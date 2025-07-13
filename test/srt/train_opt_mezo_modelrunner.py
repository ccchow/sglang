#!/usr/bin/env python3
"""
MeZO training for OPT-125m using SGLang ModelRunner and MeZOTrainer.
This implementation uses the proper SGLang infrastructure with RadixAttention optimization.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python"))

import torch
import numpy as np
import time
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import matplotlib.pyplot as plt

# SGLang imports
from sglang.srt.server_args import ServerArgs
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.lora.lora_config import LoRAConfig
from sglang.srt.mezo_trainer import MeZOTrainer, MeZODataset, create_dataloader
from sglang.srt.hf_transformers_utils import get_config, get_tokenizer
from sglang.srt.utils import get_device_memory_capacity

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for MeZO training."""
    model_name: str = "facebook/opt-125m"
    dataset: str = "SST-2"
    num_steps: int = 10000
    batch_size: int = 16
    learning_rate: float = 1e-6
    epsilon: float = 1e-3
    eval_interval: int = 500
    checkpoint_interval: int = 2000
    max_seq_length: int = 128
    seed: int = 42
    output_dir: str = "./opt_125m_sst2_mezo_modelrunner"
    # LoRA config
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: List[str] = None
    # SGLang specific
    tp_size: int = 1
    mem_fraction_static: float = 0.8
    
    def __post_init__(self):
        if self.lora_target_modules is None:
            # OPT attention modules
            self.lora_target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]


def load_sst2_data(data_dir, split, max_examples=None):
    """Load SST-2 data in the format expected by MeZO."""
    file_path = f"{data_dir}/{split}.tsv"
    examples = []
    
    # Sentiment templates
    templates = {
        0: "This movie review expresses a negative sentiment: ",
        1: "This movie review expresses a positive sentiment: "
    }
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()[1:]  # Skip header
            for i, line in enumerate(lines):
                if max_examples and i >= max_examples:
                    break
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    text, label = parts
                    # Create prompt-completion format
                    prompt = f"Classify the sentiment of this movie review: {text}\nSentiment:"
                    completion = " positive" if int(label) == 1 else " negative"
                    examples.append({
                        'prompt': prompt,
                        'completion': completion
                    })
    except FileNotFoundError:
        logger.warning(f"Data file not found: {file_path}")
        logger.warning("Using synthetic data for testing")
        # Create synthetic data
        positive_texts = [
            "This movie is absolutely fantastic!",
            "I loved every moment of this brilliant film.",
            "Outstanding performances and amazing story.",
            "One of the best movies I've ever seen.",
            "Highly recommend this masterpiece!"
        ]
        negative_texts = [
            "Terrible movie, complete waste of time.",
            "I couldn't even finish watching this.",
            "Poorly written and badly acted.",
            "One of the worst films ever made.",
            "Absolutely disappointing experience."
        ]
        
        for _ in range(50):
            if np.random.random() > 0.5:
                text = np.random.choice(positive_texts)
                prompt = f"Classify the sentiment of this movie review: {text}\nSentiment:"
                completion = " positive"
            else:
                text = np.random.choice(negative_texts)
                prompt = f"Classify the sentiment of this movie review: {text}\nSentiment:"
                completion = " negative"
                
            examples.append({
                'prompt': prompt,
                'completion': completion
            })
    
    return examples


def setup_model_runner_and_lora(config: TrainingConfig):
    """Set up ModelRunner and LoRA manager."""
    # First, let's check if distributed is initialized
    import torch.distributed as dist
    
    # Initialize distributed if not already
    if not dist.is_initialized() and config.tp_size > 1:
        dist.init_process_group(backend="nccl")
    
    # Get model configuration
    model_config = get_config(config.model_name, trust_remote_code=True)
    
    # Create ServerArgs
    server_args = ServerArgs(
        model_path=config.model_name,
        trust_remote_code=True,
        tp_size=config.tp_size,
        mem_fraction_static=config.mem_fraction_static,
        disable_cuda_graph=True,  # Disable for training flexibility
        disable_radix_cache=False,  # Enable RadixAttention
        dtype="float16" if torch.cuda.is_available() else "float32",
    )
    
    # Initialize ModelRunner
    logger.info("Initializing ModelRunner...")
    device_memory = get_device_memory_capacity()
    logger.info(f"Available device memory: {device_memory / 1e9:.2f} GB")
    
    model_runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=0,
        tp_rank=0,
        tp_size=server_args.tp_size,
        pp_rank=0,
        pp_size=1,
        nccl_port=server_args.nccl_port if server_args.nccl_port else 28765,
        server_args=server_args,
    )
    
    # Initialize LoRA manager
    logger.info("Initializing LoRA manager...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = model_config.hf_config.vocab_size
    
    lora_manager = LoRAManager(
        model_runner=model_runner,
        max_num_batched_tokens=16384,
        vocab_size=vocab_size,
        lora_config=None,  # Will be set when adding LoRA
        max_num_seqs=256,
        device=device,
    )
    
    # Create LoRA configuration
    lora_config = LoRAConfig(
        model_path=config.model_name,
        rank=config.lora_rank,
        alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
    )
    
    # Add LoRA adapter
    lora_name = "mezo_adapter"
    lora_manager.add_lora(lora_name, lora_config)
    logger.info(f"Added LoRA adapter '{lora_name}' with rank {config.lora_rank}")
    
    # Count trainable parameters
    total_params = 0
    trainable_params = 0
    for name, param in model_runner.model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable LoRA parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    return model_runner, lora_manager, lora_name


def evaluate_model(trainer, eval_dataloader, num_batches=10):
    """Evaluate the model."""
    total_loss = 0
    num_samples = 0
    
    dataloader_iter = iter(eval_dataloader)
    
    for i in range(min(num_batches, len(eval_dataloader))):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            break
        
        # Get loss without gradient update
        with torch.no_grad():
            loss = trainer._forward_pass(batch)
        
        total_loss += loss
        num_samples += 1
    
    avg_loss = total_loss / num_samples if num_samples > 0 else 0
    return avg_loss


def main():
    """Run OPT-125m MeZO training with ModelRunner."""
    config = TrainingConfig()
    
    # Create output directory
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Set seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    
    # Load tokenizer
    logger.info(f"Loading tokenizer for {config.model_name}...")
    tokenizer = get_tokenizer(config.model_name, trust_remote_code=True)
    
    # Load data
    data_dir = "/home/lei/git/princeton-nlp/MeZO/medium_models/data/k-shot-1k-test/SST-2/16-13"
    logger.info("Loading SST-2 dataset...")
    train_data = load_sst2_data(data_dir, "train", max_examples=1000)
    eval_data = load_sst2_data(data_dir, "dev", max_examples=200)
    logger.info(f"Loaded {len(train_data)} train, {len(eval_data)} eval examples")
    
    # Create datasets
    train_dataset = MeZODataset(train_data, tokenizer, max_length=config.max_seq_length)
    eval_dataset = MeZODataset(eval_data, tokenizer, max_length=config.max_seq_length)
    
    # Create dataloaders
    train_dataloader = create_dataloader(train_dataset, batch_size=config.batch_size, shuffle=True)
    eval_dataloader = create_dataloader(eval_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Set up model and LoRA
    model_runner, lora_manager, lora_name = setup_model_runner_and_lora(config)
    
    # Initialize MeZO trainer
    logger.info("Initializing MeZO trainer...")
    trainer = MeZOTrainer(
        model_runner=model_runner,
        lora_manager=lora_manager,
        lora_name=lora_name,
        tokenizer=tokenizer,
        normalize_perturbations=False,  # Follow paper default
    )
    
    # Initial evaluation
    logger.info("Running initial evaluation...")
    init_loss = evaluate_model(trainer, eval_dataloader)
    logger.info(f"Initial loss: {init_loss:.4f}")
    
    # Training metrics
    train_losses = []
    eval_losses = []
    eval_steps = []
    
    # Training loop
    logger.info("Starting MeZO training...")
    start_time = time.time()
    
    # Use the trainer's built-in train method
    trainer.train(
        train_dataloader=train_dataloader,
        learning_rate=config.learning_rate,
        num_steps=config.num_steps,
        epsilon=config.epsilon
    )
    
    # Final evaluation
    final_loss = evaluate_model(trainer, eval_dataloader)
    total_time = time.time() - start_time
    
    # Save results
    results = {
        'config': asdict(config),
        'initial_loss': init_loss,
        'final_loss': final_loss,
        'total_time_seconds': total_time,
        'steps_per_second': config.num_steps / total_time,
        'kv_cache_hit_rate': trainer.kv_cache_hit_rate if hasattr(trainer, 'kv_cache_hit_rate') else 0.0,
    }
    
    results_path = Path(config.output_dir) / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save LoRA weights
    lora_path = Path(config.output_dir) / "lora_weights"
    lora_manager.save_lora(lora_name, str(lora_path))
    logger.info(f"Saved LoRA weights to {lora_path}")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("Training Complete!")
    logger.info("="*60)
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Total time: {total_time/60:.1f} minutes")
    logger.info(f"Training speed: {config.num_steps/total_time:.2f} steps/sec")
    logger.info(f"Initial loss: {init_loss:.4f}")
    logger.info(f"Final loss: {final_loss:.4f}")
    logger.info(f"Loss reduction: {(init_loss - final_loss)/init_loss*100:.1f}%")
    if hasattr(trainer, 'radix_optimizer') and trainer.radix_optimizer:
        stats = trainer.radix_optimizer.get_optimization_stats()
        logger.info(f"KV Cache hit rate: {stats.get('cache_hit_rate', 0):.1%}")
    logger.info("="*60)


if __name__ == "__main__":
    main()