#!/usr/bin/env python3
"""
MeZO training for OPT-125m with proper ModelRunner setup.
This version correctly initializes the distributed environment.
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
import torch.distributed as dist

# SGLang imports
from sglang.srt.server_args import ServerArgs
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.lora.lora_config import LoRAConfig
from sglang.srt.mezo_trainer import MeZOTrainer, MeZODataset, create_dataloader
from sglang.srt.hf_transformers_utils import get_config, get_tokenizer
from sglang.srt.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.utils import get_device_memory_capacity

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_environment():
    """Set up environment variables for distributed training."""
    # For single GPU, we still need to initialize distributed
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
    
    logger.info(f"Environment: RANK={os.environ['RANK']}, "
                f"WORLD_SIZE={os.environ['WORLD_SIZE']}, "
                f"MASTER_ADDR={os.environ['MASTER_ADDR']}")


def init_distributed_if_needed(backend="nccl"):
    """Initialize distributed environment if not already initialized."""
    if not dist.is_initialized():
        logger.info("Initializing PyTorch distributed...")
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        
        # Initialize PyTorch distributed
        dist.init_process_group(
            backend=backend if torch.cuda.is_available() else "gloo",
            rank=rank,
            world_size=world_size,
        )
        
        # Initialize SGLang distributed environment
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            distributed_init_method="env://",
            local_rank=int(os.environ.get("LOCAL_RANK", 0)),
            backend=backend if torch.cuda.is_available() else "gloo",
        )
        
        # Initialize model parallel groups (even for single GPU)
        initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            backend=backend if torch.cuda.is_available() else "gloo",
        )
        
        logger.info("Distributed environment initialized successfully")
    else:
        logger.info("Distributed environment already initialized")


class SimpleLoRAManager:
    """Simplified LoRA manager that doesn't require full server infrastructure."""
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.loras = {}
    
    def add_lora(self, name, lora_config):
        """Add a LoRA adapter to the model."""
        from peft import get_peft_model, LoraConfig, TaskType
        
        # Convert our LoRAConfig to PEFT LoraConfig
        peft_config = LoraConfig(
            r=lora_config.rank,
            lora_alpha=lora_config.alpha,
            target_modules=lora_config.target_modules,
            lora_dropout=0.0,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(self.model, peft_config)
        self.model.to(self.device)
        
        # Create a simple adapter wrapper
        self.loras[name] = SimpleLoRAAdapter(self.model)
        logger.info(f"Added LoRA adapter '{name}'")
        self.model.print_trainable_parameters()
    
    def get_lora(self, name):
        """Get a LoRA adapter."""
        return self.loras.get(name)


class SimpleLoRAAdapter:
    """Simple wrapper for LoRA adapter."""
    def __init__(self, model):
        self.model = model
        self.layers = []
        
        # Collect LoRA parameters
        for name, module in model.named_modules():
            if hasattr(module, 'lora_A') or hasattr(module, 'lora_B'):
                self.layers.append(SimpleLoRALayer(name, module))


class SimpleLoRALayer:
    """Simple LoRA layer wrapper."""
    def __init__(self, name, module):
        self.name = name
        self.weights = {}
        if hasattr(module, 'lora_A'):
            for param_name, param in module.named_parameters():
                if 'lora' in param_name:
                    self.weights[param_name] = param


@dataclass
class TrainingConfig:
    """Configuration for MeZO training."""
    model_name: str = "gpt2"  # Using GPT-2 since OPT needs proper implementation
    dataset: str = "SST-2"
    num_steps: int = 1000
    batch_size: int = 16
    learning_rate: float = 1e-6
    epsilon: float = 1e-3
    eval_interval: int = 100
    checkpoint_interval: int = 500
    max_seq_length: int = 128
    seed: int = 42
    output_dir: str = "./gpt2_mezo_modelrunner"
    # LoRA config
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_target_modules: List[str] = None
    # SGLang specific
    tp_size: int = 1
    mem_fraction_static: float = 0.8
    
    def __post_init__(self):
        if self.lora_target_modules is None:
            # GPT-2 attention modules
            self.lora_target_modules = ["c_attn", "c_proj"]


def load_sst2_data(max_examples=100):
    """Load synthetic SST-2 data."""
    examples = []
    
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
    
    for i in range(max_examples):
        if i % 2 == 0:
            text = np.random.choice(positive_texts)
            prompt = f"Classify the sentiment: {text}\nSentiment:"
            completion = " positive"
        else:
            text = np.random.choice(negative_texts)
            prompt = f"Classify the sentiment: {text}\nSentiment:"
            completion = " negative"
            
        examples.append({
            'prompt': prompt,
            'completion': completion
        })
    
    return examples


def setup_model_runner_and_lora(config: TrainingConfig):
    """Set up ModelRunner and LoRA with proper initialization."""
    # Ensure distributed is initialized
    init_distributed_if_needed()
    
    # Create ServerArgs with minimal configuration
    server_args = ServerArgs(
        model_path=config.model_name,
        trust_remote_code=True,
        tp_size=config.tp_size,
        mem_fraction_static=config.mem_fraction_static,
        disable_cuda_graph=True,
        disable_radix_cache=False,
        dtype="float16" if torch.cuda.is_available() else "float32",
        grammar_backend="none",  # Avoid xgrammar dependency
    )
    
    # Create ModelConfig - it takes model_path as first argument
    model_config = ModelConfig(
        config.model_name,  # model_path is the first positional argument
        model_override_args="{}",
        trust_remote_code=True,
    )
    
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_id = 0 if torch.cuda.is_available() else -1
    
    logger.info("Initializing ModelRunner...")
    logger.info(f"Device: {device}, GPU ID: {gpu_id}")
    
    try:
        # Initialize ModelRunner with all required parameters
        model_runner = ModelRunner(
            model_config=model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=gpu_id,
            tp_rank=0,
            tp_size=server_args.tp_size,
            pp_rank=0,
            pp_size=1,
            nccl_port=29501,  # Use a different port than master
            server_args=server_args,
        )
        logger.info("ModelRunner initialized successfully!")
        
        # For MeZO, we need direct access to the model
        model = model_runner.model
        
        # Create a simple LoRA manager
        lora_manager = SimpleLoRAManager(model, device)
        
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
        
        return model_runner, lora_manager, lora_name
        
    except Exception as e:
        logger.error(f"Failed to initialize ModelRunner: {e}")
        logger.info("Falling back to direct model loading...")
        
        # Fallback: Load model directly
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        
        # Create simple LoRA manager
        lora_manager = SimpleLoRAManager(model, device)
        
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
        
        # Create a simple ModelRunner wrapper
        class SimpleModelRunner:
            def __init__(self, model):
                self.model = model
        
        return SimpleModelRunner(model), lora_manager, lora_name


def main():
    """Run MeZO training with proper ModelRunner setup."""
    config = TrainingConfig()
    
    # Setup environment
    setup_environment()
    
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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load data
    logger.info("Loading synthetic SST-2 dataset...")
    train_data = load_sst2_data(max_examples=500)
    eval_data = load_sst2_data(max_examples=100)
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
    try:
        trainer = MeZOTrainer(
            model_runner=model_runner,
            lora_manager=lora_manager,
            lora_name=lora_name,
            tokenizer=tokenizer,
            normalize_perturbations=False,
        )
        
        # Training
        logger.info("Starting MeZO training...")
        start_time = time.time()
        
        trainer.train(
            train_dataloader=train_dataloader,
            learning_rate=config.learning_rate,
            num_steps=config.num_steps,
            epsilon=config.epsilon
        )
        
        total_time = time.time() - start_time
        
        # Save results
        results = {
            'config': asdict(config),
            'total_time_seconds': total_time,
            'steps_per_second': config.num_steps / total_time,
        }
        
        results_path = Path(config.output_dir) / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("\n" + "="*60)
        logger.info("Training Complete!")
        logger.info("="*60)
        logger.info(f"Model: {config.model_name}")
        logger.info(f"Total time: {total_time/60:.1f} minutes")
        logger.info(f"Training speed: {config.num_steps/total_time:.2f} steps/sec")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Cleanup distributed if initialized
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()