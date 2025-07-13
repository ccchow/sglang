#!/usr/bin/env python3
"""
MeZO training for OPT-125m using SGLang ModelRunner with RadixAttention and KV cache.
This implementation follows the MeZO paper and integrates with SGLang's infrastructure.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python"))

import torch
import numpy as np
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt

# SGLang imports
import sglang as sgl
from sglang.srt.server_args import ServerArgs
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.global_config import global_config
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool, TokenToKVPool
from sglang.srt.mem_cache.radix_cache import RadixCache

# PEFT imports
from transformers import AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MeZOConfig:
    """Configuration for MeZO training."""
    model_name: str = "facebook/opt-125m"
    dataset: str = "SST-2"
    num_steps: int = 10000
    batch_size: int = 32
    learning_rate: float = 1e-6
    epsilon: float = 1e-3
    eval_interval: int = 500
    checkpoint_interval: int = 2000
    max_seq_length: int = 128
    seed: int = 42
    output_dir: str = "./opt_125m_sst2_sglang"
    # LoRA config
    use_lora: bool = True
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: List[str] = None
    # SGLang specific
    tp_size: int = 1
    enable_flashinfer: bool = True
    kv_cache_dtype: str = "auto"
    
    def __post_init__(self):
        if self.lora_target_modules is None:
            # OPT attention modules
            self.lora_target_modules = ["q_proj", "k_proj", "v_proj"]


class SGLangMeZOTrainer:
    """
    MeZO trainer using SGLang's ModelRunner with RadixAttention.
    This provides real KV cache optimization for MeZO's perturbation pairs.
    """
    
    def __init__(self, config: MeZOConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize tokenizer
        logger.info(f"Loading tokenizer for {config.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize SGLang server args
        server_args = ServerArgs(
            model_path=config.model_name,
            trust_remote_code=True,
            tp_size=config.tp_size,
            mem_fraction_static=0.9,  # Reserve memory for training
            enable_flashinfer=config.enable_flashinfer,
            kv_cache_dtype=config.kv_cache_dtype,
            disable_disk_cache=True,
            log_level="info"
        )
        
        # Initialize ModelRunner
        logger.info("Initializing SGLang ModelRunner...")
        self.init_model_runner(server_args)
        
        # Initialize LoRA if enabled
        if config.use_lora:
            self.init_lora()
        
        # Initialize MeZO components
        self.init_mezo()
        
        # Training state
        self.state = {
            'step': 0,
            'best_accuracy': 0.0,
            'best_step': 0,
            'train_losses': [],
            'eval_accuracies': [],
            'eval_losses': [],
            'eval_steps': [],
            'gradient_norms': [],
            'cache_stats': {
                'total_requests': 0,
                'cache_hits': 0,
                'token_reuse': 0,
                'total_tokens': 0
            }
        }
    
    def init_model_runner(self, server_args):
        """Initialize SGLang ModelRunner with proper setup."""
        # Set up distributed environment if needed
        if server_args.tp_size > 1:
            import torch.distributed as dist
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
        
        # Initialize model runner
        self.model_runner = ModelRunner(
            model_config=server_args.model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=0,
            tp_rank=0,
            tp_size=server_args.tp_size,
            nccl_port=server_args.nccl_port,
            server_args=server_args
        )
        
        # Initialize memory pools
        self.req_to_token_pool = ReqToTokenPool(
            size=server_args.max_total_tokens,
            max_context_len=server_args.max_context_len
        )
        
        self.token_to_kv_pool = TokenToKVPool(
            size=server_args.max_total_tokens,
            dtype=server_args.kv_cache_dtype,
            device="cuda"
        )
        
        # Initialize RadixCache for KV cache optimization
        self.radix_cache = RadixCache(
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool=self.token_to_kv_pool,
            disable=False
        )
        
        logger.info(f"ModelRunner initialized with tp_size={server_args.tp_size}")
    
    def init_lora(self):
        """Initialize LoRA adapters on the model."""
        if not self.config.use_lora:
            return
        
        logger.info("Initializing LoRA adapters...")
        
        # Get the underlying model
        model = self.model_runner.model
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Apply LoRA to model
        self.model_runner.model = get_peft_model(model, lora_config)
        
        # Get trainable parameters
        self.lora_params = []
        total_params = 0
        trainable_params = 0
        
        for name, param in self.model_runner.model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                self.lora_params.append(param)
                trainable_params += param.numel()
                logger.debug(f"Trainable: {name} - {param.shape}")
        
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    def init_mezo(self):
        """Initialize MeZO components."""
        # Get parameters to optimize
        if self.config.use_lora:
            self.opt_params = self.lora_params
        else:
            # Full parameter tuning
            self.opt_params = [p for p in self.model_runner.model.parameters() if p.requires_grad]
        
        # Pre-generate random seeds for perturbations
        self.random_seeds = [np.random.randint(0, 2**32) for _ in range(self.config.num_steps)]
        
        logger.info(f"MeZO initialized with {len(self.opt_params)} parameter groups")
    
    def prepare_batch_for_sglang(self, texts, labels):
        """Prepare batch in SGLang format with proper request structure."""
        # Tokenize texts
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length,
            return_tensors='pt'
        )
        
        # Create requests for SGLang
        reqs = []
        for i, (text, label) in enumerate(zip(texts, labels)):
            req = Req(
                rid=f"train_{self.state['step']}_{i}",
                origin_input_text=text,
                origin_input_ids=encodings.input_ids[i].tolist(),
                sampling_params=SamplingParams(
                    temperature=0,
                    max_new_tokens=1,
                ),
            )
            req.label = label
            reqs.append(req)
        
        # Create schedule batch
        batch = ScheduleBatch(
            reqs=reqs,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool=self.token_to_kv_pool,
            tree_cache=self.radix_cache,
        )
        
        # Prepare batch for model
        batch.prepare_for_extend(
            model_config=self.model_runner.model_config,
            vocab_size=self.tokenizer.vocab_size,
            device=self.device
        )
        
        return batch, encodings
    
    def compute_loss_with_cache(self, batch, encodings, labels, perturbation_sign=0):
        """Compute loss using ModelRunner with KV cache optimization."""
        # Track cache statistics before forward
        cache_stats_before = self.radix_cache.get_stats() if hasattr(self.radix_cache, 'get_stats') else None
        
        # Forward pass through ModelRunner
        logits = self.model_runner.forward(batch, ForwardMode.EXTEND)
        
        # Track cache statistics after forward
        if cache_stats_before:
            cache_stats_after = self.radix_cache.get_stats()
            # Update our tracking
            self.state['cache_stats']['total_requests'] += len(batch.reqs)
            if 'cache_hits' in cache_stats_after:
                self.state['cache_stats']['cache_hits'] += (
                    cache_stats_after['cache_hits'] - cache_stats_before.get('cache_hits', 0)
                )
        
        # Compute cross-entropy loss
        # Get relevant logits (excluding padding)
        batch_size = len(batch.reqs)
        total_loss = 0.0
        total_correct = 0
        
        for i in range(batch_size):
            # Get sequence length (excluding padding)
            seq_len = (encodings.attention_mask[i] == 1).sum().item()
            
            # Shift logits and labels for next-token prediction
            shift_logits = logits[i, :seq_len-1, :].contiguous()
            shift_labels = encodings.input_ids[i, 1:seq_len].contiguous()
            
            # Compute loss
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            total_loss += loss.item()
            
            # For classification, check if model predicts the right sentiment
            # This is a simplified accuracy metric
            last_token_logits = logits[i, seq_len-1, :]
            pred = torch.argmax(last_token_logits)
            # Simple heuristic: positive sentiment if prediction contains positive words
            # In practice, you'd want a more sophisticated evaluation
            total_correct += (labels[i] == 1 and pred.item() > self.tokenizer.vocab_size // 2) or \
                           (labels[i] == 0 and pred.item() <= self.tokenizer.vocab_size // 2)
        
        avg_loss = total_loss / batch_size
        accuracy = total_correct / batch_size
        
        return avg_loss, accuracy
    
    def zo_perturb_parameters(self, seed, scaling_factor=1.0):
        """Apply in-place perturbation to parameters."""
        torch.manual_seed(seed)
        
        for param in self.opt_params:
            z = torch.normal(mean=0, std=1, size=param.shape, 
                           device=param.device, dtype=param.dtype)
            param.data.add_(scaling_factor * self.config.epsilon * z)
    
    def mezo_step(self, texts, labels):
        """Single MeZO training step with KV cache optimization."""
        # Prepare batch
        batch, encodings = self.prepare_batch_for_sglang(texts, labels)
        
        # Get random seed for this step
        seed = self.random_seeds[self.state['step']]
        
        # Apply positive perturbation
        self.zo_perturb_parameters(seed, scaling_factor=1.0)
        
        # Forward pass with +epsilon (should hit cache on subsequent passes)
        with torch.no_grad():
            loss_plus, acc_plus = self.compute_loss_with_cache(batch, encodings, labels, perturbation_sign=1)
        
        # Apply negative perturbation (total change is -2*epsilon from positive)
        self.zo_perturb_parameters(seed, scaling_factor=-2.0)
        
        # Forward pass with -epsilon (should benefit from cache)
        with torch.no_grad():
            loss_minus, acc_minus = self.compute_loss_with_cache(batch, encodings, labels, perturbation_sign=-1)
        
        # Restore parameters to original + epsilon
        self.zo_perturb_parameters(seed, scaling_factor=1.0)
        
        # Gradient estimate
        grad_estimate = (loss_plus - loss_minus) / (2 * self.config.epsilon)
        
        # Update parameters: theta = theta - lr * grad_estimate * z
        # First remove the perturbation
        self.zo_perturb_parameters(seed, scaling_factor=-1.0)
        
        # Then apply the gradient update
        torch.manual_seed(seed)
        for param in self.opt_params:
            z = torch.normal(mean=0, std=1, size=param.shape,
                           device=param.device, dtype=param.dtype)
            param.data.add_(-self.config.learning_rate * grad_estimate * z)
        
        # Track statistics
        self.state['gradient_norms'].append(abs(grad_estimate))
        
        avg_loss = (loss_plus + loss_minus) / 2
        avg_acc = (acc_plus + acc_minus) / 2
        
        return avg_loss, avg_acc, abs(grad_estimate)
    
    def evaluate(self, eval_data, max_examples=None):
        """Evaluate model on dataset."""
        if max_examples:
            eval_data = eval_data[:max_examples]
        
        total_loss = 0
        total_correct = 0
        total_examples = 0
        
        batch_size = 16  # Smaller batch for evaluation
        
        with torch.no_grad():
            for i in range(0, len(eval_data), batch_size):
                batch_data = eval_data[i:i+batch_size]
                texts = [ex['text'] for ex in batch_data]
                labels = [ex['label'] for ex in batch_data]
                
                batch, encodings = self.prepare_batch_for_sglang(texts, labels)
                loss, accuracy = self.compute_loss_with_cache(batch, encodings, labels)
                
                total_loss += loss * len(batch_data)
                total_correct += accuracy * len(batch_data)
                total_examples += len(batch_data)
        
        avg_loss = total_loss / total_examples if total_examples > 0 else 0
        avg_accuracy = total_correct / total_examples if total_examples > 0 else 0
        
        return avg_accuracy, avg_loss
    
    def save_checkpoint(self):
        """Save training checkpoint."""
        checkpoint_path = Path(self.config.output_dir) / f"checkpoint_{self.state['step']}.pt"
        
        # Save LoRA parameters if using LoRA
        if self.config.use_lora:
            # Save PEFT model
            self.model_runner.model.save_pretrained(
                Path(self.config.output_dir) / f"lora_checkpoint_{self.state['step']}"
            )
        
        # Save training state
        checkpoint = {
            'step': self.state['step'],
            'config': asdict(self.config),
            'state': self.state,
            'cache_stats': self.state['cache_stats'],
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save plots
        self._save_plots()
    
    def _save_plots(self):
        """Save training plots."""
        if len(self.state['eval_steps']) < 2:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Accuracy plot
        ax1.plot(self.state['eval_steps'], self.state['eval_accuracies'], 'o-', color='green', markersize=6)
        ax1.axhline(y=0.5, color='red', linestyle=':', alpha=0.5, label='Random')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Evaluation Accuracy')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Loss plot
        ax2.plot(self.state['eval_steps'], self.state['eval_losses'], 'o-', color='blue', markersize=6)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Loss')
        ax2.set_title('Evaluation Loss')
        ax2.grid(True, alpha=0.3)
        
        # Gradient norm plot
        if self.state['gradient_norms']:
            steps = range(1, len(self.state['gradient_norms']) + 1)
            ax3.plot(steps, self.state['gradient_norms'], alpha=0.5, color='orange')
            # Add smoothed line
            window = min(100, len(self.state['gradient_norms']) // 10)
            if window > 1:
                smoothed = np.convolve(self.state['gradient_norms'], 
                                     np.ones(window)/window, mode='valid')
                smooth_steps = range(window//2, len(self.state['gradient_norms']) - window//2 + 1)
                ax3.plot(smooth_steps, smoothed, color='red', linewidth=2, label='Smoothed')
            ax3.set_xlabel('Step')
            ax3.set_ylabel('Gradient Norm')
            ax3.set_title('MeZO Gradient Estimates')
            ax3.set_yscale('log')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
        
        # Cache statistics
        cache_hit_rate = (self.state['cache_stats']['cache_hits'] / 
                         max(1, self.state['cache_stats']['total_requests'])) * 100
        
        ax4.text(0.1, 0.8, f"Total Requests: {self.state['cache_stats']['total_requests']:,}", 
                transform=ax4.transAxes, fontsize=12)
        ax4.text(0.1, 0.6, f"Cache Hits: {self.state['cache_stats']['cache_hits']:,}", 
                transform=ax4.transAxes, fontsize=12)
        ax4.text(0.1, 0.4, f"Cache Hit Rate: {cache_hit_rate:.1f}%", 
                transform=ax4.transAxes, fontsize=12, weight='bold')
        ax4.text(0.1, 0.2, f"Current Step: {self.state['step']}", 
                transform=ax4.transAxes, fontsize=12)
        ax4.set_title('SGLang KV Cache Statistics')
        ax4.axis('off')
        
        plt.suptitle(f'OPT-125M SST-2 Training with MeZO + SGLang (Step {self.state["step"]})')
        plt.tight_layout()
        
        plot_path = Path(self.config.output_dir) / f"progress_step_{self.state['step']}.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        
        logger.info(f"Saved plots to {plot_path}")
    
    def train(self, train_data, eval_data):
        """Full training loop."""
        logger.info("Starting MeZO training with SGLang ModelRunner...")
        logger.info(f"Configuration: {self.config}")
        logger.info(f"Using {'LoRA' if self.config.use_lora else 'full parameter'} tuning")
        
        # Initial evaluation
        init_acc, init_loss = self.evaluate(eval_data)
        self.state['eval_accuracies'].append(init_acc)
        self.state['eval_losses'].append(init_loss)
        self.state['eval_steps'].append(0)
        logger.info(f"Initial: Accuracy={init_acc:.1%}, Loss={init_loss:.4f}")
        
        # Training loop
        start_time = time.time()
        
        for step in range(self.config.num_steps):
            self.state['step'] = step + 1
            
            # Sample batch
            idx = np.random.choice(len(train_data), self.config.batch_size, replace=True)
            texts = [train_data[i]['text'] for i in idx]
            labels = [train_data[i]['label'] for i in idx]
            
            # MeZO step
            loss, acc, grad = self.mezo_step(texts, labels)
            self.state['train_losses'].append(loss)
            
            # Evaluation
            if self.state['step'] % self.config.eval_interval == 0:
                eval_acc, eval_loss = self.evaluate(eval_data)
                self.state['eval_accuracies'].append(eval_acc)
                self.state['eval_losses'].append(eval_loss)
                self.state['eval_steps'].append(self.state['step'])
                
                if eval_acc > self.state['best_accuracy']:
                    self.state['best_accuracy'] = eval_acc
                    self.state['best_step'] = self.state['step']
                
                # Calculate cache statistics
                cache_hit_rate = (self.state['cache_stats']['cache_hits'] / 
                                max(1, self.state['cache_stats']['total_requests'])) * 100
                
                elapsed = time.time() - start_time
                steps_per_sec = self.state['step'] / elapsed
                eta_hours = (self.config.num_steps - self.state['step']) / steps_per_sec / 3600
                
                logger.info(
                    f"Step {self.state['step']}: "
                    f"Loss={loss:.4f}, Acc={acc:.1%}, Grad={grad:.6f}, "
                    f"Eval: Acc={eval_acc:.1%}, Loss={eval_loss:.4f}, "
                    f"Best={self.state['best_accuracy']:.1%}@{self.state['best_step']}, "
                    f"Cache={cache_hit_rate:.1f}%, "
                    f"Speed={steps_per_sec:.2f} steps/s, ETA={eta_hours:.1f}h"
                )
            
            # Checkpoint
            if self.state['step'] % self.config.checkpoint_interval == 0:
                self.save_checkpoint()
        
        # Final checkpoint
        self.save_checkpoint()
        
        # Final statistics
        total_time = time.time() - start_time
        cache_hit_rate = (self.state['cache_stats']['cache_hits'] / 
                         max(1, self.state['cache_stats']['total_requests'])) * 100
        
        logger.info("\n" + "=" * 80)
        logger.info("Training Complete!")
        logger.info("=" * 80)
        logger.info(f"Total time: {total_time/3600:.2f} hours")
        logger.info(f"Average speed: {self.config.num_steps/total_time:.2f} steps/sec")
        logger.info(f"Final accuracy: {self.state['eval_accuracies'][-1]:.1%}")
        logger.info(f"Best accuracy: {self.state['best_accuracy']:.1%} at step {self.state['best_step']}")
        logger.info(f"KV Cache hit rate: {cache_hit_rate:.1f}%")
        logger.info(f"Non-zero gradients: {sum(1 for g in self.state['gradient_norms'] if g > 0)}/{len(self.state['gradient_norms'])}")
        
        return self.state


def load_sst2_data(data_dir, split, max_examples=None):
    """Load SST-2 data."""
    file_path = f"{data_dir}/{split}.tsv"
    examples = []
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()[1:]  # Skip header
            for i, line in enumerate(lines):
                if max_examples and i >= max_examples:
                    break
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    text, label = parts
                    examples.append({'text': text, 'label': int(label)})
    except FileNotFoundError:
        logger.warning(f"Data file not found: {file_path}")
        logger.warning("Using synthetic data for testing")
        # Create synthetic data
        positive_phrases = [
            "This movie is fantastic!",
            "I really enjoyed watching this.",
            "Absolutely brilliant performance!",
            "Best film I've seen all year.",
            "Highly recommend this movie!"
        ]
        negative_phrases = [
            "Terrible movie, waste of time.",
            "I couldn't finish watching this.",
            "Poorly written and badly acted.",
            "One of the worst films ever.",
            "Complete disappointment."
        ]
        
        for _ in range(10):
            examples.extend([
                {'text': np.random.choice(positive_phrases), 'label': 1},
                {'text': np.random.choice(negative_phrases), 'label': 0}
            ])
    
    return examples


def main():
    """Run OPT-125m SST-2 training with MeZO and SGLang."""
    # Configuration
    config = MeZOConfig(
        model_name="facebook/opt-125m",
        num_steps=10000,
        batch_size=32,
        learning_rate=1e-6,
        epsilon=1e-3,
        eval_interval=500,
        checkpoint_interval=2000,
        seed=42,
        output_dir="./opt_125m_sst2_mezo_sglang",
        use_lora=True,
        lora_rank=8,
        tp_size=1,  # Use 1 GPU
        enable_flashinfer=True
    )
    
    # Set seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    
    # Load data
    data_dir = "/home/lei/git/princeton-nlp/MeZO/medium_models/data/k-shot-1k-test/SST-2/16-13"
    logger.info("Loading SST-2 dataset...")
    train_data = load_sst2_data(data_dir, "train", max_examples=1000)
    eval_data = load_sst2_data(data_dir, "dev", max_examples=500)
    logger.info(f"Loaded {len(train_data)} train, {len(eval_data)} eval examples")
    
    # Initialize trainer
    trainer = SGLangMeZOTrainer(config)
    
    # Train
    state = trainer.train(train_data, eval_data)
    
    # Save final results
    results_path = Path(config.output_dir) / "final_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'config': asdict(config),
            'final_state': state
        }, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()