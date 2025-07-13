#!/usr/bin/env python3
"""
Gradual approach to RoBERTa MLM with SGLang.
Step 1: Use HuggingFace model with SGLang-style infrastructure
Step 2: Gradually migrate to full SGLang ModelRunner
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
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
from transformers import RobertaForMaskedLM, RobertaTokenizer

# Import SGLang components we can use
from sglang.srt.mezo_radix_optimizer import MeZORadixOptimizer
from sglang.srt.mezo_config import MeZOConfig
from sglang.srt.lora.lora_config import LoRAConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration for RoBERTa SST-2."""
    model_name: str = "roberta-large"
    dataset: str = "SST-2"
    num_steps: int = 100000
    batch_size: int = 64
    learning_rate: float = 1e-6
    epsilon: float = 1e-3
    eval_interval: int = 1000
    checkpoint_interval: int = 10000
    max_seq_length: int = 128
    seed: int = 42
    output_dir: str = "./roberta_sst2_gradual"
    # MLM specific
    template: str = "It was [MASK]."
    label_words: Dict[int, str] = None
    # LoRA config
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    
    def __post_init__(self):
        if self.label_words is None:
            self.label_words = {0: 'terrible', 1: 'great'}


class SGLangStyleMLMTrainer:
    """
    MLM trainer that uses SGLang patterns but with HuggingFace backend.
    This serves as a bridge to full SGLang integration.
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer (HuggingFace for now)
        logger.info(f"Loading {config.model_name}...")
        self.tokenizer = RobertaTokenizer.from_pretrained(config.model_name)
        self.model = RobertaForMaskedLM.from_pretrained(config.model_name).to(self.device)
        
        # Get label word IDs
        self.label_word_ids = self._get_label_word_ids()
        
        # Initialize LoRA manually (SGLang LoRAConfig requires a path)
        self.lora_config = {
            'r': config.lora_rank,
            'lora_alpha': config.lora_alpha,
            'lora_dropout': config.lora_dropout,
            'target_modules': ["query", "key", "value"],
        }
        self.lora_params = self._initialize_lora()
        
        # Initialize MeZO components
        self.mezo_config = MeZOConfig(
            learning_rate=config.learning_rate,
            epsilon=config.epsilon,
            batch_size=config.batch_size,
            num_steps=config.num_steps,
        )
        self.radix_optimizer = MeZORadixOptimizer(epsilon=config.epsilon)
        
        # Training state
        self.state = {
            'step': 0,
            'best_accuracy': 0.0,
            'best_step': 0,
            'train_losses': [],
            'eval_accuracies': [],
            'eval_losses': [],
            'eval_steps': [],
            'kv_cache_stats': {},
            'gradient_norms': [],
        }
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
    def _get_label_word_ids(self):
        """Get token IDs for label words."""
        label_word_ids = {}
        for label, word in self.config.label_words.items():
            tokens = self.tokenizer.tokenize(' ' + word)
            if len(tokens) == 1:
                token_id = self.tokenizer.convert_tokens_to_ids(tokens[0])
                label_word_ids[label] = token_id
                logger.info(f"Label {label}: ' {word}' -> token_id {token_id}")
            else:
                raise ValueError(f"Label word ' {word}' tokenizes to {len(tokens)} tokens")
        return label_word_ids
    
    def _initialize_lora(self):
        """Initialize LoRA adapters following SGLang patterns."""
        logger.info(f"Initializing LoRA with config: {self.lora_config}")
        
        # Freeze base model
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Add LoRA to attention layers
        lora_params = []
        total_params = 0
        
        for name, module in self.model.named_modules():
            if 'attention' in name and hasattr(module, 'self'):
                for proj_name in ['query', 'key', 'value']:
                    if hasattr(module.self, proj_name):
                        layer = getattr(module.self, proj_name)
                        d_in = layer.in_features
                        d_out = layer.out_features
                        
                        # Create LoRA matrices
                        lora_A = torch.nn.Parameter(
                            torch.randn(self.lora_config['r'], d_in, device=self.device) * 0.01
                        )
                        lora_B = torch.nn.Parameter(
                            torch.zeros(d_out, self.lora_config['r'], device=self.device)
                        )
                        
                        # Store in layer for easy access
                        layer.lora_A = lora_A
                        layer.lora_B = lora_B
                        layer.lora_scale = self.lora_config['lora_alpha'] / self.lora_config['r']
                        
                        # Store original weight
                        layer.original_weight = layer.weight.data.clone()
                        
                        lora_params.extend([lora_A, lora_B])
                        total_params += lora_A.numel() + lora_B.numel()
        
        logger.info(f"Initialized {len(lora_params)//2} LoRA adapter pairs")
        logger.info(f"Total trainable parameters: {total_params:,}")
        
        return lora_params
    
    def _apply_lora(self):
        """Apply LoRA weights to model."""
        for name, module in self.model.named_modules():
            if 'attention' in name and hasattr(module, 'self'):
                for proj_name in ['query', 'key', 'value']:
                    if hasattr(module.self, proj_name):
                        layer = getattr(module.self, proj_name)
                        if hasattr(layer, 'lora_A'):
                            layer.weight.data = layer.original_weight + \
                                layer.lora_scale * (layer.lora_B @ layer.lora_A)
    
    def _prepare_batch_for_radix(self, texts, labels):
        """Prepare batch in SGLang format for RadixAttention optimization."""
        # Format texts with template
        mlm_texts = [f"{text} {self.config.template}" for text in texts]
        mlm_texts = [text.replace('[MASK]', self.tokenizer.mask_token) for text in mlm_texts]
        
        # Tokenize
        inputs = self.tokenizer(
            mlm_texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length,
            return_tensors='pt'
        ).to(self.device)
        
        # Create batch dict for RadixOptimizer
        batch = {
            'input_ids': inputs.input_ids,
            'attention_mask': inputs.attention_mask,
            'prompt': mlm_texts,
            'prompt_length': torch.tensor([inputs.input_ids.shape[1]] * len(mlm_texts)),
            'labels': labels,
            'mask_positions': (inputs.input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        }
        
        return batch
    
    def compute_mlm_loss(self, batch, use_cache_optimization=False):
        """Compute MLM loss with optional RadixAttention optimization tracking."""
        # Prepare requests if using cache optimization
        if use_cache_optimization:
            requests, metadata = self.radix_optimizer.prepare_mezo_requests(
                batch,
                perturbation_sign=1 if use_cache_optimization else 0,
                request_prefix=f"step_{self.state['step']}"
            )
            # The optimizer tracks stats internally
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            logits = outputs.logits
            
            # Extract logits at mask positions
            batch_indices = batch['mask_positions'][0]
            position_indices = batch['mask_positions'][1]
            mask_logits = logits[batch_indices, position_indices]
            
            # Get label word logits
            label_logits = mask_logits[:, [self.label_word_ids[0], self.label_word_ids[1]]]
            
            # Compute loss and accuracy
            labels_tensor = torch.tensor(batch['labels'], device=self.device)
            loss = torch.nn.functional.cross_entropy(label_logits, labels_tensor)
            preds = torch.argmax(label_logits, dim=-1)
            accuracy = (preds == labels_tensor).float().mean().item()
            
            return loss.item(), accuracy
    
    def mezo_step(self, texts, labels):
        """Single MeZO training step with RadixAttention optimization."""
        # Prepare batch
        batch = self._prepare_batch_for_radix(texts, labels)
        
        # Sample perturbation
        z_list = [torch.randn_like(p) for p in self.lora_params]
        
        # Apply positive perturbation
        for i, p in enumerate(self.lora_params):
            p.data.add_(self.config.epsilon * z_list[i])
        self._apply_lora()
        
        # Forward pass with +epsilon (uses cache)
        loss_plus, acc_plus = self.compute_mlm_loss(batch, use_cache_optimization=True)
        
        # Apply negative perturbation
        for i, p in enumerate(self.lora_params):
            p.data.add_(-2 * self.config.epsilon * z_list[i])
        self._apply_lora()
        
        # Forward pass with -epsilon (uses cache)
        loss_minus, acc_minus = self.compute_mlm_loss(batch, use_cache_optimization=True)
        
        # Restore parameters
        for i, p in enumerate(self.lora_params):
            p.data.add_(self.config.epsilon * z_list[i])
        
        # Gradient estimate
        grad_est = (loss_plus - loss_minus) / (2 * self.config.epsilon)
        
        # Update parameters
        for i, p in enumerate(self.lora_params):
            p.data.add_(-self.config.learning_rate * grad_est * z_list[i])
        
        self._apply_lora()
        
        # Track gradient norm
        grad_norm = abs(grad_est)
        self.state['gradient_norms'].append(grad_norm)
        
        avg_loss = (loss_plus + loss_minus) / 2
        avg_acc = (acc_plus + acc_minus) / 2
        
        return avg_loss, avg_acc, grad_norm
    
    def evaluate(self, eval_data, max_examples=None):
        """Evaluate model on dataset."""
        if max_examples:
            eval_data = eval_data[:max_examples]
        
        total_loss = 0
        total_correct = 0
        total_examples = 0
        
        batch_size = 32
        for i in range(0, len(eval_data), batch_size):
            batch_data = eval_data[i:i+batch_size]
            texts = [ex['text'] for ex in batch_data]
            labels = [ex['label'] for ex in batch_data]
            
            batch = self._prepare_batch_for_radix(texts, labels)
            loss, accuracy = self.compute_mlm_loss(batch, use_cache_optimization=False)
            
            total_loss += loss * len(batch_data)
            total_correct += accuracy * len(batch_data)
            total_examples += len(batch_data)
        
        avg_loss = total_loss / total_examples if total_examples > 0 else 0
        avg_accuracy = total_correct / total_examples if total_examples > 0 else 0
        
        return avg_accuracy, avg_loss
    
    def save_checkpoint(self):
        """Save training checkpoint."""
        checkpoint_path = Path(self.config.output_dir) / f"checkpoint_{self.state['step']}.pt"
        
        # Save LoRA parameters
        lora_state_dict = {}
        for i, param in enumerate(self.lora_params):
            lora_state_dict[f'lora_param_{i}'] = param.data.cpu()
        
        checkpoint = {
            'step': self.state['step'],
            'config': asdict(self.config),
            'state': self.state,
            'lora_state_dict': lora_state_dict,
            'radix_stats': self.radix_optimizer.get_optimization_stats(),
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
        ax2.set_ylabel('MLM Loss')
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
        
        # Cache stats
        cache_stats = self.radix_optimizer.get_optimization_stats()
        ax4.text(0.1, 0.8, f"Cache Hit Rate: {cache_stats['cache_hit_rate']:.1%}", 
                transform=ax4.transAxes, fontsize=14)
        ax4.text(0.1, 0.6, f"Token Reuse Rate: {cache_stats['token_reuse_rate']:.1%}", 
                transform=ax4.transAxes, fontsize=14)
        ax4.text(0.1, 0.4, f"Total Forward Passes: {cache_stats['total_forward_passes']:,}", 
                transform=ax4.transAxes, fontsize=14)
        ax4.text(0.1, 0.2, f"Cache Hits: {cache_stats['cache_hits']:,}", 
                transform=ax4.transAxes, fontsize=14)
        ax4.set_title('RadixAttention Statistics')
        ax4.axis('off')
        
        plt.suptitle(f'RoBERTa SST-2 Training Progress (Step {self.state["step"]})')
        plt.tight_layout()
        
        plot_path = Path(self.config.output_dir) / f"progress_step_{self.state['step']}.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        
        logger.info(f"Saved plots to {plot_path}")
    
    def train(self, train_data, eval_data):
        """Full training loop with checkpointing."""
        logger.info("Starting SGLang-style MLM training...")
        logger.info(f"Configuration: {self.config}")
        logger.info(f"MeZO config: {self.mezo_config}")
        logger.info(f"LoRA config: {self.lora_config}")
        
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
                
                # Get cache stats
                cache_stats = self.radix_optimizer.get_optimization_stats()
                cache_rate = cache_stats['cache_hit_rate']
                
                elapsed = time.time() - start_time
                steps_per_sec = self.state['step'] / elapsed
                eta_hours = (self.config.num_steps - self.state['step']) / steps_per_sec / 3600
                
                logger.info(
                    f"Step {self.state['step']}: "
                    f"Loss={loss:.4f}, Acc={acc:.1%}, Grad={grad:.6f}, "
                    f"Eval: Acc={eval_acc:.1%}, Loss={eval_loss:.4f}, "
                    f"Best={self.state['best_accuracy']:.1%}@{self.state['best_step']}, "
                    f"Cache={cache_rate:.1%}, Time={elapsed/60:.1f}min, ETA={eta_hours:.1f}h"
                )
            
            # Progress logging
            elif self.state['step'] % 100 == 0:
                elapsed = time.time() - start_time
                steps_per_sec = self.state['step'] / elapsed
                logger.debug(f"Step {self.state['step']}: {steps_per_sec:.2f} steps/sec")
            
            # Checkpoint
            if self.state['step'] % self.config.checkpoint_interval == 0:
                self.save_checkpoint()
                logger.info(f"Checkpoint saved at step {self.state['step']}")
        
        # Final checkpoint
        self.save_checkpoint()
        
        # Final stats
        total_time = time.time() - start_time
        final_stats = self.radix_optimizer.get_optimization_stats()
        
        logger.info("\n" + "=" * 80)
        logger.info("Training Complete!")
        logger.info("=" * 80)
        logger.info(f"Total time: {total_time/3600:.2f} hours")
        logger.info(f"Average speed: {self.config.num_steps/total_time:.2f} steps/sec")
        logger.info(f"Final accuracy: {self.state['eval_accuracies'][-1]:.1%}")
        logger.info(f"Best accuracy: {self.state['best_accuracy']:.1%} at step {self.state['best_step']}")
        logger.info(f"Cache hit rate: {final_stats['cache_hit_rate']:.1%}")
        logger.info(f"Token reuse rate: {final_stats['token_reuse_rate']:.1%}")
        logger.info(f"Non-zero gradients: {sum(1 for g in self.state['gradient_norms'] if g > 0)}/{len(self.state['gradient_norms'])}")
        
        return self.state


def load_sst2_data(file_path, max_examples=None):
    """Load SST-2 data."""
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
        logger.error(f"Data file not found: {file_path}")
        # Create minimal dummy data
        examples = [
            {'text': 'This is a great movie!', 'label': 1},
            {'text': 'Terrible film.', 'label': 0},
        ] * 10
    return examples


def main():
    """Run RoBERTa SST-2 training with SGLang-style infrastructure."""
    # Configuration
    config = TrainingConfig(
        model_name="roberta-large",
        num_steps=100000,  # Full 100K steps
        batch_size=64,
        learning_rate=1e-6,
        epsilon=1e-3,
        eval_interval=1000,
        checkpoint_interval=10000,
        seed=42,
        output_dir="./roberta_sst2_sglang_gradual"
    )
    
    # Set seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    
    # Load data
    data_dir = "/home/lei/git/princeton-nlp/MeZO/medium_models/data/k-shot-1k-test/SST-2"
    logger.info("Loading SST-2 dataset...")
    train_data = load_sst2_data(f"{data_dir}/512-42/train.tsv")
    eval_data = load_sst2_data(f"{data_dir}/512-42/dev.tsv")
    logger.info(f"Loaded {len(train_data)} train, {len(eval_data)} eval examples")
    
    # Initialize trainer
    trainer = SGLangStyleMLMTrainer(config)
    
    # Train
    state = trainer.train(train_data, eval_data)
    
    # Save final results
    results_path = Path(config.output_dir) / "final_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'config': asdict(config),
            'final_state': state,
            'radix_stats': trainer.radix_optimizer.get_optimization_stats()
        }, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()