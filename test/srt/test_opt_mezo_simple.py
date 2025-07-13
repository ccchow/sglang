#!/usr/bin/env python3
"""
Simplified MeZO training for OPT-125m using SGLang Engine API.
This version properly handles distributed initialization.
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

# SGLang imports
import sglang as sgl
from sglang import Engine
from sglang.srt.sampling.sampling_params import SamplingParams

# Transformers and PEFT
from transformers import AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MeZOConfig:
    """Configuration for MeZO training."""
    model_name: str = "facebook/opt-125m"
    dataset: str = "SST-2"
    num_steps: int = 5000
    batch_size: int = 16
    learning_rate: float = 1e-6
    epsilon: float = 1e-3
    eval_interval: int = 250
    checkpoint_interval: int = 1000
    max_seq_length: int = 128
    seed: int = 42
    output_dir: str = "./opt_125m_sst2_mezo"
    # LoRA config
    use_lora: bool = True
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0


class MeZOTrainerWithEngine:
    """
    MeZO trainer using SGLang Engine API.
    This provides a simpler interface while still leveraging SGLang's optimizations.
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
        
        # Initialize SGLang Engine
        logger.info("Initializing SGLang Engine...")
        self.engine = Engine(
            model_path=config.model_name,
            tp_size=1,
            random_seed=config.seed,
            mem_fraction_static=0.8,  # Reserve memory for training
            trust_remote_code=True,
        )
        
        # Get model reference for MeZO
        self.model = self.engine.model_runner.model
        
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
            'generation_times': [],
        }
    
    def init_lora(self):
        """Initialize LoRA adapters."""
        logger.info("Initializing LoRA adapters...")
        
        # Configure LoRA for OPT
        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Apply LoRA
        self.engine.model_runner.model = get_peft_model(self.model, lora_config)
        self.model = self.engine.model_runner.model
        
        # Get trainable parameters
        self.lora_params = []
        total_params = 0
        trainable_params = 0
        
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                self.lora_params.append(param)
                trainable_params += param.numel()
        
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    def init_mezo(self):
        """Initialize MeZO components."""
        # Get parameters to optimize
        if self.config.use_lora:
            self.opt_params = self.lora_params
        else:
            # Full parameter tuning - freeze all but selected layers
            self.opt_params = []
            for name, param in self.model.named_parameters():
                # Only train the last few transformer layers for memory efficiency
                if "decoder.layers.10" in name or "decoder.layers.11" in name:
                    param.requires_grad = True
                    self.opt_params.append(param)
                else:
                    param.requires_grad = False
        
        # Pre-generate random seeds
        self.random_seeds = [np.random.randint(0, 2**32) for _ in range(self.config.num_steps)]
        
        logger.info(f"MeZO initialized with {len(self.opt_params)} parameter groups")
    
    def create_sst2_prompt(self, text, label=None):
        """Create a prompt for SST-2 sentiment classification."""
        prompt = f"Review: {text}\nSentiment:"
        if label is not None:
            sentiment = "positive" if label == 1 else "negative"
            prompt += f" {sentiment}"
        return prompt
    
    def compute_loss_batch(self, texts, labels):
        """Compute loss for a batch using SGLang Engine."""
        prompts = []
        for text, label in zip(texts, labels):
            prompt = self.create_sst2_prompt(text, label)
            prompts.append(prompt)
        
        # Use engine to get logits
        sampling_params = SamplingParams(
            temperature=0,
            max_new_tokens=0,  # Just get logits, no generation
            return_logits=True,
        )
        
        start_time = time.time()
        outputs = self.engine.generate(prompts, sampling_params)
        gen_time = time.time() - start_time
        self.state['generation_times'].append(gen_time)
        
        # Compute loss from logits
        total_loss = 0.0
        total_correct = 0
        
        positive_token_id = self.tokenizer.encode(" positive", add_special_tokens=False)[0]
        negative_token_id = self.tokenizer.encode(" negative", add_special_tokens=False)[0]
        
        for i, output in enumerate(outputs):
            if hasattr(output, 'logits') and output.logits is not None:
                # Get logits for positive/negative tokens
                logits = output.logits[-1]  # Last position
                sentiment_logits = torch.tensor([
                    logits[negative_token_id],
                    logits[positive_token_id]
                ])
                
                # Compute cross-entropy
                label_tensor = torch.tensor(labels[i])
                loss = torch.nn.functional.cross_entropy(
                    sentiment_logits.unsqueeze(0),
                    label_tensor.unsqueeze(0)
                )
                total_loss += loss.item()
                
                # Compute accuracy
                pred = 1 if logits[positive_token_id] > logits[negative_token_id] else 0
                total_correct += (pred == labels[i])
        
        avg_loss = total_loss / len(texts)
        accuracy = total_correct / len(texts)
        
        return avg_loss, accuracy
    
    def zo_perturb_parameters(self, seed, scaling_factor=1.0):
        """Apply in-place perturbation to parameters."""
        torch.manual_seed(seed)
        
        for param in self.opt_params:
            z = torch.normal(mean=0, std=1, size=param.shape, 
                           device=param.device, dtype=param.dtype)
            param.data.add_(scaling_factor * self.config.epsilon * z)
    
    def mezo_step(self, texts, labels):
        """Single MeZO training step."""
        # Get random seed for this step
        seed = self.random_seeds[self.state['step']]
        
        # Apply positive perturbation
        self.zo_perturb_parameters(seed, scaling_factor=1.0)
        
        # Forward pass with +epsilon
        with torch.no_grad():
            loss_plus, acc_plus = self.compute_loss_batch(texts, labels)
        
        # Apply negative perturbation
        self.zo_perturb_parameters(seed, scaling_factor=-2.0)
        
        # Forward pass with -epsilon
        with torch.no_grad():
            loss_minus, acc_minus = self.compute_loss_batch(texts, labels)
        
        # Restore parameters
        self.zo_perturb_parameters(seed, scaling_factor=1.0)
        
        # Gradient estimate
        grad_estimate = (loss_plus - loss_minus) / (2 * self.config.epsilon)
        
        # Update parameters
        # First remove perturbation
        self.zo_perturb_parameters(seed, scaling_factor=-1.0)
        
        # Then apply gradient update
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
    
    def evaluate(self, eval_data, max_examples=200):
        """Evaluate model on dataset."""
        if max_examples and len(eval_data) > max_examples:
            eval_data = eval_data[:max_examples]
        
        total_loss = 0
        total_correct = 0
        total_examples = 0
        
        batch_size = 8  # Smaller batch for evaluation
        
        with torch.no_grad():
            for i in range(0, len(eval_data), batch_size):
                batch_data = eval_data[i:i+batch_size]
                texts = [ex['text'] for ex in batch_data]
                labels = [ex['label'] for ex in batch_data]
                
                loss, accuracy = self.compute_loss_batch(texts, labels)
                
                total_loss += loss * len(batch_data)
                total_correct += accuracy * len(batch_data)
                total_examples += len(batch_data)
        
        avg_loss = total_loss / total_examples if total_examples > 0 else 0
        avg_accuracy = total_correct / total_examples if total_examples > 0 else 0
        
        return avg_accuracy, avg_loss
    
    def save_checkpoint(self):
        """Save training checkpoint."""
        checkpoint_path = Path(self.config.output_dir) / f"checkpoint_{self.state['step']}.pt"
        
        # Save LoRA model if using LoRA
        if self.config.use_lora:
            self.model.save_pretrained(
                Path(self.config.output_dir) / f"lora_checkpoint_{self.state['step']}"
            )
        
        # Save training state
        checkpoint = {
            'step': self.state['step'],
            'config': asdict(self.config),
            'state': self.state,
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
            window = min(50, len(self.state['gradient_norms']) // 10)
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
        
        # Generation time stats
        if self.state['generation_times']:
            recent_times = self.state['generation_times'][-100:]
            avg_time = np.mean(recent_times)
            ax4.text(0.1, 0.8, f"Avg Generation Time: {avg_time:.3f}s", 
                    transform=ax4.transAxes, fontsize=12)
            ax4.text(0.1, 0.6, f"Total Steps: {self.state['step']}", 
                    transform=ax4.transAxes, fontsize=12)
            ax4.text(0.1, 0.4, f"Best Accuracy: {self.state['best_accuracy']:.1%}", 
                    transform=ax4.transAxes, fontsize=12, weight='bold')
            ax4.text(0.1, 0.2, f"Best Step: {self.state['best_step']}", 
                    transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Training Statistics')
        ax4.axis('off')
        
        plt.suptitle(f'OPT-125M SST-2 MeZO Training (Step {self.state["step"]})')
        plt.tight_layout()
        
        plot_path = Path(self.config.output_dir) / f"progress_step_{self.state['step']}.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        
        logger.info(f"Saved plots to {plot_path}")
    
    def train(self, train_data, eval_data):
        """Full training loop."""
        logger.info("Starting MeZO training with SGLang Engine...")
        logger.info(f"Configuration: {self.config}")
        logger.info(f"Using {'LoRA' if self.config.use_lora else 'partial parameter'} tuning")
        
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
                
                elapsed = time.time() - start_time
                steps_per_sec = self.state['step'] / elapsed
                eta_hours = (self.config.num_steps - self.state['step']) / steps_per_sec / 3600
                
                logger.info(
                    f"Step {self.state['step']}: "
                    f"Loss={loss:.4f}, Acc={acc:.1%}, Grad={grad:.6f}, "
                    f"Eval: Acc={eval_acc:.1%}, Loss={eval_loss:.4f}, "
                    f"Best={self.state['best_accuracy']:.1%}@{self.state['best_step']}, "
                    f"Speed={steps_per_sec:.2f} steps/s, ETA={eta_hours:.1f}h"
                )
            
            # Checkpoint
            if self.state['step'] % self.config.checkpoint_interval == 0:
                self.save_checkpoint()
        
        # Final checkpoint
        self.save_checkpoint()
        
        # Final statistics
        total_time = time.time() - start_time
        
        logger.info("\n" + "=" * 80)
        logger.info("Training Complete!")
        logger.info("=" * 80)
        logger.info(f"Total time: {total_time/3600:.2f} hours")
        logger.info(f"Average speed: {self.config.num_steps/total_time:.2f} steps/sec")
        logger.info(f"Final accuracy: {self.state['eval_accuracies'][-1]:.1%}")
        logger.info(f"Best accuracy: {self.state['best_accuracy']:.1%} at step {self.state['best_step']}")
        
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
            "This movie is absolutely fantastic!",
            "I really enjoyed every moment of this film.",
            "Brilliant acting and amazing storyline!",
            "One of the best movies I've ever seen.",
            "Highly recommend this masterpiece!"
        ]
        negative_phrases = [
            "Terrible movie, complete waste of time.",
            "I couldn't even finish watching this.",
            "Poorly written and badly executed.",
            "One of the worst films I've encountered.",
            "Absolute disappointment from start to finish."
        ]
        
        for _ in range(50):
            examples.extend([
                {'text': np.random.choice(positive_phrases), 'label': 1},
                {'text': np.random.choice(negative_phrases), 'label': 0}
            ])
    
    return examples


def main():
    """Run OPT-125m SST-2 training with MeZO."""
    # Configuration
    config = MeZOConfig(
        model_name="facebook/opt-125m",
        num_steps=5000,
        batch_size=16,
        learning_rate=1e-6,
        epsilon=1e-3,
        eval_interval=250,
        checkpoint_interval=1000,
        seed=42,
        output_dir="./opt_125m_sst2_mezo_engine",
        use_lora=True,
        lora_rank=8
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
    eval_data = load_sst2_data(data_dir, "dev", max_examples=200)
    logger.info(f"Loaded {len(train_data)} train, {len(eval_data)} eval examples")
    
    # Initialize trainer
    trainer = MeZOTrainerWithEngine(config)
    
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
    
    # Cleanup
    trainer.engine.shutdown()


if __name__ == "__main__":
    main()