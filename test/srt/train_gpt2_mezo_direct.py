#!/usr/bin/env python3
"""
Direct MeZO training implementation for GPT-2 using SGLang infrastructure.
This bypasses the high-level API to have more control over the setup.
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
from dataclasses import dataclass
from typing import List, Dict
from torch.utils.data import Dataset, DataLoader

# SGLang imports
from sglang.srt.mezo_trainer import MeZOTrainer, MeZODataset
from sglang.srt.hf_transformers_utils import get_tokenizer

# Direct model imports
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Config
from peft import LoraConfig, get_peft_model, TaskType

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleModelRunner:
    """Simplified ModelRunner that wraps a HuggingFace model."""
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Simple forward pass."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs.loss if labels is not None else outputs.logits


class SimpleLoRAManager:
    """Simplified LoRA manager."""
    def __init__(self, device):
        self.device = device
        self.loras = {}
    
    def add_lora(self, name, model):
        """Add a LoRA-enabled model."""
        self.loras[name] = SimpleLoRAAdapter(model)
    
    def save_lora(self, name, path):
        """Save LoRA weights."""
        if name in self.loras:
            self.loras[name].model.save_pretrained(path)


class SimpleLoRAAdapter:
    """Simple wrapper for LoRA adapter."""
    def __init__(self, model):
        self.model = model
        self.layers = []
        
        # Collect LoRA parameters
        for name, module in model.named_modules():
            if hasattr(module, 'lora_A'):
                self.layers.append(SimpleLoRALayer(name, module))


class SimpleLoRALayer:
    """Simple LoRA layer wrapper."""
    def __init__(self, name, module):
        self.name = name
        self.weights = {}
        if hasattr(module, 'lora_A'):
            self.weights['lora_A'] = module.lora_A
        if hasattr(module, 'lora_B'):
            self.weights['lora_B'] = module.lora_B


class SimpleMeZOTrainer:
    """Simplified MeZO trainer for demonstration."""
    def __init__(self, model, tokenizer, lora_config):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model.to(self.device)
        self.model.print_trainable_parameters()
        
        # Get trainable parameters
        self.trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        logger.info(f"Number of trainable parameters: {len(self.trainable_params)}")
        
        # MeZO settings
        self.epsilon = 1e-3
        self.learning_rate = 1e-6
    
    def compute_loss(self, batch):
        """Compute loss for a batch."""
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        
        # For language modeling, labels are the same as input_ids
        # shifted by one position
        labels = input_ids.clone()
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs.loss
    
    def mezo_step(self, batch):
        """Single MeZO optimization step."""
        # Generate random perturbation
        z_list = [torch.randn_like(p) for p in self.trainable_params]
        
        # Apply positive perturbation
        for p, z in zip(self.trainable_params, z_list):
            p.data.add_(self.epsilon * z)
        
        # Forward pass with +epsilon
        with torch.no_grad():
            loss_plus = self.compute_loss(batch).item()
        
        # Apply negative perturbation (net change: -2*epsilon from original)
        for p, z in zip(self.trainable_params, z_list):
            p.data.add_(-2 * self.epsilon * z)
        
        # Forward pass with -epsilon
        with torch.no_grad():
            loss_minus = self.compute_loss(batch).item()
        
        # Restore original parameters
        for p, z in zip(self.trainable_params, z_list):
            p.data.add_(self.epsilon * z)
        
        # Compute gradient estimate
        grad_estimate = (loss_plus - loss_minus) / (2 * self.epsilon)
        
        # Update parameters
        for p, z in zip(self.trainable_params, z_list):
            p.data.add_(-self.learning_rate * grad_estimate * z)
        
        return (loss_plus + loss_minus) / 2
    
    def train(self, train_dataloader, num_steps):
        """Training loop."""
        dataloader_iter = iter(train_dataloader)
        
        for step in range(num_steps):
            # Get batch
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(train_dataloader)
                batch = next(dataloader_iter)
            
            # MeZO step
            loss = self.mezo_step(batch)
            
            if step % 100 == 0:
                logger.info(f"Step {step}/{num_steps}, Loss: {loss:.4f}")
        
        return loss


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


def main():
    """Run GPT-2 MeZO training."""
    # Configuration
    model_name = "gpt2"
    num_steps = 500  # Reduced for demo
    batch_size = 8
    output_dir = "./gpt2_mezo_direct"
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load model and tokenizer
    logger.info(f"Loading {model_name}...")
    tokenizer = get_tokenizer(model_name)
    # GPT-2 doesn't have a pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn", "c_proj"],  # GPT-2 attention modules
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Load data
    logger.info("Loading synthetic SST-2 data...")
    train_data = load_sst2_data(max_examples=200)
    
    # Create dataset and dataloader
    dataset = MeZODataset(train_data, tokenizer, max_length=128)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize trainer
    logger.info("Initializing MeZO trainer...")
    trainer = SimpleMeZOTrainer(model, tokenizer, lora_config)
    
    # Training
    logger.info("Starting MeZO training...")
    start_time = time.time()
    
    final_loss = trainer.train(dataloader, num_steps)
    
    total_time = time.time() - start_time
    
    # Save model
    model_path = Path(output_dir) / "lora_model"
    trainer.model.save_pretrained(model_path)
    logger.info(f"Saved LoRA model to {model_path}")
    
    # Save results
    results = {
        'model': model_name,
        'num_steps': num_steps,
        'batch_size': batch_size,
        'final_loss': final_loss,
        'total_time_seconds': total_time,
        'steps_per_second': num_steps / total_time,
    }
    
    results_path = Path(output_dir) / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("\n" + "="*60)
    logger.info("Training Complete!")
    logger.info("="*60)
    logger.info(f"Model: {model_name}")
    logger.info(f"Total time: {total_time:.1f} seconds")
    logger.info(f"Training speed: {num_steps/total_time:.2f} steps/sec")
    logger.info(f"Final loss: {final_loss:.4f}")
    logger.info("="*60)


if __name__ == "__main__":
    main()