#!/usr/bin/env python3
"""Simple test of OPT model with MeZO training using direct approach."""

import os
import sys
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, OPTConfig
from datasets import load_dataset
from peft import get_peft_model, LoraConfig

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from python.sglang.srt.mezo_trainer import MeZOTrainer

def test_opt_with_mezo():
    """Test OPT-125m with MeZO training using direct model loading."""
    print("Testing OPT-125m with MeZO Training (Direct Approach)")
    print("=" * 50)
    
    # Load model and tokenizer
    print("\n1. Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        "facebook/opt-125m",
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✓ Model loaded: {model.config.model_type}")
    print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Apply LoRA
    print("\n2. Applying LoRA...")
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Load dataset
    print("\n3. Loading dataset...")
    dataset = load_dataset("imdb", split="train[:100]")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask"])
    
    dataloader = DataLoader(tokenized_dataset, batch_size=4, shuffle=True)
    print(f"✓ Dataset loaded: {len(tokenized_dataset)} samples")
    
    # Initialize MeZO trainer
    print("\n4. Initializing MeZO trainer...")
    trainer = MeZOTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=dataloader,
        eval_dataloader=dataloader,
        epsilon=1e-3,
        learning_rate=1e-5,
        num_epochs=1,
        output_dir="./test_mezo_opt_output",
    )
    print("✓ MeZO trainer initialized")
    
    # Train for a few steps
    print("\n5. Training for 10 steps...")
    losses = []
    for i, batch in enumerate(dataloader):
        if i >= 10:
            break
        
        # Move batch to GPU
        batch = {k: v.cuda() for k, v in batch.items()}
        
        loss = trainer.train_step(batch)
        losses.append(loss)
        print(f"Step {i+1}: Loss = {loss:.4f}")
    
    # Report results
    print("\n6. Results:")
    print(f"✓ Average loss: {sum(losses)/len(losses):.4f}")
    print(f"✓ Initial loss: {losses[0]:.4f}")
    print(f"✓ Final loss: {losses[-1]:.4f}")
    print(f"✓ Loss reduction: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")
    
    print("\n" + "=" * 50)
    print("Testing completed successfully!")

if __name__ == "__main__":
    test_opt_with_mezo()