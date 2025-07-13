#!/usr/bin/env python3
"""
Simple test to reproduce RoBERTa SST-2 results with MeZO.
Direct implementation without using the full SGLang server infrastructure.
"""

import torch
import numpy as np
import time
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import matplotlib.pyplot as plt
from typing import List, Dict


def load_sst2_data(file_path: str) -> List[Dict]:
    """Load SST-2 examples from TSV file."""
    examples = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:  # Skip header
            parts = line.strip().split('\t')
            if len(parts) == 2:
                text, label = parts
                examples.append({
                    'text': text,
                    'label': int(label)
                })
    return examples


def create_lora_layers(model, rank=8, alpha=16):
    """Add LoRA to RoBERTa following the paper."""
    lora_params = []
    device = model.device
    
    # Add LoRA to query and value projections in all layers
    for name, module in model.named_modules():
        if 'attention' in name and hasattr(module, 'self'):
            # Query projection
            if hasattr(module.self, 'query'):
                layer = module.self.query
                d_in = layer.in_features
                d_out = layer.out_features
                
                # LoRA matrices
                lora_A = torch.nn.Parameter(torch.randn(rank, d_in, device=device) * 0.01)
                lora_B = torch.nn.Parameter(torch.zeros(d_out, rank, device=device))
                
                # Store original weight and LoRA components
                layer.original_weight = layer.weight.data.clone()
                layer.lora_A = lora_A
                layer.lora_B = lora_B
                layer.lora_scale = alpha / rank
                
                lora_params.extend([lora_A, lora_B])
            
            # Value projection
            if hasattr(module.self, 'value'):
                layer = module.self.value
                d_in = layer.in_features
                d_out = layer.out_features
                
                lora_A = torch.nn.Parameter(torch.randn(rank, d_in, device=device) * 0.01)
                lora_B = torch.nn.Parameter(torch.zeros(d_out, rank, device=device))
                
                layer.original_weight = layer.weight.data.clone()
                layer.lora_A = lora_A
                layer.lora_B = lora_B
                layer.lora_scale = alpha / rank
                
                lora_params.extend([lora_A, lora_B])
    
    return lora_params


def apply_lora(model):
    """Apply LoRA weights to model."""
    for name, module in model.named_modules():
        if 'attention' in name and hasattr(module, 'self'):
            for proj in ['query', 'value']:
                if hasattr(module.self, proj):
                    layer = getattr(module.self, proj)
                    if hasattr(layer, 'lora_A'):
                        layer.weight.data = layer.original_weight + \
                            layer.lora_scale * (layer.lora_B @ layer.lora_A)


def mezo_train_step(model, tokenizer, batch, lora_params, epsilon=1e-3, lr=1e-6):
    """Single MeZO training step following the paper."""
    # Sample perturbation (NO normalization, following paper)
    z_list = [torch.randn_like(p) for p in lora_params]
    
    # Tokenize batch
    inputs = tokenizer(
        batch['texts'],
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    ).to(model.device)
    
    labels = torch.tensor(batch['labels']).to(model.device)
    
    # Forward pass with +epsilon*z
    for i, p in enumerate(lora_params):
        p.data.add_(epsilon * z_list[i])
    apply_lora(model)
    
    with torch.no_grad():
        outputs = model(**inputs, labels=labels)
        loss_plus = outputs.loss.item()
    
    # Forward pass with -epsilon*z
    for i, p in enumerate(lora_params):
        p.data.add_(-2 * epsilon * z_list[i])
    apply_lora(model)
    
    with torch.no_grad():
        outputs = model(**inputs, labels=labels)
        loss_minus = outputs.loss.item()
    
    # Restore parameters
    for i, p in enumerate(lora_params):
        p.data.add_(epsilon * z_list[i])
    
    # Gradient estimate
    grad_est = (loss_plus - loss_minus) / (2 * epsilon)
    
    # Update parameters
    for i, p in enumerate(lora_params):
        p.data.add_(-lr * grad_est * z_list[i])
    
    apply_lora(model)
    
    return (loss_plus + loss_minus) / 2


def evaluate(model, tokenizer, examples):
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    
    with torch.no_grad():
        for ex in examples:
            inputs = tokenizer(
                ex['text'],
                truncation=True,
                max_length=128,
                return_tensors='pt'
            ).to(model.device)
            
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=-1).item()
            correct += (pred == ex['label'])
    
    return correct / len(examples)


def main():
    print("=" * 80)
    print("RoBERTa-large SST-2 MeZO Reproduction (Simple)")
    print("=" * 80)
    
    # Configuration
    model_name = "roberta-large"
    batch_size = 64
    learning_rate = 1e-6
    epsilon = 1e-3
    num_steps = 5000
    eval_interval = 500
    
    print(f"\nConfiguration:")
    print(f"  Model: {model_name}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Epsilon: {epsilon}")
    print(f"  Steps: {num_steps}")
    
    # Load model and tokenizer
    print("\nLoading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    ).to(device)
    
    # Freeze base model
    for param in model.parameters():
        param.requires_grad = False
    
    # Add LoRA
    print("Adding LoRA adapters...")
    lora_params = create_lora_layers(model, rank=8, alpha=16)
    print(f"  LoRA parameters: {sum(p.numel() for p in lora_params):,}")
    
    # Load data
    print("\nLoading SST-2 data...")
    data_dir = "/home/lei/git/princeton-nlp/MeZO/medium_models/data/k-shot-1k-test/SST-2"
    train_data = load_sst2_data(f"{data_dir}/512-42/train.tsv")
    eval_data = load_sst2_data(f"{data_dir}/512-42/dev.tsv")[:200]  # Subset for speed
    
    print(f"  Train: {len(train_data)} examples")
    print(f"  Eval: {len(eval_data)} examples")
    
    # Initial evaluation
    print("\nInitial evaluation...")
    apply_lora(model)
    init_acc = evaluate(model, tokenizer, eval_data)
    print(f"  Accuracy: {init_acc:.1%}")
    
    # Training
    print(f"\nTraining for {num_steps} steps...")
    print("-" * 60)
    print("Step  | Loss    | Eval Acc | Time")
    print("-" * 60)
    
    losses = []
    eval_accs = [init_acc]
    start_time = time.time()
    
    for step in range(num_steps):
        # Sample batch
        idx = np.random.choice(len(train_data), batch_size, replace=True)
        batch = {
            'texts': [train_data[i]['text'] for i in idx],
            'labels': [train_data[i]['label'] for i in idx]
        }
        
        # Train step
        loss = mezo_train_step(model, tokenizer, batch, lora_params, epsilon, learning_rate)
        losses.append(loss)
        
        # Evaluate
        if (step + 1) % eval_interval == 0:
            eval_acc = evaluate(model, tokenizer, eval_data)
            eval_accs.append(eval_acc)
            elapsed = time.time() - start_time
            print(f"{step+1:5d} | {loss:.5f} | {eval_acc:8.1%} | {elapsed:5.0f}s")
    
    print("-" * 60)
    
    # Final evaluation
    final_acc = evaluate(model, tokenizer, eval_data)
    total_time = time.time() - start_time
    
    print(f"\nTraining complete in {total_time:.0f}s")
    print(f"Final accuracy: {final_acc:.1%} (from {init_acc:.1%})")
    print(f"Improvement: {(final_acc - init_acc) * 100:+.1f} percentage points")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss
    ax1.plot(losses, alpha=0.5)
    ax1.plot(np.convolve(losses, np.ones(100)/100, mode='valid'), linewidth=2)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    eval_steps = list(range(0, num_steps+1, eval_interval))[:len(eval_accs)]
    ax2.plot(eval_steps, eval_accs, 'o-')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Evaluation Accuracy')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('roberta_large_sst2_simple.png', dpi=150)
    print(f"\nPlots saved to: roberta_large_sst2_simple.png")
    
    return final_acc > init_acc


if __name__ == "__main__":
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.manual_seed(42)
    
    success = main()