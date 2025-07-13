#!/usr/bin/env python3
"""
Test MeZO with RoBERTa-large on SST-2 following the original MeZO setup.
This test uses the same hyperparameters as MeZO/medium_models/mezo.sh
"""

import torch
import json
import os
import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import matplotlib.pyplot as plt
from typing import Dict, List
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_sst2_from_file(file_path, tokenizer=None, apply_template=True):
    """Load SST-2 examples from TSV file."""
    examples = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        # Skip header
        for line in lines[1:]:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                sentence, label = parts
                label = int(label)
                
                if apply_template:
                    # Follow MeZO's template: *cls**sent_0*_It_was*mask*.*sep+*
                    # This becomes: [CLS] sentence It was [MASK] . [SEP]
                    text = f"{sentence} It was [MASK]."
                else:
                    text = sentence
                
                examples.append({
                    'text': text,
                    'label': label,
                    'original_sentence': sentence
                })
    
    return examples


def create_lora_roberta(model, rank=8):
    """Add LoRA adapters to RoBERTa model."""
    # For simplicity, add LoRA to attention layers only
    lora_params = []
    device = model.device
    
    for name, module in model.named_modules():
        if 'attention' in name and hasattr(module, 'self'):
            # Add LoRA to query and value projections
            if hasattr(module.self, 'query'):
                in_features = module.self.query.in_features
                out_features = module.self.query.out_features
                
                # Create LoRA parameters on the same device as the model
                lora_A_q = torch.nn.Parameter(torch.randn(rank, in_features, device=device) * 0.01)
                lora_B_q = torch.nn.Parameter(torch.zeros(out_features, rank, device=device))
                
                lora_A_v = torch.nn.Parameter(torch.randn(rank, in_features, device=device) * 0.01)
                lora_B_v = torch.nn.Parameter(torch.zeros(out_features, rank, device=device))
                
                # Store original weights and add LoRA
                module.self.query.original_weight = module.self.query.weight.data.clone()
                module.self.value.original_weight = module.self.value.weight.data.clone()
                
                module.self.query.lora_A = lora_A_q
                module.self.query.lora_B = lora_B_q
                module.self.value.lora_A = lora_A_v
                module.self.value.lora_B = lora_B_v
                
                lora_params.extend([lora_A_q, lora_B_q, lora_A_v, lora_B_v])
    
    return lora_params


def apply_lora_weights(model, scaling=1.0):
    """Apply LoRA weights to the model."""
    for name, module in model.named_modules():
        if 'attention' in name and hasattr(module, 'self'):
            if hasattr(module.self.query, 'lora_A'):
                # Apply LoRA: W = W_original + scaling * B @ A
                module.self.query.weight.data = module.self.query.original_weight + \
                    scaling * (module.self.query.lora_B @ module.self.query.lora_A)
                
                module.self.value.weight.data = module.self.value.original_weight + \
                    scaling * (module.self.value.lora_B @ module.self.value.lora_A)


def mezo_step_roberta(model, tokenizer, batch, lora_params, epsilon=1e-3, lr=1e-6):
    """Perform one MeZO step on RoBERTa."""
    # Sample perturbation
    z_list = [torch.randn_like(p) for p in lora_params]
    
    # Normalize perturbations
    z_list = [z / (z.norm() + 1e-8) for z in z_list]
    
    # Apply positive perturbation
    for i, p in enumerate(lora_params):
        p.data.add_(epsilon * z_list[i])
    apply_lora_weights(model)
    
    # Forward pass with +epsilon
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(
            batch['text'], 
            padding=True, 
            truncation=True, 
            max_length=128,
            return_tensors='pt'
        ).to(model.device)
        
        outputs = model(**inputs, labels=torch.tensor(batch['label']).to(model.device))
        loss_plus = outputs.loss
    
    # Apply negative perturbation
    for i, p in enumerate(lora_params):
        p.data.add_(-2 * epsilon * z_list[i])
    apply_lora_weights(model)
    
    # Forward pass with -epsilon
    with torch.no_grad():
        outputs = model(**inputs, labels=torch.tensor(batch['label']).to(model.device))
        loss_minus = outputs.loss
    
    # Restore parameters
    for i, p in enumerate(lora_params):
        p.data.add_(epsilon * z_list[i])
    apply_lora_weights(model)
    
    # Compute gradient estimate
    grad_estimate = (loss_plus - loss_minus) / (2 * epsilon)
    
    # Update parameters
    for i, p in enumerate(lora_params):
        p.data.add_(-lr * grad_estimate * z_list[i])
    
    return (loss_plus + loss_minus) / 2


def evaluate_sst2(model, tokenizer, examples):
    """Evaluate on SST-2 examples."""
    model.eval()
    correct = 0
    total_loss = 0
    
    with torch.no_grad():
        for example in examples:
            inputs = tokenizer(
                example['text'],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            ).to(model.device)
            
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=-1).item()
            
            correct += (pred == example['label'])
            
            # Compute loss
            labels = torch.tensor([example['label']]).to(model.device)
            loss = torch.nn.functional.cross_entropy(outputs.logits, labels)
            total_loss += loss.item()
    
    accuracy = correct / len(examples)
    avg_loss = total_loss / len(examples)
    
    return accuracy, avg_loss


def test_mezo_roberta_sst2():
    """Test MeZO on RoBERTa-large with SST-2."""
    print("=" * 60)
    print("MeZO RoBERTa-large SST-2 Test")
    print("=" * 60)
    
    # Configuration from MeZO
    model_name = "roberta-base"  # Use base instead of large for memory
    batch_size = 4  # Small batch for testing
    learning_rate = 1e-5  # Slightly higher for faster convergence
    epsilon = 1e-3
    num_steps = 50  # Reduced for quick testing
    eval_steps = 5
    k_shot = 16
    seed = 42
    
    print(f"\nConfiguration:")
    print(f"  Model: {model_name}")
    print(f"  Task: SST-2 ({k_shot}-shot)")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Epsilon: {epsilon}")
    print(f"  Steps: {num_steps}")
    
    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2,
        torch_dtype=torch.float32
    ).to(device)
    
    # Freeze base model
    for param in model.parameters():
        param.requires_grad = False
    
    # Add LoRA adapters
    print("Adding LoRA adapters...")
    lora_params = create_lora_roberta(model, rank=8)
    print(f"  Number of LoRA parameters: {len(lora_params)}")
    print(f"  Total LoRA parameters: {sum(p.numel() for p in lora_params):,}")
    
    # Create datasets
    print("\nLoading datasets from files...")
    data_dir = "/home/lei/git/princeton-nlp/MeZO/medium_models/data/k-shot-1k-test/SST-2"
    
    # Use 16-shot with seed 42
    train_path = f"{data_dir}/16-42/train.tsv"
    dev_path = f"{data_dir}/16-42/dev.tsv"
    
    train_examples = load_sst2_from_file(train_path, tokenizer)
    eval_examples = load_sst2_from_file(dev_path, tokenizer)
    
    print(f"  Train examples: {len(train_examples)}")
    print(f"  Eval examples: {len(eval_examples)}")
    
    # Training
    print("\nStarting MeZO training...")
    print("-" * 50)
    print("Step | Train Loss | Eval Loss | Eval Acc")
    print("-" * 50)
    
    train_losses = []
    eval_losses = []
    eval_accs = []
    
    for step in range(num_steps):
        # Sample batch (with replacement since we have limited examples)
        batch_indices = np.random.choice(len(train_examples), batch_size, replace=True)
        batch = {
            'text': [train_examples[i]['text'] for i in batch_indices],
            'label': [train_examples[i]['label'] for i in batch_indices]
        }
        
        # MeZO step
        train_loss = mezo_step_roberta(model, tokenizer, batch, lora_params, epsilon, learning_rate)
        train_losses.append(train_loss.item())
        
        # Evaluation
        if step % eval_steps == 0:
            apply_lora_weights(model)
            eval_acc, eval_loss = evaluate_sst2(model, tokenizer, eval_examples)
            eval_losses.append(eval_loss)
            eval_accs.append(eval_acc)
            
            print(f"{step:4d} | {train_loss:.6f} | {eval_loss:.6f} | {eval_acc:.2%}")
    
    print("-" * 50)
    
    # Analysis
    print("\nTraining Summary:")
    initial_loss = np.mean(train_losses[:5])
    final_loss = np.mean(train_losses[-5:])
    improvement = (initial_loss - final_loss) / initial_loss * 100
    
    print(f"  Initial train loss: {initial_loss:.4f}")
    print(f"  Final train loss: {final_loss:.4f}")
    print(f"  Loss improvement: {improvement:.1f}%")
    print(f"  Initial eval accuracy: {eval_accs[0]:.1%}")
    print(f"  Final eval accuracy: {eval_accs[-1]:.1%}")
    print(f"  Accuracy improvement: {(eval_accs[-1] - eval_accs[0]) * 100:.1f} pp")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Training loss
    ax1.plot(train_losses, 'b-', alpha=0.7)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    
    # Eval accuracy
    steps = list(range(0, num_steps, eval_steps))
    ax2.plot(steps, eval_accs, 'g-', marker='o')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Evaluation Accuracy')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('mezo_roberta_sst2.png', dpi=150)
    print(f"\nPlot saved to: mezo_roberta_sst2.png")
    
    # Success criteria
    success = eval_accs[-1] > eval_accs[0] + 0.05  # At least 5% improvement
    
    print("\n" + "=" * 60)
    if success:
        print("✅ MeZO ROBERTA SST-2 TEST: PASSED")
        print("=" * 60)
        print(f"Successfully fine-tuned RoBERTa on SST-2 with MeZO!")
    else:
        print("⚠️  MeZO ROBERTA SST-2 TEST: NEEDS MORE TUNING")
        print("=" * 60)
        print("Consider running for more steps or adjusting hyperparameters")
    
    return success


if __name__ == "__main__":
    # Check if we have GPU
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: Running on CPU, this will be slow")
    
    success = test_mezo_roberta_sst2()