#!/usr/bin/env python3
"""
Test MeZO with RoBERTa-large on SST-2 using accuracy as the optimization objective.
Following the original MeZO implementation that uses accuracy for classification tasks.
"""

import torch
import torch.nn.functional as F
import json
import os
import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import matplotlib.pyplot as plt
from typing import Dict, List
import logging

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
    lora_params = []
    device = model.device
    
    for name, module in model.named_modules():
        if 'attention' in name and hasattr(module, 'self'):
            if hasattr(module.self, 'query'):
                in_features = module.self.query.in_features
                out_features = module.self.query.out_features
                
                # Create LoRA parameters on the same device as the model
                lora_A_q = torch.nn.Parameter(torch.randn(rank, in_features, device=device) * 0.01)
                lora_B_q = torch.nn.Parameter(torch.zeros(out_features, rank, device=device))
                
                lora_A_v = torch.nn.Parameter(torch.randn(rank, in_features, device=device) * 0.01)
                lora_B_v = torch.nn.Parameter(torch.zeros(out_features, rank, device=device))
                
                # Store original weights
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


def compute_accuracy_objective(model, inputs, labels):
    """Compute accuracy-based objective (negative accuracy for minimization)."""
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Get predictions
        preds = torch.argmax(logits, dim=-1)
        
        # Compute accuracy
        correct = (preds == labels).float()
        accuracy = correct.mean()
        
        # Return negative accuracy so minimization maximizes accuracy
        return -accuracy


def mezo_step_roberta_accuracy(model, tokenizer, batch, lora_params, epsilon=1e-3, lr=1e-6):
    """Perform one MeZO step using accuracy as the objective."""
    # Sample perturbation
    z_list = [torch.randn_like(p) for p in lora_params]
    
    # Normalize perturbations
    z_list = [z / (z.norm() + 1e-8) for z in z_list]
    
    # Prepare inputs
    inputs = tokenizer(
        batch['text'], 
        padding=True, 
        truncation=True, 
        max_length=128,
        return_tensors='pt'
    ).to(model.device)
    
    labels = torch.tensor(batch['label']).to(model.device)
    
    # Apply positive perturbation
    for i, p in enumerate(lora_params):
        p.data.add_(epsilon * z_list[i])
    apply_lora_weights(model)
    
    # Forward pass with +epsilon (using accuracy objective)
    model.eval()
    obj_plus = compute_accuracy_objective(model, inputs, labels)
    
    # Apply negative perturbation
    for i, p in enumerate(lora_params):
        p.data.add_(-2 * epsilon * z_list[i])
    apply_lora_weights(model)
    
    # Forward pass with -epsilon (using accuracy objective)
    obj_minus = compute_accuracy_objective(model, inputs, labels)
    
    # Restore parameters
    for i, p in enumerate(lora_params):
        p.data.add_(epsilon * z_list[i])
    apply_lora_weights(model)
    
    # Compute gradient estimate
    grad_estimate = (obj_plus - obj_minus) / (2 * epsilon)
    
    # Update parameters
    for i, p in enumerate(lora_params):
        p.data.add_(-lr * grad_estimate * z_list[i])
    
    # Return positive accuracy for logging
    return -(obj_plus + obj_minus) / 2


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


def test_mezo_roberta_sst2_accuracy():
    """Test MeZO on RoBERTa with SST-2 using accuracy optimization."""
    print("=" * 60)
    print("MeZO RoBERTa SST-2 Test (Accuracy Optimization)")
    print("=" * 60)
    
    # Configuration
    model_name = "roberta-base"
    batch_size = 16  # Larger batch for more stable accuracy estimation
    learning_rate = 1e-3  # Higher LR for accuracy optimization
    epsilon = 1e-3
    num_steps = 500  # More steps with more data
    eval_steps = 50
    k_shot = 512  # Use 512-shot dataset
    seed = 42
    use_template = False  # Try without template for now
    
    print(f"\nConfiguration:")
    print(f"  Model: {model_name}")
    print(f"  Task: SST-2 ({k_shot}-shot)")
    print(f"  Optimization: ACCURACY (not loss)")
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
    
    # Load datasets
    print("\nLoading datasets from files...")
    data_dir = "/home/lei/git/princeton-nlp/MeZO/medium_models/data/k-shot-1k-test/SST-2"
    
    # Use 512-shot with seed 42
    train_path = f"{data_dir}/512-42/train.tsv"
    dev_path = f"{data_dir}/512-42/dev.tsv"
    
    train_examples = load_sst2_from_file(train_path, tokenizer, apply_template=use_template)
    eval_examples = load_sst2_from_file(dev_path, tokenizer, apply_template=use_template)
    
    print(f"  Train examples: {len(train_examples)}")
    print(f"  Eval examples: {len(eval_examples)}")
    
    # Training
    print("\nStarting MeZO training with ACCURACY optimization...")
    print("-" * 60)
    print("Step | Train Acc | Eval Loss | Eval Acc | Improvement")
    print("-" * 60)
    
    train_accs = []
    eval_losses = []
    eval_accs = []
    
    # Initial evaluation
    apply_lora_weights(model)
    initial_acc, initial_loss = evaluate_sst2(model, tokenizer, eval_examples)
    eval_accs.append(initial_acc)
    eval_losses.append(initial_loss)
    
    for step in range(num_steps):
        # Sample batch with replacement
        batch_indices = np.random.choice(len(train_examples), batch_size, replace=True)
        batch = {
            'text': [train_examples[i]['text'] for i in batch_indices],
            'label': [train_examples[i]['label'] for i in batch_indices]
        }
        
        # MeZO step with accuracy optimization
        train_acc = mezo_step_roberta_accuracy(model, tokenizer, batch, lora_params, epsilon, learning_rate)
        train_accs.append(train_acc)
        
        # Evaluation
        if (step + 1) % eval_steps == 0:
            apply_lora_weights(model)
            eval_acc, eval_loss = evaluate_sst2(model, tokenizer, eval_examples)
            eval_losses.append(eval_loss)
            eval_accs.append(eval_acc)
            
            improvement = (eval_acc - initial_acc) * 100
            print(f"{step+1:4d} | {train_acc:9.2%} | {eval_loss:.6f} | {eval_acc:8.2%} | {improvement:+.1f}pp")
    
    print("-" * 60)
    
    # Final evaluation
    apply_lora_weights(model)
    final_acc, final_loss = evaluate_sst2(model, tokenizer, eval_examples)
    
    # Analysis
    print("\nTraining Summary:")
    print(f"  Initial eval accuracy: {initial_acc:.1%}")
    print(f"  Final eval accuracy: {final_acc:.1%}")
    print(f"  Accuracy improvement: {(final_acc - initial_acc) * 100:+.1f} percentage points")
    print(f"  Average train accuracy: {np.mean([acc.item() if torch.is_tensor(acc) else acc for acc in train_accs]):.1%}")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Training accuracy
    train_accs_np = [acc.item() if torch.is_tensor(acc) else acc for acc in train_accs]
    ax1.plot(train_accs_np, 'b-', alpha=0.7)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Training Accuracy')
    ax1.set_title('Training Accuracy (Batch)')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Add smoothed line
    if len(train_accs_np) > 10:
        window = 10
        smoothed = np.convolve(train_accs_np, np.ones(window)/window, mode='valid')
        ax1.plot(range(window//2, len(train_accs_np)-window//2+1), smoothed, 'r-', 
                linewidth=2, label='Smoothed')
        ax1.legend()
    
    # Eval accuracy
    steps = list(range(0, num_steps + 1, eval_steps))
    ax2.plot(steps[:len(eval_accs)], eval_accs, 'g-', marker='o', markersize=8)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Evaluation Accuracy')
    ax2.set_title('Evaluation Accuracy')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    ax2.axhline(y=initial_acc, color='gray', linestyle='--', alpha=0.5, label='Initial')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('mezo_roberta_sst2_accuracy.png', dpi=150)
    print(f"\nPlot saved to: mezo_roberta_sst2_accuracy.png")
    
    # Success criteria
    success = final_acc > initial_acc + 0.05  # At least 5% improvement
    
    print("\n" + "=" * 60)
    if success:
        print("✅ MeZO ROBERTA SST-2 TEST (ACCURACY): PASSED")
        print("=" * 60)
        print(f"Successfully fine-tuned RoBERTa on SST-2 using accuracy optimization!")
        print(f"Achieved {(final_acc - initial_acc) * 100:+.1f}pp improvement")
    else:
        print("⚠️  MeZO ROBERTA SST-2 TEST (ACCURACY): NEEDS MORE STEPS")
        print("=" * 60)
        print(f"Current improvement: {(final_acc - initial_acc) * 100:+.1f}pp")
        print("Consider running for more steps (original uses 100K steps)")
    
    return success


if __name__ == "__main__":
    # Check if we have GPU
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: Running on CPU, this will be slow")
    
    success = test_mezo_roberta_sst2_accuracy()