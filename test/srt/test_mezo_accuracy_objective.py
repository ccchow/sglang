#!/usr/bin/env python3
"""
MeZO implementation using accuracy as the objective for SST-2.
Following the paper's approach for non-differentiable objectives.
"""

import torch
import numpy as np
import time
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import matplotlib.pyplot as plt


def compute_accuracy_objective(model, inputs, labels):
    """
    Compute negative accuracy as the objective for MeZO.
    The paper uses accuracy maximization, so we minimize negative accuracy.
    """
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Get predictions
        predictions = torch.argmax(logits, dim=-1)
        
        # Compute accuracy
        correct = (predictions == labels).float()
        accuracy = correct.mean().item()
        
        # Return negative accuracy for minimization
        return -accuracy


def mezo_step_accuracy(model, tokenizer, batch, lora_params, epsilon=1e-3, lr=1e-6):
    """
    MeZO step using accuracy as the objective (following the paper).
    """
    # Sample perturbation (NO normalization)
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
    
    # Compute negative accuracy
    obj_plus = compute_accuracy_objective(model, inputs, labels)
    
    # Forward pass with -epsilon*z
    for i, p in enumerate(lora_params):
        p.data.add_(-2 * epsilon * z_list[i])
    apply_lora(model)
    
    obj_minus = compute_accuracy_objective(model, inputs, labels)
    
    # Restore parameters
    for i, p in enumerate(lora_params):
        p.data.add_(epsilon * z_list[i])
    
    # Gradient estimate
    grad_est = (obj_plus - obj_minus) / (2 * epsilon)
    
    # Update parameters
    for i, p in enumerate(lora_params):
        p.data.add_(-lr * grad_est * z_list[i])
    
    apply_lora(model)
    
    # Return average objective (negative accuracy)
    avg_obj = (obj_plus + obj_minus) / 2
    
    # Also compute cross-entropy loss for monitoring
    with torch.no_grad():
        outputs = model(**inputs, labels=labels)
        ce_loss = outputs.loss.item()
    
    return -avg_obj, ce_loss  # Return positive accuracy and CE loss


def create_lora_layers(model, rank=8, alpha=16):
    """Add LoRA to RoBERTa."""
    lora_params = []
    device = model.device
    
    for name, module in model.named_modules():
        if 'attention' in name and hasattr(module, 'self'):
            if hasattr(module.self, 'query'):
                layer = module.self.query
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
    """Apply LoRA weights."""
    for name, module in model.named_modules():
        if 'attention' in name and hasattr(module, 'self'):
            if hasattr(module.self.query, 'lora_A'):
                layer = module.self.query
                layer.weight.data = layer.original_weight + \
                    layer.lora_scale * (layer.lora_B @ layer.lora_A)


def load_sst2_data(file_path, max_examples=None):
    """Load SST-2 data."""
    examples = []
    with open(file_path, 'r') as f:
        lines = f.readlines()[1:]  # Skip header
        for i, line in enumerate(lines):
            if max_examples and i >= max_examples:
                break
            parts = line.strip().split('\t')
            if len(parts) == 2:
                text, label = parts
                examples.append({'text': text, 'label': int(label)})
    return examples


def evaluate_full(model, tokenizer, examples):
    """Full evaluation."""
    model.eval()
    correct = 0
    total_loss = 0
    
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
            
            # Also compute loss
            labels = torch.tensor([ex['label']]).to(model.device)
            loss = torch.nn.functional.cross_entropy(outputs.logits, labels)
            total_loss += loss.item()
    
    accuracy = correct / len(examples)
    avg_loss = total_loss / len(examples)
    
    return accuracy, avg_loss


def main():
    print("=" * 80)
    print("MeZO RoBERTa SST-2 with Accuracy Objective (Paper Setting)")
    print("=" * 80)
    
    # Configuration from paper
    model_name = "roberta-base"
    batch_size = 64
    learning_rate = 1e-6
    epsilon = 1e-3
    num_steps = 5000
    eval_interval = 500
    
    print(f"\nConfiguration:")
    print(f"  Model: {model_name}")
    print(f"  Objective: ACCURACY (not cross-entropy)")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Epsilon: {epsilon}")
    print(f"  Steps: {num_steps}")
    
    # Load model
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
    eval_data = load_sst2_data(f"{data_dir}/512-42/dev.tsv", 200)
    
    print(f"  Train: {len(train_data)} examples")
    print(f"  Eval: {len(eval_data)} examples")
    
    # Initial evaluation
    print("\nInitial evaluation...")
    apply_lora(model)
    init_acc, init_loss = evaluate_full(model, tokenizer, eval_data)
    print(f"  Accuracy: {init_acc:.1%}")
    print(f"  Loss: {init_loss:.4f}")
    
    # Training
    print(f"\nTraining with ACCURACY objective...")
    print("-" * 70)
    print("Step  | Train Acc | CE Loss | Eval Acc | Eval Loss | Time")
    print("-" * 70)
    
    train_accs = []
    ce_losses = []
    eval_accs = [init_acc]
    eval_losses = [init_loss]
    
    start_time = time.time()
    
    for step in range(num_steps):
        # Sample batch
        idx = np.random.choice(len(train_data), batch_size, replace=True)
        batch = {
            'texts': [train_data[i]['text'] for i in idx],
            'labels': [train_data[i]['label'] for i in idx]
        }
        
        # MeZO step with accuracy objective
        train_acc, ce_loss = mezo_step_accuracy(
            model, tokenizer, batch, lora_params, epsilon, learning_rate
        )
        
        train_accs.append(train_acc)
        ce_losses.append(ce_loss)
        
        # Evaluate
        if (step + 1) % eval_interval == 0:
            eval_acc, eval_loss = evaluate_full(model, tokenizer, eval_data)
            eval_accs.append(eval_acc)
            eval_losses.append(eval_loss)
            
            elapsed = time.time() - start_time
            print(f"{step+1:5d} | {train_acc:9.1%} | {ce_loss:7.4f} | "
                  f"{eval_acc:8.1%} | {eval_loss:9.4f} | {elapsed:5.0f}s")
    
    print("-" * 70)
    
    # Final evaluation
    final_acc, final_loss = evaluate_full(model, tokenizer, eval_data)
    total_time = time.time() - start_time
    
    print(f"\nTraining complete in {total_time:.0f}s")
    print(f"Final accuracy: {final_acc:.1%} (from {init_acc:.1%})")
    print(f"Improvement: {(final_acc - init_acc) * 100:+.1f} percentage points")
    print(f"Loss change: {init_loss:.4f} → {final_loss:.4f}")
    
    # Analysis
    print("\nTraining dynamics:")
    early_acc = np.mean(train_accs[:100])
    late_acc = np.mean(train_accs[-100:])
    print(f"  Train accuracy: {early_acc:.1%} → {late_acc:.1%}")
    
    early_loss = np.mean(ce_losses[:100])
    late_loss = np.mean(ce_losses[-100:])
    print(f"  CE loss: {early_loss:.4f} → {late_loss:.4f}")
    
    # Plot results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Training accuracy (what we optimize)
    ax1.plot(train_accs, alpha=0.5)
    if len(train_accs) > 50:
        smoothed = np.convolve(train_accs, np.ones(50)/50, mode='valid')
        ax1.plot(smoothed, linewidth=2, label='Smoothed')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Training Accuracy')
    ax1.set_title('Training Accuracy (Optimized Objective)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # CE loss (for monitoring)
    ax2.plot(ce_losses, alpha=0.5, color='orange')
    if len(ce_losses) > 50:
        smoothed = np.convolve(ce_losses, np.ones(50)/50, mode='valid')
        ax2.plot(smoothed, linewidth=2, color='red')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Cross-Entropy Loss')
    ax2.set_title('CE Loss (Monitoring Only)')
    ax2.grid(True, alpha=0.3)
    
    # Eval metrics
    eval_steps = list(range(0, num_steps+1, eval_interval))[:len(eval_accs)]
    
    ax3.plot(eval_steps, eval_accs, 'o-', color='green')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Evaluation Accuracy')
    ax3.set_title('Evaluation Accuracy')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    ax4.plot(eval_steps[1:], eval_losses[1:], 'o-', color='red')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Evaluation Loss')
    ax4.set_title('Evaluation Loss')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mezo_accuracy_objective.png', dpi=150)
    print(f"\nPlots saved to: mezo_accuracy_objective.png")
    
    print("\n" + "=" * 80)
    if final_acc > init_acc + 0.02:
        print("✅ SUCCESS: MeZO with accuracy objective is working!")
        print(f"   Achieved {(final_acc-init_acc)*100:.1f}pp improvement")
    else:
        print("⚠️  Limited improvement - more steps needed")
        print("   Paper uses 100K steps for full convergence")
    
    return final_acc > init_acc


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    main()