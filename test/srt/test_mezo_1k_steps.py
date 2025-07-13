#!/usr/bin/env python3
"""
1K step test of MeZO with accuracy objective on RoBERTa SST-2.
Following the paper's exact approach.
"""

import torch
import numpy as np
import time
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import matplotlib.pyplot as plt
from datetime import datetime


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


def create_lora_layers(model, rank=8, alpha=16):
    """Add LoRA to RoBERTa attention layers."""
    lora_params = []
    device = model.device
    
    for name, module in model.named_modules():
        if 'attention' in name and hasattr(module, 'self'):
            # Add LoRA to query and value projections
            for proj_name in ['query', 'value']:
                if hasattr(module.self, proj_name):
                    layer = getattr(module.self, proj_name)
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
            for proj_name in ['query', 'value']:
                if hasattr(module.self, proj_name):
                    layer = getattr(module.self, proj_name)
                    if hasattr(layer, 'lora_A'):
                        layer.weight.data = layer.original_weight + \
                            layer.lora_scale * (layer.lora_B @ layer.lora_A)


def compute_accuracy_objective(model, inputs, labels):
    """Compute negative accuracy (for minimization)."""
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
        accuracy = (predictions == labels).float().mean().item()
        return -accuracy  # Negative for minimization


def mezo_step_accuracy(model, tokenizer, batch, lora_params, epsilon=1e-3, lr=1e-6):
    """MeZO step using accuracy objective."""
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
    
    # Return training accuracy and gradient info
    train_acc = -(obj_plus + obj_minus) / 2
    
    # Also compute CE loss for monitoring
    with torch.no_grad():
        outputs = model(**inputs, labels=labels)
        ce_loss = outputs.loss.item()
    
    return train_acc, ce_loss, grad_est


def evaluate(model, tokenizer, examples, max_examples=None):
    """Evaluate model on examples."""
    model.eval()
    correct = 0
    total_loss = 0
    
    eval_examples = examples[:max_examples] if max_examples else examples
    
    for ex in eval_examples:
        inputs = tokenizer(
            ex['text'],
            truncation=True,
            max_length=128,
            return_tensors='pt'
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=-1).item()
            correct += (pred == ex['label'])
            
            labels = torch.tensor([ex['label']]).to(model.device)
            loss = torch.nn.functional.cross_entropy(outputs.logits, labels)
            total_loss += loss.item()
    
    accuracy = correct / len(eval_examples)
    avg_loss = total_loss / len(eval_examples)
    
    return accuracy, avg_loss


def run_1k_step_test():
    """Run 1K step MeZO test with accuracy objective."""
    print("=" * 80)
    print("MeZO 1K Step Test - RoBERTa SST-2 with Accuracy Objective")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration from paper
    model_name = "roberta-base"  # Using base for faster execution
    batch_size = 64  # Paper setting
    learning_rate = 1e-6  # Paper setting
    epsilon = 1e-3  # Paper setting
    num_steps = 1000
    eval_interval = 100
    
    print(f"\nConfiguration (MeZO paper settings):")
    print(f"  Model: {model_name}")
    print(f"  Task: SST-2")
    print(f"  Objective: ACCURACY (not cross-entropy)")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Epsilon: {epsilon}")
    print(f"  Steps: {num_steps}")
    print(f"  Perturbations: Unnormalized (paper default)")
    
    # Load model
    print("\n[1/6] Loading model and tokenizer...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    ).to(device)
    
    # Freeze base model
    for param in model.parameters():
        param.requires_grad = False
    
    # Add LoRA
    print("\n[2/6] Adding LoRA adapters...")
    lora_params = create_lora_layers(model, rank=8, alpha=16)
    total_params = sum(p.numel() for p in lora_params)
    print(f"  LoRA parameters: {total_params:,}")
    print(f"  Number of LoRA matrices: {len(lora_params)}")
    
    # Load data
    print("\n[3/6] Loading SST-2 dataset...")
    data_dir = "/home/lei/git/princeton-nlp/MeZO/medium_models/data/k-shot-1k-test/SST-2"
    train_data = load_sst2_data(f"{data_dir}/512-42/train.tsv")
    eval_data = load_sst2_data(f"{data_dir}/512-42/dev.tsv", max_examples=200)
    
    print(f"  Train: {len(train_data)} examples")
    print(f"  Eval: {len(eval_data)} examples")
    
    # Initial evaluation
    print("\n[4/6] Initial evaluation...")
    apply_lora(model)
    init_acc, init_loss = evaluate(model, tokenizer, eval_data)
    print(f"  Initial accuracy: {init_acc:.1%}")
    print(f"  Initial CE loss: {init_loss:.4f}")
    
    # Training
    print(f"\n[5/6] Training for {num_steps} steps with ACCURACY objective...")
    print("-" * 80)
    print("Step  | Train Acc | CE Loss | Grad Est | Eval Acc | Eval Loss | Time")
    print("-" * 80)
    
    # Tracking
    train_accs = []
    ce_losses = []
    grad_estimates = []
    eval_accs = [init_acc]
    eval_losses = [init_loss]
    zero_grad_count = 0
    
    start_time = time.time()
    
    for step in range(num_steps):
        # Sample batch
        idx = np.random.choice(len(train_data), batch_size, replace=True)
        batch = {
            'texts': [train_data[i]['text'] for i in idx],
            'labels': [train_data[i]['label'] for i in idx]
        }
        
        # MeZO step with accuracy objective
        train_acc, ce_loss, grad_est = mezo_step_accuracy(
            model, tokenizer, batch, lora_params, epsilon, learning_rate
        )
        
        train_accs.append(train_acc)
        ce_losses.append(ce_loss)
        grad_estimates.append(abs(grad_est))
        
        if grad_est == 0:
            zero_grad_count += 1
        
        # Evaluate periodically
        if (step + 1) % eval_interval == 0:
            eval_acc, eval_loss = evaluate(model, tokenizer, eval_data)
            eval_accs.append(eval_acc)
            eval_losses.append(eval_loss)
            
            elapsed = time.time() - start_time
            print(f"{step+1:5d} | {train_acc:9.1%} | {ce_loss:7.4f} | {grad_est:8.5f} | "
                  f"{eval_acc:8.1%} | {eval_loss:9.4f} | {elapsed:5.0f}s")
    
    print("-" * 80)
    
    # Final evaluation
    print("\n[6/6] Final evaluation...")
    final_acc, final_loss = evaluate(model, tokenizer, eval_data)
    total_time = time.time() - start_time
    
    # Results summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    print(f"\nTraining time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Time per step: {total_time/num_steps:.2f}s")
    
    print(f"\nAccuracy improvement:")
    print(f"  Initial: {init_acc:.1%}")
    print(f"  Final: {final_acc:.1%}")
    print(f"  Change: {(final_acc - init_acc) * 100:+.2f} percentage points")
    
    print(f"\nCE Loss change:")
    print(f"  Initial: {init_loss:.4f}")
    print(f"  Final: {final_loss:.4f}")
    print(f"  Change: {final_loss - init_loss:+.4f}")
    
    print(f"\nTraining dynamics:")
    print(f"  Average train accuracy: {np.mean(train_accs):.1%}")
    print(f"  Average CE loss: {np.mean(ce_losses):.4f}")
    print(f"  Zero gradient steps: {zero_grad_count}/{num_steps} ({zero_grad_count/num_steps*100:.1f}%)")
    print(f"  Average |gradient|: {np.mean(grad_estimates):.6f}")
    
    # Trend analysis
    early_acc = np.mean(train_accs[:100])
    late_acc = np.mean(train_accs[-100:])
    print(f"\nTraining accuracy trend:")
    print(f"  First 100 steps: {early_acc:.1%}")
    print(f"  Last 100 steps: {late_acc:.1%}")
    print(f"  Change: {(late_acc - early_acc) * 100:+.2f}pp")
    
    # Create plots
    print("\nGenerating plots...")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Training accuracy
    ax1.plot(train_accs, alpha=0.5, color='blue')
    if len(train_accs) > 50:
        smoothed = np.convolve(train_accs, np.ones(50)/50, mode='valid')
        ax1.plot(range(25, len(train_accs)-24), smoothed, 'b-', linewidth=2)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Training Accuracy')
    ax1.set_title('Training Accuracy (Optimized Objective)')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # CE Loss
    ax2.plot(ce_losses, alpha=0.5, color='orange')
    if len(ce_losses) > 50:
        smoothed = np.convolve(ce_losses, np.ones(50)/50, mode='valid')
        ax2.plot(range(25, len(ce_losses)-24), smoothed, 'r-', linewidth=2)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Cross-Entropy Loss')
    ax2.set_title('CE Loss (Monitoring Only)')
    ax2.grid(True, alpha=0.3)
    
    # Evaluation metrics
    eval_steps = list(range(0, num_steps+1, eval_interval))[:len(eval_accs)]
    
    ax3.plot(eval_steps, eval_accs, 'o-', color='green', markersize=8)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Evaluation Accuracy')
    ax3.set_title('Evaluation Accuracy')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    ax3.axhline(y=init_acc, color='gray', linestyle='--', alpha=0.5, label='Initial')
    ax3.legend()
    
    # Gradient estimates
    ax4.plot(grad_estimates, alpha=0.5, color='purple')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('|Gradient Estimate|')
    ax4.set_title(f'Gradient Magnitude ({zero_grad_count/num_steps*100:.0f}% zeros)')
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('symlog')
    
    plt.tight_layout()
    plt.savefig('mezo_1k_steps_results.png', dpi=150)
    print(f"Plots saved to: mezo_1k_steps_results.png")
    
    # Final verdict
    print("\n" + "=" * 80)
    if final_acc > init_acc + 0.01:  # 1% improvement threshold
        print("✅ SUCCESS: MeZO with accuracy objective shows improvement!")
        print(f"   Achieved {(final_acc-init_acc)*100:.2f}pp gain in 1K steps")
    elif final_acc > init_acc:
        print("⚠️  MARGINAL: Small improvement detected")
        print(f"   Only {(final_acc-init_acc)*100:.2f}pp gain - more steps needed")
    else:
        print("❌ NO IMPROVEMENT: Accuracy unchanged or decreased")
        print("   This is common with accuracy objective in early training")
    
    print("\nKey insights:")
    print(f"- {zero_grad_count/num_steps*100:.0f}% of steps had zero gradient")
    print("- Accuracy objective requires many more steps than CE loss")
    print("- Paper uses 100K steps for good reason!")
    print("=" * 80)
    
    return {
        'initial_acc': init_acc,
        'final_acc': final_acc,
        'improvement': final_acc - init_acc,
        'zero_grad_ratio': zero_grad_count / num_steps,
        'time': total_time
    }


if __name__ == "__main__":
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Run the test
    results = run_1k_step_test()
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")