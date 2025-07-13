#!/usr/bin/env python3
"""
10K step test of MeZO using MLM approach (as in the paper) on RoBERTa SST-2.
This should show actual accuracy improvements with continuous gradients.
"""

import torch
import numpy as np
import time
from transformers import RobertaTokenizer, RobertaForMaskedLM
import matplotlib.pyplot as plt
from datetime import datetime
import json


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


def create_balanced_eval_set(examples, size_per_class=50):
    """Create a balanced evaluation set."""
    pos_examples = [ex for ex in examples if ex['label'] == 1]
    neg_examples = [ex for ex in examples if ex['label'] == 0]
    
    # Take equal number from each class
    balanced = pos_examples[:size_per_class] + neg_examples[:size_per_class]
    
    # Shuffle
    np.random.shuffle(balanced)
    
    return balanced


def create_lora_layers(model, rank=8, alpha=16):
    """Add LoRA to RoBERTa MLM model attention layers."""
    lora_params = []
    device = model.device
    
    # Add LoRA to attention layers in RoBERTa
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


def compute_mlm_loss(model, tokenizer, inputs, labels, label_word_ids):
    """Compute MLM loss using label words."""
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Find mask positions
        mask_token_id = tokenizer.mask_token_id
        mask_positions = (inputs.input_ids == mask_token_id).nonzero(as_tuple=True)[1]
        
        # Get logits at mask positions
        batch_size = inputs.input_ids.size(0)
        mask_logits = logits[torch.arange(batch_size), mask_positions]
        
        # Extract logits for label words only
        label_logits = mask_logits[:, label_word_ids]
        
        # Compute cross-entropy loss
        loss = torch.nn.functional.cross_entropy(label_logits, labels)
        
        return loss.item(), label_logits


def mezo_step_mlm(model, tokenizer, batch, lora_params, label_word_ids, epsilon=1e-3, lr=1e-6):
    """MeZO step using MLM objective."""
    # Sample perturbation (NO normalization)
    z_list = [torch.randn_like(p) for p in lora_params]
    
    # Format inputs with MLM template: "[text] It was [MASK]."
    mlm_texts = [f"{text} It was {tokenizer.mask_token}." for text in batch['texts']]
    
    inputs = tokenizer(
        mlm_texts,
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
    
    loss_plus, _ = compute_mlm_loss(model, tokenizer, inputs, labels, label_word_ids)
    
    # Forward pass with -epsilon*z
    for i, p in enumerate(lora_params):
        p.data.add_(-2 * epsilon * z_list[i])
    apply_lora(model)
    
    loss_minus, _ = compute_mlm_loss(model, tokenizer, inputs, labels, label_word_ids)
    
    # Restore parameters
    for i, p in enumerate(lora_params):
        p.data.add_(epsilon * z_list[i])
    
    # Gradient estimate
    grad_est = (loss_plus - loss_minus) / (2 * epsilon)
    
    # Update parameters
    for i, p in enumerate(lora_params):
        p.data.add_(-lr * grad_est * z_list[i])
    
    apply_lora(model)
    
    # Compute accuracy for monitoring
    _, label_logits = compute_mlm_loss(model, tokenizer, inputs, labels, label_word_ids)
    preds = torch.argmax(label_logits, dim=-1)
    accuracy = (preds == labels).float().mean().item()
    
    avg_loss = (loss_plus + loss_minus) / 2
    
    return accuracy, avg_loss, grad_est


def evaluate_mlm(model, tokenizer, examples, label_word_ids):
    """Evaluate model using MLM approach."""
    model.eval()
    correct = 0
    total_loss = 0
    predictions = []
    
    for ex in examples:
        # Format with MLM template
        mlm_text = f"{ex['text']} It was {tokenizer.mask_token}."
        inputs = tokenizer(
            mlm_text,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        ).to(model.device)
        
        label = torch.tensor([ex['label']]).to(model.device)
        
        loss, label_logits = compute_mlm_loss(model, tokenizer, inputs, label, label_word_ids)
        total_loss += loss
        
        pred = torch.argmax(label_logits, dim=-1).item()
        predictions.append(pred)
        correct += (pred == ex['label'])
    
    accuracy = correct / len(examples)
    avg_loss = total_loss / len(examples)
    
    return accuracy, avg_loss, predictions


def run_mlm_10k_step_test():
    """Run 10K step MeZO test with MLM objective."""
    print("=" * 80)
    print("MeZO 10K Step Test - RoBERTa SST-2 with MLM Objective")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration from paper
    model_name = "roberta-base"  # Using base for faster execution
    batch_size = 64  # Paper setting
    learning_rate = 1e-6  # Paper setting
    epsilon = 1e-3  # Paper setting
    num_steps = 10000  # 10K steps (paper uses 100K)
    eval_interval = 1000  # Evaluate every 1K steps
    
    print(f"\nConfiguration (MeZO paper settings):")
    print(f"  Model: {model_name} (MLM)")
    print(f"  Task: SST-2")
    print(f"  Template: '[text] It was [MASK].'")
    print(f"  Label words: {{'terrible': 0, 'great': 1}}")
    print(f"  Objective: Cross-entropy on vocabulary logits")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Epsilon: {epsilon}")
    print(f"  Steps: {num_steps:,}")
    print(f"  Eval interval: {eval_interval:,}")
    
    # Load model and tokenizer
    print("\n[1/6] Loading MLM model and tokenizer...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForMaskedLM.from_pretrained(model_name).to(device)
    
    # Get label word IDs with space prefix
    terrible_token = tokenizer.tokenize(' terrible')[0]
    great_token = tokenizer.tokenize(' great')[0]
    terrible_id = tokenizer.convert_tokens_to_ids(terrible_token)
    great_id = tokenizer.convert_tokens_to_ids(great_token)
    label_word_ids = [terrible_id, great_id]
    
    print(f"  Label word tokens: ' terrible'={terrible_token} (id={terrible_id}), ' great'={great_token} (id={great_id})")
    
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
    
    # Create balanced eval set
    full_dev_data = load_sst2_data(f"{data_dir}/512-42/dev.tsv")
    eval_data = create_balanced_eval_set(full_dev_data, size_per_class=100)
    
    print(f"  Train: {len(train_data)} examples")
    print(f"  Eval: {len(eval_data)} examples (balanced)")
    
    # Initial evaluation
    print("\n[4/6] Initial evaluation...")
    apply_lora(model)
    init_acc, init_loss, init_preds = evaluate_mlm(model, tokenizer, eval_data, label_word_ids)
    print(f"  Initial accuracy: {init_acc:.1%}")
    print(f"  Initial MLM loss: {init_loss:.4f}")
    
    # Check initial prediction distribution
    init_pred_dist = {0: sum(1 for p in init_preds if p == 0), 
                      1: sum(1 for p in init_preds if p == 1)}
    print(f"  Initial predictions: {init_pred_dist[0]} terrible, {init_pred_dist[1]} great")
    
    # Training
    print(f"\n[5/6] Training for {num_steps:,} steps with MLM objective...")
    print("-" * 90)
    print("Step   | Train Acc | MLM Loss | Grad Est | Eval Acc | Eval Loss | Pred Dist | Time")
    print("-" * 90)
    
    # Tracking
    train_accs = []
    mlm_losses = []
    grad_estimates = []
    eval_accs = [init_acc]
    eval_losses = [init_loss]
    zero_grad_count = 0
    
    # Best model tracking
    best_eval_acc = init_acc
    best_step = 0
    
    start_time = time.time()
    
    for step in range(num_steps):
        # Sample batch
        idx = np.random.choice(len(train_data), batch_size, replace=True)
        batch = {
            'texts': [train_data[i]['text'] for i in idx],
            'labels': [train_data[i]['label'] for i in idx]
        }
        
        # MeZO step with MLM objective
        train_acc, mlm_loss, grad_est = mezo_step_mlm(
            model, tokenizer, batch, lora_params, label_word_ids, epsilon, learning_rate
        )
        
        train_accs.append(train_acc)
        mlm_losses.append(mlm_loss)
        grad_estimates.append(abs(grad_est))
        
        if grad_est == 0:
            zero_grad_count += 1
        
        # Evaluate periodically
        if (step + 1) % eval_interval == 0:
            eval_acc, eval_loss, eval_preds = evaluate_mlm(model, tokenizer, eval_data, label_word_ids)
            eval_accs.append(eval_acc)
            eval_losses.append(eval_loss)
            
            # Track best model
            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                best_step = step + 1
            
            # Prediction distribution
            pred_dist = {0: sum(1 for p in eval_preds if p == 0), 
                        1: sum(1 for p in eval_preds if p == 1)}
            
            elapsed = time.time() - start_time
            print(f"{step+1:6d} | {train_acc:9.1%} | {mlm_loss:8.4f} | {grad_est:8.5f} | "
                  f"{eval_acc:8.1%} | {eval_loss:9.4f} | T:{pred_dist[0]:3d} G:{pred_dist[1]:3d} | {elapsed:5.0f}s")
    
    print("-" * 90)
    
    # Final evaluation
    print("\n[6/6] Final evaluation...")
    final_acc, final_loss, final_preds = evaluate_mlm(model, tokenizer, eval_data, label_word_ids)
    total_time = time.time() - start_time
    
    # Final prediction distribution
    final_pred_dist = {0: sum(1 for p in final_preds if p == 0), 
                       1: sum(1 for p in final_preds if p == 1)}
    
    # Results summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    print(f"\nTraining time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Time per step: {total_time/num_steps:.2f}s")
    print(f"Steps per minute: {num_steps/(total_time/60):.1f}")
    
    print(f"\nAccuracy improvement:")
    print(f"  Initial: {init_acc:.1%}")
    print(f"  Final: {final_acc:.1%}")
    print(f"  Best: {best_eval_acc:.1%} (at step {best_step})")
    print(f"  Change: {(final_acc - init_acc) * 100:+.2f} percentage points")
    
    print(f"\nMLM Loss change:")
    print(f"  Initial: {init_loss:.4f}")
    print(f"  Final: {final_loss:.4f}")
    print(f"  Change: {final_loss - init_loss:+.4f} ({(final_loss - init_loss)/init_loss*100:+.1f}%)")
    
    print(f"\nPrediction distribution change:")
    print(f"  Initial: {init_pred_dist[0]} terrible, {init_pred_dist[1]} great")
    print(f"  Final: {final_pred_dist[0]} terrible, {final_pred_dist[1]} great")
    
    print(f"\nTraining dynamics:")
    print(f"  Average train accuracy: {np.mean(train_accs):.1%}")
    print(f"  Average MLM loss: {np.mean(mlm_losses):.4f}")
    print(f"  Zero gradient steps: {zero_grad_count}/{num_steps} ({zero_grad_count/num_steps*100:.1f}%)")
    print(f"  Average |gradient|: {np.mean(grad_estimates):.6f}")
    
    # Gradient statistics
    non_zero_grads = [g for g in grad_estimates if g > 0]
    if non_zero_grads:
        print(f"\nGradient statistics:")
        print(f"  Min |gradient|: {min(non_zero_grads):.6f}")
        print(f"  Max |gradient|: {max(grad_estimates):.6f}")
        print(f"  Std |gradient|: {np.std(grad_estimates):.6f}")
    
    # Create plots
    print("\nGenerating plots...")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('MeZO 10K Steps - RoBERTa SST-2 with MLM Objective', fontsize=16)
    
    # Training accuracy
    ax1.plot(train_accs, alpha=0.3, color='blue')
    if len(train_accs) > 100:
        smoothed = np.convolve(train_accs, np.ones(100)/100, mode='valid')
        ax1.plot(range(50, len(train_accs)-49), smoothed, 'b-', linewidth=2)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Training Accuracy')
    ax1.set_title('Training Accuracy (MLM Predictions)')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # MLM Loss
    ax2.plot(mlm_losses, alpha=0.3, color='orange')
    if len(mlm_losses) > 100:
        smoothed = np.convolve(mlm_losses, np.ones(100)/100, mode='valid')
        ax2.plot(range(50, len(mlm_losses)-49), smoothed, 'r-', linewidth=2)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('MLM Loss')
    ax2.set_title('MLM Cross-Entropy Loss (Optimized)')
    ax2.grid(True, alpha=0.3)
    
    # Evaluation metrics
    eval_steps = list(range(0, num_steps+1, eval_interval))[:len(eval_accs)]
    
    ax3.plot(eval_steps, eval_accs, 'o-', color='green', markersize=8, linewidth=2)
    ax3.plot(eval_steps, eval_losses, 'o-', color='red', markersize=6, linewidth=1, alpha=0.7)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Metric Value')
    ax3.set_title('Evaluation Metrics')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=init_acc, color='gray', linestyle='--', alpha=0.5, label='Initial Acc')
    ax3.axhline(y=0.5, color='black', linestyle=':', alpha=0.5, label='Random')
    ax3.legend(['Accuracy', 'Loss', 'Initial', 'Random'])
    
    # Gradient magnitude over time
    ax4.scatter(range(len(grad_estimates)), grad_estimates, alpha=0.1, s=1, color='purple')
    if len(grad_estimates) > 100:
        # Show moving average
        smoothed = np.convolve(grad_estimates, np.ones(100)/100, mode='valid')
        ax4.plot(range(50, len(grad_estimates)-49), smoothed, 'purple', linewidth=2)
    ax4.set_xlabel('Step')
    ax4.set_ylabel('|Gradient Estimate|')
    ax4.set_title(f'Gradient Magnitude ({zero_grad_count/num_steps*100:.1f}% zeros)')
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    ax4.set_ylim(1e-3, max(grad_estimates)*2)
    
    plt.tight_layout()
    plt.savefig('mezo_mlm_10k_steps_results.png', dpi=150)
    print(f"Plots saved to: mezo_mlm_10k_steps_results.png")
    
    # Save results to JSON
    results = {
        'config': {
            'model': model_name,
            'steps': num_steps,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'epsilon': epsilon,
            'lora_rank': 8,
            'lora_alpha': 16
        },
        'results': {
            'initial_accuracy': init_acc,
            'final_accuracy': final_acc,
            'best_accuracy': best_eval_acc,
            'best_step': best_step,
            'accuracy_improvement': final_acc - init_acc,
            'initial_loss': init_loss,
            'final_loss': final_loss,
            'loss_reduction': init_loss - final_loss,
            'zero_gradient_ratio': zero_grad_count / num_steps,
            'avg_gradient': float(np.mean(grad_estimates)),
            'training_time_seconds': total_time,
            'training_time_minutes': total_time / 60
        },
        'dynamics': {
            'eval_accuracies': eval_accs,
            'eval_losses': eval_losses,
            'eval_steps': eval_steps
        }
    }
    
    with open('mezo_mlm_10k_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Results saved to: mezo_mlm_10k_results.json")
    
    # Final verdict
    print("\n" + "=" * 80)
    if final_acc > init_acc + 0.05:  # 5% improvement threshold
        print("✅ SUCCESS: MeZO with MLM objective shows significant improvement!")
        print(f"   Achieved {(final_acc-init_acc)*100:.2f}pp gain in 10K steps")
    elif final_acc > init_acc + 0.01:
        print("📈 PROGRESS: Clear improvement with MLM approach")
        print(f"   Gained {(final_acc-init_acc)*100:.2f}pp - trending upward")
    elif abs(final_acc - init_acc) <= 0.01:
        print("⚠️  MARGINAL: Little improvement yet")
        print(f"   Change: {(final_acc-init_acc)*100:.2f}pp - needs more steps")
    else:
        print("📉 UNEXPECTED: Accuracy decreased")
        print(f"   Lost {(init_acc-final_acc)*100:.2f}pp - check hyperparameters")
    
    print("\nKey insights:")
    print(f"- {zero_grad_count/num_steps*100:.1f}% zero gradients (vs 100% with accuracy objective)")
    print(f"- Continuous optimization with average gradient {np.mean(grad_estimates):.4f}")
    print(f"- Loss reduced by {(init_loss - final_loss)/init_loss*100:.1f}%")
    print(f"- Paper uses 100K steps - we're at 10% of that")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Run the test
    results = run_mlm_10k_step_test()
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")