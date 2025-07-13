#!/usr/bin/env python3
"""
MeZO test with paper-aligned hyperparameters and implementation.
This test demonstrates MeZO working correctly with the original paper's settings.
"""

import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaForMaskedLM
import matplotlib.pyplot as plt
import time
import logging
from typing import List, Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_sst2_data(file_path: str, max_examples: int = None) -> List[Dict]:
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
                if max_examples and len(examples) >= max_examples:
                    break
    return examples


def create_lora_roberta(model, rank=8, target_modules=["query", "value"]):
    """
    Add LoRA adapters to RoBERTa model following the paper.
    
    Args:
        model: The RoBERTa model
        rank: LoRA rank (paper default: 8)
        target_modules: Which modules to apply LoRA to
    """
    lora_params = []
    device = model.device
    
    # Add LoRA to attention layers
    for name, module in model.named_modules():
        if 'attention' in name and hasattr(module, 'self'):
            for target in target_modules:
                if hasattr(module.self, target):
                    layer = getattr(module.self, target)
                    in_features = layer.in_features
                    out_features = layer.out_features
                    
                    # Initialize LoRA matrices (Kaiming initialization)
                    lora_A = torch.nn.Parameter(
                        torch.randn(rank, in_features, device=device) * np.sqrt(2.0 / rank)
                    )
                    lora_B = torch.nn.Parameter(
                        torch.zeros(out_features, rank, device=device)
                    )
                    
                    # Store original weight
                    layer.original_weight = layer.weight.data.clone()
                    
                    # Attach LoRA parameters
                    layer.lora_A = lora_A
                    layer.lora_B = lora_B
                    layer.lora_scale = 16.0 / rank  # alpha / r from paper
                    
                    lora_params.extend([lora_A, lora_B])
    
    return lora_params


def apply_lora_weights(model):
    """Apply LoRA weights to the model."""
    for name, module in model.named_modules():
        if 'attention' in name and hasattr(module, 'self'):
            for attr in ['query', 'value']:
                if hasattr(module.self, attr):
                    layer = getattr(module.self, attr)
                    if hasattr(layer, 'lora_A'):
                        # W = W_original + (alpha/r) * B @ A
                        layer.weight.data = layer.original_weight + \
                            layer.lora_scale * (layer.lora_B @ layer.lora_A)


def mezo_step_paper_aligned(
    model, 
    tokenizer, 
    batch: Dict[str, List], 
    lora_params: List[torch.nn.Parameter], 
    epsilon: float = 1e-3, 
    lr: float = 1e-6,
    weight_decay: float = 0.0
) -> Tuple[float, Dict[str, float]]:
    """
    MeZO step following the exact implementation from the paper.
    
    Key differences from our previous implementation:
    1. NO perturbation normalization
    2. Uses paper's hyperparameters
    3. Includes weight decay option
    """
    # Sample perturbation z ~ N(0, I) - NO NORMALIZATION
    z_list = [torch.randn_like(p) for p in lora_params]
    
    # Prepare inputs
    inputs = tokenizer(
        batch['text'], 
        padding=True, 
        truncation=True, 
        max_length=128,
        return_tensors='pt'
    ).to(model.device)
    
    labels = torch.tensor(batch['label']).to(model.device)
    
    # Apply positive perturbation: θ + εz
    for i, p in enumerate(lora_params):
        p.data.add_(epsilon * z_list[i])
    apply_lora_weights(model)
    
    # First forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs, labels=labels)
        loss_plus = outputs.loss.item()
    
    # Apply negative perturbation: θ - εz (from θ + εz, so -2εz)
    for i, p in enumerate(lora_params):
        p.data.add_(-2 * epsilon * z_list[i])
    apply_lora_weights(model)
    
    # Second forward pass
    with torch.no_grad():
        outputs = model(**inputs, labels=labels)
        loss_minus = outputs.loss.item()
    
    # Restore original parameters: θ (from θ - εz, so +εz)
    for i, p in enumerate(lora_params):
        p.data.add_(epsilon * z_list[i])
    
    # Compute gradient estimate: g = (L(θ+εz) - L(θ-εz)) / (2ε)
    grad_estimate = (loss_plus - loss_minus) / (2 * epsilon)
    
    # Update parameters: θ = θ - lr * (g * z + λθ)
    for i, p in enumerate(lora_params):
        # Gradient term
        p.data.add_(-lr * grad_estimate * z_list[i])
        # Weight decay term (if enabled)
        if weight_decay > 0:
            p.data.mul_(1 - lr * weight_decay)
    
    # Apply updated weights
    apply_lora_weights(model)
    
    # Return average loss and statistics
    avg_loss = (loss_plus + loss_minus) / 2
    stats = {
        'loss_plus': loss_plus,
        'loss_minus': loss_minus,
        'grad_estimate': grad_estimate,
        'grad_norm': sum(torch.norm(z * grad_estimate) for z in z_list).item()
    }
    
    return avg_loss, stats


def evaluate_model(model, tokenizer, examples, max_examples=200):
    """Evaluate model on examples."""
    model.eval()
    correct = 0
    total_loss = 0
    
    # Limit evaluation size for speed
    eval_examples = examples[:max_examples] if max_examples else examples
    
    with torch.no_grad():
        for example in eval_examples:
            inputs = tokenizer(
                example['text'],
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
    
    accuracy = correct / len(eval_examples)
    avg_loss = total_loss / len(eval_examples)
    
    return accuracy, avg_loss


def test_mezo_paper_aligned():
    """Test MeZO with paper-aligned implementation and hyperparameters."""
    print("=" * 70)
    print("MeZO Paper-Aligned Implementation Test")
    print("=" * 70)
    
    # Configuration exactly from the paper
    model_name = "roberta-base"  # Paper uses roberta-large, but base is faster
    task_name = "sst-2"
    
    # Hyperparameters from paper Table 15
    batch_size = 64
    learning_rate = 1e-6  # Middle of paper grid {1e-7, 1e-6, 1e-5}
    epsilon = 1e-3
    weight_decay = 0.0
    lora_rank = 8
    lora_alpha = 16
    
    # We'll run fewer steps for demonstration (paper uses 100K)
    num_steps = 2000  # Enough to see convergence starting
    eval_steps = 200
    
    print(f"\nConfiguration (from MeZO paper):")
    print(f"  Model: {model_name}")
    print(f"  Task: {task_name}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate:.0e}")
    print(f"  Epsilon: {epsilon:.0e}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  LoRA rank: {lora_rank}")
    print(f"  LoRA alpha: {lora_alpha}")
    print(f"  Steps: {num_steps:,} (paper uses 100K)")
    print(f"  Evaluation interval: {eval_steps}")
    
    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2,
        torch_dtype=torch.float32
    ).to(device)
    
    print(f"Model loaded on {device}")
    
    # Freeze base model
    for param in model.parameters():
        param.requires_grad = False
    
    # Add LoRA adapters (following paper)
    print("\nAdding LoRA adapters...")
    lora_params = create_lora_roberta(model, rank=lora_rank)
    total_lora_params = sum(p.numel() for p in lora_params)
    base_params = sum(p.numel() for p in model.parameters())
    print(f"  LoRA parameters: {total_lora_params:,} ({total_lora_params/base_params*100:.2f}% of base)")
    print(f"  Number of LoRA matrices: {len(lora_params)}")
    
    # Load datasets
    print("\nLoading SST-2 dataset...")
    data_dir = "/home/lei/git/princeton-nlp/MeZO/medium_models/data/k-shot-1k-test/SST-2"
    train_path = f"{data_dir}/512-42/train.tsv"
    dev_path = f"{data_dir}/512-42/dev.tsv"
    
    train_examples = load_sst2_data(train_path)
    eval_examples = load_sst2_data(dev_path, max_examples=500)  # Limit for speed
    
    print(f"  Train examples: {len(train_examples)}")
    print(f"  Eval examples: {len(eval_examples)}")
    
    # Initial evaluation
    print("\nInitial evaluation...")
    apply_lora_weights(model)
    initial_acc, initial_loss = evaluate_model(model, tokenizer, eval_examples)
    print(f"  Accuracy: {initial_acc:.1%}")
    print(f"  Loss: {initial_loss:.4f}")
    
    # Training
    print(f"\nStarting MeZO training (paper-aligned)...")
    print("-" * 70)
    print("Step   | Train Loss | Grad Norm | Eval Loss | Eval Acc | Time")
    print("-" * 70)
    
    # Tracking
    train_losses = []
    eval_losses = []
    eval_accs = []
    grad_norms = []
    
    start_time = time.time()
    
    for step in range(num_steps):
        # Sample batch
        batch_indices = np.random.choice(len(train_examples), batch_size, replace=True)
        batch = {
            'text': [train_examples[i]['text'] for i in batch_indices],
            'label': [train_examples[i]['label'] for i in batch_indices]
        }
        
        # MeZO step with paper settings
        train_loss, stats = mezo_step_paper_aligned(
            model, tokenizer, batch, lora_params, 
            epsilon=epsilon, lr=learning_rate, weight_decay=weight_decay
        )
        
        train_losses.append(train_loss)
        grad_norms.append(stats['grad_norm'])
        
        # Evaluation
        if (step + 1) % eval_steps == 0:
            eval_acc, eval_loss = evaluate_model(model, tokenizer, eval_examples)
            eval_losses.append(eval_loss)
            eval_accs.append(eval_acc)
            
            elapsed = time.time() - start_time
            print(f"{step+1:6d} | {train_loss:10.6f} | {stats['grad_norm']:9.6f} | "
                  f"{eval_loss:9.6f} | {eval_acc:8.1%} | {elapsed:6.1f}s")
    
    print("-" * 70)
    
    # Final evaluation
    final_acc, final_loss = evaluate_model(model, tokenizer, eval_examples)
    total_time = time.time() - start_time
    
    # Analysis
    print(f"\nTraining Summary:")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"  Time per step: {total_time/num_steps:.3f}s")
    print(f"  Initial → Final:")
    print(f"    Accuracy: {initial_acc:.1%} → {final_acc:.1%} ({(final_acc-initial_acc)*100:+.1f}pp)")
    print(f"    Loss: {initial_loss:.4f} → {final_loss:.4f} ({initial_loss-final_loss:+.4f})")
    
    # Check if loss is decreasing
    if len(train_losses) > 100:
        early_avg = np.mean(train_losses[:100])
        late_avg = np.mean(train_losses[-100:])
        print(f"  Training loss trend: {early_avg:.4f} → {late_avg:.4f}")
    
    # Plot results
    if len(eval_accs) > 1:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Training loss
        ax1.plot(train_losses, 'b-', alpha=0.5)
        ax1.plot(np.convolve(train_losses, np.ones(100)/100, mode='valid'), 'b-', linewidth=2)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Training Loss')
        ax1.set_title('Training Loss (with 100-step moving average)')
        ax1.grid(True, alpha=0.3)
        
        # Eval metrics
        eval_steps_list = list(range(eval_steps, num_steps+1, eval_steps))
        
        ax2.plot(eval_steps_list[:len(eval_losses)], eval_losses, 'r-', marker='o')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Eval Loss')
        ax2.set_title('Evaluation Loss')
        ax2.grid(True, alpha=0.3)
        
        ax3.plot(eval_steps_list[:len(eval_accs)], eval_accs, 'g-', marker='o')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Eval Accuracy')
        ax3.set_title('Evaluation Accuracy')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        # Gradient norm
        ax4.plot(grad_norms, 'purple', alpha=0.5)
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Gradient Norm')
        ax4.set_title('Gradient Norm During Training')
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig('mezo_paper_aligned_results.png', dpi=150)
        print(f"\nPlots saved to: mezo_paper_aligned_results.png")
    
    # Verdict
    print("\n" + "=" * 70)
    if final_acc > initial_acc + 0.02:  # At least 2% improvement
        print("✅ SUCCESS: MeZO is learning with paper-aligned settings!")
        print("=" * 70)
        print(f"Achieved {(final_acc-initial_acc)*100:.1f}pp improvement in {num_steps} steps")
        print("With full 100K steps (as in paper), convergence would be much stronger.")
    else:
        print("⚠️  LIMITED IMPROVEMENT: More steps needed")
        print("=" * 70)
        print(f"Current improvement: {(final_acc-initial_acc)*100:.1f}pp")
        print("This is expected - MeZO typically needs 10K+ steps to show clear gains.")
        print("The paper uses 100K steps for full convergence.")
    
    return final_acc > initial_acc


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        torch.cuda.manual_seed(42)
    else:
        print("WARNING: Running on CPU, this will be slower")
    
    success = test_mezo_paper_aligned()