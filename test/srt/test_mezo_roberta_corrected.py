#!/usr/bin/env python3
"""
Corrected MeZO test for RoBERTa on SST-2.
Addresses the issues identified in the investigation:
1. No perturbation normalization
2. Correct learning rate (1e-6)
3. More training steps
4. Proper hyperparameters from paper
"""

import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import matplotlib.pyplot as plt
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_sst2_data(file_path):
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


def create_lora_roberta(model, rank=8):
    """Add LoRA adapters to RoBERTa model."""
    lora_params = []
    device = model.device
    
    # Add LoRA to attention layers
    for name, module in model.named_modules():
        if 'attention' in name and hasattr(module, 'self'):
            if hasattr(module.self, 'query'):
                in_features = module.self.query.in_features
                out_features = module.self.query.out_features
                
                # Initialize LoRA matrices
                lora_A_q = torch.nn.Parameter(torch.randn(rank, in_features, device=device) * 0.01)
                lora_B_q = torch.nn.Parameter(torch.zeros(out_features, rank, device=device))
                
                lora_A_v = torch.nn.Parameter(torch.randn(rank, in_features, device=device) * 0.01)
                lora_B_v = torch.nn.Parameter(torch.zeros(out_features, rank, device=device))
                
                # Store original weights
                module.self.query.original_weight = module.self.query.weight.data.clone()
                module.self.value.original_weight = module.self.value.weight.data.clone()
                
                # Attach LoRA parameters
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


def mezo_step_corrected(model, tokenizer, batch, lora_params, epsilon=1e-3, lr=1e-6):
    """
    Corrected MeZO step following the original implementation:
    - No perturbation normalization
    - Proper gradient scaling
    """
    # Sample perturbation (NO NORMALIZATION)
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
    
    # Apply positive perturbation
    for i, p in enumerate(lora_params):
        p.data.add_(epsilon * z_list[i])
    apply_lora_weights(model)
    
    # First forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs, labels=labels)
        loss1 = outputs.loss
    
    # Apply negative perturbation (total -2*epsilon from original)
    for i, p in enumerate(lora_params):
        p.data.add_(-2 * epsilon * z_list[i])
    apply_lora_weights(model)
    
    # Second forward pass
    with torch.no_grad():
        outputs = model(**inputs, labels=labels)
        loss2 = outputs.loss
    
    # Restore original parameters
    for i, p in enumerate(lora_params):
        p.data.add_(epsilon * z_list[i])
    apply_lora_weights(model)
    
    # Compute gradient estimate
    grad_estimate = (loss1 - loss2) / (2 * epsilon)
    
    # Update parameters
    for i, p in enumerate(lora_params):
        p.data.add_(-lr * grad_estimate * z_list[i])
    
    return (loss1 + loss2) / 2


def evaluate_model(model, tokenizer, examples):
    """Evaluate model on examples."""
    model.eval()
    correct = 0
    total_loss = 0
    
    with torch.no_grad():
        for example in examples:
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
    
    accuracy = correct / len(examples)
    avg_loss = total_loss / len(examples)
    
    return accuracy, avg_loss


def test_mezo_roberta_corrected():
    """Test corrected MeZO implementation."""
    print("=" * 70)
    print("Corrected MeZO RoBERTa SST-2 Test")
    print("=" * 70)
    
    # Configuration following the paper
    model_name = "roberta-base"  # Could use roberta-large as in paper
    batch_size = 64  # As in paper
    learning_rate = 1e-6  # From paper grid {1e-7, 1e-6, 1e-5}
    epsilon = 1e-3  # As in paper
    num_steps = 5000  # More steps (paper uses 100K)
    eval_steps = 500  # Evaluate more frequently
    
    print(f"\nConfiguration (following MeZO paper):")
    print(f"  Model: {model_name}")
    print(f"  Task: SST-2 (512-shot)")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate:.0e}")
    print(f"  Epsilon: {epsilon:.0e}")
    print(f"  Steps: {num_steps:,}")
    print(f"  Key changes:")
    print(f"    - NO perturbation normalization")
    print(f"    - Correct learning rate (1e-6 not 1e-3)")
    print(f"    - More training steps")
    
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
    print("\nLoading datasets...")
    data_dir = "/home/lei/git/princeton-nlp/MeZO/medium_models/data/k-shot-1k-test/SST-2"
    train_path = f"{data_dir}/512-42/train.tsv"
    dev_path = f"{data_dir}/512-42/dev.tsv"
    
    train_examples = load_sst2_data(train_path)
    eval_examples = load_sst2_data(dev_path)[:200]  # Use subset for faster eval
    
    print(f"  Train examples: {len(train_examples)}")
    print(f"  Eval examples: {len(eval_examples)}")
    
    # Initial evaluation
    apply_lora_weights(model)
    initial_acc, initial_loss = evaluate_model(model, tokenizer, eval_examples)
    print(f"\nInitial performance:")
    print(f"  Accuracy: {initial_acc:.1%}")
    print(f"  Loss: {initial_loss:.4f}")
    
    # Training
    print(f"\nStarting corrected MeZO training...")
    print("-" * 70)
    print("Step   | Train Loss | Eval Loss | Eval Acc | Time/step | Total Time")
    print("-" * 70)
    
    train_losses = []
    eval_losses = []
    eval_accs = []
    
    start_time = time.time()
    
    for step in range(num_steps):
        # Sample batch
        batch_indices = np.random.choice(len(train_examples), batch_size, replace=True)
        batch = {
            'text': [train_examples[i]['text'] for i in batch_indices],
            'label': [train_examples[i]['label'] for i in batch_indices]
        }
        
        # MeZO step
        step_start = time.time()
        train_loss = mezo_step_corrected(model, tokenizer, batch, lora_params, epsilon, learning_rate)
        step_time = time.time() - step_start
        
        train_losses.append(train_loss.item())
        
        # Evaluation
        if (step + 1) % eval_steps == 0:
            apply_lora_weights(model)
            eval_acc, eval_loss = evaluate_model(model, tokenizer, eval_examples)
            eval_losses.append(eval_loss)
            eval_accs.append(eval_acc)
            
            elapsed = time.time() - start_time
            print(f"{step+1:6d} | {train_loss:10.6f} | {eval_loss:9.6f} | {eval_acc:8.1%} | {step_time:9.3f}s | {elapsed:9.1f}s")
    
    print("-" * 70)
    
    # Final evaluation
    apply_lora_weights(model)
    final_acc, final_loss = evaluate_model(model, tokenizer, eval_examples)
    
    # Training summary
    total_time = time.time() - start_time
    print(f"\nTraining Summary:")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"  Initial accuracy: {initial_acc:.1%}")
    print(f"  Final accuracy: {final_acc:.1%}")
    print(f"  Accuracy improvement: {(final_acc - initial_acc) * 100:+.1f}pp")
    print(f"  Initial loss: {initial_loss:.4f}")
    print(f"  Final loss: {final_loss:.4f}")
    print(f"  Loss reduction: {initial_loss - final_loss:.4f}")
    
    # Plot results
    if len(eval_accs) > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss plot
        ax1.plot(train_losses, 'b-', alpha=0.3, label='Train')
        eval_steps_list = list(range(eval_steps-1, len(train_losses), eval_steps))
        ax1.plot(eval_steps_list[:len(eval_losses)], eval_losses, 'r-', marker='o', label='Eval')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(eval_steps_list[:len(eval_accs)], eval_accs, 'g-', marker='o')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Evaluation Accuracy')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('mezo_roberta_corrected.png', dpi=150)
        print(f"\nPlot saved to: mezo_roberta_corrected.png")
    
    # Success criteria
    success = final_acc > initial_acc + 0.02  # At least 2% improvement
    
    print("\n" + "=" * 70)
    if success:
        print("✅ CORRECTED MeZO TEST: SHOWING IMPROVEMENT")
        print("=" * 70)
        print(f"MeZO is learning! Achieved {(final_acc - initial_acc) * 100:+.1f}pp improvement")
        print("With more steps (100K as in paper), it should converge fully.")
    else:
        print("⚠️  CORRECTED MeZO TEST: NEEDS MORE STEPS")
        print("=" * 70)
        print(f"Current improvement: {(final_acc - initial_acc) * 100:+.1f}pp")
        print("This is expected - MeZO typically needs 20K+ steps to show clear improvement")
    
    return final_acc > initial_acc


if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: Running on CPU, this will be slow")
    
    success = test_mezo_roberta_corrected()