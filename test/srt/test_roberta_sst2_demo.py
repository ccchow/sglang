#!/usr/bin/env python3
"""
Demonstration of MeZO on RoBERTa SST-2 with paper settings.
Uses roberta-base for faster execution and shows the training dynamics.
"""

import torch
import numpy as np
import time
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import matplotlib.pyplot as plt


def load_sst2_subset(file_path, max_examples=500):
    """Load a subset of SST-2 for faster demo."""
    examples = []
    with open(file_path, 'r') as f:
        lines = f.readlines()[1:]  # Skip header
        for i, line in enumerate(lines):
            if i >= max_examples:
                break
            parts = line.strip().split('\t')
            if len(parts) == 2:
                text, label = parts
                examples.append({'text': text, 'label': int(label)})
    return examples


def create_minimal_lora(model, rank=8, alpha=16):
    """Add LoRA to first few attention layers for demo."""
    lora_params = []
    device = model.device
    num_layers = 0
    max_layers = 6  # Only first 6 layers for speed
    
    for name, module in model.named_modules():
        if num_layers >= max_layers:
            break
            
        if 'attention' in name and hasattr(module, 'self') and hasattr(module.self, 'query'):
            # Add LoRA to query only for simplicity
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
            num_layers += 1
    
    return lora_params


def apply_lora_minimal(model):
    """Apply LoRA weights."""
    for name, module in model.named_modules():
        if 'attention' in name and hasattr(module, 'self'):
            if hasattr(module.self.query, 'lora_A'):
                layer = module.self.query
                layer.weight.data = layer.original_weight + \
                    layer.lora_scale * (layer.lora_B @ layer.lora_A)


def run_mezo_demo():
    print("=" * 70)
    print("MeZO RoBERTa SST-2 Demonstration")
    print("=" * 70)
    
    # Configuration
    model_name = "roberta-base"  # Faster than large
    batch_size = 32  # Smaller for demo
    learning_rate = 1e-6  # Paper setting
    epsilon = 1e-3  # Paper setting
    num_steps = 1000  # Quick demo
    
    print(f"\nConfiguration (MeZO paper settings):")
    print(f"  Model: {model_name} (using base for speed)")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Epsilon: {epsilon}")
    print(f"  Steps: {num_steps}")
    print(f"  Perturbations: Unnormalized (paper default)")
    
    # Load model
    print("\nLoading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    ).to(device)
    
    # Freeze base model
    for param in model.parameters():
        param.requires_grad = False
    
    # Add LoRA
    print("Adding LoRA adapters...")
    lora_params = create_minimal_lora(model)
    print(f"  LoRA parameters: {sum(p.numel() for p in lora_params):,}")
    
    # Load data
    print("\nLoading SST-2 data...")
    data_dir = "/home/lei/git/princeton-nlp/MeZO/medium_models/data/k-shot-1k-test/SST-2"
    train_data = load_sst2_subset(f"{data_dir}/512-42/train.tsv", 500)
    eval_data = load_sst2_subset(f"{data_dir}/512-42/dev.tsv", 100)
    
    print(f"  Train: {len(train_data)} examples")
    print(f"  Eval: {len(eval_data)} examples")
    
    # Quick evaluation function
    def quick_eval(examples, max_ex=50):
        model.eval()
        correct = 0
        for ex in examples[:max_ex]:
            inputs = tokenizer(ex['text'], truncation=True, max_length=128, 
                             return_tensors='pt').to(device)
            with torch.no_grad():
                pred = torch.argmax(model(**inputs).logits, dim=-1).item()
            correct += (pred == ex['label'])
        return correct / min(len(examples), max_ex)
    
    # Initial accuracy
    apply_lora_minimal(model)
    init_acc = quick_eval(eval_data)
    print(f"\nInitial accuracy: {init_acc:.1%}")
    
    # Training
    print(f"\nTraining for {num_steps} steps...")
    print("-" * 50)
    print("Step  | Loss    | Acc   | Time")
    print("-" * 50)
    
    losses = []
    accs = []
    start_time = time.time()
    
    for step in range(num_steps):
        # Sample batch
        idx = np.random.choice(len(train_data), batch_size, replace=True)
        texts = [train_data[i]['text'] for i in idx]
        labels = torch.tensor([train_data[i]['label'] for i in idx]).to(device)
        
        # Tokenize
        inputs = tokenizer(texts, padding=True, truncation=True, 
                          max_length=128, return_tensors='pt').to(device)
        
        # MeZO step (paper algorithm)
        z_list = [torch.randn_like(p) for p in lora_params]  # NO normalization
        
        # Forward +epsilon
        for i, p in enumerate(lora_params):
            p.data.add_(epsilon * z_list[i])
        apply_lora_minimal(model)
        
        with torch.no_grad():
            loss_plus = model(**inputs, labels=labels).loss.item()
        
        # Forward -epsilon
        for i, p in enumerate(lora_params):
            p.data.add_(-2 * epsilon * z_list[i])
        apply_lora_minimal(model)
        
        with torch.no_grad():
            loss_minus = model(**inputs, labels=labels).loss.item()
        
        # Restore and update
        for i, p in enumerate(lora_params):
            p.data.add_(epsilon * z_list[i])
        
        grad_est = (loss_plus - loss_minus) / (2 * epsilon)
        
        for i, p in enumerate(lora_params):
            p.data.add_(-learning_rate * grad_est * z_list[i])
        
        apply_lora_minimal(model)
        
        avg_loss = (loss_plus + loss_minus) / 2
        losses.append(avg_loss)
        
        # Periodic evaluation
        if (step + 1) % 200 == 0:
            acc = quick_eval(eval_data)
            accs.append(acc)
            elapsed = time.time() - start_time
            print(f"{step+1:5d} | {avg_loss:.5f} | {acc:.1%} | {elapsed:4.0f}s")
    
    print("-" * 50)
    
    # Final evaluation
    final_acc = quick_eval(eval_data)
    total_time = time.time() - start_time
    
    print(f"\nTraining complete in {total_time:.0f}s")
    print(f"Accuracy: {init_acc:.1%} → {final_acc:.1%} ({(final_acc-init_acc)*100:+.1f}pp)")
    
    # Analysis
    print("\nLoss trend analysis:")
    early_loss = np.mean(losses[:50])
    late_loss = np.mean(losses[-50:])
    print(f"  Early loss (first 50): {early_loss:.4f}")
    print(f"  Late loss (last 50): {late_loss:.4f}")
    print(f"  Reduction: {early_loss - late_loss:.4f}")
    
    # Simple plot
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses, alpha=0.5)
    if len(losses) > 20:
        smoothed = np.convolve(losses, np.ones(20)/20, mode='valid')
        plt.plot(smoothed, 'r-', linewidth=2, label='Smoothed')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    eval_steps = list(range(200, num_steps+1, 200))
    plt.plot([0] + eval_steps, [init_acc] + accs, 'o-')
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.title('Evaluation Accuracy')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('roberta_sst2_demo.png', dpi=150)
    print(f"\nPlot saved to: roberta_sst2_demo.png")
    
    print("\n" + "=" * 70)
    if final_acc > init_acc + 0.02:
        print("✅ SUCCESS: MeZO is learning with paper settings!")
        print(f"   Achieved {(final_acc-init_acc)*100:.1f}pp improvement")
    else:
        print("⚠️  Limited improvement - this is expected for short runs")
        print("   MeZO typically needs 10K+ steps for clear gains")
    
    return final_acc > init_acc


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    run_mezo_demo()