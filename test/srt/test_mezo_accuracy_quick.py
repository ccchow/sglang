#!/usr/bin/env python3
"""
Quick demonstration of MeZO using accuracy as the objective.
Shows the key difference between optimizing accuracy vs cross-entropy loss.
"""

import torch
import numpy as np
import time
from transformers import RobertaTokenizer, RobertaForSequenceClassification


def test_accuracy_vs_loss_objective():
    print("=" * 70)
    print("MeZO: Accuracy vs Loss Objective Comparison")
    print("=" * 70)
    
    # Simple setup
    model_name = "roberta-base"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\nSetup:")
    print(f"  Model: {model_name}")
    print(f"  Device: {device}")
    
    # Load model and tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    ).to(device)
    
    # Create a simple batch
    texts = [
        "This movie is absolutely fantastic!",
        "Terrible film, waste of time.",
        "Great acting and amazing story!",
        "Boring and poorly made."
    ]
    labels = torch.tensor([1, 0, 1, 0]).to(device)  # 1=positive, 0=negative
    
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(device)
    
    # Get initial predictions
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(logits, dim=-1)
        
        # Compute both objectives
        accuracy = (preds == labels).float().mean().item()
        loss = torch.nn.functional.cross_entropy(logits, labels).item()
    
    print(f"\nInitial state:")
    print(f"  Predictions: {preds.tolist()}")
    print(f"  True labels: {labels.tolist()}")
    print(f"  Accuracy: {accuracy:.1%}")
    print(f"  Cross-entropy loss: {loss:.4f}")
    
    # Demonstrate MeZO gradient estimation
    print("\n" + "-" * 50)
    print("MeZO Gradient Estimation:")
    print("-" * 50)
    
    # Use a simple parameter for demonstration
    param = model.classifier.dense.weight
    original_param = param.data.clone()
    
    epsilon = 1e-3
    
    # Sample perturbation
    z = torch.randn_like(param)
    
    print("\n1. Using CROSS-ENTROPY LOSS objective:")
    
    # Forward with +epsilon
    param.data = original_param + epsilon * z
    with torch.no_grad():
        loss_plus = model(**inputs, labels=labels).loss.item()
    
    # Forward with -epsilon
    param.data = original_param - epsilon * z
    with torch.no_grad():
        loss_minus = model(**inputs, labels=labels).loss.item()
    
    # Gradient estimate
    grad_est_loss = (loss_plus - loss_minus) / (2 * epsilon)
    print(f"   L(θ+εz) = {loss_plus:.6f}")
    print(f"   L(θ-εz) = {loss_minus:.6f}")
    print(f"   Gradient estimate: {grad_est_loss:.6f}")
    
    print("\n2. Using ACCURACY objective (paper approach):")
    
    # Forward with +epsilon
    param.data = original_param + epsilon * z
    with torch.no_grad():
        preds_plus = torch.argmax(model(**inputs).logits, dim=-1)
        acc_plus = (preds_plus == labels).float().mean().item()
        neg_acc_plus = -acc_plus  # Negative for minimization
    
    # Forward with -epsilon
    param.data = original_param - epsilon * z
    with torch.no_grad():
        preds_minus = torch.argmax(model(**inputs).logits, dim=-1)
        acc_minus = (preds_minus == labels).float().mean().item()
        neg_acc_minus = -acc_minus
    
    # Gradient estimate
    grad_est_acc = (neg_acc_plus - neg_acc_minus) / (2 * epsilon)
    print(f"   Acc(θ+εz) = {acc_plus:.1%} (neg: {neg_acc_plus:.3f})")
    print(f"   Acc(θ-εz) = {acc_minus:.1%} (neg: {neg_acc_minus:.3f})")
    print(f"   Gradient estimate: {grad_est_acc:.6f}")
    
    # Restore parameter
    param.data = original_param
    
    # Key insight
    print("\n" + "=" * 70)
    print("KEY INSIGHTS:")
    print("=" * 70)
    print("1. Cross-entropy loss provides continuous gradient signal")
    print("2. Accuracy objective is discrete (0 or 1 per sample)")
    print("3. Accuracy gradient is often 0 when predictions don't change")
    print("4. Paper shows accuracy optimization still works with many steps")
    print("\nFor SST-2, the paper uses:")
    print("- Objective: Negative accuracy (to minimize)")
    print("- Steps: 100K (allows discrete changes to accumulate)")
    print("- Batch size: 64 (more samples for gradient estimation)")
    
    # Quick training demo
    print("\n" + "-" * 50)
    print("Quick Training Demo (100 steps):")
    print("-" * 50)
    
    # Simple LoRA on classifier
    lora_A = torch.nn.Parameter(torch.randn(8, param.size(1), device=device) * 0.01)
    lora_B = torch.nn.Parameter(torch.zeros(param.size(0), 8, device=device))
    param.original = param.data.clone()
    
    lr = 1e-5  # Higher LR for demo
    
    # Load more data
    data_dir = "/home/lei/git/princeton-nlp/MeZO/medium_models/data/k-shot-1k-test/SST-2"
    examples = []
    with open(f"{data_dir}/512-42/train.tsv", 'r') as f:
        lines = f.readlines()[1:101]  # First 100 examples
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                examples.append({'text': parts[0], 'label': int(parts[1])})
    
    print(f"Loaded {len(examples)} training examples")
    
    accs = []
    for step in range(100):
        # Sample batch
        idx = np.random.choice(len(examples), 16)
        batch_texts = [examples[i]['text'] for i in idx]
        batch_labels = torch.tensor([examples[i]['label'] for i in idx]).to(device)
        
        batch_inputs = tokenizer(
            batch_texts, padding=True, truncation=True, 
            max_length=128, return_tensors='pt'
        ).to(device)
        
        # MeZO step with accuracy objective
        z_A = torch.randn_like(lora_A)
        z_B = torch.randn_like(lora_B)
        
        # Apply LoRA + perturbation
        param.data = param.original + (lora_B @ lora_A) + epsilon * (z_B @ lora_A + lora_B @ z_A)
        with torch.no_grad():
            acc_plus = -(torch.argmax(model(**batch_inputs).logits, dim=-1) == batch_labels).float().mean().item()
        
        param.data = param.original + (lora_B @ lora_A) - epsilon * (z_B @ lora_A + lora_B @ z_A)
        with torch.no_grad():
            acc_minus = -(torch.argmax(model(**batch_inputs).logits, dim=-1) == batch_labels).float().mean().item()
        
        # Update
        grad_est = (acc_plus - acc_minus) / (2 * epsilon)
        lora_A.data -= lr * grad_est * z_A
        lora_B.data -= lr * grad_est * z_B
        
        # Track accuracy
        param.data = param.original + (lora_B @ lora_A)
        with torch.no_grad():
            acc = (torch.argmax(model(**batch_inputs).logits, dim=-1) == batch_labels).float().mean().item()
        accs.append(acc)
        
        if step % 20 == 0:
            print(f"  Step {step}: Batch accuracy = {acc:.1%}")
    
    print(f"\nAverage accuracy: First 20 steps = {np.mean(accs[:20]):.1%}, "
          f"Last 20 steps = {np.mean(accs[-20:]):.1%}")
    
    return True


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    test_accuracy_vs_loss_objective()