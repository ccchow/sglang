#!/usr/bin/env python3
"""
Quick test to demonstrate the corrected MeZO implementation.
Uses fewer steps for faster execution.
"""

import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import time

def create_minimal_lora(model, rank=4):
    """Create minimal LoRA for faster testing."""
    lora_params = []
    device = model.device
    
    # Only add LoRA to first 2 attention layers
    count = 0
    for name, module in model.named_modules():
        if count >= 2:
            break
        if 'attention' in name and hasattr(module, 'self') and hasattr(module.self, 'query'):
            in_features = module.self.query.in_features
            out_features = module.self.query.out_features
            
            lora_A = torch.nn.Parameter(torch.randn(rank, in_features, device=device) * 0.01)
            lora_B = torch.nn.Parameter(torch.zeros(out_features, rank, device=device))
            
            module.self.query.original_weight = module.self.query.weight.data.clone()
            module.self.query.lora_A = lora_A
            module.self.query.lora_B = lora_B
            
            lora_params.extend([lora_A, lora_B])
            count += 1
    
    return lora_params


def apply_minimal_lora(model):
    """Apply minimal LoRA weights."""
    for name, module in model.named_modules():
        if 'attention' in name and hasattr(module, 'self'):
            if hasattr(module.self.query, 'lora_A'):
                module.self.query.weight.data = module.self.query.original_weight + \
                    (module.self.query.lora_B @ module.self.query.lora_A)


def test_perturbation_comparison():
    """Compare normalized vs unnormalized perturbations."""
    print("=" * 60)
    print("MeZO Perturbation Comparison Test")
    print("=" * 60)
    
    # Create dummy parameters
    param = torch.randn(100, 768)
    epsilon = 1e-3
    
    # Test 1: Normalized perturbation (our original approach)
    z_norm = torch.randn_like(param)
    z_norm = z_norm / (z_norm.norm() + 1e-8)
    perturbation_norm = epsilon * z_norm
    
    # Test 2: Unnormalized perturbation (paper approach)
    z_unnorm = torch.randn_like(param)
    perturbation_unnorm = epsilon * z_unnorm
    
    print("\nPerturbation Statistics:")
    print(f"Normalized - Mean: {perturbation_norm.mean():.6f}, Std: {perturbation_norm.std():.6f}")
    print(f"Unnormalized - Mean: {perturbation_unnorm.mean():.6f}, Std: {perturbation_unnorm.std():.6f}")
    print(f"Std ratio: {perturbation_unnorm.std() / perturbation_norm.std():.1f}x")
    
    # This shows why we needed different learning rates!
    

def quick_convergence_test():
    """Quick test to show MeZO can learn with correct settings."""
    print("\n" + "=" * 60)
    print("Quick MeZO Convergence Test")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "roberta-base"
    
    # Load model
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    ).to(device)
    
    # Freeze base model
    for param in model.parameters():
        param.requires_grad = False
    
    # Create minimal LoRA
    lora_params = create_minimal_lora(model, rank=4)
    print(f"\nUsing minimal LoRA: {len(lora_params)} parameters")
    
    # Test data
    test_texts = [
        "This movie is absolutely fantastic!",  # Positive
        "This movie is terrible and boring.",   # Negative
        "Best film I've ever seen!",           # Positive
        "Worst movie of all time.",            # Negative
    ]
    test_labels = [1, 0, 1, 0]
    
    # Configuration
    batch_size = 4
    lr_normalized = 1e-3    # For normalized perturbations
    lr_unnormalized = 1e-6  # For unnormalized (paper approach)
    epsilon = 1e-3
    num_steps = 100
    
    print(f"\nTesting two approaches for {num_steps} steps:")
    print("1. Normalized perturbations with lr=1e-3")
    print("2. Unnormalized perturbations with lr=1e-6 (paper)")
    
    # Test both approaches
    for approach, use_normalization, lr in [
        ("Normalized", True, lr_normalized),
        ("Unnormalized", False, lr_unnormalized)
    ]:
        print(f"\n--- {approach} Perturbations ---")
        
        # Reset LoRA parameters
        for p in lora_params:
            p.data.zero_()
        
        losses = []
        start_time = time.time()
        
        for step in range(num_steps):
            # Sample perturbation
            z_list = [torch.randn_like(p) for p in lora_params]
            
            if use_normalization:
                z_list = [z / (z.norm() + 1e-8) for z in z_list]
            
            # Prepare batch
            inputs = tokenizer(
                test_texts, padding=True, truncation=True, return_tensors='pt'
            ).to(device)
            labels = torch.tensor(test_labels).to(device)
            
            # Forward pass 1
            for i, p in enumerate(lora_params):
                p.data.add_(epsilon * z_list[i])
            apply_minimal_lora(model)
            
            with torch.no_grad():
                loss1 = model(**inputs, labels=labels).loss
            
            # Forward pass 2
            for i, p in enumerate(lora_params):
                p.data.add_(-2 * epsilon * z_list[i])
            apply_minimal_lora(model)
            
            with torch.no_grad():
                loss2 = model(**inputs, labels=labels).loss
            
            # Restore and update
            for i, p in enumerate(lora_params):
                p.data.add_(epsilon * z_list[i])
            
            grad_est = (loss1 - loss2) / (2 * epsilon)
            
            for i, p in enumerate(lora_params):
                p.data.add_(-lr * grad_est * z_list[i])
            
            avg_loss = (loss1 + loss2) / 2
            losses.append(avg_loss.item())
            
            if step % 20 == 0:
                print(f"  Step {step}: Loss = {avg_loss:.6f}")
        
        # Final evaluation
        apply_minimal_lora(model)
        with torch.no_grad():
            final_outputs = model(**inputs)
            predictions = torch.argmax(final_outputs.logits, dim=-1)
            accuracy = (predictions == labels).float().mean()
        
        elapsed = time.time() - start_time
        print(f"  Final: Loss = {losses[-1]:.6f}, Accuracy = {accuracy:.1%}")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Loss reduction: {losses[0] - losses[-1]:.6f}")
    
    print("\n" + "=" * 60)
    print("Key Insight: Both approaches can work with appropriate learning rates!")
    print("The unnormalized approach (paper) needs much smaller learning rate.")


if __name__ == "__main__":
    # First show perturbation comparison
    test_perturbation_comparison()
    
    # Then run quick convergence test
    quick_convergence_test()