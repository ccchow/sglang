#!/usr/bin/env python3
"""
Analyze when MeZO gets non-zero gradients with accuracy objective.
"""

import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification


def test_gradient_conditions():
    """Test different conditions to understand when we get non-zero gradients."""
    print("=" * 70)
    print("MeZO Gradient Analysis - When Do We Get Non-Zero Gradients?")
    print("=" * 70)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base", num_labels=2
    ).to(device)
    
    # Test different scenarios
    scenarios = [
        {
            "name": "Easy examples with large epsilon",
            "texts": ["I love this movie!", "This is terrible!"],
            "labels": [1, 0],
            "epsilon": 0.1,  # 100x larger
        },
        {
            "name": "Ambiguous examples with normal epsilon",
            "texts": ["It's okay.", "Not bad."],
            "labels": [1, 0],
            "epsilon": 0.001,
        },
        {
            "name": "Larger batch size",
            "texts": ["Good"] * 32 + ["Bad"] * 32,
            "labels": [1] * 32 + [0] * 32,
            "epsilon": 0.001,
        },
        {
            "name": "Mixed confidence examples",
            "texts": [
                "Absolutely fantastic!",  # High confidence positive
                "Completely awful!",      # High confidence negative
                "It's alright.",         # Low confidence
                "Not sure.",             # Low confidence
            ],
            "labels": [1, 0, 1, 0],
            "epsilon": 0.001,
        }
    ]
    
    # Use a simple parameter for testing
    param = model.classifier.dense.weight
    original_param = param.data.clone()
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        print(f"  Batch size: {len(scenario['texts'])}")
        print(f"  Epsilon: {scenario['epsilon']}")
        
        inputs = tokenizer(
            scenario['texts'],
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(device)
        labels = torch.tensor(scenario['labels']).to(device)
        
        # Test multiple random perturbations
        non_zero_count = 0
        grad_magnitudes = []
        
        for trial in range(10):
            z = torch.randn_like(param)
            
            # Forward with +epsilon
            param.data = original_param + scenario['epsilon'] * z
            with torch.no_grad():
                preds_plus = torch.argmax(model(**inputs).logits, dim=-1)
                acc_plus = (preds_plus == labels).float().mean().item()
            
            # Forward with -epsilon  
            param.data = original_param - scenario['epsilon'] * z
            with torch.no_grad():
                preds_minus = torch.argmax(model(**inputs).logits, dim=-1)
                acc_minus = (preds_minus == labels).float().mean().item()
            
            # Gradient
            grad = ((-acc_plus) - (-acc_minus)) / (2 * scenario['epsilon'])
            
            if grad != 0:
                non_zero_count += 1
                grad_magnitudes.append(abs(grad))
                
                # Show details for first non-zero
                if non_zero_count == 1:
                    print(f"    Example non-zero gradient:")
                    print(f"      Acc(+ε): {acc_plus:.2%}, Acc(-ε): {acc_minus:.2%}")
                    print(f"      Gradient: {grad:.6f}")
        
        print(f"  Results from 10 trials:")
        print(f"    Non-zero gradients: {non_zero_count}/10 ({non_zero_count*10}%)")
        if grad_magnitudes:
            print(f"    Average magnitude: {np.mean(grad_magnitudes):.6f}")
    
    # Restore
    param.data = original_param
    
    # Test with very carefully chosen examples
    print("\n" + "=" * 70)
    print("Special test: Examples near decision boundary")
    print("=" * 70)
    
    # Create examples that should be near the decision boundary
    # by using neutral/ambiguous language
    boundary_texts = [
        "The movie was okay, I guess.",
        "It had some good parts and some bad parts.",
        "I'm not sure how I feel about it.",
        "Mixed feelings about this one.",
        "Could have been better, could have been worse.",
        "Average film with average acting.",
        "Nothing special but not terrible.",
        "Decent effort but forgettable.",
    ]
    boundary_labels = [1, 0, 1, 0, 1, 0, 1, 0]  # Arbitrary labels
    
    inputs = tokenizer(boundary_texts, padding=True, return_tensors='pt').to(device)
    labels = torch.tensor(boundary_labels).to(device)
    
    # Check initial predictions and confidence
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(logits, dim=-1)
        confidences = torch.max(probs, dim=-1).values
    
    print("\nInitial predictions:")
    for i, (text, label, pred, conf) in enumerate(
        zip(boundary_texts[:4], boundary_labels[:4], preds[:4], confidences[:4])
    ):
        print(f"  '{text[:30]}...'")
        print(f"    True: {label}, Pred: {pred.item()}, Conf: {conf.item():.2%}")
    
    # Test gradients with different epsilon values
    print("\nGradient analysis with different epsilon values:")
    for eps in [0.0001, 0.001, 0.01, 0.1]:
        non_zero = 0
        for _ in range(20):
            z = torch.randn_like(param)
            
            param.data = original_param + eps * z
            acc_plus = -(torch.argmax(model(**inputs).logits, dim=-1) == labels).float().mean().item()
            
            param.data = original_param - eps * z
            acc_minus = -(torch.argmax(model(**inputs).logits, dim=-1) == labels).float().mean().item()
            
            if acc_plus != acc_minus:
                non_zero += 1
        
        print(f"  ε = {eps:g}: {non_zero}/20 non-zero gradients ({non_zero*5}%)")
    
    param.data = original_param
    
    print("\n" + "=" * 70)
    print("CONCLUSIONS:")
    print("=" * 70)
    print("1. Larger epsilon → more likely to flip predictions → non-zero gradients")
    print("2. Examples near decision boundary → more sensitive to perturbations")
    print("3. Larger batches → higher chance at least one example flips")
    print("4. With ε=0.001 and confident predictions → almost always zero gradient")
    print("\nThis explains why 100K steps are needed for convergence!")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    test_gradient_conditions()