#!/usr/bin/env python3
"""
Test MeZO using MLM (Masked Language Model) approach as in the original paper.
This uses cross-entropy loss on vocabulary logits, not accuracy on classifications.
"""

import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaForSequenceClassification


def test_mlm_vs_classification_approach():
    """Compare MLM approach (paper) vs classification approach."""
    print("=" * 80)
    print("MeZO: MLM Approach (Paper) vs Classification Approach")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    
    # Load BOTH models
    mlm_model = RobertaForMaskedLM.from_pretrained("roberta-base").to(device)
    cls_model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2).to(device)
    
    # Test examples using MLM template from paper
    # Template: "*cls**sent_0*_It_was*mask*.*sep+*"
    # Mapping: {'0':'terrible','1':'great'}
    
    texts = [
        "This movie is absolutely fantastic!",
        "Terrible film, waste of time.",
        "Great acting and amazing story!",
        "Boring and poorly made."
    ]
    labels = [1, 0, 1, 0]  # 1=positive (great), 0=negative (terrible)
    
    # Get token IDs for label words
    terrible_id = tokenizer.convert_tokens_to_ids('terrible')
    great_id = tokenizer.convert_tokens_to_ids('great')
    mask_id = tokenizer.mask_token_id
    
    print(f"\nLabel word mappings:")
    print(f"  'terrible' -> token {terrible_id}")
    print(f"  'great' -> token {great_id}")
    print(f"  '<mask>' -> token {mask_id}")
    
    # Format inputs with MLM template
    mlm_texts = [f"{text} It was {tokenizer.mask_token}." for text in texts]
    
    print("\n1. MLM APPROACH (Paper Method):")
    print("-" * 50)
    
    # Test gradient computation with MLM
    inputs = tokenizer(mlm_texts, padding=True, return_tensors='pt').to(device)
    
    # Find mask positions
    mask_positions = (inputs.input_ids == mask_id).nonzero(as_tuple=True)[1]
    
    # Use a simple parameter for testing
    param = mlm_model.lm_head.dense.weight
    original_param = param.data.clone()
    
    epsilon = 1e-3
    z = torch.randn_like(param)
    
    # Forward with +epsilon
    param.data = original_param + epsilon * z
    with torch.no_grad():
        outputs = mlm_model(**inputs)
        logits_plus = outputs.logits
        
        # Get logits at mask positions for label words
        mask_logits = logits_plus[torch.arange(len(texts)), mask_positions]
        label_logits = torch.stack([
            mask_logits[:, terrible_id],
            mask_logits[:, great_id]
        ], dim=1)
        
        # Cross-entropy loss
        label_tensor = torch.tensor(labels).to(device)
        loss_plus = torch.nn.functional.cross_entropy(label_logits, label_tensor).item()
    
    # Forward with -epsilon
    param.data = original_param - epsilon * z
    with torch.no_grad():
        outputs = mlm_model(**inputs)
        logits_minus = outputs.logits
        
        mask_logits = logits_minus[torch.arange(len(texts)), mask_positions]
        label_logits = torch.stack([
            mask_logits[:, terrible_id],
            mask_logits[:, great_id]
        ], dim=1)
        
        loss_minus = torch.nn.functional.cross_entropy(label_logits, label_tensor).item()
    
    # Gradient
    grad_mlm = (loss_plus - loss_minus) / (2 * epsilon)
    
    print(f"  Loss(+ε) = {loss_plus:.6f}")
    print(f"  Loss(-ε) = {loss_minus:.6f}")
    print(f"  Gradient = {grad_mlm:.6f} (CONTINUOUS!)")
    
    # Also show predictions
    param.data = original_param
    with torch.no_grad():
        outputs = mlm_model(**inputs)
        mask_logits = outputs.logits[torch.arange(len(texts)), mask_positions]
        label_logits = torch.stack([
            mask_logits[:, terrible_id],
            mask_logits[:, great_id]
        ], dim=1)
        probs = torch.softmax(label_logits, dim=-1)
        preds = torch.argmax(label_logits, dim=-1)
    
    print(f"\n  MLM Predictions:")
    for i, (text, label, pred, prob) in enumerate(zip(texts, labels, preds, probs)):
        print(f"    '{text[:30]}...' -> pred: {'great' if pred == 1 else 'terrible'} "
              f"(prob: {prob[pred].item():.2%}), true: {'great' if label == 1 else 'terrible'}")
    
    # 2. Classification approach for comparison
    print("\n2. CLASSIFICATION APPROACH (What we tried):")
    print("-" * 50)
    
    cls_inputs = tokenizer(texts, padding=True, return_tensors='pt').to(device)
    cls_param = cls_model.classifier.dense.weight
    cls_original = cls_param.data.clone()
    
    # Forward with +epsilon
    cls_param.data = cls_original + epsilon * z[:cls_param.size(0), :cls_param.size(1)]
    with torch.no_grad():
        preds_plus = torch.argmax(cls_model(**cls_inputs).logits, dim=-1)
        acc_plus = (preds_plus == label_tensor).float().mean().item()
    
    # Forward with -epsilon
    cls_param.data = cls_original - epsilon * z[:cls_param.size(0), :cls_param.size(1)]
    with torch.no_grad():
        preds_minus = torch.argmax(cls_model(**cls_inputs).logits, dim=-1)
        acc_minus = (preds_minus == label_tensor).float().mean().item()
    
    # Gradient (accuracy objective)
    grad_acc = ((-acc_plus) - (-acc_minus)) / (2 * epsilon)
    
    print(f"  Acc(+ε) = {acc_plus:.2%}")
    print(f"  Acc(-ε) = {acc_minus:.2%}")
    print(f"  Gradient = {grad_acc:.6f} (DISCRETE!)")
    
    cls_param.data = cls_original
    
    # Summary
    print("\n" + "=" * 80)
    print("KEY INSIGHT:")
    print("=" * 80)
    print("The paper uses MLM with cross-entropy loss, NOT accuracy objective!")
    print("- MLM approach: Continuous gradients from vocabulary logits")
    print("- Classification approach: Discrete gradients from accuracy")
    print("- This explains why the paper's approach converges!")
    
    # Test multiple perturbations
    print("\n3. GRADIENT ANALYSIS (10 random perturbations):")
    print("-" * 50)
    
    mlm_grads = []
    cls_grads = []
    
    for _ in range(10):
        z = torch.randn_like(param)
        
        # MLM gradient
        param.data = original_param + epsilon * z
        outputs = mlm_model(**inputs)
        mask_logits = outputs.logits[torch.arange(len(texts)), mask_positions]
        label_logits = torch.stack([mask_logits[:, terrible_id], mask_logits[:, great_id]], dim=1)
        loss_plus = torch.nn.functional.cross_entropy(label_logits, label_tensor).item()
        
        param.data = original_param - epsilon * z
        outputs = mlm_model(**inputs)
        mask_logits = outputs.logits[torch.arange(len(texts)), mask_positions]
        label_logits = torch.stack([mask_logits[:, terrible_id], mask_logits[:, great_id]], dim=1)
        loss_minus = torch.nn.functional.cross_entropy(label_logits, label_tensor).item()
        
        mlm_grads.append((loss_plus - loss_minus) / (2 * epsilon))
        
        # Classification gradient  
        z_cls = z[:cls_param.size(0), :cls_param.size(1)]
        cls_param.data = cls_original + epsilon * z_cls
        acc_plus = -(torch.argmax(cls_model(**cls_inputs).logits, dim=-1) == label_tensor).float().mean().item()
        
        cls_param.data = cls_original - epsilon * z_cls
        acc_minus = -(torch.argmax(cls_model(**cls_inputs).logits, dim=-1) == label_tensor).float().mean().item()
        
        cls_grads.append((acc_plus - acc_minus) / (2 * epsilon))
    
    param.data = original_param
    cls_param.data = cls_original
    
    mlm_nonzero = sum(1 for g in mlm_grads if g != 0)
    cls_nonzero = sum(1 for g in cls_grads if g != 0)
    
    print(f"MLM approach:")
    print(f"  Non-zero gradients: {mlm_nonzero}/10 ({mlm_nonzero*10}%)")
    print(f"  Average |gradient|: {np.mean(np.abs(mlm_grads)):.6f}")
    print(f"  Gradient range: [{min(mlm_grads):.6f}, {max(mlm_grads):.6f}]")
    
    print(f"\nClassification approach:")
    print(f"  Non-zero gradients: {cls_nonzero}/10 ({cls_nonzero*10}%)")
    print(f"  Average |gradient|: {np.mean(np.abs(cls_grads)):.6f}")
    
    print("\n" + "=" * 80)
    print("CONCLUSION: The paper's MLM approach provides continuous gradients!")
    print("This is why MeZO converges - it's NOT using discrete accuracy objective.")
    print("=" * 80)


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    test_mlm_vs_classification_approach()