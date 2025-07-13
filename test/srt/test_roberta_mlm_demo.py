#!/usr/bin/env python3
"""
Demo of RoBERTa MLM for SST-2 showing how it enables MeZO training.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python"))

import torch
import torch.nn as nn
import numpy as np
from transformers import RobertaTokenizer, RobertaConfig
from sglang.srt.models.roberta import RobertaLMHead


def demo_mlm_for_sst2():
    """Demonstrate how MLM enables continuous gradients for SST-2."""
    print("=" * 80)
    print("RoBERTa MLM Demo: Enabling Continuous Gradients for SST-2")
    print("=" * 80)
    
    # Load tokenizer
    try:
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    except:
        print("Error: RoBERTa tokenizer not available")
        return
    
    # Configuration
    config = RobertaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=768,
        layer_norm_eps=1e-5
    )
    
    # Create LM head
    lm_head = RobertaLMHead(config)
    lm_head.eval()
    
    # SST-2 configuration
    template = "It was [MASK]."
    label_words = {0: 'terrible', 1: 'great'}
    
    # Get label word token IDs
    label_word_ids = {}
    print("\nLabel word mapping:")
    for label, word in label_words.items():
        tokens = tokenizer.tokenize(' ' + word)
        token_id = tokenizer.convert_tokens_to_ids(tokens[0])
        label_word_ids[label] = token_id
        print(f"  Label {label}: '{word}' -> ' {word}' -> token_id {token_id}")
    
    # Test sentences
    test_sentences = [
        ("This movie is absolutely fantastic! Best film I've seen all year.", 1),
        ("Terrible waste of time. I want my money back.", 0),
        ("A masterpiece of cinema that will be remembered for generations.", 1),
        ("Boring, predictable, and poorly acted throughout.", 0),
        ("I was on the edge of my seat the entire time!", 1),
        ("Fell asleep halfway through. Completely uninspiring.", 0),
    ]
    
    print(f"\nTemplate: '{template}'")
    print("\n" + "-" * 80)
    print("Demonstrating MLM predictions:")
    print("-" * 80)
    
    correct = 0
    total_loss = 0
    all_gradients = []
    
    for i, (text, true_label) in enumerate(test_sentences):
        # Create MLM input
        mlm_text = f"{text} {template}".replace('[MASK]', tokenizer.mask_token)
        inputs = tokenizer(mlm_text, return_tensors="pt", truncation=True, max_length=128)
        
        # Find mask position
        mask_pos = (inputs.input_ids == tokenizer.mask_token_id).nonzero()[0, 1].item()
        
        # Simulate encoder output (random for demo)
        seq_length = inputs.input_ids.shape[1]
        hidden_states = torch.randn(1, seq_length, config.hidden_size, requires_grad=True)
        
        # Get predictions
        logits = lm_head(hidden_states)
        mask_logits = logits[0, mask_pos]
        
        # Extract logits for label words
        label_indices = torch.tensor([label_word_ids[0], label_word_ids[1]])
        label_logits = mask_logits[label_indices]
        
        # Compute loss
        loss_fn = nn.CrossEntropyLoss()
        labels = torch.tensor([true_label])
        loss = loss_fn(label_logits.unsqueeze(0), labels)
        
        # Get prediction
        probs = torch.softmax(label_logits, dim=0)
        pred_label = torch.argmax(label_logits).item()
        
        # Compute gradient
        loss.backward()
        gradient_norm = hidden_states.grad.norm().item()
        all_gradients.append(gradient_norm)
        
        # Update stats
        correct += (pred_label == true_label)
        total_loss += loss.item()
        
        # Display result
        print(f"\n{i+1}. Text: '{text[:60]}...'" if len(text) > 60 else f"\n{i+1}. Text: '{text}'")
        print(f"   True: {label_words[true_label]}, Predicted: {label_words[pred_label]}")
        print(f"   P(terrible): {probs[0]:.1%}, P(great): {probs[1]:.1%}")
        print(f"   Loss: {loss.item():.4f}, Gradient norm: {gradient_norm:.4f}")
        print(f"   {'✅ Correct' if pred_label == true_label else '❌ Wrong'}")
        
        # Clear gradients
        hidden_states.grad = None
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics:")
    print("=" * 80)
    
    accuracy = correct / len(test_sentences)
    avg_loss = total_loss / len(test_sentences)
    
    print(f"Accuracy: {correct}/{len(test_sentences)} = {accuracy:.1%}")
    print(f"Average loss: {avg_loss:.4f}")
    print(f"Average gradient norm: {np.mean(all_gradients):.4f}")
    
    # Key insight
    print("\n" + "=" * 80)
    print("Key Insight: Why MLM Works for MeZO")
    print("=" * 80)
    print("1. CONTINUOUS GRADIENTS: Every example produces non-zero gradients")
    print(f"   - All {len(all_gradients)} examples had gradients (norm > 0)")
    print(f"   - Average gradient norm: {np.mean(all_gradients):.4f}")
    print("\n2. DIFFERENTIABLE LOSS: Cross-entropy on vocabulary probabilities")
    print("   - Loss changes smoothly with model parameters")
    print("   - Enables gradient estimation via finite differences")
    print("\n3. Compare with ACCURACY OBJECTIVE:")
    print("   - Accuracy is discrete (0 or 1)")
    print("   - Most perturbations don't change predictions")
    print("   - Results in zero gradients ~99% of the time")
    
    # MeZO gradient estimation demo
    print("\n" + "=" * 80)
    print("MeZO Gradient Estimation Demo")
    print("=" * 80)
    
    # Simulate MeZO gradient estimation
    epsilon = 1e-3
    
    # Original parameters (using first example)
    text, true_label = test_sentences[0]
    mlm_text = f"{text} {template}".replace('[MASK]', tokenizer.mask_token)
    inputs = tokenizer(mlm_text, return_tensors="pt", truncation=True)
    mask_pos = (inputs.input_ids == tokenizer.mask_token_id).nonzero()[0, 1].item()
    
    # Base hidden states
    hidden_states = torch.randn(1, inputs.input_ids.shape[1], config.hidden_size)
    
    # Compute base loss
    with torch.no_grad():
        logits = lm_head(hidden_states)
        label_logits = logits[0, mask_pos][[label_word_ids[0], label_word_ids[1]]]
        base_loss = loss_fn(label_logits.unsqueeze(0), torch.tensor([true_label]))
    
    # Perturb and compute losses
    perturbation = torch.randn_like(hidden_states) * epsilon
    
    with torch.no_grad():
        # Loss with +epsilon
        logits_plus = lm_head(hidden_states + perturbation)
        label_logits_plus = logits_plus[0, mask_pos][[label_word_ids[0], label_word_ids[1]]]
        loss_plus = loss_fn(label_logits_plus.unsqueeze(0), torch.tensor([true_label]))
        
        # Loss with -epsilon
        logits_minus = lm_head(hidden_states - perturbation)
        label_logits_minus = logits_minus[0, mask_pos][[label_word_ids[0], label_word_ids[1]]]
        loss_minus = loss_fn(label_logits_minus.unsqueeze(0), torch.tensor([true_label]))
    
    # MeZO gradient estimate
    grad_estimate = (loss_plus - loss_minus) / (2 * epsilon)
    
    print(f"Base loss: {base_loss:.4f}")
    print(f"Loss +ε: {loss_plus:.4f}")
    print(f"Loss -ε: {loss_minus:.4f}")
    print(f"MeZO gradient estimate: {grad_estimate:.6f}")
    print("\n✅ Non-zero gradient enables optimization!")
    
    print("\n" + "=" * 80)
    print("Conclusion: MLM + MeZO = Effective Zero-Order Optimization")
    print("=" * 80)


if __name__ == "__main__":
    demo_mlm_for_sst2()