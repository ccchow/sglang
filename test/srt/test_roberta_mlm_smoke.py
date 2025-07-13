#!/usr/bin/env python3
"""
Smoke test for RoBERTa MLM implementation.
Quick verification that all components work together.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python"))

import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaConfig

# Test imports
print("Testing imports...")
try:
    from sglang.srt.models.roberta import XLMRobertaForMaskedLM, RobertaLMHead
    print("✅ XLMRobertaForMaskedLM imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def test_lm_head():
    """Test LM head component."""
    print("\n1. Testing RobertaLMHead...")
    
    config = RobertaConfig(
        vocab_size=1000,
        hidden_size=128,
        layer_norm_eps=1e-5
    )
    
    lm_head = RobertaLMHead(config)
    
    # Test forward pass
    batch_size = 2
    seq_length = 10
    hidden_states = torch.randn(batch_size, seq_length, config.hidden_size)
    
    output = lm_head(hidden_states)
    assert output.shape == (batch_size, seq_length, config.vocab_size), f"Wrong shape: {output.shape}"
    
    print("✅ LM head forward pass successful")
    return True


def test_mlm_loss():
    """Test MLM loss computation."""
    print("\n2. Testing MLM loss computation...")
    
    config = RobertaConfig(
        vocab_size=100,
        hidden_size=64,
        layer_norm_eps=1e-5
    )
    
    lm_head = RobertaLMHead(config)
    
    # Forward pass
    hidden_states = torch.randn(4, 8, config.hidden_size)
    logits = lm_head(hidden_states)
    
    # Create labels with some masked positions
    labels = torch.randint(0, config.vocab_size, (4, 8))
    labels[:, [0, 2, 4, 6]] = -100  # Non-masked positions
    
    # Compute loss
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fn(logits.view(-1, config.vocab_size), labels.view(-1))
    
    assert loss.item() > 0, "Loss should be positive"
    print(f"✅ MLM loss computed: {loss.item():.4f}")
    return True


def test_sst2_mlm():
    """Test SST-2 MLM setup."""
    print("\n3. Testing SST-2 MLM configuration...")
    
    try:
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    except:
        print("⚠️  Skipping tokenizer test (not available)")
        return True
    
    # Label words
    label_words = {0: 'terrible', 1: 'great'}
    template = "It was [MASK]."
    
    # Get token IDs
    label_word_ids = {}
    for label, word in label_words.items():
        tokens = tokenizer.tokenize(' ' + word)
        if len(tokens) == 1:
            token_id = tokenizer.convert_tokens_to_ids(tokens[0])
            label_word_ids[label] = token_id
            print(f"  Label {label}: ' {word}' -> token {tokens[0]} -> id {token_id}")
    
    # Test MLM formatting
    test_text = "This movie is great!"
    mlm_text = f"{test_text} {template}".replace('[MASK]', tokenizer.mask_token)
    
    inputs = tokenizer(mlm_text, return_tensors='pt')
    mask_pos = (inputs.input_ids == tokenizer.mask_token_id).nonzero()
    
    assert len(mask_pos) > 0, "No mask token found"
    print(f"✅ Mask position found at: {mask_pos[0, 1].item()}")
    
    return True


def test_mezo_gradient_estimation():
    """Test MeZO gradient estimation with MLM."""
    print("\n4. Testing MeZO gradient estimation...")
    
    # Simple model
    config = RobertaConfig(vocab_size=100, hidden_size=64)
    lm_head = RobertaLMHead(config)
    
    # Parameters
    param = lm_head.dense.weight
    original_param = param.data.clone()
    epsilon = 1e-3
    
    # Mock data
    hidden_states = torch.randn(1, 10, config.hidden_size)
    mask_pos = 5
    true_label = 42
    
    # MeZO gradient estimation
    z = torch.randn_like(param)
    
    # Forward with +epsilon
    param.data = original_param + epsilon * z
    logits_plus = lm_head(hidden_states)
    loss_plus = torch.nn.functional.cross_entropy(
        logits_plus[0, mask_pos].unsqueeze(0),
        torch.tensor([true_label])
    )
    
    # Forward with -epsilon
    param.data = original_param - epsilon * z
    logits_minus = lm_head(hidden_states)
    loss_minus = torch.nn.functional.cross_entropy(
        logits_minus[0, mask_pos].unsqueeze(0),
        torch.tensor([true_label])
    )
    
    # Gradient estimate
    grad_est = (loss_plus - loss_minus) / (2 * epsilon)
    
    print(f"  Loss +ε: {loss_plus.item():.4f}")
    print(f"  Loss -ε: {loss_minus.item():.4f}")
    print(f"  Gradient estimate: {grad_est.item():.6f}")
    
    assert grad_est != 0, "Gradient should be non-zero"
    print("✅ Non-zero gradient obtained!")
    
    return True


def test_model_in_entry_class():
    """Verify model is registered."""
    print("\n5. Testing model registration...")
    
    from sglang.srt.models.roberta import EntryClass
    
    assert XLMRobertaForMaskedLM in EntryClass, "XLMRobertaForMaskedLM not in EntryClass"
    print("✅ XLMRobertaForMaskedLM registered in EntryClass")
    
    return True


def run_smoke_test():
    """Run all smoke tests."""
    print("=" * 60)
    print("RoBERTa MLM Smoke Test")
    print("=" * 60)
    
    tests = [
        ("LM Head", test_lm_head),
        ("MLM Loss", test_mlm_loss),
        ("SST-2 MLM", test_sst2_mlm),
        ("MeZO Gradient", test_mezo_gradient_estimation),
        ("Model Registration", test_model_in_entry_class),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✅ All smoke tests passed!")
        print("\nNext steps:")
        print("1. Run simplified test: python test/srt/test_full_roberta_large_mlm_simple.py")
        print("2. Run full test: python test/srt/test_full_roberta_large_mlm.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)