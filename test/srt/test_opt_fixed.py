#!/usr/bin/env python3
"""Test OPT model with fixed configuration."""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Enable flexible distributed initialization
os.environ["SGLANG_ALLOW_REUSE_DISTRIBUTED"] = "true"

import torch
from transformers import OPTConfig, AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset

from python.sglang.srt.distributed.distributed_utils import (
    setup_environment_for_testing,
    cleanup_distributed,
)


def test_opt_direct():
    """Test OPT model directly to verify it works."""
    print("\nTest: Direct OPT Model Loading")
    print("=" * 50)
    
    try:
        # Load model directly
        print("1. Loading OPT-125m directly...")
        model = AutoModelForCausalLM.from_pretrained(
            "facebook/opt-125m",
            torch_dtype=torch.float16,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
        )
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        
        print(f"   ✓ Model loaded: {model.config.model_type}")
        print(f"   Model config attributes:")
        print(f"     - num_attention_heads: {model.config.num_attention_heads}")
        print(f"     - num_key_value_heads: {getattr(model.config, 'num_key_value_heads', 'NOT SET')}")
        print(f"     - hidden_size: {model.config.hidden_size}")
        
        # Test forward pass
        print("\n2. Testing forward pass...")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        inputs = tokenizer("Hello world", return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
        with torch.no_grad():
            outputs = model(**inputs)
            
        print(f"   ✓ Forward pass successful, output shape: {outputs.logits.shape}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_opt_with_sglang():
    """Test OPT with SGLang after fixing config."""
    print("\n\nTest: OPT with SGLang (Fixed Config)")
    print("=" * 50)
    
    try:
        # Setup environment
        setup_environment_for_testing(
            tp_size=1,
            pp_size=1,
            rank=0,
            local_rank=0,
            master_port="29508",
        )
        
        # Register OPT model and fix the config loading
        print("1. Fixing OPT model registration...")
        
        # Monkey patch the transformers loader to fix OPT config
        original_from_pretrained = OPTConfig.from_pretrained
        
        def fixed_from_pretrained(cls, *args, **kwargs):
            config = original_from_pretrained(*args, **kwargs)
            # Add missing attributes for SGLang compatibility
            if not hasattr(config, 'num_key_value_heads'):
                config.num_key_value_heads = config.num_attention_heads
            if not hasattr(config, 'is_generation'):
                config.is_generation = True
            if not hasattr(config, 'is_multimodal'):
                config.is_multimodal = False
            if not hasattr(config, 'is_embedding'):
                config.is_embedding = False
            return config
            
        OPTConfig.from_pretrained = classmethod(fixed_from_pretrained)
        
        print("   ✓ Applied OPT config fix")
        
        # Now try with ModelRunner
        from python.sglang.srt.server_args import ServerArgs
        from python.sglang.srt.configs.model_config import ModelConfig
        from python.sglang.srt.model_executor.model_runner import ModelRunner
        
        server_args = ServerArgs(
            model_path="facebook/opt-125m",
            trust_remote_code=True,
            tp_size=1,
            pp_size=1,
            mem_fraction_static=0.8,
            disable_cuda_graph=True,
            dtype="float16",
            grammar_backend="none",
        )
        
        model_config = ModelConfig.from_server_args(server_args)
        
        print("\n2. Creating ModelRunner for OPT-125m...")
        runner = ModelRunner(
            model_config=model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=0,
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            nccl_port=29508,
            server_args=server_args,
        )
        
        print("   ✓ ModelRunner created successfully!")
        print(f"   Model type: {type(runner.model)}")
        
        # Test creating second instance
        print("\n3. Creating second ModelRunner (reusing distributed)...")
        runner2 = ModelRunner(
            model_config=model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=0,
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            nccl_port=29508,
            server_args=server_args,
        )
        print("   ✓ Second ModelRunner created successfully!")
        
        # Restore original method
        OPTConfig.from_pretrained = original_from_pretrained
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        cleanup_distributed()


def main():
    """Run OPT model tests."""
    print("Testing OPT Model with Fixes")
    print("=" * 70)
    
    results = []
    
    # Test 1: Direct OPT loading
    results.append(("Direct OPT", test_opt_direct()))
    
    # Test 2: OPT with SGLang
    results.append(("OPT with SGLang", test_opt_with_sglang()))
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary:")
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    if all_passed:
        print("\n✓ All tests passed!")
        print("\nKey fixes applied:")
        print("  1. SGLANG_ALLOW_REUSE_DISTRIBUTED=true enables flexible distributed init")
        print("  2. OPT config patched to include num_key_value_heads")
        print("  3. Multiple ModelRunner instances can now be created without conflicts")
    else:
        print("\n✗ Some tests failed.")


if __name__ == "__main__":
    main()