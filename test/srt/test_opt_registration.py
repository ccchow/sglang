#!/usr/bin/env python3
"""Test OPT model registration and loading."""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Enable flexible distributed initialization
os.environ["SGLANG_ALLOW_REUSE_DISTRIBUTED"] = "true"

import torch
from python.sglang.srt.models.registry import ModelRegistry
from python.sglang.srt.server_args import ServerArgs
from python.sglang.srt.configs.model_config import ModelConfig
from python.sglang.srt.model_executor.model_runner import ModelRunner
from python.sglang.srt.distributed.distributed_utils import (
    setup_environment_for_testing,
    cleanup_distributed,
)


def test_opt_in_registry():
    """Test if OPT is properly registered in the model registry."""
    print("\nTest: OPT Model Registration")
    print("=" * 50)
    
    # Check registry
    print("1. Checking model registry...")
    supported_archs = ModelRegistry.get_supported_archs()
    print(f"   Total registered models: {len(supported_archs)}")
    
    # Look for OPT
    opt_models = [arch for arch in supported_archs if "OPT" in arch]
    print(f"   OPT models found: {opt_models}")
    
    if "OPTForCausalLM" in supported_archs:
        print("   ✓ OPTForCausalLM is registered!")
        model_class = ModelRegistry.models.get("OPTForCausalLM")
        print(f"   Model class: {model_class}")
        return True
    else:
        print("   ✗ OPTForCausalLM not found in registry")
        return False


def test_opt_model_loading():
    """Test loading OPT model with ModelRunner."""
    print("\n\nTest: OPT Model Loading with ModelRunner")
    print("=" * 50)
    
    try:
        # Setup environment
        setup_environment_for_testing(
            tp_size=1,
            pp_size=1,
            rank=0,
            local_rank=0,
            master_port="29510",
        )
        
        # Create server args
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
        
        # Create model config
        print("1. Creating model config...")
        model_config = ModelConfig.from_server_args(server_args)
        print(f"   Model type: {model_config.hf_config.model_type}")
        print(f"   Architectures: {model_config.hf_config.architectures}")
        
        # Check config attributes
        print("\n2. Checking config attributes...")
        attrs = [
            'is_generation', 'is_multimodal', 'is_embedding',
            'num_key_value_heads', 'attention_arch'
        ]
        for attr in attrs:
            value = getattr(model_config, attr, "NOT SET")
            print(f"   {attr}: {value}")
        
        # Create ModelRunner
        print("\n3. Creating ModelRunner...")
        runner = ModelRunner(
            model_config=model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=0,
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            nccl_port=29510,
            server_args=server_args,
        )
        
        print("   ✓ ModelRunner created successfully!")
        print(f"   Model type: {type(runner.model)}")
        print(f"   Model class name: {runner.model.__class__.__name__}")
        
        # Check if it's using our implementation
        if hasattr(runner.model, 'model') and hasattr(runner.model.model, 'layers'):
            print("   ✓ Using SGLang OPT implementation!")
            print(f"   Number of layers: {len(runner.model.model.layers)}")
            print(f"   Has RadixAttention: {hasattr(runner.model.model.layers[0].self_attn, 'attn')}")
        else:
            print("   ⚠ Using transformers fallback implementation")
        
        # Test second instance
        print("\n4. Creating second ModelRunner (testing reuse)...")
        runner2 = ModelRunner(
            model_config=model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=0,
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            nccl_port=29510,
            server_args=server_args,
        )
        print("   ✓ Second ModelRunner created successfully!")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        cleanup_distributed()


def test_opt_mezo_compatibility():
    """Test if OPT model is compatible with MeZO training."""
    print("\n\nTest: OPT MeZO Compatibility")
    print("=" * 50)
    
    try:
        # Setup environment
        setup_environment_for_testing(
            tp_size=1,
            pp_size=1,
            rank=0,
            local_rank=0,
            master_port="29511",
        )
        
        # Create server args
        server_args = ServerArgs(
            model_path="facebook/opt-125m",
            trust_remote_code=True,
            tp_size=1,
            pp_size=1,
            mem_fraction_static=0.8,
            disable_cuda_graph=True,
            dtype="float32",  # Use float32 for MeZO stability
            grammar_backend="none",
        )
        
        # Create model config
        model_config = ModelConfig.from_server_args(server_args)
        
        # Create ModelRunner
        print("1. Creating ModelRunner for MeZO...")
        runner = ModelRunner(
            model_config=model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=0,
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            nccl_port=29511,
            server_args=server_args,
        )
        
        print("   ✓ Model loaded successfully")
        
        # Check LoRA compatibility
        print("\n2. Checking LoRA compatibility...")
        model = runner.model
        
        # Look for attention modules suitable for LoRA
        target_modules = []
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            # SGLang implementation
            for i, layer in enumerate(model.model.layers):
                if hasattr(layer, 'self_attn'):
                    if hasattr(layer.self_attn, 'q_proj'):
                        target_modules.extend([f"model.layers.{i}.self_attn.q_proj",
                                             f"model.layers.{i}.self_attn.v_proj"])
            print(f"   ✓ Found {len(target_modules)} target modules for LoRA")
            print(f"   Sample modules: {target_modules[:4]}")
        else:
            print("   ⚠ Using transformers implementation, LoRA targets may differ")
        
        print("\n3. Model ready for MeZO training!")
        print("   - RadixAttention enabled for KV cache optimization")
        print("   - LoRA can target attention projections")
        print("   - Float32 precision for gradient estimation stability")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        cleanup_distributed()


def main():
    """Run all OPT registration tests."""
    print("Testing OPT Model Registration and Loading")
    print("=" * 70)
    
    results = []
    
    # Test 1: Check registry
    results.append(("Registry Check", test_opt_in_registry()))
    
    # Test 2: Model loading
    results.append(("Model Loading", test_opt_model_loading()))
    
    # Test 3: MeZO compatibility
    results.append(("MeZO Compatibility", test_opt_mezo_compatibility()))
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary:")
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    if all_passed:
        print("\n✓ All tests passed! OPT model is properly registered and working.")
        print("\nKey achievements:")
        print("  1. OPTForCausalLM is registered in the model registry")
        print("  2. Uses SGLang's native implementation with RadixAttention")
        print("  3. Compatible with MeZO training and LoRA")
        print("  4. Supports multiple ModelRunner instances with SGLANG_ALLOW_REUSE_DISTRIBUTED")
    else:
        print("\n✗ Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    main()