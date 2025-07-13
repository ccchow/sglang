#!/usr/bin/env python3
"""Test distributed initialization with SGLANG_ALLOW_REUSE_DISTRIBUTED environment variable."""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Enable flexible distributed initialization
os.environ["SGLANG_ALLOW_REUSE_DISTRIBUTED"] = "true"

from python.sglang.srt.distributed.distributed_utils import (
    setup_environment_for_testing,
    cleanup_distributed,
)
from python.sglang.srt.model_executor.model_runner import ModelRunner
from python.sglang.srt.server_args import ServerArgs
from python.sglang.srt.configs.model_config import ModelConfig


def test_multiple_modelrunners_with_env():
    """Test creating multiple ModelRunner instances with environment variable."""
    print("\nTest: Multiple ModelRunners with SGLANG_ALLOW_REUSE_DISTRIBUTED=true")
    print("=" * 70)
    
    try:
        # Setup environment
        setup_environment_for_testing(
            tp_size=1,
            pp_size=1,
            rank=0,
            local_rank=0,
            master_port="29506",
        )
        
        print(f"Environment variable set: SGLANG_ALLOW_REUSE_DISTRIBUTED={os.environ.get('SGLANG_ALLOW_REUSE_DISTRIBUTED')}")
        
        # Create server args
        server_args = ServerArgs(
            model_path="gpt2",
            trust_remote_code=True,
            tp_size=1,
            pp_size=1,
            mem_fraction_static=0.8,
            disable_cuda_graph=True,
            dtype="float16",
            grammar_backend="none",
        )
        
        # Create model config
        model_config = ModelConfig.from_server_args(server_args)
        
        print("\n1. Creating first ModelRunner...")
        runner1 = ModelRunner(
            model_config=model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=0,
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            nccl_port=29506,
            server_args=server_args,
        )
        print("   ✓ First ModelRunner created successfully")
        print(f"   Model loaded: {type(runner1.model)}")
        
        print("\n2. Creating second ModelRunner (should reuse distributed)...")
        runner2 = ModelRunner(
            model_config=model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=0,
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            nccl_port=29506,
            server_args=server_args,
        )
        print("   ✓ Second ModelRunner created successfully (reused distributed)")
        print(f"   Model loaded: {type(runner2.model)}")
        
        print("\n3. Creating third ModelRunner (draft worker)...")
        runner3 = ModelRunner(
            model_config=model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=0,
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            nccl_port=29506,
            server_args=server_args,
            is_draft_worker=True,
        )
        print("   ✓ Draft worker ModelRunner created successfully")
        
        print("\n✓ All ModelRunners created successfully with flexible distributed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        cleanup_distributed()


def test_opt_with_flexible_env():
    """Test OPT model with flexible distributed environment."""
    print("\n\nTest: OPT Model with Flexible Distributed")
    print("=" * 70)
    
    try:
        # Register OPT model
        from python.sglang.srt.models.opt_complete import OPTForCausalLM
        from python.sglang.srt.models.registry import ModelRegistry
        
        ModelRegistry.models["OPTForCausalLM"] = OPTForCausalLM
        print("1. Registered OPTForCausalLM model")
        
        # Setup environment
        setup_environment_for_testing(
            tp_size=1,
            pp_size=1,
            rank=0,
            local_rank=0,
            master_port="29507",
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
            nccl_port=29507,
            server_args=server_args,
        )
        
        print("   ✓ ModelRunner created successfully!")
        print(f"   Model type: {type(runner.model)}")
        print(f"   Model parameters: {sum(p.numel() for p in runner.model.parameters()):,}")
        
        # Test multiple instances
        print("\n3. Creating second OPT ModelRunner (reusing distributed)...")
        runner2 = ModelRunner(
            model_config=model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=0,
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            nccl_port=29507,
            server_args=server_args,
        )
        print("   ✓ Second OPT ModelRunner created successfully!")
        
        print("\n✓ OPT model works with flexible distributed initialization!")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        cleanup_distributed()


def main():
    """Run all flexible distributed tests."""
    print("Testing Flexible Distributed Initialization")
    print("=" * 70)
    print(f"\nSGLANG_ALLOW_REUSE_DISTRIBUTED = {os.environ.get('SGLANG_ALLOW_REUSE_DISTRIBUTED')}")
    
    results = []
    
    # Test 1: Multiple ModelRunners
    results.append(("Multiple ModelRunners", test_multiple_modelrunners_with_env()))
    
    # Test 2: OPT Model
    results.append(("OPT Model", test_opt_with_flexible_env()))
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary:")
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    if all_passed:
        print("\n✓ All tests passed! Distributed initialization is now flexible.")
        print("\nTo use this feature in your code:")
        print("  1. Set environment variable: export SGLANG_ALLOW_REUSE_DISTRIBUTED=true")
        print("  2. Create multiple ModelRunner instances without conflicts")
        print("  3. Reuse existing distributed initialization across instances")
    else:
        print("\n✗ Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    main()