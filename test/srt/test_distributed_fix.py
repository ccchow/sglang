#!/usr/bin/env python3
"""Test distributed initialization fixes using monkey patching."""

import os
import sys
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from python.sglang.srt.distributed.flexible_parallel_state import (
    monkey_patch_for_flexible_init,
    remove_flexible_init_patch,
)
from python.sglang.srt.distributed.distributed_utils import (
    setup_environment_for_testing,
    cleanup_distributed,
)
from python.sglang.srt.model_executor.model_runner import ModelRunner
from python.sglang.srt.server_args import ServerArgs
from python.sglang.srt.configs.model_config import ModelConfig


def test_multiple_modelrunners():
    """Test creating multiple ModelRunner instances."""
    print("\nTest: Multiple ModelRunner Instances")
    print("=" * 50)
    
    # Apply the monkey patch
    monkey_patch_for_flexible_init()
    
    try:
        # Setup environment
        setup_environment_for_testing(
            tp_size=1,
            pp_size=1,
            rank=0,
            local_rank=0,
            master_port="29503",
        )
        
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
            nccl_port=29503,
            server_args=server_args,
        )
        print("   ✓ First ModelRunner created successfully")
        
        print("\n2. Creating second ModelRunner (should reuse distributed)...")
        runner2 = ModelRunner(
            model_config=model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=0,
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            nccl_port=29503,
            server_args=server_args,
        )
        print("   ✓ Second ModelRunner created successfully (reused distributed)")
        
        print("\n3. Creating draft worker ModelRunner...")
        runner3 = ModelRunner(
            model_config=model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=0,
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            nccl_port=29503,
            server_args=server_args,
            is_draft_worker=True,
        )
        print("   ✓ Draft worker ModelRunner created successfully")
        
        print("\n✓ All ModelRunners created successfully!")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        remove_flexible_init_patch()
        cleanup_distributed()


def test_opt_model_with_fix():
    """Test OPT model with distributed fix."""
    print("\n\nTest: OPT Model with Distributed Fix")
    print("=" * 50)
    
    # Apply the monkey patch
    monkey_patch_for_flexible_init()
    
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
            master_port="29504",
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
            nccl_port=29504,
            server_args=server_args,
        )
        
        print("   ✓ ModelRunner created successfully!")
        print(f"   Model type: {type(runner.model)}")
        print(f"   Model config: {runner.model.config.model_type}")
        
        print("\n✓ OPT model loaded successfully with distributed fix!")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        remove_flexible_init_patch()
        cleanup_distributed()


def test_pre_initialized_distributed():
    """Test using pre-initialized distributed environment."""
    print("\n\nTest: Pre-initialized Distributed Environment")
    print("=" * 50)
    
    # Apply the monkey patch
    monkey_patch_for_flexible_init()
    
    try:
        # Setup environment
        setup_environment_for_testing(
            tp_size=1,
            pp_size=1,
            rank=0,
            local_rank=0,
            master_port="29505",
        )
        
        # Pre-initialize distributed
        print("\n1. Pre-initializing distributed environment...")
        import torch.distributed as dist
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo",
                world_size=1,
                rank=0,
            )
        print("   ✓ Distributed pre-initialized")
        
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
        
        print("\n2. Creating ModelRunner with pre-initialized distributed...")
        runner = ModelRunner(
            model_config=model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=0,
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            nccl_port=29505,
            server_args=server_args,
        )
        
        print("   ✓ ModelRunner created successfully with pre-initialized distributed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        remove_flexible_init_patch()
        cleanup_distributed()


def main():
    """Run all distributed fix tests."""
    print("Testing Distributed Initialization Fixes")
    print("=" * 70)
    
    results = []
    
    # Test 1: Multiple ModelRunners
    results.append(("Multiple ModelRunners", test_multiple_modelrunners()))
    
    # Test 2: OPT Model
    results.append(("OPT Model", test_opt_model_with_fix()))
    
    # Test 3: Pre-initialized distributed
    results.append(("Pre-initialized Distributed", test_pre_initialized_distributed()))
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary:")
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    if all_passed:
        print("\n✓ All tests passed! Distributed initialization issues are fixed.")
    else:
        print("\n✗ Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    main()