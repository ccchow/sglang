# MeZO ModelRunner Initialization Fix Plan

## Current Issues

1. **Distributed Environment Not Initialized**
   - Error: "world group is not initialized"
   - Root cause: ModelRunner expects distributed environment to be set up
   - Location: When creating ModelRunner in MeZO trainer

2. **Missing Required ServerArgs Fields**
   - Several fields in ServerArgs need proper defaults for MeZO
   - Some fields are MeZO-specific (e.g., memory allocation for perturbations)

3. **Model Architecture Detection**
   - Need to properly detect and handle different model architectures
   - Some models require special configuration (e.g., attention backend)

## Proposed Solution

### 1. Create MeZO-Specific Initialization Module

Create `python/sglang/srt/mezo_init.py`:

```python
import os
import torch
import torch.distributed as dist
from typing import Optional, Dict, Any
from sglang.srt.server_args import ServerArgs
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed import init_distributed_environment, initialize_model_parallel
from sglang.srt.model_executor.model_runner import ModelRunner

def init_mezo_distributed(
    rank: int = 0,
    world_size: int = 1,
    local_rank: int = 0,
    master_addr: str = "127.0.0.1",
    master_port: str = "29500",
    backend: str = "nccl"
):
    """Initialize distributed environment for MeZO training."""
    # Set environment variables
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    
    # Initialize PyTorch distributed if not already initialized
    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size
        )
        
        # Set CUDA device if available
        if backend == "nccl" and torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    
    # Initialize SGLang distributed environment
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        distributed_init_method="env://",
        local_rank=local_rank,
        backend=backend
    )
    
    # Initialize model parallel groups (MeZO typically uses data parallel only)
    initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1
    )

def create_mezo_server_args(
    model_path: str,
    tokenizer_path: Optional[str] = None,
    device: str = "cuda",
    batch_size: int = 1,
    max_seq_length: int = 512,
    dtype: str = "auto",
    mem_fraction_static: float = 0.8,
    **kwargs
) -> ServerArgs:
    """Create ServerArgs optimized for MeZO training."""
    # Calculate max tokens needed for MeZO (2x for +/- perturbations)
    max_total_tokens = batch_size * max_seq_length * 2
    
    # Base arguments
    args_dict = {
        "model_path": model_path,
        "tokenizer_path": tokenizer_path or model_path,
        "device": device,
        "dtype": dtype,
        "mem_fraction_static": mem_fraction_static,
        "trust_remote_code": True,
        "tp_size": 1,
        "pp_size": 1,
        "max_total_tokens": max_total_tokens,
        "max_running_requests": batch_size * 2,  # For +/- perturbations
        "disable_radix_cache": False,  # Enable for KV cache efficiency
        "enable_nan_detection": True,  # Help debug gradient issues
        "disable_cuda_graph": True,  # Not needed for training
        "skip_tokenizer_init": False,
        "context_length": max_seq_length,
    }
    
    # Merge with any additional kwargs
    args_dict.update(kwargs)
    
    return ServerArgs(**args_dict)

def create_mezo_model_runner(
    model_path: str,
    tokenizer_path: Optional[str] = None,
    device: str = "cuda",
    batch_size: int = 1,
    max_seq_length: int = 512,
    dtype: str = "auto",
    mem_fraction_static: float = 0.8,
    gpu_id: int = 0,
    nccl_port: int = 28000,
    model_override_args: Optional[Dict[str, Any]] = None,
    **kwargs
) -> ModelRunner:
    """Create a ModelRunner instance configured for MeZO training."""
    # Initialize distributed if needed
    if not dist.is_initialized():
        init_mezo_distributed(
            backend="nccl" if device == "cuda" else "gloo"
        )
    
    # Create ServerArgs
    server_args = create_mezo_server_args(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        device=device,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        dtype=dtype,
        mem_fraction_static=mem_fraction_static,
        **kwargs
    )
    
    # Handle model override args
    if model_override_args:
        import json
        server_args.json_model_override_args = json.dumps(model_override_args)
    
    # Create ModelConfig
    model_config = ModelConfig.from_server_args(
        server_args,
        model_path=model_path
    )
    
    # Create ModelRunner
    model_runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=mem_fraction_static,
        gpu_id=gpu_id,
        tp_rank=0,  # MeZO typically uses single GPU
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        nccl_port=nccl_port,
        server_args=server_args
    )
    
    return model_runner
```

### 2. Update MeZO Trainer Initialization

Modify `python/sglang/srt/mezo_trainer.py`:

```python
# Add import
from sglang.srt.mezo_init import create_mezo_model_runner

# Update mezo_finetune function
def mezo_finetune(
    model_path: str,
    data_path: str,
    output_dir: str,
    # ... other parameters
):
    # Replace direct ModelRunner creation with:
    model_runner = create_mezo_model_runner(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        device=device,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        dtype=dtype,
        model_override_args=model_override_args,
        # Pass any other needed arguments
    )
    
    # Continue with trainer creation...
```

### 3. Add Validation and Error Handling

Create validation utilities:

```python
def validate_mezo_environment():
    """Validate that the environment is properly set up for MeZO."""
    issues = []
    
    # Check distributed initialization
    if not dist.is_initialized():
        issues.append("PyTorch distributed not initialized")
    
    # Check CUDA availability if needed
    if torch.cuda.is_available() and not torch.cuda.is_initialized():
        issues.append("CUDA not initialized")
    
    # Check SGLang distributed
    try:
        from sglang.srt.distributed import get_world_group
        get_world_group()
    except AssertionError:
        issues.append("SGLang world group not initialized")
    
    if issues:
        raise RuntimeError(f"MeZO environment validation failed:\n" + "\n".join(issues))
```

### 4. Create Helper Script for Testing

Create `test/srt/test_mezo_modelrunner_init.py`:

```python
"""Test MeZO ModelRunner initialization."""

import unittest
import torch
from sglang.srt.mezo_init import (
    init_mezo_distributed,
    create_mezo_server_args,
    create_mezo_model_runner,
    validate_mezo_environment
)

class TestMeZOModelRunnerInit(unittest.TestCase):
    def test_distributed_init(self):
        """Test distributed environment initialization."""
        init_mezo_distributed()
        validate_mezo_environment()
        
    def test_server_args_creation(self):
        """Test ServerArgs creation with MeZO defaults."""
        args = create_mezo_server_args(
            model_path="gpt2",
            batch_size=4,
            max_seq_length=128
        )
        
        self.assertEqual(args.model_path, "gpt2")
        self.assertEqual(args.tp_size, 1)
        self.assertEqual(args.max_total_tokens, 4 * 128 * 2)  # 2x for perturbations
        
    def test_model_runner_creation(self):
        """Test ModelRunner creation."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
            
        runner = create_mezo_model_runner(
            model_path="gpt2",
            batch_size=2,
            max_seq_length=64
        )
        
        self.assertIsNotNone(runner)
        self.assertEqual(runner.tp_size, 1)

if __name__ == "__main__":
    unittest.main()
```

## Implementation Steps

1. **Create mezo_init.py** with initialization utilities
2. **Update mezo_trainer.py** to use the new initialization
3. **Add validation** to catch configuration errors early
4. **Create tests** to verify initialization works correctly
5. **Update documentation** with initialization examples
6. **Test with various models** (GPT-2, RoBERTa, Llama, etc.)

## Benefits

1. **Simplified initialization** - Users don't need to understand distributed setup
2. **Better error messages** - Clear feedback on what's wrong
3. **MeZO-optimized defaults** - Configurations tuned for MeZO workloads
4. **Reusable components** - Can be used in different MeZO implementations
5. **Easier testing** - Consistent initialization across tests

## Testing Plan

1. **Unit tests**:
   - Test distributed initialization
   - Test ServerArgs creation
   - Test ModelRunner creation
   - Test validation functions

2. **Integration tests**:
   - Test with different models
   - Test with different batch sizes
   - Test error handling
   - Test multi-GPU scenarios

3. **Performance tests**:
   - Verify KV cache efficiency
   - Check memory usage
   - Measure initialization time

## Next Steps

1. Implement the mezo_init.py module
2. Update existing MeZO code to use new initialization
3. Add comprehensive tests
4. Update documentation
5. Test with real MeZO training scenarios