# ModelRunner Initialization Requirements

## Overview

The ModelRunner class in SGLang requires specific initialization steps and configuration to work properly. The "world group is not initialized" error occurs when the distributed environment is not properly set up before creating ModelRunner.

## Required Components

### 1. Distributed Environment Initialization

Before creating ModelRunner, you must initialize the distributed environment:

```python
import torch.distributed as dist
from sglang.srt.distributed import init_distributed_environment, initialize_model_parallel

# Initialize PyTorch distributed
if not dist.is_initialized():
    dist.init_process_group(
        backend="nccl",  # or "gloo" for CPU
        rank=0,          # Process rank
        world_size=1     # Total number of processes
    )

# Initialize SGLang distributed environment
init_distributed_environment(
    world_size=1,
    rank=0,
    distributed_init_method="env://",
    local_rank=0,
    backend="nccl"
)

# Initialize model parallel groups
initialize_model_parallel(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1
)
```

### 2. ModelConfig Requirements

ModelConfig requires the following attributes:
- `model_path`: Path to the model
- `trust_remote_code`: Whether to trust remote code (default True)
- `dtype`: Data type ("auto", "float16", "bfloat16", etc.)
- `quantization`: Optional quantization method
- `context_length`: Optional context length override
- `is_embedding`: Whether it's an embedding model
- `enable_multimodal`: Whether to enable multimodal support

Example:
```python
from sglang.srt.configs.model_config import ModelConfig

model_config = ModelConfig(
    model_path="gpt2",
    trust_remote_code=True,
    dtype="auto",
    quantization=None,
    context_length=None,
    is_embedding=False,
    enable_multimodal=False
)
```

### 3. ServerArgs Requirements

ServerArgs is a comprehensive dataclass with many fields. Key required fields:
- `model_path`: Path to the model
- `tokenizer_path`: Path to tokenizer (can be same as model_path)
- `device`: Device type ("cuda", "cpu", etc.)
- `tp_size`: Tensor parallel size (default 1)
- `pp_size`: Pipeline parallel size (default 1)
- `mem_fraction_static`: Memory fraction for static allocation
- `dtype`: Data type
- `trust_remote_code`: Whether to trust remote code

Example:
```python
from sglang.srt.server_args import ServerArgs

server_args = ServerArgs(
    model_path="gpt2",
    tokenizer_path="gpt2",
    device="cuda",
    tp_size=1,
    pp_size=1,
    mem_fraction_static=0.8,
    dtype="auto",
    trust_remote_code=True,
    # Many other fields have defaults
)
```

### 4. ModelRunner Constructor Parameters

ModelRunner requires:
- `model_config`: ModelConfig instance
- `mem_fraction_static`: Memory fraction (0.0-1.0)
- `gpu_id`: GPU device ID
- `tp_rank`: Tensor parallel rank
- `tp_size`: Tensor parallel size
- `pp_rank`: Pipeline parallel rank (default 0)
- `pp_size`: Pipeline parallel size (default 1)
- `nccl_port`: NCCL communication port
- `server_args`: ServerArgs instance
- `is_draft_worker`: Whether this is a draft worker (default False)
- `req_to_token_pool`: Optional request to token pool
- `token_to_kv_pool_allocator`: Optional KV cache allocator

## Initialization Sequence

1. **Set environment variables** (if needed):
   ```python
   import os
   os.environ["RANK"] = "0"
   os.environ["LOCAL_RANK"] = "0"
   os.environ["WORLD_SIZE"] = "1"
   os.environ["MASTER_ADDR"] = "127.0.0.1"
   os.environ["MASTER_PORT"] = "29500"
   ```

2. **Initialize PyTorch distributed**:
   ```python
   import torch.distributed as dist
   if not dist.is_initialized():
       dist.init_process_group(backend="nccl", rank=0, world_size=1)
   ```

3. **Initialize SGLang distributed environment**:
   ```python
   from sglang.srt.distributed import init_distributed_environment, initialize_model_parallel
   
   init_distributed_environment(
       world_size=1,
       rank=0,
       distributed_init_method="env://",
       local_rank=0,
       backend="nccl"
   )
   
   initialize_model_parallel(
       tensor_model_parallel_size=1,
       pipeline_model_parallel_size=1
   )
   ```

4. **Create ServerArgs**:
   ```python
   from sglang.srt.server_args import ServerArgs
   
   server_args = ServerArgs(
       model_path="gpt2",
       tokenizer_path="gpt2",
       device="cuda",
       tp_size=1,
       pp_size=1,
       # ... other fields
   )
   ```

5. **Create ModelConfig**:
   ```python
   from sglang.srt.configs.model_config import ModelConfig
   
   model_config = ModelConfig.from_server_args(server_args)
   ```

6. **Create ModelRunner**:
   ```python
   from sglang.srt.model_executor.model_runner import ModelRunner
   
   model_runner = ModelRunner(
       model_config=model_config,
       mem_fraction_static=0.8,
       gpu_id=0,
       tp_rank=0,
       tp_size=1,
       pp_rank=0,
       pp_size=1,
       nccl_port=28000,
       server_args=server_args
   )
   ```

## Common Issues and Solutions

### 1. "world group is not initialized"
- **Cause**: Distributed environment not initialized
- **Solution**: Call `init_distributed_environment()` before creating ModelRunner

### 2. "tensor model parallel group is not initialized"
- **Cause**: Model parallel groups not initialized
- **Solution**: Call `initialize_model_parallel()` after `init_distributed_environment()`

### 3. CUDA device errors
- **Cause**: GPU not properly set
- **Solution**: Set CUDA device with `torch.cuda.set_device(gpu_id)` if using CUDA

### 4. Port conflicts
- **Cause**: NCCL port already in use
- **Solution**: Use a different port for `nccl_port` parameter

## Environment Variables

Important environment variables that affect initialization:
- `RANK`: Process rank in distributed training
- `LOCAL_RANK`: Local rank on the node
- `WORLD_SIZE`: Total number of processes
- `MASTER_ADDR`: Master node address
- `MASTER_PORT`: Master node port
- `CUDA_VISIBLE_DEVICES`: Which GPUs to use
- `SGLANG_*`: Various SGLang-specific settings

## Minimal Working Example

```python
import os
import torch
import torch.distributed as dist
from sglang.srt.server_args import ServerArgs
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.distributed import init_distributed_environment, initialize_model_parallel

# Set environment variables
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"

# Initialize PyTorch distributed
if not dist.is_initialized():
    dist.init_process_group(backend="nccl", rank=0, world_size=1)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

# Initialize SGLang distributed
init_distributed_environment(
    world_size=1,
    rank=0,
    distributed_init_method="env://",
    local_rank=0,
    backend="nccl"
)

initialize_model_parallel(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1
)

# Create configurations
server_args = ServerArgs(
    model_path="gpt2",
    tokenizer_path="gpt2",
    device="cuda",
    tp_size=1,
    pp_size=1,
    mem_fraction_static=0.8,
    dtype="auto",
    trust_remote_code=True
)

model_config = ModelConfig.from_server_args(server_args)

# Create ModelRunner
model_runner = ModelRunner(
    model_config=model_config,
    mem_fraction_static=0.8,
    gpu_id=0,
    tp_rank=0,
    tp_size=1,
    pp_rank=0,
    pp_size=1,
    nccl_port=28000,
    server_args=server_args
)
```

## Implementation Plan for MeZO Integration

1. **Update MeZO initialization**:
   - Add distributed environment initialization before ModelRunner creation
   - Ensure all required ServerArgs fields are properly set
   - Handle both single-GPU and multi-GPU scenarios

2. **Create helper functions**:
   - `init_mezo_distributed()`: Initialize distributed environment for MeZO
   - `create_mezo_server_args()`: Create ServerArgs with MeZO-specific defaults
   - `validate_mezo_config()`: Validate configuration before initialization

3. **Add error handling**:
   - Check if distributed is already initialized
   - Validate GPU availability
   - Handle port conflicts gracefully
   - Provide clear error messages

4. **Test scenarios**:
   - Single GPU training
   - Multi-GPU data parallel
   - Multi-GPU tensor parallel
   - CPU-only training
   - Mixed precision training

5. **Documentation**:
   - Add examples for common MeZO use cases
   - Document environment variable requirements
   - Provide troubleshooting guide