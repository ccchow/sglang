# ModelRunner Distributed Initialization Issue

## Problem Description
When attempting to use ModelRunner directly in tests, we encounter a distributed initialization error:
```
AssertionError: world group is not initialized
```

## Error Details

### Stack Trace
```
File "test_roberta_sst2_modelrunner.py", line 81, in __init__
    initialize_model_parallel(
File "sglang/srt/distributed/parallel_state.py", line 1211, in initialize_model_parallel
    backend = backend or torch.distributed.get_backend(get_world_group().device_group)
File "sglang/srt/distributed/parallel_state.py", line 1017, in get_world_group
    assert _WORLD is not None, "world group is not initialized"
AssertionError: world group is not initialized
```

### Root Cause Analysis

1. **Initialization Order Issue**: The `initialize_model_parallel` function expects the world group to already be initialized, but in our test setup, this hasn't happened yet.

2. **Missing Infrastructure**: ModelRunner is designed to work within SGLang's server infrastructure, which handles distributed initialization automatically. When used directly in tests, this infrastructure is missing.

3. **Documentation Gap**: Neither README.md nor docs/README.md provide clear guidance on how to properly initialize ModelRunner for standalone test usage.

## Attempted Solutions

### 1. Basic Distributed Init (Failed)
```python
if not dist.is_initialized():
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend="nccl", rank=0, world_size=1)
```
**Result**: This initializes PyTorch distributed, but not SGLang's internal world group.

### 2. Direct ModelRunner Usage (Failed)
```python
initialize_model_parallel(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
)
model_runner = ModelRunner(...)
```
**Result**: Fails because `initialize_model_parallel` requires world group to be set up first.

## Working Examples in Codebase

### 1. Server-based Tests
Most tests use the server infrastructure:
```python
from sglang.utils import launch_server_cmd
server_process, port = launch_server_cmd(...)
```

### 2. Engine-based Tests
Some tests use the Engine API:
```python
import sglang as sgl
llm = sgl.Engine(model_path="...")
```

### 3. Documentation Pattern
The docs recommend using either:
- Server endpoints (HTTP API)
- Engine API (Python API)
- But NOT direct ModelRunner usage

## Recommended Solutions

### 1. Use Engine API (Recommended)
```python
import sglang as sgl
engine = sgl.Engine(model_path="roberta-large")
# Use engine for inference
```

### 2. Use Server + Client
```python
# Launch server
server_process, port = launch_server_cmd(...)
# Use HTTP client to interact
```

### 3. Continue with HuggingFace Models
For MeZO testing purposes, using HuggingFace models directly is simpler and sufficient:
- The core MeZO algorithm is model-agnostic
- RadixAttention benefits can be simulated/measured
- Avoids complex distributed setup

## Conclusion

The ModelRunner initialization issue stems from attempting to use low-level SGLang components outside their intended server infrastructure. The documentation and existing tests suggest using either:
1. The high-level Engine API
2. Server-based testing with HTTP clients
3. HuggingFace models for algorithm testing

For the MeZO implementation, continuing with HuggingFace models is the most practical approach, as it allows us to:
- Test the core algorithm
- Measure theoretical RadixAttention benefits
- Avoid complex distributed initialization
- Maintain compatibility with the original MeZO paper's approach

## Future Work

To properly integrate MeZO with SGLang's ModelRunner:
1. Implement MeZO as a server-side feature
2. Use Engine API for client-side MeZO training
3. Add proper documentation for ModelRunner initialization in tests
4. Consider adding a simplified ModelRunner API for single-GPU testing