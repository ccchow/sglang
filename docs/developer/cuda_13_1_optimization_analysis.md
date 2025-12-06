# CUDA Toolkit 13.1 LLM Inference Optimization Analysis for SGLang

## Executive Summary

CUDA Toolkit 13.1, released December 4, 2025, represents NVIDIA's largest platform update in two decades. This document analyzes how sglang can leverage new CUDA 13.1 features to improve LLM inference performance, building upon sglang's existing extensive CUDA infrastructure.

## Current State of CUDA in SGLang

SGLang already has sophisticated CUDA infrastructure including:

- **69 CUDA kernel files** (~26,600 lines) covering attention, GEMM, MoE, and quantization
- **SM100 (Blackwell) support** via CUTLASS for compute capability 10.x
- **Green contexts** implementation in `csrc/spatial/greenctx_stream.cu`
- **NVFP4 quantization** support for 4-bit floating point inference
- **FP8 blockwise GEMM** for mixed-precision computation
- **TMA (Tensor Memory Accelerator)** usage in MLA kernels
- **DeepGEMM JIT compilation** via NVRTC

## CUDA 13.1 Key Features and Optimization Opportunities

### 1. CUDA Tile Programming Model

**Feature**: A new tile-based programming abstraction with Virtual ISA (CUDA Tile IR) that automatically leverages Tensor Cores and TMA without explicit programming.

**Opportunity for sglang**:
- **cuTile Python DSL** can simplify development of new attention kernels
- Automatic hardware abstraction enables forward compatibility with future GPU architectures
- Could replace some Triton kernels with cuTile implementations for better Tensor Core utilization

**Affected components**:
- `python/sglang/srt/layers/attention/triton_ops/` - Triton attention implementations
- `sgl-kernel/csrc/attention/` - Custom attention kernels

**Potential benefits**:
- Reduced kernel development time
- Automatic optimization for Tensor Cores without manual tuning
- Better portability across Hopper → Blackwell → future architectures

### 2. Green Contexts Runtime API Exposure

**Feature**: CUDA 13.1 exposes green contexts in the Runtime API (previously driver API only), enabling fine-grained SM partitioning with simpler code.

**Current sglang state**: Already uses green contexts via driver API in `csrc/spatial/greenctx_stream.cu`

**Opportunity**:
- Migrate from driver API (`cuGreenCtxCreate`, `cuGreenCtxDestroy`) to simpler runtime API
- Improved SM allocation visibility in profiling tools (timeline rows show SM allocation in tooltips)
- Better integration with PyTorch's CUDA runtime

**Benefits**:
- Simpler code maintenance
- Better debugging and profiling
- Improved determinism for latency-sensitive workloads

### 3. cuBLAS Grouped GEMM with Autotune

**Feature**: New experimental grouped GEMM API in cuBLASLt with automatic algorithm selection via `CUBLAS_GEMM_AUTOTUNE`.

**Details**:
- Supports FP8 (E4M3/E5M2), FP16, BF16 input types
- Per-batch tensor-wide scaling for FP8
- Algorithm caching within `cublasHandle_t`

**Opportunity for sglang**:
- **MoE (Mixture of Experts) kernels**: Replace custom grouped GEMM implementations
- **Batch inference**: Autotune can select optimal algorithms for variable batch sizes

**Affected components**:
- `sgl-kernel/csrc/moe/fp8_blockwise_moe_kernel.cu`
- `sgl-kernel/csrc/gemm/bmm_fp8.cu`
- Expert specialization kernels

**Potential benefits**:
- 10-20% performance improvement from automatic algorithm selection
- Reduced need for manual kernel tuning
- Better FP8 scaling support

### 4. NVRTC Compilation Improvements

**Feature**:
- Faster compile times for small programs (builtin functions moved to compiler)
- `cuda_fp16.h` and `cuda_bf16.h` moved to compiler bitcode
- New `--Ofast-compile=<level>` option for faster development cycles
- Deterministic PTX generation with `--frandom-seed=<seed>`

**Opportunity for sglang**:
- **DeepGEMM JIT**: Faster runtime kernel compilation
- **Deterministic builds**: Reproducible kernel generation for debugging

**Affected components**:
- `sgl-kernel/csrc/` - All JIT-compiled kernels via DeepGEMM
- `CMakeLists.txt` - Build configuration

**Benefits**:
- Faster server startup with JIT compilation
- More reproducible builds for CI/CD

### 5. Memory Locality Optimization (MLOPart)

**Feature**: On Blackwell GPUs, create specialized CUDA devices optimized for memory locality with fewer compute resources but better memory access patterns.

**Opportunity**:
- Optimize memory-bound operations like KV cache management
- Better utilization for disaggregated prefill-decode scenarios

**Affected components**:
- `python/sglang/srt/mem_cache/memory_pool.py`
- `sgl-kernel/csrc/kvcacheio/transfer.cu`

### 6. cuBLAS Performance Improvements for Blackwell

**Feature**:
- FP32 emulation with BF16 Tensor Cores (higher performance, preserved accuracy)
- FP64 emulation with INT8 Tensor Cores
- Improved heuristics for Blackwell FP32 GEMMs where M, N >> K
- Better FP16, FP8 GEMM performance on DGX Spark

**Opportunity**:
- Enable FP32 emulation for numerical stability scenarios
- Leverage improved heuristics for LLM linear layers

**Benefits**:
- 1.5x-3x speedup for operations requiring FP32 precision
- Better performance on Blackwell Thor (Jetson)

### 7. CUTLASS 4.0 Integration

**Feature**:
- CuTe DSL for Python-native kernel development
- Block-scaled data types (NVFP4, MXFP4, MXFP6, MXFP8)
- Optimized group GEMM with async TMA descriptor updates
- SM103 (Blackwell GeForce) support

**Current sglang state**: Uses CUTLASS via FetchContent (commit 57e3cfb)

**Opportunity**:
- Upgrade to CUTLASS 4.x for latest optimizations
- Use CuTe DSL for faster kernel development
- Enable MXFP4/MXFP6 formats for even lower precision

**Affected components**:
- `sgl-kernel/CMakeLists.txt` - CUTLASS version
- All CUTLASS-based kernels in `csrc/gemm/` and `csrc/attention/`

## Implementation Recommendations

### High Priority (Immediate Impact)

1. **cuBLAS Autotune Integration**
   - Location: `sgl-kernel/csrc/gemm/bmm_fp8.cu`
   - Action: Add `CUBLAS_GEMM_AUTOTUNE` support for FP8 batched GEMM
   - Expected benefit: 10-15% performance improvement

2. **Grouped GEMM for MoE**
   - Location: `sgl-kernel/csrc/moe/fp8_blockwise_moe_kernel.cu`
   - Action: Evaluate cuBLASLt grouped GEMM API vs current CUTLASS implementation
   - Expected benefit: Simpler code, potential performance gains

3. **NVRTC Compilation Flags**
   - Location: `sgl-kernel/CMakeLists.txt`
   - Action: Add `--Ofast-compile` for debug builds, `--frandom-seed` for reproducibility
   - Expected benefit: Faster development iteration

### Medium Priority (Near-term Optimization)

4. **Green Contexts Runtime API Migration**
   - Location: `sgl-kernel/csrc/spatial/greenctx_stream.cu`
   - Action: Migrate from driver API to runtime API when CUDA 13.1 is baseline
   - Expected benefit: Simpler code, better profiling

5. **CUTLASS 4.0 Upgrade**
   - Location: `sgl-kernel/CMakeLists.txt`
   - Action: Update CUTLASS FetchContent to 4.x release
   - Expected benefit: Latest optimizations, SM103 support

6. **FP32 Emulation Evaluation**
   - Location: Model execution paths requiring FP32
   - Action: Benchmark FP32 emulation vs native for applicable workloads
   - Expected benefit: Up to 3x speedup for FP32 operations

### Future Exploration

7. **cuTile Python Kernels**
   - Evaluate replacing Triton kernels with cuTile for specific operations
   - Start with simpler kernels (elementwise, reduction) before attention

8. **MLOPart for Disaggregated Inference**
   - Investigate memory locality optimization for prefill-decode separation
   - Requires Blackwell hardware for testing

## Build Configuration Updates

### CMakeLists.txt Changes for CUDA 13.1

```cmake
# Detect CUDA 13.1+
if ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "13.1")
    message("CUDA_VERSION ${CUDA_VERSION} >= 13.1 - Enabling CUDA 13.1 features")

    # Add new SM targets
    list(APPEND SGL_KERNEL_CUDA_FLAGS
        "-gencode=arch=compute_100a,code=sm_100a"
        "-gencode=arch=compute_103a,code=sm_103a"
        "-gencode=arch=compute_120a,code=sm_120a"
    )

    # Enable faster compilation for development
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        list(APPEND SGL_KERNEL_CUDA_FLAGS "--Ofast-compile=1")
    endif()

    # Deterministic builds
    list(APPEND SGL_KERNEL_CUDA_FLAGS "--frandom-seed=sglang")
endif()
```

## Performance Impact Estimates

| Optimization | Estimated Improvement | Effort | Priority |
|--------------|----------------------|--------|----------|
| cuBLAS Autotune | 10-15% GEMM ops | Low | High |
| Grouped GEMM API | 5-10% MoE layers | Medium | High |
| NVRTC improvements | 20-30% JIT time | Low | Medium |
| Green contexts runtime | Code simplification | Low | Medium |
| CUTLASS 4.0 | 5-15% GEMM ops | Medium | Medium |
| FP32 emulation | Up to 3x for FP32 | Low | Context-dependent |
| cuTile adoption | Variable | High | Exploratory |

## Compatibility Notes

- CUDA 13.1 is **backward compatible** with existing CUDA 12.x code
- New features require **Blackwell (SM100+)** for full benefit
- **Grouped GEMM** requires compute capability 10.x or 11.0
- **cuTile Python** requires CUDA 13.1 runtime

## Testing Requirements

1. **Hardware**: Blackwell B100/B200/GB200 for SM100 optimizations
2. **Baseline**: Compare against CUDA 12.8/13.0 performance
3. **Workloads**:
   - Prefill latency (attention-bound)
   - Decode throughput (memory-bound)
   - MoE expert routing (grouped GEMM)
4. **Models**: DeepSeek-V3, Qwen, LLaMA with FP8/FP4 quantization

## References

- [NVIDIA CUDA 13.1 Powers Next-Gen GPU Programming](https://developer.nvidia.com/blog/nvidia-cuda-13-1-powers-next-gen-gpu-programming-with-nvidia-cuda-tile-and-performance-gains/)
- [Focus on Your Algorithm—NVIDIA CUDA Tile Handles the Hardware](https://developer.nvidia.com/blog/focus-on-your-algorithm-nvidia-cuda-tile-handles-the-hardware/)
- [Simplify GPU Programming with NVIDIA CUDA Tile in Python](https://developer.nvidia.com/blog/simplify-gpu-programming-with-nvidia-cuda-tile-in-python/)
- [Introducing Grouped GEMM APIs in cuBLAS](https://developer.nvidia.com/blog/introducing-grouped-gemm-apis-in-cublas-and-more-performance-updates)
- [cuTile Python Documentation](https://docs.nvidia.com/cuda/cutile-python/)
- [CUDA Green Contexts API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html)
- [CUTLASS 4.0 GitHub](https://github.com/NVIDIA/cutlass)
- [Introducing NVFP4 for Efficient Low-Precision Inference](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
