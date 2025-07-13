# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

SGLang is a fast serving framework for large language models (LLMs) and vision language models (VLMs). It co-designs the backend runtime and frontend language for better performance and control. The project consists of several main components:

1. **SGLang Runtime (`python/sglang/`)** - The core Python package
2. **SGLang Kernel (`sgl-kernel/`)** - High-performance CUDA/ROCm kernels
3. **SGLang Router (`sgl-router/`)** - Rust-based load balancer for data parallelism
4. **Documentation (`docs/`)** - Jupyter notebooks and markdown documentation

## Key Commands

### Build & Installation

```bash
# Install from source (recommended for development)
cd python
pip install -e .

# For development with all dependencies
pip install -e ".[dev]"

# Build sgl-kernel separately if needed
cd sgl-kernel
python setup.py install
```

### Code Quality & Linting

```bash
# Install and run pre-commit hooks (REQUIRED before committing)
pip install pre-commit
pre-commit install
pre-commit run --all-files

# Format Python files using Makefile
make format

# The pre-commit config includes:
# - isort (import sorting)
# - black (Python formatting)
# - ruff (linting)
# - codespell (spelling)
# - clang-format (C++/CUDA)
# - nbstripout (Jupyter notebooks)
```

### Testing

```bash
# Run backend runtime tests
cd test/srt
python run_suite.py --suite per-commit

# Run frontend language tests
cd test/lang
python run_suite.py --suite per-commit

# Run a single test file
python test_srt_endpoint.py

# Run a specific test method
python -m unittest test_srt_endpoint.TestSRTEndpoint.test_simple_decode
```

### Documentation

```bash
cd docs
pip install -r requirements.txt

# Compile notebooks
make compile

# Serve documentation locally
bash serve.sh
# or
make serve
```

## Architecture Overview

### Core Components

1. **Frontend Language (`python/sglang/lang/`)**
   - Provides intuitive APIs for LLM programming
   - Supports chained generation calls, control flow, multi-modal inputs
   - Key files: `api.py`, `interpreter.py`, `compiler.py`, `backend/`

2. **Backend Runtime (`python/sglang/srt/`)**
   - High-performance serving engine
   - Features: RadixAttention, continuous batching, tensor/pipeline/expert parallelism
   - Key modules:
     - `models/` - Model implementations (Llama, Mistral, etc.)
     - `layers/` - Core layers (attention, linear, etc.)
     - `managers/` - Scheduling, caching, tokenization
     - `mem_cache/` - Memory and cache management

3. **SGLang Kernel (`sgl-kernel/`)**
   - Custom CUDA/ROCm kernels for performance
   - Includes: MoE kernels, quantization, attention kernels
   - Built as a separate package `sgl-kernel`

4. **Router (`sgl-router/`)**
   - Rust-based load balancer with Python bindings
   - Supports cache-aware routing and service discovery
   - Build with: `cd sgl-router && python -m build`

### Model Support

Models are implemented in `python/sglang/srt/models/`. Each model file typically includes:
- Model configuration loading
- Architecture definition
- Custom attention/layer implementations
- Weight loading logic

### Key Design Patterns

1. **Lazy imports** - Used throughout to reduce startup time
2. **Modular backends** - Support for different inference backends (CUDA, ROCm, CPU)
3. **Plugin architecture** - Easy to add new models, layers, and features
4. **Async processing** - Extensive use of asyncio for concurrent request handling

## Development Workflow

1. **Before making changes**: Fork the repository (contributors don't have write access)
2. **Make changes**: Create a feature branch, never commit to main
3. **Test locally**: Run relevant test suites. Create temp test scripts and md files under claude/
4. **Format code**: Run `pre-commit run --all-files`
5. **Update docs**: If adding features, update relevant documentation
6. **Create PR**: Push to your fork and open a PR

## Important Notes

- The project uses unittest for testing, not pytest
- Documentation prefers Jupyter notebooks over plain markdown
- When working on models, check existing implementations for patterns
- The router is a separate Rust project with Python bindings
- CI runs extensive tests including accuracy evaluations

## Environment Variables

Key environment variables that affect behavior:
- `SGLANG_*` - Various SGLang-specific settings
- Check `python/sglang/srt/utils.py` and `docs/references/environment_variables.md` for full list

## Debugging Tips

- Use `--log-level debug` when launching servers
- Check `python/sglang/srt/debug_utils.py` for debugging utilities
- Structured logging is available via `--log-dir` option
- Prometheus metrics available for monitoring