# Task 7: Create Example Script

## Status: In Progress

## Description
Create a comprehensive example script demonstrating how to use the MeZO trainer for LoRA fine-tuning with SGLang.

## Example Script Features

### 1. Basic Usage Example
- Simple fine-tuning on a small model
- Clear documentation and comments
- Minimal dependencies

### 2. Advanced Features
- Dataset loading from various sources
- Quantization support demonstration
- Distributed training setup
- Hyperparameter tuning

### 3. Benchmarking Tools
- Performance comparison with standard fine-tuning
- Memory usage monitoring
- Training progress visualization

## Script Structure

```
examples/runtime/mezo_example.py
├── Basic MeZO fine-tuning
├── Dataset preparation examples
├── Quantized model training
├── Distributed training setup
└── Performance benchmarking
```

## Key Demonstrations

1. **Simple Fine-tuning**:
   - Load a small model (e.g., OPT-125M)
   - Fine-tune on a sample dataset
   - Save and load LoRA weights

2. **Dataset Formats**:
   - JSONL file loading
   - Hugging Face datasets
   - In-memory examples

3. **Advanced Options**:
   - Bitsandbytes quantization
   - Custom epsilon values
   - Batch size tuning

## Documentation
- Inline comments explaining each step
- Expected outputs and benchmarks
- Troubleshooting tips

## Progress
- [ ] Create basic example script
- [ ] Add dataset loading examples
- [ ] Include quantization demo
- [ ] Add performance benchmarking
- [ ] Write comprehensive documentation