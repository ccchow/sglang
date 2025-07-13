# Task 6: Create Unit Tests for MeZO Trainer

## Status: In Progress

## Description
Create comprehensive unit and integration tests for the MeZO trainer implementation to ensure correctness, efficiency, and robustness.

## Test Categories

### 1. Gradient Estimation Tests
- Verify MeZO gradient estimation accuracy with toy models
- Compare with finite difference approximations
- Test gradient magnitude and direction

### 2. KV Cache Optimization Tests
- Measure cache efficiency with different epsilon values
- Verify in-place perturbation correctness
- Test memory usage reduction

### 3. Dataset Handling Tests
- Test MeZODataset with various input formats
- Verify tokenization and padding
- Test distributed data loading

### 4. Integration Tests
- End-to-end training on small models
- Convergence tests on simple tasks
- Memory and performance benchmarks

### 5. Edge Case Tests
- Handle empty batches
- Test with very small/large epsilon values
- OOM handling with quantization

## Test Implementation

### Test File Structure
```
test/srt/test_mezo_trainer.py
├── TestMeZOGradientEstimation
├── TestMeZODataset
├── TestKVCacheOptimization
├── TestMeZOIntegration
└── TestMeZOEdgeCases
```

### Key Test Cases

1. **Gradient Correctness**:
   ```python
   def test_gradient_estimation_linear_model():
       # Use simple linear model for exact gradient comparison
       # Verify MeZO gradient approximation accuracy
   ```

2. **Dataset Loading**:
   ```python
   def test_dataset_formats():
       # Test JSONL, JSON, HF datasets, list formats
       # Verify proper tokenization and batching
   ```

3. **Memory Efficiency**:
   ```python
   def test_memory_usage_optimized_vs_original():
       # Compare memory usage between methods
       # Verify in-place operations don't corrupt weights
   ```

## Progress
- [ ] Create test file structure
- [ ] Implement gradient estimation tests
- [ ] Add dataset handling tests
- [ ] Create KV cache optimization tests
- [ ] Add integration tests
- [ ] Implement edge case tests

## Testing Strategy
- Use pytest framework (following SGLang conventions)
- Mock heavy components (model loading) for unit tests
- Use small models for integration tests
- Measure performance metrics in benchmarks