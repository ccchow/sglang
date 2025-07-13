# Task 5: Optimize KV Cache Using Perturbation Symmetry

## Status: In Progress

## Description
Leverage MeZO's fixed two-way perturbation characteristic to optimize KV cache usage and implement efficient in-memory weight updates.

## Background
MeZO uses symmetric perturbations (+εz and -εz) along a fixed direction z. This symmetry can be exploited for:
1. KV cache reuse between the two forward passes
2. More efficient weight perturbation operations
3. Reduced memory bandwidth requirements

## Implementation Strategy

### Sub-task 1: Shared Prefix KV Caching
- Identify common computation between +εz and -εz perturbations
- Implement prefix sharing using SGLang's RadixAttention
- Cache KV states that are invariant to small perturbations

### Sub-task 2: In-place Weight Updates
- Optimize perturbation application as scalar-scaled additions
- Minimize memory allocations during perturbation
- Implement efficient restoration of original weights

### Sub-task 3: Perturbation Analysis
- Profile cache hit rates for different epsilon values
- Tune epsilon for optimal cache reuse (target >80% hit rate)
- Implement adaptive epsilon scheduling

## Technical Details

### KV Cache Optimization
```python
# Pseudo-code for optimized forward passes
def optimized_mezo_step():
    # Compute shared prefix once
    shared_kv = compute_shared_prefix(batch, original_weights)
    
    # Reuse for both perturbations
    loss_plus = forward_with_cache(batch, weights + ε*z, shared_kv)
    loss_minus = forward_with_cache(batch, weights - ε*z, shared_kv)
```

### Memory-Efficient Perturbations
```python
# In-place perturbation
weights.add_(epsilon * z)  # +εz
loss_plus = forward()
weights.add_(-2 * epsilon * z)  # -εz (from +εz to -εz)
loss_minus = forward()
weights.add_(epsilon * z)  # Restore original
```

## Challenges
1. SGLang's KV cache is optimized for inference, not training perturbations
2. Weight perturbations may invalidate cached activations
3. Need to balance cache reuse with perturbation magnitude

## Progress
- [ ] Analyze SGLang's RadixAttention for perturbation caching
- [ ] Implement shared prefix computation
- [ ] Add in-place weight perturbation operations
- [ ] Profile and optimize cache hit rates
- [ ] Benchmark memory and compute savings

## Expected Benefits
- 30-50% reduction in forward pass computation
- Reduced memory bandwidth usage
- Faster training iterations