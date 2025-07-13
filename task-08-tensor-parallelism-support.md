# Task 08: Support for Tensor Parallelism in MeZO

## Overview

This task implements tensor parallelism (TP) support for MeZO training in SGLang. The implementation ensures that:
1. LoRA adapters are properly sharded across TP ranks
2. The perturbation direction z is synchronized across all TP ranks
3. Loss aggregation is handled correctly

## Key Components

### 1. TP-Aware LoRA Parameter Collection

When collecting LoRA parameters in a TP setup, we need to ensure each rank only updates its shard of the parameters.

### 2. Perturbation Synchronization

The random perturbation z must be identical across all TP ranks to ensure consistent gradient estimation. This is achieved by:
- Broadcasting the random seed from rank 0
- Using the same seed to generate perturbations on all ranks

### 3. Loss Aggregation

Since each TP rank computes loss on its shard of the model, we need to aggregate losses across ranks before computing gradients.

## Implementation

The implementation modifies `mezo_trainer.py` to:
1. Detect tensor parallelism configuration
2. Synchronize perturbations across TP ranks
3. Aggregate losses before gradient computation
4. Handle LoRA weight updates correctly for sharded parameters

## Testing

Test with:
```bash
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-hf \
    --port 30000 --tensor-parallel-size 2

python examples/runtime/mezo_example.py --tensor-parallel
```

## Benefits

- Enables MeZO training on larger models that require tensor parallelism
- Maintains MeZO's memory efficiency advantages
- Minimal communication overhead (only loss scalars are aggregated)