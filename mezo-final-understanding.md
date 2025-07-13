# MeZO Final Understanding - MLM vs Accuracy Objective

## The Key Discovery

The MeZO paper does NOT use accuracy as the objective for SST-2. Instead, it uses:

1. **Masked Language Modeling (MLM)** with RoBERTa's MLM head
2. **Prompt template**: `"[TEXT] It was [MASK]."`
3. **Label mapping**: `{0: 'terrible', 1: 'great'}`
4. **Loss function**: Cross-entropy on vocabulary logits at mask position

## Why This Works

### MLM Approach (Paper's Method)
- **Continuous gradients**: Every perturbation produces a gradient
- **Average gradient magnitude**: ~25 in our tests
- **Non-zero gradients**: 100% of the time
- **Loss function**: Cross-entropy (differentiable)

### Accuracy Approach (What We Implemented)
- **Discrete gradients**: Only when predictions flip
- **Average gradient magnitude**: 0 in our tests  
- **Non-zero gradients**: 0% with small epsilon
- **Loss function**: Accuracy (non-differentiable)

## Implementation Details

The paper's approach:
```python
# Template: "[sentence] It was [MASK]."
# Map labels to vocabulary tokens
label_words = {0: 'terrible', 1: 'great'}

# Get MLM logits at mask position
mlm_logits = model(**inputs).logits[batch_idx, mask_pos]

# Extract logits for label words
label_logits = mlm_logits[:, [terrible_id, great_id]]

# Compute cross-entropy loss
loss = F.cross_entropy(label_logits, labels)
```

This is fundamentally different from classification with accuracy objective!

## SGLang Implementation Status

Our SGLang implementation is correct for the MeZO algorithm itself, but we implemented the wrong objective function for reproducing the paper's results:

✅ **Correct**: MeZO gradient estimation algorithm (2 forward passes)
✅ **Correct**: LoRA integration and tensor parallelism
✅ **Correct**: RadixAttention optimization
❌ **Wrong for paper reproduction**: Used classification head with accuracy objective
✅ **Can be fixed**: Need to add MLM-based loss computation

## Recommendations

1. **To reproduce paper results**: Implement MLM-based loss with label word mapping
2. **For practical use**: The current implementation with cross-entropy loss on classification head works well
3. **For research**: Both approaches are valid - MLM for continuous gradients, accuracy for true discrete optimization

## Conclusion

The MeZO paper cleverly converts classification tasks into language modeling tasks to obtain continuous gradients. This explains why it converges in 100K steps - it's optimizing a smooth loss function, not a discrete accuracy metric. Our implementation of the core MeZO algorithm is correct, but we need to use MLM-based loss to exactly reproduce the paper's SST-2 results.