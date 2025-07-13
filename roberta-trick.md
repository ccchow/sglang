# The RoBERTa MLM Trick for MeZO

## The Problem

When using MeZO (zeroth-order optimization) with classification tasks:
- **Accuracy objective is discrete**: Only changes when predictions flip
- **Result**: 100% zero gradients with small perturbations
- **Cannot optimize**: No learning signal

## The Clever Solution

The MeZO paper converts classification into a **language modeling task** to get continuous gradients.

### How It Works

1. **Use RoBERTa's MLM head** instead of classification head
2. **Add a prompt template** with mask token:
   ```
   Original: "This movie is terrible"
   Template: "This movie is terrible. It was [MASK]."
   ```

3. **Map labels to vocabulary words**:
   ```python
   label_words = {
       0: 'terrible',  # negative sentiment
       1: 'great'      # positive sentiment
   }
   ```
   
   **Critical detail**: The implementation automatically adds space prefix for RoBERTa:
   ```python
   # Configuration mapping (without spaces)
   mapping = {'0': 'terrible', '1': 'great'}
   
   # MeZO automatically adds space for RoBERTa/BART/T5
   if label_word[0] not in ['<', '[', '.', ',']:
       # Converts 'terrible' → ' terrible' internally
       token_id = tokenizer.convert_tokens_to_ids(' ' + label_word)
   ```
   
   This ensures proper tokenization as single vocabulary tokens, not subwords.

4. **Optimize cross-entropy on vocabulary logits**:
   ```python
   # Get logits at [MASK] position
   mask_logits = model(inputs).logits[batch, mask_pos]
   
   # Extract logits for label words only
   label_logits = mask_logits[:, [terrible_id, great_id]]
   
   # Compute cross-entropy loss (continuous!)
   loss = F.cross_entropy(label_logits, labels)
   ```

## Why This Works

| Aspect | Classification Head | MLM Trick |
|--------|-------------------|-----------|
| Objective | Accuracy (discrete) | Cross-entropy (continuous) |
| Gradient | 0% non-zero | 100% non-zero |
| Optimization | Impossible | Smooth gradient descent |
| Convergence | Never | 100K steps |

## Results

- **Without trick**: 0% non-zero gradients, no learning
- **With trick**: 100% non-zero gradients, steady optimization
- **Average gradient**: ~5 (vs 0 without trick)
- **Loss improvement**: -0.035 in just 1K steps

## Implementation Tips

1. **Choose semantically meaningful label words**:
   - SST-2: 'terrible' vs 'great'
   - MNLI: 'Yes', 'No', 'Maybe'
   - RTE: 'Yes' vs 'No'

2. **Use space-prefixed tokens for tokenization**:
   ```python
   # Wrong: might tokenize as subwords
   tokenizer.convert_tokens_to_ids('terrible')  # Could be 'ter' + 'rible'
   
   # Correct: ensures single token
   tokenizer.convert_tokens_to_ids(' terrible')  # Single token with space prefix
   ```

3. **Design appropriate templates**:
   - Sentiment: "It was [MASK]."
   - Entailment: "? [MASK],"
   - Questions: "[MASK]:"

4. **Use the model's existing vocabulary**:
   - No new tokens needed
   - Leverages pre-trained knowledge

## Conclusion

This trick transforms an impossible discrete optimization problem into a smooth continuous one by leveraging the model's language modeling capabilities. It's a brilliant example of reformulating the problem to match the optimization method's requirements.