# Task 4: Improve Loss Calculation

## Status: In Progress

## Description
Enhance the loss calculation to handle full sequence predictions, support attention masks for padding tokens, and implement proper loss averaging over batches.

## Current Issues
- Loss is only calculated on the first token of the completion
- No support for attention masks to ignore padding tokens
- No handling of variable sequence lengths
- Loss not properly averaged over sequences

## Implementation Plan

### Sub-task 1: Full Sequence Loss Calculation
- Modify forward pass to generate full completion sequences
- Calculate cross-entropy loss over all completion tokens
- Handle teacher forcing during training

### Sub-task 2: Attention Mask Support
- Use attention masks from dataset to ignore padding tokens
- Implement proper loss masking for padded positions
- Ensure gradient computation only considers valid tokens

### Sub-task 3: Variable Sequence Support
- Dynamic handling of different sequence lengths in a batch
- Proper loss normalization across sequences
- Support for completion-only loss vs full sequence loss

## Technical Details

### Loss Computation Strategy
1. **Forward Pass Enhancement**:
   - Generate logits for full sequences, not just next token
   - Use teacher forcing with ground truth tokens
   
2. **Loss Masking**:
   ```python
   # Pseudo-code for masked loss
   loss = F.cross_entropy(logits.view(-1, vocab_size), 
                         targets.view(-1), 
                         reduction='none')
   masked_loss = loss * attention_mask.view(-1)
   final_loss = masked_loss.sum() / attention_mask.sum()
   ```

3. **MeZO-Specific Considerations**:
   - Ensure symmetric loss computation for +ε and -ε perturbations
   - Handle numerical stability with proper scaling

## Progress
- [ ] Implement full sequence forward pass
- [ ] Add attention mask support in loss calculation
- [ ] Support variable length sequences
- [ ] Add loss normalization options
- [ ] Test with different sequence lengths

## Testing Plan
- Verify loss values are reasonable (not NaN/Inf)
- Compare with standard fine-tuning loss values
- Test with sequences of different lengths
- Validate gradient flow through masked positions