# MeZO MLM Analysis

## Key Discovery

The original MeZO implementation for RoBERTa on SST-2 uses **Masked Language Modeling (MLM)** for classification, not the standard sequence classification head. This is a fundamentally different approach:

### Standard Approach (RobertaForSequenceClassification)
- Uses a classification head on top of [CLS] token
- Outputs: 2 logits for binary classification
- Loss: Cross-entropy over 2 classes

### MeZO's MLM Approach (RobertaForMaskedLM)
- Uses the MLM head to predict words at [MASK] position
- Prompt template: `"{sentence} It was [MASK]."`
- Label mapping:
  - Negative (0) → "terrible"
  - Positive (1) → "great"
- Loss: Cross-entropy over vocabulary, but only considering label words

## Why MLM for Classification?

1. **Leverages Pre-trained Knowledge**: The MLM head is pre-trained and already knows word meanings
2. **Natural Language Interface**: Maps labels to meaningful words
3. **Better Initialization**: Unlike randomly initialized classification heads
4. **Prompt-based Fine-tuning**: Aligns with modern prompt-based methods

## Implementation Details

### Original MeZO (from modeling_roberta.py)
```python
class RobertaModelForPromptFinetuning(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)  # MLM head
```

### Key Functions
- `model_for_prompting_forward`: Handles prompt templates and label word predictions
- Uses prompt template with special tokens: `*cls**sent_0*_It_was*mask*.*sep+*`
- Extracts logits at [MASK] position
- Computes loss only over label words

## Our Implementation

Created `test_mezo_roberta_mlm.py` that:
1. Uses `RobertaForMaskedLM` instead of `RobertaForSequenceClassification`
2. Applies prompt template: `"{sentence} It was [MASK]."`
3. Maps predictions at [MASK] to label words
4. Optimizes accuracy of label word predictions
5. Adds LoRA to both attention and FFN layers (following original)

## Expected Benefits

1. **Faster Convergence**: Pre-trained MLM head vs random classification head
2. **Better Performance**: Leverages language understanding
3. **More Stable**: MLM head has meaningful initialization

## Next Steps

1. Run the MLM-based test to see if it converges better
2. Consider using roberta-large (as in original MeZO)
3. Tune hyperparameters based on MLM setup
4. Compare with standard classification approach