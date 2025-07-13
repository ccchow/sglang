# MeZO Accuracy Optimization

## Key Insight from Original MeZO

The original MeZO implementation (`MeZO/medium_models/src/trainer.py`) includes an important feature for classification tasks:

```python
def zo_forward(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
    model.eval()
    inputs = self._prepare_inputs(inputs)
    if self.args.optimize_acc:
        loss, logits = model(**inputs)
        preds = F.softmax(logits, dim=-1)
        acc = torch.sum(torch.argmax(preds, 1) == inputs['labels']) / len(preds)
        loss = -acc  # Negative accuracy for minimization
    else:
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
```

## When to Use Accuracy Optimization

1. **Classification Tasks**: Especially useful for tasks where accuracy is the primary metric
2. **Few-Shot Learning**: Can be more effective than loss optimization when data is limited
3. **Non-Differentiable Objectives**: MeZO can optimize any scalar objective, not just differentiable losses

## Implementation in SGLang

We've implemented accuracy optimization in `test_mezo_roberta_sst2_accuracy.py`:

```python
def compute_accuracy_objective(model, inputs, labels):
    """Compute accuracy-based objective (negative accuracy for minimization)."""
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Get predictions
        preds = torch.argmax(logits, dim=-1)
        
        # Compute accuracy
        correct = (preds == labels).float()
        accuracy = correct.mean()
        
        # Return negative accuracy so minimization maximizes accuracy
        return -accuracy
```

## Key Differences from Loss Optimization

1. **Objective**: Directly optimizes accuracy instead of cross-entropy loss
2. **Scale**: Accuracy is bounded [0, 1], while loss can be unbounded
3. **Gradient Signal**: Can be noisier but more aligned with evaluation metric
4. **Learning Rate**: Often requires different learning rate tuning

## Recommendations

1. **Batch Size**: Use larger batches for more stable accuracy estimates
2. **Learning Rate**: Start with higher learning rates (1e-4 to 1e-3) for accuracy optimization
3. **Steps**: Accuracy optimization typically requires more steps to converge
4. **Evaluation**: Monitor both accuracy and loss to understand training dynamics

## Trade-offs

### Advantages:
- Directly optimizes the metric we care about
- Can work better for imbalanced datasets
- Useful for non-differentiable objectives

### Disadvantages:
- Noisier gradient estimates
- Requires more forward passes for stable estimates
- May converge more slowly

## Integration with SGLang MeZO

To add accuracy optimization to the main MeZO trainer:

```python
class MeZOTrainer:
    def __init__(self, ..., optimize_accuracy=False):
        self.optimize_accuracy = optimize_accuracy
        
    def _compute_objective(self, model, batch):
        if self.optimize_accuracy:
            # Use accuracy objective
            with torch.no_grad():
                outputs = model(...)
                preds = torch.argmax(outputs.logits, dim=-1)
                accuracy = (preds == labels).float().mean()
                return -accuracy  # Negative for minimization
        else:
            # Use standard loss
            return self._forward_pass(batch)
```

This feature makes MeZO more versatile for different optimization objectives beyond standard loss minimization.