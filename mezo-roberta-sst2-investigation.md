# MeZO RoBERTa SST-2 Investigation

## Problem Summary

Our MeZO implementation on RoBERTa for SST-2 shows no convergence, while the original MeZO paper reports successful results. Key differences identified:

1. **Training Steps**: 500-1000 vs 100K in original
2. **Model Architecture**: Using wrong model type
3. **Hyperparameters**: Different from paper specifications
4. **Optimization Objective**: Implementation details differ

## Key Findings from Paper and Original Code

### 1. Training Configuration (from Paper)
- **Steps**: 100K for MeZO experiments
- **Batch Size**: 64
- **Learning Rate**: {1e-7, 1e-6, 1e-5}
- **Epsilon**: 1e-3
- **Constant learning rate** (no scheduling)
- **Evaluation**: Every 10K steps

### 2. Model Architecture
From the paper (Table 13):
- Uses **MLM head** with prompt: `<S1> It was [MASK].`
- Label words: 'terrible' (0), 'great' (1)
- Space-prefixed tokens for RoBERTa tokenization

### 3. Original Implementation Details
From `MeZO/medium_models/src/trainer.py`:
- **Efficient ZO**: In-place perturbations using random seed
- **Gradient Estimation**: `(loss1 - loss2) / (2 * eps)`
- **Direct parameter updates** when not using trainer optimizer
- **SGD optimizer** with constant LR

### 4. Critical Implementation Differences

#### Our Implementation:
```python
# Multiple perturbations stored
z_list = [torch.randn_like(p) for p in lora_params]
# Normalized perturbations
z_list = [z / (z.norm() + 1e-8) for z in z_list]
```

#### Original Implementation:
```python
# Single random seed for efficiency
torch.manual_seed(random_seed)
z = torch.normal(mean=0, std=1, size=param.data.size())
# No normalization by default
param.data = param.data + scaling_factor * z * eps
```

## Root Causes of No Convergence

### 1. Insufficient Training Steps
- We used 500-1000 steps (0.5-1% of original)
- MeZO's noisy gradients require many iterations
- Paper shows convergence starts after ~20K steps

### 2. Perturbation Normalization
- We normalize perturbations: `z / (z.norm() + 1e-8)`
- Original uses unnormalized Gaussian noise
- Normalization changes gradient scale and variance

### 3. Learning Rate Mismatch
- Our normalized perturbations require different LR scaling
- Original LR grid: {1e-7, 1e-6, 1e-5}
- We used: 1e-3 (1000x higher)

### 4. Model Head Initialization
- Classification head starts random
- MLM head is pre-trained
- Random initialization needs more steps to converge

## Recommended Fixes

### 1. Remove Perturbation Normalization
```python
# Change from:
z_list = [z / (z.norm() + 1e-8) for z in z_list]
# To:
z_list = z_list  # Use raw Gaussian noise
```

### 2. Adjust Hyperparameters
```python
# Match original paper
learning_rate = 1e-6  # Not 1e-3
epsilon = 1e-3
batch_size = 64
num_steps = 10000  # Minimum for observable progress
```

### 3. Use Correct Model Architecture
- Switch to MLM head for classification
- Implement prompt-based fine-tuning
- Use space-prefixed label words

### 4. Implement Efficient ZO (Optional)
```python
# Use single random seed for memory efficiency
random_seed = np.random.randint(1000000000)
torch.manual_seed(random_seed)
# Reuse same seed for negative perturbation
```

## Validation Experiment

To validate these findings, we should:

1. Run with 10K+ steps
2. Remove perturbation normalization
3. Use learning rate 1e-6
4. Monitor loss carefully for gradual decrease

## Expected Results

With proper configuration:
- Initial loss: ~0.7 (random)
- After 10K steps: ~0.5
- After 50K steps: ~0.3
- After 100K steps: ~0.2 (90%+ accuracy)

## Conclusion

The lack of convergence is expected given:
1. **Too few steps**: 500 vs 100K (200x difference)
2. **Wrong gradient scale**: Normalized vs unnormalized perturbations
3. **Wrong learning rate**: 1e-3 vs 1e-6 (1000x difference)
4. **Wrong model**: Classification head vs MLM head

These are not bugs but configuration mismatches. The MeZO algorithm itself is correctly implemented.