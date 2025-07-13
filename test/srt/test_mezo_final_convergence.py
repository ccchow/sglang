#!/usr/bin/env python3
"""
Final MeZO convergence test with careful hyperparameter tuning and gradient clipping.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class ToyModel(nn.Module):
    """Very simple model for convergence testing."""
    def __init__(self, input_dim: int = 10, output_dim: int = 2):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        # Initialize with small weights
        nn.init.normal_(self.linear.weight, 0, 0.01)
        
    def forward(self, x):
        return self.linear(x)


class ToyLoRA(nn.Module):
    """Simple LoRA for the toy model."""
    def __init__(self, input_dim: int = 10, output_dim: int = 2, rank: int = 2):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(rank, input_dim) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(output_dim, rank))
        
    def forward(self, x):
        return x @ self.lora_A.T @ self.lora_B.T


def create_simple_dataset(n_samples: int = 100):
    """Create a very simple binary classification dataset."""
    # Create linearly separable data
    X = torch.randn(n_samples, 10)
    # Simple rule: if sum of first 3 features > 0, class 1, else class 0
    y = (X[:, :3].sum(dim=1) > 0).long()
    return X, y


def mezo_update(model, lora, X, y, epsilon=1e-3, lr=0.01, momentum=0.9, velocity=None):
    """MeZO update with momentum and gradient clipping."""
    params = [lora.lora_A, lora.lora_B]
    
    # Initialize velocity if needed
    if velocity is None:
        velocity = [torch.zeros_like(p) for p in params]
    
    # Sample perturbation
    z_list = [torch.randn_like(p) for p in params]
    
    # Normalize perturbations
    z_list = [z / (z.norm() + 1e-8) for z in z_list]
    
    # Forward with +epsilon
    for i, p in enumerate(params):
        p.data.add_(epsilon * z_list[i])
    
    with torch.no_grad():
        output = model(X) + lora(X)
        loss_plus = nn.functional.cross_entropy(output, y)
    
    # Forward with -epsilon
    for i, p in enumerate(params):
        p.data.add_(-2 * epsilon * z_list[i])
    
    with torch.no_grad():
        output = model(X) + lora(X)
        loss_minus = nn.functional.cross_entropy(output, y)
    
    # Restore parameters
    for i, p in enumerate(params):
        p.data.add_(epsilon * z_list[i])
    
    # Compute gradient estimate
    grad_estimate = (loss_plus - loss_minus) / (2 * epsilon)
    
    # Clip gradient estimate
    grad_estimate = torch.clamp(grad_estimate, -1.0, 1.0)
    
    # Update with momentum
    for i, p in enumerate(params):
        velocity[i] = momentum * velocity[i] - lr * grad_estimate * z_list[i]
        p.data.add_(velocity[i])
    
    return (loss_plus + loss_minus) / 2, velocity


def test_convergence():
    """Test MeZO convergence with careful setup."""
    print("=" * 60)
    print("MeZO Final Convergence Test")
    print("=" * 60)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create simple model and data
    model = ToyModel(10, 2)
    lora = ToyLoRA(10, 2, rank=2)
    
    # Freeze base model
    for p in model.parameters():
        p.requires_grad = False
    
    # Create data
    X_train, y_train = create_simple_dataset(200)
    X_test, y_test = create_simple_dataset(50)
    
    print(f"\nSetup:")
    print(f"  Model: Linear(10 -> 2)")
    print(f"  LoRA rank: 2")
    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    
    # Hyperparameters (carefully tuned)
    num_steps = 500
    epsilon = 1e-3
    learning_rate = 0.01
    momentum = 0.9
    
    print(f"\nHyperparameters:")
    print(f"  Steps: {num_steps}")
    print(f"  Epsilon: {epsilon}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Momentum: {momentum}")
    
    # Training
    train_losses = []
    test_losses = []
    test_accs = []
    velocity = None
    
    print("\nTraining...")
    print("-" * 40)
    print("Step | Train Loss | Test Loss | Test Acc")
    print("-" * 40)
    
    for step in range(num_steps):
        # Train step
        train_loss, velocity = mezo_update(
            model, lora, X_train, y_train, 
            epsilon, learning_rate, momentum, velocity
        )
        train_losses.append(train_loss.item())
        
        # Evaluate every 50 steps
        if step % 50 == 0:
            with torch.no_grad():
                test_output = model(X_test) + lora(X_test)
                test_loss = nn.functional.cross_entropy(test_output, y_test)
                test_acc = (test_output.argmax(1) == y_test).float().mean()
                
                test_losses.append(test_loss.item())
                test_accs.append(test_acc.item())
                
                print(f"{step:4d} | {train_loss:.6f} | {test_loss:.6f} | {test_acc:.2%}")
    
    print("-" * 40)
    
    # Analysis
    print("\nResults:")
    initial_loss = np.mean(train_losses[:10])
    final_loss = np.mean(train_losses[-10:])
    improvement = (initial_loss - final_loss) / initial_loss * 100
    
    print(f"  Initial train loss: {initial_loss:.4f}")
    print(f"  Final train loss: {final_loss:.4f}")
    print(f"  Improvement: {improvement:.1f}%")
    print(f"  Final test accuracy: {test_accs[-1]:.1%}")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Training loss
    ax1.plot(train_losses, 'b-', alpha=0.7)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    
    # Smooth the curve
    window = 20
    if len(train_losses) > window:
        smoothed = np.convolve(train_losses, np.ones(window)/window, mode='valid')
        ax1.plot(range(window//2, len(train_losses)-window//2+1), smoothed, 'r-', 
                linewidth=2, label='Smoothed')
        ax1.legend()
    
    # Test accuracy
    steps = list(range(0, num_steps, 50))
    ax2.plot(steps, test_accs, 'g-', marker='o', markersize=8)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Test Accuracy')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('mezo_final_convergence.png', dpi=150)
    print(f"\nPlot saved to: mezo_final_convergence.png")
    
    # Success criteria
    converged = improvement > 5 and test_accs[-1] > 0.6
    
    print("\n" + "=" * 60)
    if converged:
        print("✅ MeZO CONVERGENCE TEST: PASSED")
        print("=" * 60)
        print(f"MeZO successfully optimized LoRA parameters:")
        print(f"  - Loss decreased by {improvement:.1f}%")
        print(f"  - Achieved {test_accs[-1]:.1%} test accuracy")
    else:
        print("❌ MeZO CONVERGENCE TEST: NEEDS TUNING")
        print("=" * 60)
        print("Consider adjusting hyperparameters further")
    
    return converged


if __name__ == "__main__":
    converged = test_convergence()