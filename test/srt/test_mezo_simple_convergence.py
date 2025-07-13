#!/usr/bin/env python3
"""
Simple convergence test for MeZO algorithm using a synthetic task.
This test verifies the core MeZO algorithm can minimize loss without full SGLang infrastructure.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


class SimpleModel(nn.Module):
    """A simple 2-layer neural network for testing."""
    def __init__(self, input_dim: int = 10, hidden_dim: int = 20, output_dim: int = 5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class SimpleLoRALayer(nn.Module):
    """Simple LoRA adapter for testing."""
    def __init__(self, in_features: int, out_features: int, rank: int = 4):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = 1.0
        
    def forward(self, x):
        return (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


def create_synthetic_data(n_samples: int = 100, input_dim: int = 10, output_dim: int = 5):
    """Create synthetic classification data."""
    # Create random inputs
    X = torch.randn(n_samples, input_dim)
    
    # Create random linear transformation for ground truth
    W_true = torch.randn(output_dim, input_dim) * 0.5
    
    # Generate labels
    logits = X @ W_true.T
    y = torch.argmax(logits + torch.randn_like(logits) * 0.1, dim=1)
    
    return X, y


def mezo_step(model, lora_layer, X, y, epsilon=1e-3):
    """Perform one MeZO step."""
    # Get LoRA parameters
    params = [lora_layer.lora_A, lora_layer.lora_B]
    
    # Sample random perturbation
    z_list = [torch.randn_like(p) for p in params]
    
    # Forward pass with +εz
    for i, p in enumerate(params):
        p.data.add_(epsilon * z_list[i])
    
    with torch.no_grad():
        output = model(X) + lora_layer(X)
        loss_plus = nn.functional.cross_entropy(output, y)
    
    # Forward pass with -εz (from +εz to -εz)
    for i, p in enumerate(params):
        p.data.add_(-2 * epsilon * z_list[i])
    
    with torch.no_grad():
        output = model(X) + lora_layer(X)
        loss_minus = nn.functional.cross_entropy(output, y)
    
    # Restore original parameters
    for i, p in enumerate(params):
        p.data.add_(epsilon * z_list[i])
    
    # Compute gradient estimate
    grad_estimate = (loss_plus - loss_minus) / (2 * epsilon)
    
    # Update parameters
    learning_rate = 0.1  # Increased learning rate for better convergence
    for i, p in enumerate(params):
        p.data.add_(-learning_rate * grad_estimate * z_list[i])
    
    return (loss_plus + loss_minus) / 2


def test_mezo_convergence():
    """Test MeZO convergence on a simple task."""
    print("=" * 60)
    print("MeZO Simple Convergence Test")
    print("=" * 60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create model and data
    input_dim, hidden_dim, output_dim = 10, 20, 5
    model = SimpleModel(input_dim, hidden_dim, output_dim)
    lora_layer = SimpleLoRALayer(input_dim, output_dim, rank=4)
    
    # Freeze base model
    for param in model.parameters():
        param.requires_grad = False
    
    # Create data
    X_train, y_train = create_synthetic_data(200, input_dim, output_dim)
    X_test, y_test = create_synthetic_data(50, input_dim, output_dim)
    
    print(f"\nModel: {input_dim} -> {hidden_dim} -> {output_dim}")
    print(f"LoRA rank: 4")
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Training parameters
    num_steps = 200
    epsilon = 1e-3
    eval_interval = 10
    
    # Track losses
    train_losses = []
    test_losses = []
    test_accuracies = []
    steps = []
    
    print("\nStarting MeZO training...")
    print("-" * 40)
    print("Step | Train Loss | Test Loss | Test Acc")
    print("-" * 40)
    
    for step in range(num_steps):
        # Training step
        train_loss = mezo_step(model, lora_layer, X_train, y_train, epsilon)
        train_losses.append(train_loss.item())
        
        # Evaluation
        if step % eval_interval == 0:
            with torch.no_grad():
                # Test loss
                test_output = model(X_test) + lora_layer(X_test)
                test_loss = nn.functional.cross_entropy(test_output, y_test)
                
                # Test accuracy
                predictions = torch.argmax(test_output, dim=1)
                accuracy = (predictions == y_test).float().mean()
                
                test_losses.append(test_loss.item())
                test_accuracies.append(accuracy.item())
                steps.append(step)
                
                print(f"{step:4d} | {train_loss:.6f} | {test_loss:.6f} | {accuracy:.2%}")
    
    print("-" * 40)
    
    # Analyze convergence
    print("\nConvergence Analysis:")
    
    # Check loss reduction
    initial_train_loss = np.mean(train_losses[:10])
    final_train_loss = np.mean(train_losses[-10:])
    train_reduction = (initial_train_loss - final_train_loss) / initial_train_loss * 100
    
    initial_test_loss = test_losses[0]
    final_test_loss = test_losses[-1]
    test_reduction = (initial_test_loss - final_test_loss) / initial_test_loss * 100
    
    initial_accuracy = test_accuracies[0]
    final_accuracy = test_accuracies[-1]
    accuracy_improvement = final_accuracy - initial_accuracy
    
    print(f"  Initial train loss: {initial_train_loss:.4f}")
    print(f"  Final train loss: {final_train_loss:.4f}")
    print(f"  Train loss reduction: {train_reduction:.1f}%")
    print(f"  Initial test loss: {initial_test_loss:.4f}")
    print(f"  Final test loss: {final_test_loss:.4f}")
    print(f"  Test loss reduction: {test_reduction:.1f}%")
    print(f"  Initial accuracy: {initial_accuracy:.1%}")
    print(f"  Final accuracy: {final_accuracy:.1%}")
    print(f"  Accuracy improvement: {accuracy_improvement:.1%}")
    
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Training loss
    ax = axes[0]
    ax.plot(train_losses, 'b-', alpha=0.6)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(range(len(train_losses)), train_losses, 1)
    p = np.poly1d(z)
    ax.plot(range(len(train_losses)), p(range(len(train_losses))), "r--", 
            alpha=0.8, label=f'Trend (slope={z[0]:.4f})')
    ax.legend()
    
    # Test loss
    ax = axes[1]
    ax.plot(steps, test_losses, 'g-', marker='o')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Test Loss')
    ax.grid(True, alpha=0.3)
    
    # Test accuracy
    ax = axes[2]
    ax.plot(steps, test_accuracies, 'r-', marker='s')
    ax.set_xlabel('Step')
    ax.set_ylabel('Accuracy')
    ax.set_title('Test Accuracy')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('mezo_simple_convergence.png', dpi=150)
    print(f"\nPlots saved to: mezo_simple_convergence.png")
    
    # Compare with SGD
    print("\nComparing with SGD baseline...")
    
    # Reset LoRA parameters
    torch.nn.init.normal_(lora_layer.lora_A, 0, 0.01)
    torch.nn.init.zeros_(lora_layer.lora_B)
    
    # SGD training
    sgd_optimizer = torch.optim.SGD([lora_layer.lora_A, lora_layer.lora_B], lr=0.01)
    sgd_losses = []
    
    for step in range(num_steps):
        sgd_optimizer.zero_grad()
        output = model(X_train) + lora_layer(X_train)
        loss = nn.functional.cross_entropy(output, y_train)
        loss.backward()
        sgd_optimizer.step()
        sgd_losses.append(loss.item())
    
    # Final SGD performance
    with torch.no_grad():
        test_output = model(X_test) + lora_layer(X_test)
        sgd_test_loss = nn.functional.cross_entropy(test_output, y_test)
        sgd_accuracy = (torch.argmax(test_output, dim=1) == y_test).float().mean()
    
    print(f"  SGD final test loss: {sgd_test_loss:.4f}")
    print(f"  SGD final accuracy: {sgd_accuracy:.1%}")
    print(f"  MeZO vs SGD accuracy: {final_accuracy:.1%} vs {sgd_accuracy:.1%}")
    
    # Determine success
    converged = (train_reduction > 20 and test_reduction > 10 and final_accuracy > 0.3)
    
    print("\n" + "=" * 60)
    print(f"CONVERGENCE TEST: {'PASSED' if converged else 'FAILED'}")
    print("=" * 60)
    
    if converged:
        print("✓ MeZO successfully converged!")
        print(f"  - Training loss reduced by {train_reduction:.1f}%")
        print(f"  - Test loss reduced by {test_reduction:.1f}%")
        print(f"  - Accuracy improved by {accuracy_improvement:.1%}")
        print(f"  - Final accuracy: {final_accuracy:.1%}")
    else:
        print("✗ MeZO convergence needs tuning")
    
    return converged


if __name__ == "__main__":
    converged = test_mezo_convergence()
    
    if converged:
        print("\n✅ Simple MeZO convergence test passed!")
    else:
        print("\n❌ Simple MeZO convergence test needs investigation")