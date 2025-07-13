#!/usr/bin/env python3
"""
Stable convergence test for MeZO with batched processing and better hyperparameters.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader


class SimpleModel(nn.Module):
    """Simple model for testing - mimics a small transformer."""
    def __init__(self, vocab_size: int = 100, hidden_dim: int = 64, num_classes: int = 5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        x = self.embedding(x)  # (batch_size, seq_len, hidden_dim)
        x = x.mean(dim=1)  # Simple pooling: (batch_size, hidden_dim)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class LoRAAdapter(nn.Module):
    """LoRA adapter for the final layer."""
    def __init__(self, hidden_dim: int = 64, num_classes: int = 5, rank: int = 8):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(rank, hidden_dim) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(num_classes, rank))
        self.scaling = 1.0  # Increased scaling for stronger LoRA contribution
        
    def forward(self, x):
        # x is the input to fc2, shape: (batch_size, hidden_dim)
        return (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


def create_classification_dataset(n_samples: int = 500, seq_len: int = 10, vocab_size: int = 100, num_classes: int = 5):
    """Create a simple classification dataset with learnable patterns."""
    X = torch.zeros((n_samples, seq_len), dtype=torch.long)
    y = torch.zeros(n_samples, dtype=torch.long)
    
    for i in range(n_samples):
        # Create class-specific patterns
        label = i % num_classes
        y[i] = label
        
        # Each class has a distinctive pattern in the first token
        X[i, 0] = label + 1  # First token encodes the class
        
        # Rest of the sequence is random but from a class-specific range
        base = label * 20 + 10
        X[i, 1:] = torch.randint(base, base + 10, (seq_len - 1,))
    
    # Shuffle the dataset
    perm = torch.randperm(n_samples)
    return X[perm], y[perm]


def compute_accuracy(model, lora_adapter, dataloader):
    """Compute accuracy on a dataset."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            # Get base model features
            embeddings = model.embedding(X_batch).mean(dim=1)
            hidden = model.relu(model.fc1(embeddings))
            
            # Combine base model and LoRA outputs
            base_output = model.fc2(hidden)
            lora_output = lora_adapter(hidden)
            output = base_output + lora_output
            
            predictions = torch.argmax(output, dim=1)
            correct += (predictions == y_batch).sum().item()
            total += y_batch.size(0)
    
    return correct / total


def mezo_step_batched(model, lora_adapter, dataloader, epsilon=1e-3, lr=0.01):
    """Perform one MeZO step with batched data."""
    # Collect LoRA parameters
    params = [lora_adapter.lora_A, lora_adapter.lora_B]
    
    # Sample perturbation
    z_list = [torch.randn_like(p) for p in params]
    
    # Initialize loss accumulators
    total_loss_plus = 0.0
    total_loss_minus = 0.0
    total_samples = 0
    
    model.eval()  # Ensure model is in eval mode
    
    # Process all batches
    for X_batch, y_batch in dataloader:
        batch_size = X_batch.size(0)
        
        # Forward pass with +εz
        for i, p in enumerate(params):
            p.data.add_(epsilon * z_list[i])
        
        with torch.no_grad():
            embeddings = model.embedding(X_batch).mean(dim=1)
            hidden = model.relu(model.fc1(embeddings))
            base_output = model.fc2(hidden)
            lora_output = lora_adapter(hidden)
            output = base_output + lora_output
            loss_plus = nn.functional.cross_entropy(output, y_batch, reduction='sum')
        
        # Forward pass with -εz
        for i, p in enumerate(params):
            p.data.add_(-2 * epsilon * z_list[i])
        
        with torch.no_grad():
            embeddings = model.embedding(X_batch).mean(dim=1)
            hidden = model.relu(model.fc1(embeddings))
            base_output = model.fc2(hidden)
            lora_output = lora_adapter(hidden)
            output = base_output + lora_output
            loss_minus = nn.functional.cross_entropy(output, y_batch, reduction='sum')
        
        # Restore parameters
        for i, p in enumerate(params):
            p.data.add_(epsilon * z_list[i])
        
        total_loss_plus += loss_plus.item()
        total_loss_minus += loss_minus.item()
        total_samples += batch_size
    
    # Average losses
    avg_loss_plus = total_loss_plus / total_samples
    avg_loss_minus = total_loss_minus / total_samples
    
    # Compute gradient estimate
    grad_estimate = (avg_loss_plus - avg_loss_minus) / (2 * epsilon)
    
    # Update parameters
    for i, p in enumerate(params):
        p.data.add_(-lr * grad_estimate * z_list[i])
    
    return (avg_loss_plus + avg_loss_minus) / 2


def test_mezo_convergence():
    """Test MeZO convergence with stable hyperparameters."""
    print("=" * 60)
    print("MeZO Stable Convergence Test")
    print("=" * 60)
    
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configuration
    vocab_size = 100
    hidden_dim = 64
    num_classes = 5
    seq_len = 10
    batch_size = 32
    rank = 8
    
    # Create model and LoRA adapter
    model = SimpleModel(vocab_size, hidden_dim, num_classes)
    lora_adapter = LoRAAdapter(hidden_dim, num_classes, rank)
    
    # Freeze base model
    for param in model.parameters():
        param.requires_grad = False
    
    # Create datasets
    X_train, y_train = create_classification_dataset(1000, seq_len, vocab_size, num_classes)
    X_test, y_test = create_classification_dataset(200, seq_len, vocab_size, num_classes)
    
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\nConfiguration:")
    print(f"  Model: vocab_size={vocab_size}, hidden_dim={hidden_dim}, num_classes={num_classes}")
    print(f"  LoRA rank: {rank}")
    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Batch size: {batch_size}")
    
    # Training parameters
    num_epochs = 100
    epsilon = 5e-3  # Moderate epsilon
    learning_rate = 0.5  # Moderate learning rate
    
    print(f"\nTraining parameters:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Epsilon: {epsilon}")
    print(f"  Learning rate: {learning_rate}")
    
    # Track metrics
    train_losses = []
    test_losses = []
    test_accuracies = []
    
    print("\nStarting MeZO training...")
    print("-" * 50)
    print("Epoch | Train Loss | Test Loss | Test Accuracy")
    print("-" * 50)
    
    for epoch in range(num_epochs):
        # Training step
        train_loss = mezo_step_batched(model, lora_adapter, train_loader, epsilon, learning_rate)
        train_losses.append(train_loss)
        
        # Evaluation every 5 epochs
        if epoch % 5 == 0:
            # Compute test loss
            model.eval()
            test_loss = 0.0
            total_samples = 0
            
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    embeddings = model.embedding(X_batch).mean(dim=1)
                    hidden = model.relu(model.fc1(embeddings))
                    base_output = model.fc2(hidden)
                    lora_output = lora_adapter(hidden)
                    output = base_output + lora_output
                    loss = nn.functional.cross_entropy(output, y_batch, reduction='sum')
                    test_loss += loss.item()
                    total_samples += y_batch.size(0)
            
            avg_test_loss = test_loss / total_samples
            test_losses.append(avg_test_loss)
            
            # Compute accuracy
            test_acc = compute_accuracy(model, lora_adapter, test_loader)
            test_accuracies.append(test_acc)
            
            print(f"{epoch:5d} | {train_loss:.6f} | {avg_test_loss:.6f} | {test_acc:.2%}")
    
    print("-" * 50)
    
    # Analyze results
    print("\nConvergence Analysis:")
    
    initial_train_loss = np.mean(train_losses[:5])
    final_train_loss = np.mean(train_losses[-5:])
    train_improvement = (initial_train_loss - final_train_loss) / initial_train_loss * 100
    
    initial_test_loss = test_losses[0]
    final_test_loss = test_losses[-1]
    test_improvement = (initial_test_loss - final_test_loss) / initial_test_loss * 100
    
    initial_accuracy = test_accuracies[0]
    final_accuracy = test_accuracies[-1]
    accuracy_gain = (final_accuracy - initial_accuracy) * 100
    
    print(f"  Initial train loss: {initial_train_loss:.4f}")
    print(f"  Final train loss: {final_train_loss:.4f}")
    print(f"  Train improvement: {train_improvement:.1f}%")
    print(f"  Initial test accuracy: {initial_accuracy:.1%}")
    print(f"  Final test accuracy: {final_accuracy:.1%}")
    print(f"  Accuracy gain: {accuracy_gain:.1f} percentage points")
    
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Training loss
    ax = axes[0]
    ax.plot(train_losses, 'b-', alpha=0.7, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)
    
    # Test loss
    ax = axes[1]
    epochs = list(range(0, num_epochs, 5))
    ax.plot(epochs, test_losses, 'g-', marker='o', linewidth=2, markersize=8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Test Loss')
    ax.grid(True, alpha=0.3)
    
    # Test accuracy
    ax = axes[2]
    ax.plot(epochs, test_accuracies, 'r-', marker='s', linewidth=2, markersize=8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Test Accuracy')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('mezo_stable_convergence.png', dpi=150)
    print(f"\nPlots saved to: mezo_stable_convergence.png")
    
    # Success criteria
    converged = (train_improvement > 10 and final_accuracy > initial_accuracy + 0.1)
    
    print("\n" + "=" * 60)
    print(f"CONVERGENCE TEST: {'PASSED' if converged else 'FAILED'}")
    print("=" * 60)
    
    if converged:
        print("✓ MeZO successfully converged!")
        print(f"  - Training loss improved by {train_improvement:.1f}%")
        print(f"  - Test accuracy improved from {initial_accuracy:.1%} to {final_accuracy:.1%}")
        print(f"  - This demonstrates MeZO can optimize LoRA parameters effectively")
    else:
        print("✗ MeZO needs hyperparameter tuning for this task")
    
    return converged, {
        'final_accuracy': final_accuracy,
        'train_improvement': train_improvement,
        'accuracy_gain': accuracy_gain
    }


if __name__ == "__main__":
    converged, results = test_mezo_convergence()
    
    if converged:
        print("\n✅ MeZO convergence verified!")
    else:
        print("\n⚠️  MeZO convergence needs further investigation")