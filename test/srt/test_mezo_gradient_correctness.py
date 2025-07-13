"""
Unit tests for MeZO gradient estimation correctness.
Tests the gradient approximation against known analytical gradients.
"""

import unittest
import torch
import numpy as np
from typing import List, Tuple

class TestMeZOGradientCorrectness(unittest.TestCase):
    """Test MeZO gradient estimation accuracy against analytical gradients."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.epsilon = 1e-3
        self.tolerance = 0.3  # 30% relative error tolerance
        
    def test_linear_model_gradient(self):
        """Test gradient estimation on a simple linear model."""
        # Setup: y = Wx + b, loss = MSE
        batch_size, input_dim, output_dim = 10, 20, 5
        X = torch.randn(batch_size, input_dim)
        W_true = torch.randn(output_dim, input_dim)
        b_true = torch.randn(output_dim)
        y_true = X @ W_true.t() + b_true
        
        # Initialize parameters
        W = torch.randn(output_dim, input_dim, requires_grad=True)
        b = torch.randn(output_dim, requires_grad=True)
        
        # Analytical gradient
        y_pred = X @ W.t() + b
        loss = torch.nn.functional.mse_loss(y_pred, y_true)
        loss.backward()
        grad_W_analytical = W.grad.clone()
        grad_b_analytical = b.grad.clone()
        
        # MeZO gradient estimation
        W.grad = None
        b.grad = None
        W.requires_grad = False
        b.requires_grad = False
        
        # Sample perturbation direction
        z_W = torch.randn_like(W)
        z_b = torch.randn_like(b)
        
        # Forward passes with perturbations
        y_plus = X @ (W + self.epsilon * z_W).t() + (b + self.epsilon * z_b)
        loss_plus = torch.nn.functional.mse_loss(y_plus, y_true)
        
        y_minus = X @ (W - self.epsilon * z_W).t() + (b - self.epsilon * z_b)
        loss_minus = torch.nn.functional.mse_loss(y_minus, y_true)
        
        # Gradient estimation
        grad_scale = (loss_plus - loss_minus) / (2 * self.epsilon)
        grad_W_mezo = z_W * grad_scale
        grad_b_mezo = z_b * grad_scale
        
        # Compare gradients
        cosine_sim_W = torch.nn.functional.cosine_similarity(
            grad_W_analytical.flatten(), grad_W_mezo.flatten(), dim=0
        )
        cosine_sim_b = torch.nn.functional.cosine_similarity(
            grad_b_analytical.flatten(), grad_b_mezo.flatten(), dim=0
        )
        
        # Relative error
        rel_error_W = (grad_W_mezo - grad_W_analytical).norm() / grad_W_analytical.norm()
        rel_error_b = (grad_b_mezo - grad_b_analytical).norm() / grad_b_analytical.norm()
        
        print(f"\nLinear Model Gradient Test:")
        print(f"  W cosine similarity: {cosine_sim_W:.4f}")
        print(f"  b cosine similarity: {cosine_sim_b:.4f}")
        print(f"  W relative error: {rel_error_W:.4f}")
        print(f"  b relative error: {rel_error_b:.4f}")
        
        # Assert gradients are aligned
        self.assertGreater(cosine_sim_W, 0.5, "Weight gradient misaligned")
        self.assertGreater(cosine_sim_b, 0.5, "Bias gradient misaligned")
        
    def test_quadratic_function_gradient(self):
        """Test on quadratic function with known gradient."""
        # f(x) = 0.5 * x^T * A * x + b^T * x
        dim = 10
        A = torch.randn(dim, dim)
        A = (A + A.t()) / 2  # Make symmetric
        b = torch.randn(dim)
        x = torch.randn(dim, requires_grad=True)
        
        # Analytical gradient: grad = Ax + b
        grad_analytical = A @ x + b
        
        # MeZO gradient estimation
        x.requires_grad = False
        z = torch.randn_like(x)
        
        # Loss function
        def loss_fn(x):
            return 0.5 * torch.dot(x, A @ x) + torch.dot(b, x)
        
        loss_plus = loss_fn(x + self.epsilon * z)
        loss_minus = loss_fn(x - self.epsilon * z)
        
        grad_scale = (loss_plus - loss_minus) / (2 * self.epsilon)
        grad_mezo = z * grad_scale
        
        # Compare
        cosine_sim = torch.nn.functional.cosine_similarity(
            grad_analytical, grad_mezo, dim=0
        )
        rel_error = (grad_mezo - grad_analytical).norm() / grad_analytical.norm()
        
        print(f"\nQuadratic Function Gradient Test:")
        print(f"  Cosine similarity: {cosine_sim:.4f}")
        print(f"  Relative error: {rel_error:.4f}")
        
        self.assertGreater(cosine_sim, 0.8, "Quadratic gradient misaligned")
        self.assertLess(rel_error, self.tolerance, "Quadratic gradient error too large")
        
    def test_multi_sample_averaging(self):
        """Test that averaging over multiple samples improves accuracy."""
        # Simple linear function
        dim = 5
        w_true = torch.randn(dim)
        x = torch.randn(dim)
        
        def loss_fn(w):
            return torch.sum((w - w_true) ** 2)
        
        # True gradient
        grad_true = 2 * (x - w_true)
        
        # Test with different numbers of samples
        n_samples_list = [1, 10, 50, 100]
        errors = []
        
        for n_samples in n_samples_list:
            grad_estimates = []
            
            for _ in range(n_samples):
                z = torch.randn_like(x)
                loss_plus = loss_fn(x + self.epsilon * z)
                loss_minus = loss_fn(x - self.epsilon * z)
                grad_scale = (loss_plus - loss_minus) / (2 * self.epsilon)
                grad_estimates.append(z * grad_scale)
            
            # Average gradient estimate
            grad_avg = torch.stack(grad_estimates).mean(dim=0)
            rel_error = (grad_avg - grad_true).norm() / grad_true.norm()
            errors.append(rel_error.item())
        
        print(f"\nMulti-Sample Averaging Test:")
        for n, err in zip(n_samples_list, errors):
            print(f"  Samples: {n:3d}, Relative Error: {err:.4f}")
        
        # Error should decrease with more samples
        self.assertLess(errors[-1], errors[0] * 0.5, "Multi-sample averaging not improving")
        
    def test_perturbation_scale_sensitivity(self):
        """Test gradient estimation with different epsilon values."""
        # Simple squared loss
        x = torch.randn(10)
        target = torch.randn(10)
        
        def loss_fn(x):
            return torch.sum((x - target) ** 2)
        
        # True gradient
        grad_true = 2 * (x - target)
        
        # Test different epsilon values
        epsilons = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        errors = []
        
        for eps in epsilons:
            z = torch.randn_like(x)
            loss_plus = loss_fn(x + eps * z)
            loss_minus = loss_fn(x - eps * z)
            grad_scale = (loss_plus - loss_minus) / (2 * eps)
            grad_est = z * grad_scale
            
            rel_error = (grad_est - grad_true).norm() / grad_true.norm()
            errors.append(rel_error.item())
        
        print(f"\nPerturbation Scale Sensitivity Test:")
        for eps, err in zip(epsilons, errors):
            print(f"  Epsilon: {eps:.1e}, Relative Error: {err:.4f}")
        
        # Optimal epsilon should be around 1e-3 to 1e-2
        min_error_idx = np.argmin(errors)
        self.assertIn(min_error_idx, [2, 3], "Optimal epsilon not in expected range")


class TestMeZOLoRAGradients(unittest.TestCase):
    """Test MeZO gradient estimation specifically for LoRA parameters."""
    
    def setUp(self):
        """Set up LoRA-specific test fixtures."""
        torch.manual_seed(42)
        self.epsilon = 1e-3
        
    def test_lora_parameter_gradient(self):
        """Test gradient estimation for LoRA A and B matrices."""
        # Simulate LoRA: W = W_base + BA
        input_dim, output_dim, rank = 20, 10, 4
        
        W_base = torch.randn(output_dim, input_dim)
        A = torch.randn(rank, input_dim, requires_grad=True)
        B = torch.randn(output_dim, rank, requires_grad=True)
        
        # Data
        X = torch.randn(5, input_dim)
        y_true = torch.randn(5, output_dim)
        
        # Forward with LoRA
        def forward(A, B):
            W = W_base + B @ A
            return X @ W.t()
        
        # Analytical gradient
        y_pred = forward(A, B)
        loss = torch.nn.functional.mse_loss(y_pred, y_true)
        loss.backward()
        grad_A_true = A.grad.clone()
        grad_B_true = B.grad.clone()
        
        # MeZO estimation
        A.grad = None
        B.grad = None
        A.requires_grad = False
        B.requires_grad = False
        
        z_A = torch.randn_like(A)
        z_B = torch.randn_like(B)
        
        # Perturbed forward passes
        y_plus = forward(A + self.epsilon * z_A, B + self.epsilon * z_B)
        loss_plus = torch.nn.functional.mse_loss(y_plus, y_true)
        
        y_minus = forward(A - self.epsilon * z_A, B - self.epsilon * z_B)
        loss_minus = torch.nn.functional.mse_loss(y_minus, y_true)
        
        # Gradient estimation
        grad_scale = (loss_plus - loss_minus) / (2 * self.epsilon)
        grad_A_mezo = z_A * grad_scale
        grad_B_mezo = z_B * grad_scale
        
        # Compare
        cosine_A = torch.nn.functional.cosine_similarity(
            grad_A_true.flatten(), grad_A_mezo.flatten(), dim=0
        )
        cosine_B = torch.nn.functional.cosine_similarity(
            grad_B_true.flatten(), grad_B_mezo.flatten(), dim=0
        )
        
        print(f"\nLoRA Parameter Gradient Test:")
        print(f"  A matrix cosine similarity: {cosine_A:.4f}")
        print(f"  B matrix cosine similarity: {cosine_B:.4f}")
        
        self.assertGreater(cosine_A, 0.5, "LoRA A gradient misaligned")
        self.assertGreater(cosine_B, 0.5, "LoRA B gradient misaligned")


if __name__ == "__main__":
    unittest.main(verbosity=2)