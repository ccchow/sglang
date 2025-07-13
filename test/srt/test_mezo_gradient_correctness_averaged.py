"""
Unit tests for MeZO gradient estimation correctness with sample averaging.
Tests show that MeZO gradient approximation improves with multiple samples.
"""

import unittest
import torch
import numpy as np

class TestMeZOGradientWithAveraging(unittest.TestCase):
    """Test MeZO gradient estimation with multiple samples for accuracy."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.epsilon = 1e-3
        self.n_samples = 100  # Use multiple samples for better accuracy
        
    def estimate_gradient_mezo(self, loss_fn, params, epsilon, n_samples=1):
        """Estimate gradient using MeZO with multiple samples."""
        grad_estimates = []
        
        for _ in range(n_samples):
            # Sample perturbation direction
            z_list = [torch.randn_like(p) for p in params]
            
            # Perturbed losses
            params_plus = [p + epsilon * z for p, z in zip(params, z_list)]
            loss_plus = loss_fn(params_plus)
            
            params_minus = [p - epsilon * z for p, z in zip(params, z_list)]
            loss_minus = loss_fn(params_minus)
            
            # Gradient estimate for this sample
            grad_scale = (loss_plus - loss_minus) / (2 * epsilon)
            grads = [z * grad_scale for z in z_list]
            grad_estimates.append(grads)
        
        # Average over samples
        avg_grads = []
        for i in range(len(params)):
            avg_grad = torch.stack([g[i] for g in grad_estimates]).mean(dim=0)
            avg_grads.append(avg_grad)
        
        return avg_grads
    
    def test_convergence_with_samples(self):
        """Test that gradient estimation improves with more samples."""
        # Simple quadratic loss
        dim = 10
        x_true = torch.randn(dim)
        x = torch.randn(dim, requires_grad=True)
        
        def loss_fn(params):
            x = params[0]
            return torch.sum((x - x_true) ** 2)
        
        # True gradient
        loss = loss_fn([x])
        loss.backward()
        grad_true = x.grad.clone()
        
        # Test with different sample sizes
        sample_sizes = [1, 10, 50, 100, 500]
        errors = []
        cosine_sims = []
        
        for n in sample_sizes:
            x.grad = None
            x.requires_grad = False
            
            grad_mezo = self.estimate_gradient_mezo(loss_fn, [x], self.epsilon, n)[0]
            
            # Metrics
            cosine_sim = torch.nn.functional.cosine_similarity(
                grad_true.flatten(), grad_mezo.flatten(), dim=0
            )
            rel_error = (grad_mezo - grad_true).norm() / grad_true.norm()
            
            errors.append(rel_error.item())
            cosine_sims.append(cosine_sim.item())
        
        print("\nGradient Convergence with Sample Size:")
        print("Samples | Cosine Sim | Rel Error")
        print("--------|------------|----------")
        for n, cos, err in zip(sample_sizes, cosine_sims, errors):
            print(f"{n:7d} | {cos:10.4f} | {err:9.4f}")
        
        # Should converge to high accuracy with enough samples
        self.assertGreater(cosine_sims[-1], 0.95, "Gradient not converging with samples")
        self.assertLess(errors[-1], 0.1, "Error not decreasing with samples")
        
    def test_lora_gradient_estimation(self):
        """Test MeZO gradient estimation for LoRA parameters with averaging."""
        # LoRA setup
        input_dim, output_dim, rank = 20, 10, 4
        batch_size = 8
        
        W_base = torch.randn(output_dim, input_dim)
        A = torch.randn(rank, input_dim, requires_grad=True)
        B = torch.randn(output_dim, rank, requires_grad=True)
        
        X = torch.randn(batch_size, input_dim)
        y_true = torch.randn(batch_size, output_dim)
        
        def loss_fn(params):
            A, B = params
            W = W_base + B @ A
            y_pred = X @ W.t()
            return torch.nn.functional.mse_loss(y_pred, y_true)
        
        # True gradient
        loss = loss_fn([A, B])
        loss.backward()
        grad_A_true = A.grad.clone()
        grad_B_true = B.grad.clone()
        
        # MeZO gradient with averaging
        A.grad = None
        B.grad = None
        A.requires_grad = False
        B.requires_grad = False
        
        grads_mezo = self.estimate_gradient_mezo(
            loss_fn, [A, B], self.epsilon, self.n_samples
        )
        grad_A_mezo, grad_B_mezo = grads_mezo
        
        # Metrics
        cosine_A = torch.nn.functional.cosine_similarity(
            grad_A_true.flatten(), grad_A_mezo.flatten(), dim=0
        )
        cosine_B = torch.nn.functional.cosine_similarity(
            grad_B_true.flatten(), grad_B_mezo.flatten(), dim=0
        )
        
        print(f"\nLoRA Gradient Estimation ({self.n_samples} samples):")
        print(f"  A matrix cosine similarity: {cosine_A:.4f}")
        print(f"  B matrix cosine similarity: {cosine_B:.4f}")
        
        self.assertGreater(cosine_A, 0.85, "LoRA A gradient poor approximation")
        self.assertGreater(cosine_B, 0.85, "LoRA B gradient poor approximation")
        
    def test_epsilon_selection(self):
        """Test optimal epsilon selection for different loss landscapes."""
        # Test on different function types
        dim = 5
        x = torch.randn(dim)
        
        # Smooth quadratic
        def smooth_loss(params):
            x = params[0]
            return torch.sum(x ** 2)
        
        # Sharp loss with high curvature
        def sharp_loss(params):
            x = params[0]
            return torch.sum(torch.exp(x ** 2))
        
        epsilons = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        
        print("\nEpsilon Selection for Different Loss Landscapes:")
        
        for loss_fn, name in [(smooth_loss, "Smooth"), (sharp_loss, "Sharp")]:
            print(f"\n{name} Loss Function:")
            print("Epsilon | Gradient Norm | Variance")
            print("--------|---------------|----------")
            
            for eps in epsilons:
                # Estimate gradient multiple times
                grad_norms = []
                for _ in range(10):
                    grad = self.estimate_gradient_mezo(loss_fn, [x], eps, n_samples=10)[0]
                    grad_norms.append(grad.norm().item())
                
                mean_norm = np.mean(grad_norms)
                std_norm = np.std(grad_norms)
                print(f"{eps:7.1e} | {mean_norm:13.4f} | {std_norm:9.4f}")
        
        # No specific assertion - this is for analysis
        
    def test_batch_gradient_estimation(self):
        """Test gradient estimation with batched data."""
        # Neural network simulation
        input_dim, hidden_dim, output_dim = 10, 20, 5
        batch_sizes = [1, 4, 16, 64]
        
        W1 = torch.randn(hidden_dim, input_dim, requires_grad=True)
        W2 = torch.randn(output_dim, hidden_dim, requires_grad=True)
        
        print("\nBatch Gradient Estimation:")
        print("Batch Size | Cosine Sim | Time Ratio")
        print("-----------|------------|------------")
        
        for batch_size in batch_sizes:
            X = torch.randn(batch_size, input_dim)
            y_true = torch.randn(batch_size, output_dim)
            
            def loss_fn(params):
                W1, W2 = params
                h = torch.relu(X @ W1.t())
                y = h @ W2.t()
                return torch.nn.functional.mse_loss(y, y_true)
            
            # True gradient
            loss = loss_fn([W1, W2])
            loss.backward()
            grad_W1_true = W1.grad.clone()
            grad_W2_true = W2.grad.clone()
            
            # MeZO gradient
            W1.grad = None
            W2.grad = None
            W1.requires_grad = False
            W2.requires_grad = False
            
            import time
            start = time.time()
            grads_mezo = self.estimate_gradient_mezo(
                loss_fn, [W1, W2], self.epsilon, n_samples=50
            )
            elapsed = time.time() - start
            
            # Compare
            cosine_W1 = torch.nn.functional.cosine_similarity(
                grad_W1_true.flatten(), grads_mezo[0].flatten(), dim=0
            )
            cosine_W2 = torch.nn.functional.cosine_similarity(
                grad_W2_true.flatten(), grads_mezo[1].flatten(), dim=0
            )
            avg_cosine = (cosine_W1 + cosine_W2) / 2
            
            # Time ratio (normalized by batch size)
            time_ratio = elapsed / batch_size
            
            print(f"{batch_size:10d} | {avg_cosine:10.4f} | {time_ratio:11.4f}")
            
            W1.requires_grad = True
            W2.requires_grad = True


if __name__ == "__main__":
    unittest.main(verbosity=2)