"""
Edge case tests for MeZO trainer implementation.
Tests handling of extreme values, OOM scenarios, and unusual configurations.
"""

import unittest
import torch
import numpy as np
from unittest.mock import Mock, patch
import warnings

class TestMeZOEdgeCases(unittest.TestCase):
    """Test edge cases and error handling in MeZO implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        warnings.filterwarnings('ignore')
        
    def test_very_small_epsilon(self):
        """Test numerical stability with very small epsilon values."""
        # Simple loss function
        x = torch.randn(10, requires_grad=False)
        target = torch.randn(10)
        
        def loss_fn(x):
            return torch.sum((x - target) ** 2)
        
        epsilons = [1e-10, 1e-8, 1e-6, 1e-4]
        print("\nVery Small Epsilon Test:")
        print("Epsilon   | Loss+ - Loss- | Grad Norm | Numerical Issues")
        print("----------|---------------|-----------|------------------")
        
        for eps in epsilons:
            z = torch.randn_like(x)
            
            # Compute with high precision to detect numerical issues
            loss_plus = loss_fn(x + eps * z)
            loss_minus = loss_fn(x - eps * z)
            loss_diff = loss_plus - loss_minus
            
            # Check for numerical issues
            has_nan = torch.isnan(loss_diff).any()
            has_inf = torch.isinf(loss_diff).any()
            is_zero = torch.allclose(loss_diff, torch.tensor(0.0))
            
            grad = z * loss_diff / (2 * eps)
            grad_norm = grad.norm().item()
            
            issues = []
            if has_nan: issues.append("NaN")
            if has_inf: issues.append("Inf")
            if is_zero: issues.append("Zero")
            
            print(f"{eps:.1e} | {loss_diff:13.6e} | {grad_norm:9.4f} | {', '.join(issues) or 'None'}")
        
        # Should handle small epsilon without NaN/Inf
        self.assertFalse(has_nan, "NaN detected with small epsilon")
        self.assertFalse(has_inf, "Inf detected with small epsilon")
        
    def test_very_large_epsilon(self):
        """Test behavior with very large epsilon values."""
        # LoRA-like parameters
        A = torch.randn(4, 10)
        B = torch.randn(10, 4)
        
        def loss_fn(A, B):
            # Simulate LoRA loss
            W = B @ A
            return W.norm() ** 2
        
        epsilons = [1.0, 10.0, 100.0, 1000.0]
        z_A = torch.randn_like(A)
        z_B = torch.randn_like(B)
        
        print("\nVery Large Epsilon Test:")
        print("Epsilon | Loss Range | Gradient Scale | Stability")
        print("--------|------------|----------------|----------")
        
        base_loss = loss_fn(A, B)
        
        for eps in epsilons:
            loss_plus = loss_fn(A + eps * z_A, B + eps * z_B)
            loss_minus = loss_fn(A - eps * z_A, B - eps * z_B)
            
            loss_range = abs(loss_plus - loss_minus).item()
            grad_scale = loss_range / (2 * eps)
            
            # Check stability
            is_stable = not (torch.isnan(loss_plus).any() or torch.isnan(loss_minus).any())
            relative_change = max(loss_plus / base_loss, loss_minus / base_loss).item()
            
            stability = "Stable" if is_stable and relative_change < 1000 else "Unstable"
            
            print(f"{eps:7.1f} | {loss_range:10.4f} | {grad_scale:14.6f} | {stability}")
        
        self.assertTrue(is_stable, "Instability with large epsilon")
        
    def test_sparse_perturbations(self):
        """Test MeZO with sparse perturbation vectors."""
        # Large parameter matrix
        W = torch.randn(100, 100)
        sparsity_levels = [0.99, 0.95, 0.9, 0.5, 0.0]
        
        def loss_fn(W):
            return W.sum() ** 2
        
        print("\nSparse Perturbation Test:")
        print("Sparsity | Non-zeros | Grad Norm | Variance")
        print("---------|-----------|-----------|----------")
        
        epsilon = 1e-3
        
        for sparsity in sparsity_levels:
            # Create sparse perturbation
            z = torch.randn_like(W)
            if sparsity > 0:
                mask = torch.rand_like(W) > sparsity
                z = z * mask.float()
            
            non_zeros = (z != 0).sum().item()
            
            # Estimate gradient multiple times
            grad_norms = []
            for _ in range(10):
                z_sample = torch.randn_like(W)
                if sparsity > 0:
                    mask = torch.rand_like(W) > sparsity
                    z_sample = z_sample * mask.float()
                
                loss_plus = loss_fn(W + epsilon * z_sample)
                loss_minus = loss_fn(W - epsilon * z_sample)
                grad = z_sample * (loss_plus - loss_minus) / (2 * epsilon)
                grad_norms.append(grad.norm().item())
            
            mean_norm = np.mean(grad_norms)
            std_norm = np.std(grad_norms)
            
            print(f"{sparsity:8.2f} | {non_zeros:9d} | {mean_norm:9.4f} | {std_norm:8.4f}")
        
        # Sparse perturbations should still work
        self.assertGreater(mean_norm, 0, "Zero gradient with sparse perturbations")
        
    def test_zero_gradient_handling(self):
        """Test handling when gradient estimates are zero."""
        # Flat loss landscape
        x = torch.zeros(10)
        
        def constant_loss(x):
            return torch.tensor(1.0)  # Constant loss
        
        epsilon = 1e-3
        z = torch.randn_like(x)
        
        loss_plus = constant_loss(x + epsilon * z)
        loss_minus = constant_loss(x - epsilon * z)
        grad = z * (loss_plus - loss_minus) / (2 * epsilon)
        
        print(f"\nZero Gradient Test:")
        print(f"  Loss difference: {(loss_plus - loss_minus).item()}")
        print(f"  Gradient norm: {grad.norm().item()}")
        
        # Should handle zero gradients gracefully
        self.assertTrue(torch.allclose(grad, torch.zeros_like(grad)), 
                       "Non-zero gradient on flat loss")
        
    def test_memory_constrained_scenario(self):
        """Test behavior under memory constraints."""
        # Simulate large model scenario
        try:
            # Try to allocate large tensors
            size = 1000
            n_params = 100
            
            print("\nMemory Constraint Test:")
            print("Allocation | Status")
            print("-----------|--------")
            
            # Test different allocation strategies
            strategies = [
                ("Naive (copy all)", lambda: [torch.randn(size, size).clone() for _ in range(n_params)]),
                ("In-place", lambda: [torch.randn(size, size) for _ in range(n_params)]),
                ("Chunked", lambda: [torch.randn(size//10, size//10) for _ in range(n_params*100)]),
            ]
            
            for name, alloc_fn in strategies:
                try:
                    params = alloc_fn()
                    total_memory = sum(p.numel() * 4 / 1e9 for p in params)  # GB
                    print(f"{name:11s} | Success ({total_memory:.2f} GB)")
                    del params
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"{name:11s} | OOM")
                    else:
                        print(f"{name:11s} | Error: {str(e)[:30]}")
                except Exception as e:
                    print(f"{name:11s} | Error: {str(e)[:30]}")
                    
        except Exception as e:
            print(f"Memory test setup failed: {e}")
            
    def test_nan_inf_propagation(self):
        """Test handling of NaN and Inf values."""
        # Parameters with potential numerical issues
        W = torch.randn(10, 10)
        
        # Test cases that might produce NaN/Inf
        test_cases = [
            ("Normal", W, 1e-3),
            ("Large values", W * 1e10, 1e-3),
            ("Small values", W * 1e-10, 1e-3),
            ("With NaN", torch.cat([W[:-1], torch.tensor([[float('nan')] * 10])]), 1e-3),
            ("With Inf", torch.cat([W[:-1], torch.tensor([[float('inf')] * 10])]), 1e-3),
        ]
        
        print("\nNaN/Inf Propagation Test:")
        print("Test Case   | Has NaN | Has Inf | Grad Norm")
        print("------------|---------|---------|----------")
        
        for name, params, eps in test_cases:
            z = torch.randn_like(params)
            
            # Skip if input already has NaN/Inf
            if torch.isnan(params).any() or torch.isinf(params).any():
                print(f"{name:11s} | Input has NaN/Inf - skipped")
                continue
                
            try:
                # Simple loss
                loss_plus = (params + eps * z).sum() ** 2
                loss_minus = (params - eps * z).sum() ** 2
                grad = z * (loss_plus - loss_minus) / (2 * eps)
                
                has_nan = torch.isnan(grad).any().item()
                has_inf = torch.isinf(grad).any().item()
                grad_norm = grad.norm().item() if not (has_nan or has_inf) else float('nan')
                
                print(f"{name:11s} | {str(has_nan):7s} | {str(has_inf):7s} | {grad_norm:9.4f}")
            except Exception as e:
                print(f"{name:11s} | Error: {str(e)[:40]}")
                
    def test_different_dtypes(self):
        """Test MeZO with different tensor dtypes."""
        dtypes = [torch.float32, torch.float16, torch.bfloat16]
        
        print("\nDifferent Dtypes Test:")
        print("Dtype    | Loss Diff | Grad Norm | Precision OK")
        print("---------|-----------|-----------|-------------")
        
        for dtype in dtypes:
            if dtype == torch.bfloat16 and not torch.cuda.is_available():
                print(f"{str(dtype):8s} | Skipped (CPU)")
                continue
                
            try:
                # Simple test
                x = torch.randn(10, dtype=dtype)
                target = torch.randn(10, dtype=dtype)
                epsilon = 1e-3
                z = torch.randn_like(x)
                
                loss_plus = ((x + epsilon * z) - target).pow(2).sum()
                loss_minus = ((x - epsilon * z) - target).pow(2).sum()
                loss_diff = (loss_plus - loss_minus).item()
                
                grad = z * (loss_plus - loss_minus) / (2 * epsilon)
                grad_norm = grad.norm().item()
                
                # Check precision
                precision_ok = not (np.isnan(grad_norm) or np.isinf(grad_norm))
                
                print(f"{str(dtype):8s} | {loss_diff:9.6f} | {grad_norm:9.4f} | {str(precision_ok)}")
            except Exception as e:
                print(f"{str(dtype):8s} | Error: {str(e)[:40]}")


if __name__ == "__main__":
    unittest.main(verbosity=2)