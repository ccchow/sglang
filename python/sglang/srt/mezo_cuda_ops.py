"""
Python wrapper for MeZO CUDA operations.
Provides optimized implementations and fallback to PyTorch when CUDA kernels are not available.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Try to import CUDA kernels
try:
    import mezo_cuda_ops
    CUDA_AVAILABLE = True
    logger.info("MeZO CUDA kernels loaded successfully")
except ImportError:
    CUDA_AVAILABLE = False
    logger.warning("MeZO CUDA kernels not available, using PyTorch fallback")


class MeZOCudaOps:
    """Optimized CUDA operations for MeZO algorithm."""
    
    @staticmethod
    def fused_perturbation_lora(
        A: torch.Tensor,
        B: torch.Tensor, 
        W_base: torch.Tensor,
        epsilon: float,
        seed: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fused kernel for perturbation generation and LoRA computation.
        
        Args:
            A: LoRA A matrix (rank x hidden)
            B: LoRA B matrix (hidden x rank)
            W_base: Base weight matrix (hidden x hidden)
            epsilon: Perturbation scale
            seed: Random seed for perturbation generation
            
        Returns:
            W_plus: W_base + (B + eps*z_B) @ (A + eps*z_A)
            W_minus: W_base + (B - eps*z_B) @ (A - eps*z_A)
            z_A: Perturbation for A
            z_B: Perturbation for B
        """
        if CUDA_AVAILABLE and A.is_cuda:
            if seed is None:
                seed = torch.randint(0, 2**32, (1,)).item()
            
            # Call CUDA kernel
            result = mezo_cuda_ops.fused_perturbation_lora(A, B, W_base, epsilon, seed)
            return result[0], result[1], result[2], result[3]
        else:
            # PyTorch fallback
            z_A = torch.randn_like(A)
            z_B = torch.randn_like(B)
            
            # Compute perturbed weights
            W_lora_plus = (B + epsilon * z_B) @ (A + epsilon * z_A)
            W_lora_minus = (B - epsilon * z_B) @ (A - epsilon * z_A)
            
            W_plus = W_base + W_lora_plus
            W_minus = W_base + W_lora_minus
            
            return W_plus, W_minus, z_A, z_B
    
    @staticmethod
    def batched_mezo_forward(
        x: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        W_base: torch.Tensor,
        epsilon: float,
        n_perturbations: int,
        compute_loss_fn
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Batched MeZO forward passes for multiple perturbations.
        
        Args:
            x: Input tensor (batch x seq x hidden)
            A: LoRA A matrix (rank x hidden)
            B: LoRA B matrix (hidden x rank)
            W_base: Base weight matrix
            epsilon: Perturbation scale
            n_perturbations: Number of perturbations to process
            compute_loss_fn: Function to compute loss from output
            
        Returns:
            loss_diffs: Loss differences for each perturbation
            z_A_batch: Batched perturbations for A
            z_B_batch: Batched perturbations for B
            avg_loss: Average loss across perturbations
        """
        if CUDA_AVAILABLE and x.is_cuda and n_perturbations > 1:
            # Generate batched perturbations
            z_A_batch = torch.randn(n_perturbations, *A.shape, device=A.device, dtype=A.dtype)
            z_B_batch = torch.randn(n_perturbations, *B.shape, device=B.device, dtype=B.dtype)
            
            # Compute batched perturbed weights
            A_batch = A.unsqueeze(0) + epsilon * z_A_batch  # (n_pert, rank, hidden)
            B_batch = B.unsqueeze(0) + epsilon * z_B_batch  # (n_pert, hidden, rank)
            
            # Batched LoRA computation
            W_lora_plus = torch.einsum('phr,prh->phh', B_batch, A_batch)
            
            # For minus perturbation
            A_batch_minus = A.unsqueeze(0) - epsilon * z_A_batch
            B_batch_minus = B.unsqueeze(0) - epsilon * z_B_batch
            W_lora_minus = torch.einsum('phr,prh->phh', B_batch_minus, A_batch_minus)
            
            # Compute losses
            loss_plus_list = []
            loss_minus_list = []
            
            for i in range(n_perturbations):
                # Plus perturbation
                W_total = W_base + W_lora_plus[i]
                output = F.linear(x, W_total)
                loss_plus = compute_loss_fn(output)
                loss_plus_list.append(loss_plus)
                
                # Minus perturbation
                W_total = W_base + W_lora_minus[i]
                output = F.linear(x, W_total)
                loss_minus = compute_loss_fn(output)
                loss_minus_list.append(loss_minus)
            
            loss_plus = torch.stack(loss_plus_list)
            loss_minus = torch.stack(loss_minus_list)
            loss_diffs = loss_plus - loss_minus
            avg_loss = (loss_plus.mean() + loss_minus.mean()) / 2
            
            return loss_diffs, z_A_batch, z_B_batch, avg_loss
        else:
            # Sequential fallback
            loss_diffs = []
            z_A_list = []
            z_B_list = []
            loss_sum = 0
            
            for _ in range(n_perturbations):
                z_A = torch.randn_like(A)
                z_B = torch.randn_like(B)
                
                # Plus perturbation
                W_lora_plus = (B + epsilon * z_B) @ (A + epsilon * z_A)
                W_total = W_base + W_lora_plus
                output = F.linear(x, W_total)
                loss_plus = compute_loss_fn(output)
                
                # Minus perturbation
                W_lora_minus = (B - epsilon * z_B) @ (A - epsilon * z_A)
                W_total = W_base + W_lora_minus
                output = F.linear(x, W_total)
                loss_minus = compute_loss_fn(output)
                
                loss_diffs.append(loss_plus - loss_minus)
                z_A_list.append(z_A)
                z_B_list.append(z_B)
                loss_sum += loss_plus + loss_minus
            
            loss_diffs = torch.stack(loss_diffs)
            z_A_batch = torch.stack(z_A_list)
            z_B_batch = torch.stack(z_B_list)
            avg_loss = loss_sum / (2 * n_perturbations)
            
            return loss_diffs, z_A_batch, z_B_batch, avg_loss
    
    @staticmethod
    def gradient_accumulation(
        z_A_batch: torch.Tensor,
        z_B_batch: torch.Tensor,
        loss_diffs: torch.Tensor,
        epsilon: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Accumulate gradients from multiple perturbations.
        
        Args:
            z_A_batch: Batched perturbations for A (n_samples x rank x hidden)
            z_B_batch: Batched perturbations for B (n_samples x hidden x rank)
            loss_diffs: Loss differences (n_samples,)
            epsilon: Perturbation scale
            
        Returns:
            grad_A: Gradient for A
            grad_B: Gradient for B
        """
        if CUDA_AVAILABLE and z_A_batch.is_cuda:
            # Call CUDA kernel
            result = mezo_cuda_ops.gradient_accumulation(z_A_batch, z_B_batch, loss_diffs, epsilon)
            return result[0], result[1]
        else:
            # PyTorch fallback
            n_samples = z_A_batch.shape[0]
            
            # Expand loss_diffs for broadcasting
            loss_diffs = loss_diffs.view(n_samples, 1, 1)
            
            # Compute gradients
            grad_scale = loss_diffs / (2 * epsilon)
            grad_A = (z_A_batch * grad_scale).mean(dim=0)
            grad_B = (z_B_batch * grad_scale).mean(dim=0)
            
            return grad_A, grad_B


def benchmark_cuda_ops():
    """Benchmark CUDA operations vs PyTorch implementation."""
    import time
    
    print("=" * 60)
    print("MeZO CUDA Operations Benchmark")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"CUDA kernels available: {CUDA_AVAILABLE}")
    
    # Test configurations
    configs = [
        {'rank': 8, 'hidden': 768, 'batch': 4, 'seq': 128},
        {'rank': 16, 'hidden': 1024, 'batch': 8, 'seq': 256},
        {'rank': 32, 'hidden': 2048, 'batch': 16, 'seq': 512},
    ]
    
    for config in configs:
        print(f"\nConfig: rank={config['rank']}, hidden={config['hidden']}")
        
        # Setup tensors
        A = torch.randn(config['rank'], config['hidden'], device=device)
        B = torch.randn(config['hidden'], config['rank'], device=device)
        W_base = torch.randn(config['hidden'], config['hidden'], device=device)
        x = torch.randn(config['batch'], config['seq'], config['hidden'], device=device)
        
        ops = MeZOCudaOps()
        
        # Benchmark fused perturbation
        n_runs = 100
        torch.cuda.synchronize() if device == 'cuda' else None
        start = time.time()
        
        for _ in range(n_runs):
            W_plus, W_minus, z_A, z_B = ops.fused_perturbation_lora(A, B, W_base, 1e-3)
            
        torch.cuda.synchronize() if device == 'cuda' else None
        elapsed = time.time() - start
        
        print(f"  Fused perturbation: {elapsed/n_runs*1000:.3f} ms/iter")
        
        # Benchmark batched forward
        def dummy_loss(output):
            return output.mean()
        
        torch.cuda.synchronize() if device == 'cuda' else None
        start = time.time()
        
        for _ in range(10):
            loss_diffs, z_A_batch, z_B_batch, avg_loss = ops.batched_mezo_forward(
                x, A, B, W_base, 1e-3, 4, dummy_loss
            )
            
        torch.cuda.synchronize() if device == 'cuda' else None
        elapsed = time.time() - start
        
        print(f"  Batched forward (4 perturbations): {elapsed/10*1000:.3f} ms/iter")
        
        # Benchmark gradient accumulation
        z_A_batch = torch.randn(20, config['rank'], config['hidden'], device=device)
        z_B_batch = torch.randn(20, config['hidden'], config['rank'], device=device)
        loss_diffs = torch.randn(20, device=device)
        
        torch.cuda.synchronize() if device == 'cuda' else None
        start = time.time()
        
        for _ in range(100):
            grad_A, grad_B = ops.gradient_accumulation(z_A_batch, z_B_batch, loss_diffs, 1e-3)
            
        torch.cuda.synchronize() if device == 'cuda' else None
        elapsed = time.time() - start
        
        print(f"  Gradient accumulation (20 samples): {elapsed/100*1000:.3f} ms/iter")


if __name__ == "__main__":
    benchmark_cuda_ops()