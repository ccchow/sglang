#!/usr/bin/env python3
"""
Detailed profiling of MeZO compute bottlenecks.
Identifies optimization opportunities for CUDA kernel implementation.
"""

import torch
import time
import numpy as np
from typing import Dict, List, Tuple
import torch.profiler
import matplotlib.pyplot as plt

class DetailedProfiler:
    """Profile individual MeZO operations to identify bottlenecks."""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.timings = {}
        
    def time_operation(self, name: str, fn, *args, **kwargs):
        """Time a single operation."""
        # Warm up
        for _ in range(3):
            fn(*args, **kwargs)
        
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        times = []
        for _ in range(10):
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            if self.device == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        self.timings[name] = {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times)
        }
        return result

def profile_mezo_operations():
    """Profile each component of MeZO algorithm."""
    print("=" * 60)
    print("MeZO Compute Bottleneck Analysis")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Test configurations
    configs = [
        {'name': 'Small', 'batch': 4, 'seq_len': 128, 'hidden': 768, 'lora_rank': 8},
        {'name': 'Medium', 'batch': 8, 'seq_len': 256, 'hidden': 1024, 'lora_rank': 16},
        {'name': 'Large', 'batch': 16, 'seq_len': 512, 'hidden': 2048, 'lora_rank': 32},
    ]
    
    results = []
    
    for config in configs:
        print(f"\n{'-' * 40}")
        print(f"Configuration: {config['name']}")
        print(f"  Batch: {config['batch']}, Seq: {config['seq_len']}, Hidden: {config['hidden']}")
        
        profiler = DetailedProfiler(device)
        
        # Setup tensors
        batch_size = config['batch']
        seq_len = config['seq_len']
        hidden_dim = config['hidden']
        lora_rank = config['lora_rank']
        
        # Model components
        W_base = torch.randn(hidden_dim, hidden_dim, device=device)
        A = torch.randn(lora_rank, hidden_dim, device=device)
        B = torch.randn(hidden_dim, lora_rank, device=device)
        x = torch.randn(batch_size, seq_len, hidden_dim, device=device)
        
        # MeZO parameters
        epsilon = 1e-3
        n_samples = 20
        
        # 1. Profile perturbation generation
        def generate_perturbation():
            z_A = torch.randn_like(A)
            z_B = torch.randn_like(B)
            return z_A, z_B
        
        profiler.time_operation("1. Generate Perturbation", generate_perturbation)
        
        # 2. Profile perturbation application (in-place)
        z_A, z_B = generate_perturbation()
        def apply_perturbation_inplace():
            A.add_(epsilon * z_A)
            B.add_(epsilon * z_B)
        
        profiler.time_operation("2. Apply Perturbation (in-place)", apply_perturbation_inplace)
        
        # 3. Profile LoRA weight computation
        def compute_lora_weight():
            return B @ A
        
        profiler.time_operation("3. Compute LoRA Weight (B @ A)", compute_lora_weight)
        
        # 4. Profile combined weight computation
        def compute_combined_weight():
            return W_base + B @ A
        
        profiler.time_operation("4. Combine Base + LoRA", compute_combined_weight)
        
        # 5. Profile forward pass
        def forward_pass():
            W = W_base + B @ A
            # Simulate transformer layer
            h = x
            for _ in range(12):  # 12 layers
                h = torch.nn.functional.linear(h, W)
                h = torch.nn.functional.relu(h)
            return h.mean()
        
        profiler.time_operation("5. Forward Pass (12 layers)", forward_pass)
        
        # 6. Profile MeZO gradient estimation (full)
        def mezo_gradient_step():
            grad_A = torch.zeros_like(A)
            grad_B = torch.zeros_like(B)
            
            for _ in range(n_samples):
                # Generate perturbation
                z_A = torch.randn_like(A)
                z_B = torch.randn_like(B)
                
                # Forward +epsilon
                A.add_(epsilon * z_A)
                B.add_(epsilon * z_B)
                loss_plus = forward_pass()
                
                # Forward -epsilon (from +epsilon state)
                A.add_(-2 * epsilon * z_A)
                B.add_(-2 * epsilon * z_B)
                loss_minus = forward_pass()
                
                # Restore original
                A.add_(epsilon * z_A)
                B.add_(epsilon * z_B)
                
                # Accumulate gradient
                grad_scale = (loss_plus - loss_minus) / (2 * epsilon)
                grad_A.add_(z_A * grad_scale)
                grad_B.add_(z_B * grad_scale)
            
            grad_A.div_(n_samples)
            grad_B.div_(n_samples)
            return grad_A, grad_B
        
        profiler.time_operation("6. Full MeZO Step (20 samples)", mezo_gradient_step)
        
        # 7. Profile potential optimizations
        
        # 7a. Fused perturbation and forward
        def fused_perturbation_forward():
            """Simulate fused kernel that applies perturbation and starts forward pass."""
            z_A = torch.randn_like(A)
            z_B = torch.randn_like(B)
            # In real CUDA kernel, this would be fused
            W_perturbed = W_base + (B + epsilon * z_B) @ (A + epsilon * z_A)
            h = torch.nn.functional.linear(x, W_perturbed)
            return h.mean(), z_A, z_B
        
        profiler.time_operation("7a. Fused Perturbation+Forward", fused_perturbation_forward)
        
        # 7b. Batched perturbation computation
        def batched_perturbation(n_batch=4):
            """Process multiple perturbations in parallel."""
            # Stack perturbations
            z_A_batch = torch.randn(n_batch, *A.shape, device=device)
            z_B_batch = torch.randn(n_batch, *B.shape, device=device)
            
            # Batched LoRA computation
            A_batch = A.unsqueeze(0) + epsilon * z_A_batch
            B_batch = B.unsqueeze(0) + epsilon * z_B_batch
            
            # Batched matrix multiplication
            W_batch = torch.einsum('bij,bjk->bik', B_batch, A_batch)
            return W_batch
        
        profiler.time_operation("7b. Batched Perturbations (4)", lambda: batched_perturbation(4))
        
        # Store results
        result = {
            'config': config,
            'timings': profiler.timings
        }
        results.append(result)
        
        # Print timing breakdown
        print("\nTiming Breakdown:")
        total_mezo = profiler.timings["6. Full MeZO Step (20 samples)"]['mean']
        for name, stats in profiler.timings.items():
            mean_time = stats['mean'] * 1000  # Convert to ms
            if name == "6. Full MeZO Step (20 samples)":
                print(f"{name:40s}: {mean_time:8.3f} ms")
            else:
                # Estimate contribution to total
                if name in ["1. Generate Perturbation", "2. Apply Perturbation (in-place)"]:
                    contrib = (stats['mean'] * 40) / total_mezo * 100  # 40 = 2 * 20 samples
                elif name == "5. Forward Pass (12 layers)":
                    contrib = (stats['mean'] * 40) / total_mezo * 100  # 40 = 2 * 20 samples
                else:
                    contrib = 0
                print(f"{name:40s}: {mean_time:8.3f} ms ({contrib:5.1f}% of MeZO)")
    
    # Analyze optimization opportunities
    print("\n" + "=" * 60)
    print("Optimization Opportunities")
    print("=" * 60)
    
    # Calculate potential speedups
    for i, result in enumerate(results):
        config = result['config']
        timings = result['timings']
        
        print(f"\n{config['name']} Configuration:")
        
        # Current MeZO time
        mezo_time = timings["6. Full MeZO Step (20 samples)"]['mean']
        
        # Breakdown
        perturbation_time = timings["1. Generate Perturbation"]['mean'] * 40
        apply_time = timings["2. Apply Perturbation (in-place)"]['mean'] * 40
        forward_time = timings["5. Forward Pass (12 layers)"]['mean'] * 40
        
        overhead = mezo_time - forward_time
        
        print(f"  Total MeZO time: {mezo_time*1000:.2f} ms")
        print(f"  Forward pass time: {forward_time*1000:.2f} ms ({forward_time/mezo_time*100:.1f}%)")
        print(f"  Overhead: {overhead*1000:.2f} ms ({overhead/mezo_time*100:.1f}%)")
        
        # Potential optimizations
        fused_time = timings["7a. Fused Perturbation+Forward"]['mean']
        batched_time = timings["7b. Batched Perturbations (4)"]['mean']
        
        print(f"\n  Optimization Potential:")
        print(f"    - Fused kernels could save: ~{(perturbation_time + apply_time)*1000:.2f} ms")
        print(f"    - Batched computation speedup: {timings['3. Compute LoRA Weight (B @ A)']['mean'] / (batched_time/4):.2f}x")
        print(f"    - Theoretical speedup: up to {mezo_time / (forward_time + overhead*0.2):.2f}x")
    
    return results

def design_cuda_kernels():
    """Design specifications for optimized CUDA kernels."""
    print("\n" + "=" * 60)
    print("CUDA Kernel Design for MeZO")
    print("=" * 60)
    
    kernels = [
        {
            'name': 'mezo_fused_perturbation_lora',
            'description': 'Fused kernel for perturbation generation and LoRA computation',
            'operations': [
                '1. Generate random perturbations z_A, z_B using cuRAND',
                '2. Compute perturbed LoRA: W = (B + εz_B) @ (A + εz_A)',
                '3. Add to base weight: W_total = W_base + W',
                '4. Return W_total and store z_A, z_B for gradient computation'
            ],
            'benefits': [
                'Eliminates multiple kernel launches',
                'Reduces memory bandwidth for intermediate results',
                'Can use shared memory for small LoRA matrices'
            ]
        },
        {
            'name': 'mezo_batched_forward_difference',
            'description': 'Process multiple perturbations in parallel',
            'operations': [
                '1. Generate batch of perturbations',
                '2. Compute forward passes for +ε and -ε in parallel',
                '3. Compute loss differences',
                '4. Accumulate gradient estimates'
            ],
            'benefits': [
                'Better GPU utilization',
                'Amortizes kernel launch overhead',
                'Can process 4-8 perturbations simultaneously'
            ]
        },
        {
            'name': 'mezo_gradient_accumulation',
            'description': 'Optimized gradient accumulation with perturbations',
            'operations': [
                '1. Input: perturbations z, loss differences Δℓ',
                '2. Compute: grad += z * Δℓ / (2ε)',
                '3. Use atomic operations for thread-safe accumulation',
                '4. Output: accumulated gradients'
            ],
            'benefits': [
                'Eliminates temporary gradient storage',
                'Can be fused with loss computation',
                'Supports variable number of samples'
            ]
        }
    ]
    
    for kernel in kernels:
        print(f"\n{kernel['name']}:")
        print(f"  {kernel['description']}")
        print(f"\n  Operations:")
        for op in kernel['operations']:
            print(f"    {op}")
        print(f"\n  Benefits:")
        for benefit in kernel['benefits']:
            print(f"    - {benefit}")
    
    # Estimate performance gains
    print("\n" + "=" * 60)
    print("Estimated Performance Gains")
    print("=" * 60)
    
    optimizations = [
        ("Fused perturbation + LoRA computation", "30-40%"),
        ("Batched forward passes (4x)", "20-30%"),
        ("Optimized gradient accumulation", "10-15%"),
        ("Combined optimizations", "50-70%")
    ]
    
    for opt, gain in optimizations:
        print(f"  {opt:40s}: {gain} reduction in compute time")
    
    print(f"\n  Estimated final speedup: {9 / (9 * 0.3):.1f}x (from 9x to ~3x overhead)")

if __name__ == "__main__":
    # Profile MeZO operations
    results = profile_mezo_operations()
    
    # Design CUDA kernels
    design_cuda_kernels()
    
    # Additional analysis with PyTorch profiler
    if torch.cuda.is_available():
        print("\n" + "=" * 60)
        print("PyTorch Profiler Analysis")
        print("=" * 60)
        print("Running detailed profiling with PyTorch profiler...")
        
        # Simple MeZO step for profiling
        def profile_mezo_step():
            device = 'cuda'
            A = torch.randn(16, 1024, device=device)
            B = torch.randn(1024, 16, device=device)
            x = torch.randn(8, 256, 1024, device=device)
            
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            ) as prof:
                for _ in range(5):
                    # Generate perturbation
                    z_A = torch.randn_like(A)
                    z_B = torch.randn_like(B)
                    
                    # Apply perturbation
                    A_pert = A + 1e-3 * z_A
                    B_pert = B + 1e-3 * z_B
                    
                    # Forward pass
                    W = B_pert @ A_pert
                    output = torch.nn.functional.linear(x, W)
                    loss = output.mean()
                    
                    torch.cuda.synchronize()
            
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
            
            # Save trace for visualization
            prof.export_chrome_trace("mezo_trace.json")
            print("\nTrace saved to mezo_trace.json (view in chrome://tracing)")
        
        profile_mezo_step()