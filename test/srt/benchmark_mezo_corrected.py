#!/usr/bin/env python3
"""
Corrected benchmark comparing MeZO (2 forward passes) with standard backpropagation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

class SimpleModel(nn.Module):
    """Simple model for benchmarking."""
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=12):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(n_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x


class LoRALayer(nn.Module):
    """LoRA adapter layer."""
    def __init__(self, in_features, out_features, rank=8):
        super().__init__()
        self.A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        
    def forward(self, x):
        return x @ (self.B @ self.A).t()


def benchmark_single_step(model, lora_layers, x, y, method='backprop', epsilon=1e-3):
    """Benchmark a single optimization step."""
    
    if method == 'backprop':
        # Standard backpropagation
        start = time.perf_counter()
        
        # Forward pass
        output = model(x)
        for lora in lora_layers:
            output = output + lora(x)
        loss = F.mse_loss(output, y)
        
        # Backward pass
        loss.backward()
        
        # Gradient update would happen here
        
        torch.cuda.synchronize() if x.is_cuda else None
        elapsed = time.perf_counter() - start
        
        # Clear gradients
        model.zero_grad()
        for lora in lora_layers:
            lora.zero_grad()
            
    elif method == 'mezo':
        # MeZO optimization
        start = time.perf_counter()
        
        # Sample perturbation
        z_list = []
        for lora in lora_layers:
            z_A = torch.randn_like(lora.A)
            z_B = torch.randn_like(lora.B)
            z_list.append((z_A, z_B))
        
        # Forward pass with +epsilon
        for i, lora in enumerate(lora_layers):
            lora.A.data.add_(epsilon * z_list[i][0])
            lora.B.data.add_(epsilon * z_list[i][1])
        
        output = model(x)
        for lora in lora_layers:
            output = output + lora(x)
        loss_plus = F.mse_loss(output, y)
        
        # Forward pass with -epsilon (from +epsilon state)
        for i, lora in enumerate(lora_layers):
            lora.A.data.add_(-2 * epsilon * z_list[i][0])
            lora.B.data.add_(-2 * epsilon * z_list[i][1])
        
        output = model(x)
        for lora in lora_layers:
            output = output + lora(x)
        loss_minus = F.mse_loss(output, y)
        
        # Restore original parameters
        for i, lora in enumerate(lora_layers):
            lora.A.data.add_(epsilon * z_list[i][0])
            lora.B.data.add_(epsilon * z_list[i][1])
        
        # Gradient estimation would happen here
        grad_scale = (loss_plus - loss_minus) / (2 * epsilon)
        
        torch.cuda.synchronize() if x.is_cuda else None
        elapsed = time.perf_counter() - start
        
    return elapsed


def run_benchmark():
    """Run comprehensive benchmark."""
    print("=" * 60)
    print("MeZO vs Backpropagation Benchmark (Corrected)")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    # Test configurations
    configs = [
        {'name': 'Small', 'input': 768, 'hidden': 768, 'output': 768, 'layers': 12, 'rank': 8},
        {'name': 'Medium', 'input': 1024, 'hidden': 4096, 'output': 1024, 'layers': 24, 'rank': 16},
        {'name': 'Large', 'input': 4096, 'hidden': 4096, 'output': 4096, 'layers': 32, 'rank': 32},
    ]
    
    batch_sizes = [1, 4, 16, 64]
    seq_lengths = [128, 512]
    
    results = []
    
    for config in configs:
        print(f"\nTesting {config['name']} model configuration:")
        print(f"  Layers: {config['layers']}, Hidden: {config['hidden']}, LoRA rank: {config['rank']}")
        
        # Create model
        model = SimpleModel(
            config['input'], config['hidden'], config['output'], config['layers']
        ).to(device)
        
        # Create LoRA layers (one per model layer)
        lora_layers = []
        for _ in range(config['layers']):
            lora = LoRALayer(config['input'], config['output'], config['rank']).to(device)
            lora_layers.append(lora)
        
        # Count parameters
        model_params = sum(p.numel() for p in model.parameters())
        lora_params = sum(sum(p.numel() for p in lora.parameters()) for lora in lora_layers)
        print(f"  Model parameters: {model_params/1e6:.1f}M")
        print(f"  LoRA parameters: {lora_params/1e6:.1f}M")
        
        config_results = []
        
        for batch_size in batch_sizes:
            for seq_len in seq_lengths:
                # Create dummy data
                x = torch.randn(batch_size, config['input'], device=device)
                y = torch.randn(batch_size, config['output'], device=device)
                
                # Warmup
                for _ in range(5):
                    benchmark_single_step(model, lora_layers, x, y, 'backprop')
                    benchmark_single_step(model, lora_layers, x, y, 'mezo')
                
                # Benchmark
                n_runs = 20
                
                # Backprop
                backprop_times = []
                for _ in range(n_runs):
                    t = benchmark_single_step(model, lora_layers, x, y, 'backprop')
                    backprop_times.append(t)
                backprop_time = np.median(backprop_times)
                
                # MeZO
                mezo_times = []
                for _ in range(n_runs):
                    t = benchmark_single_step(model, lora_layers, x, y, 'mezo')
                    mezo_times.append(t)
                mezo_time = np.median(mezo_times)
                
                # Store results
                result = {
                    'batch_size': batch_size,
                    'seq_len': seq_len,
                    'backprop_time': backprop_time,
                    'mezo_time': mezo_time,
                    'speedup': backprop_time / mezo_time
                }
                config_results.append(result)
                
                print(f"  Batch={batch_size:3d}: Backprop={backprop_time*1000:6.2f}ms, "
                      f"MeZO={mezo_time*1000:6.2f}ms, Speedup={result['speedup']:.2f}x")
        
        results.append({
            'config': config,
            'results': config_results
        })
    
    # Analysis
    print("\n" + "=" * 60)
    print("Analysis")
    print("=" * 60)
    
    # Average speedup across all configurations
    all_speedups = []
    for r in results:
        all_speedups.extend([res['speedup'] for res in r['results']])
    
    avg_speedup = np.mean(all_speedups)
    print(f"\nAverage speedup (Backprop time / MeZO time): {avg_speedup:.2f}x")
    
    if avg_speedup > 1:
        print("✓ MeZO is FASTER than backpropagation!")
    else:
        print("✗ MeZO is slower than backpropagation")
    
    # Memory analysis
    print("\nMemory Analysis:")
    print("-" * 40)
    print("Backpropagation memory: O(model_size + batch_size × seq_len × hidden_dim)")
    print("MeZO memory: O(model_size)")
    print("\nFor large models and sequences, MeZO's memory advantage is significant.")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Time comparison
    batch_sizes_plot = [1, 4, 16, 64]
    for i, r in enumerate(results):
        config_name = r['config']['name']
        # Get times for seq_len=512
        times_backprop = []
        times_mezo = []
        for res in r['results']:
            if res['seq_len'] == 512:
                times_backprop.append(res['backprop_time'] * 1000)
                times_mezo.append(res['mezo_time'] * 1000)
        
        if len(times_backprop) == len(batch_sizes_plot):
            ax1.plot(batch_sizes_plot, times_backprop, 'o-', label=f'{config_name} (Backprop)')
            ax1.plot(batch_sizes_plot, times_mezo, 's--', label=f'{config_name} (MeZO)')
    
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Time per Step (ms)')
    ax1.set_title('MeZO vs Backpropagation Time')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Speedup by model size
    model_names = [r['config']['name'] for r in results]
    avg_speedups = []
    for r in results:
        speedups = [res['speedup'] for res in r['results']]
        avg_speedups.append(np.mean(speedups))
    
    bars = ax2.bar(model_names, avg_speedups)
    ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Equal performance')
    ax2.set_xlabel('Model Configuration')
    ax2.set_ylabel('Speedup (Backprop / MeZO)')
    ax2.set_title('MeZO Speedup by Model Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Color bars based on speedup
    for bar, speedup in zip(bars, avg_speedups):
        if speedup > 1:
            bar.set_color('green')
        else:
            bar.set_color('red')
    
    plt.tight_layout()
    plt.savefig('mezo_corrected_benchmark.png', dpi=150)
    print(f"\nPlots saved to: mezo_corrected_benchmark.png")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("1. MeZO uses exactly 2 forward passes per optimization step")
    print("2. Backpropagation uses 1 forward + 1 backward pass")
    print("3. Since backward pass ≈ 2x forward pass, total cost is similar")
    print("4. MeZO's advantage is MEMORY efficiency, not necessarily speed")
    print("5. For memory-constrained scenarios, MeZO enables training otherwise impossible models")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmark()