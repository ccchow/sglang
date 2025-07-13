#!/usr/bin/env python3
"""
Analyze and visualize MeZO training results.
"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_training_results(output_dir):
    """Load training results from output directory."""
    results_path = os.path.join(output_dir, "training_results.json")
    with open(results_path, 'r') as f:
        return json.load(f)

def plot_training_curves(history, output_dir):
    """Create and save training curve plots."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('MeZO Training Results - OPT-125m', fontsize=16)
    
    # Plot 1: Training Loss
    ax = axes[0, 0]
    ax.plot(history['train_loss'], label='Train Loss', alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Over Time')
    ax.grid(True, alpha=0.3)
    
    # Add moving average
    window = 10
    if len(history['train_loss']) > window:
        moving_avg = np.convolve(history['train_loss'], 
                                 np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(history['train_loss'])), 
                moving_avg, 'r-', label=f'{window}-step MA', linewidth=2)
    ax.legend()
    
    # Plot 2: Evaluation Loss
    ax = axes[0, 1]
    if 'eval_loss' in history and history['eval_loss']:
        eval_steps = list(range(20, len(history['train_loss'])+1, 20))[:len(history['eval_loss'])]
        ax.plot(eval_steps, history['eval_loss'], 'o-', label='Eval Loss', markersize=8)
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('Evaluation Loss')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Plot 3: KV Cache Reuse Rate
    ax = axes[1, 0]
    ax.plot(history['kv_cache_reuse'], label='KV Reuse Rate', color='green', alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Reuse Rate')
    ax.set_title('KV Cache Reuse Rate')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Expected (50%)')
    ax.legend()
    
    # Plot 4: Step Times
    ax = axes[1, 1]
    ax.plot(history['step_times'], label='Step Time', alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Step Execution Time')
    ax.grid(True, alpha=0.3)
    avg_time = np.mean(history['step_times'])
    ax.axhline(y=avg_time, color='r', linestyle='--', alpha=0.5, 
               label=f'Average ({avg_time:.3f}s)')
    ax.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  📊 Plots saved to: {plot_path}")
    
def print_summary_statistics(results):
    """Print summary statistics of the training."""
    print("\n" + "="*70)
    print("MeZO Training Summary Statistics")
    print("="*70)
    
    config = results['config']
    history = results['history']
    final_metrics = results['final_metrics']
    
    print("\n📋 Configuration:")
    print(f"  • Model: {config['model_name']}")
    print(f"  • Steps: {config['num_train_steps']}")
    print(f"  • Batch size: {config['batch_size']}")
    print(f"  • Learning rate: {config['learning_rate']}")
    print(f"  • Epsilon: {config['epsilon']}")
    print(f"  • LoRA rank: {config['lora_rank']}")
    
    print("\n📊 Final Metrics:")
    print(f"  • Final train loss: {final_metrics['final_train_loss']:.4f}")
    print(f"  • Best eval loss: {final_metrics['best_eval_loss']:.4f}")
    print(f"  • Average KV reuse: {final_metrics['average_kv_reuse']:.1%}")
    
    print("\n📈 Training Statistics:")
    train_losses = history['train_loss']
    print(f"  • Initial loss: {train_losses[0]:.4f}")
    print(f"  • Min loss: {min(train_losses):.4f}")
    print(f"  • Max loss: {max(train_losses):.4f}")
    print(f"  • Loss std dev: {np.std(train_losses):.4f}")
    
    # Calculate improvement
    improvement = (train_losses[0] - train_losses[-1]) / train_losses[0] * 100
    print(f"  • Loss improvement: {improvement:.1f}%")
    
    print("\n⏱️  Performance Statistics:")
    step_times = history['step_times']
    total_time = sum(step_times)
    print(f"  • Total time: {total_time:.1f}s ({total_time/60:.2f} minutes)")
    print(f"  • Avg step time: {np.mean(step_times):.3f}s")
    print(f"  • Min step time: {min(step_times):.3f}s")
    print(f"  • Max step time: {max(step_times):.3f}s")
    
    # Theoretical speedup from KV cache
    kv_reuse = np.mean(history['kv_cache_reuse'])
    theoretical_speedup = 1 / (1 - kv_reuse * 0.95)  # Assuming 95% of time is attention
    print(f"  • Theoretical speedup from KV cache: {theoretical_speedup:.2f}x")
    
    print("\n💾 Memory Efficiency:")
    print(f"  • MeZO memory usage: Same as inference (no gradients)")
    print(f"  • LoRA parameters: {config['lora_rank'] * 4} per layer")
    print(f"  • Full model parameters: ~125M")
    print(f"  • Memory savings: >99% vs full fine-tuning")
    
    print("="*70)

def analyze_convergence(history):
    """Analyze training convergence."""
    train_losses = history['train_loss']
    
    # Calculate smoothed loss
    window = 10
    if len(train_losses) > window:
        smoothed = np.convolve(train_losses, np.ones(window)/window, mode='valid')
        
        # Check if loss is decreasing
        first_quarter = np.mean(smoothed[:len(smoothed)//4])
        last_quarter = np.mean(smoothed[-len(smoothed)//4:])
        
        print("\n🔍 Convergence Analysis:")
        print(f"  • First quarter avg loss: {first_quarter:.4f}")
        print(f"  • Last quarter avg loss: {last_quarter:.4f}")
        
        if last_quarter < first_quarter:
            print("  ✓ Model is converging (loss decreasing)")
        else:
            print("  ⚠️ Model may not be converging")
        
        # Calculate loss variance
        loss_variance = np.var(smoothed)
        print(f"  • Loss variance (smoothed): {loss_variance:.4f}")
        
        if loss_variance < 0.01:
            print("  ✓ Training is stable")
        else:
            print("  ⚠️ High variance in loss")

def main():
    """Main analysis function."""
    # Find the most recent output directory
    output_dirs = sorted([d for d in Path('.').glob('mezo_opt125m_output_*') if d.is_dir()])
    
    if not output_dirs:
        print("❌ No training output directories found!")
        return
    
    latest_dir = output_dirs[-1]
    print(f"📂 Analyzing results from: {latest_dir}")
    
    # Load results
    results = load_training_results(latest_dir)
    
    # Print summary statistics
    print_summary_statistics(results)
    
    # Analyze convergence
    analyze_convergence(results['history'])
    
    # Create plots
    print("\n📊 Creating visualization plots...")
    plot_training_curves(results['history'], latest_dir)
    
    # Print final recommendations
    print("\n💡 Recommendations:")
    final_loss = results['final_metrics']['final_train_loss']
    if final_loss > 3.5:
        print("  • Consider increasing learning rate or training steps")
        print("  • Try adjusting epsilon for better gradient estimates")
    else:
        print("  • Training appears successful")
        print("  • Consider evaluating on downstream tasks")
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()