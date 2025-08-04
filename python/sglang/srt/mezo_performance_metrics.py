"""
Performance metrics and monitoring for MeZO KV cache optimization.

This module provides comprehensive tracking and reporting of cache performance,
memory usage, and training efficiency improvements.
"""

import time
import torch
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ForwardPassMetrics:
    """Metrics for a single forward pass."""
    step: int
    perturbation_sign: int
    epsilon: float
    start_time: float
    end_time: float
    cache_hits: int = 0
    cache_misses: int = 0
    tokens_computed: int = 0
    tokens_reused: int = 0
    loss: float = 0.0
    memory_used_mb: float = 0.0
    
    @property
    def duration(self) -> float:
        if self.end_time == 0:
            return 0.0
        return self.end_time - self.start_time
    
    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0


@dataclass
class StepMetrics:
    """Aggregated metrics for a training step (both forward passes)."""
    step: int
    plus_pass: ForwardPassMetrics
    minus_pass: ForwardPassMetrics
    gradient_norm: float = 0.0
    parameter_update_norm: float = 0.0
    
    @property
    def total_duration(self) -> float:
        return self.plus_pass.duration + self.minus_pass.duration
    
    @property
    def cache_reuse_benefit(self) -> float:
        """Estimate time saved by cache reuse in second pass."""
        if self.plus_pass.duration > 0:
            return max(0, self.plus_pass.duration - self.minus_pass.duration)
        return 0.0
    
    @property
    def overall_cache_hit_rate(self) -> float:
        total_hits = self.plus_pass.cache_hits + self.minus_pass.cache_hits
        total_requests = (self.plus_pass.cache_hits + self.plus_pass.cache_misses + 
                         self.minus_pass.cache_hits + self.minus_pass.cache_misses)
        return total_hits / total_requests if total_requests > 0 else 0.0


class MeZOPerformanceTracker:
    """
    Comprehensive performance tracking for MeZO with KV cache optimization.
    
    Tracks:
    1. Per-step metrics (timing, cache hits, memory)
    2. Aggregated statistics over windows
    3. Cache efficiency trends
    4. Memory usage patterns
    5. Speedup estimates
    """
    
    def __init__(
        self,
        window_size: int = 100,
        enable_memory_tracking: bool = True,
        log_interval: int = 50
    ):
        self.window_size = window_size
        self.enable_memory_tracking = enable_memory_tracking
        self.log_interval = log_interval
        
        # Storage
        self.step_metrics: List[StepMetrics] = []
        self.current_step_data: Dict[str, Any] = {}
        
        # Aggregated stats
        self.window_stats: List[Dict[str, float]] = []
        self.baseline_time: Optional[float] = None  # Time without cache
        
        # Memory tracking
        self.peak_memory_mb = 0.0
        self.memory_history: List[Tuple[int, float]] = []
        
        logger.info(f"MeZOPerformanceTracker initialized with window_size={window_size}")
    
    def start_forward_pass(self, step: int, perturbation_sign: int, epsilon: float):
        """Start tracking a forward pass."""
        key = f"step{step}_sign{perturbation_sign}"
        
        self.current_step_data[key] = ForwardPassMetrics(
            step=step,
            perturbation_sign=perturbation_sign,
            epsilon=epsilon,
            start_time=time.time(),
            end_time=0.0
        )
        
        # Track memory at start
        if self.enable_memory_tracking:
            memory_mb = self._get_current_memory_usage()
            self.current_step_data[key].memory_used_mb = memory_mb
    
    def end_forward_pass(
        self,
        step: int,
        perturbation_sign: int,
        loss: float,
        cache_stats: Optional[Dict[str, Any]] = None
    ):
        """End tracking a forward pass and record metrics."""
        key = f"step{step}_sign{perturbation_sign}"
        
        if key not in self.current_step_data:
            logger.warning(f"No tracking data found for {key}")
            return
        
        metrics = self.current_step_data[key]
        metrics.end_time = time.time()
        metrics.loss = loss
        
        # Update cache statistics if provided
        if cache_stats:
            metrics.cache_hits = cache_stats.get('cache_hits', 0)
            metrics.cache_misses = cache_stats.get('cache_misses', 0)
            metrics.tokens_computed = cache_stats.get('tokens_computed', 0)
            metrics.tokens_reused = cache_stats.get('tokens_reused', 0)
        
        # Track memory at end
        if self.enable_memory_tracking:
            memory_mb = self._get_current_memory_usage()
            metrics.memory_used_mb = max(metrics.memory_used_mb, memory_mb)
            self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)
    
    def complete_step(
        self,
        step: int,
        gradient_norm: float = 0.0,
        parameter_update_norm: float = 0.0
    ):
        """Complete tracking for a training step."""
        plus_key = f"step{step}_sign1"
        minus_key = f"step{step}_sign-1"
        
        if plus_key not in self.current_step_data or minus_key not in self.current_step_data:
            logger.warning(f"Incomplete step data for step {step}")
            return
        
        # Create step metrics
        step_metrics = StepMetrics(
            step=step,
            plus_pass=self.current_step_data[plus_key],
            minus_pass=self.current_step_data[minus_key],
            gradient_norm=gradient_norm,
            parameter_update_norm=parameter_update_norm
        )
        
        self.step_metrics.append(step_metrics)
        
        # Clean up current step data
        del self.current_step_data[plus_key]
        del self.current_step_data[minus_key]
        
        # Update memory history
        if self.enable_memory_tracking:
            self.memory_history.append((step, self.peak_memory_mb))
        
        # Log periodically
        if step % self.log_interval == 0:
            self._log_current_stats(step)
        
        # Update window statistics
        if len(self.step_metrics) % self.window_size == 0:
            self._update_window_stats()
    
    def _get_current_memory_usage(self) -> float:
        """Get current GPU memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2
        return 0.0
    
    def _log_current_stats(self, step: int):
        """Log current performance statistics."""
        if not self.step_metrics:
            return
        
        # Get recent metrics
        recent_steps = self.step_metrics[-min(self.window_size, len(self.step_metrics)):]
        
        # Calculate aggregated stats
        avg_duration = np.mean([s.total_duration for s in recent_steps])
        avg_cache_hit_rate = np.mean([s.overall_cache_hit_rate for s in recent_steps])
        avg_cache_benefit = np.mean([s.cache_reuse_benefit for s in recent_steps])
        
        # Token statistics
        total_tokens_computed = sum(
            s.plus_pass.tokens_computed + s.minus_pass.tokens_computed 
            for s in recent_steps
        )
        total_tokens_reused = sum(
            s.plus_pass.tokens_reused + s.minus_pass.tokens_reused 
            for s in recent_steps
        )
        
        # Estimate speedup
        if self.baseline_time is None and len(self.step_metrics) > 10:
            # Use first few steps as baseline (before cache warms up)
            self.baseline_time = np.mean([s.total_duration for s in self.step_metrics[:5]])
        
        speedup = self.baseline_time / avg_duration if self.baseline_time and avg_duration > 0 else 1.0
        
        logger.info(
            f"\n=== MeZO Performance Report (Step {step}) ===\n"
            f"Average step time: {avg_duration:.3f}s\n"
            f"Cache hit rate: {avg_cache_hit_rate:.2%}\n"
            f"Cache time benefit: {avg_cache_benefit:.3f}s/step\n"
            f"Tokens computed: {total_tokens_computed:,}\n"
            f"Tokens reused: {total_tokens_reused:,}\n"
            f"Token reuse rate: {total_tokens_reused / (total_tokens_computed + total_tokens_reused):.2%}\n"
            f"Estimated speedup: {speedup:.2f}x\n"
            f"Peak memory: {self.peak_memory_mb:.1f}MB\n"
            f"==========================================="
        )
    
    def _update_window_stats(self):
        """Update windowed statistics."""
        if len(self.step_metrics) < self.window_size:
            return
        
        window = self.step_metrics[-self.window_size:]
        
        stats = {
            'window_start_step': window[0].step,
            'window_end_step': window[-1].step,
            'avg_step_time': np.mean([s.total_duration for s in window]),
            'avg_cache_hit_rate': np.mean([s.overall_cache_hit_rate for s in window]),
            'avg_plus_pass_time': np.mean([s.plus_pass.duration for s in window]),
            'avg_minus_pass_time': np.mean([s.minus_pass.duration for s in window]),
            'avg_cache_benefit': np.mean([s.cache_reuse_benefit for s in window]),
            'total_tokens_computed': sum(s.plus_pass.tokens_computed + s.minus_pass.tokens_computed for s in window),
            'total_tokens_reused': sum(s.plus_pass.tokens_reused + s.minus_pass.tokens_reused for s in window),
            'avg_gradient_norm': np.mean([s.gradient_norm for s in window if s.gradient_norm > 0]),
            'avg_memory_mb': np.mean([s.plus_pass.memory_used_mb for s in window]),
        }
        
        self.window_stats.append(stats)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get comprehensive summary statistics."""
        if not self.step_metrics:
            return {}
        
        all_durations = [s.total_duration for s in self.step_metrics]
        all_cache_rates = [s.overall_cache_hit_rate for s in self.step_metrics]
        all_benefits = [s.cache_reuse_benefit for s in self.step_metrics]
        
        # Calculate percentiles
        duration_percentiles = np.percentile(all_durations, [25, 50, 75, 90, 99])
        
        # Token statistics
        total_computed = sum(
            s.plus_pass.tokens_computed + s.minus_pass.tokens_computed 
            for s in self.step_metrics
        )
        total_reused = sum(
            s.plus_pass.tokens_reused + s.minus_pass.tokens_reused 
            for s in self.step_metrics
        )
        
        # Memory savings estimate
        memory_saved_mb = self._estimate_memory_savings()
        
        return {
            'total_steps': len(self.step_metrics),
            'avg_step_duration': np.mean(all_durations),
            'std_step_duration': np.std(all_durations),
            'duration_p25': duration_percentiles[0],
            'duration_p50': duration_percentiles[1],
            'duration_p75': duration_percentiles[2],
            'duration_p90': duration_percentiles[3],
            'duration_p99': duration_percentiles[4],
            'avg_cache_hit_rate': np.mean(all_cache_rates),
            'avg_cache_time_benefit': np.mean(all_benefits),
            'total_cache_time_saved': sum(all_benefits),
            'total_tokens_computed': total_computed,
            'total_tokens_reused': total_reused,
            'overall_token_reuse_rate': total_reused / (total_computed + total_reused) if (total_computed + total_reused) > 0 else 0,
            'peak_memory_mb': self.peak_memory_mb,
            'estimated_memory_saved_mb': memory_saved_mb,
            'estimated_overall_speedup': self._calculate_overall_speedup(),
        }
    
    def _estimate_memory_savings(self) -> float:
        """Estimate memory saved by KV cache reuse."""
        if not self.step_metrics:
            return 0.0
        
        # Estimate based on tokens reused
        total_reused = sum(
            s.plus_pass.tokens_reused + s.minus_pass.tokens_reused 
            for s in self.step_metrics
        )
        
        # Rough estimate: 2MB per 1000 tokens (depends on model size)
        # This is a placeholder - actual calculation should use model config
        return total_reused * 0.002
    
    def _calculate_overall_speedup(self) -> float:
        """Calculate overall speedup from cache optimization."""
        if len(self.step_metrics) < 20:
            return 1.0
        
        # Compare early steps (cold cache) to recent steps (warm cache)
        early_steps = self.step_metrics[:10]
        recent_steps = self.step_metrics[-10:]
        
        early_avg = np.mean([s.total_duration for s in early_steps])
        recent_avg = np.mean([s.total_duration for s in recent_steps])
        
        return early_avg / recent_avg if recent_avg > 0 else 1.0
    
    def plot_metrics(self, save_path: Optional[str] = None):
        """Generate plots of key metrics over time."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available for plotting")
            return
        
        if not self.step_metrics:
            logger.warning("No metrics to plot")
            return
        
        steps = [s.step for s in self.step_metrics]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Step duration over time
        ax1 = axes[0, 0]
        durations = [s.total_duration for s in self.step_metrics]
        ax1.plot(steps, durations, label='Total duration')
        ax1.plot(steps, [s.plus_pass.duration for s in self.step_metrics], 
                label='+ε pass', alpha=0.7)
        ax1.plot(steps, [s.minus_pass.duration for s in self.step_metrics], 
                label='-ε pass', alpha=0.7)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Duration (s)')
        ax1.set_title('Forward Pass Duration')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cache hit rate
        ax2 = axes[0, 1]
        cache_rates = [s.overall_cache_hit_rate for s in self.step_metrics]
        ax2.plot(steps, cache_rates)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Cache Hit Rate')
        ax2.set_title('KV Cache Hit Rate')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
        
        # Plot 3: Token reuse
        ax3 = axes[1, 0]
        tokens_computed = [s.plus_pass.tokens_computed + s.minus_pass.tokens_computed 
                          for s in self.step_metrics]
        tokens_reused = [s.plus_pass.tokens_reused + s.minus_pass.tokens_reused 
                        for s in self.step_metrics]
        ax3.plot(steps, tokens_computed, label='Computed')
        ax3.plot(steps, tokens_reused, label='Reused')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Tokens')
        ax3.set_title('Token Computation vs Reuse')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Memory usage
        ax4 = axes[1, 1]
        if self.memory_history:
            mem_steps, mem_values = zip(*self.memory_history)
            ax4.plot(mem_steps, mem_values)
            ax4.set_xlabel('Step')
            ax4.set_ylabel('Memory (MB)')
            ax4.set_title('Peak GPU Memory Usage')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            logger.info(f"Metrics plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def create_performance_tracker(**kwargs) -> MeZOPerformanceTracker:
    """Factory function to create a performance tracker."""
    return MeZOPerformanceTracker(**kwargs)