"""
MeZO Distributed Trainer for multi-GPU tensor parallel training.

This module implements distributed MeZO training with tensor parallelism support,
enabling efficient training across multiple GPUs with synchronized perturbations
and gradient estimation.
"""

import torch
import torch.distributed as dist
import numpy as np
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass

from sglang.srt.mezo_trainer import MeZOTrainer
from sglang.srt.mezo_radix_optimizer import MeZORadixOptimizer
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.managers.schedule_batch import ScheduleBatch, Req
from sglang.srt.lora.tp_lora_manager import TPLoRAManager, TPLoRAOptimizer
from sglang.srt.distributed_radix_cache import (
    DistributedRadixCache, 
    DistributedMeZOCacheOptimizer
)

logger = logging.getLogger(__name__)


@dataclass
class DistributedTrainingConfig:
    """Configuration for distributed MeZO training."""
    tp_size: int = 1
    tp_rank: int = 0
    master_port: str = "29500"
    backend: str = "nccl"
    
    # Synchronization options
    sync_perturbation: bool = True
    sync_loss: bool = True
    
    # Performance options
    pipeline_forward: bool = False
    gradient_accumulation_steps: int = 1
    
    # Cache options
    enable_distributed_cache: bool = True
    cache_size_per_rank: int = 100000
    enable_cross_rank_sharing: bool = True


class MeZODistributedTrainer(MeZOTrainer):
    """
    Distributed MeZO trainer with tensor parallel support.
    
    Key features:
    - Synchronized perturbation generation across TP ranks
    - All-reduce for loss aggregation
    - TP-aware LoRA weight management
    - RadixAttention cache optimization per rank
    """
    
    def __init__(
        self,
        model_runner: ModelRunner,
        lora_manager,
        lora_name: str,
        tokenizer,
        distributed_config: DistributedTrainingConfig,
        **kwargs
    ):
        """
        Initialize distributed MeZO trainer.
        
        Args:
            model_runner: Model runner for this TP rank
            lora_manager: LoRA manager instance
            lora_name: Name of the LoRA adapter
            tokenizer: Tokenizer instance
            distributed_config: Distributed training configuration
            **kwargs: Additional arguments for base trainer
        """
        super().__init__(
            model_runner=model_runner,
            lora_manager=lora_manager,
            lora_name=lora_name,
            tokenizer=tokenizer,
            **kwargs
        )
        
        self.dist_config = distributed_config
        self.tp_size = distributed_config.tp_size
        self.tp_rank = distributed_config.tp_rank
        
        # Initialize distributed communication
        self._init_distributed()
        
        # Initialize fixed random generator for synchronized perturbations
        # All ranks use the same seed to generate identical perturbations
        self.perturbation_generator = torch.Generator(device='cuda')
        # Use a fixed seed that all ranks share
        # This eliminates the need for per-step seed broadcasting
        fixed_seed = 42  # Can be any fixed value
        self.perturbation_generator.manual_seed(fixed_seed)
        logger.info(f"Rank {self.tp_rank}: Initialized perturbation generator with fixed seed {fixed_seed}")
        
        # Create TP-aware LoRA manager
        if self.lora_manager:
            self.lora_manager = TPLoRAManager(
                base_lora_manager=self.lora_manager,
                tp_size=self.tp_size,
                tp_rank=self.tp_rank,
                tp_group=None  # Will use default group
            )
        
        # Create distributed RadixOptimizer
        if hasattr(self, 'radix_optimizer') and self.radix_optimizer:
            self.radix_optimizer = DistributedRadixOptimizer(
                epsilon=self.epsilon,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size
            )
        
        # Create distributed cache if enabled
        self.distributed_cache = None
        self.cache_optimizer = None
        if distributed_config.enable_distributed_cache:
            self.distributed_cache = DistributedRadixCache(
                tp_size=self.tp_size,
                tp_rank=self.tp_rank,
                tp_group=None,  # Will use default group
                cache_size=distributed_config.cache_size_per_rank,
                enable_cross_rank_sharing=distributed_config.enable_cross_rank_sharing
            )
            self.cache_optimizer = DistributedMeZOCacheOptimizer(
                distributed_cache=self.distributed_cache,
                epsilon=self.epsilon
            )
        
        logger.info(f"Initialized MeZODistributedTrainer on rank {self.tp_rank}/{self.tp_size}")
    
    def _init_distributed(self):
        """Initialize distributed communication."""
        if not dist.is_initialized():
            logger.info(f"Initializing distributed with backend={self.dist_config.backend}")
            dist.init_process_group(
                backend=self.dist_config.backend,
                init_method=f"tcp://localhost:{self.dist_config.master_port}",
                world_size=self.tp_size,
                rank=self.tp_rank
            )
        
        # Verify initialization
        assert dist.is_initialized(), "Distributed not initialized"
        assert dist.get_world_size() == self.tp_size, f"World size mismatch: {dist.get_world_size()} vs {self.tp_size}"
        assert dist.get_rank() == self.tp_rank, f"Rank mismatch: {dist.get_rank()} vs {self.tp_rank}"
    
    def generate_synchronized_perturbation(
        self,
        params: List[torch.nn.Parameter]
    ) -> List[torch.Tensor]:
        """
        Generate synchronized perturbation across all TP ranks.
        
        This ensures all ranks use the same random perturbation z,
        which is critical for correct MeZO gradient estimation.
        
        Since all ranks are initialized with the same fixed seed,
        they will generate identical perturbations without any
        communication overhead.
        
        Args:
            params: List of parameters to perturb
            
        Returns:
            List of perturbation tensors (same across all ranks)
        """
        if not self.dist_config.sync_perturbation:
            # Fallback to local perturbation (for testing)
            return super()._generate_perturbation(params)
        
        # All ranks use the same pre-initialized generator
        # No broadcasting needed - all ranks generate identical sequences
        z_list = []
        for p in params:
            if p.requires_grad:
                z = torch.randn_like(p, generator=self.perturbation_generator, device=p.device)
                z_list.append(z)
            else:
                z_list.append(None)
        
        logger.debug(f"Rank {self.tp_rank}: Generated synchronized perturbation (no broadcast needed)")
        
        return z_list
    
    def aggregate_losses(
        self,
        loss_plus: float,
        loss_minus: float
    ) -> Tuple[float, float]:
        """
        Aggregate losses across all TP ranks using all-reduce.
        
        Args:
            loss_plus: Local loss with positive perturbation
            loss_minus: Local loss with negative perturbation
            
        Returns:
            Tuple of (aggregated_loss_plus, aggregated_loss_minus)
        """
        if not self.dist_config.sync_loss:
            # No synchronization (for testing)
            return loss_plus, loss_minus
        
        # Convert to tensors for all-reduce
        losses_local = torch.tensor(
            [loss_plus, loss_minus],
            dtype=torch.float32,
            device='cuda'
        )
        
        # All-reduce sum
        dist.all_reduce(losses_local, op=dist.ReduceOp.SUM)
        
        # Average across TP ranks
        losses_local /= self.tp_size
        
        loss_plus_global = losses_local[0].item()
        loss_minus_global = losses_local[1].item()
        
        logger.debug(
            f"Rank {self.tp_rank}: Local losses=({loss_plus:.4f}, {loss_minus:.4f}), "
            f"Global losses=({loss_plus_global:.4f}, {loss_minus_global:.4f})"
        )
        
        return loss_plus_global, loss_minus_global
    
    def mezo_step(self, batch: Dict[str, Any]) -> float:
        """
        Perform one distributed MeZO optimization step.
        
        Overrides base class to use synchronized perturbations
        and aggregated losses.
        
        Args:
            batch: Batch of training data
            
        Returns:
            Average loss for this step
        """
        # Get LoRA parameters
        lora_adapter = self.lora_manager.get_adapter(self.lora_name)
        lora_params = [p for p in lora_adapter.parameters() if p.requires_grad]
        
        # Generate synchronized perturbation
        z_list = self.generate_synchronized_perturbation(lora_params)
        
        # Apply positive perturbation
        self._apply_perturbation(lora_params, z_list, self.epsilon)
        
        # Forward pass with positive perturbation
        loss_plus_local = self._forward_pass(batch, lora_params, self.epsilon, z_list)
        
        # Apply negative perturbation (from current state)
        self._apply_perturbation(lora_params, z_list, -2 * self.epsilon)
        
        # Forward pass with negative perturbation
        loss_minus_local = self._forward_pass(batch, lora_params, -self.epsilon, z_list)
        
        # Restore original parameters
        self._apply_perturbation(lora_params, z_list, self.epsilon)
        
        # Aggregate losses across ranks
        loss_plus, loss_minus = self.aggregate_losses(loss_plus_local, loss_minus_local)
        
        # Compute gradient estimate
        grad_estimate = (loss_plus - loss_minus) / (2 * self.epsilon)
        
        # Update parameters
        self._update_parameters(lora_params, z_list, grad_estimate)
        
        # Average loss
        avg_loss = (loss_plus + loss_minus) / 2
        
        # Update statistics
        self.steps += 1
        self.losses.append(avg_loss)
        
        # Log progress
        if self.steps % 10 == 0:
            logger.info(
                f"Rank {self.tp_rank}: Step {self.steps}, "
                f"Loss: {avg_loss:.4f}, "
                f"Gradient estimate: {grad_estimate:.6f}"
            )
        
        return avg_loss
    
    def _apply_perturbation(
        self,
        params: List[torch.nn.Parameter],
        z_list: List[Optional[torch.Tensor]],
        scale: float
    ):
        """Apply perturbation to parameters."""
        with torch.no_grad():
            for p, z in zip(params, z_list):
                if z is not None:
                    p.data.add_(z, alpha=scale)
    
    def _update_parameters(
        self,
        params: List[torch.nn.Parameter],
        z_list: List[Optional[torch.Tensor]],
        grad_estimate: float
    ):
        """Update parameters using MeZO gradient estimate."""
        with torch.no_grad():
            for p, z in zip(params, z_list):
                if z is not None:
                    p.data.add_(z, alpha=-self.learning_rate * grad_estimate)
    
    def collect_distributed_stats(self) -> Dict[str, Any]:
        """Collect and aggregate statistics from all ranks."""
        local_stats = {
            'rank': self.tp_rank,
            'steps': self.steps,
            'avg_loss': np.mean(self.losses) if self.losses else 0.0,
        }
        
        # Add RadixCache stats if available
        if hasattr(self, 'radix_optimizer') and self.radix_optimizer:
            cache_stats = self.radix_optimizer.get_distributed_stats()
            local_stats.update(cache_stats)
        
        # Gather stats from all ranks (optional)
        if self.tp_rank == 0:
            all_stats = [local_stats]
            for rank in range(1, self.tp_size):
                # In practice, use proper gather operation
                all_stats.append(local_stats)  # Placeholder
            
            # Aggregate
            global_stats = {
                'total_steps': local_stats['steps'],
                'avg_loss_global': np.mean([s['avg_loss'] for s in all_stats]),
                'tp_size': self.tp_size,
            }
            
            return global_stats
        
        return local_stats
    
    def save_checkpoint(self, path: str):
        """Save distributed checkpoint."""
        # Each rank saves its own LoRA shard
        checkpoint_path = f"{path}_rank{self.tp_rank}"
        super().save_checkpoint(checkpoint_path)
        
        # Synchronize
        dist.barrier()
        
        logger.info(f"Rank {self.tp_rank}: Saved checkpoint to {checkpoint_path}")
    
    def cleanup(self):
        """Clean up distributed resources."""
        if dist.is_initialized():
            dist.barrier()
            # Note: Don't destroy process group as it may be used by other components
        
        logger.info(f"Rank {self.tp_rank}: Cleanup complete")


class DistributedRadixOptimizer(MeZORadixOptimizer):
    """
    Distributed version of RadixOptimizer with TP support.
    
    Aggregates cache statistics across ranks and provides
    global optimization metrics.
    """
    
    def __init__(self, epsilon: float, tp_rank: int, tp_size: int):
        super().__init__(epsilon)
        self.tp_rank = tp_rank
        self.tp_size = tp_size
    
    def get_distributed_stats(self) -> Dict[str, float]:
        """Get distributed cache statistics."""
        local_stats = self.get_optimization_stats()
        
        # Prepare tensor for all-reduce
        stats_tensor = torch.tensor([
            local_stats['cache_hits'],
            local_stats['tokens_reused'],
            local_stats['tokens_computed'],
            local_stats['total_forward_passes']
        ], dtype=torch.float32, device='cuda')
        
        # All-reduce to get global stats
        dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
        
        # Compute global rates
        global_hits = stats_tensor[0].item()
        global_reused = stats_tensor[1].item()
        global_computed = stats_tensor[2].item()
        global_passes = stats_tensor[3].item()
        
        return {
            'global_cache_hit_rate': global_hits / global_passes if global_passes > 0 else 0.0,
            'global_token_reuse_rate': global_reused / (global_reused + global_computed) 
                                       if (global_reused + global_computed) > 0 else 0.0,
            'global_forward_passes': global_passes,
        }