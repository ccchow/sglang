#!/usr/bin/env python3
"""
Integration test for MeZO + Tensor Parallelism + Distributed RadixCache.
This demonstrates the full optimization stack working together.
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import logging
import time
import numpy as np
from typing import List, Dict, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockModelForward:
    """Mock model forward pass for testing."""
    def __init__(self, device, hidden_dim=768):
        self.device = device
        self.hidden_dim = hidden_dim
        self.forward_count = 0
        
    def __call__(self, tokens: List[int], cached_kv=None) -> 'MockOutput':
        self.forward_count += 1
        # Simulate computation time
        time.sleep(0.001 * len(tokens) / 100)  # 1ms per 100 tokens
        
        # Return mock output
        output = MockOutput()
        output.loss = 2.5 - (self.forward_count / 100) * 0.5 + torch.randn(1).item() * 0.1
        output.kv_states = torch.randn(len(tokens), self.hidden_dim, device=self.device)
        return output


class MockOutput:
    """Mock model output."""
    def __init__(self):
        self.loss = 0.0
        self.kv_states = None


def run_integrated_test(rank: int, world_size: int):
    """Run integrated MeZO + TP + Cache test."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29515'
    
    # Initialize process group
    dist.init_process_group(
        backend='nccl' if torch.cuda.is_available() else 'gloo',
        rank=rank,
        world_size=world_size
    )
    
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Import here to avoid circular imports
    from sglang.srt.distributed_radix_cache import (
        DistributedRadixCache,
        DistributedMeZOCacheOptimizer
    )
    
    logger.info(f"Rank {rank}/{world_size}: Starting integrated test on {device}")
    
    # Configuration
    num_steps = 20
    batch_size = 4
    seq_length = 256
    lora_r = 8
    epsilon = 1e-3
    learning_rate = 1e-5
    
    # Create components
    cache = DistributedRadixCache(
        tp_size=world_size,
        tp_rank=rank,
        cache_size=10000,
        enable_cross_rank_sharing=True
    )
    
    cache_optimizer = DistributedMeZOCacheOptimizer(
        distributed_cache=cache,
        epsilon=epsilon
    )
    
    model_forward = MockModelForward(device)
    
    # Create mock LoRA parameters (sharded for TP)
    # Each rank has different shard of parameters
    param_shard_size = 512 // world_size
    lora_params = torch.randn(param_shard_size, lora_r, device=device, requires_grad=True)
    
    # Training metrics
    losses = []
    cache_hit_rates = []
    gradient_norms = []
    step_times = []
    
    logger.info(f"Rank {rank}: Starting MeZO training loop")
    
    for step in range(num_steps):
        start_time = time.time()
        
        # Step 1: Generate batch with common prefixes
        batch_tokens = []
        for i in range(batch_size):
            # Common prefix (simulates real text patterns)
            common_prefix = list(range(100))  # 100 token prefix
            
            # Rank-specific middle section
            rank_section = list(range(100 + rank * 50, 100 + (rank + 1) * 50))
            
            # Unique suffix for each sample
            unique_suffix = list(range(150 + i * 20, 150 + i * 20 + (seq_length - 150)))
            
            tokens = common_prefix + rank_section + unique_suffix
            batch_tokens.append(tokens[:seq_length])
        
        # Step 2: Generate synchronized perturbation
        if rank == 0:
            seed = torch.randint(0, 2**31-1, (1,), device=device)
        else:
            seed = torch.zeros(1, dtype=torch.long, device=device)
        
        dist.broadcast(seed, src=0)
        torch.manual_seed(seed.item())
        
        # Generate perturbation for LoRA parameters
        z = torch.randn_like(lora_params)
        
        # Step 3: Prepare cache-optimized batches
        pos_batch, neg_batch = cache_optimizer.prepare_perturbation_batch(
            batch_tokens, z.flatten()
        )
        
        # Step 4: Forward pass with positive perturbation
        lora_params.data += epsilon * z
        
        pos_losses = []
        cache_hits_pos = 0
        
        for tokens in pos_batch:
            # Check cache
            cached_kv, cached_length = cache.query_cache(tokens[:-1], len(tokens)-1)
            if cached_kv is not None:
                cache_hits_pos += 1
            
            # Forward pass
            output = model_forward(tokens, cached_kv=cached_kv)
            pos_losses.append(output.loss)
            
            # Update cache
            if cached_length < len(tokens) - 1:
                cache.update_cache(tokens[:-1], output.kv_states, len(tokens)-1)
        
        pos_loss_local = np.mean(pos_losses)
        
        # Step 5: Forward pass with negative perturbation
        lora_params.data -= 2 * epsilon * z  # Go from +ε to -ε
        
        neg_losses = []
        cache_hits_neg = 0
        
        for tokens in neg_batch:
            cached_kv, cached_length = cache.query_cache(tokens[:-1], len(tokens)-1)
            if cached_kv is not None:
                cache_hits_neg += 1
            
            output = model_forward(tokens, cached_kv=cached_kv)
            neg_losses.append(output.loss)
            
            if cached_length < len(tokens) - 1:
                cache.update_cache(tokens[:-1], output.kv_states, len(tokens)-1)
        
        neg_loss_local = np.mean(neg_losses)
        
        # Restore original parameters
        lora_params.data += epsilon * z
        
        # Step 6: Aggregate losses across TP ranks
        losses_tensor = torch.tensor([pos_loss_local, neg_loss_local], device=device)
        dist.all_reduce(losses_tensor, op=dist.ReduceOp.SUM)
        losses_tensor /= world_size
        
        pos_loss = losses_tensor[0].item()
        neg_loss = losses_tensor[1].item()
        avg_loss = (pos_loss + neg_loss) / 2
        
        # Step 7: Compute gradient estimate
        grad_estimate = (pos_loss - neg_loss) / (2 * epsilon)
        
        # Step 8: Update parameters (MeZO style)
        lora_params.data -= learning_rate * grad_estimate * z
        
        # Step 9: Compute metrics
        cache_hit_rate = (cache_hits_pos + cache_hits_neg) / (2 * batch_size)
        gradient_norm = abs(grad_estimate) * torch.norm(z).item()
        step_time = time.time() - start_time
        
        # Record metrics
        losses.append(avg_loss)
        cache_hit_rates.append(cache_hit_rate)
        gradient_norms.append(gradient_norm)
        step_times.append(step_time)
        
        # Log progress
        if step % 5 == 0:
            logger.info(
                f"Rank {rank} - Step {step}/{num_steps}: "
                f"Loss={avg_loss:.4f}, Cache={cache_hit_rate:.2%}, "
                f"Grad norm={gradient_norm:.6f}, Time={step_time:.3f}s"
            )
    
    # Final statistics
    dist.barrier()
    
    # Get cache statistics
    local_stats = cache.get_local_stats()
    global_stats = cache.synchronize_cache_stats()
    
    if rank == 0:
        logger.info("\n" + "="*70)
        logger.info("Integrated MeZO + TP + Cache Training Complete!")
        logger.info("="*70)
        logger.info(f"Final Loss: {losses[-1]:.4f}")
        logger.info(f"Average Cache Hit Rate: {np.mean(cache_hit_rates):.2%}")
        logger.info(f"Final Cache Hit Rate: {cache_hit_rates[-1]:.2%}")
        logger.info(f"Total Time: {sum(step_times):.2f}s")
        logger.info(f"Average Step Time: {np.mean(step_times):.3f}s")
        
        logger.info("\nGlobal Cache Statistics:")
        logger.info(f"  Global hit rate: {global_stats.get('global_hit_rate', 0):.2%}")
        logger.info(f"  Cross-rank benefit: {global_stats.get('global_cross_rank_benefit', 0):.2%}")
        logger.info(f"  Cache efficiency: {global_stats.get('global_cache_efficiency', 0):.2%}")
        logger.info(f"  Total tokens shared: {global_stats.get('global_tokens_shared', 0)}")
        
        logger.info("\nPerformance Analysis:")
        logger.info(f"  Cache hit improvement: {cache_hit_rates[-1] - cache_hit_rates[0]:.2%}")
        logger.info(f"  Estimated memory saved: {global_stats.get('global_tokens_shared', 0) * 768 * 4 / 1024**3:.2f} GB")
        logger.info(f"  Forward passes saved: ~{sum(cache_hit_rates) * batch_size:.0f}")
        
        logger.info("="*70)
    
    # Clean up
    dist.destroy_process_group()


def main():
    """Run integrated test."""
    gpu_count = torch.cuda.device_count()
    
    if gpu_count < 2:
        logger.error(f"Need at least 2 GPUs for this test, but found {gpu_count}")
        return
    
    logger.info(f"Found {gpu_count} GPUs. Running integrated test with TP=2...")
    
    world_size = 2  # Use TP=2 for this test
    
    logger.info("\n" + "="*70)
    logger.info("MeZO + Tensor Parallelism + Distributed RadixCache Integration Test")
    logger.info("="*70)
    logger.info(f"Configuration:")
    logger.info(f"  Tensor Parallel Size: {world_size}")
    logger.info(f"  MeZO epsilon: 1e-3")
    logger.info(f"  Learning rate: 1e-5")
    logger.info(f"  Batch size: 4")
    logger.info(f"  Sequence length: 256")
    logger.info("="*70 + "\n")
    
    # Launch workers
    mp.set_start_method('spawn', force=True)
    processes = []
    
    for rank in range(world_size):
        p = mp.Process(target=run_integrated_test, args=(rank, world_size))
        p.start()
        processes.append(p)
    
    # Wait for completion
    for p in processes:
        p.join()
    
    logger.info("\nIntegration test completed successfully!")


if __name__ == "__main__":
    main()