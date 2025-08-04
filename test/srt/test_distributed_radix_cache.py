#!/usr/bin/env python3
"""
Test distributed RadixCache optimization for MeZO.
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import logging
import time
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_cache_sharing(rank: int, world_size: int):
    """Test cache sharing across TP ranks."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29512'
    
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
    
    # Import here to avoid issues
    from sglang.srt.distributed_radix_cache import DistributedRadixCache
    
    logger.info(f"Rank {rank}: Testing distributed cache sharing")
    
    # Create distributed cache
    cache = DistributedRadixCache(
        tp_size=world_size,
        tp_rank=rank,
        cache_size=1000,
        enable_cross_rank_sharing=True
    )
    
    # Each rank caches different prefixes
    base_tokens = list(range(100))
    
    # Rank 0 caches prefix [0, 1, 2, ...]
    # Rank 1 caches prefix [100, 101, 102, ...]
    # etc.
    rank_offset = rank * 100
    rank_tokens = [t + rank_offset for t in base_tokens]
    
    # Simulate KV states
    kv_states = torch.randn(len(rank_tokens), 768, device=device)  # 768 = hidden dim
    
    # Update cache
    cache.update_cache(rank_tokens, kv_states, len(rank_tokens))
    
    logger.info(f"Rank {rank}: Cached {len(rank_tokens)} tokens with offset {rank_offset}")
    
    # Synchronize
    dist.barrier()
    
    # Test local cache hit
    local_cached, local_length = cache.query_cache(rank_tokens[:50], 50)
    if local_cached is not None:
        logger.info(f"Rank {rank}: ✅ Local cache hit for own prefix")
    else:
        logger.error(f"Rank {rank}: ❌ Failed local cache lookup")
    
    # Test cross-rank query (would need actual implementation)
    other_rank = (rank + 1) % world_size
    other_tokens = [t + other_rank * 100 for t in base_tokens[:50]]
    
    # In real implementation, this would check cross-rank cache
    cross_cached, cross_length = cache.query_cache(other_tokens, 50)
    
    # Get cache statistics
    local_stats = cache.get_local_stats()
    logger.info(f"Rank {rank}: Local stats - {local_stats}")
    
    # Synchronize global stats
    global_stats = cache.synchronize_cache_stats()
    
    if rank == 0:
        logger.info(f"Global cache statistics: {global_stats}")
    
    dist.destroy_process_group()


def test_mezo_cache_optimization(rank: int, world_size: int):
    """Test MeZO-specific cache optimization."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29513'
    
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
    
    from sglang.srt.distributed_radix_cache import (
        DistributedRadixCache, 
        DistributedMeZOCacheOptimizer
    )
    
    logger.info(f"Rank {rank}: Testing MeZO cache optimization")
    
    # Create cache and optimizer
    cache = DistributedRadixCache(
        tp_size=world_size,
        tp_rank=rank,
        cache_size=10000
    )
    
    optimizer = DistributedMeZOCacheOptimizer(
        distributed_cache=cache,
        epsilon=1e-3
    )
    
    # Simulate batch of sequences
    batch_size = 4
    seq_length = 128
    
    batch_tokens = []
    for i in range(batch_size):
        # Create sequences with common prefixes
        # This simulates real text where sequences share common starts
        common_prefix = list(range(50))  # 50 token common prefix
        unique_suffix = list(range(50 + rank * 100 + i * 20, 50 + rank * 100 + i * 20 + 78))
        tokens = common_prefix + unique_suffix
        batch_tokens.append(tokens)
    
    # Prepare perturbation batches
    z_direction = torch.randn(1, device=device)  # Simplified
    pos_batch, neg_batch = optimizer.prepare_perturbation_batch(batch_tokens, z_direction)
    
    # Simulate forward passes
    total_cache_hits = 0
    total_requests = 0
    
    # First pass (positive perturbation)
    for tokens in pos_batch:
        # Query cache
        cached_kv, cached_length = cache.query_cache(tokens[:-1], len(tokens)-1)
        
        if cached_kv is not None:
            total_cache_hits += 1
        total_requests += 1
        
        # Update cache (simulate computing non-cached portion)
        if cached_length < len(tokens) - 1:
            kv_states = torch.randn(len(tokens), 768, device=device)
            cache.update_cache(tokens[:-1], kv_states, len(tokens)-1)
    
    # Second pass (negative perturbation) - should have high cache reuse
    for tokens in neg_batch:
        cached_kv, cached_length = cache.query_cache(tokens[:-1], len(tokens)-1)
        
        if cached_kv is not None:
            total_cache_hits += 1
        total_requests += 1
    
    # Calculate cache efficiency
    cache_hit_rate = total_cache_hits / total_requests if total_requests > 0 else 0
    
    logger.info(
        f"Rank {rank}: MeZO cache performance - "
        f"Hits: {total_cache_hits}/{total_requests} ({cache_hit_rate:.2%})"
    )
    
    # Get final statistics
    final_stats = cache.get_local_stats()
    logger.info(f"Rank {rank}: Final cache stats - {final_stats}")
    
    # Synchronize and report global stats
    dist.barrier()
    
    if rank == 0:
        global_report = optimizer.get_global_cache_report()
        logger.info("\n" + "="*60)
        logger.info("Global MeZO Cache Optimization Report:")
        logger.info(f"  Global hit rate: {global_report.get('global_hit_rate', 0):.2%}")
        logger.info(f"  Cross-rank benefit: {global_report.get('global_cross_rank_benefit', 0):.2%}")
        logger.info(f"  Tokens shared: {global_report.get('global_tokens_shared', 0)}")
        logger.info(f"  Cache efficiency: {global_report.get('global_cache_efficiency', 0):.2%}")
        logger.info("="*60)
    
    dist.destroy_process_group()


def test_cache_scalability(rank: int, world_size: int):
    """Test cache scalability with increasing sequence lengths."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29514'
    
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
    
    from sglang.srt.distributed_radix_cache import DistributedRadixCache
    
    logger.info(f"Rank {rank}: Testing cache scalability")
    
    # Create cache
    cache = DistributedRadixCache(
        tp_size=world_size,
        tp_rank=rank,
        cache_size=50000  # Larger cache for scalability test
    )
    
    # Test with increasing sequence lengths
    sequence_lengths = [128, 256, 512, 1024, 2048]
    results = []
    
    for seq_len in sequence_lengths:
        start_time = time.time()
        
        # Generate sequences
        num_sequences = 10
        cache_hits = 0
        
        for i in range(num_sequences):
            # Create sequence with rank-specific pattern
            tokens = list(range(rank * 1000 + i * 100, rank * 1000 + i * 100 + seq_len))
            
            # Query cache
            cached_kv, cached_length = cache.query_cache(tokens[:-1], len(tokens)-1)
            
            if cached_kv is not None:
                cache_hits += 1
            
            # Update cache
            kv_states = torch.randn(seq_len, 768, device=device)
            cache.update_cache(tokens[:-1], kv_states, len(tokens)-1)
        
        elapsed_time = time.time() - start_time
        hit_rate = cache_hits / num_sequences
        
        results.append({
            'seq_len': seq_len,
            'hit_rate': hit_rate,
            'time': elapsed_time
        })
        
        logger.info(
            f"Rank {rank} - Seq length {seq_len}: "
            f"Hit rate={hit_rate:.2%}, Time={elapsed_time:.3f}s"
        )
    
    # Report scalability results
    dist.barrier()
    
    if rank == 0:
        logger.info("\n" + "="*60)
        logger.info("Cache Scalability Results:")
        for result in results:
            logger.info(
                f"  Seq length {result['seq_len']:4d}: "
                f"Hit rate={result['hit_rate']:.2%}, "
                f"Time={result['time']:.3f}s"
            )
        logger.info("="*60)
    
    dist.destroy_process_group()


def main():
    """Run distributed cache tests."""
    gpu_count = torch.cuda.device_count()
    
    if gpu_count < 2:
        logger.warning(f"Only {gpu_count} GPU(s) available. Testing with CPU backend...")
        world_size = 2
    else:
        logger.info(f"Found {gpu_count} GPUs. Running distributed cache tests...")
        world_size = min(gpu_count, 4)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(world_size))
    
    # Test 1: Basic cache sharing
    logger.info("\n" + "="*60)
    logger.info("Test 1: Distributed Cache Sharing")
    logger.info("="*60)
    
    mp.set_start_method('spawn', force=True)
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=test_cache_sharing, args=(rank, world_size))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    # Test 2: MeZO cache optimization
    logger.info("\n" + "="*60)
    logger.info("Test 2: MeZO Cache Optimization")
    logger.info("="*60)
    
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=test_mezo_cache_optimization, args=(rank, world_size))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    # Test 3: Cache scalability
    logger.info("\n" + "="*60)
    logger.info("Test 3: Cache Scalability")
    logger.info("="*60)
    
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=test_cache_scalability, args=(rank, world_size))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    logger.info("\n" + "="*60)
    logger.info("All distributed cache tests completed!")
    logger.info("="*60)


if __name__ == "__main__":
    main()