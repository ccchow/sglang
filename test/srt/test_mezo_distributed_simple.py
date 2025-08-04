#!/usr/bin/env python3
"""
Simple test for MeZO distributed functionality without full sglang dependencies.
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_distributed_perturbation(rank, world_size):
    """Test synchronized perturbation generation across ranks."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
    
    # Initialize process group
    dist.init_process_group(
        backend='nccl' if torch.cuda.is_available() else 'gloo',
        rank=rank,
        world_size=world_size
    )
    
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    
    # Generate seed on rank 0 and broadcast
    if rank == 0:
        seed = torch.randint(0, 2**63, (1,), device=device)
    else:
        seed = torch.zeros(1, dtype=torch.long, device=device)
    
    dist.broadcast(seed, src=0)
    
    # All ranks use same seed to generate perturbation
    generator = torch.Generator(device=device)
    generator.manual_seed(seed.item())
    
    # Generate perturbation
    z = torch.randn(100, 100, generator=generator, device=device)
    
    # Compute checksum
    checksum = z.sum().item()
    
    logger.info(f"Rank {rank}: Seed={seed.item()}, Checksum={checksum:.6f}")
    
    # Gather checksums from all ranks
    checksums = [torch.tensor(0.0, device=device) for _ in range(world_size)]
    dist.all_gather(checksums, torch.tensor(checksum, device=device))
    
    # Verify all checksums match
    checksums_list = [c.item() for c in checksums]
    all_match = all(abs(c - checksums_list[0]) < 1e-6 for c in checksums_list)
    
    if rank == 0:
        if all_match:
            logger.info("✅ PASSED: All ranks generated identical perturbations")
            logger.info(f"   Checksums: {checksums_list}")
        else:
            logger.error("❌ FAILED: Perturbations differ across ranks")
            logger.error(f"   Checksums: {checksums_list}")
    
    dist.destroy_process_group()


def test_loss_aggregation(rank, world_size):
    """Test loss aggregation across ranks."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29501'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
    
    dist.init_process_group(
        backend='nccl' if torch.cuda.is_available() else 'gloo',
        rank=rank,
        world_size=world_size
    )
    
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    
    # Each rank has different local losses
    loss_plus_local = 1.0 + rank * 0.1
    loss_minus_local = 2.0 + rank * 0.2
    
    # Convert to tensors for all-reduce
    losses_local = torch.tensor(
        [loss_plus_local, loss_minus_local],
        dtype=torch.float32,
        device=device
    )
    
    # All-reduce sum
    dist.all_reduce(losses_local, op=dist.ReduceOp.SUM)
    
    # Average across ranks
    losses_local /= world_size
    
    loss_plus_global = losses_local[0].item()
    loss_minus_global = losses_local[1].item()
    
    logger.info(
        f"Rank {rank}: Local=({loss_plus_local:.4f}, {loss_minus_local:.4f}), "
        f"Global=({loss_plus_global:.4f}, {loss_minus_global:.4f})"
    )
    
    # Verify aggregation is correct
    expected_plus = sum(1.0 + r * 0.1 for r in range(world_size)) / world_size
    expected_minus = sum(2.0 + r * 0.2 for r in range(world_size)) / world_size
    
    if rank == 0:
        if (abs(loss_plus_global - expected_plus) < 1e-6 and 
            abs(loss_minus_global - expected_minus) < 1e-6):
            logger.info("✅ PASSED: Loss aggregation is correct")
            logger.info(f"   Expected: ({expected_plus:.4f}, {expected_minus:.4f})")
            logger.info(f"   Got: ({loss_plus_global:.4f}, {loss_minus_global:.4f})")
        else:
            logger.error("❌ FAILED: Loss aggregation is incorrect")
    
    dist.destroy_process_group()


def main():
    """Run distributed tests."""
    gpu_count = torch.cuda.device_count()
    
    if gpu_count < 2:
        logger.warning(f"Only {gpu_count} GPU(s) available. Need at least 2 for distributed tests.")
        logger.info("Testing with CPU backend instead...")
        world_size = 2
    else:
        logger.info(f"Found {gpu_count} GPUs. Running distributed tests...")
        world_size = min(gpu_count, 4)
    
    # Test 1: Perturbation consistency
    logger.info("\n" + "="*60)
    logger.info("Test 1: Synchronized Perturbation Generation")
    logger.info("="*60)
    
    mp.set_start_method('spawn', force=True)
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=test_distributed_perturbation, args=(rank, world_size))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    # Test 2: Loss aggregation
    logger.info("\n" + "="*60)
    logger.info("Test 2: Loss Aggregation")
    logger.info("="*60)
    
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=test_loss_aggregation, args=(rank, world_size))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    logger.info("\n" + "="*60)
    logger.info("All tests completed!")
    logger.info("="*60)


if __name__ == "__main__":
    main()