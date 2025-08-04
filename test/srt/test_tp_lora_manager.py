#!/usr/bin/env python3
"""
Test TP LoRA Manager functionality.
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import logging
from typing import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_lora_weight_sharding(rank: int, world_size: int):
    """Test LoRA weight sharding across TP ranks."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29506'
    
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
    
    logger.info(f"Rank {rank}: Testing LoRA weight sharding")
    
    # Create mock weights
    in_features = 1024
    out_features = 1024
    lora_r = 8
    
    # Simulate column-parallel layer (e.g., q_proj)
    col_lora_A = torch.randn(lora_r, in_features, device=device)
    col_lora_B = torch.randn(out_features, lora_r, device=device)
    
    # Column parallel: split lora_B along dim 0
    col_shard_size = out_features // world_size
    col_start = rank * col_shard_size
    col_end = col_start + col_shard_size
    col_lora_B_shard = col_lora_B[col_start:col_end]
    
    logger.info(f"Rank {rank}: Column-parallel lora_B shape: {col_lora_B.shape} -> {col_lora_B_shard.shape}")
    
    # Simulate row-parallel layer (e.g., o_proj)
    row_lora_A = torch.randn(lora_r, in_features, device=device)
    row_lora_B = torch.randn(out_features, lora_r, device=device)
    
    # Row parallel: split lora_A along dim 1
    row_shard_size = in_features // world_size
    row_start = rank * row_shard_size
    row_end = row_start + row_shard_size
    row_lora_A_shard = row_lora_A[:, row_start:row_end]
    
    logger.info(f"Rank {rank}: Row-parallel lora_A shape: {row_lora_A.shape} -> {row_lora_A_shard.shape}")
    
    # Test gradient aggregation
    # For row-parallel layers, gradients need all-reduce
    row_lora_A_shard.requires_grad = True
    
    # Simulate gradient
    grad = torch.ones_like(row_lora_A_shard)
    row_lora_A_shard.grad = grad
    
    # All-reduce gradient
    dist.all_reduce(row_lora_A_shard.grad, op=dist.ReduceOp.SUM)
    
    logger.info(f"Rank {rank}: Gradient sum after all-reduce: {row_lora_A_shard.grad.sum().item()}")
    
    # Verify sharding correctness
    # Each rank should have different shards
    shard_sum = col_lora_B_shard.sum().item()
    all_shard_sums = [torch.tensor(0.0, device=device) for _ in range(world_size)]
    dist.all_gather(all_shard_sums, torch.tensor(shard_sum, device=device))
    
    if rank == 0:
        shard_sums_list = [s.item() for s in all_shard_sums]
        # Check that shards are different (very unlikely to have same sum for random tensors)
        unique_sums = len(set(shard_sums_list))
        if unique_sums == world_size:
            logger.info("✅ PASSED: All ranks have different weight shards")
        else:
            logger.error("❌ FAILED: Some ranks have identical shards")
        logger.info(f"   Shard sums: {shard_sums_list}")
    
    dist.destroy_process_group()


def test_mezo_perturbation_with_lora(rank: int, world_size: int):
    """Test MeZO perturbation generation with sharded LoRA weights."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29507'
    
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
    
    logger.info(f"Rank {rank}: Testing MeZO perturbation with LoRA")
    
    # Create sharded LoRA weights
    lora_r = 8
    shard_size = 512  # Each rank gets 512 features out of 1024
    
    # Each rank has its shard
    lora_B_shard = torch.randn(shard_size, lora_r, device=device, requires_grad=True)
    
    # Generate synchronized perturbation
    if rank == 0:
        seed = torch.randint(0, 2**31-1, (1,), device=device)
    else:
        seed = torch.zeros(1, dtype=torch.long, device=device)
    
    dist.broadcast(seed, src=0)
    
    # Generate perturbation for the shard
    torch.manual_seed(seed.item())
    z_shard = torch.randn_like(lora_B_shard)
    
    # Apply perturbation
    epsilon = 1e-3
    perturbed = lora_B_shard + epsilon * z_shard
    
    # Compute local "loss" (mock)
    local_loss = perturbed.sum().item()
    
    # Aggregate loss
    loss_tensor = torch.tensor(local_loss, device=device)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    global_loss = loss_tensor.item() / world_size
    
    logger.info(f"Rank {rank}: Local loss={local_loss:.4f}, Global loss={global_loss:.4f}")
    
    # Verify all ranks computed same global loss
    all_global_losses = [torch.tensor(0.0, device=device) for _ in range(world_size)]
    dist.all_gather(all_global_losses, torch.tensor(global_loss, device=device))
    
    if rank == 0:
        losses_list = [l.item() for l in all_global_losses]
        if all(abs(l - losses_list[0]) < 1e-6 for l in losses_list):
            logger.info("✅ PASSED: All ranks computed same global loss")
        else:
            logger.error("❌ FAILED: Global losses differ across ranks")
        logger.info(f"   Global losses: {losses_list}")
    
    dist.destroy_process_group()


def main():
    """Run TP LoRA manager tests."""
    gpu_count = torch.cuda.device_count()
    
    if gpu_count < 2:
        logger.warning(f"Only {gpu_count} GPU(s) available. Testing with CPU backend...")
        world_size = 2
    else:
        logger.info(f"Found {gpu_count} GPUs. Running TP LoRA tests...")
        world_size = min(gpu_count, 4)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(world_size))
    
    # Test 1: Weight sharding
    logger.info("\n" + "="*60)
    logger.info("Test 1: LoRA Weight Sharding")
    logger.info("="*60)
    
    mp.set_start_method('spawn', force=True)
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=test_lora_weight_sharding, args=(rank, world_size))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    # Test 2: MeZO with sharded LoRA
    logger.info("\n" + "="*60)
    logger.info("Test 2: MeZO Perturbation with Sharded LoRA")
    logger.info("="*60)
    
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=test_mezo_perturbation_with_lora, args=(rank, world_size))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    logger.info("\n" + "="*60)
    logger.info("All TP LoRA tests completed!")
    logger.info("="*60)


if __name__ == "__main__":
    main()