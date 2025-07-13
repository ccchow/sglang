"""Flexible parallel state management that allows reusing existing initialization."""

import logging
from typing import Optional

import torch
import torch.distributed as dist

from sglang.srt.distributed.parallel_state import (
    _TP,
    _PP,
    _WORLD,
    model_parallel_is_initialized,
    get_tensor_model_parallel_world_size,
    get_pp_group,
    init_distributed_environment as _original_init_distributed_environment,
    initialize_model_parallel as _original_initialize_model_parallel,
)

logger = logging.getLogger(__name__)


def flexible_init_distributed_environment(
    backend: str,
    world_size: int,
    rank: int,
    local_rank: int,
    distributed_init_method: str,
    timeout: int = 1800,
) -> bool:
    """
    Initialize distributed environment if not already initialized.
    
    Returns:
        True if initialization was performed, False if already initialized.
    """
    global _WORLD
    
    # Check if already initialized
    if dist.is_initialized() and _WORLD is not None:
        # Verify the existing setup matches requirements
        existing_world_size = dist.get_world_size()
        existing_rank = dist.get_rank()
        
        if existing_world_size != world_size or existing_rank != rank:
            raise ValueError(
                f"Distributed already initialized with different configuration. "
                f"Existing: world_size={existing_world_size}, rank={existing_rank}. "
                f"Requested: world_size={world_size}, rank={rank}"
            )
        
        logger.info("Distributed environment already initialized, reusing existing setup")
        return False
    
    # Initialize using original function
    _original_init_distributed_environment(
        backend=backend,
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method=distributed_init_method,
        timeout=timeout,
    )
    return True


def flexible_initialize_model_parallel(
    tensor_model_parallel_size: int,
    pipeline_model_parallel_size: int,
    backend: Optional[str] = None,
) -> bool:
    """
    Initialize model parallel groups if not already initialized.
    
    Returns:
        True if initialization was performed, False if already initialized.
    """
    global _TP, _PP
    
    # Check if already initialized
    if _TP is not None and _PP is not None:
        # Verify the existing groups match requirements
        existing_tp_size = get_tensor_model_parallel_world_size()
        existing_pp_size = get_pp_group().world_size
        
        if (existing_tp_size != tensor_model_parallel_size or 
            existing_pp_size != pipeline_model_parallel_size):
            raise ValueError(
                f"Model parallel already initialized with different sizes. "
                f"Existing: tp={existing_tp_size}, pp={existing_pp_size}. "
                f"Requested: tp={tensor_model_parallel_size}, pp={pipeline_model_parallel_size}"
            )
        
        logger.info(
            f"Model parallel groups already initialized with tp={existing_tp_size}, "
            f"pp={existing_pp_size}, reusing existing setup"
        )
        return False
    
    # Initialize using original function
    _original_initialize_model_parallel(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        backend=backend,
    )
    return True


def monkey_patch_for_flexible_init():
    """Monkey patch the parallel state module to use flexible initialization."""
    import sglang.srt.distributed.parallel_state as ps
    
    # Store originals if not already stored
    if not hasattr(ps, '_original_init_distributed_environment'):
        ps._original_init_distributed_environment = ps.init_distributed_environment
        ps._original_initialize_model_parallel = ps.initialize_model_parallel
    
    # Replace with flexible versions
    ps.init_distributed_environment = flexible_init_distributed_environment
    ps.initialize_model_parallel = flexible_initialize_model_parallel
    
    logger.info("Applied flexible initialization monkey patch")


def remove_flexible_init_patch():
    """Remove the flexible initialization monkey patch."""
    import sglang.srt.distributed.parallel_state as ps
    
    if hasattr(ps, '_original_init_distributed_environment'):
        ps.init_distributed_environment = ps._original_init_distributed_environment
        ps.initialize_model_parallel = ps._original_initialize_model_parallel
        delattr(ps, '_original_init_distributed_environment')
        delattr(ps, '_original_initialize_model_parallel')
        
        logger.info("Removed flexible initialization monkey patch")