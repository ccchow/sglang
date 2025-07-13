"""Utilities for distributed initialization and management."""

import os
import torch
import torch.distributed as dist
from typing import Optional

from sglang.srt.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
    model_parallel_is_initialized,
    get_world_group,
    destroy_model_parallel,
    destroy_distributed_environment,
)


def is_distributed_initialized() -> bool:
    """Check if torch distributed is initialized."""
    return dist.is_initialized()


def is_sglang_parallel_initialized() -> bool:
    """Check if SGLang parallel groups are initialized."""
    return model_parallel_is_initialized()


def safe_init_distributed_environment(
    backend: str = "nccl",
    world_size: int = 1,
    rank: int = 0,
    local_rank: int = 0,
    distributed_init_method: str = "tcp://127.0.0.1:29500",
    timeout: int = 1800,
) -> bool:
    """
    Safely initialize distributed environment.
    
    Returns:
        True if initialization was performed, False if already initialized.
    """
    if is_distributed_initialized():
        # Already initialized
        return False
    
    init_distributed_environment(
        backend=backend,
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method=distributed_init_method,
        timeout=timeout,
    )
    return True


def safe_initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
) -> bool:
    """
    Safely initialize model parallel groups.
    
    Returns:
        True if initialization was performed, False if already initialized.
    """
    if model_parallel_is_initialized():
        # Already initialized - verify sizes match
        from sglang.srt.distributed.parallel_state import (
            get_tensor_model_parallel_world_size,
            get_pp_group,
        )
        
        current_tp_size = get_tensor_model_parallel_world_size()
        current_pp_size = get_pp_group().world_size
        
        if (current_tp_size != tensor_model_parallel_size or 
            current_pp_size != pipeline_model_parallel_size):
            raise ValueError(
                f"Model parallel already initialized with different sizes. "
                f"Current: tp={current_tp_size}, pp={current_pp_size}. "
                f"Requested: tp={tensor_model_parallel_size}, pp={pipeline_model_parallel_size}"
            )
        return False
    
    initialize_model_parallel(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
    )
    return True


def cleanup_distributed() -> None:
    """Clean up all distributed resources."""
    # Destroy model parallel groups first
    if model_parallel_is_initialized():
        destroy_model_parallel()
    
    # Then destroy distributed environment
    if is_distributed_initialized():
        destroy_distributed_environment()
        if dist.is_initialized():
            dist.destroy_process_group()


def setup_environment_for_testing(
    tp_size: int = 1,
    pp_size: int = 1,
    rank: int = 0,
    local_rank: int = 0,
    master_port: str = "29500",
) -> None:
    """
    Set up environment variables for distributed testing.
    
    This should be called before creating ModelRunner to ensure
    proper environment setup.
    """
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(tp_size * pp_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = master_port
    
    # Optional but recommended
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ModelRunnerDistributedConfig:
    """Configuration for distributed initialization in ModelRunner."""
    
    def __init__(
        self,
        skip_distributed_init: bool = False,
        reuse_existing_parallel: bool = True,
    ):
        """
        Args:
            skip_distributed_init: If True, skip all distributed initialization.
                Useful for testing or when using pre-initialized distributed.
            reuse_existing_parallel: If True, reuse existing parallel groups if
                they match the requested configuration.
        """
        self.skip_distributed_init = skip_distributed_init
        self.reuse_existing_parallel = reuse_existing_parallel