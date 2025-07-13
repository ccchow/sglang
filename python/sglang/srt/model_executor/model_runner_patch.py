"""Patch for ModelRunner to support flexible distributed initialization."""

import logging
from typing import Optional

from sglang.srt.distributed.parallel_state import (
    model_parallel_is_initialized,
    get_tensor_model_parallel_world_size,
    get_pp_group,
)
from sglang.srt.distributed.distributed_utils import (
    safe_init_distributed_environment,
    safe_initialize_model_parallel,
)

logger = logging.getLogger(__name__)


def init_torch_distributed_flexible(self, skip_if_initialized: bool = True):
    """
    Modified version of init_torch_distributed that can skip initialization
    if distributed is already set up.
    
    Args:
        skip_if_initialized: If True, skip initialization if already initialized.
    """
    logger.info("Init torch distributed begin (flexible mode).")
    
    # Device setup
    try:
        import torch
        torch.get_device_module(self.device).set_device(self.gpu_id)
    except Exception:
        logger.warning(
            f"Context: {self.device=} {self.gpu_id=} {self.tp_rank=} {self.tp_size=}"
        )
        raise
    
    # Determine backend
    if self.device == "cuda":
        backend = "nccl"
    elif self.device == "xpu":
        backend = "xccl"
    elif self.device == "hpu":
        backend = "hccl"
    elif self.device == "cpu":
        backend = "gloo"
    elif self.device == "npu":
        backend = "hccl"
    else:
        backend = "nccl"  # fallback
    
    # Get memory before initialization
    from sglang.srt.utils import get_available_gpu_memory
    before_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
    
    # Setup for custom all-reduce
    from sglang.srt.distributed.utils import (
        set_custom_all_reduce,
        set_mscclpp_all_reduce,
    )
    set_custom_all_reduce(not self.server_args.disable_custom_all_reduce)
    set_mscclpp_all_reduce(self.server_args.enable_mscclpp)
    
    # Determine init method
    if self.server_args.dist_init_addr:
        dist_init_method = f"tcp://{self.server_args.dist_init_addr}"
    else:
        dist_init_method = f"tcp://127.0.0.1:{self.dist_port}"
    
    if not self.is_draft_worker:
        # CPU-specific setup
        if self.device == "cpu":
            try:
                from sglang.srt.layers.attention.flashinfer_backend import (
                    _is_cpu_amx_available,
                )
                if _is_cpu_amx_available:
                    import torch
                    torch.ops.sgl_kernel.init_cpu_threads_env(self.local_omp_cpuid)
                    import os
                    os.environ["LOCAL_SIZE"] = str(self.tp_size)
                    torch.ops.sgl_kernel.initialize(self.tp_size, self.tp_rank)
                else:
                    logger.warning(
                        "CPU AMX backend not available, skipping optimization"
                    )
            except ImportError:
                logger.warning("Could not import CPU-specific optimizations")
        
        # Try to initialize distributed environment
        if skip_if_initialized:
            dist_initialized = safe_init_distributed_environment(
                backend=backend,
                world_size=self.tp_size * self.pp_size,
                rank=self.tp_size * self.pp_rank + self.tp_rank,
                local_rank=self.gpu_id,
                distributed_init_method=dist_init_method,
                timeout=self.server_args.dist_timeout,
            )
            if dist_initialized:
                logger.info("Initialized distributed environment")
            else:
                logger.info("Distributed environment already initialized, reusing")
            
            # Try to initialize model parallel
            mp_initialized = safe_initialize_model_parallel(
                tensor_model_parallel_size=self.tp_size,
                pipeline_model_parallel_size=self.pp_size,
            )
            if mp_initialized:
                logger.info("Initialized model parallel groups")
            else:
                logger.info("Model parallel groups already initialized, verifying...")
                # Verify the existing groups match our requirements
                if model_parallel_is_initialized():
                    current_tp_size = get_tensor_model_parallel_world_size()
                    current_pp_size = get_pp_group().world_size
                    if (current_tp_size != self.tp_size or 
                        current_pp_size != self.pp_size):
                        raise ValueError(
                            f"Existing model parallel groups don't match requirements. "
                            f"Current: tp={current_tp_size}, pp={current_pp_size}. "
                            f"Required: tp={self.tp_size}, pp={self.pp_size}"
                        )
        else:
            # Original initialization (always initializes)
            from sglang.srt.distributed.parallel_state import (
                init_distributed_environment,
                initialize_model_parallel,
            )
            init_distributed_environment(
                backend=backend,
                world_size=self.tp_size * self.pp_size,
                rank=self.tp_size * self.pp_rank + self.tp_rank,
                local_rank=self.gpu_id,
                distributed_init_method=dist_init_method,
                timeout=self.server_args.dist_timeout,
            )
            initialize_model_parallel(
                tensor_model_parallel_size=self.tp_size,
                pipeline_model_parallel_size=self.pp_size,
            )
        
        # Initialize DP attention
        from sglang.srt.layers.attention.flashinfer_backend import initialize_dp_attention
        initialize_dp_attention(
            enable_dp_attention=self.server_args.enable_dp_attention,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            dp_size=self.server_args.dp_size,
            moe_dense_tp_size=self.server_args.moe_dense_tp_size,
            pp_size=self.server_args.pp_size,
        )
    
    # Get groups and check memory
    from sglang.srt.distributed.parallel_state import (
        get_world_group,
        get_tp_group,
        get_attention_tp_group,
    )
    from sglang.srt.utils import get_bool_env_var
    
    min_per_gpu_memory = get_available_gpu_memory(
        self.device,
        self.gpu_id,
        distributed=get_world_group().world_size > 1,
        cpu_group=get_world_group().cpu_group,
    )
    self.tp_group = get_tp_group()
    self.attention_tp_group = get_attention_tp_group()
    
    # Check memory balance for tensor parallelism
    local_gpu_memory = get_available_gpu_memory(self.device, self.gpu_id)
    if self.tp_size > 1:
        if min_per_gpu_memory < local_gpu_memory * 0.9:
            if get_bool_env_var("SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK"):
                logger.warning(
                    "Memory capacity is unbalanced. Some GPUs may be occupied. "
                    f"{min_per_gpu_memory=}, {local_gpu_memory=}"
                )
            else:
                raise ValueError(
                    "Memory capacity is unbalanced. Some GPUs may be occupied. "
                    f"{min_per_gpu_memory=}, {local_gpu_memory=}"
                )
    
    logger.info(
        f"Init torch distributed ends. mem usage="
        f"{(before_avail_memory - local_gpu_memory):.2f} GB"
    )
    return min_per_gpu_memory


def patch_model_runner_for_flexible_init():
    """Apply the flexible initialization patch to ModelRunner."""
    from sglang.srt.model_executor.model_runner import ModelRunner
    
    # Store original method
    if not hasattr(ModelRunner, '_original_init_torch_distributed'):
        ModelRunner._original_init_torch_distributed = ModelRunner.init_torch_distributed
    
    # Replace with flexible version
    ModelRunner.init_torch_distributed = init_torch_distributed_flexible
    
    logger.info("Applied flexible distributed initialization patch to ModelRunner")


def unpatch_model_runner():
    """Remove the flexible initialization patch from ModelRunner."""
    from sglang.srt.model_executor.model_runner import ModelRunner
    
    if hasattr(ModelRunner, '_original_init_torch_distributed'):
        ModelRunner.init_torch_distributed = ModelRunner._original_init_torch_distributed
        delattr(ModelRunner, '_original_init_torch_distributed')
        logger.info("Removed flexible distributed initialization patch from ModelRunner")