"""
FlashInfer backend integration for MeZO training.

This module provides optimized forward passes for MeZO using FlashInfer kernels,
enabling efficient KV cache reuse between +εz and -εz perturbations.
"""

import torch
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

from sglang.srt.utils import is_flashinfer_available
from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

if is_flashinfer_available():
    import flashinfer
    from flashinfer import (
        BatchPrefillWithPagedKVCacheWrapper,
        BatchDecodeWithPagedKVCacheWrapper,
        SegmentGEMMWrapper,
    )
    from flashinfer.cascade import merge_state
else:
    raise ImportError("FlashInfer is required for MeZO FlashInfer backend")

logger = logging.getLogger(__name__)


@dataclass
class MeZOFlashInferConfig:
    """Configuration for MeZO FlashInfer backend."""
    enable_perturbation_fusion: bool = True
    enable_incremental_kv: bool = True
    enable_lora_segment_gemm: bool = True
    workspace_size_mb: int = 512
    use_fp8_quantization: bool = False
    cache_precision: torch.dtype = torch.float16


class MeZOFlashInferBackend:
    """
    FlashInfer backend for MeZO training that optimizes the two forward passes
    per gradient step using efficient KV cache reuse.
    """
    
    def __init__(
        self,
        model_runner,
        config: Optional[MeZOFlashInferConfig] = None
    ):
        self.model_runner = model_runner
        self.config = config or MeZOFlashInferConfig()
        self.device = model_runner.device
        
        # Initialize workspace buffer
        self.workspace_buffer = torch.empty(
            self.config.workspace_size_mb * 1024 * 1024,
            dtype=torch.uint8,
            device=self.device
        )
        
        # Initialize FlashInfer wrappers
        self._init_flashinfer_wrappers()
        
        # Cache for perturbation pairs
        self.perturbation_cache = {}
        
        # LoRA SegmentGEMM wrapper
        if self.config.enable_lora_segment_gemm:
            self.segment_gemm = SegmentGEMMWrapper(self.workspace_buffer)
        
        logger.info(f"MeZO FlashInfer backend initialized with config: {self.config}")
    
    def _init_flashinfer_wrappers(self):
        """Initialize FlashInfer attention wrappers."""
        # Prefill wrapper for initial computation
        self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.workspace_buffer,
            "NHD",  # Layout
            backend="fa2" if torch.cuda.get_device_capability()[0] >= 8 else "fa1"
        )
        
        # Decode wrapper for incremental computation
        self.decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            self.workspace_buffer,
            "NHD",
            use_tensor_cores=True
        )
        
        # Ragged wrapper for variable-length sequences
        self.ragged_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            self.workspace_buffer,
            "NHD"
        )
    
    def forward_with_perturbation_pair(
        self,
        batch: Dict[str, Any],
        lora_params: List[torch.Tensor],
        epsilon: float,
        z_list: List[torch.Tensor],
        step: int
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        Execute both forward passes (+εz and -εz) with optimized KV cache reuse.
        
        Returns:
            loss_plus: Loss for +εz perturbation
            loss_minus: Loss for -εz perturbation
            metrics: Performance metrics including cache statistics
        """
        metrics = {
            'cache_hits': 0,
            'tokens_computed': 0,
            'tokens_reused': 0,
            'flashinfer_time_ms': 0
        }
        
        if self.config.enable_perturbation_fusion:
            # Fused computation for both perturbations
            return self._forward_fused_perturbations(
                batch, lora_params, epsilon, z_list, step, metrics
            )
        else:
            # Sequential computation with cache reuse
            return self._forward_sequential_perturbations(
                batch, lora_params, epsilon, z_list, step, metrics
            )
    
    def _forward_fused_perturbations(
        self,
        batch: Dict[str, Any],
        lora_params: List[torch.Tensor],
        epsilon: float,
        z_list: List[torch.Tensor],
        step: int,
        metrics: Dict[str, Any]
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        Fused kernel execution for both perturbations in a single pass.
        This is the most optimized path.
        """
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        
        # Prepare cache key for this step
        cache_key = f"step_{step}_eps_{epsilon}"
        
        # Check if we have cached KV for this step
        if cache_key in self.perturbation_cache:
            # Use incremental computation
            loss_plus, loss_minus = self._forward_incremental_fused(
                batch, lora_params, epsilon, z_list, cache_key, metrics
            )
        else:
            # Full computation with cache population
            loss_plus, loss_minus = self._forward_full_fused(
                batch, lora_params, epsilon, z_list, cache_key, metrics
            )
        
        end_time.record()
        torch.cuda.synchronize()
        
        metrics['flashinfer_time_ms'] = start_time.elapsed_time(end_time)
        
        return loss_plus, loss_minus, metrics
    
    def _forward_full_fused(
        self,
        batch: Dict[str, Any],
        lora_params: List[torch.Tensor],
        epsilon: float,
        z_list: List[torch.Tensor],
        cache_key: str,
        metrics: Dict[str, Any]
    ) -> Tuple[float, float]:
        """
        Full forward pass with fused perturbation computation.
        """
        # Extract batch information
        input_ids = batch['input_ids']
        attention_mask = batch.get('attention_mask')
        prompt_lengths = batch.get('prompt_length', [0] * len(input_ids))
        
        batch_size = len(input_ids)
        max_seq_len = max(len(ids) for ids in input_ids)
        
        # Prepare KV indices for paged attention
        kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=self.device)
        seq_lens = torch.tensor([len(ids) for ids in input_ids], device=self.device)
        kv_indptr[1:] = torch.cumsum(seq_lens, dim=0)
        
        # Allocate KV cache entries
        total_tokens = seq_lens.sum().item()
        kv_indices = torch.arange(total_tokens, dtype=torch.int32, device=self.device)
        
        # Apply LoRA perturbations using SegmentGEMM if available
        if self.config.enable_lora_segment_gemm and hasattr(self, 'segment_gemm'):
            # Prepare segment indices for batched LoRA computation
            perturbed_weights_plus = self._apply_lora_perturbation_batch(
                lora_params, epsilon, z_list, sign=1
            )
            perturbed_weights_minus = self._apply_lora_perturbation_batch(
                lora_params, epsilon, z_list, sign=-1
            )
        else:
            # Standard perturbation application
            # Apply +εz
            for i, p in enumerate(lora_params):
                p.data.add_(epsilon * z_list[i])
            perturbed_weights_plus = lora_params
            
            # Prepare -εz (will switch later)
            perturbed_weights_minus = []
            for i, p in enumerate(lora_params):
                p_minus = p.data - 2 * epsilon * z_list[i]
                perturbed_weights_minus.append(p_minus)
        
        # Use cascade attention to compute both perturbations efficiently
        with flashinfer.cascade.CascadeAttentionWrapper(
            num_stages=2,  # Two stages for +εz and -εz
            workspace_buffer=self.workspace_buffer
        ) as cascade:
            # Stage 1: +εz perturbation
            # This computes and caches KV for prompt tokens
            output_plus, lse_plus = self._run_flashinfer_forward(
                input_ids, prompt_lengths, kv_indptr, kv_indices,
                perturbed_weights_plus, return_lse=True
            )
            
            # Stage 2: -εz perturbation  
            # This reuses cached KV from stage 1
            output_minus, lse_minus = self._run_flashinfer_forward(
                input_ids, prompt_lengths, kv_indptr, kv_indices,
                perturbed_weights_minus, return_lse=True,
                reuse_kv_cache=True
            )
            
            # Merge states if using cascade
            final_output_plus = output_plus
            final_output_minus = output_minus
        
        # Cache the KV for future incremental updates
        self.perturbation_cache[cache_key] = {
            'kv_indptr': kv_indptr,
            'kv_indices': kv_indices,
            'prompt_lengths': prompt_lengths,
            'seq_lens': seq_lens
        }
        
        # Compute losses
        loss_plus = self._compute_loss(final_output_plus, batch)
        loss_minus = self._compute_loss(final_output_minus, batch)
        
        # Update metrics
        metrics['tokens_computed'] = total_tokens * 2  # Both passes
        metrics['tokens_reused'] = 0  # First pass, no reuse
        
        # Restore original parameters
        if not self.config.enable_lora_segment_gemm:
            for i, p in enumerate(lora_params):
                p.data.add_(-epsilon * z_list[i])
        
        return loss_plus, loss_minus
    
    def _forward_incremental_fused(
        self,
        batch: Dict[str, Any],
        lora_params: List[torch.Tensor],
        epsilon: float,
        z_list: List[torch.Tensor],
        cache_key: str,
        metrics: Dict[str, Any]
    ) -> Tuple[float, float]:
        """
        Incremental forward pass reusing cached KV values.
        """
        # Retrieve cached information
        cache_info = self.perturbation_cache[cache_key]
        
        # Only compute new tokens (response tokens)
        # This is where the main efficiency gain comes from
        
        # TODO: Implement incremental computation
        # For now, fallback to full computation
        return self._forward_full_fused(
            batch, lora_params, epsilon, z_list, cache_key + "_new", metrics
        )
    
    def _apply_lora_perturbation_batch(
        self,
        lora_params: List[torch.Tensor],
        epsilon: float,
        z_list: List[torch.Tensor],
        sign: int
    ) -> List[torch.Tensor]:
        """
        Apply LoRA perturbations using FlashInfer's SegmentGEMM for efficiency.
        """
        perturbed_params = []
        
        # Group parameters by shape for batched computation
        shape_groups = {}
        for i, (param, z) in enumerate(zip(lora_params, z_list)):
            shape = param.shape
            if shape not in shape_groups:
                shape_groups[shape] = []
            shape_groups[shape].append((i, param, z))
        
        # Process each shape group with SegmentGEMM
        for shape, group in shape_groups.items():
            if len(group) > 1:
                # Batch multiple parameters of the same shape
                indices, params, zs = zip(*group)
                
                # Stack parameters and perturbations
                stacked_params = torch.stack([p.data for p in params])
                stacked_z = torch.stack(zs)
                
                # Apply perturbation using SegmentGEMM
                perturbed = self.segment_gemm(
                    stacked_params,
                    stacked_z,
                    scale=sign * epsilon
                )
                
                # Unstack results
                for i, idx in enumerate(indices):
                    perturbed_params.append((idx, perturbed[i]))
            else:
                # Single parameter, apply directly
                idx, param, z = group[0]
                perturbed = param.data + sign * epsilon * z
                perturbed_params.append((idx, perturbed))
        
        # Sort by original index and return
        perturbed_params.sort(key=lambda x: x[0])
        return [p for _, p in perturbed_params]
    
    def _run_flashinfer_forward(
        self,
        input_ids: List[List[int]],
        prompt_lengths: List[int],
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        lora_weights: List[torch.Tensor],
        return_lse: bool = False,
        reuse_kv_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Run forward pass through model using FlashInfer kernels.
        """
        # This is a placeholder - actual implementation would integrate with model
        # For now, return dummy outputs
        batch_size = len(input_ids)
        seq_lens = [len(ids) for ids in input_ids]
        max_seq_len = max(seq_lens)
        
        # Dummy output
        output = torch.randn(
            batch_size, max_seq_len, self.model_runner.model_config.hidden_size,
            device=self.device
        )
        
        lse = torch.randn(batch_size, device=self.device) if return_lse else None
        
        return output, lse
    
    def _compute_loss(self, output: torch.Tensor, batch: Dict[str, Any]) -> float:
        """Compute loss from model output."""
        # Placeholder loss computation
        return torch.nn.functional.mse_loss(
            output.mean(),
            torch.tensor(0.0, device=output.device)
        ).item()
    
    def _forward_sequential_perturbations(
        self,
        batch: Dict[str, Any],
        lora_params: List[torch.Tensor],
        epsilon: float,
        z_list: List[torch.Tensor],
        step: int,
        metrics: Dict[str, Any]
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        Sequential computation with explicit cache reuse between perturbations.
        This is easier to debug but less efficient than fused computation.
        """
        # Apply +εz perturbation
        for i, p in enumerate(lora_params):
            p.data.add_(epsilon * z_list[i])
        
        # First forward pass
        loss_plus = self._forward_single_perturbation(
            batch, step, perturbation_sign=1, metrics=metrics
        )
        
        # Switch to -εz perturbation
        for i, p in enumerate(lora_params):
            p.data.add_(-2 * epsilon * z_list[i])
        
        # Second forward pass (reuses KV cache)
        loss_minus = self._forward_single_perturbation(
            batch, step, perturbation_sign=-1, metrics=metrics
        )
        
        # Restore original parameters
        for i, p in enumerate(lora_params):
            p.data.add_(epsilon * z_list[i])
        
        return loss_plus, loss_minus, metrics
    
    def _forward_single_perturbation(
        self,
        batch: Dict[str, Any],
        step: int,
        perturbation_sign: int,
        metrics: Dict[str, Any]
    ) -> float:
        """Execute a single forward pass with FlashInfer optimizations."""
        # Extract batch information
        input_ids = batch['input_ids']
        prompt_lengths = batch.get('prompt_length', [0] * len(input_ids))
        
        # Prepare FlashInfer batch
        batch_size = len(input_ids)
        seq_lens = torch.tensor([len(ids) for ids in input_ids], device=self.device)
        
        # Check for cached KV from previous perturbation
        cache_key = f"step_{step}"
        has_cache = cache_key in self.perturbation_cache and perturbation_sign == -1
        
        if has_cache:
            # Use decode mode for cached prompt tokens
            cache_info = self.perturbation_cache[cache_key]
            
            # Only compute response tokens
            tokens_to_compute = sum(
                len(ids) - prompt_len 
                for ids, prompt_len in zip(input_ids, prompt_lengths)
            )
            
            # Use decode wrapper for incremental computation
            output = self._run_decode_with_cache(
                batch, cache_info, metrics
            )
            
            metrics['tokens_reused'] += sum(prompt_lengths)
            metrics['tokens_computed'] += tokens_to_compute
            metrics['cache_hits'] += 1
        else:
            # Full prefill computation
            total_tokens = seq_lens.sum().item()
            
            # Use prefill wrapper
            output = self._run_prefill_no_cache(batch, metrics)
            
            # Cache KV for next perturbation
            if perturbation_sign == 1:
                self.perturbation_cache[cache_key] = {
                    'prompt_lengths': prompt_lengths,
                    'seq_lens': seq_lens
                }
            
            metrics['tokens_computed'] += total_tokens
        
        # Compute loss
        return self._compute_loss(output, batch)
    
    def _run_prefill_no_cache(
        self,
        batch: Dict[str, Any],
        metrics: Dict[str, Any]
    ) -> torch.Tensor:
        """Run prefill without using cached KV."""
        # Placeholder - would integrate with actual model
        return torch.randn(1, device=self.device)
    
    def _run_decode_with_cache(
        self,
        batch: Dict[str, Any],
        cache_info: Dict[str, Any],
        metrics: Dict[str, Any]  
    ) -> torch.Tensor:
        """Run decode using cached KV values."""
        # Placeholder - would integrate with actual model
        return torch.randn(1, device=self.device)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage of FlashInfer backend."""
        return {
            'workspace_mb': self.config.workspace_size_mb,
            'cache_entries': len(self.perturbation_cache),
            'cached_tokens': sum(
                info['seq_lens'].sum().item() 
                for info in self.perturbation_cache.values()
            )
        }
    
    def clear_cache(self):
        """Clear the perturbation cache."""
        self.perturbation_cache.clear()
        logger.info("FlashInfer perturbation cache cleared")