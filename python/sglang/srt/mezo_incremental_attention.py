"""
Incremental attention mechanism for MeZO's prompt KV reuse optimization.

This module provides utilities to enable efficient incremental forward passes
that reuse cached KV values for prompt tokens across MeZO perturbations.
"""

import torch
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch, ModelWorkerBatch

# Robust import for ForwardMode with fallback
try:
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
except ImportError:
    try:
        from sglang.srt.model_executor.model_runner import ForwardMode
    except ImportError:
        # Define a fallback if neither import works
        from enum import Enum
        class ForwardMode(Enum):
            DECODE = "decode"
            EXTEND = "extend"
            MIXED = "mixed"

logger = logging.getLogger(__name__)


@dataclass
class IncrementalForwardConfig:
    """Configuration for incremental forward passes."""
    enable_prompt_caching: bool = True
    cache_prompt_only: bool = True
    merge_attention_states: bool = True
    use_chunked_prefill: bool = False
    chunk_size: int = 512


class MeZOIncrementalAttention:
    """
    Manages incremental attention computation for MeZO training.
    
    Key features:
    1. Splits attention computation into cached (prompt) and new (response) portions
    2. Configures SGLang's attention backends for optimal cache reuse
    3. Handles position embeddings correctly for incremental computation
    4. Provides utilities for merging attention states
    """
    
    def __init__(self, config: IncrementalForwardConfig = None):
        self.config = config or IncrementalForwardConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def prepare_incremental_batch(
        self,
        schedule_batch: ScheduleBatch,
        cache_metadata: Dict[str, Any]
    ) -> ScheduleBatch:
        """
        Prepare a batch for incremental forward pass with KV cache reuse.
        
        Args:
            schedule_batch: The base schedule batch
            cache_metadata: Metadata about cached portions
            
        Returns:
            Modified schedule batch configured for incremental computation
        """
        # Configure each request for incremental computation
        for i, req in enumerate(schedule_batch.reqs):
            req_metadata = cache_metadata.get(req.rid, {})
            
            if req_metadata.get('cache_hit', False):
                # Set up for incremental computation
                prefix_len = req_metadata.get('prefix_len', 0)
                
                # Mark the cached prefix length
                req.prefix_len = prefix_len
                
                # If we have cached KV indices, set them
                if req_metadata.get('cache_entry'):
                    cache_entry = req_metadata['cache_entry']
                    if cache_entry.kv_indices is not None:
                        req.prefix_indices = cache_entry.kv_indices
                
                self.logger.debug(
                    f"Request {req.rid}: Using cached prefix of {prefix_len} tokens"
                )
        
        # Ensure batch is configured for extend mode with prefix support
        schedule_batch.forward_mode = ForwardMode.EXTEND
        
        return schedule_batch
    
    def configure_model_worker_batch(
        self,
        model_batch: ModelWorkerBatch,
        cache_metadata: Dict[str, Any]
    ) -> ModelWorkerBatch:
        """
        Configure ModelWorkerBatch for efficient incremental computation.
        
        Args:
            model_batch: The base model worker batch
            cache_metadata: Metadata about cached portions
            
        Returns:
            Configured model worker batch
        """
        # Extract prefix lengths from metadata
        prefix_lens = []
        for i in range(model_batch.batch_size):
            # Find corresponding metadata
            req_metadata = None
            for rid, metadata in cache_metadata.items():
                if metadata.get('batch_idx') == i:
                    req_metadata = metadata
                    break
            
            if req_metadata and req_metadata.get('cache_hit'):
                prefix_lens.append(req_metadata.get('prefix_len', 0))
            else:
                prefix_lens.append(0)
        
        # Update model batch with prefix information
        model_batch.prefix_lens = prefix_lens
        
        # For extend mode, we need to set extend-specific attributes
        if hasattr(model_batch, 'extend_seq_lens') and hasattr(model_batch, 'extend_prefix_lens'):
            # Configure for incremental extend
            model_batch.extend_prefix_lens = prefix_lens
            
            # Extend lengths are the new tokens to process
            extend_lens = []
            for i in range(model_batch.batch_size):
                total_len = model_batch.seq_lens[i] if hasattr(model_batch, 'seq_lens') else 0
                prefix_len = prefix_lens[i]
                extend_lens.append(total_len - prefix_len)
            
            model_batch.extend_lens = extend_lens
        
        self.logger.debug(
            f"Configured incremental batch: prefix_lens={prefix_lens}, "
            f"forward_mode={model_batch.forward_mode}"
        )
        
        return model_batch
    
    def create_position_ids_incremental(
        self,
        seq_lens: List[int],
        prefix_lens: List[int],
        device: torch.device
    ) -> torch.Tensor:
        """
        Create position IDs for incremental attention computation.
        
        For cached prefixes, we don't need position IDs as they're already computed.
        We only need positions for the new tokens.
        
        Args:
            seq_lens: Total sequence lengths
            prefix_lens: Lengths of cached prefixes
            device: Target device
            
        Returns:
            Position IDs tensor for new tokens only
        """
        position_ids = []
        
        for seq_len, prefix_len in zip(seq_lens, prefix_lens):
            # Only create positions for new tokens
            new_len = seq_len - prefix_len
            if new_len > 0:
                # Positions start from prefix_len
                positions = torch.arange(
                    prefix_len, seq_len, dtype=torch.long, device=device
                )
                position_ids.append(positions)
        
        if position_ids:
            return torch.cat(position_ids, dim=0)
        else:
            return torch.empty(0, dtype=torch.long, device=device)
    
    def merge_attention_outputs(
        self,
        cached_output: Optional[torch.Tensor],
        new_output: torch.Tensor,
        prefix_lens: List[int],
        seq_lens: List[int]
    ) -> torch.Tensor:
        """
        Merge attention outputs from cached and newly computed portions.
        
        Args:
            cached_output: Output for cached prefix (if separate)
            new_output: Output for newly computed tokens
            prefix_lens: Lengths of cached prefixes
            seq_lens: Total sequence lengths
            
        Returns:
            Merged attention output
        """
        if cached_output is None or not self.config.merge_attention_states:
            # No merging needed
            return new_output
        
        # In SGLang's architecture, the attention backend handles merging internally
        # This is a placeholder for custom merging logic if needed
        return new_output
    
    def compute_incremental_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        prefix_lens: List[int],
        compute_on_cached: bool = False
    ) -> torch.Tensor:
        """
        Compute loss considering the incremental computation structure.
        
        Args:
            logits: Model output logits
            targets: Target token IDs
            prefix_lens: Lengths of cached prefixes
            compute_on_cached: Whether to include cached portion in loss
            
        Returns:
            Computed loss value
        """
        if compute_on_cached:
            # Standard loss computation over all tokens
            return torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )
        else:
            # Only compute loss on non-cached tokens
            loss_mask = torch.ones_like(targets, dtype=torch.bool)
            
            # Mask out cached portions
            offset = 0
            for i, prefix_len in enumerate(prefix_lens):
                if prefix_len > 0:
                    # Find the range for this sequence
                    seq_start = offset
                    seq_end = offset + targets.size(1)  # Assuming fixed seq length
                    
                    # Mask cached portion
                    loss_mask[seq_start:seq_start + prefix_len] = False
                    
                    offset = seq_end
            
            # Compute masked loss
            if loss_mask.any():
                masked_logits = logits[loss_mask]
                masked_targets = targets[loss_mask]
                
                return torch.nn.functional.cross_entropy(
                    masked_logits,
                    masked_targets
                )
            else:
                # No non-cached tokens
                return torch.tensor(0.0, device=logits.device)
    
    def analyze_cache_efficiency(
        self,
        cache_metadata: Dict[str, Any],
        model_config: Any
    ) -> Dict[str, float]:
        """
        Analyze the efficiency of incremental attention with caching.
        
        Args:
            cache_metadata: Metadata about cached portions
            model_config: Model configuration
            
        Returns:
            Efficiency metrics
        """
        total_sequences = len(cache_metadata)
        sequences_with_cache = sum(
            1 for m in cache_metadata.values() if m.get('cache_hit', False)
        )
        
        total_tokens = sum(
            m.get('total_length', 0) for m in cache_metadata.values()
        )
        cached_tokens = sum(
            m.get('prefix_len', 0) for m in cache_metadata.values() 
            if m.get('cache_hit', False)
        )
        
        # Estimate compute savings
        # Attention is O(n^2) so caching prefix of length p saves p^2 computation
        compute_saved = 0
        compute_total = 0
        
        for metadata in cache_metadata.values():
            seq_len = metadata.get('total_length', 0)
            prefix_len = metadata.get('prefix_len', 0) if metadata.get('cache_hit') else 0
            
            # Full computation cost
            compute_total += seq_len * seq_len
            
            # Saved computation (don't need to compute prefix attention)
            compute_saved += prefix_len * prefix_len
        
        # Memory bandwidth savings
        # Each token requires loading K, V from all layers
        num_layers = model_config.num_hidden_layers
        hidden_size = model_config.hidden_size
        bytes_per_token = 2 * hidden_size * num_layers * 2  # K, V, fp16
        
        memory_bandwidth_saved = cached_tokens * bytes_per_token
        
        return {
            'sequences_with_cache_ratio': sequences_with_cache / total_sequences if total_sequences > 0 else 0,
            'cached_token_ratio': cached_tokens / total_tokens if total_tokens > 0 else 0,
            'compute_savings_ratio': compute_saved / compute_total if compute_total > 0 else 0,
            'memory_bandwidth_saved_gb': memory_bandwidth_saved / (1024**3),
            'estimated_speedup': 1 / (1 - compute_saved / compute_total * 0.7) if compute_total > 0 else 1,
        }


def create_incremental_attention_manager(
    enable_prompt_caching: bool = True,
    cache_prompt_only: bool = True
) -> MeZOIncrementalAttention:
    """
    Factory function to create an incremental attention manager.
    
    Args:
        enable_prompt_caching: Whether to enable prompt caching
        cache_prompt_only: Whether to cache only prompts (not completions)
        
    Returns:
        Configured MeZOIncrementalAttention instance
    """
    config = IncrementalForwardConfig(
        enable_prompt_caching=enable_prompt_caching,
        cache_prompt_only=cache_prompt_only,
        merge_attention_states=True,
        use_chunked_prefill=False
    )
    
    return MeZOIncrementalAttention(config)