"""
RadixAttention optimization for MeZO forward passes.

This module implements KV cache optimization strategies for MeZO's two forward passes
by leveraging SGLang's RadixAttention to reuse cached computations.

Enhanced with:
- Prompt-aware KV cache management
- LoRA-aware cache invalidation
- Incremental forward pass support
"""

import torch
import torch.nn as nn
import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
import hashlib

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.mezo_kv_cache_manager import MeZOKVCacheManager, CacheEntry

logger = logging.getLogger(__name__)


@dataclass
class MeZOCacheStats:
    """Statistics for MeZO cache optimization."""
    total_forward_passes: int = 0
    cache_hits: int = 0
    tokens_reused: int = 0
    tokens_computed: int = 0
    
    @property
    def cache_hit_rate(self) -> float:
        if self.total_forward_passes == 0:
            return 0.0
        return self.cache_hits / self.total_forward_passes
    
    @property
    def token_reuse_rate(self) -> float:
        total_tokens = self.tokens_reused + self.tokens_computed
        if total_tokens == 0:
            return 0.0
        return self.tokens_reused / total_tokens


class MeZORadixOptimizer:
    """
    Optimizes MeZO forward passes using RadixAttention's KV cache.
    
    Key optimizations:
    1. Prompt/response splitting for selective KV caching
    2. LoRA-aware cache invalidation
    3. Incremental forward passes reusing prompt KV
    4. Perturbation tolerance for approximate cache matching
    """
    
    def __init__(self, epsilon: float = 1e-3, cache_prompt_only: bool = True):
        self.epsilon = epsilon
        self.stats = MeZOCacheStats()
        self._cache_state = {}
        self.cache_prompt_only = cache_prompt_only
        
        # Initialize KV cache manager
        self.kv_cache_manager = MeZOKVCacheManager(
            max_cache_size_gb=4.0,
            epsilon_tolerance=1e-5,
            enable_partial_reuse=True,
            cache_prompt_only=cache_prompt_only
        )
        
        # Track LoRA configuration
        self.lora_layers: Set[int] = set()
        self.current_step = 0
        
    def prepare_mezo_requests(
        self,
        base_batch: Dict,
        perturbation_sign: int,  # +1 or -1
        request_prefix: str = "mezo",
        epsilon: float = 1e-3,
        step: int = 0
    ) -> Tuple[List[Req], Dict[str, Any]]:
        """
        Prepare requests for MeZO forward passes with enhanced cache optimization.
        
        Args:
            base_batch: Original batch data
            perturbation_sign: +1 for positive perturbation, -1 for negative
            request_prefix: Prefix for request IDs to enable cache sharing
            epsilon: Current perturbation magnitude
            step: Current training step
            
        Returns:
            Tuple of (requests, cache_metadata)
        """
        requests = []
        cache_metadata = {}
        self.current_step = step
        
        batch_size = base_batch['input_ids'].size(0)
        
        for i in range(batch_size):
            # Extract sequence data
            prompt_len = base_batch['prompt_length'][i].item()
            full_ids = base_batch['input_ids'][i]
            attention_mask = base_batch['attention_mask'][i]
            seq_len = attention_mask.sum().item()
            
            # Get full sequence (prompt + completion)
            input_ids = full_ids[:seq_len].tolist()
            
            # Check cache for this sequence
            cache_entry, segments_to_compute, cache_hit = self.kv_cache_manager.get_or_allocate_cache(
                token_ids=input_ids,
                prompt_length=prompt_len,
                epsilon=epsilon,
                perturbation_sign=perturbation_sign,
                step=step
            )
            
            # Create request with cache awareness
            base_rid = f"{request_prefix}_s{step}_b{i}"
            rid = f"{base_rid}_{perturbation_sign:+d}eps{epsilon:.6f}"
            
            # Determine what portion needs computation
            if cache_hit and self.cache_prompt_only:
                # Only compute response portion
                compute_start = prompt_len
                prefix_len = prompt_len
            else:
                # Compute from the first segment that needs computation
                compute_start = segments_to_compute[0][0] if segments_to_compute else 0
                prefix_len = compute_start
            
            # Create request
            req = Req(
                rid=rid,
                origin_input_text=base_batch.get('prompt', [''])[i] if 'prompt' in base_batch else '',
                origin_input_ids=input_ids,
                sampling_params=SamplingParams(
                    temperature=0,
                    max_new_tokens=0,
                ),
                lora_path=base_batch.get('lora_path', None),
            )
            
            # Set prefix length for cache reuse
            req.prefix_len = prefix_len
            
            # Store enhanced metadata
            cache_metadata[rid] = {
                'base_rid': base_rid,
                'perturbation_sign': perturbation_sign,
                'epsilon': epsilon,
                'prompt_length': prompt_len,
                'total_length': len(input_ids),
                'cache_hit': cache_hit,
                'cache_entry': cache_entry,
                'segments_to_compute': segments_to_compute,
                'prefix_len': prefix_len,
                'batch_idx': i,
            }
            
            requests.append(req)
        
        # Update statistics
        self.stats.total_forward_passes += len(requests)
        
        return requests, cache_metadata
    
    def register_lora_configuration(self, lora_adapter):
        """
        Register LoRA adapter configuration for cache optimization.
        """
        lora_layers = []
        for i, layer in enumerate(lora_adapter.layers):
            if hasattr(layer, 'layer_idx'):
                layer_idx = layer.layer_idx
            else:
                layer_idx = i
            
            # Check which modules have LoRA
            for module_name in layer.weights.keys():
                if 'lora_A' in module_name or 'lora_B' in module_name:
                    module_type = module_name.split('_lora')[0]
                    lora_layers.append((layer_idx, module_type))
                    self.lora_layers.add(layer_idx)
        
        # Register with cache manager
        self.kv_cache_manager.register_lora_layers(lora_layers)
        logger.info(f"Registered {len(self.lora_layers)} LoRA layers for cache optimization")
    
    def analyze_cache_potential(
        self,
        model_config,
        batch_info: Dict[str, Any],
        epsilon: float
    ) -> Dict[str, float]:
        """
        Analyze potential cache hit rates based on current configuration and batch.
        """
        # Get current cache statistics
        cache_stats = self.kv_cache_manager.get_cache_stats()
        
        # Analyze prompt/completion ratio
        total_tokens = 0
        prompt_tokens = 0
        
        if 'prompt_length' in batch_info:
            prompt_lengths = batch_info['prompt_length']
            total_lengths = batch_info['total_length']
            
            if isinstance(prompt_lengths, torch.Tensor):
                prompt_tokens = prompt_lengths.sum().item()
                total_tokens = total_lengths.sum().item() if isinstance(total_lengths, torch.Tensor) else sum(total_lengths)
            else:
                prompt_tokens = sum(prompt_lengths)
                total_tokens = sum(total_lengths)
        
        prompt_ratio = prompt_tokens / total_tokens if total_tokens > 0 else 0
        
        # Calculate theoretical maximum reuse
        if self.cache_prompt_only:
            # Can reuse prompt tokens between +ε and -ε passes
            max_reuse_rate = prompt_ratio * 0.5  # 50% of tokens (second pass)
        else:
            # Full sequence caching (less common)
            max_reuse_rate = 0.5
        
        # Adjust for LoRA impact
        if self.lora_layers:
            # LoRA affects some layers, reducing reuse potential
            lora_impact = len(self.lora_layers) / model_config.num_hidden_layers
            max_reuse_rate *= (1 - lora_impact * 0.5)  # 50% impact per LoRA layer
        
        # Epsilon impact on cache validity
        epsilon_factors = {
            1e-5: 0.99,
            1e-4: 0.95,
            1e-3: 0.85,
            1e-2: 0.60,
            1e-1: 0.20,
        }
        
        closest_epsilon = min(epsilon_factors.keys(), key=lambda x: abs(x - epsilon))
        epsilon_factor = epsilon_factors[closest_epsilon]
        
        # Calculate expected performance
        expected_reuse_rate = max_reuse_rate * epsilon_factor
        expected_speedup = 1 / (1 - expected_reuse_rate * 0.9)  # 90% of theoretical
        
        # Include actual performance if available
        actual_stats = {
            'actual_hit_rate': cache_stats['hit_rate'],
            'actual_token_reuse_rate': cache_stats['token_reuse_rate'],
            'cache_entries': cache_stats['cache_entries'],
            'cache_size_mb': cache_stats['cache_size_mb'],
        }
        
        return {
            'prompt_ratio': prompt_ratio,
            'max_reuse_rate': max_reuse_rate,
            'epsilon_adjusted_rate': expected_reuse_rate,
            'estimated_speedup': expected_speedup,
            'lora_impact_factor': 1 - len(self.lora_layers) / model_config.num_hidden_layers if model_config.num_hidden_layers > 0 else 1,
            **actual_stats
        }
    
    def optimize_forward_schedule(
        self,
        plus_requests: List[Req],
        minus_requests: List[Req],
        tree_cache
    ) -> Tuple[ScheduleBatch, ScheduleBatch]:
        """
        Optimize the scheduling of +εz and -εz forward passes to maximize cache reuse.
        
        Strategy:
        1. Schedule all +εz passes first to populate the cache
        2. Schedule -εz passes to maximize prefix sharing
        3. Use request ordering to improve cache locality
        """
        # Sort requests to maximize prefix sharing
        # Requests with similar prompts should be scheduled together
        plus_requests_sorted = sorted(plus_requests, key=lambda r: r.origin_input_ids[:10])
        minus_requests_sorted = sorted(minus_requests, key=lambda r: r.origin_input_ids[:10])
        
        # Track cache state for optimization
        self._update_cache_state(plus_requests_sorted, minus_requests_sorted)
        
        return plus_requests_sorted, minus_requests_sorted
    
    def _update_cache_state(
        self,
        plus_requests: List[Req],
        minus_requests: List[Req]
    ):
        """Update internal cache state tracking."""
        for req in plus_requests:
            self._cache_state[req.rid] = {
                'input_ids': req.origin_input_ids,
                'cached': True,
                'perturbation': 'plus'
            }
        
        for req in minus_requests:
            # Check if we can reuse from corresponding plus request
            base_rid = req.rid.replace('_minus', '')
            plus_rid = f"{base_rid}_plus"
            
            if plus_rid in self._cache_state:
                # Can potentially reuse cache
                self.stats.cache_hits += 1
                self.stats.tokens_reused += len(req.origin_input_ids)
            else:
                self.stats.tokens_computed += len(req.origin_input_ids)
            
            self._cache_state[req.rid] = {
                'input_ids': req.origin_input_ids,
                'cached': True,
                'perturbation': 'minus'
            }
        
        self.stats.total_forward_passes += len(plus_requests) + len(minus_requests)
    
    def get_optimization_stats(self) -> Dict[str, float]:
        """Get comprehensive optimization statistics."""
        # Get base stats
        base_stats = {
            'cache_hit_rate': self.stats.cache_hit_rate,
            'token_reuse_rate': self.stats.token_reuse_rate,
            'total_forward_passes': self.stats.total_forward_passes,
            'cache_hits': self.stats.cache_hits,
            'tokens_reused': self.stats.tokens_reused,
            'tokens_computed': self.stats.tokens_computed,
        }
        
        # Get KV cache manager stats
        kv_stats = self.kv_cache_manager.get_cache_stats()
        
        # Combine statistics
        combined_stats = base_stats.copy()
        for k, v in kv_stats.items():
            combined_stats['kv_' + k] = v
        return combined_stats
    
    def invalidate_cache_for_step(self, step: int, updated_lora_layers: Optional[Set[int]] = None):
        """Invalidate cache entries when moving to a new training step."""
        # Invalidate based on LoRA updates
        self.kv_cache_manager.invalidate_cache_for_lora_update(updated_lora_layers)
        
        # Update step
        self.current_step = step
        
    def update_cache_after_forward(
        self,
        requests: List[Req],
        cache_metadata: Dict[str, Any],
        kv_indices: Optional[torch.Tensor] = None
    ):
        """Update cache state after forward pass completion."""
        for req, (rid, metadata) in zip(requests, cache_metadata.items()):
            if metadata.get('cache_entry') and kv_indices is not None:
                # Update cache with computed KV indices
                cache_key = metadata['cache_entry'].cache_key
                self.kv_cache_manager.update_cache_indices(cache_key, kv_indices)
                
                # Update our internal tracking
                if metadata['cache_hit']:
                    self.stats.cache_hits += 1
                    self.stats.tokens_reused += metadata['prompt_length']
                else:
                    self.stats.tokens_computed += metadata['total_length']
    
    def estimate_memory_savings(
        self,
        model_config,
        batch_size: int,
        sequence_length: int,
        prompt_length: int
    ) -> Dict[str, float]:
        """Estimate memory savings from cache optimization."""
        # Get KV cache manager's estimate
        kv_savings = self.kv_cache_manager.estimate_memory_savings(model_config)
        
        # Calculate theoretical savings
        hidden_size = model_config.hidden_size
        num_layers = model_config.num_hidden_layers
        
        # Each token needs K and V storage per layer
        kv_size_per_token = 2 * hidden_size * num_layers * 2  # float16
        
        # Without optimization: store KV for both passes
        total_tokens_no_opt = 2 * batch_size * sequence_length
        memory_no_opt = total_tokens_no_opt * kv_size_per_token
        
        # With optimization: reuse prompt KV in second pass
        if self.cache_prompt_only:
            # First pass: full sequence
            # Second pass: only response tokens (prompt KV reused)
            unique_tokens = batch_size * sequence_length + batch_size * (sequence_length - prompt_length)
        else:
            # Reuse based on actual cache hit rate
            cache_reuse_rate = self.kv_cache_manager.stats.token_reuse_rate
            unique_tokens = batch_size * sequence_length * (2 - cache_reuse_rate)
        
        memory_with_opt = unique_tokens * kv_size_per_token
        
        # Calculate speedup estimate
        compute_saved_ratio = self.kv_cache_manager.stats.token_reuse_rate
        estimated_speedup = 1 / (1 - compute_saved_ratio * 0.7)  # 70% of theoretical speedup
        
        return {
            'memory_no_optimization_gb': memory_no_opt / (1024**3),
            'memory_with_optimization_gb': memory_with_opt / (1024**3),
            'memory_savings_gb': (memory_no_opt - memory_with_opt) / (1024**3),
            'memory_reduction_percent': (1 - memory_with_opt / memory_no_opt) * 100 if memory_no_opt > 0 else 0,
            'kv_manager_savings_mb': kv_savings,
            'estimated_speedup': estimated_speedup,
            'prompt_token_ratio': prompt_length / sequence_length if sequence_length > 0 else 0,
        }