"""
MeZO KV Cache Manager for optimizing prompt token reuse across forward passes.

This module implements sophisticated KV cache management specifically designed for MeZO's
training pattern where multiple forward passes share the same prompt tokens.
"""

import torch
import hashlib
import logging
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cached KV entry for a sequence segment."""
    cache_key: str
    token_ids: List[int]
    kv_indices: Optional[torch.Tensor] = None
    lora_version: int = 0
    epsilon: float = 0.0
    last_access_step: int = 0
    hit_count: int = 0
    is_prompt: bool = True
    affected_layers: Set[int] = field(default_factory=set)
    
    @property
    def cached_prompt_length(self) -> int:
        """Length of cached prompt tokens."""
        return len(self.token_ids) if self.is_prompt else 0


@dataclass 
class CacheStats:
    """Statistics for cache performance monitoring."""
    total_requests: int = 0
    cache_hits: int = 0
    partial_hits: int = 0
    cache_misses: int = 0
    tokens_reused: int = 0
    tokens_computed: int = 0
    memory_saved_mb: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return (self.cache_hits + self.partial_hits) / self.total_requests
    
    @property
    def token_reuse_rate(self) -> float:
        total = self.tokens_reused + self.tokens_computed
        if total == 0:
            return 0.0
        return self.tokens_reused / total


class MeZOKVCacheManager:
    """
    Manages KV cache for MeZO training with prompt-aware optimization.
    
    Key features:
    1. Prompt/response splitting for selective caching
    2. LoRA-aware cache invalidation
    3. Perturbation tolerance for approximate cache matching
    4. Memory-efficient cache eviction
    """
    
    def __init__(
        self,
        max_cache_size_gb: float = 4.0,
        epsilon_tolerance: float = 1e-5,
        enable_partial_reuse: bool = True,
        cache_prompt_only: bool = True,
    ):
        self.max_cache_size_bytes = int(max_cache_size_gb * 1024**3)
        self.epsilon_tolerance = epsilon_tolerance
        self.enable_partial_reuse = enable_partial_reuse
        self.cache_prompt_only = cache_prompt_only
        
        # Cache storage
        self.cache_entries: Dict[str, CacheEntry] = {}
        self.lora_version = 0
        self.current_step = 0
        
        # LoRA tracking
        self.lora_affected_layers: Set[int] = set()
        self.layer_wise_cache: Dict[int, Dict[str, Any]] = defaultdict(dict)
        
        # Statistics
        self.stats = CacheStats()
        self.current_cache_size_bytes = 0
        
        logger.info(f"MeZOKVCacheManager initialized: max_cache={max_cache_size_gb:.1f}GB, "
                   f"epsilon_tolerance={epsilon_tolerance}, partial_reuse={enable_partial_reuse}")
    
    def register_lora_layers(self, lora_layers: List[Tuple[int, str]]):
        """Register which layers have LoRA adapters for smart invalidation."""
        self.lora_affected_layers.clear()
        for layer_idx, module_type in lora_layers:
            self.lora_affected_layers.add(layer_idx)
        logger.info(f"Registered {len(self.lora_affected_layers)} LoRA-affected layers")
    
    def generate_cache_key(
        self,
        token_ids: List[int],
        lora_version: Optional[int] = None,
        perturbation_sign: int = 0,
        include_epsilon: bool = False,
        epsilon: float = 0.0
    ) -> str:
        """Generate a unique cache key for a token sequence."""
        # Use hash for efficiency with long sequences
        token_hash = hashlib.md5(str(token_ids).encode()).hexdigest()[:16]
        
        # Include LoRA version for cache invalidation
        lora_v = lora_version if lora_version is not None else self.lora_version
        
        # Base key
        key_parts = [f"mezo_v2", token_hash, f"lora{lora_v}"]
        
        # Add perturbation info if needed
        if include_epsilon and abs(epsilon) > self.epsilon_tolerance:
            # Only include epsilon magnitude, not sign
            # This allows +ε and -ε to share the same cache
            key_parts.append(f"eps{abs(epsilon):.6f}")
            # Note: We intentionally exclude perturbation_sign to enable cache sharing
        
        return "_".join(key_parts)
    
    def can_reuse_cache(
        self,
        cache_entry: CacheEntry,
        current_epsilon: float,
        current_lora_version: int
    ) -> Tuple[bool, str]:
        """
        Determine if a cache entry can be reused.
        
        Returns:
            (can_reuse, reason)
        """
        # Check LoRA version
        if cache_entry.lora_version != current_lora_version:
            return False, "lora_version_mismatch"
        
        # Check epsilon tolerance
        if abs(cache_entry.epsilon - current_epsilon) > self.epsilon_tolerance:
            return False, "epsilon_out_of_tolerance"
        
        # Note: We don't check kv_indices here because in MeZO's use case,
        # we're tracking which tokens need to be computed, not actual KV storage
        
        return True, "reusable"
    
    def split_sequence_for_caching(
        self,
        input_ids: List[int],
        prompt_length: int,
        completion_length: Optional[int] = None
    ) -> List[Tuple[List[int], bool, int, int]]:
        """
        Split sequence into cacheable segments.
        
        Returns:
            List of (token_ids, is_cacheable, start_idx, end_idx)
        """
        segments = []
        
        if self.cache_prompt_only:
            # Only cache prompt portion
            if prompt_length > 0:
                segments.append((
                    input_ids[:prompt_length],
                    True,  # cacheable
                    0,
                    prompt_length
                ))
            
            # Response portion is not cached
            if len(input_ids) > prompt_length:
                segments.append((
                    input_ids[prompt_length:],
                    False,  # not cacheable
                    prompt_length,
                    len(input_ids)
                ))
        else:
            # Cache entire sequence (less common for MeZO)
            segments.append((
                input_ids,
                True,
                0,
                len(input_ids)
            ))
        
        return segments
    
    def get_or_allocate_cache(
        self,
        token_ids: List[int],
        prompt_length: int,
        epsilon: float,
        perturbation_sign: int,
        step: int
    ) -> Tuple[Optional[CacheEntry], List[Tuple[int, int]], bool]:
        """
        Get existing cache or allocate new entry.
        
        Returns:
            (cache_entry, segments_to_compute, cache_hit)
            segments_to_compute: List of (start, end) indices that need computation
        """
        self.stats.total_requests += 1
        self.current_step = step
        
        # Split sequence into segments
        segments = self.split_sequence_for_caching(token_ids, prompt_length)
        
        cache_hits = []
        segments_to_compute = []
        
        for segment_tokens, is_cacheable, start_idx, end_idx in segments:
            if not is_cacheable:
                # Always compute non-cacheable segments
                segments_to_compute.append((start_idx, end_idx))
                self.stats.tokens_computed += len(segment_tokens)
                continue
            
            # Generate cache key for this segment
            cache_key = self.generate_cache_key(
                segment_tokens,
                self.lora_version,
                perturbation_sign if abs(epsilon) > self.epsilon_tolerance else 0,
                include_epsilon=True,
                epsilon=epsilon
            )
            
            # Check if we have a valid cache entry
            if cache_key in self.cache_entries:
                cache_entry = self.cache_entries[cache_key]
                can_reuse, reason = self.can_reuse_cache(cache_entry, epsilon, self.lora_version)
                
                if can_reuse:
                    # Cache hit!
                    cache_entry.hit_count += 1
                    cache_entry.last_access_step = step
                    cache_hits.append(cache_entry)
                    self.stats.tokens_reused += len(segment_tokens)
                    logger.debug(f"Cache hit for segment [{start_idx}:{end_idx}], key={cache_key}")
                else:
                    # Cache miss due to invalidation
                    segments_to_compute.append((start_idx, end_idx))
                    self.stats.tokens_computed += len(segment_tokens)
                    logger.debug(f"Cache miss for segment [{start_idx}:{end_idx}]: {reason}")
            else:
                # New cache entry needed
                segments_to_compute.append((start_idx, end_idx))
                self.stats.tokens_computed += len(segment_tokens)
                
                # Create new entry
                new_entry = CacheEntry(
                    cache_key=cache_key,
                    token_ids=segment_tokens,
                    lora_version=self.lora_version,
                    epsilon=epsilon,
                    last_access_step=step,
                    is_prompt=(start_idx < prompt_length)
                )
                self.cache_entries[cache_key] = new_entry
                logger.debug(f"Created new cache entry for segment [{start_idx}:{end_idx}], key={cache_key}")
        
        # Update statistics
        if len(cache_hits) == len(segments):
            self.stats.cache_hits += 1
            full_hit = True
        elif len(cache_hits) > 0:
            self.stats.partial_hits += 1
            full_hit = False
        else:
            self.stats.cache_misses += 1
            full_hit = False
        
        # Return the first cache entry if available (for prompt caching)
        primary_cache = cache_hits[0] if cache_hits else None
        
        return primary_cache, segments_to_compute, full_hit
    
    def invalidate_cache_for_lora_update(self, updated_layers: Optional[Set[int]] = None):
        """Invalidate cache entries affected by LoRA parameter updates."""
        self.lora_version += 1
        
        if updated_layers is None:
            # Full invalidation
            invalidated = len(self.cache_entries)
            self.cache_entries.clear()
            self.current_cache_size_bytes = 0
            logger.info(f"Full cache invalidation: cleared {invalidated} entries")
        else:
            # Selective invalidation based on affected layers
            if not self.enable_partial_reuse:
                # If partial reuse is disabled, invalidate everything
                self.invalidate_cache_for_lora_update(None)
                return
            
            # Track which entries to remove
            to_remove = []
            for key, entry in self.cache_entries.items():
                if entry.affected_layers.intersection(updated_layers):
                    to_remove.append(key)
            
            # Remove affected entries
            for key in to_remove:
                del self.cache_entries[key]
            
            logger.info(f"Selective cache invalidation: removed {len(to_remove)} entries "
                       f"affected by layers {updated_layers}")
    
    def evict_least_recently_used(self, required_bytes: int):
        """Evict cache entries using LRU policy to free up memory."""
        if not self.cache_entries:
            return
        
        # Sort by last access time
        sorted_entries = sorted(
            self.cache_entries.values(),
            key=lambda e: (e.last_access_step, -e.hit_count)
        )
        
        freed_bytes = 0
        evicted_count = 0
        
        for entry in sorted_entries:
            if freed_bytes >= required_bytes:
                break
            
            # Estimate memory usage (simplified)
            entry_size = len(entry.token_ids) * 2 * 4096  # Rough estimate
            
            # Mark as evicted but keep metadata
            entry.kv_indices = None
            freed_bytes += entry_size
            evicted_count += 1
        
        self.current_cache_size_bytes -= freed_bytes
        logger.info(f"Evicted {evicted_count} cache entries, freed {freed_bytes / 1024**2:.1f}MB")
    
    def update_cache_indices(self, cache_key: str, kv_indices: torch.Tensor):
        """Update cache entry with actual KV indices after computation."""
        if cache_key in self.cache_entries:
            self.cache_entries[cache_key].kv_indices = kv_indices
            
            # Update memory tracking
            entry_size = len(self.cache_entries[cache_key].token_ids) * 2 * 4096
            self.current_cache_size_bytes += entry_size
            
            # Check if eviction is needed
            if self.current_cache_size_bytes > self.max_cache_size_bytes:
                self.evict_least_recently_used(
                    self.current_cache_size_bytes - self.max_cache_size_bytes
                )
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        return {
            'hit_rate': self.stats.hit_rate,
            'token_reuse_rate': self.stats.token_reuse_rate,
            'total_requests': self.stats.total_requests,
            'cache_hits': self.stats.cache_hits,
            'partial_hits': self.stats.partial_hits,
            'cache_misses': self.stats.cache_misses,
            'tokens_reused': self.stats.tokens_reused,
            'tokens_computed': self.stats.tokens_computed,
            'cache_entries': len(self.cache_entries),
            'cache_size_mb': self.current_cache_size_bytes / 1024**2,
            'max_cache_size_mb': self.max_cache_size_bytes / 1024**2,
            'lora_version': self.lora_version,
            'memory_saved_mb': self.stats.memory_saved_mb,
        }
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.stats = CacheStats()
        logger.info("Cache statistics reset")
    
    def estimate_memory_savings(self, model_config) -> float:
        """Estimate memory savings from cache reuse."""
        if self.stats.tokens_reused == 0:
            return 0.0
        
        # Estimate based on model size
        hidden_size = model_config.hidden_size
        num_layers = model_config.num_hidden_layers
        
        # Each token requires K and V storage per layer
        bytes_per_token = 2 * hidden_size * num_layers * 2  # fp16
        
        # Calculate savings
        saved_bytes = self.stats.tokens_reused * bytes_per_token
        self.stats.memory_saved_mb = saved_bytes / 1024**2
        
        return self.stats.memory_saved_mb