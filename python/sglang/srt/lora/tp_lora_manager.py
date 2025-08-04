"""
Tensor Parallel LoRA Manager for distributed MeZO training.

This module implements TP-aware LoRA weight management, ensuring correct
weight distribution and gradient accumulation across tensor parallel ranks.
"""

import torch
import torch.distributed as dist
from typing import Dict, List, Optional, Tuple
import logging
import re

from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.lora.lora import LoRAAdapter, LoRALayer
from sglang.srt.model_loader.loader import DefaultModelLoader

logger = logging.getLogger(__name__)


class TPLoRAManager(LoRAManager):
    """
    Tensor Parallel aware LoRA manager.
    
    Key features:
    - Distributes LoRA weights across TP ranks
    - Handles column-parallel and row-parallel layers correctly
    - Ensures gradient accumulation is TP-aware
    """
    
    def __init__(
        self,
        base_lora_manager: LoRAManager,
        tp_size: int,
        tp_rank: int,
        tp_group: Optional[dist.ProcessGroup] = None
    ):
        """
        Initialize TP-aware LoRA manager.
        
        Args:
            base_lora_manager: Base LoRA manager instance
            tp_size: Number of tensor parallel ranks
            tp_rank: Current rank in TP group
            tp_group: Process group for tensor parallelism
        """
        # Copy attributes from base manager
        self.__dict__.update(base_lora_manager.__dict__)
        
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.tp_group = tp_group
        
        # Track which layers are column/row parallel
        self._identify_parallel_layers()
        
        logger.info(f"Initialized TPLoRAManager: rank={tp_rank}/{tp_size}")
    
    def _identify_parallel_layers(self):
        """Identify which layers are column-parallel vs row-parallel."""
        self.column_parallel_layers = set()
        self.row_parallel_layers = set()
        
        # Common patterns for different model architectures
        column_patterns = [
            r"q_proj", r"k_proj", r"v_proj",  # Attention QKV
            r"gate_proj", r"up_proj",          # MLP gates
            r"fc1", r"c_fc",                   # Various architectures
            r"query", r"key", r"value",        # Alternative names
        ]
        
        row_patterns = [
            r"o_proj", r"out_proj",            # Attention output
            r"down_proj",                      # MLP output
            r"fc2", r"c_proj",                 # Various architectures
            r"dense",                          # Some architectures
        ]
        
        # Will be populated when adapters are loaded
        self.layer_parallel_info = {}
    
    def add_adapter(self, adapter: LoRAAdapter):
        """
        Add a LoRA adapter with TP-aware weight distribution.
        
        Args:
            adapter: LoRA adapter to add
        """
        # First, let base class handle the addition
        super().add_adapter(adapter)
        
        # Then apply TP-specific weight sharding
        self._shard_adapter_weights(adapter)
    
    def _shard_adapter_weights(self, adapter: LoRAAdapter):
        """
        Shard LoRA weights according to tensor parallel strategy.
        
        For column-parallel layers:
        - Split along output dimension (columns)
        - Each rank gets a slice of columns
        
        For row-parallel layers:
        - Split along input dimension (rows)
        - Each rank gets a slice of rows
        """
        logger.info(f"Sharding weights for adapter {adapter.uid} on rank {self.tp_rank}")
        
        # Process each layer
        for layer_idx, layer in enumerate(adapter.layers):
            new_weights = {}
            
            for weight_name, weight in layer.weights.items():
                # Determine if this is column or row parallel
                is_column_parallel = any(
                    re.search(pattern, weight_name) 
                    for pattern in [r"q_proj", r"k_proj", r"v_proj", r"gate_proj", r"up_proj"]
                )
                is_row_parallel = any(
                    re.search(pattern, weight_name)
                    for pattern in [r"o_proj", r"out_proj", r"down_proj"]
                )
                
                if is_column_parallel:
                    # Split along output dimension (dim 0 for lora_A, dim 1 for lora_B)
                    if "lora_A" in weight_name:
                        # lora_A: [r, in_features] - don't split
                        new_weights[weight_name] = weight
                    elif "lora_B" in weight_name:
                        # lora_B: [out_features, r] - split dim 0
                        shard_size = weight.shape[0] // self.tp_size
                        start_idx = self.tp_rank * shard_size
                        end_idx = start_idx + shard_size
                        new_weights[weight_name] = weight[start_idx:end_idx].contiguous()
                        logger.debug(f"Column-parallel {weight_name}: {weight.shape} -> {new_weights[weight_name].shape}")
                    else:
                        new_weights[weight_name] = weight
                
                elif is_row_parallel:
                    # Split along input dimension
                    if "lora_A" in weight_name:
                        # lora_A: [r, in_features] - split dim 1
                        shard_size = weight.shape[1] // self.tp_size
                        start_idx = self.tp_rank * shard_size
                        end_idx = start_idx + shard_size
                        new_weights[weight_name] = weight[:, start_idx:end_idx].contiguous()
                        logger.debug(f"Row-parallel {weight_name}: {weight.shape} -> {new_weights[weight_name].shape}")
                    elif "lora_B" in weight_name:
                        # lora_B: [out_features, r] - don't split
                        new_weights[weight_name] = weight
                    else:
                        new_weights[weight_name] = weight
                
                else:
                    # Not parallel, keep as is
                    new_weights[weight_name] = weight
            
            # Update layer weights
            layer.weights = new_weights
    
    def get_adapter_for_training(self, adapter_name: str) -> LoRAAdapter:
        """
        Get adapter configured for TP-aware training.
        
        Args:
            adapter_name: Name of the adapter
            
        Returns:
            TP-configured LoRA adapter
        """
        adapter = self.get_adapter(adapter_name)
        
        # Ensure weights are properly sharded
        if adapter and not hasattr(adapter, '_tp_sharded'):
            self._shard_adapter_weights(adapter)
            adapter._tp_sharded = True
        
        return adapter
    
    def aggregate_gradients(self, adapter: LoRAAdapter):
        """
        Aggregate gradients across TP ranks for correct updates.
        
        For column-parallel layers:
        - Gradients are already summed by backward pass
        
        For row-parallel layers:
        - Need to all-reduce gradients
        """
        if not self.tp_group:
            return
        
        for layer in adapter.layers:
            for name, param in layer.named_parameters():
                if param.grad is None:
                    continue
                
                # Check if this needs all-reduce
                is_row_parallel_A = "lora_A" in name and any(
                    re.search(pattern, name)
                    for pattern in [r"o_proj", r"out_proj", r"down_proj"]
                )
                
                if is_row_parallel_A:
                    # All-reduce gradient for row-parallel lora_A
                    dist.all_reduce(
                        param.grad,
                        op=dist.ReduceOp.SUM,
                        group=self.tp_group
                    )
    
    def save_adapter(self, adapter_name: str, save_path: str):
        """
        Save adapter with TP rank suffix.
        
        Each rank saves its shard separately.
        """
        adapter = self.get_adapter(adapter_name)
        if not adapter:
            logger.warning(f"Adapter {adapter_name} not found")
            return
        
        # Create rank-specific save path
        rank_save_path = f"{save_path}_tp{self.tp_rank}"
        
        # Collect weights from this rank
        state_dict = {}
        
        for layer_idx, layer in enumerate(adapter.layers):
            for weight_name, weight in layer.weights.items():
                state_dict[weight_name] = weight
        
        # Save
        torch.save({
            'state_dict': state_dict,
            'config': adapter.config,
            'tp_rank': self.tp_rank,
            'tp_size': self.tp_size,
        }, rank_save_path)
        
        logger.info(f"Saved TP shard to {rank_save_path}")
    
    def load_adapter_shards(self, adapter_name: str, load_path: str):
        """
        Load and merge adapter shards from all TP ranks.
        
        This is used when switching from TP training to single GPU inference.
        """
        merged_weights = {}
        
        # Load all shards
        shards = []
        for rank in range(self.tp_size):
            shard_path = f"{load_path}_tp{rank}"
            shard = torch.load(shard_path, map_location='cpu')
            shards.append(shard)
        
        # Merge shards
        # TODO: Implement proper merging logic based on parallel strategy
        logger.info(f"Loaded {len(shards)} shards for adapter {adapter_name}")
        
        return merged_weights


class TPLoRAOptimizer:
    """
    TP-aware optimizer wrapper for LoRA parameters.
    
    Ensures correct gradient accumulation and parameter updates
    in tensor parallel setting.
    """
    
    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        tp_lora_manager: TPLoRAManager
    ):
        self.base_optimizer = base_optimizer
        self.tp_manager = tp_lora_manager
    
    def step(self, adapter_name: str):
        """
        Perform optimization step with TP-aware gradient handling.
        """
        adapter = self.tp_manager.get_adapter(adapter_name)
        
        # Aggregate gradients across TP ranks if needed
        self.tp_manager.aggregate_gradients(adapter)
        
        # Perform optimization step
        self.base_optimizer.step()
    
    def zero_grad(self):
        """Zero gradients."""
        self.base_optimizer.zero_grad()