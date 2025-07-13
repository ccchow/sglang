"""
Inference-only OPT model compatible with HuggingFace weights.
Complete implementation with all required attributes for SGLang ModelRunner.
"""
from typing import Iterable, Optional, Tuple, Type

import torch
from torch import nn
from transformers import OPTConfig

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.distributed.parallel_state import get_tensor_model_parallel_world_size
from sglang.srt.layers.activation import NewGELU
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix


def add_missing_config_attributes(config: OPTConfig) -> OPTConfig:
    """Add missing attributes required by SGLang to OPT config."""
    # Model type attributes
    if not hasattr(config, 'is_generation'):
        config.is_generation = True
    if not hasattr(config, 'is_multimodal'):
        config.is_multimodal = False
    if not hasattr(config, 'is_embedding'):
        config.is_embedding = False
    if not hasattr(config, 'is_encoder_decoder'):
        config.is_encoder_decoder = False
    
    # Attention architecture
    if not hasattr(config, 'attention_arch'):
        config.attention_arch = AttentionArch.MHA
    
    # For multi-head attention, num_key_value_heads equals num_attention_heads
    if not hasattr(config, 'num_key_value_heads'):
        config.num_key_value_heads = config.num_attention_heads
    
    # Additional attributes
    if not hasattr(config, 'is_hybrid'):
        config.is_hybrid = False
    if not hasattr(config, 'is_multimodal_chunked_prefill_supported'):
        config.is_multimodal_chunked_prefill_supported = False
    
    # Sliding window attention (OPT doesn't use it)
    if not hasattr(config, 'sliding_window'):
        config.sliding_window = None
    
    # Bias settings (OPT uses bias by default)
    if not hasattr(config, 'enable_bias'):
        config.enable_bias = True
    
    return config


class OPTAttention(nn.Module):
    """OPT attention layer with RadixAttention support."""
    
    def __init__(
        self,
        layer_id: int,
        config: OPTConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        
        # Ensure we have num_key_value_heads
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = getattr(config, 'num_key_value_heads', self.total_num_heads)
        
        assert self.total_num_heads % tp_size == 0
        assert self.total_num_kv_heads % tp_size == 0
        
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = self.hidden_size // self.total_num_heads
        self.scaling = self.head_dim**-0.5

        # OPT uses separate q, k, v projections
        self.q_proj = ColumnParallelLinear(
            config.hidden_size,
            config.hidden_size,
            bias=config.enable_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )
        self.k_proj = ColumnParallelLinear(
            config.hidden_size,
            config.hidden_size,
            bias=config.enable_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.k_proj",
        )
        self.v_proj = ColumnParallelLinear(
            config.hidden_size,
            config.hidden_size,
            bias=config.enable_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.v_proj",
        )
        self.out_proj = RowParallelLinear(
            config.hidden_size,
            config.hidden_size,
            bias=config.enable_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

        # RadixAttention for KV cache optimization
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
        )

    def forward(self, hidden_states: torch.Tensor, forward_batch: ForwardBatch):
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)
        
        # Reshape for attention computation
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        
        # Apply RadixAttention
        attn_output = self.attn(q, k, v, forward_batch)
        
        # Project output
        output, _ = self.out_proj(attn_output)
        return output


class OPTMLP(nn.Module):
    """OPT feedforward network."""
    
    def __init__(
        self,
        config: OPTConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.ffn_dim,
            bias=config.enable_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
        )
        self.fc2 = RowParallelLinear(
            config.ffn_dim,
            config.hidden_size,
            bias=config.enable_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
        )
        self.act = NewGELU()

    def forward(self, hidden_states):
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        return hidden_states


class OPTDecoderLayer(nn.Module):
    """OPT decoder layer."""
    
    def __init__(
        self,
        layer_id: int,
        config: OPTConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_id = layer_id
        
        # Layer normalization
        self.self_attn_layer_norm = nn.LayerNorm(
            config.hidden_size, 
            elementwise_affine=config.layer_norm_elementwise_affine
        )
        self.final_layer_norm = nn.LayerNorm(
            config.hidden_size, 
            elementwise_affine=config.layer_norm_elementwise_affine
        )
        
        # Attention and MLP
        self.self_attn = OPTAttention(
            layer_id=layer_id,
            config=config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.fc = OPTMLP(
            config, 
            quant_config=quant_config,
            prefix=f"{prefix}",
        )

    def forward(self, hidden_states: torch.Tensor, forward_batch: ForwardBatch):
        # Self Attention with pre-norm
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, forward_batch)
        hidden_states = residual + hidden_states

        # Fully Connected with pre-norm
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class OPTModel(nn.Module):
    """OPT model without language modeling head."""
    
    def __init__(
        self,
        config: OPTConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.max_position_embeddings = config.max_position_embeddings
        self.vocab_size = config.vocab_size
        
        # Token embeddings
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, 
            config.word_embed_proj_dim, 
            quant_config=quant_config,
        )
        
        # Position embeddings (OPT uses learned position embeddings starting from 2)
        self.embed_positions = nn.Embedding(
            config.max_position_embeddings + 2, 
            config.hidden_size
        )

        # Project embeddings if dimensions don't match
        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = nn.Linear(
                config.word_embed_proj_dim, 
                config.hidden_size, 
                bias=False
            )
        else:
            self.project_in = None

        # Decoder layers
        self.layers = nn.ModuleList([
            OPTDecoderLayer(
                layer_id=i, 
                config=config, 
                quant_config=quant_config,
                prefix=f"model.decoder.layers.{i}",
            )
            for i in range(config.num_hidden_layers)
        ])

        # Project out if needed
        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = nn.Linear(
                config.hidden_size, 
                config.word_embed_proj_dim, 
                bias=False
            )
        else:
            self.project_out = None

        # Final layer norm
        self.final_layer_norm = nn.LayerNorm(
            config.hidden_size, 
            elementwise_affine=config.layer_norm_elementwise_affine
        )

    def forward(self, input_ids: torch.Tensor, forward_batch: ForwardBatch):
        # Token embeddings
        inputs_embeds = self.embed_tokens(input_ids)
        
        # Position embeddings (OPT adds 2 to positions)
        seq_length = input_ids.shape[1]
        positions = torch.arange(
            0, seq_length, dtype=torch.long, device=input_ids.device
        )
        positions = positions.unsqueeze(0).expand_as(input_ids) + 2
        pos_embeds = self.embed_positions(positions)

        # Project embeddings if needed
        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)

        # Add position embeddings
        hidden_states = inputs_embeds + pos_embeds

        # Apply decoder layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, forward_batch)

        # Final layer norm
        hidden_states = self.final_layer_norm(hidden_states)

        # Project out if needed
        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)

        return hidden_states


class OPTForCausalLM(nn.Module):
    """OPT model with language modeling head."""
    
    def __init__(
        self,
        config: OPTConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        # Add missing attributes to config
        config = add_missing_config_attributes(config)
        
        self.config = config
        self.quant_config = quant_config
        
        # Model
        self.model = OPTModel(config, quant_config)
        
        # Language modeling head (shares embeddings with token embeddings)
        self.lm_head = ParallelLMHead(
            config.vocab_size, 
            config.word_embed_proj_dim,
            quant_config=quant_config,
        )
        
        # Logits processor
        self.logits_processor = LogitsProcessor(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, forward_batch)
        logits = self.lm_head(hidden_states)
        logits = self.logits_processor(logits, hidden_states, forward_batch)
        return logits

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights from HuggingFace checkpoint."""
        stacked_params_mapping = []
        params_dict = dict(self.named_parameters(remove_duplicate=False))

        for name, loaded_weight in weights:
            # Skip the language modeling head weight as it's tied with embeddings
            if "lm_head.weight" in name:
                continue
            
            # Map HuggingFace weight names to our model structure
            # HF uses: model.decoder.layers.X.* -> our model uses: model.layers.X.*
            if name.startswith("model.decoder."):
                # Remove 'decoder.' from the path
                param_name = name.replace("model.decoder.", "model.")
            else:
                param_name = name
            
            # Get the parameter
            param = params_dict.get(param_name)
            
            if param is not None:
                weight_loader = getattr(
                    param, "weight_loader", default_weight_loader
                )
                weight_loader(param, loaded_weight)
            else:
                # Log unmatched weights for debugging (only in debug mode)
                # print(f"Warning: No parameter found for weight: {name}")
                pass
        
        # Tie the language modeling head with token embeddings
        self.lm_head.weight = self.model.embed_tokens.weight


# Export the model class for registry
EntryClass = OPTForCausalLM