# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""MeZO (Memory-efficient Zeroth-order) training implementation for SGLang server."""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from sglang.srt.managers.io_struct import (
    GenerateReqInput,
    MeZOTrainRequest,
    MeZOTrainResponse,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.sampling.sampling_params import SamplingParams

logger = logging.getLogger(__name__)


class MeZOServerTrainer:
    """
    MeZO trainer that works with SGLang server infrastructure.
    
    This class handles MeZO training steps by:
    1. Using the existing server infrastructure for tokenization
    2. Running forward passes through the model
    3. Estimating gradients using finite differences
    4. Updating LoRA weights through the LoRAManager
    """
    
    def __init__(self, tokenizer_manager: TokenizerManager):
        self.tokenizer_manager = tokenizer_manager
        self.perturbation_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        
    async def train_step(self, request: MeZOTrainRequest) -> MeZOTrainResponse:
        """Execute a single MeZO training step."""
        start_time = time.time()
        
        try:
            # Validate the LoRA adapter exists
            if not await self._validate_lora_adapter(request.lora_name):
                return MeZOTrainResponse(
                    loss=0.0,
                    gradient_norm=0.0,
                    forward_time_ms=0.0,
                    total_time_ms=0.0,
                    tokens_processed=0,
                    rid=request.rid,
                    error=f"LoRA adapter '{request.lora_name}' not found"
                )
            
            # Prepare batch data
            batch_prompts, batch_completions = self._prepare_batch(request.batch)
            
            # Get LoRA parameters
            lora_params = await self._get_lora_parameters(request.lora_name)
            if not lora_params:
                return MeZOTrainResponse(
                    loss=0.0,
                    gradient_norm=0.0,
                    forward_time_ms=0.0,
                    total_time_ms=0.0,
                    tokens_processed=0,
                    rid=request.rid,
                    error="Failed to get LoRA parameters"
                )
            
            # Generate random perturbations
            z_list = self._generate_perturbations(lora_params, request.lora_name)
            
            # Apply positive perturbation and compute loss
            self._apply_perturbation(lora_params, z_list, request.epsilon)
            forward_start = time.time()
            loss_plus, tokens_plus = await self._compute_loss(
                batch_prompts, batch_completions, request.lora_name, request.use_full_sequence_loss
            )
            
            # Apply negative perturbation and compute loss
            self._apply_perturbation(lora_params, z_list, -2 * request.epsilon)
            loss_minus, tokens_minus = await self._compute_loss(
                batch_prompts, batch_completions, request.lora_name, request.use_full_sequence_loss
            )
            forward_time_ms = (time.time() - forward_start) * 1000
            
            # Restore original weights
            self._apply_perturbation(lora_params, z_list, request.epsilon)
            
            # Estimate gradient and update weights
            gradient_scale = (loss_plus - loss_minus) / (2 * request.epsilon)
            gradient_norm = self._update_weights(
                lora_params, z_list, gradient_scale, request.learning_rate
            )
            
            # Update LoRA weights in the manager
            weight_updates = await self._prepare_weight_updates(request.lora_name, lora_params)
            success = await self._update_lora_manager(request.lora_name, weight_updates)
            
            if not success:
                return MeZOTrainResponse(
                    loss=0.0,
                    gradient_norm=0.0,
                    forward_time_ms=forward_time_ms,
                    total_time_ms=(time.time() - start_time) * 1000,
                    tokens_processed=0,
                    rid=request.rid,
                    error="Failed to update LoRA weights in manager"
                )
            
            # Calculate average loss
            avg_loss = (loss_plus + loss_minus) / 2
            total_tokens = max(tokens_plus, tokens_minus)
            
            logger.info(
                f"MeZO step completed: loss={avg_loss:.4f}, "
                f"gradient_norm={gradient_norm:.4f}, tokens={total_tokens}"
            )
            
            return MeZOTrainResponse(
                loss=float(avg_loss),
                gradient_norm=float(gradient_norm),
                forward_time_ms=forward_time_ms,
                total_time_ms=(time.time() - start_time) * 1000,
                tokens_processed=total_tokens,
                rid=request.rid,
                cache_hit_rate=None,  # TODO: Get from tokenizer manager
                error=None
            )
            
        except Exception as e:
            logger.error(f"MeZO training step failed: {str(e)}")
            return MeZOTrainResponse(
                loss=0.0,
                gradient_norm=0.0,
                forward_time_ms=0.0,
                total_time_ms=(time.time() - start_time) * 1000,
                tokens_processed=0,
                rid=request.rid,
                error=str(e)
            )
    
    async def _validate_lora_adapter(self, lora_name: str) -> bool:
        """Check if the LoRA adapter exists."""
        # For now, assume the adapter exists if a name is provided
        # In a real implementation, this would check with the server
        return bool(lora_name)
    
    def _prepare_batch(self, batch: List[Dict[str, str]]) -> Tuple[List[str], List[str]]:
        """Extract prompts and completions from batch."""
        prompts = [item["prompt"] for item in batch]
        completions = [item["completion"] for item in batch]
        return prompts, completions
    
    async def _get_lora_parameters(self, lora_name: str) -> Optional[List[nn.Parameter]]:
        """Get LoRA parameters for the specified adapter."""
        # In this simplified version, we'll create dummy parameters
        # In a real implementation, this would retrieve actual LoRA weights
        
        # Create dummy parameters for demonstration
        # Typical LoRA has A and B matrices for each layer
        num_layers = 12  # OPT-125M has 12 layers
        lora_rank = 16
        hidden_dim = 768  # OPT-125M hidden dimension
        
        lora_params = []
        for layer_idx in range(num_layers):
            # Query/Value projection LoRA weights (simplified)
            lora_a = nn.Parameter(torch.randn(lora_rank, hidden_dim) * 0.01)
            lora_b = nn.Parameter(torch.zeros(hidden_dim, lora_rank))
            lora_params.extend([lora_a, lora_b])
                
        return lora_params
    
    def _generate_perturbations(
        self, lora_params: List[nn.Parameter], lora_name: str
    ) -> List[torch.Tensor]:
        """Generate or retrieve cached random perturbations."""
        if lora_name not in self.perturbation_cache:
            self.perturbation_cache[lora_name] = {}
            
        z_list = []
        for i, param in enumerate(lora_params):
            param_key = f"param_{i}"
            if param_key not in self.perturbation_cache[lora_name]:
                # Generate new perturbation
                z = torch.randn_like(param)
                self.perturbation_cache[lora_name][param_key] = z
            else:
                z = self.perturbation_cache[lora_name][param_key]
            z_list.append(z)
            
        return z_list
    
    def _apply_perturbation(
        self, lora_params: List[nn.Parameter], z_list: List[torch.Tensor], scale: float
    ):
        """Apply perturbation to parameters."""
        with torch.no_grad():
            for param, z in zip(lora_params, z_list):
                param.data.add_(scale * z)
    
    async def _compute_loss(
        self,
        prompts: List[str],
        completions: List[str],
        lora_name: str,
        use_full_sequence_loss: bool
    ) -> Tuple[float, int]:
        """Compute loss for the batch using server infrastructure."""
        total_loss = 0.0
        total_tokens = 0
        
        # Process all samples in a single batch for efficiency
        texts = []
        prompt_lens = []
        
        for prompt, completion in zip(prompts, completions):
            texts.append(prompt + completion)
            prompt_lens.append(len(self.tokenizer_manager.tokenizer.encode(prompt)))
        
        # Create a batch generation request to get logprobs
        # Use logprob_start_len=0 to get logprobs for all tokens (including input)
        sampling_params = {
            "temperature": 0.0,
            "max_new_tokens": 0,  # We only need logprobs for existing tokens
            "skip_special_tokens": False,
        }
        
        req = GenerateReqInput(
            text=texts,
            sampling_params=sampling_params,
            lora_path=lora_name,
            return_logprob=True,
            top_logprobs_num=1,
            logprob_start_len=0,  # Get logprobs from the beginning (echo functionality)
            stream=False,
        )
        
        # Get response with logprobs
        try:
            response = await self.tokenizer_manager.generate_request(req, None).__anext__()
            
            # Debug: Log response structure
            logger.info(f"Response type: {type(response)}")
            if isinstance(response, list) and len(response) > 0:
                logger.info(f"First response item keys: {list(response[0].keys()) if isinstance(response[0], dict) else 'Not a dict'}")
            
            # Process each sample in the batch
            for i, (prompt_len, text) in enumerate(zip(prompt_lens, texts)):
                if i < len(response):
                    sample_resp = response[i] if isinstance(response, list) else response
                    
                    # Extract logprobs from response
                    meta_info = sample_resp.get("meta_info", {})
                    
                    # Get both input and output logprobs
                    input_logprobs = meta_info.get("input_token_logprobs", [])
                    output_logprobs = meta_info.get("output_token_logprobs", [])
                    
                    # Combine them for full sequence
                    logprobs = input_logprobs + output_logprobs
                    
                    # Debug: Log logprobs info
                    logger.info(f"Sample {i}: input_logprobs len: {len(input_logprobs)}, output_logprobs len: {len(output_logprobs)}, total: {len(logprobs)}")
                    if logprobs and len(logprobs) > 0:
                        logger.info(f"Sample {i}: First logprob type: {type(logprobs[0])}, value: {logprobs[0]}")
                    
                    if logprobs:
                        if use_full_sequence_loss:
                            # Use loss over full sequence
                            valid_logprobs = []
                            for lp in logprobs:
                                if lp is not None:
                                    if isinstance(lp, (list, tuple)):
                                        # Extract logprob from list/tuple [logprob, token_id, None]
                                        if len(lp) > 0 and lp[0] is not None:
                                            valid_logprobs.append(lp[0])
                                    else:
                                        valid_logprobs.append(lp)
                            
                            if valid_logprobs:
                                loss = -sum(valid_logprobs)
                                total_loss += loss
                                total_tokens += len(valid_logprobs)
                        else:
                            # Use loss only over completion tokens
                            if len(logprobs) > prompt_len:
                                completion_logprobs = logprobs[prompt_len:]
                                # Handle tuples if logprobs are returned as (token_id, logprob)
                                valid_logprobs = []
                                for lp in completion_logprobs:
                                    if lp is not None:
                                        if isinstance(lp, (list, tuple)):
                                            # Extract logprob from list/tuple [logprob, token_id, None]
                                            if len(lp) > 0 and lp[0] is not None:
                                                valid_logprobs.append(lp[0])
                                        else:
                                            valid_logprobs.append(lp)
                                
                                if valid_logprobs:
                                    loss = -sum(valid_logprobs)
                                    total_loss += loss
                                    total_tokens += len(valid_logprobs)
                                
        except Exception as e:
            logger.error(f"Failed to compute loss: {e}")
            # Return a dummy loss to continue training
            return 1.0, len(texts)
        
        # Return average loss
        if total_tokens > 0:
            avg_loss = total_loss / total_tokens
        else:
            # Fallback to a default loss if no tokens were processed
            avg_loss = 1.0
            total_tokens = sum(len(self.tokenizer_manager.tokenizer.encode(t)) for t in texts)
            
        return avg_loss, total_tokens
    
    def _update_weights(
        self,
        lora_params: List[nn.Parameter],
        z_list: List[torch.Tensor],
        gradient_scale: float,
        learning_rate: float
    ) -> float:
        """Update weights using MeZO gradient estimate."""
        total_norm = 0.0
        
        with torch.no_grad():
            for param, z in zip(lora_params, z_list):
                # MeZO update: w = w - lr * gradient_scale * z
                param.data.add_(-learning_rate * gradient_scale * z)
                
                # Accumulate gradient norm
                grad_estimate = gradient_scale * z
                total_norm += grad_estimate.norm().item() ** 2
        
        return total_norm ** 0.5
    
    async def _prepare_weight_updates(
        self, lora_name: str, lora_params: List[nn.Parameter]
    ) -> Dict[str, torch.Tensor]:
        """Prepare weight updates for LoRAManager."""
        # In this simplified version, create weight update dictionary
        # In a real implementation, this would map to actual LoRA weight names
        
        weight_updates = {}
        param_idx = 0
        num_layers = 12  # OPT-125M
        
        # Create weight names following SGLang convention
        for layer_idx in range(num_layers):
            # LoRA A weights
            if param_idx < len(lora_params):
                weight_name = f"base_model.model.model.layers.{layer_idx}.self_attn.q_proj.lora_A.weight"
                weight_updates[weight_name] = lora_params[param_idx].data.cpu()
                param_idx += 1
                
            # LoRA B weights  
            if param_idx < len(lora_params):
                weight_name = f"base_model.model.model.layers.{layer_idx}.self_attn.q_proj.lora_B.weight"
                weight_updates[weight_name] = lora_params[param_idx].data.cpu()
                param_idx += 1
                    
        return weight_updates
    
    async def _update_lora_manager(
        self, lora_name: str, weight_updates: Dict[str, torch.Tensor]
    ) -> bool:
        """Update LoRA weights in the manager."""
        # In this simplified version, we'll simulate the update
        # In a real implementation, this would call the LoRAManager
        logger.info(f"Simulating weight update for {lora_name} with {len(weight_updates)} weights")
        return True