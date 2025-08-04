"""
MeZO Integration with SGLang ModelRunner
Provides seamless integration of MeZO training with SGLang's infrastructure.
"""

import torch
import torch.distributed as dist
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass

# Import these when ModelRunner is available
try:
    from sglang.srt.model_runner import ModelRunner
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.sampling.sampling_params import SamplingParams
    MODELRUNNER_AVAILABLE = True
except ImportError:
    MODELRUNNER_AVAILABLE = False
    ModelRunner = None

logger = logging.getLogger(__name__)


@dataclass
class MeZOStep:
    """Represents a single MeZO optimization step."""
    batch_texts: List[str]
    learning_rate: float
    epsilon: float
    lora_weights: Optional[Dict[str, torch.Tensor]] = None


class MeZOModelRunner:
    """
    Extends SGLang's ModelRunner with MeZO optimization capabilities.
    This integrates MeZO training directly into SGLang's inference engine.
    """
    
    def __init__(self, model_runner, enable_radix_cache: bool = True):
        self.model_runner = model_runner
        self.enable_radix_cache = enable_radix_cache
        self.tp_rank = model_runner.tp_rank
        self.tp_size = model_runner.tp_size
        
        # Track MeZO statistics
        self.mezo_stats = {
            'total_steps': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_forward_passes': 0
        }
        
    def mezo_forward_pass(self, 
                         batch: ScheduleBatch,
                         lora_weights: Optional[Dict[str, torch.Tensor]] = None) -> float:
        """
        Perform a forward pass with optional LoRA weights.
        Returns the loss value.
        """
        # Apply LoRA weights if provided
        if lora_weights is not None:
            self._apply_lora_weights(lora_weights)
        
        # Run forward pass through ModelRunner
        logits_output = self.model_runner.forward(batch)
        
        # Extract loss from output
        loss = self._compute_loss(logits_output, batch)
        
        # Update statistics
        self.mezo_stats['total_forward_passes'] += 1
        
        return loss
    
    def mezo_optimization_step(self, mezo_step: MeZOStep) -> Tuple[float, Dict]:
        """
        Perform a complete MeZO optimization step.
        Returns average loss and cache statistics.
        """
        # Create batch from texts
        batch = self._create_batch(mezo_step.batch_texts)
        
        # Generate synchronized perturbation for distributed training
        if self.tp_size > 1:
            perturbation = self._generate_synchronized_perturbation(mezo_step.lora_weights)
        else:
            perturbation = self._generate_perturbation(mezo_step.lora_weights)
        
        # Forward pass with positive perturbation
        perturbed_weights_pos = self._apply_perturbation(
            mezo_step.lora_weights, perturbation, mezo_step.epsilon, positive=True
        )
        loss_plus = self.mezo_forward_pass(batch, perturbed_weights_pos)
        
        # Track cache hit for negative pass (should reuse KV cache)
        cache_hit_before = self._get_cache_hits()
        
        # Forward pass with negative perturbation
        perturbed_weights_neg = self._apply_perturbation(
            mezo_step.lora_weights, perturbation, mezo_step.epsilon, positive=False
        )
        loss_minus = self.mezo_forward_pass(batch, perturbed_weights_neg)
        
        # Check if cache was reused
        cache_hit_after = self._get_cache_hits()
        cache_reused = cache_hit_after > cache_hit_before
        
        if cache_reused:
            self.mezo_stats['cache_hits'] += 1
        else:
            self.mezo_stats['cache_misses'] += 1
        
        # Compute gradient estimate
        if self.tp_size > 1:
            # Aggregate losses across TP ranks
            losses = torch.tensor([loss_plus, loss_minus], device='cuda')
            dist.all_reduce(losses, op=dist.ReduceOp.SUM)
            losses /= self.tp_size
            loss_plus, loss_minus = losses[0].item(), losses[1].item()
        
        grad_estimate = (loss_plus - loss_minus) / (2 * mezo_step.epsilon)
        
        # Update weights
        self._update_weights(
            mezo_step.lora_weights, 
            perturbation, 
            grad_estimate, 
            mezo_step.learning_rate
        )
        
        # Update statistics
        self.mezo_stats['total_steps'] += 1
        
        avg_loss = (loss_plus + loss_minus) / 2
        cache_stats = {
            'cache_hit_rate': self.mezo_stats['cache_hits'] / max(1, self.mezo_stats['cache_hits'] + self.mezo_stats['cache_misses']),
            'total_cache_hits': self.mezo_stats['cache_hits'],
            'cache_reused_this_step': cache_reused
        }
        
        return avg_loss, cache_stats
    
    def _create_batch(self, texts: List[str]):
        """Create a ScheduleBatch from text inputs."""
        # Tokenize inputs
        tokenizer = self.model_runner.tokenizer
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        
        # Create requests
        reqs = []
        for i, text in enumerate(texts):
            req = Req(
                request_id=f"mezo_req_{i}",
                prompt_text=text,
                prompt_ids=inputs['input_ids'][i].tolist(),
                sampling_params=SamplingParams(max_new_tokens=1, temperature=0),
            )
            reqs.append(req)
        
        # Create batch
        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            tree_cache=self.model_runner.tree_cache
        )
        
        return batch
    
    def _generate_perturbation(self, weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Generate random perturbation for weights."""
        perturbation = {}
        for name, weight in weights.items():
            perturbation[name] = torch.randn_like(weight)
        return perturbation
    
    def _generate_synchronized_perturbation(self, weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Generate synchronized perturbation across TP ranks."""
        # Broadcast seed from rank 0
        if self.tp_rank == 0:
            seed = torch.randint(0, 2**31-1, (1,), device='cuda')
        else:
            seed = torch.zeros(1, dtype=torch.long, device='cuda')
        
        dist.broadcast(seed, src=0)
        torch.manual_seed(seed.item())
        
        return self._generate_perturbation(weights)
    
    def _apply_perturbation(self, weights: Dict[str, torch.Tensor], 
                          perturbation: Dict[str, torch.Tensor], 
                          epsilon: float, 
                          positive: bool) -> Dict[str, torch.Tensor]:
        """Apply perturbation to weights."""
        sign = 1 if positive else -1
        perturbed = {}
        for name, weight in weights.items():
            perturbed[name] = weight + sign * epsilon * perturbation[name]
        return perturbed
    
    def _update_weights(self, weights: Dict[str, torch.Tensor], 
                       perturbation: Dict[str, torch.Tensor], 
                       grad_estimate: float, 
                       learning_rate: float):
        """Update weights using MeZO gradient estimate."""
        for name in weights:
            weights[name] -= learning_rate * grad_estimate * perturbation[name]
    
    def _apply_lora_weights(self, lora_weights: Dict[str, torch.Tensor]):
        """Apply LoRA weights to the model."""
        # This would integrate with SGLang's LoRA manager
        # For now, this is a placeholder
        pass
    
    def _compute_loss(self, logits_output, batch: ScheduleBatch) -> float:
        """Compute loss from model output."""
        # Extract logits and compute cross-entropy loss
        # This is a simplified version - actual implementation would handle
        # different model types and loss functions
        return torch.randn(1).item() + 3.0  # Mock loss
    
    def _get_cache_hits(self) -> int:
        """Get current cache hit count from RadixAttention."""
        if hasattr(self.model_runner, 'attn_backend') and hasattr(self.model_runner.attn_backend, 'get_cache_hits'):
            return self.model_runner.attn_backend.get_cache_hits()
        return 0
    
    def get_statistics(self) -> Dict:
        """Get MeZO training statistics."""
        stats = self.mezo_stats.copy()
        stats['avg_cache_hit_rate'] = stats['cache_hits'] / max(1, stats['cache_hits'] + stats['cache_misses'])
        stats['forward_passes_per_step'] = stats['total_forward_passes'] / max(1, stats['total_steps'])
        return stats


def create_mezo_model_runner(model_path: str, 
                           tp_size: int = 1,
                           enable_radix_cache: bool = True,
                           **kwargs) -> MeZOModelRunner:
    """
    Create a MeZO-enabled ModelRunner.
    
    Args:
        model_path: Path to the model
        tp_size: Tensor parallel size
        enable_radix_cache: Enable RadixAttention for cache optimization
        **kwargs: Additional arguments for ModelRunner
    
    Returns:
        MeZOModelRunner instance
    """
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.model_runner import ModelRunner
    
    # Create server args
    server_args = ServerArgs(
        model_path=model_path,
        tp_size=tp_size,
        trust_remote_code=kwargs.get('trust_remote_code', True),
        mem_fraction_static=kwargs.get('mem_fraction_static', 0.8),
        disable_radix_cache=not enable_radix_cache,
        **kwargs
    )
    
    # Initialize ModelRunner
    # Note: This is simplified - actual implementation would handle
    # distributed initialization, model loading, etc.
    model_runner = ModelRunner(
        server_args=server_args,
        # Additional initialization parameters
    )
    
    # Wrap with MeZO capabilities
    mezo_runner = MeZOModelRunner(model_runner, enable_radix_cache)
    
    return mezo_runner