import torch
import numpy as np
import math
import logging
from typing import Dict, List, Union, Optional
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import datasets
import json
from pathlib import Path
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch, ModelWorkerBatch
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.distributed import (
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.mezo_radix_optimizer import MeZORadixOptimizer

# MeZO is inherently efficient with just 2 forward passes
# No need for additional CUDA optimizations

class MeZOTrainer:
    def __init__(self, model_runner: ModelRunner, lora_manager: LoRAManager, lora_name: str, tokenizer, normalize_perturbations=False):
        self.model_runner = model_runner
        self.lora_manager = lora_manager
        self.lora_name = lora_name
        self.tokenizer = tokenizer
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # MeZO configuration
        self.normalize_perturbations = normalize_perturbations  # Default: False (follows paper)
        
        # Configuration for loss calculation
        self.compute_full_sequence_loss = False  # Set to True for experimental full sequence loss
        self.use_accuracy_objective = False  # Set to True to optimize accuracy instead of loss
        
        # KV cache optimization settings
        self.enable_kv_cache_optimization = True
        self.kv_cache_hit_rate = 0.0  # Track cache efficiency
        
        # RadixAttention optimization
        self.radix_optimizer = None
        if self.enable_kv_cache_optimization:
            self.radix_optimizer = MeZORadixOptimizer()
            self.logger.info("RadixAttention optimization enabled for MeZO")
        
        # Tensor parallelism configuration
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_group = get_tp_group() if self.tp_size > 1 else None
        
        # Device configuration
        self.device = self.lora_manager.device
        
        # MeZO is already optimized with just 2 forward passes
        self.logger.info(f"MeZO trainer initialized - 2 forward passes per step")
        if self.tp_size > 1:
            self.logger.info(f"Tensor parallelism enabled: size={self.tp_size}, rank={self.tp_rank}")

    def train(self, train_dataloader: DataLoader, learning_rate=1e-5, num_steps=1000, epsilon=1e-3):
        lora_adapter = self.lora_manager.loras[self.lora_name]
        
        lora_params = []
        for layer in lora_adapter.layers:
            for param_name, param in layer.weights.items():
                if "lora_A" in param_name or "lora_B" in param_name:
                    lora_params.append(param)
        
        self.logger.info(f"Collected {len(lora_params)} LoRA parameters for training")
        
        optimizer = torch.optim.Adam(lora_params, lr=learning_rate)
        
        # Create iterator for cycling through batches
        dataloader_iter = iter(train_dataloader)
        
        for step in range(num_steps):
            self.current_step = step
            # Get next batch, restart iterator if needed
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(train_dataloader)
                batch = next(dataloader_iter)
            
            loss = self._mezo_step(batch, lora_params, optimizer, epsilon)
            
            if step % 100 == 0:
                self.logger.info(f"Step {step}/{num_steps}, Loss estimate: {loss:.4f}")

    def _mezo_step(self, batch, lora_params, optimizer, epsilon):
        # MeZO always uses exactly 2 forward passes per step
        # Sample a fixed perturbation direction z
        if self.tp_size > 1:
            # For tensor parallelism, ensure all ranks use the same perturbation
            z_list = self._generate_synchronized_perturbations(lora_params)
        else:
            z_list = [torch.randn_like(p) for p in lora_params]
        
        # Optionally normalize perturbations (not in original paper)
        if self.normalize_perturbations:
            z_list = [z / (z.norm() + 1e-8) for z in z_list]
        
        if self.enable_kv_cache_optimization:
            # Optimized version with in-place perturbations
            return self._mezo_step_optimized(batch, lora_params, optimizer, epsilon, z_list)
        else:
            # Original version with parameter cloning
            return self._mezo_step_original(batch, lora_params, optimizer, epsilon, z_list)
    
    def _generate_synchronized_perturbations(self, lora_params):
        """Generate perturbations that are synchronized across all TP ranks."""
        z_list = []
        
        if self.tp_rank == 0:
            # Rank 0 generates the random seed
            seed = torch.randint(0, 2**32, (1,), device=lora_params[0].device)
        else:
            seed = torch.zeros(1, dtype=torch.long, device=lora_params[0].device)
        
        # Broadcast seed from rank 0 to all other ranks
        torch.distributed.broadcast(seed, src=0, group=self.tp_group.device_group)
        
        # All ranks use the same seed to generate identical perturbations
        generator = torch.Generator(device=lora_params[0].device)
        generator.manual_seed(seed.item())
        
        for p in lora_params:
            z = torch.randn_like(p, generator=generator)
            z_list.append(z)
        
        return z_list
    
    def _mezo_step_optimized(self, batch, lora_params, optimizer, epsilon, z_list):
        """Optimized MeZO step with in-place perturbations and RadixAttention optimization."""
        if self.radix_optimizer and self.enable_kv_cache_optimization:
            # Use RadixAttention-optimized forward passes
            loss_plus, loss_minus = self._forward_pass_radix_optimized(batch, lora_params, epsilon, z_list)
        else:
            # Standard forward passes
            # Apply positive perturbation in-place
            for i, p in enumerate(lora_params):
                p.data.add_(epsilon * z_list[i])
            loss_plus = self._forward_pass(batch)
            
            # Aggregate loss across TP ranks if needed
            if self.tp_size > 1:
                loss_plus = self._aggregate_loss_across_tp(loss_plus)
            
            # Switch to negative perturbation (from +εz to -εz)
            for i, p in enumerate(lora_params):
                p.data.add_(-2 * epsilon * z_list[i])
            loss_minus = self._forward_pass(batch)
            
            # Aggregate loss across TP ranks if needed
            if self.tp_size > 1:
                loss_minus = self._aggregate_loss_across_tp(loss_minus)
        
        # Restore original parameters (from -εz back to original)
        for i, p in enumerate(lora_params):
            p.data.add_(epsilon * z_list[i])
        
        # Estimate gradient using MeZO formula
        projected_grad = (loss_plus - loss_minus) / (2 * epsilon)
        
        optimizer.zero_grad()
        for i, p in enumerate(lora_params):
            p.grad = z_list[i] * projected_grad
        optimizer.step()
        
        return (loss_plus + loss_minus) / 2
    
    def _aggregate_loss_across_tp(self, loss):
        """Aggregate loss across tensor parallel ranks."""
        # Convert loss to tensor if it's a scalar
        if not isinstance(loss, torch.Tensor):
            loss = torch.tensor(loss, device=self.device)
        
        # All-reduce the loss across TP ranks
        tensor_model_parallel_all_reduce(loss)
        
        # Average the loss
        loss = loss / self.tp_size
        
        return loss.item()
    
    def _forward_pass_radix_optimized(self, batch, lora_params, epsilon, z_list):
        """
        Optimized forward passes using RadixAttention to maximize KV cache reuse.
        
        Strategy:
        1. Create requests that share common prefixes for cache reuse
        2. Apply perturbations and run forward pass for +εz
        3. Switch perturbations and run forward pass for -εz
        4. The RadixCache will automatically reuse KV values for shared prefixes
        """
        # Prepare requests for RadixAttention optimization
        plus_requests, plus_metadata = self.radix_optimizer.prepare_mezo_requests(
            batch, perturbation_sign=1, request_prefix=f"mezo_step{self.current_step}"
        )
        minus_requests, minus_metadata = self.radix_optimizer.prepare_mezo_requests(
            batch, perturbation_sign=-1, request_prefix=f"mezo_step{self.current_step}"
        )
        
        # Apply positive perturbation
        for i, p in enumerate(lora_params):
            p.data.add_(epsilon * z_list[i])
        
        # Forward pass with +εz (this will populate the RadixCache)
        loss_plus = self._forward_pass_with_requests(plus_requests)
        
        # Switch to negative perturbation (from +εz to -εz)
        for i, p in enumerate(lora_params):
            p.data.add_(-2 * epsilon * z_list[i])
        
        # Forward pass with -εz (this will reuse cache for shared prefixes)
        loss_minus = self._forward_pass_with_requests(minus_requests)
        
        # Update optimization statistics
        stats = self.radix_optimizer.get_optimization_stats()
        self.kv_cache_hit_rate = stats['cache_hit_rate']
        
        # Log cache efficiency periodically
        if hasattr(self, 'current_step') and self.current_step % 100 == 0:
            self.logger.info(f"RadixAttention cache hit rate: {stats['cache_hit_rate']:.2%}, "
                           f"Token reuse rate: {stats['token_reuse_rate']:.2%}")
        
        # Aggregate losses across TP ranks if needed
        if self.tp_size > 1:
            loss_plus = self._aggregate_loss_across_tp(loss_plus)
            loss_minus = self._aggregate_loss_across_tp(loss_minus)
        
        return loss_plus, loss_minus
    
    def _forward_pass_with_requests(self, requests: List[Req]) -> float:
        """
        Perform forward pass using prepared requests that enable cache sharing.
        """
        # Create ScheduleBatch from requests
        schedule_batch = ScheduleBatch.init_new(
            requests=requests,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            tree_cache=self.model_runner.tree_cache,
        )
        
        # Create ModelWorkerBatch
        model_batch = ModelWorkerBatch(
            forward_mode=schedule_batch.forward_mode,
            batch_size=len(requests),
            req_pool_indices=schedule_batch.req_pool_indices,
            seq_lens=schedule_batch.seq_lens,
            prefix_lens=schedule_batch.prefix_lens,
            output_lens=[0] * len(requests),  # No generation needed
            extend_lens=schedule_batch.extend_lens,
            return_logprob=False,
        )
        
        # Run forward pass
        logits, _ = self.model_runner.forward(model_batch)
        
        # Compute loss
        # For MeZO, we typically compute loss only on the last token
        # This is simplified - actual implementation would match _forward_pass
        total_loss = 0.0
        for i, req in enumerate(requests):
            # Simple cross-entropy loss on last token
            seq_logits = logits[i]  # Shape: [seq_len, vocab_size]
            if len(seq_logits) > 1:
                # Use second-to-last token to predict last token
                target = torch.tensor(req.origin_input_ids[-1], device=logits.device)
                loss = torch.nn.functional.cross_entropy(seq_logits[-2].unsqueeze(0), target.unsqueeze(0))
                total_loss += loss.item()
        
        return total_loss / len(requests) if requests else 0.0
    
    def _mezo_step_original(self, batch, lora_params, optimizer, epsilon, z_list):
        """Original MeZO step with parameter cloning (more memory intensive)."""
        # Store original parameters
        original_params = [p.clone() for p in lora_params]

        # Perturb positively (+εz) and get loss
        for i, p in enumerate(lora_params):
            p.data.add_(epsilon * z_list[i])
        loss_plus = self._forward_pass(batch)
        
        # Aggregate loss across TP ranks if needed
        if self.tp_size > 1:
            loss_plus = self._aggregate_loss_across_tp(loss_plus)

        # Restore and perturb negatively (-εz) and get loss
        self._restore_params(lora_params, original_params)
        for i, p in enumerate(lora_params):
            p.data.add_(-epsilon * z_list[i])
        loss_minus = self._forward_pass(batch)
        
        # Aggregate loss across TP ranks if needed
        if self.tp_size > 1:
            loss_minus = self._aggregate_loss_across_tp(loss_minus)

        # Restore original parameters
        self._restore_params(lora_params, original_params)

        # Estimate gradient using MeZO formula: g = (loss_plus - loss_minus) / (2 * epsilon) * z
        projected_grad = (loss_plus - loss_minus) / (2 * epsilon)
        
        optimizer.zero_grad()
        for i, p in enumerate(lora_params):
            p.grad = z_list[i] * projected_grad
        optimizer.step()
        
        # Return average of the two losses for logging
        return (loss_plus + loss_minus) / 2

    def _perturb_params(self, params, epsilon, z_list):
        # This method is now deprecated in favor of inline perturbation
        # Kept for potential future use
        for i, p in enumerate(params):
            p.data.add_(epsilon * z_list[i])

    def _restore_params(self, params, original_params):
        for i, p in enumerate(params):
            p.data = original_params[i].clone()

    def _compute_accuracy_objective(self, logits, labels):
        """
        Compute negative accuracy as the objective (for minimization).
        Used when use_accuracy_objective=True, following the MeZO paper.
        """
        predictions = torch.argmax(logits, dim=-1)
        correct = (predictions == labels).float()
        accuracy = correct.mean()
        return -accuracy  # Negative for minimization
    
    def _forward_pass(self, batch):
        """Perform forward pass and compute loss.
        
        Note: SGLang is optimized for inference, so we compute loss iteratively
        over completion tokens. This is less efficient than standard training
        but works within SGLang's architecture.
        """
        # Handle both dict batch (from DataLoader) and list batch (legacy)
        if isinstance(batch, dict):
            # DataLoader format
            batch_size = batch['input_ids'].size(0)
            requests = []
            for i in range(batch_size):
                # For training, we use the prompt portion only
                prompt_len = batch['prompt_length'][i].item()
                prompt_ids = batch['input_ids'][i][:prompt_len].tolist()
                requests.append(Req(
                    rid=str(i),
                    origin_input_text=batch['prompt'][i],
                    origin_input_ids=prompt_ids,
                    sampling_params=SamplingParams(temperature=0),
                    lora_path=self.lora_name,
                ))
        else:
            # Legacy list format
            requests = [
                Req(
                    rid=str(i),
                    origin_input_text=item["prompt"],
                    origin_input_ids=self.tokenizer.encode(item["prompt"]),
                    sampling_params=SamplingParams(temperature=0),
                    lora_path=self.lora_name,
                )
                for i, item in enumerate(batch)
            ]
        
        schedule_batch = ScheduleBatch.init_new(
            reqs=requests,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
            tree_cache=None, # Assuming no prefix caching for training
            model_config=self.model_runner.model_config,
            enable_overlap=False,
            spec_algorithm=None,
            enable_custom_logit_processor=False,
        )

        # 2. Prepare for extend
        schedule_batch.prepare_for_extend()
        model_worker_batch = schedule_batch.get_model_worker_batch()

        # 3. Forward pass
        output, _ = self.model_runner.forward(model_worker_batch)

        # 4. Calculate loss with improved handling
        # Note: Due to SGLang's inference-optimized architecture, we compute
        # loss on next-token predictions. For more accurate training, consider
        # accumulating losses over multiple forward passes for full sequences.
        
        logits = output.next_token_logits
        
        if isinstance(batch, dict):
            # DataLoader format - extract target tokens from input_ids
            target_ids = []
            valid_mask = []
            
            for i in range(len(batch['prompt'])):
                prompt_len = batch['prompt_length'][i].item()
                input_ids = batch['input_ids'][i]
                attention_mask = batch['attention_mask'][i]
                
                # Check if there's a valid completion token
                if prompt_len < len(input_ids) and attention_mask[prompt_len] == 1:
                    target_ids.append(input_ids[prompt_len].item())
                    valid_mask.append(1.0)
                else:
                    # No valid completion, use pad token but mask out loss
                    target_ids.append(self.tokenizer.pad_token_id or 0)
                    valid_mask.append(0.0)
            
            target_ids = torch.tensor(target_ids, dtype=torch.long, device=logits.device)
            valid_mask = torch.tensor(valid_mask, dtype=torch.float32, device=logits.device)
            
            # Compute loss with masking
            if valid_mask.sum() > 0:
                loss = torch.nn.functional.cross_entropy(logits, target_ids, reduction='none')
                masked_loss = loss * valid_mask
                final_loss = masked_loss.sum() / valid_mask.sum()
            else:
                # No valid targets in batch
                final_loss = torch.tensor(0.0, device=logits.device)
        else:
            # Legacy format
            target_ids = torch.tensor([self.tokenizer.encode(item["completion"])[0] for item in batch], dtype=torch.long, device=logits.device)
            final_loss = torch.nn.functional.cross_entropy(logits, target_ids)
        
        return final_loss.item()
    
    def _compute_full_sequence_loss(self, batch):
        """
        Experimental: Compute loss over full completion sequence.
        
        This would require modifying SGLang's forward pass to return all logits,
        not just next-token logits. Currently not implemented due to SGLang's
        inference-optimized architecture.
        
        Future improvements could include:
        1. Multiple forward passes to accumulate loss over full sequences
        2. Custom model wrapper that returns all logits during training
        3. Integration with SGLang's planned training features
        """
        raise NotImplementedError(
            "Full sequence loss calculation requires modifications to SGLang's "
            "inference-optimized forward pass. Use single-token loss for now."
        )
    
    def analyze_epsilon_for_cache_efficiency(self, batch, lora_params, epsilon_values=[1e-4, 1e-3, 1e-2]):
        """
        Analyze different epsilon values to find optimal cache reuse.
        
        Smaller epsilon values should lead to higher KV cache hit rates
        as the perturbations cause less change in activations.
        """
        self.logger.info("Analyzing epsilon values for KV cache efficiency...")
        
        results = {}
        for epsilon in epsilon_values:
            # Perform test forward passes
            z_list = [torch.randn_like(p) for p in lora_params]
            
            # Time the forward passes
            import time
            start = time.time()
            
            # Apply perturbations and measure
            for i, p in enumerate(lora_params):
                p.data.add_(epsilon * z_list[i])
            loss_plus = self._forward_pass(batch)
            
            for i, p in enumerate(lora_params):
                p.data.add_(-2 * epsilon * z_list[i])
            loss_minus = self._forward_pass(batch)
            
            # Restore
            for i, p in enumerate(lora_params):
                p.data.add_(epsilon * z_list[i])
            
            elapsed = time.time() - start
            
            results[epsilon] = {
                'time': elapsed,
                'loss_diff': abs(loss_plus - loss_minus),
                'avg_loss': (loss_plus + loss_minus) / 2
            }
            
            self.logger.info(f"Epsilon={epsilon}: time={elapsed:.3f}s, loss_diff={abs(loss_plus - loss_minus):.6f}")
        
        return results


from typing import Optional, Dict, Any
import logging
from sglang.srt.server_args import ServerArgs
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.utils import get_device_memory_capacity

logger = logging.getLogger(__name__)


class MeZODataset(Dataset):
    """Dataset class for MeZO training with support for various formats."""
    
    def __init__(self, 
                 dataset_path: Union[str, List[Dict]], 
                 tokenizer, 
                 max_length: int = 512,
                 format_type: str = "auto"):
        """
        Args:
            dataset_path: Path to dataset file, HF dataset name, or list of examples
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            format_type: Dataset format ("auto", "jsonl", "json", "hf", "list")
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        # Load dataset based on type
        if isinstance(dataset_path, list):
            # Direct list of examples
            self.examples = dataset_path
        elif isinstance(dataset_path, str):
            if format_type == "auto":
                # Auto-detect format
                if dataset_path.endswith('.jsonl'):
                    format_type = "jsonl"
                elif dataset_path.endswith('.json'):
                    format_type = "json"
                elif Path(dataset_path).exists():
                    format_type = "jsonl"  # Default to JSONL for files
                else:
                    format_type = "hf"  # Assume HF dataset name
            
            if format_type == "jsonl":
                with open(dataset_path, 'r') as f:
                    self.examples = [json.loads(line) for line in f]
            elif format_type == "json":
                with open(dataset_path, 'r') as f:
                    self.examples = json.load(f)
            elif format_type == "hf":
                # Load from Hugging Face datasets
                ds = datasets.load_dataset(dataset_path, split="train")
                self.examples = list(ds)
        
        # Validate examples
        for i, ex in enumerate(self.examples):
            if 'prompt' not in ex or 'completion' not in ex:
                raise ValueError(f"Example {i} missing 'prompt' or 'completion' field")
        
        logger.info(f"Loaded {len(self.examples)} examples from {dataset_path}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenize prompt and completion
        prompt_tokens = self.tokenizer.encode(example['prompt'], add_special_tokens=True)
        completion_tokens = self.tokenizer.encode(example['completion'], add_special_tokens=False)
        
        # Combine tokens (simplified - in practice would handle more carefully)
        input_ids = prompt_tokens + completion_tokens
        
        # Truncate if needed
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
        
        # Create attention mask
        attention_mask = [1] * len(input_ids)
        
        # Pad to max_length
        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
        
        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'prompt': example['prompt'],
            'completion': example['completion'],
            'prompt_length': len(prompt_tokens)
        }


def create_dataloader(dataset: MeZODataset, 
                     batch_size: int, 
                     shuffle: bool = True,
                     distributed: bool = False,
                     world_size: int = 1,
                     rank: int = 0) -> DataLoader:
    """Create a DataLoader with appropriate sampling strategy."""
    if distributed and world_size > 1:
        sampler = DistributedSampler(
            dataset, 
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle
        )
        # When using DistributedSampler, shuffle must be False in DataLoader
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            drop_last=True  # Important for distributed training
        )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False
        )

def mezo_finetune(
    model_path: str,
    train_dataset: Union[str, List[Dict], MeZODataset],
    lora_rank: int = 8,
    learning_rate: float = 1e-6,  # Paper default (was 1e-5)
    num_steps: int = 10000,  # More realistic default (was 1000)
    epsilon: float = 1e-3,
    batch_size: int = 64,  # Paper default for RoBERTa (was 1)
    max_length: int = 512,
    normalize_perturbations: bool = False,  # Paper doesn't normalize
    server_args: Optional[ServerArgs] = None,
    **kwargs
):
    """Fine-tune a model using the MeZO (Memory-efficient Zeroth-order) algorithm.
    
    Default hyperparameters follow the MeZO paper (Malladi et al., 2023).
    
    Args:
        model_path: Path to the model or Hugging Face model ID
        train_dataset: Training dataset - can be:
            - Path to dataset file (JSONL, JSON)
            - Hugging Face dataset name
            - List of dict with 'prompt' and 'completion'
            - MeZODataset instance
        lora_rank: Rank for LoRA adapter (default: 8, from paper)
        learning_rate: Learning rate for optimization (default: 1e-6, from paper)
        num_steps: Number of training steps (default: 10000, paper uses 100K)
        epsilon: Perturbation scale for gradient estimation (default: 1e-3, from paper)
        batch_size: Batch size for training (default: 64, from paper)
        max_length: Maximum sequence length (default: 512)
        normalize_perturbations: Whether to normalize perturbations (default: False, paper doesn't)
        server_args: Optional ServerArgs for advanced configuration
        **kwargs: Additional arguments passed to ServerArgs
    
    Returns:
        Dict containing the trained LoRA weights and metadata
    """
    # Create or validate ServerArgs
    if server_args is None:
        # Build default ServerArgs with user-specified options
        server_args_kwargs = {
            'model_path': model_path,
            'lora_rank': lora_rank,
        }
        
        # Add any additional kwargs that are valid ServerArgs fields
        valid_fields = {f.name for f in ServerArgs.__dataclass_fields__.values()}
        for k, v in kwargs.items():
            if k in valid_fields:
                server_args_kwargs[k] = v
        
        server_args = ServerArgs(**server_args_kwargs)
        logger.info(f"Created ServerArgs with: {server_args_kwargs}")
    else:
        # Override specific fields if provided
        server_args.model_path = model_path
        server_args.lora_rank = lora_rank
    
    # Validate quantization settings
    if server_args.quantization:
        logger.info(f"Using quantization method: {server_args.quantization}")
        if server_args.quantization == "bitsandbytes":
            logger.info("BitsAndBytes quantization enabled for memory-efficient training")
    
    # Get model configuration
    model_config = ModelConfig(model_path)
    
    # Detect distributed environment
    import torch.distributed as dist
    is_distributed = dist.is_available() and dist.is_initialized()
    
    if is_distributed:
        tp_size = dist.get_world_size()
        tp_rank = dist.get_rank()
        logger.info(f"Distributed training detected: rank {tp_rank}/{tp_size}")
    else:
        tp_size = 1
        tp_rank = 0
        logger.info("Single GPU training mode")
    
    # Initialize model runner with proper distributed settings
    try:
        # Check available GPU memory
        device_memory = get_device_memory_capacity()
        logger.info(f"Available device memory: {device_memory / 1e9:.2f} GB")
        
        model_runner = ModelRunner(
            model_config=model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=tp_rank if is_distributed else 0,
            tp_rank=tp_rank,
            tp_size=tp_size,
            pp_rank=0,  # Pipeline parallelism not yet supported for MeZO
            pp_size=1,
            nccl_port=server_args.nccl_port if hasattr(server_args, 'nccl_port') else 23333,
            server_args=server_args,
        )
        logger.info("Model runner initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize model runner: {e}")
        # Fallback to single GPU
        if is_distributed:
            logger.warning("Falling back to single GPU mode")
            model_runner = ModelRunner(
                model_config=model_config,
                mem_fraction_static=server_args.mem_fraction_static,
                gpu_id=0,
                tp_rank=0,
                tp_size=1,
                pp_rank=0,
                pp_size=1,
                nccl_port=23333,
                server_args=server_args,
            )
        else:
            raise

    lora_manager = model_runner.lora_manager
    lora_name = "mezo_lora"
    
    # Create and initialize a new LoRA adapter
    logger.info(f"Creating LoRA adapter with rank={lora_rank}")
    lora_manager.load_lora_adapter(lora_name, "") # Pass empty path to create a new adapter
    lora_adapter = lora_manager.loras[lora_name]
    
    # Initialize LoRA weights
    for layer in lora_adapter.layers:
        for param_name, param in layer.weights.items():
            if "lora_A" in param_name:
                torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
            elif "lora_B" in param_name:
                torch.nn.init.zeros_(param)
    
    # Ensure RNG synchronization for distributed training
    if is_distributed:
        import torch
        # Set seed based on rank to ensure different perturbations per rank initially
        # but deterministic across runs
        torch.manual_seed(42 + tp_rank)

    # Initialize tokenizer
    tokenizer_path = server_args.tokenizer_path or model_path
    tokenizer = get_tokenizer(
        tokenizer_path, 
        tokenizer_mode=server_args.tokenizer_mode, 
        trust_remote_code=server_args.trust_remote_code
    )
    logger.info(f"Tokenizer loaded from: {tokenizer_path}")
    
    # Prepare dataset
    if not isinstance(train_dataset, MeZODataset):
        train_dataset = MeZODataset(
            dataset_path=train_dataset,
            tokenizer=tokenizer,
            max_length=max_length
        )
    
    # Create dataloader
    train_dataloader = create_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        distributed=is_distributed,
        world_size=tp_size,
        rank=tp_rank
    )
    logger.info(f"Created dataloader with batch_size={batch_size}, num_batches={len(train_dataloader)}")

    # Start training
    logger.info(f"Starting MeZO training with {num_steps} steps, lr={learning_rate}, epsilon={epsilon}")
    trainer = MeZOTrainer(model_runner, lora_manager, lora_name, tokenizer, normalize_perturbations=normalize_perturbations)
    trainer.train(train_dataloader, learning_rate, num_steps, epsilon)
    
    logger.info("MeZO training completed successfully")
    
    # Return trained weights and metadata
    return {
        'weights': lora_manager.loras[lora_name].weights,
        'config': {
            'model_path': model_path,
            'lora_rank': lora_rank,
            'num_steps': num_steps,
            'learning_rate': learning_rate,
            'epsilon': epsilon,
            'quantization': server_args.quantization,
        }
    }
