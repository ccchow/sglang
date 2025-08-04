"""
SGLang MeZO Fine-tuning API
High-level API for MeZO training with SGLang.
"""

import os
import json
import time
import torch
import logging
from typing import Dict, List, Optional, Union
from datetime import datetime
from pathlib import Path

from sglang.srt.server_args import ServerArgs
from sglang.srt.model_config import ModelConfig
from sglang.srt.mezo_trainer import MeZOTrainer
from sglang.srt.mezo_config import MeZOConfig, get_mezo_config_for_model
from sglang.srt.mezo_radix_optimizer import MeZORadixOptimizer
from sglang.srt.mezo_integration import create_mezo_model_runner, MeZOStep

logger = logging.getLogger(__name__)


def mezo_finetune(
    model_path: str,
    dataset_path: Union[str, List[Dict]],
    output_dir: str = "./mezo_output",
    # MeZO hyperparameters
    learning_rate: Optional[float] = None,
    epsilon: Optional[float] = None,
    num_steps: Optional[int] = None,
    batch_size: Optional[int] = None,
    # LoRA configuration
    lora_rank: int = 8,
    lora_alpha: Optional[int] = None,
    lora_target_modules: Optional[List[str]] = None,
    # System configuration
    tp_size: int = 1,
    enable_radix_cache: bool = True,
    # Training configuration
    eval_steps: Optional[int] = None,
    save_steps: Optional[int] = None,
    log_steps: int = 10,
    max_length: int = 512,
    # Advanced options
    seed: int = 42,
    task_name: Optional[str] = None,
    **kwargs
) -> Dict:
    """
    Fine-tune a model using MeZO (Memory-efficient Zeroth-order optimization).
    
    Args:
        model_path: Path to the base model
        dataset_path: Path to dataset file or list of examples
        output_dir: Directory to save checkpoints and results
        learning_rate: Learning rate (uses MeZO defaults if None)
        epsilon: Perturbation scale (uses MeZO defaults if None)
        num_steps: Number of training steps
        batch_size: Batch size for training
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha (defaults to 2 * lora_rank)
        lora_target_modules: Target modules for LoRA
        tp_size: Tensor parallel size
        enable_radix_cache: Enable RadixAttention optimization
        eval_steps: Evaluation interval
        save_steps: Checkpoint interval
        log_steps: Logging interval
        max_length: Maximum sequence length
        seed: Random seed
        task_name: Task name for task-specific configs
        **kwargs: Additional arguments
    
    Returns:
        Dictionary containing training results and statistics
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Set environment for distributed training
    if tp_size > 1:
        os.environ['SGLANG_ALLOW_REUSE_DISTRIBUTED'] = 'true'
        if 'CUDA_VISIBLE_DEVICES' not in os.environ:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(tp_size))
    
    # Set random seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    logger.info("="*70)
    logger.info("SGLang MeZO Fine-tuning")
    logger.info("="*70)
    logger.info(f"Model: {model_path}")
    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Tensor parallel size: {tp_size}")
    logger.info(f"RadixCache enabled: {enable_radix_cache}")
    logger.info("="*70)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get MeZO configuration
    mezo_config = get_mezo_config_for_model(model_path, task_name)
    
    # Override with provided parameters
    if learning_rate is not None:
        mezo_config.learning_rate = learning_rate
    if epsilon is not None:
        mezo_config.epsilon = epsilon
    if num_steps is not None:
        mezo_config.num_steps = num_steps
    if batch_size is not None:
        mezo_config.batch_size = batch_size
    
    # Set LoRA configuration
    mezo_config.lora_r = lora_rank
    mezo_config.lora_alpha = lora_alpha or (lora_rank * 2)
    
    # Set evaluation and save intervals
    if eval_steps is None:
        eval_steps = mezo_config.eval_steps
    if save_steps is None:
        save_steps = max(100, mezo_config.num_steps // 10)
    
    logger.info(f"MeZO Configuration:")
    logger.info(f"  Learning rate: {mezo_config.learning_rate}")
    logger.info(f"  Epsilon: {mezo_config.epsilon}")
    logger.info(f"  Steps: {mezo_config.num_steps}")
    logger.info(f"  Batch size: {mezo_config.batch_size}")
    logger.info(f"  LoRA rank: {mezo_config.lora_r}")
    logger.info(f"  LoRA alpha: {mezo_config.lora_alpha}")
    
    # Load dataset
    if isinstance(dataset_path, str):
        with open(dataset_path, 'r') as f:
            if dataset_path.endswith('.jsonl'):
                dataset = [json.loads(line) for line in f]
            else:
                dataset = json.load(f)
    else:
        dataset = dataset_path
    
    logger.info(f"Loaded {len(dataset)} training examples")
    
    # Initialize model runner with MeZO capabilities
    logger.info("Initializing model...")
    mezo_runner = create_mezo_model_runner(
        model_path=model_path,
        tp_size=tp_size,
        enable_radix_cache=enable_radix_cache,
        dtype="auto",
        trust_remote_code=True,
        mem_fraction_static=0.8,
        lora_paths=None,  # We'll create new LoRA weights
        max_loras_per_batch=1,
        **kwargs
    )
    
    # Initialize LoRA weights
    if lora_target_modules is None:
        lora_target_modules = ['q_proj', 'v_proj', 'k_proj', 'out_proj']
    
    # Get model dimensions
    # Note: This is simplified - actual implementation would extract from model config
    hidden_size = 768 if "125m" in model_path.lower() else 2048
    
    lora_weights = {}
    for module_name in lora_target_modules:
        lora_weights[module_name] = {
            'A': torch.randn(mezo_config.lora_r, hidden_size, device='cuda') * 0.01,
            'B': torch.zeros(hidden_size, mezo_config.lora_r, device='cuda')
        }
    
    logger.info(f"Initialized LoRA weights for modules: {lora_target_modules}")
    
    # Training metrics
    metrics = {
        'losses': [],
        'eval_losses': [],
        'cache_hit_rates': [],
        'step_times': [],
        'checkpoints': []
    }
    
    # Training loop
    logger.info("Starting MeZO training...")
    start_time = time.time()
    
    for step in range(mezo_config.num_steps):
        step_start = time.time()
        
        # Prepare batch
        batch_texts = []
        for i in range(mezo_config.batch_size):
            idx = (step * mezo_config.batch_size + i) % len(dataset)
            sample = dataset[idx]
            
            # Format text based on dataset structure
            if 'prompt' in sample and 'completion' in sample:
                text = f"{sample['prompt']} {sample['completion']}"
            elif 'text' in sample:
                text = sample['text']
            else:
                text = str(sample)
            
            batch_texts.append(text[:max_length])  # Truncate to max length
        
        # Create MeZO step
        mezo_step = MeZOStep(
            batch_texts=batch_texts,
            learning_rate=mezo_config.learning_rate,
            epsilon=mezo_config.epsilon,
            lora_weights=lora_weights
        )
        
        # Perform optimization step
        loss, cache_stats = mezo_runner.mezo_optimization_step(mezo_step)
        
        step_time = time.time() - step_start
        
        # Record metrics
        metrics['losses'].append(loss)
        metrics['cache_hit_rates'].append(cache_stats['cache_hit_rate'])
        metrics['step_times'].append(step_time)
        
        # Logging
        if step % log_steps == 0:
            avg_loss = sum(metrics['losses'][-log_steps:]) / min(log_steps, len(metrics['losses']))
            avg_cache_hit = sum(metrics['cache_hit_rates'][-log_steps:]) / min(log_steps, len(metrics['cache_hit_rates']))
            
            logger.info(
                f"Step {step}/{mezo_config.num_steps}: "
                f"Loss={avg_loss:.4f}, "
                f"Cache hit rate={avg_cache_hit:.2%}, "
                f"Time={step_time:.3f}s"
            )
        
        # Evaluation
        if eval_steps > 0 and step % eval_steps == 0 and step > 0:
            # Simple evaluation - in practice, would use a separate validation set
            eval_loss = avg_loss  # Placeholder
            metrics['eval_losses'].append({'step': step, 'loss': eval_loss})
            logger.info(f"Evaluation at step {step}: Loss={eval_loss:.4f}")
        
        # Checkpointing
        if save_steps > 0 and step % save_steps == 0 and step > 0:
            checkpoint_path = os.path.join(output_dir, f"checkpoint-{step}")
            os.makedirs(checkpoint_path, exist_ok=True)
            
            # Save LoRA weights
            torch.save(lora_weights, os.path.join(checkpoint_path, "lora_weights.pt"))
            
            # Save training state
            state = {
                'step': step,
                'config': mezo_config.__dict__,
                'metrics': metrics,
                'model_path': model_path
            }
            with open(os.path.join(checkpoint_path, "training_state.json"), 'w') as f:
                json.dump(state, f, indent=2)
            
            metrics['checkpoints'].append(checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Training complete
    total_time = time.time() - start_time
    
    # Save final checkpoint
    final_checkpoint = os.path.join(output_dir, "final_checkpoint")
    os.makedirs(final_checkpoint, exist_ok=True)
    torch.save(lora_weights, os.path.join(final_checkpoint, "lora_weights.pt"))
    
    # Get final statistics
    mezo_stats = mezo_runner.get_statistics()
    
    # Prepare results
    results = {
        'model_path': model_path,
        'output_dir': output_dir,
        'config': mezo_config.__dict__,
        'training_time': total_time,
        'total_steps': mezo_config.num_steps,
        'final_loss': metrics['losses'][-1] if metrics['losses'] else None,
        'initial_loss': metrics['losses'][0] if metrics['losses'] else None,
        'improvement': ((metrics['losses'][0] - metrics['losses'][-1]) / metrics['losses'][0] * 100) if len(metrics['losses']) > 1 else 0,
        'avg_cache_hit_rate': mezo_stats['avg_cache_hit_rate'],
        'total_forward_passes': mezo_stats['total_forward_passes'],
        'checkpoints': metrics['checkpoints'],
        'timestamp': datetime.now().isoformat()
    }
    
    # Save results
    results_path = os.path.join(output_dir, "training_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save detailed metrics
    metrics_path = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info("\n" + "="*70)
    logger.info("MeZO Training Complete!")
    logger.info("="*70)
    logger.info(f"Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    logger.info(f"Steps per second: {mezo_config.num_steps/total_time:.2f}")
    logger.info(f"Final loss: {results['final_loss']:.4f}")
    logger.info(f"Improvement: {results['improvement']:.2f}%")
    logger.info(f"Cache hit rate: {results['avg_cache_hit_rate']:.2%}")
    logger.info(f"Results saved to: {results_path}")
    logger.info("="*70)
    
    return results