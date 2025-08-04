#!/usr/bin/env python3
"""
Train OPT model using SGLang with MeZO, LoRA, and tensor parallelism (TP=2).
This script uses the actual SGLang ModelRunner with our MeZO implementation.
"""

import os
import sys
import json
import time
import torch
import argparse
import logging
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_dataset():
    """Create training dataset."""
    return [
        {
            "prompt": "What is machine learning?",
            "completion": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."
        },
        {
            "prompt": "Explain climate change briefly.",
            "completion": "Climate change refers to long-term shifts in global temperatures and weather patterns, primarily driven by human activities like burning fossil fuels."
        },
        {
            "prompt": "What is the speed of light?",
            "completion": "The speed of light in vacuum is approximately 299,792,458 meters per second, a fundamental constant in physics denoted by 'c'."
        },
        {
            "prompt": "Describe photosynthesis.",
            "completion": "Photosynthesis is the process by which plants and other organisms convert light energy into chemical energy, producing oxygen and glucose from carbon dioxide and water."
        },
        {
            "prompt": "What is DNA?",
            "completion": "DNA (Deoxyribonucleic acid) is a molecule that contains the genetic instructions for the development, functioning, growth and reproduction of all known living organisms."
        }
    ]


def main():
    parser = argparse.ArgumentParser(description="Train OPT with MeZO using SGLang")
    parser.add_argument("--model-path", type=str, default="facebook/opt-125m",
                        help="Path to model")
    parser.add_argument("--tp-size", type=int, default=2,
                        help="Tensor parallel size")
    parser.add_argument("--num-steps", type=int, default=100,
                        help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--epsilon", type=float, default=1e-3,
                        help="MeZO epsilon")
    parser.add_argument("--lora-rank", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--max-length", type=int, default=256,
                        help="Maximum sequence length")
    parser.add_argument("--output-dir", type=str, default="./sglang_mezo_results",
                        help="Output directory")
    
    args = parser.parse_args()
    
    # Set environment variables
    os.environ['SGLANG_ALLOW_REUSE_DISTRIBUTED'] = 'true'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # Use first 2 GPUs
    
    logger.info("="*70)
    logger.info("SGLang MeZO Training with Tensor Parallelism")
    logger.info("="*70)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"TP Size: {args.tp_size}")
    logger.info(f"Steps: {args.num_steps}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Learning Rate: {args.learning_rate}")
    logger.info(f"Epsilon: {args.epsilon}")
    logger.info(f"LoRA Rank: {args.lora_rank}")
    logger.info("="*70)
    
    try:
        # Import SGLang components
        from sglang.srt.server_args import ServerArgs
        from sglang.srt.model_runner import ModelRunner
        from sglang.srt.mezo_trainer import MeZOTrainer
        from sglang.srt.managers.lora_manager import LoRAManager
        from sglang.srt.mezo_radix_optimizer import MeZORadixOptimizer
        
        # Create server arguments
        server_args = ServerArgs(
            model_path=args.model_path,
            tokenizer_path=args.model_path,
            tp_size=args.tp_size,
            mem_fraction_static=0.8,
            trust_remote_code=True,
            disable_disk_cache=True,
            enable_lora=True,
            lora_rank=args.lora_rank,
            max_loras=1,
            log_level="info"
        )
        
        logger.info("Initializing SGLang ModelRunner...")
        
        # Initialize model runner
        model_runner = ModelRunner(
            model_config=server_args.model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            tp_size=server_args.tp_size,
            tp_rank=0,  # Will be set properly in distributed context
            tp_group=None,  # Will be initialized
            nccl_port=server_args.nccl_port,
            server_args=server_args
        )
        
        # Initialize LoRA manager
        lora_manager = LoRAManager(
            base_model=model_runner.model,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj"]
        )
        
        # Create new LoRA adapter for training
        lora_name = f"mezo_lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        lora_id = lora_manager.add_lora(lora_name)
        
        logger.info(f"Created LoRA adapter: {lora_name} (ID: {lora_id})")
        
        # Initialize MeZO trainer
        mezo_trainer = MeZOTrainer(
            model_runner=model_runner,
            lora_manager=lora_manager,
            lora_id=lora_id,
            learning_rate=args.learning_rate,
            epsilon=args.epsilon
        )
        
        # Initialize RadixCache optimizer
        radix_optimizer = MeZORadixOptimizer(
            model_runner=model_runner,
            epsilon=args.epsilon,
            cache_config={
                'enable_prefix_caching': True,
                'context_window': args.max_length,
                'block_size': 16
            }
        )
        
        # Create dataset
        dataset = create_dataset()
        dataset_path = os.path.join(args.output_dir, "training_data.json")
        os.makedirs(args.output_dir, exist_ok=True)
        
        with open(dataset_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        logger.info(f"Created dataset with {len(dataset)} examples")
        
        # Training configuration
        training_config = {
            'num_steps': args.num_steps,
            'batch_size': args.batch_size,
            'max_length': args.max_length,
            'log_interval': 10,
            'checkpoint_interval': 50,
            'evaluation_interval': 25,
            'output_dir': args.output_dir,
            'enable_radix_optimization': True
        }
        
        logger.info("Starting MeZO training...")
        start_time = time.time()
        
        # Training metrics
        metrics = {
            'losses': [],
            'cache_hit_rates': [],
            'step_times': [],
            'gradient_norms': []
        }
        
        # Training loop
        for step in range(args.num_steps):
            step_start = time.time()
            
            # Get batch
            batch_idx = step % len(dataset)
            sample = dataset[batch_idx]
            
            # Prepare input
            text = f"{sample['prompt']} {sample['completion']}"
            
            if radix_optimizer:
                # Use RadixCache-optimized forward passes
                loss, cache_stats = radix_optimizer.mezo_step_with_cache(
                    trainer=mezo_trainer,
                    batch_texts=[text] * args.batch_size,
                    learning_rate=args.learning_rate
                )
                
                cache_hit_rate = cache_stats.get('cache_hit_rate', 0.0)
            else:
                # Standard MeZO step
                loss = mezo_trainer.mezo_step(
                    batch_texts=[text] * args.batch_size,
                    learning_rate=args.learning_rate
                )
                cache_hit_rate = 0.0
            
            step_time = time.time() - step_start
            
            # Record metrics
            metrics['losses'].append(loss)
            metrics['cache_hit_rates'].append(cache_hit_rate)
            metrics['step_times'].append(step_time)
            
            # Log progress
            if step % training_config['log_interval'] == 0:
                avg_loss = sum(metrics['losses'][-10:]) / min(10, len(metrics['losses']))
                avg_cache_hit = sum(metrics['cache_hit_rates'][-10:]) / min(10, len(metrics['cache_hit_rates']))
                
                logger.info(
                    f"Step {step}/{args.num_steps}: "
                    f"Loss={avg_loss:.4f}, "
                    f"Cache={avg_cache_hit:.2%}, "
                    f"Time={step_time:.3f}s"
                )
            
            # Checkpoint
            if step > 0 and step % training_config['checkpoint_interval'] == 0:
                checkpoint_path = os.path.join(
                    args.output_dir,
                    f"checkpoint_step_{step}.pt"
                )
                lora_manager.save_lora(lora_id, checkpoint_path)
                logger.info(f"Saved checkpoint: {checkpoint_path}")
            
            # Evaluation
            if step > 0 and step % training_config['evaluation_interval'] == 0:
                logger.info(f"Evaluation at step {step}:")
                logger.info(f"  Average loss: {avg_loss:.4f}")
                logger.info(f"  Cache hit rate: {avg_cache_hit:.2%}")
                
                if radix_optimizer:
                    optimizer_stats = radix_optimizer.get_statistics()
                    logger.info(f"  Total cache hits: {optimizer_stats['total_cache_hits']}")
                    logger.info(f"  Memory saved: {optimizer_stats['memory_saved_mb']:.0f} MB")
        
        # Training complete
        training_time = time.time() - start_time
        
        # Save final model
        final_checkpoint = os.path.join(args.output_dir, "final_checkpoint.pt")
        lora_manager.save_lora(lora_id, final_checkpoint)
        
        # Compute final metrics
        initial_loss = metrics['losses'][0] if metrics['losses'] else 0
        final_loss = metrics['losses'][-1] if metrics['losses'] else 0
        improvement = (initial_loss - final_loss) / initial_loss * 100 if initial_loss > 0 else 0
        avg_cache_hit_rate = sum(metrics['cache_hit_rates']) / len(metrics['cache_hit_rates']) if metrics['cache_hit_rates'] else 0
        
        # Print results
        print("\n" + "="*70)
        print("SGLang MeZO Training Complete!")
        print("="*70)
        print(f"Model: {args.model_path}")
        print(f"Tensor Parallel Size: {args.tp_size}")
        print(f"Total Steps: {args.num_steps}")
        print(f"Total Time: {training_time:.2f}s")
        print(f"Avg Time/Step: {training_time/args.num_steps:.3f}s")
        print(f"\nResults:")
        print(f"  Initial Loss: {initial_loss:.4f}")
        print(f"  Final Loss: {final_loss:.4f}")
        print(f"  Improvement: {improvement:.2f}%")
        print(f"  Avg Cache Hit Rate: {avg_cache_hit_rate:.2%}")
        print(f"\nCheckpoints saved to: {args.output_dir}")
        print("="*70)
        
        # Save training summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'config': vars(args),
            'training_config': training_config,
            'results': {
                'initial_loss': initial_loss,
                'final_loss': final_loss,
                'improvement_pct': improvement,
                'avg_cache_hit_rate': avg_cache_hit_rate,
                'training_time': training_time,
                'steps_per_second': args.num_steps / training_time
            },
            'metrics': {
                'losses': metrics['losses'],
                'cache_hit_rates': metrics['cache_hit_rates'],
                'step_times': metrics['step_times']
            }
        }
        
        summary_path = os.path.join(args.output_dir, 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Training summary saved to: {summary_path}")
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback to direct implementation test
        logger.info("\nFalling back to direct implementation test...")
        from examples.mezo_opt125m_100steps import main as run_direct
        run_direct()


if __name__ == "__main__":
    main()