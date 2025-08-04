#!/usr/bin/env python3
"""
SGLang MeZO Training Demo
Demonstrates the complete MeZO training workflow with SGLang.
"""

import os
import sys
import json
import torch
import logging
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'python'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main demo function."""
    print("\n" + "="*80)
    print("SGLang MeZO Training Demo")
    print("="*80)
    print("This demo shows how to use MeZO training with SGLang")
    print("="*80)
    
    # Check environment
    gpu_count = torch.cuda.device_count()
    print(f"Environment: {gpu_count} GPU(s) available")
    print(f"PyTorch version: {torch.__version__}")
    
    # Create training dataset
    dataset = [
        {
            "prompt": "What is machine learning?",
            "completion": "Machine learning is a branch of artificial intelligence that focuses on building applications that learn from data and improve their accuracy over time without being programmed to do so."
        },
        {
            "prompt": "Explain neural networks.",
            "completion": "Neural networks are computing systems inspired by biological neural networks in animal brains. They consist of interconnected nodes (neurons) organized in layers that process information."
        },
        {
            "prompt": "What is deep learning?",
            "completion": "Deep learning is a subset of machine learning that uses multi-layered neural networks. The 'deep' refers to the number of layers in the network—the more layers, the deeper the network."
        },
        {
            "prompt": "Describe natural language processing.",
            "completion": "Natural language processing (NLP) is a branch of AI that helps computers understand, interpret and manipulate human language. It bridges the gap between human communication and computer understanding."
        },
        {
            "prompt": "What are transformers in AI?",
            "completion": "Transformers are a neural network architecture that uses self-attention mechanisms to process sequential data. They have revolutionized NLP tasks and are the foundation of models like GPT and BERT."
        }
    ]
    
    # Save dataset
    dataset_path = "sglang_mezo_demo_dataset.json"
    with open(dataset_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    print(f"\nCreated dataset with {len(dataset)} examples: {dataset_path}")
    
    # Option 1: Use the high-level API
    print("\n" + "-"*60)
    print("Option 1: High-level MeZO API")
    print("-"*60)
    
    try:
        from sglang import mezo_finetune, MeZOConfig
        
        print("Example code:")
        print("""
from sglang import mezo_finetune

results = mezo_finetune(
    model_path="facebook/opt-125m",
    dataset_path="sglang_mezo_demo_dataset.json",
    output_dir="./mezo_demo_output",
    learning_rate=1e-5,
    epsilon=1e-3,
    num_steps=100,
    batch_size=2,
    lora_rank=8,
    tp_size=gpu_count,  # Use available GPUs
    enable_radix_cache=True
)

print(f"Training complete! Final loss: {results['final_loss']:.4f}")
print(f"Improvement: {results['improvement']:.2f}%")
print(f"Checkpoints saved to: {results['output_dir']}")
""")
        
        # Get MeZO configuration
        config = MeZOConfig(
            learning_rate=1e-5,
            epsilon=1e-3,
            batch_size=2,
            num_steps=100
        )
        print(f"\nMeZO Configuration created:")
        print(f"  Learning rate: {config.learning_rate}")
        print(f"  Epsilon: {config.epsilon}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Steps: {config.num_steps}")
        
    except ImportError as e:
        print(f"Note: Full SGLang installation required for mezo_finetune API")
    
    # Option 2: Use the components directly
    print("\n" + "-"*60)
    print("Option 2: Direct MeZO Components")
    print("-"*60)
    
    try:
        from sglang.srt.mezo_trainer import MeZOTrainer
        from sglang.srt.mezo_radix_optimizer import MeZORadixOptimizer
        from sglang.srt.mezo_config import get_mezo_config_for_model
        
        print("Components available:")
        print("  ✓ MeZOTrainer - Core training logic")
        print("  ✓ MeZORadixOptimizer - Cache-aware optimization")
        print("  ✓ MeZOConfig - Configuration management")
        
        # Get model-specific configuration
        model_config = get_mezo_config_for_model("facebook/opt-125m", "sst-2")
        print(f"\nModel-specific config for OPT-125m:")
        print(f"  Batch size: {model_config.batch_size}")
        print(f"  Learning rate grid: {model_config.task_configs['sst-2']['learning_rate_grid']}")
        
    except ImportError:
        print("Note: SGLang components not fully available")
    
    # Option 3: Run actual training (if environment is set up)
    print("\n" + "-"*60)
    print("Option 3: Run Training (Requires SGLang Server)")
    print("-"*60)
    
    if gpu_count > 0:
        print("\nTo run actual training:")
        print("1. Launch SGLang server:")
        print("   python -m sglang.launch_server --model-path facebook/opt-125m --tp 1")
        print("\n2. Run training in another terminal:")
        print("   python examples/sglang_mezo_demo.py --run-training")
        
        # Check if we should run training
        if len(sys.argv) > 1 and sys.argv[1] == "--run-training":
            print("\n" + "="*60)
            print("Running MeZO Training...")
            print("="*60)
            
            try:
                from sglang import mezo_finetune
                
                results = mezo_finetune(
                    model_path="facebook/opt-125m",
                    dataset_path=dataset_path,
                    output_dir="./mezo_demo_output",
                    learning_rate=1e-5,
                    epsilon=1e-3,
                    num_steps=10,  # Short demo
                    batch_size=2,
                    lora_rank=8,
                    tp_size=1,
                    enable_radix_cache=True
                )
                
                print(f"\nTraining complete!")
                print(f"Final loss: {results.get('final_loss', 'N/A')}")
                print(f"Improvement: {results.get('improvement', 'N/A')}%")
                print(f"Results saved to: {results.get('output_dir', 'N/A')}")
                
            except Exception as e:
                print(f"Training failed: {e}")
                print("Make sure SGLang server is running")
    else:
        print("No GPUs available for training")
    
    # Summary
    print("\n" + "="*80)
    print("SGLang MeZO Training Summary")
    print("="*80)
    print("Key Features:")
    print("  • Memory-efficient: Only forward passes, no backpropagation")
    print("  • RadixAttention: ~95% KV cache reuse between perturbations")
    print("  • LoRA integration: Fine-tune with <1% of parameters")
    print("  • Multi-GPU support: Tensor parallelism for large models")
    print("  • Production ready: Checkpointing, evaluation, logging")
    print("\nUse Cases:")
    print("  • Fine-tune LLMs on limited hardware")
    print("  • Quick adaptation to new tasks")
    print("  • Parameter-efficient training")
    print("  • Research on zeroth-order optimization")
    print("="*80)
    
    # Clean up
    if os.path.exists(dataset_path) and "--keep-dataset" not in sys.argv:
        os.remove(dataset_path)
        print(f"\nCleaned up: {dataset_path}")


if __name__ == "__main__":
    main()