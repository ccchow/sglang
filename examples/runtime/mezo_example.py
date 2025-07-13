#!/usr/bin/env python3
"""
Example script demonstrating MeZO (Memory-efficient Zeroth-order) fine-tuning with SGLang.

This script shows how to:
1. Fine-tune language models using MeZO with LoRA adapters
2. Load datasets from various sources
3. Use quantization for memory efficiency
4. Monitor training progress and performance

Requirements:
    pip install sglang datasets torch

Usage:
    python mezo_example.py
"""

import json
import logging
import time
from pathlib import Path
import torch

# SGLang imports
from sglang.srt.mezo_trainer import mezo_finetune, MeZODataset, MeZOTrainer
from sglang.srt.server_args import ServerArgs

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_sample_dataset():
    """Create a small sample dataset for demonstration."""
    samples = [
        {
            "prompt": "What is machine learning?",
            "completion": "Machine learning is a subset of artificial intelligence that enables systems to learn from data."
        },
        {
            "prompt": "Explain neural networks briefly.",
            "completion": "Neural networks are computing systems inspired by biological neural networks that learn from examples."
        },
        {
            "prompt": "What is deep learning?",
            "completion": "Deep learning uses artificial neural networks with multiple layers to progressively extract features."
        },
        {
            "prompt": "Define natural language processing.",
            "completion": "Natural language processing is the field of AI focused on enabling computers to understand human language."
        },
        {
            "prompt": "What are transformers in AI?",
            "completion": "Transformers are neural network architectures that use attention mechanisms for sequence processing."
        }
    ]
    return samples


def save_dataset_to_jsonl(dataset, filepath):
    """Save dataset to JSONL format."""
    with open(filepath, 'w') as f:
        for item in dataset:
            f.write(json.dumps(item) + '\n')
    logger.info(f"Saved {len(dataset)} examples to {filepath}")


def example_basic_training():
    """Basic MeZO fine-tuning example."""
    print("\n=== Basic MeZO Fine-tuning Example ===\n")
    
    # Create sample dataset
    train_data = create_sample_dataset()
    
    # Fine-tune model
    logger.info("Starting MeZO fine-tuning...")
    start_time = time.time()
    
    result = mezo_finetune(
        model_path="facebook/opt-125m",  # Small model for demonstration
        train_dataset=train_data,
        lora_rank=8,
        learning_rate=1e-4,
        num_steps=10,  # Small number for demo
        epsilon=1e-3,
        batch_size=2,
        max_length=128
    )
    
    elapsed = time.time() - start_time
    logger.info(f"Training completed in {elapsed:.2f} seconds")
    
    # Display results
    print(f"\nTraining Configuration:")
    for key, value in result['config'].items():
        print(f"  {key}: {value}")
    
    print(f"\nLoRA weights shape: {list(result['weights'].keys())[:3]}...")  # Show first 3 keys


def example_dataset_formats():
    """Demonstrate different dataset loading formats."""
    print("\n=== Dataset Format Examples ===\n")
    
    # 1. From list (already shown above)
    list_dataset = create_sample_dataset()
    print(f"1. List format: {len(list_dataset)} examples")
    
    # 2. From JSONL file
    jsonl_path = "sample_data.jsonl"
    save_dataset_to_jsonl(list_dataset, jsonl_path)
    print(f"2. JSONL format: saved to {jsonl_path}")
    
    # 3. From JSON file
    json_path = "sample_data.json"
    with open(json_path, 'w') as f:
        json.dump(list_dataset, f, indent=2)
    print(f"3. JSON format: saved to {json_path}")
    
    # Clean up
    Path(jsonl_path).unlink()
    Path(json_path).unlink()


def example_quantized_training():
    """Example using quantization for memory efficiency."""
    print("\n=== Quantized Model Training Example ===\n")
    
    train_data = create_sample_dataset()
    
    # Create ServerArgs with quantization
    server_args = ServerArgs(
        model_path="facebook/opt-125m",
        quantization="bitsandbytes",  # 8-bit quantization
        lora_rank=8
    )
    
    logger.info("Training with 8-bit quantization...")
    
    # Note: This requires bitsandbytes to be installed
    # In practice, you would run:
    # result = mezo_finetune(
    #     model_path="facebook/opt-125m",
    #     train_dataset=train_data,
    #     server_args=server_args,
    #     num_steps=10
    # )
    
    print("Quantization reduces memory usage by ~50% for large models")


def example_performance_analysis():
    """Analyze MeZO performance characteristics."""
    print("\n=== Performance Analysis Example ===\n")
    
    # Mock analysis for demonstration
    epsilon_values = [1e-4, 1e-3, 1e-2]
    
    print("Epsilon Analysis (simulated):")
    print("Epsilon | Time (s) | Loss Diff")
    print("--------|----------|----------")
    
    for eps in epsilon_values:
        # Simulated values
        time_taken = 2.0 + eps * 100  # Larger epsilon = slightly slower
        loss_diff = eps * 10  # Larger epsilon = bigger loss difference
        print(f"{eps:7.1e} | {time_taken:8.2f} | {loss_diff:9.4f}")
    
    print("\nSmaller epsilon values provide:")
    print("- Better KV cache reuse")
    print("- More stable gradients")
    print("- But may need more training steps")


def example_advanced_options():
    """Demonstrate advanced training options."""
    print("\n=== Advanced Options Example ===\n")
    
    # Show various configuration options
    config_examples = {
        "Memory Efficient": {
            "batch_size": 1,
            "max_length": 512,
            "quantization": "bitsandbytes",
            "lora_rank": 4
        },
        "High Quality": {
            "batch_size": 4,
            "max_length": 1024,
            "epsilon": 1e-4,
            "learning_rate": 5e-5,
            "num_steps": 1000
        },
        "Fast Training": {
            "batch_size": 8,
            "max_length": 256,
            "epsilon": 1e-2,
            "learning_rate": 1e-3,
            "num_steps": 100
        }
    }
    
    for name, config in config_examples.items():
        print(f"\n{name} Configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")


def main():
    """Run all examples."""
    print("=" * 60)
    print("SGLang MeZO Fine-tuning Examples")
    print("=" * 60)
    
    # Run examples
    example_basic_training()
    example_dataset_formats()
    example_quantized_training()
    example_performance_analysis()
    example_advanced_options()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("\nKey Takeaways:")
    print("1. MeZO enables memory-efficient fine-tuning without backpropagation")
    print("2. Works seamlessly with SGLang's inference engine")
    print("3. Supports various dataset formats and quantization methods")
    print("4. Ideal for large models on limited hardware")
    print("=" * 60)


if __name__ == "__main__":
    main()