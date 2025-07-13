#!/usr/bin/env python3
"""
Simple MeZO fine-tuning for OPT-125m using the mezo_finetune function.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python"))

import torch
import numpy as np
import json
import logging
from pathlib import Path
from typing import List, Dict

# SGLang imports
from sglang.srt.mezo_trainer import mezo_finetune

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_sst2_data(data_dir, split, max_examples=None):
    """Load SST-2 data in the format expected by MeZO."""
    file_path = f"{data_dir}/{split}.tsv"
    examples = []
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()[1:]  # Skip header
            for i, line in enumerate(lines):
                if max_examples and i >= max_examples:
                    break
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    text, label = parts
                    # Create prompt-completion format
                    prompt = f"Classify the sentiment of this movie review: {text}\nSentiment:"
                    completion = " positive" if int(label) == 1 else " negative"
                    examples.append({
                        'prompt': prompt,
                        'completion': completion
                    })
    except FileNotFoundError:
        logger.warning(f"Data file not found: {file_path}")
        logger.warning("Using synthetic data for testing")
        # Create synthetic data
        positive_texts = [
            "This movie is absolutely fantastic!",
            "I loved every moment of this brilliant film.",
            "Outstanding performances and amazing story.",
            "One of the best movies I've ever seen.",
            "Highly recommend this masterpiece!"
        ]
        negative_texts = [
            "Terrible movie, complete waste of time.",
            "I couldn't even finish watching this.",
            "Poorly written and badly acted.",
            "One of the worst films ever made.",
            "Absolutely disappointing experience."
        ]
        
        for _ in range(100):
            if np.random.random() > 0.5:
                text = np.random.choice(positive_texts)
                prompt = f"Classify the sentiment of this movie review: {text}\nSentiment:"
                completion = " positive"
            else:
                text = np.random.choice(negative_texts)
                prompt = f"Classify the sentiment of this movie review: {text}\nSentiment:"
                completion = " negative"
                
            examples.append({
                'prompt': prompt,
                'completion': completion
            })
    
    return examples


def main():
    """Run OPT-125m MeZO fine-tuning."""
    # Configuration
    model_name = "gpt2"  # Use GPT-2 since OPT is not yet fully integrated
    output_dir = "./gpt2_sst2_mezo_finetune"
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Load data
    data_dir = "/home/lei/git/princeton-nlp/MeZO/medium_models/data/k-shot-1k-test/SST-2/16-13"
    logger.info("Loading SST-2 dataset...")
    train_data = load_sst2_data(data_dir, "train", max_examples=500)
    logger.info(f"Loaded {len(train_data)} training examples")
    
    # Save training data to JSONL file
    train_file = Path(output_dir) / "train.jsonl"
    with open(train_file, 'w') as f:
        for example in train_data:
            f.write(json.dumps(example) + '\n')
    logger.info(f"Saved training data to {train_file}")
    
    # Run MeZO fine-tuning
    logger.info("Starting MeZO fine-tuning...")
    logger.info(f"Model: {model_name}")
    logger.info("Configuration:")
    logger.info("  - Learning rate: 1e-6")
    logger.info("  - Epsilon: 1e-3")
    logger.info("  - Batch size: 16")
    logger.info("  - LoRA rank: 8")
    logger.info("  - Number of steps: 1000")
    
    try:
        result = mezo_finetune(
            model_path=model_name,
            train_dataset=str(train_file),
            lora_rank=8,
            learning_rate=1e-6,
            num_steps=1000,  # Reduced for demo
            epsilon=1e-3,
            batch_size=16,
            max_length=128,
            normalize_perturbations=False,
            # Additional args
            mem_fraction_static=0.8,
            disable_cuda_graph=True,
            disable_radix_cache=False,  # Enable RadixAttention
            grammar_backend="none",  # Avoid xgrammar dependency
        )
        
        logger.info("MeZO fine-tuning completed!")
        
        # Save results
        results_path = Path(output_dir) / "results.json"
        with open(results_path, 'w') as f:
            json.dump({
                'model': model_name,
                'training_examples': len(train_data),
                'config': {
                    'learning_rate': 1e-6,
                    'epsilon': 1e-3,
                    'batch_size': 16,
                    'lora_rank': 8,
                    'num_steps': 1000,
                },
                'result': str(result) if result else "Training completed"
            }, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")
        
    except Exception as e:
        logger.error(f"Error during fine-tuning: {e}")
        import traceback
        traceback.print_exc()
        
        # Try a simpler approach with just the data
        logger.info("\nTrying direct data format...")
        try:
            result = mezo_finetune(
                model_path=model_name,
                train_dataset=train_data[:100],  # Use smaller subset
                lora_rank=8,
                learning_rate=1e-6,
                num_steps=100,  # Even fewer steps
                epsilon=1e-3,
                batch_size=8,
                max_length=128,
            )
            logger.info("Direct data format succeeded!")
        except Exception as e2:
            logger.error(f"Direct data format also failed: {e2}")


if __name__ == "__main__":
    main()