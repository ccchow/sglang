#!/usr/bin/env python3
"""
Reproduce RoBERTa-large SST-2 results using SGLang MeZO trainer.
Following the exact setup from the MeZO paper.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import torch
import numpy as np
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt

# Import SGLang MeZO components
from python.sglang.srt.mezo_trainer import mezo_finetune, MeZODataset
from python.sglang.srt.configs.model_config import ModelConfig


def prepare_sst2_dataset(data_path: str, tokenizer_name: str = "roberta-large"):
    """
    Prepare SST-2 dataset in the format expected by MeZO trainer.
    """
    from transformers import RobertaTokenizer
    
    examples = []
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
    
    # Load SST-2 data
    with open(data_path, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:  # Skip header
            parts = line.strip().split('\t')
            if len(parts) == 2:
                text, label = parts
                # Format as prompt/completion for fine-tuning
                # Following standard classification format
                examples.append({
                    'prompt': text,
                    'completion': str(label)  # "0" or "1"
                })
    
    return examples


def test_roberta_large_sst2():
    """Test MeZO on RoBERTa-large with SST-2 using SGLang trainer."""
    print("=" * 80)
    print("RoBERTa-large SST-2 Reproduction Test with SGLang MeZO")
    print("=" * 80)
    
    # Configuration from MeZO paper
    model_name = "roberta-large"
    task_name = "sst-2"
    
    # Hyperparameters from paper (Table 15)
    batch_size = 64
    learning_rate = 1e-6  # Middle of grid {1e-7, 1e-6, 1e-5}
    epsilon = 1e-3
    lora_rank = 8
    lora_alpha = 16
    num_steps = 5000  # Start with 5K steps as requested
    eval_interval = 500  # Evaluate every 500 steps
    
    print(f"\nConfiguration (MeZO paper settings):")
    print(f"  Model: {model_name}")
    print(f"  Task: {task_name}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate:.0e}")
    print(f"  Epsilon: {epsilon:.0e}")
    print(f"  LoRA rank: {lora_rank}")
    print(f"  LoRA alpha: {lora_alpha}")
    print(f"  Steps: {num_steps:,}")
    print(f"  Normalize perturbations: False (paper default)")
    
    # Prepare dataset
    print("\nPreparing SST-2 dataset...")
    data_dir = "/home/lei/git/princeton-nlp/MeZO/medium_models/data/k-shot-1k-test/SST-2"
    train_path = f"{data_dir}/512-42/train.tsv"
    
    train_examples = prepare_sst2_dataset(train_path, model_name)
    print(f"  Loaded {len(train_examples)} training examples")
    
    # Create temporary file for dataset (MeZO trainer expects file path)
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for example in train_examples:
            f.write(json.dumps(example) + '\n')
        dataset_path = f.name
    
    print(f"  Saved dataset to: {dataset_path}")
    
    # Prepare evaluation dataset
    eval_path = f"{data_dir}/512-42/dev.tsv"
    eval_examples = prepare_sst2_dataset(eval_path, model_name)
    print(f"  Loaded {len(eval_examples)} evaluation examples")
    
    # Run MeZO fine-tuning
    print(f"\nStarting MeZO fine-tuning with SGLang...")
    print("-" * 80)
    
    start_time = time.time()
    
    try:
        # Use mezo_finetune with paper settings
        result = mezo_finetune(
            model_path=model_name,
            train_dataset=dataset_path,
            lora_rank=lora_rank,
            learning_rate=learning_rate,
            num_steps=num_steps,
            epsilon=epsilon,
            batch_size=batch_size,
            normalize_perturbations=False,  # Paper doesn't normalize
            max_length=128,  # RoBERTa max for SST-2
            # Additional configs
            lora_alpha=lora_alpha,
            weight_decay=0.0,  # Paper default
            warmup_steps=0,  # No warmup in paper
            seed=42,
            eval_steps=eval_interval,
            logging_steps=100,
            save_steps=1000,
            output_dir="./mezo_roberta_large_sst2",
            # Use fp32 for reproducibility
            fp16=False,
            bf16=False,
        )
        
        training_time = time.time() - start_time
        
        print("-" * 80)
        print(f"\nTraining completed in {training_time:.1f} seconds ({training_time/60:.1f} minutes)")
        
        # Analyze results
        if 'training_history' in result:
            history = result['training_history']
            
            # Plot training curves
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Loss curve
            if 'loss' in history:
                steps = history['steps']
                losses = history['loss']
                ax1.plot(steps, losses, 'b-')
                ax1.set_xlabel('Steps')
                ax1.set_ylabel('Training Loss')
                ax1.set_title('Training Loss Over Time')
                ax1.grid(True, alpha=0.3)
            
            # Evaluation metrics
            if 'eval_loss' in history:
                eval_steps = history['eval_steps']
                eval_losses = history['eval_loss']
                ax2.plot(eval_steps, eval_losses, 'r-', marker='o', label='Eval Loss')
                ax2.set_xlabel('Steps')
                ax2.set_ylabel('Evaluation Loss')
                ax2.set_title('Evaluation Metrics')
                ax2.grid(True, alpha=0.3)
                
                if 'eval_accuracy' in history:
                    ax2_twin = ax2.twinx()
                    eval_accs = history['eval_accuracy']
                    ax2_twin.plot(eval_steps, eval_accs, 'g-', marker='s', label='Accuracy')
                    ax2_twin.set_ylabel('Accuracy')
                    ax2_twin.set_ylim(0, 1)
            
            plt.tight_layout()
            plt.savefig('roberta_large_sst2_training.png', dpi=150)
            print(f"\nTraining curves saved to: roberta_large_sst2_training.png")
        
        # Report final metrics
        print("\nFinal Results:")
        print("-" * 40)
        if 'final_loss' in result:
            print(f"  Final training loss: {result['final_loss']:.4f}")
        if 'final_eval_loss' in result:
            print(f"  Final eval loss: {result['final_eval_loss']:.4f}")
        if 'final_eval_accuracy' in result:
            print(f"  Final eval accuracy: {result['final_eval_accuracy']:.1%}")
        
        # Save LoRA weights
        if 'lora_weights' in result:
            weights_path = "./roberta_large_sst2_lora_weights.pt"
            torch.save(result['lora_weights'], weights_path)
            print(f"\nLoRA weights saved to: {weights_path}")
        
        # Cleanup
        os.unlink(dataset_path)
        
        return result
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        
        # Cleanup
        if os.path.exists(dataset_path):
            os.unlink(dataset_path)
        
        return None


def evaluate_trained_model(model_path: str, lora_weights_path: str, eval_examples):
    """Evaluate the trained model on test data."""
    from transformers import RobertaForSequenceClassification, RobertaTokenizer
    
    print("\nEvaluating trained model...")
    
    # Load model and tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = RobertaForSequenceClassification.from_pretrained(
        model_path,
        num_labels=2
    )
    
    # Load LoRA weights
    if os.path.exists(lora_weights_path):
        lora_state = torch.load(lora_weights_path)
        # Apply LoRA weights to model
        # (Implementation depends on how weights are stored)
        print(f"  Loaded LoRA weights from: {lora_weights_path}")
    
    # Evaluate
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for example in eval_examples:
            inputs = tokenizer(
                example['prompt'],
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors='pt'
            )
            
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=-1).item()
            label = int(example['completion'])
            
            correct += (pred == label)
            total += 1
    
    accuracy = correct / total
    print(f"  Test accuracy: {accuracy:.1%} ({correct}/{total})")
    
    return accuracy


if __name__ == "__main__":
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        torch.cuda.manual_seed(42)
    else:
        print("WARNING: Running on CPU, this will be very slow")
    
    # Run the test
    result = test_roberta_large_sst2()
    
    if result:
        print("\n" + "=" * 80)
        print("Test completed successfully!")
        
        # Expected results from paper (Table 18):
        # RoBERTa-large on SST-2 with MeZO: ~92% accuracy after 100K steps
        # With 5K steps, we expect to see some improvement but not full convergence
        
        print("\nExpected results from MeZO paper:")
        print("  RoBERTa-large SST-2 (100K steps): ~92% accuracy")
        print("  Our test (5K steps): Should show improvement trend")
    else:
        print("\n" + "=" * 80)
        print("Test failed. Please check the error messages above.")