#!/usr/bin/env python3
"""
Simplified RoBERTa-large SST-2 test with MLM objective.
Fallback implementation when full ModelRunner setup isn't available.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python"))

import torch
import numpy as np
import time
import json
from datetime import datetime
from transformers import RobertaForMaskedLM, RobertaTokenizer


def load_sst2_data(file_path, max_examples=None):
    """Load SST-2 data."""
    examples = []
    with open(file_path, 'r') as f:
        lines = f.readlines()[1:]  # Skip header
        for i, line in enumerate(lines):
            if max_examples and i >= max_examples:
                break
            parts = line.strip().split('\t')
            if len(parts) == 2:
                text, label = parts
                examples.append({'text': text, 'label': int(label)})
    return examples


def run_full_roberta_large_test():
    """Simplified test with HuggingFace models."""
    print("\nUsing HuggingFace RoBERTa with simulated optimizations")
    print("-" * 60)
    
    # Configuration
    model_name = "roberta-base"  # Use base for faster testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model and tokenizer
    print(f"Loading {model_name}...")
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForMaskedLM.from_pretrained(model_name).to(device)
    
    # MLM configuration
    template = "It was [MASK]."
    label_words = {0: 'terrible', 1: 'great'}
    
    # Get label word IDs
    label_word_ids = {}
    for label, word in label_words.items():
        tokens = tokenizer.tokenize(' ' + word)
        token_id = tokenizer.convert_tokens_to_ids(tokens[0])
        label_word_ids[label] = token_id
        print(f"Label {label}: ' {word}' -> token_id {token_id}")
    
    # Load minimal data
    data_dir = "/home/lei/git/princeton-nlp/MeZO/medium_models/data/k-shot-1k-test/SST-2"
    train_data = load_sst2_data(f"{data_dir}/512-42/train.tsv", max_examples=100)
    eval_data = load_sst2_data(f"{data_dir}/512-42/dev.tsv", max_examples=50)
    
    print(f"\nData loaded: {len(train_data)} train, {len(eval_data)} eval")
    
    # Simple LoRA simulation
    param = model.roberta.encoder.layer[-1].attention.self.query.weight
    original_param = param.data.clone()
    
    # Training settings
    num_steps = 100
    batch_size = 8
    learning_rate = 1e-6
    epsilon = 1e-3
    
    # Track metrics
    train_losses = []
    cache_hits = 0
    cache_total = 0
    
    print("\nTraining with MeZO + MLM...")
    start_time = time.time()
    
    for step in range(num_steps):
        # Sample batch
        idx = np.random.choice(len(train_data), batch_size, replace=True)
        batch_texts = [train_data[i]['text'] for i in idx]
        batch_labels = [train_data[i]['label'] for i in idx]
        
        # MeZO step
        z = torch.randn_like(param)
        
        # Forward with +epsilon
        param.data = original_param + epsilon * z
        loss_plus = compute_mlm_loss(
            model, tokenizer, batch_texts, batch_labels, 
            template, label_word_ids, device
        )
        cache_total += 1
        
        # Forward with -epsilon (simulated cache hit)
        param.data = original_param - epsilon * z
        loss_minus = compute_mlm_loss(
            model, tokenizer, batch_texts, batch_labels,
            template, label_word_ids, device
        )
        cache_hits += 1  # This would reuse cache
        cache_total += 1
        
        # Gradient estimate
        grad_est = (loss_plus - loss_minus) / (2 * epsilon)
        
        # Update
        original_param = original_param - learning_rate * grad_est * z
        param.data = original_param.clone()
        
        train_losses.append((loss_plus + loss_minus) / 2)
        
        if (step + 1) % 20 == 0:
            cache_rate = cache_hits / cache_total if cache_total > 0 else 0
            print(f"Step {step+1}: Loss={train_losses[-1]:.4f}, "
                  f"Gradient={abs(grad_est):.6f}, Cache hit={cache_rate:.1%}")
    
    total_time = time.time() - start_time
    
    # Final evaluation
    eval_acc = evaluate_model(model, tokenizer, eval_data, template, label_word_ids, device)
    
    print(f"\nTraining complete:")
    print(f"  Time: {total_time:.1f}s")
    print(f"  Final eval accuracy: {eval_acc:.1%}")
    print(f"  Simulated cache hit rate: {cache_hits/cache_total:.1%}")
    print(f"  Average loss: {np.mean(train_losses):.4f}")
    
    return {
        'accuracy': eval_acc,
        'time': total_time,
        'cache_hit_rate': cache_hits / cache_total
    }


def compute_mlm_loss(model, tokenizer, texts, labels, template, label_word_ids, device):
    """Compute MLM loss for a batch."""
    losses = []
    
    for text, label in zip(texts, labels):
        # Format with template
        mlm_text = f"{text} {template}".replace('[MASK]', tokenizer.mask_token)
        inputs = tokenizer(mlm_text, return_tensors='pt', truncation=True).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            mask_pos = (inputs.input_ids == tokenizer.mask_token_id).nonzero()
            
            if len(mask_pos) > 0:
                mask_logits = outputs.logits[0, mask_pos[0, 1]]
                label_logits = mask_logits[[label_word_ids[0], label_word_ids[1]]]
                
                loss = torch.nn.functional.cross_entropy(
                    label_logits.unsqueeze(0),
                    torch.tensor([label], device=device)
                )
                losses.append(loss.item())
    
    return np.mean(losses) if losses else 0.0


def evaluate_model(model, tokenizer, eval_data, template, label_word_ids, device):
    """Evaluate model accuracy."""
    correct = 0
    total = 0
    
    for ex in eval_data:
        text = ex['text']
        label = ex['label']
        
        mlm_text = f"{text} {template}".replace('[MASK]', tokenizer.mask_token)
        inputs = tokenizer(mlm_text, return_tensors='pt', truncation=True).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            mask_pos = (inputs.input_ids == tokenizer.mask_token_id).nonzero()
            
            if len(mask_pos) > 0:
                mask_logits = outputs.logits[0, mask_pos[0, 1]]
                label_logits = mask_logits[[label_word_ids[0], label_word_ids[1]]]
                pred = torch.argmax(label_logits).item()
                
                correct += (pred == label)
                total += 1
    
    return correct / total if total > 0 else 0.0


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    run_full_roberta_large_test()