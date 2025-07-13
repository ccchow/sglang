#!/usr/bin/env python3
"""
Fast test of SGLang MeZO trainer with MLM objective.
Optimized for speed with fewer evaluations and efficient batching.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python"))

import torch
import numpy as np
import time
from datetime import datetime
from transformers import RobertaTokenizer, RobertaForMaskedLM
import torch.nn.functional as F

# Import SGLang MeZO components  
from sglang.srt.mezo_mlm_trainer import MLMConfig, create_mlm_config_for_task


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


def run_fast_mlm_test(num_steps=5000):
    """Run fast MeZO MLM test."""
    print("=" * 80)
    print(f"SGLang MeZO MLM Fast Test - {num_steps} Steps on SST-2")
    print("=" * 80)
    
    # Configuration
    model_name = "roberta-base"
    batch_size = 64
    learning_rate = 1e-6
    epsilon = 1e-3
    lora_rank = 8
    lora_alpha = 16
    
    # Create MLM config
    mlm_config = create_mlm_config_for_task("sst-2")
    
    print(f"\nConfiguration:")
    print(f"  Model: {model_name}")
    print(f"  MLM template: {mlm_config.template}")
    print(f"  Label words: {mlm_config.label_word_mapping}")
    print(f"  Steps: {num_steps}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForMaskedLM.from_pretrained(model_name).to(device)
    
    # Get label word IDs
    label_word_ids = []
    for label in [0, 1]:
        word = mlm_config.label_word_mapping[label]
        tokens = tokenizer.tokenize(' ' + word)  # Space prefix for RoBERTa
        token_id = tokenizer.convert_tokens_to_ids(tokens[0])
        label_word_ids.append(token_id)
    print(f"  Label word IDs: terrible={label_word_ids[0]}, great={label_word_ids[1]}")
    
    # Create simplified LoRA parameters
    lora_params = []
    layer_count = 0
    for name, module in model.named_modules():
        if 'attention' in name and 'self.query' in name:
            layer = module
            if hasattr(layer, 'weight'):
                # Add LoRA matrices
                lora_A = torch.nn.Parameter(
                    torch.randn(lora_rank, layer.weight.shape[1], device=device) * 0.01
                )
                lora_B = torch.nn.Parameter(
                    torch.zeros(layer.weight.shape[0], lora_rank, device=device)
                )
                lora_params.extend([lora_A, lora_B])
                layer_count += 1
                
                # Store original weight
                layer.original_weight = layer.weight.data.clone()
                layer.lora_A = lora_A
                layer.lora_B = lora_B
                layer.lora_scale = lora_alpha / lora_rank
    
    print(f"  LoRA layers: {layer_count}")
    print(f"  LoRA parameters: {sum(p.numel() for p in lora_params):,}")
    
    # Load data
    data_dir = "/home/lei/git/princeton-nlp/MeZO/medium_models/data/k-shot-1k-test/SST-2"
    train_data = load_sst2_data(f"{data_dir}/512-42/train.tsv")
    eval_data = load_sst2_data(f"{data_dir}/512-42/dev.tsv", max_examples=100)
    
    print(f"\nData loaded:")
    print(f"  Train: {len(train_data)} examples")
    print(f"  Eval: {len(eval_data)} examples")
    
    # Quick initial evaluation
    model.eval()
    correct = 0
    for i in range(min(20, len(eval_data))):
        ex = eval_data[i]
        text = ex['text'] + " It was " + tokenizer.mask_token + "."
        inputs = tokenizer(text, return_tensors='pt', truncation=True).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            mask_pos = (inputs.input_ids == tokenizer.mask_token_id).nonzero()[0, 1]
            mask_logits = outputs.logits[0, mask_pos]
            label_logits = mask_logits[label_word_ids]
            pred = torch.argmax(label_logits).item()
            correct += (pred == ex['label'])
    
    init_acc = correct / min(20, len(eval_data))
    print(f"\nInitial accuracy (20 examples): {init_acc:.1%}")
    
    # Training
    print(f"\nTraining for {num_steps} steps...")
    print("-" * 60)
    print("Step  | Loss    | Gradient | Progress | Time")
    print("-" * 60)
    
    def apply_lora():
        """Apply LoRA weights to model."""
        for name, module in model.named_modules():
            if 'attention' in name and 'self.query' in name:
                layer = module
                if hasattr(layer, 'weight') and hasattr(layer, 'lora_A'):
                    layer.weight.data = layer.original_weight + layer.lora_scale * (layer.lora_B @ layer.lora_A)
    
    model.train()
    start_time = time.time()
    losses = []
    gradients = []
    
    for step in range(num_steps):
        # Sample batch
        idx = np.random.choice(len(train_data), batch_size, replace=True)
        texts = []
        labels = []
        
        for i in idx:
            text = train_data[i]['text'] + " It was " + tokenizer.mask_token + "."
            texts.append(text)
            labels.append(train_data[i]['label'])
        
        # Tokenize batch
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
        labels_tensor = torch.tensor(labels, device=device)
        
        # MeZO step
        z_list = [torch.randn_like(p) for p in lora_params]
        
        # Forward with +epsilon
        for i, p in enumerate(lora_params):
            p.data.add_(epsilon * z_list[i])
        apply_lora()
        
        outputs = model(**inputs)
        mask_positions = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)
        batch_indices = mask_positions[0]
        position_indices = mask_positions[1]
        
        mask_logits = outputs.logits[batch_indices, position_indices]
        label_logits = mask_logits[:, label_word_ids]
        loss_plus = F.cross_entropy(label_logits, labels_tensor).item()
        
        # Forward with -epsilon
        for i, p in enumerate(lora_params):
            p.data.add_(-2 * epsilon * z_list[i])
        apply_lora()
        
        outputs = model(**inputs)
        mask_logits = outputs.logits[batch_indices, position_indices]
        label_logits = mask_logits[:, label_word_ids]
        loss_minus = F.cross_entropy(label_logits, labels_tensor).item()
        
        # Gradient estimate and update
        grad_est = (loss_plus - loss_minus) / (2 * epsilon)
        
        for i, p in enumerate(lora_params):
            p.data.add_(epsilon * z_list[i])  # Restore
            p.data.add_(-learning_rate * grad_est * z_list[i])  # Update
        
        avg_loss = (loss_plus + loss_minus) / 2
        losses.append(avg_loss)
        gradients.append(abs(grad_est))
        
        # Print progress
        if (step + 1) % 500 == 0:
            elapsed = time.time() - start_time
            progress = (step + 1) / num_steps * 100
            recent_loss = np.mean(losses[-100:])
            recent_grad = np.mean(gradients[-100:])
            print(f"{step+1:5d} | {recent_loss:7.4f} | {recent_grad:8.5f} | {progress:6.1f}% | {elapsed:4.0f}s")
    
    print("-" * 60)
    apply_lora()
    
    # Final evaluation
    print("\nFinal evaluation...")
    model.eval()
    correct = 0
    total_loss = 0
    
    for ex in eval_data[:50]:  # Evaluate on 50 examples
        text = ex['text'] + " It was " + tokenizer.mask_token + "."
        inputs = tokenizer(text, return_tensors='pt', truncation=True).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            mask_pos = (inputs.input_ids == tokenizer.mask_token_id).nonzero()
            if len(mask_pos) > 0:
                mask_pos = mask_pos[0, 1]
                mask_logits = outputs.logits[0, mask_pos]
                label_logits = mask_logits[label_word_ids]
                pred = torch.argmax(label_logits).item()
                correct += (pred == ex['label'])
                
                label_tensor = torch.tensor([ex['label']], device=device)
                loss = F.cross_entropy(label_logits.unsqueeze(0), label_tensor)
                total_loss += loss.item()
    
    final_acc = correct / 50
    final_loss = total_loss / 50
    total_time = time.time() - start_time
    
    # Results
    print(f"\nResults:")
    print(f"  Initial accuracy: ~{init_acc:.1%}")
    print(f"  Final accuracy: {final_acc:.1%}")
    print(f"  Improvement: {(final_acc - init_acc) * 100:+.2f}pp")
    print(f"  Final loss: {final_loss:.4f}")
    print(f"  Average gradient: {np.mean(gradients):.5f}")
    print(f"  Training time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    
    print("\n" + "=" * 80)
    if final_acc > init_acc + 0.05:
        print("✅ SUCCESS: Significant improvement with MLM approach!")
    elif final_acc > init_acc:
        print("📈 PROGRESS: Some improvement observed")
    else:
        print("⚠️  No clear improvement - may need more steps")
    print("=" * 80)
    
    # Show loss trend
    print("\nLoss trend (last 500 steps):")
    for i in range(0, min(500, len(losses)), 100):
        window = losses[-(500-i):-(400-i) if i < 400 else None]
        if window:
            print(f"  Steps {num_steps-500+i}-{num_steps-500+i+100}: {np.mean(window):.4f}")
    
    return {
        'initial_acc': init_acc,
        'final_acc': final_acc,
        'improvement': final_acc - init_acc,
        'final_loss': final_loss,
        'avg_gradient': np.mean(gradients),
        'training_time': total_time
    }


if __name__ == "__main__":
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Run test
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    results = run_fast_mlm_test(num_steps=5000)
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")