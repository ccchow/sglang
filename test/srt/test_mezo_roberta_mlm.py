#!/usr/bin/env python3
"""
Test MeZO with RoBERTa using Masked Language Model (MLM) for SST-2.
Following the original MeZO implementation that uses MLM head for classification.
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformers import RobertaTokenizer, RobertaForMaskedLM
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_sst2_mlm(file_path, tokenizer, label_words=["terrible", "great"]):
    """
    Load SST-2 examples and format them for MLM-based classification.
    Uses prompt template: "{sentence} It was [MASK]."
    """
    examples = []
    
    # Get token IDs for label words
    label_word_ids = []
    for word in label_words:
        # Tokenize and ensure it's a single token
        tokens = tokenizer.tokenize(word)
        assert len(tokens) == 1, f"Label word '{word}' tokenizes to multiple tokens: {tokens}"
        token_id = tokenizer.convert_tokens_to_ids(tokens[0])
        label_word_ids.append(token_id)
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        # Skip header
        for line in lines[1:]:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                sentence, label = parts
                label = int(label)
                
                # Apply prompt template - use actual mask token
                text = f"{sentence} It was {tokenizer.mask_token}."
                
                examples.append({
                    'text': text,
                    'label': label,
                    'label_word_id': label_word_ids[label],
                    'original_sentence': sentence
                })
    
    return examples, label_word_ids


def create_lora_roberta_mlm(model, rank=8):
    """Add LoRA adapters to RoBERTa MLM model."""
    lora_params = []
    device = model.device
    
    # Add LoRA to both attention and FFN layers (following original MeZO)
    for name, module in model.named_modules():
        # Attention layers
        if 'attention' in name and hasattr(module, 'self'):
            if hasattr(module.self, 'query'):
                in_features = module.self.query.in_features
                out_features = module.self.query.out_features
                
                # Query LoRA
                lora_A_q = torch.nn.Parameter(torch.randn(rank, in_features, device=device) * 0.01)
                lora_B_q = torch.nn.Parameter(torch.zeros(out_features, rank, device=device))
                
                # Value LoRA
                lora_A_v = torch.nn.Parameter(torch.randn(rank, in_features, device=device) * 0.01)
                lora_B_v = torch.nn.Parameter(torch.zeros(out_features, rank, device=device))
                
                # Store original weights
                module.self.query.original_weight = module.self.query.weight.data.clone()
                module.self.value.original_weight = module.self.value.weight.data.clone()
                
                module.self.query.lora_A = lora_A_q
                module.self.query.lora_B = lora_B_q
                module.self.value.lora_A = lora_A_v
                module.self.value.lora_B = lora_B_v
                
                lora_params.extend([lora_A_q, lora_B_q, lora_A_v, lora_B_v])
        
        # FFN layers (following original MeZO)
        elif 'intermediate.dense' in name and isinstance(module, torch.nn.Linear):
            in_features = module.in_features
            out_features = module.out_features
            
            lora_A = torch.nn.Parameter(torch.randn(rank, in_features, device=device) * 0.01)
            lora_B = torch.nn.Parameter(torch.zeros(out_features, rank, device=device))
            
            module.original_weight = module.weight.data.clone()
            module.lora_A = lora_A
            module.lora_B = lora_B
            
            lora_params.extend([lora_A, lora_B])
    
    return lora_params


def apply_lora_weights_mlm(model, scaling=1.0):
    """Apply LoRA weights to the MLM model."""
    for name, module in model.named_modules():
        # Attention layers
        if 'attention' in name and hasattr(module, 'self'):
            if hasattr(module.self.query, 'lora_A'):
                module.self.query.weight.data = module.self.query.original_weight + \
                    scaling * (module.self.query.lora_B @ module.self.query.lora_A)
                
                module.self.value.weight.data = module.self.value.original_weight + \
                    scaling * (module.self.value.lora_B @ module.self.value.lora_A)
        
        # FFN layers
        elif hasattr(module, 'lora_A') and hasattr(module, 'original_weight'):
            module.weight.data = module.original_weight + \
                scaling * (module.lora_B @ module.lora_A)


def compute_mlm_accuracy_objective(model, inputs, mask_positions, label_word_ids, labels):
    """
    Compute accuracy using MLM predictions at mask positions.
    Returns negative accuracy for minimization.
    """
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        
        # Get logits at mask positions
        batch_size = logits.size(0)
        mask_logits = []
        
        for i in range(batch_size):
            mask_pos = mask_positions[i]
            mask_logits.append(logits[i, mask_pos])
        
        mask_logits = torch.stack(mask_logits)  # [batch_size, vocab_size]
        
        # Get logits for label words only
        label_logits = mask_logits[:, label_word_ids]  # [batch_size, num_labels]
        
        # Get predictions
        preds = torch.argmax(label_logits, dim=-1)
        
        # Compute accuracy
        correct = (preds == labels).float()
        accuracy = correct.mean()
        
        return -accuracy  # Negative for minimization


def mezo_step_mlm(model, tokenizer, batch, lora_params, label_word_ids, epsilon=1e-3, lr=1e-6):
    """Perform one MeZO step using MLM accuracy objective."""
    # Sample perturbation
    z_list = [torch.randn_like(p) for p in lora_params]
    
    # Normalize perturbations
    z_list = [z / (z.norm() + 1e-8) for z in z_list]
    
    # Prepare inputs
    inputs = tokenizer(
        batch['text'], 
        padding=True, 
        truncation=True, 
        max_length=128,
        return_tensors='pt'
    ).to(model.device)
    
    # Find mask positions
    mask_token_id = tokenizer.mask_token_id
    mask_positions = []
    for i in range(inputs.input_ids.size(0)):
        mask_pos = (inputs.input_ids[i] == mask_token_id).nonzero(as_tuple=True)[0]
        if len(mask_pos) > 0:
            mask_positions.append(mask_pos[0].item())
        else:
            mask_positions.append(0)  # Fallback
    
    labels = torch.tensor(batch['label']).to(model.device)
    
    # Apply positive perturbation
    for i, p in enumerate(lora_params):
        p.data.add_(epsilon * z_list[i])
    apply_lora_weights_mlm(model)
    
    # Forward pass with +epsilon
    model.eval()
    obj_plus = compute_mlm_accuracy_objective(model, inputs, mask_positions, label_word_ids, labels)
    
    # Apply negative perturbation
    for i, p in enumerate(lora_params):
        p.data.add_(-2 * epsilon * z_list[i])
    apply_lora_weights_mlm(model)
    
    # Forward pass with -epsilon
    obj_minus = compute_mlm_accuracy_objective(model, inputs, mask_positions, label_word_ids, labels)
    
    # Restore parameters
    for i, p in enumerate(lora_params):
        p.data.add_(epsilon * z_list[i])
    apply_lora_weights_mlm(model)
    
    # Compute gradient estimate
    grad_estimate = (obj_plus - obj_minus) / (2 * epsilon)
    
    # Update parameters
    for i, p in enumerate(lora_params):
        p.data.add_(-lr * grad_estimate * z_list[i])
    
    # Return positive accuracy for logging
    return -(obj_plus + obj_minus) / 2


def evaluate_mlm_sst2(model, tokenizer, examples, label_word_ids):
    """Evaluate MLM-based classification on SST-2."""
    model.eval()
    correct = 0
    
    mask_token_id = tokenizer.mask_token_id
    
    with torch.no_grad():
        for example in examples:
            inputs = tokenizer(
                example['text'],
                truncation=True,
                max_length=128,
                return_tensors='pt'
            ).to(model.device)
            
            # Find mask position
            mask_pos = (inputs.input_ids[0] == mask_token_id).nonzero(as_tuple=True)[0]
            if len(mask_pos) == 0:
                continue
            mask_pos = mask_pos[0].item()
            
            # Get predictions
            outputs = model(**inputs)
            logits = outputs.logits[0, mask_pos]  # [vocab_size]
            
            # Get logits for label words
            label_logits = logits[label_word_ids]
            pred = torch.argmax(label_logits).item()
            
            correct += (pred == example['label'])
    
    accuracy = correct / len(examples)
    return accuracy


def test_mezo_roberta_mlm():
    """Test MeZO with RoBERTa MLM on SST-2."""
    print("=" * 60)
    print("MeZO RoBERTa SST-2 Test (MLM-based Classification)")
    print("=" * 60)
    
    # Configuration
    model_name = "roberta-base"
    batch_size = 16
    learning_rate = 1e-3
    epsilon = 1e-3
    num_steps = 1000  # More steps for MLM
    eval_steps = 100
    label_words = [" terrible", " great"]  # Negative, Positive - space prefix for RoBERTa
    
    print(f"\nConfiguration:")
    print(f"  Model: {model_name} (MLM)")
    print(f"  Task: SST-2 (512-shot)")
    print(f"  Label words: {label_words}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Epsilon: {epsilon}")
    print(f"  Steps: {num_steps}")
    
    # Load model and tokenizer
    print("\nLoading MLM model and tokenizer...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForMaskedLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32
    ).to(device)
    
    # Freeze base model
    for param in model.parameters():
        param.requires_grad = False
    
    # Add LoRA adapters
    print("Adding LoRA adapters (attention + FFN)...")
    lora_params = create_lora_roberta_mlm(model, rank=8)
    print(f"  Number of LoRA parameters: {len(lora_params)}")
    print(f"  Total LoRA parameters: {sum(p.numel() for p in lora_params):,}")
    
    # Load datasets
    print("\nLoading datasets...")
    data_dir = "/home/lei/git/princeton-nlp/MeZO/medium_models/data/k-shot-1k-test/SST-2"
    train_path = f"{data_dir}/512-42/train.tsv"
    dev_path = f"{data_dir}/512-42/dev.tsv"
    
    train_examples, label_word_ids = load_sst2_mlm(train_path, tokenizer, label_words)
    eval_examples, _ = load_sst2_mlm(dev_path, tokenizer, label_words)
    
    print(f"  Train examples: {len(train_examples)}")
    print(f"  Eval examples: {len(eval_examples)}")
    print(f"  Label word IDs: {label_word_ids}")
    
    # Initial evaluation
    apply_lora_weights_mlm(model)
    initial_acc = evaluate_mlm_sst2(model, tokenizer, eval_examples, label_word_ids)
    print(f"\nInitial accuracy: {initial_acc:.1%}")
    
    # Training
    print("\nStarting MeZO training with MLM...")
    print("-" * 60)
    print("Step | Train Acc | Eval Acc | Improvement")
    print("-" * 60)
    
    best_acc = initial_acc
    
    for step in range(num_steps):
        # Sample batch
        batch_indices = np.random.choice(len(train_examples), batch_size, replace=True)
        batch = {
            'text': [train_examples[i]['text'] for i in batch_indices],
            'label': [train_examples[i]['label'] for i in batch_indices]
        }
        
        # MeZO step
        train_acc = mezo_step_mlm(model, tokenizer, batch, lora_params, label_word_ids, epsilon, learning_rate)
        
        # Evaluation
        if (step + 1) % eval_steps == 0:
            apply_lora_weights_mlm(model)
            eval_acc = evaluate_mlm_sst2(model, tokenizer, eval_examples, label_word_ids)
            best_acc = max(best_acc, eval_acc)
            
            improvement = (eval_acc - initial_acc) * 100
            print(f"{step+1:4d} | {train_acc:9.2%} | {eval_acc:8.2%} | {improvement:+.1f}pp")
    
    print("-" * 60)
    
    # Final evaluation
    apply_lora_weights_mlm(model)
    final_acc = evaluate_mlm_sst2(model, tokenizer, eval_examples, label_word_ids)
    
    print("\nTraining Summary:")
    print(f"  Initial accuracy: {initial_acc:.1%}")
    print(f"  Final accuracy: {final_acc:.1%}")
    print(f"  Best accuracy: {best_acc:.1%}")
    print(f"  Improvement: {(final_acc - initial_acc) * 100:+.1f}pp")
    
    # Success criteria
    success = final_acc > initial_acc + 0.05
    
    print("\n" + "=" * 60)
    if success:
        print("✅ MeZO ROBERTA MLM TEST: PASSED")
        print("=" * 60)
        print(f"Successfully fine-tuned RoBERTa using MLM!")
    else:
        print("⚠️  MeZO ROBERTA MLM TEST: NEEDS MORE STEPS")
        print("=" * 60)
        print("Consider:")
        print("- Running for more steps (original uses 100K)")
        print("- Tuning hyperparameters")
        print("- Using roberta-large as in original")
    
    return success


if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: Running on CPU, this will be slow")
    
    success = test_mezo_roberta_mlm()