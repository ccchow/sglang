#!/usr/bin/env python3
"""
Minimal MeZO MLM test to demonstrate the approach works.
"""

import torch
import numpy as np
import time
from transformers import RobertaTokenizer, RobertaForMaskedLM


def run_minimal_test():
    print("Minimal MeZO MLM Test (2000 steps)")
    print("=" * 50)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForMaskedLM.from_pretrained('roberta-base').to(device)
    
    # Get label word IDs (with space prefix for RoBERTa)
    terrible_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(' terrible')[0])
    great_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(' great')[0])
    label_ids = torch.tensor([terrible_id, great_id], device=device)
    print(f"Label word IDs: terrible={terrible_id}, great={great_id}")
    
    # Create simple LoRA parameters (just first 2 attention layers)
    lora_params = []
    for name, module in model.named_modules():
        if 'encoder.layer.0.attention.self.query' in name and hasattr(module, 'weight'):
            # Simple LoRA for query projection
            lora_A = torch.nn.Parameter(torch.randn(8, 768, device=device) * 0.01)
            lora_B = torch.nn.Parameter(torch.zeros(768, 8, device=device))
            lora_params.extend([lora_A, lora_B])
            
            # Store original weight
            module.original_weight = module.weight.data.clone()
            module.lora_A = lora_A
            module.lora_B = lora_B
            break
    
    print(f"LoRA parameters: {sum(p.numel() for p in lora_params):,}")
    
    # Training settings
    num_steps = 2000
    batch_size = 32
    lr = 1e-6
    epsilon = 1e-3
    
    # Load some training data
    train_texts = [
        "This movie is absolutely fantastic",
        "Terrible film, complete waste of time",
        "I loved every minute of it", 
        "Boring and poorly made",
        "Best movie I've seen all year",
        "Could not watch till the end",
        "Highly recommend to everyone",
        "Do not waste your money",
    ]
    train_labels = [1, 0, 1, 0, 1, 0, 1, 0]  # 1=great, 0=terrible
    
    print(f"\nTraining with {num_steps} steps...")
    print("-" * 50)
    
    losses = []
    gradients = []
    start_time = time.time()
    
    # Simple apply LoRA function
    def apply_lora():
        for name, module in model.named_modules():
            if 'encoder.layer.0.attention.self.query' in name and hasattr(module, 'lora_A'):
                module.weight.data = module.original_weight + (module.lora_B @ module.lora_A)
    
    for step in range(num_steps):
        # Sample batch
        idx = np.random.choice(len(train_texts), batch_size, replace=True)
        texts = []
        labels = []
        
        for i in idx:
            # Add MLM template
            text = train_texts[i] + " It was " + tokenizer.mask_token + "."
            texts.append(text)
            labels.append(train_labels[i])
        
        # Tokenize
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors='pt').to(device)
        labels_tensor = torch.tensor(labels, device=device)
        
        # MeZO step
        z_list = [torch.randn_like(p) for p in lora_params]
        
        # Forward with +epsilon
        for i, p in enumerate(lora_params):
            p.data.add_(epsilon * z_list[i])
        apply_lora()
        
        with torch.no_grad():
            outputs = model(**inputs)
            
            # Find mask positions
            mask_positions = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)
            if len(mask_positions[0]) == 0:
                continue
                
            # Get logits at mask positions
            mask_logits = outputs.logits[mask_positions[0], mask_positions[1]]
            label_logits = mask_logits[:, label_ids]
            
            loss_plus = torch.nn.functional.cross_entropy(label_logits, labels_tensor).item()
        
        # Forward with -epsilon
        for i, p in enumerate(lora_params):
            p.data.add_(-2 * epsilon * z_list[i])
        apply_lora()
        
        with torch.no_grad():
            outputs = model(**inputs)
            mask_logits = outputs.logits[mask_positions[0], mask_positions[1]]
            label_logits = mask_logits[:, label_ids]
            
            loss_minus = torch.nn.functional.cross_entropy(label_logits, labels_tensor).item()
        
        # Restore and update
        grad_est = (loss_plus - loss_minus) / (2 * epsilon)
        
        for i, p in enumerate(lora_params):
            p.data.add_(epsilon * z_list[i])  # Restore
            p.data.add_(-lr * grad_est * z_list[i])  # Update
        
        avg_loss = (loss_plus + loss_minus) / 2
        losses.append(avg_loss)
        gradients.append(abs(grad_est))
        
        # Print progress
        if (step + 1) % 400 == 0:
            recent_loss = np.mean(losses[-100:])
            recent_grad = np.mean(gradients[-100:])
            elapsed = time.time() - start_time
            print(f"Step {step+1:4d}: loss={recent_loss:.4f}, |grad|={recent_grad:.5f}, time={elapsed:.1f}s")
    
    print("-" * 50)
    
    # Final evaluation
    apply_lora()
    print("\nEvaluating on training examples...")
    
    correct = 0
    for i, (text, label) in enumerate(zip(train_texts, train_labels)):
        mlm_text = text + " It was " + tokenizer.mask_token + "."
        inputs = tokenizer(mlm_text, return_tensors='pt').to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            mask_pos = (inputs.input_ids == tokenizer.mask_token_id).nonzero()
            if len(mask_pos) > 0:
                mask_logits = outputs.logits[0, mask_pos[0, 1]]
                label_logits = mask_logits[label_ids]
                pred = torch.argmax(label_logits).item()
                correct += (pred == label)
                
                if i < 4:  # Show first few predictions
                    pred_word = 'great' if pred == 1 else 'terrible'
                    true_word = 'great' if label == 1 else 'terrible'
                    print(f"  '{text[:30]}...' -> pred: {pred_word}, true: {true_word}")
    
    accuracy = correct / len(train_texts)
    
    # Results
    print(f"\nResults:")
    print(f"  Training accuracy: {accuracy:.1%}")
    print(f"  Initial loss: {np.mean(losses[:100]):.4f}")
    print(f"  Final loss: {np.mean(losses[-100:]):.4f}")
    print(f"  Loss reduction: {np.mean(losses[:100]) - np.mean(losses[-100:]):.4f}")
    print(f"  Average gradient: {np.mean(gradients):.5f}")
    print(f"  Zero gradients: {sum(1 for g in gradients if g == 0)}/{len(gradients)} ({sum(1 for g in gradients if g == 0)/len(gradients)*100:.1f}%)")
    print(f"  Training time: {time.time() - start_time:.1f}s")
    
    print("\n" + "=" * 50)
    if np.mean(losses[-100:]) < np.mean(losses[:100]) - 0.01:
        print("✅ SUCCESS: MLM loss is decreasing!")
    else:
        print("⚠️  No clear improvement")
    print("=" * 50)
    
    return {
        'accuracy': accuracy,
        'initial_loss': np.mean(losses[:100]),
        'final_loss': np.mean(losses[-100:]),
        'avg_gradient': np.mean(gradients)
    }


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    results = run_minimal_test()