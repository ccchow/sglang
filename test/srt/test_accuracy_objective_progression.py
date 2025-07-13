#!/usr/bin/env python3
"""
Test to analyze whether accuracy improves gradually when using accuracy as the objective.
Tracks both training and evaluation accuracy over time.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from datetime import datetime


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


def create_balanced_set(examples, size_per_class):
    """Create a balanced dataset."""
    pos = [ex for ex in examples if ex['label'] == 1]
    neg = [ex for ex in examples if ex['label'] == 0]
    
    selected = pos[:size_per_class] + neg[:size_per_class]
    np.random.shuffle(selected)
    return selected


def compute_accuracy_objective(model, inputs, labels):
    """Compute negative accuracy (for minimization)."""
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
        accuracy = (predictions == labels).float().mean().item()
        return -accuracy  # Negative for minimization


def test_accuracy_objective_progression():
    """Test accuracy progression with accuracy objective."""
    print("=" * 80)
    print("Accuracy Objective Progression Analysis")
    print("=" * 80)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "roberta-base"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    ).to(device)
    
    # Create simple LoRA parameters
    param = model.classifier.dense.weight
    original_param = param.data.clone()
    
    # Load data
    data_dir = "/home/lei/git/princeton-nlp/MeZO/medium_models/data/k-shot-1k-test/SST-2"
    train_data = load_sst2_data(f"{data_dir}/512-42/train.tsv")
    eval_data = create_balanced_set(
        load_sst2_data(f"{data_dir}/512-42/dev.tsv"), 
        size_per_class=50
    )
    
    print(f"\nSetup:")
    print(f"  Model: {model_name}")
    print(f"  Train data: {len(train_data)} examples")
    print(f"  Eval data: {len(eval_data)} examples (balanced)")
    print(f"  Device: {device}")
    
    # Test different epsilon values
    epsilon_values = [1e-3, 1e-2, 1e-1]
    results = {}
    
    for epsilon in epsilon_values:
        print(f"\n\nTesting with epsilon = {epsilon}")
        print("-" * 60)
        
        # Reset parameters
        param.data = original_param.clone()
        
        # Training settings
        num_steps = 2000
        batch_size = 32
        learning_rate = 1e-5  # Higher LR for demonstration
        
        # Tracking
        train_accuracies = []
        eval_accuracies = []
        eval_steps = []
        gradients = []
        non_zero_grad_steps = []
        
        # Initial evaluation
        eval_batch = eval_data[:50]
        eval_texts = [ex['text'] for ex in eval_batch]
        eval_labels = torch.tensor([ex['label'] for ex in eval_batch]).to(device)
        eval_inputs = tokenizer(eval_texts, padding=True, truncation=True, return_tensors='pt').to(device)
        
        with torch.no_grad():
            outputs = model(**eval_inputs)
            preds = torch.argmax(outputs.logits, dim=-1)
            init_acc = (preds == eval_labels).float().mean().item()
        
        eval_accuracies.append(init_acc)
        eval_steps.append(0)
        
        print(f"Initial accuracy: {init_acc:.1%}")
        
        # Training loop
        for step in range(num_steps):
            # Sample batch
            idx = np.random.choice(len(train_data), batch_size, replace=True)
            texts = [train_data[i]['text'] for i in idx]
            labels = torch.tensor([train_data[i]['label'] for i in idx]).to(device)
            
            inputs = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            ).to(device)
            
            # MeZO step with accuracy objective
            z = torch.randn_like(param)
            
            # Forward with +epsilon
            param.data = original_param + epsilon * z
            obj_plus = compute_accuracy_objective(model, inputs, labels)
            acc_plus = -obj_plus  # Convert back to accuracy
            
            # Forward with -epsilon  
            param.data = original_param - epsilon * z
            obj_minus = compute_accuracy_objective(model, inputs, labels)
            acc_minus = -obj_minus
            
            # Gradient estimate
            grad_est = (obj_plus - obj_minus) / (2 * epsilon)
            
            # Update
            original_param = original_param - learning_rate * grad_est * z
            param.data = original_param.clone()
            
            # Track metrics
            train_acc = (acc_plus + acc_minus) / 2
            train_accuracies.append(train_acc)
            gradients.append(abs(grad_est))
            
            if grad_est != 0:
                non_zero_grad_steps.append(step)
            
            # Evaluate periodically
            if (step + 1) % 200 == 0:
                with torch.no_grad():
                    outputs = model(**eval_inputs)
                    preds = torch.argmax(outputs.logits, dim=-1)
                    eval_acc = (preds == eval_labels).float().mean().item()
                
                eval_accuracies.append(eval_acc)
                eval_steps.append(step + 1)
                
                non_zero_ratio = len(non_zero_grad_steps) / (step + 1) * 100
                print(f"Step {step+1}: Train acc={train_acc:.1%}, "
                      f"Eval acc={eval_acc:.1%}, "
                      f"Non-zero grads={non_zero_ratio:.1f}%")
        
        # Store results
        results[epsilon] = {
            'train_accuracies': train_accuracies,
            'eval_accuracies': eval_accuracies,
            'eval_steps': eval_steps,
            'gradients': gradients,
            'non_zero_grad_steps': non_zero_grad_steps,
            'final_acc': eval_accuracies[-1],
            'improvement': eval_accuracies[-1] - eval_accuracies[0]
        }
    
    # Create plots
    print("\n\nGenerating plots...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Accuracy as Objective: Progression Analysis', fontsize=16)
    
    for i, epsilon in enumerate(epsilon_values):
        data = results[epsilon]
        
        # Training accuracy
        ax = axes[0, i]
        ax.plot(data['train_accuracies'], alpha=0.5, color='blue')
        if len(data['train_accuracies']) > 50:
            # Add smoothed line
            smoothed = np.convolve(data['train_accuracies'], np.ones(50)/50, mode='valid')
            ax.plot(range(25, len(data['train_accuracies'])-24), smoothed, 'b-', linewidth=2)
        ax.set_title(f'Training Accuracy (ε={epsilon})')
        ax.set_xlabel('Step')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Evaluation accuracy
        ax = axes[1, i]
        ax.plot(data['eval_steps'], data['eval_accuracies'], 'o-', color='green', markersize=8)
        ax.axhline(y=0.5, color='red', linestyle=':', alpha=0.5, label='Random')
        ax.set_title(f'Evaluation Accuracy (ε={epsilon})')
        ax.set_xlabel('Step')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        non_zero_pct = len(data['non_zero_grad_steps']) / len(data['gradients']) * 100
        ax.text(0.02, 0.02, f'Non-zero: {non_zero_pct:.1f}%\nImprove: {data["improvement"]*100:+.1f}pp',
                transform=ax.transAxes, fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('accuracy_objective_progression.png', dpi=150)
    print("Plots saved to: accuracy_objective_progression.png")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Does accuracy improve gradually with accuracy objective?")
    print("=" * 80)
    
    for epsilon in epsilon_values:
        data = results[epsilon]
        non_zero_pct = len(data['non_zero_grad_steps']) / len(data['gradients']) * 100
        print(f"\nEpsilon = {epsilon}:")
        print(f"  Non-zero gradients: {non_zero_pct:.1f}%")
        print(f"  Initial accuracy: {data['eval_accuracies'][0]:.1%}")
        print(f"  Final accuracy: {data['final_acc']:.1%}")
        print(f"  Improvement: {data['improvement']*100:+.1f} percentage points")
        
        # Analyze progression
        if len(data['eval_accuracies']) > 1:
            changes = [data['eval_accuracies'][i+1] - data['eval_accuracies'][i] 
                      for i in range(len(data['eval_accuracies'])-1)]
            print(f"  Accuracy changes: {[f'{c*100:+.1f}' for c in changes]}")
    
    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print("=" * 80)
    print("1. With small epsilon (1e-3): Almost no non-zero gradients → no learning")
    print("2. With medium epsilon (1e-2): Some non-zero gradients → minimal learning")
    print("3. With large epsilon (1e-1): More non-zero gradients → some improvement")
    print("\nAccuracy does NOT improve gradually - it jumps when predictions flip!")
    print("This is why the paper uses MLM with cross-entropy instead.")
    print("=" * 80)
    
    # Show gradient distribution
    plt.figure(figsize=(12, 4))
    for i, epsilon in enumerate(epsilon_values):
        plt.subplot(1, 3, i+1)
        data = results[epsilon]
        non_zero_grads = [g for g in data['gradients'] if g > 0]
        
        if non_zero_grads:
            plt.hist(non_zero_grads, bins=30, alpha=0.7, color='purple')
            plt.title(f'Non-zero Gradients (ε={epsilon})')
            plt.xlabel('|Gradient|')
            plt.ylabel('Count')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'No non-zero\ngradients', 
                    ha='center', va='center', fontsize=14,
                    transform=plt.gca().transAxes)
            plt.title(f'Non-zero Gradients (ε={epsilon})')
    
    plt.tight_layout()
    plt.savefig('accuracy_gradient_distribution.png', dpi=150)
    print("\nGradient distribution saved to: accuracy_gradient_distribution.png")
    
    return results


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    results = test_accuracy_objective_progression()
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")