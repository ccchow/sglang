#!/usr/bin/env python3
"""
Fast test to show accuracy progression with accuracy objective.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import RobertaTokenizer, RobertaForSequenceClassification


def test_accuracy_progression():
    """Quick test of accuracy progression."""
    print("Accuracy Objective Progression Test")
    print("=" * 50)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2).to(device)
    
    # Simple parameter
    param = model.classifier.dense.weight
    original_param = param.data.clone()
    
    # Create simple data
    texts = [
        "This movie is absolutely fantastic!",
        "Terrible film, waste of time.",
        "Great acting and amazing story!",
        "Boring and poorly made.",
        "I loved every minute of it!",
        "Could not finish watching it.",
        "Highly recommend to everyone!",
        "Stay away from this movie.",
    ] * 4  # Repeat to get 32 examples
    labels = [1, 0, 1, 0, 1, 0, 1, 0] * 4
    
    # Test with different epsilon values
    epsilon_values = [1e-3, 1e-2, 5e-2]
    num_steps = 1000
    learning_rate = 1e-5
    
    results = {}
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Accuracy as Objective: Does It Improve Gradually?', fontsize=16)
    
    for idx, epsilon in enumerate(epsilon_values):
        print(f"\n\nTesting epsilon = {epsilon}")
        print("-" * 40)
        
        # Reset
        param.data = original_param.clone()
        current_param = original_param.clone()
        
        # Track metrics
        train_accs = []
        eval_accs = []
        eval_steps = []
        non_zero_count = 0
        
        # Initial eval
        inputs = tokenizer(texts[:8], padding=True, truncation=True, return_tensors='pt').to(device)
        labels_tensor = torch.tensor(labels[:8]).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1)
            init_acc = (preds == labels_tensor).float().mean().item()
        
        eval_accs.append(init_acc)
        eval_steps.append(0)
        
        print(f"Initial accuracy: {init_acc:.1%}")
        
        # Training
        batch_texts = texts
        batch_labels = torch.tensor(labels).to(device)
        batch_inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt').to(device)
        
        for step in range(num_steps):
            # MeZO step
            z = torch.randn_like(param)
            
            # Forward +eps
            param.data = current_param + epsilon * z
            with torch.no_grad():
                preds = torch.argmax(model(**batch_inputs).logits, dim=-1)
                acc_plus = (preds == batch_labels).float().mean().item()
            
            # Forward -eps
            param.data = current_param - epsilon * z
            with torch.no_grad():
                preds = torch.argmax(model(**batch_inputs).logits, dim=-1)
                acc_minus = (preds == batch_labels).float().mean().item()
            
            # Gradient
            grad = ((-acc_plus) - (-acc_minus)) / (2 * epsilon)
            
            if grad != 0:
                non_zero_count += 1
            
            # Update
            current_param = current_param - learning_rate * grad * z
            param.data = current_param.clone()
            
            # Track
            train_acc = (acc_plus + acc_minus) / 2
            train_accs.append(train_acc)
            
            # Eval every 100 steps
            if (step + 1) % 100 == 0:
                with torch.no_grad():
                    outputs = model(**inputs)
                    preds = torch.argmax(outputs.logits, dim=-1)
                    eval_acc = (preds == labels_tensor).float().mean().item()
                
                eval_accs.append(eval_acc)
                eval_steps.append(step + 1)
                
                print(f"Step {step+1}: Train={train_acc:.1%}, Eval={eval_acc:.1%}, "
                      f"Non-zero={non_zero_count}/{step+1} ({non_zero_count/(step+1)*100:.1f}%)")
        
        # Store results
        results[epsilon] = {
            'train_accs': train_accs,
            'eval_accs': eval_accs,
            'eval_steps': eval_steps,
            'non_zero_ratio': non_zero_count / num_steps
        }
        
        # Plot training accuracy
        ax = axes[0, idx]
        ax.plot(train_accs, alpha=0.3, color='blue')
        # Smooth
        if len(train_accs) > 20:
            smoothed = np.convolve(train_accs, np.ones(20)/20, mode='valid')
            ax.plot(range(10, len(train_accs)-9), smoothed, 'b-', linewidth=2)
        ax.set_title(f'Training Accuracy (ε={epsilon})')
        ax.set_xlabel('Step')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0.5, color='red', linestyle=':', alpha=0.5)
        
        # Plot eval accuracy
        ax = axes[1, idx]
        ax.plot(eval_steps, eval_accs, 'o-', color='green', markersize=8, linewidth=2)
        ax.set_title(f'Evaluation Accuracy (ε={epsilon})')
        ax.set_xlabel('Step')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0.5, color='red', linestyle=':', alpha=0.5)
        
        # Add text
        improvement = eval_accs[-1] - eval_accs[0]
        ax.text(0.05, 0.95, f'Non-zero: {non_zero_count/num_steps*100:.1f}%\n'
                           f'Change: {improvement*100:+.1f}pp',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('accuracy_progression.png', dpi=150)
    print("\nPlot saved to: accuracy_progression.png")
    
    # Summary
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    
    for eps in epsilon_values:
        r = results[eps]
        print(f"\nε = {eps}:")
        print(f"  Non-zero gradients: {r['non_zero_ratio']*100:.1f}%")
        print(f"  Accuracy change: {(r['eval_accs'][-1] - r['eval_accs'][0])*100:+.1f}pp")
        print(f"  Final accuracy: {r['eval_accs'][-1]:.1%}")
    
    # Plot gradient occurrences
    plt.figure(figsize=(10, 4))
    
    # Show when non-zero gradients occur
    for i, eps in enumerate(epsilon_values):
        plt.subplot(1, 3, i+1)
        
        # Recompute to track when gradients are non-zero
        param.data = original_param.clone()
        current_param = original_param.clone()
        gradient_indicators = []
        
        for step in range(min(500, num_steps)):
            z = torch.randn_like(param)
            
            param.data = current_param + eps * z
            acc_plus = -(torch.argmax(model(**batch_inputs).logits, dim=-1) == batch_labels).float().mean().item()
            
            param.data = current_param - eps * z
            acc_minus = -(torch.argmax(model(**batch_inputs).logits, dim=-1) == batch_labels).float().mean().item()
            
            grad = (acc_plus - acc_minus) / (2 * eps)
            gradient_indicators.append(1 if grad != 0 else 0)
            
            if grad != 0:
                current_param = current_param - learning_rate * grad * z
            param.data = current_param.clone()
        
        # Plot as scatter
        non_zero_steps = [i for i, g in enumerate(gradient_indicators) if g == 1]
        if non_zero_steps:
            plt.scatter(non_zero_steps, [1]*len(non_zero_steps), alpha=0.5, s=10)
        
        plt.title(f'Non-zero Gradient Steps (ε={eps})')
        plt.xlabel('Step')
        plt.ylabel('Gradient Present')
        plt.ylim(-0.1, 1.1)
        plt.yticks([0, 1], ['Zero', 'Non-zero'])
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gradient_occurrence.png', dpi=150)
    print("\nGradient occurrence plot saved to: gradient_occurrence.png")
    
    print("\n" + "=" * 50)
    print("CONCLUSION:")
    print("=" * 50)
    print("Accuracy does NOT improve gradually with accuracy objective!")
    print("- Small ε: Almost no gradients → no learning")
    print("- Large ε: Occasional gradients → sporadic jumps")
    print("- No smooth progression like with continuous losses")
    print("This is why the paper uses MLM with cross-entropy!")
    print("=" * 50)


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    test_accuracy_progression()