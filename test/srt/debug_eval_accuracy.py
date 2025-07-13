#!/usr/bin/env python3
"""Debug why evaluation accuracy is 100%."""

import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

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

def evaluate(model, tokenizer, examples):
    """Evaluate model on examples."""
    model.eval()
    correct = 0
    predictions = []
    labels = []
    
    for i, ex in enumerate(examples[:10]):  # First 10 examples
        inputs = tokenizer(
            ex['text'],
            truncation=True,
            max_length=128,
            return_tensors='pt'
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=-1).item()
            correct += (pred == ex['label'])
            predictions.append(pred)
            labels.append(ex['label'])
            
            if i < 5:  # Print first 5
                print(f"\nExample {i+1}:")
                print(f"  Text: {ex['text'][:50]}...")
                print(f"  Label: {ex['label']}")
                print(f"  Prediction: {pred}")
                print(f"  Logits: {logits.squeeze().tolist()}")
    
    accuracy = correct / len(examples[:10])
    print(f"\nAccuracy on first 10: {accuracy:.1%}")
    print(f"Predictions: {predictions}")
    print(f"Labels: {labels}")
    
    return accuracy

# Test
print("Loading model...")
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

print("\nLoading evaluation data...")
data_dir = "/home/lei/git/princeton-nlp/MeZO/medium_models/data/k-shot-1k-test/SST-2"
eval_data = load_sst2_data(f"{data_dir}/512-42/dev.tsv", max_examples=20)

print(f"\nEvaluating on {len(eval_data)} examples...")
evaluate(model, tokenizer, eval_data)

# Check label distribution
label_counts = {0: 0, 1: 0}
for ex in eval_data:
    label_counts[ex['label']] += 1
print(f"\nLabel distribution in first 20: {label_counts}")