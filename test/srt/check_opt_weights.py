#!/usr/bin/env python3
"""Check OPT weight names from HuggingFace."""

from transformers import AutoModelForCausalLM
import torch

# Load model
print("Loading OPT-125m from HuggingFace...")
model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")

# Get all parameter names
param_names = list(model.state_dict().keys())

print(f"\nTotal parameters: {len(param_names)}")
print("\nFirst 20 parameter names:")
for i, name in enumerate(param_names[:20]):
    print(f"  {i}: {name}")

print("\nLast 10 parameter names:")
for i, name in enumerate(param_names[-10:]):
    print(f"  {i}: {name}")

# Check layer structure
print("\nLayer 0 parameters:")
layer0_params = [name for name in param_names if "layers.0." in name]
for name in layer0_params:
    print(f"  {name}")

# Check embed/final structure
print("\nEmbedding and final layer parameters:")
embed_params = [name for name in param_names if "embed" in name or "final" in name or "project" in name]
for name in embed_params:
    print(f"  {name}")