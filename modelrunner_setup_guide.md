# Complete ModelRunner Setup Guide for MeZO Training

## Overview
This guide documents the complete requirements for setting up ModelRunner in SGLang for MeZO training, based on investigation of the codebase and testing.

## Key Issues Identified

### 1. Distributed Initialization
- **Problem**: ModelRunner internally initializes distributed environment
- **Error**: "tensor model parallel group is already initialized"
- **Solution**: Don't pre-initialize; let ModelRunner handle it

### 2. Model Implementation
- **Problem**: OPT model missing required attributes
- **Error**: "'OPTConfig' object has no attribute 'is_generation'"
- **Solution**: Need to properly implement OPT model with all required attributes

### 3. LoRA Integration
- **Problem**: SGLang's LoRAManager expects full server infrastructure
- **Error**: Complex initialization requirements
- **Solution**: Create simplified LoRA wrapper for standalone use

## Complete Setup Requirements

### 1. Environment Setup
```python
# Required environment variables (for distributed)
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0" 
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"

# Optional but recommended
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Specify GPU
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid warnings
```

### 2. Model Implementation Requirements
For a model to work with ModelRunner, it needs:

```python
# In model config
class OPTConfig:
    # Required attributes
    is_generation = True  # For text generation models
    is_multimodal = False  # For vision-language models
    is_embedding = False  # For embedding models
    attention_arch = AttentionArch.MHA  # or MLA for newer models
    
    # For attention layers
    num_attention_heads = 12
    num_key_value_heads = 12  # Same as num_attention_heads for MHA
    hidden_size = 768
    
    # Optional but recommended
    is_hybrid = False
    is_multimodal_chunked_prefill_supported = False
```

### 3. Proper ModelRunner Initialization
```python
from sglang.srt.server_args import ServerArgs
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.model_executor.model_runner import ModelRunner

# Step 1: Create ServerArgs
server_args = ServerArgs(
    model_path="facebook/opt-125m",
    trust_remote_code=True,
    tp_size=1,  # Tensor parallel size
    pp_size=1,  # Pipeline parallel size
    mem_fraction_static=0.8,
    disable_cuda_graph=True,  # For training flexibility
    disable_radix_cache=False,  # Enable KV cache optimization
    dtype="float16",
    grammar_backend="none",  # Avoid xgrammar dependency
)

# Step 2: Create ModelConfig
model_config = ModelConfig(
    model_path,  # First positional argument
    model_override_args="{}",
    trust_remote_code=True,
)

# Step 3: Initialize ModelRunner
# This will internally handle all distributed initialization
model_runner = ModelRunner(
    model_config=model_config,
    mem_fraction_static=server_args.mem_fraction_static,
    gpu_id=0,  # GPU device ID
    tp_rank=0,  # Tensor parallel rank
    tp_size=server_args.tp_size,
    pp_rank=0,  # Pipeline parallel rank
    pp_size=server_args.pp_size,
    nccl_port=29500,  # NCCL communication port
    server_args=server_args,
)
```

### 4. LoRA Setup for MeZO
```python
from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.lora.lora_config import LoRAConfig

# Initialize LoRA manager
lora_manager = LoRAManager(
    model_runner=model_runner,
    max_num_batched_tokens=16384,
    vocab_size=model_config.vocab_size,
    lora_config=None,
    max_num_seqs=256,
    device="cuda",
)

# Create LoRA config (SGLang style)
lora_config = LoRAConfig(
    rank=8,
    alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    # Note: SGLang LoRAConfig is different from PEFT
)

# Add LoRA adapter
lora_manager.add_lora("mezo_adapter", lora_config)
```

## Working Examples

### Example 1: Using GPT-2 (Working Model)
```python
# GPT-2 works because it's properly implemented
config = TrainingConfig(model_name="gpt2")
# ... rest of setup
```

### Example 2: Direct Model Loading (Current Workaround)
```python
# Skip ModelRunner, use transformers directly
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig

model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
peft_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, peft_config)
```

## Required Fixes for OPT Support

### 1. Update OPT Model Implementation
```python
# In opt.py
class OPTForCausalLM(nn.Module):
    def __init__(self, config: OPTConfig, ...):
        # Add required config attributes
        config.is_generation = True
        config.is_multimodal = False
        config.is_embedding = False
        config.num_key_value_heads = config.num_attention_heads
        # ... rest of implementation
```

### 2. Fix Attention Implementation
```python
# Update OPTAttention to match expected interface
class OPTAttention(nn.Module):
    def __init__(self, layer_id, config, ...):
        # Ensure compatibility with RadixAttention
        self.num_kv_heads = config.num_attention_heads  # For MHA
        # ... rest of implementation
```

## Best Practices

### 1. Memory Management
- Use `mem_fraction_static=0.8` for training (leaves room for gradients)
- Disable CUDA graphs during training: `disable_cuda_graph=True`
- Enable memory saver if needed: `enable_memory_saver=True`

### 2. Performance Optimization
- Keep RadixCache enabled: `disable_radix_cache=False`
- Use appropriate batch sizes based on model size
- Monitor cache hit rates for MeZO's perturbation pairs

### 3. Error Handling
- Always check if distributed is already initialized
- Properly cleanup with `dist.destroy_process_group()` if needed
- Use try-except blocks around ModelRunner initialization

## Future Improvements

1. **Complete OPT Implementation**: Add all required attributes and methods
2. **Simplified LoRA API**: Create MeZO-specific LoRA manager
3. **Direct MeZO Support**: Add MeZO as first-class training method in SGLang
4. **Server Integration**: Implement MeZO training endpoints in server

## Conclusion
While ModelRunner provides advanced features like RadixAttention optimization, the current implementation requires models to follow specific interfaces. For immediate MeZO training needs, the direct model loading approach is more practical. Long-term, proper model implementation and simplified APIs would make ModelRunner integration seamless.