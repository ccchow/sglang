# ModelRunner Setup Investigation Results

## Key Findings

### 1. **Distributed Initialization Conflict**
The main issue is that ModelRunner internally calls `initialize_model_parallel()` in its `init_torch_distributed()` method. When we pre-initialize the distributed environment, it causes a conflict:
- Error: "tensor model parallel group is already initialized"
- ModelRunner expects to handle distributed initialization itself

### 2. **Required Setup Sequence**
Based on the investigation, the proper sequence is:
1. Set environment variables (RANK, WORLD_SIZE, etc.)
2. Let ModelRunner handle all distributed initialization
3. ModelRunner will internally:
   - Initialize PyTorch distributed
   - Initialize model parallel groups
   - Set up communication groups

### 3. **Configuration Issues Found**
- `ModelConfig` expects model_path as first positional argument
- `LoRAConfig` from SGLang doesn't take `model_path` parameter
- OPT model needs proper attribute mapping (missing `is_generation`, `num_key_value_heads`)

### 4. **Working Approach**
For MeZO training with ModelRunner, we need to:
1. **Don't pre-initialize distributed** - Let ModelRunner handle it
2. **Use supported models** - GPT-2 works, OPT needs implementation fixes
3. **Create custom LoRA wrapper** - SGLang's LoRAManager expects full server infrastructure

## Recommended Solutions

### Option 1: Direct Model Loading (Current Working Approach)
```python
# Load model directly with transformers + PEFT
model = AutoModelForCausalLM.from_pretrained(model_name)
model = get_peft_model(model, lora_config)
# Use MeZO algorithm directly
```

### Option 2: Fix OPT Implementation
1. Add missing attributes to OPT config mapping
2. Implement proper attention configuration
3. Register model correctly in SGLang registry

### Option 3: Use Engine API
Instead of ModelRunner, use the higher-level Engine API:
```python
from sglang import Engine
engine = Engine(model_path=model_name)
# Access model through engine.model_runner.model
```

### Option 4: Server-Based Approach
1. Launch SGLang server with the model
2. Use the server's MLM endpoints (once implemented)
3. Send MeZO training requests to server

## Technical Details

### ModelRunner Initialization Requirements
1. **Environment Variables** (if distributed):
   - RANK, LOCAL_RANK, WORLD_SIZE
   - MASTER_ADDR, MASTER_PORT
   
2. **Required Parameters**:
   - `model_config`: ModelConfig instance
   - `mem_fraction_static`: Memory allocation (0.0-1.0)
   - `gpu_id`: GPU device ID
   - `tp_rank`, `tp_size`: Tensor parallel settings
   - `pp_rank`, `pp_size`: Pipeline parallel settings
   - `nccl_port`: NCCL communication port
   - `server_args`: Full ServerArgs instance

3. **Model Config Attributes**:
   - `is_generation`: Whether model generates text
   - `is_multimodal`: Whether model supports images
   - `num_key_value_heads`: For attention configuration
   - `attention_arch`: Architecture type

### Why Direct Approach Works Better for MeZO
1. **Memory Efficiency**: MeZO only needs forward passes
2. **Simplicity**: No need for complex server infrastructure
3. **Flexibility**: Easy to modify parameters on the fly
4. **Compatibility**: Works with any PyTorch model

## Conclusion
While full ModelRunner integration would provide benefits like RadixAttention optimization, the current direct approach is more practical for MeZO training experiments. The main bottleneck is the OPT model implementation, which needs updates to work with SGLang's infrastructure.