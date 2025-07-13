# Plan for SGLang Server Support of RoBERTa MLM

## Overview
This document outlines a comprehensive plan to add proper RoBERTa MLM (Masked Language Modeling) support to SGLang's server infrastructure, enabling MeZO training with real RadixAttention benefits.

## Current State Analysis

### What We Have
1. **XLMRobertaForMaskedLM** implemented in `python/sglang/srt/models/roberta.py`
   - Model architecture with MLM head
   - Proper loss computation
   - Registered in EntryClass

2. **MeZO Infrastructure**
   - MeZOTrainer in `python/sglang/srt/mezo_trainer.py`
   - MeZORadixOptimizer in `python/sglang/srt/mezo_radix_optimizer.py`
   - Basic integration tests

### What's Missing
1. **Server MLM Support**
   - Server expects generative models (next-token prediction)
   - No MLM-specific endpoints
   - No mask token handling in server API

2. **ModelRunner MLM Integration**
   - ModelRunner designed for autoregressive generation
   - No MLM forward pass support
   - Missing mask position tracking

3. **RadixAttention for MLM**
   - Current RadixAttention optimized for prefix caching in generation
   - Need adaptation for MLM's bidirectional attention

## Implementation Plan

### Phase 1: Server API Extensions (Week 1)

#### 1.1 Add MLM Endpoint
```python
# In python/sglang/srt/entrypoints/http_server.py
@app.post("/v1/mlm")
async def mlm_inference(request: MLMRequest):
    """Handle masked language modeling requests."""
    # Process texts with mask tokens
    # Return logits at mask positions
    # Support batch processing
```

#### 1.2 Define MLM Request/Response Schema
```python
# In python/sglang/srt/managers/io_struct.py
@dataclass
class MLMRequest:
    texts: List[str]  # Texts with [MASK] tokens
    return_logits: bool = False
    return_top_k: int = 10
    request_ids: Optional[List[str]] = None  # For caching

@dataclass
class MLMResponse:
    predictions: List[Dict[str, Any]]  # Token predictions at mask positions
    logits: Optional[List[torch.Tensor]] = None
    cache_hit: bool = False
```

#### 1.3 Update Server Arguments
```python
# In python/sglang/srt/server_args.py
# Add --enable-mlm flag
# Add --mlm-models list for supported MLM models
```

### Phase 2: ModelRunner MLM Support (Week 2)

#### 2.1 Add MLM Forward Mode
```python
# In python/sglang/srt/model_executor/forward_batch_info.py
class ForwardMode(Enum):
    PREFILL = "prefill"
    DECODE = "decode"
    MLM = "mlm"  # New mode for masked language modeling
```

#### 2.2 Implement MLM Forward Pass
```python
# In python/sglang/srt/model_executor/model_runner.py
def forward_mlm(self, batch: MLMBatch):
    """Forward pass for masked language modeling."""
    # Track mask positions
    # Run bidirectional attention
    # Extract logits at mask positions only
    # Return MLMOutput with logits and mask info
```

#### 2.3 MLM Batch Management
```python
# In python/sglang/srt/managers/schedule_batch.py
class MLMBatch(ScheduleBatch):
    """Batch specifically for MLM requests."""
    mask_positions: List[List[int]]
    original_texts: List[str]
    request_ids: List[str]
```

### Phase 3: RadixAttention for MLM (Week 3)

#### 3.1 Bidirectional KV Cache
```python
# In python/sglang/srt/mem_cache/radix_cache.py
class MLMRadixCache(RadixCache):
    """RadixCache adapted for MLM's bidirectional attention."""
    # Cache full sequences (not just prefixes)
    # Support exact match for MeZO's +εz/-εz patterns
```

#### 3.2 MLM Cache Key Design
```python
# Cache key includes:
# - Full tokenized sequence
# - Model state hash (for MeZO perturbations)
# - Request ID prefix (for MeZO step tracking)
```

#### 3.3 Cache Metrics for MLM
```python
# Track:
# - Exact sequence matches
# - Perturbation pair matches (+εz/-εz)
# - Cache hit rate per MeZO step
```

### Phase 4: MeZO Server Integration (Week 4)

#### 4.1 Server-side MeZO Coordinator
```python
# In python/sglang/srt/managers/mezo_coordinator.py
class MeZOCoordinator:
    """Coordinates MeZO training on server side."""
    def register_mezo_session(self, session_id: str, config: MeZOConfig):
        # Track active MeZO sessions
        # Manage LoRA parameters server-side
        # Coordinate perturbations
    
    def mezo_forward_pair(self, session_id: str, batch: MLMBatch):
        # Execute +εz and -εz forwards
        # Maximize cache reuse
        # Return gradient estimates
```

#### 4.2 MeZO Training Endpoint
```python
# In python/sglang/srt/entrypoints/http_server.py
@app.post("/v1/mezo/train_step")
async def mezo_train_step(request: MeZOStepRequest):
    """Execute one MeZO training step."""
    # Receive batch and perturbation info
    # Run forward pair with cache optimization
    # Update LoRA parameters
    # Return loss and gradient info
```

#### 4.3 Stateful MeZO Sessions
```python
# Support:
# - Session creation/deletion
# - LoRA parameter persistence
# - Training state tracking
# - Checkpoint management
```

### Phase 5: Testing and Optimization (Week 5)

#### 5.1 Comprehensive Test Suite
```python
# Tests to add:
# - test_mlm_endpoint.py
# - test_mlm_modelrunner.py
# - test_mlm_radix_cache.py
# - test_mezo_server_integration.py
# - test_roberta_mlm_e2e.py
```

#### 5.2 Performance Benchmarks
```python
# Benchmark:
# - MLM inference throughput
# - RadixCache hit rates for MeZO
# - Memory usage with LoRA
# - Scaling with batch size
```

#### 5.3 Documentation
- Add MLM tutorial notebook
- Document MeZO server API
- Provide RoBERTa SST-2 example
- Cache optimization guide

## Implementation Priority

### Immediate (This Week)
1. Basic MLM endpoint (without optimization)
2. Simple MLM forward in ModelRunner
3. Test with RoBERTa model

### Short Term (2-3 Weeks)
1. Full MLM batch management
2. Basic RadixCache adaptation
3. MeZO coordinator prototype

### Medium Term (1 Month)
1. Optimized bidirectional caching
2. Full MeZO server integration
3. Production-ready API

## Technical Challenges

### 1. Attention Pattern Differences
- **Challenge**: MLM uses bidirectional attention, not causal
- **Solution**: Add attention_type flag to ModelRunner

### 2. Cache Key Design
- **Challenge**: MeZO needs exact sequence matching
- **Solution**: Use full sequence hash + perturbation ID

### 3. Memory Management
- **Challenge**: Storing full sequences vs prefixes
- **Solution**: Adaptive cache eviction for MLM

### 4. API Compatibility
- **Challenge**: Maintaining backward compatibility
- **Solution**: Separate MLM endpoints, gradual migration

## Success Metrics

1. **Functionality**
   - RoBERTa MLM runs on SGLang server
   - MeZO training achieves paper results
   - RadixCache provides >90% hit rate for MeZO

2. **Performance**
   - MLM inference within 20% of HuggingFace
   - MeZO training 2-3x faster with caching
   - Memory usage comparable to generation

3. **Usability**
   - Clean API for MLM tasks
   - Easy MeZO session management
   - Good documentation and examples

## Next Steps

1. **Prototype Basic MLM Endpoint** (Today)
   - Minimal viable MLM API
   - Test with existing RoBERTa model
   - Measure baseline performance

2. **Design Review** (This Week)
   - Review plan with SGLang team
   - Get feedback on API design
   - Prioritize features

3. **Start Implementation** (Next Week)
   - Begin with Phase 1
   - Create tracking issues
   - Set up CI tests

## Conclusion

This plan provides a systematic approach to adding proper MLM support to SGLang, enabling efficient MeZO training with real RadixAttention benefits. The phased approach allows for incremental progress while maintaining system stability.