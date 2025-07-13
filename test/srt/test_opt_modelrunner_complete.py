#!/usr/bin/env python3
"""Test OPT model with complete implementation using ModelRunner."""

import os
import sys
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import required modules
from python.sglang.srt.model_executor.model_runner import ModelRunner
from python.sglang.srt.server_args import ServerArgs
from python.sglang.srt.configs.model_config import ModelConfig
from python.sglang.srt.mezo_trainer import MeZOTrainer
from transformers import AutoTokenizer, OPTConfig
from datasets import load_dataset

def setup_environment():
    """Set up environment for ModelRunner."""
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

def test_opt_config():
    """Test that OPT config has all required attributes."""
    print("\n=== Testing OPT Config Attributes ===")
    
    # Import the complete OPT implementation here to avoid circular imports
    from python.sglang.srt.models.opt_complete import add_missing_config_attributes
    
    # Load config
    config = OPTConfig.from_pretrained("facebook/opt-125m")
    
    # Add missing attributes
    config = add_missing_config_attributes(config)
    
    # Check required attributes
    required_attrs = [
        'is_generation', 'is_multimodal', 'is_embedding', 
        'num_key_value_heads', 'attention_arch', 'is_hybrid',
        'is_multimodal_chunked_prefill_supported'
    ]
    
    for attr in required_attrs:
        if hasattr(config, attr):
            print(f"✓ {attr}: {getattr(config, attr)}")
        else:
            print(f"✗ Missing: {attr}")
            
    return config

def test_modelrunner_initialization():
    """Test ModelRunner initialization with complete OPT model."""
    print("\n=== Testing ModelRunner Initialization ===")
    
    setup_environment()
    
    # Register the complete OPT model
    from python.sglang.srt.models.opt_complete import OPTForCausalLM
    from python.sglang.srt.models.registry import _MODELS
    _MODELS["OPTForCausalLM"] = OPTForCausalLM
    print("✓ Registered OPTForCausalLM model")
    
    try:
        # Create server args
        server_args = ServerArgs(
            model_path="facebook/opt-125m",
            trust_remote_code=True,
            tp_size=1,
            mem_fraction_static=0.8,
            disable_cuda_graph=True,
            disable_radix_cache=False,
            dtype="float16",
            grammar_backend="none",
        )
        
        # Create model config
        model_config = ModelConfig.from_server_args(server_args)
        print(f"Model config created: {model_config.hf_config.model_type}")
        
        # Initialize ModelRunner
        model_runner = ModelRunner(
            model_config=model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=0,
            tp_rank=0,
            tp_size=server_args.tp_size,
            nccl_port=29500,
            server_args=server_args,
        )
        
        print("✓ ModelRunner initialized successfully!")
        
        # Test model access
        model = model_runner.model
        print(f"✓ Model type: {type(model)}")
        print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model_runner
        
    except Exception as e:
        print(f"✗ ModelRunner initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_mezo_training():
    """Test MeZO training with ModelRunner."""
    print("\n=== Testing MeZO Training ===")
    
    model_runner = test_modelrunner_initialization()
    if model_runner is None:
        print("Skipping MeZO training test due to initialization failure")
        return
        
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Load small dataset
        dataset = load_dataset("imdb", split="train[:100]")
        
        def tokenize_function(examples):
            return tokenizer(
                examples["text"], 
                padding="max_length", 
                truncation=True, 
                max_length=128
            )
            
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask"])
        
        # Create dataloader
        dataloader = DataLoader(tokenized_dataset, batch_size=4, shuffle=True)
        
        # Initialize MeZO trainer
        trainer = MeZOTrainer(
            model=model_runner.model,
            model_runner=model_runner,
            tokenizer=tokenizer,
            train_dataloader=dataloader,
            eval_dataloader=dataloader,
            epsilon=1e-3,
            learning_rate=1e-5,
            num_epochs=1,
            lora_rank=8,
            lora_alpha=16,
            lora_target_modules=["q_proj", "v_proj"],
            output_dir="./test_mezo_opt_output",
        )
        
        print("✓ MeZO trainer initialized")
        
        # Train for a few steps
        print("\nTraining for 5 steps...")
        for i, batch in enumerate(dataloader):
            if i >= 5:
                break
                
            loss = trainer.train_step(batch)
            print(f"Step {i+1}: Loss = {loss:.4f}")
            
        print("✓ MeZO training completed successfully!")
        
    except Exception as e:
        print(f"✗ MeZO training failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if dist.is_initialized():
            dist.destroy_process_group()

def main():
    """Run all tests."""
    print("Testing Complete OPT Implementation with ModelRunner")
    print("=" * 50)
    
    # Test 1: Config attributes
    config = test_opt_config()
    
    # Test 2: ModelRunner initialization
    model_runner = test_modelrunner_initialization()
    
    # Test 3: MeZO training (if ModelRunner works)
    if model_runner:
        test_mezo_training()
    
    print("\n" + "=" * 50)
    print("Testing completed!")

if __name__ == "__main__":
    main()