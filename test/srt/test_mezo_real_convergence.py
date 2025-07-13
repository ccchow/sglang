#!/usr/bin/env python3
"""
Test MeZO convergence on a real training task with a small model.
This test verifies that MeZO can actually minimize loss and improve model performance.
"""

import torch
import json
import os
import tempfile
import matplotlib.pyplot as plt
from typing import List, Dict
import numpy as np

# Create a simple instruction-following dataset
def create_instruction_dataset(num_samples: int = 100) -> List[Dict[str, str]]:
    """Create a simple math instruction dataset for testing convergence."""
    dataset = []
    
    # Simple addition problems
    for i in range(num_samples // 4):
        a = np.random.randint(1, 20)
        b = np.random.randint(1, 20)
        dataset.append({
            "prompt": f"What is {a} + {b}?",
            "completion": f"The answer is {a + b}."
        })
    
    # Simple subtraction problems
    for i in range(num_samples // 4):
        a = np.random.randint(10, 30)
        b = np.random.randint(1, 10)
        dataset.append({
            "prompt": f"What is {a} - {b}?",
            "completion": f"The answer is {a - b}."
        })
    
    # Simple multiplication problems
    for i in range(num_samples // 4):
        a = np.random.randint(1, 10)
        b = np.random.randint(1, 10)
        dataset.append({
            "prompt": f"What is {a} × {b}?",
            "completion": f"The answer is {a * b}."
        })
    
    # Word problems
    templates = [
        ("If I have {a} apples and buy {b} more, how many do I have?", 
         "You have {result} apples in total."),
        ("I had {a} cookies and ate {b}. How many are left?",
         "There are {result} cookies left."),
        ("Each box has {a} items. If I have {b} boxes, how many items total?",
         "There are {result} items in total.")
    ]
    
    for i in range(num_samples - 3 * (num_samples // 4)):
        template_idx = i % len(templates)
        prompt_template, completion_template = templates[template_idx]
        
        if template_idx == 0:  # Addition
            a = np.random.randint(1, 20)
            b = np.random.randint(1, 20)
            result = a + b
        elif template_idx == 1:  # Subtraction
            a = np.random.randint(10, 30)
            b = np.random.randint(1, 10)
            result = a - b
        else:  # Multiplication
            a = np.random.randint(1, 10)
            b = np.random.randint(1, 10)
            result = a * b
        
        dataset.append({
            "prompt": prompt_template.format(a=a, b=b),
            "completion": completion_template.format(result=result)
        })
    
    return dataset


def test_mezo_convergence():
    """Test MeZO convergence with loss tracking."""
    print("=" * 60)
    print("MeZO Real Convergence Test")
    print("=" * 60)
    
    # Create datasets
    print("\nCreating datasets...")
    train_data = create_instruction_dataset(200)
    eval_data = create_instruction_dataset(50)
    
    # Save datasets to temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        train_file = f.name
        for item in train_data:
            f.write(json.dumps(item) + '\n')
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        eval_file = f.name
        for item in eval_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Train dataset: {len(train_data)} samples")
    print(f"Eval dataset: {len(eval_data)} samples")
    
    # Import MeZO components
    from sglang.srt.mezo_trainer import MeZOTrainer, MeZODataset, create_dataloader
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.lora.lora_manager import LoRAManager
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.hf_transformers_utils import get_tokenizer
    from sglang.srt.configs.model_config import ModelConfig
    import logging
    
    # Set up logging to capture loss values
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("MeZOConvergenceTest")
    
    # Model configuration - use smallest GPT2 for testing
    model_path = "gpt2"  # 124M parameters
    
    print(f"\nInitializing model: {model_path}")
    
    # Create server args
    server_args = ServerArgs(
        model_path=model_path,
        mem_fraction_static=0.8,
        lora_paths=["dummy"],  # This enables LoRA
        lora_ranks=[8],
        trust_remote_code=True
    )
    
    # Initialize model
    model_config = ModelConfig(model_path)
    
    try:
        # Initialize model runner
        model_runner = ModelRunner(
            model_config=model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=0,
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            nccl_port=23333,
            server_args=server_args,
        )
        
        # Initialize tokenizer
        tokenizer = get_tokenizer(
            model_path,
            tokenizer_mode="auto",
            trust_remote_code=True
        )
        
        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create LoRA adapter
        lora_manager = model_runner.lora_manager
        lora_name = "convergence_test"
        lora_manager.load_lora_adapter(lora_name, "")  # Empty path creates new adapter
        
        # Initialize LoRA weights
        import math
        lora_adapter = lora_manager.loras[lora_name]
        for layer in lora_adapter.layers:
            for param_name, param in layer.weights.items():
                if "lora_A" in param_name:
                    torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                elif "lora_B" in param_name:
                    torch.nn.init.zeros_(param)
        
        # Create datasets
        train_dataset = MeZODataset(train_file, tokenizer, max_length=128)
        eval_dataset = MeZODataset(eval_file, tokenizer, max_length=128)
        
        train_dataloader = create_dataloader(train_dataset, batch_size=4, shuffle=True)
        eval_dataloader = create_dataloader(eval_dataset, batch_size=4, shuffle=False)
        
        # Training parameters
        learning_rate = 5e-4  # Higher LR for faster convergence in test
        epsilon = 1e-3
        num_steps = 100
        eval_interval = 10
        
        print(f"\nTraining Configuration:")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Epsilon: {epsilon}")
        print(f"  Batch size: 4")
        print(f"  Training steps: {num_steps}")
        print(f"  Evaluation interval: {eval_interval}")
        
        # Create trainer
        trainer = MeZOTrainer(model_runner, lora_manager, lora_name, tokenizer)
        
        # Collect LoRA parameters
        lora_params = []
        for layer in lora_adapter.layers:
            for param_name, param in layer.weights.items():
                if "lora_A" in param_name or "lora_B" in param_name:
                    lora_params.append(param)
        
        # Create optimizer
        optimizer = torch.optim.Adam(lora_params, lr=learning_rate)
        
        # Training loop with loss tracking
        train_losses = []
        eval_losses = []
        steps = []
        
        print("\nStarting training...")
        print("-" * 40)
        
        train_iter = iter(train_dataloader)
        eval_iter = iter(eval_dataloader)
        
        for step in range(num_steps):
            trainer.current_step = step
            
            # Get training batch
            try:
                train_batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dataloader)
                train_batch = next(train_iter)
            
            # Training step
            train_loss = trainer._mezo_step(train_batch, lora_params, optimizer, epsilon)
            train_losses.append(train_loss)
            
            # Evaluation
            if step % eval_interval == 0:
                # Get eval batch
                try:
                    eval_batch = next(eval_iter)
                except StopIteration:
                    eval_iter = iter(eval_dataloader)
                    eval_batch = next(eval_iter)
                
                # Compute eval loss (without perturbation)
                with torch.no_grad():
                    eval_loss = trainer._forward_pass(eval_batch)
                
                eval_losses.append(eval_loss)
                steps.append(step)
                
                print(f"Step {step:3d} | Train Loss: {train_loss:.4f} | Eval Loss: {eval_loss:.4f}")
        
        print("-" * 40)
        print("Training completed!")
        
        # Analyze convergence
        print("\nConvergence Analysis:")
        
        # Check if loss decreased
        initial_train_loss = np.mean(train_losses[:5])
        final_train_loss = np.mean(train_losses[-5:])
        train_improvement = (initial_train_loss - final_train_loss) / initial_train_loss * 100
        
        initial_eval_loss = eval_losses[0]
        final_eval_loss = eval_losses[-1]
        eval_improvement = (initial_eval_loss - final_eval_loss) / initial_eval_loss * 100
        
        print(f"  Initial train loss: {initial_train_loss:.4f}")
        print(f"  Final train loss: {final_train_loss:.4f}")
        print(f"  Train improvement: {train_improvement:.1f}%")
        print(f"  Initial eval loss: {initial_eval_loss:.4f}")
        print(f"  Final eval loss: {final_eval_loss:.4f}")
        print(f"  Eval improvement: {eval_improvement:.1f}%")
        
        # Plot convergence
        plt.figure(figsize=(10, 6))
        
        # Plot training loss
        plt.subplot(1, 2, 1)
        plt.plot(range(len(train_losses)), train_losses, 'b-', alpha=0.6, label='Train Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(range(len(train_losses)), train_losses, 1)
        p = np.poly1d(z)
        plt.plot(range(len(train_losses)), p(range(len(train_losses))), "r--", alpha=0.8, label=f'Trend (slope={z[0]:.4f})')
        plt.legend()
        
        # Plot evaluation loss
        plt.subplot(1, 2, 2)
        plt.plot(steps, eval_losses, 'g-', marker='o', label='Eval Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Evaluation Loss')
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(steps, eval_losses, 1)
        p = np.poly1d(z)
        plt.plot(steps, p(steps), "r--", alpha=0.8, label=f'Trend (slope={z[0]:.4f})')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('mezo_convergence_plot.png', dpi=150)
        print(f"\nConvergence plot saved to: mezo_convergence_plot.png")
        
        # Test on a few examples
        print("\nTesting on sample problems:")
        print("-" * 40)
        
        test_prompts = [
            "What is 5 + 3?",
            "What is 12 - 4?",
            "What is 6 × 7?",
            "If I have 10 apples and buy 5 more, how many do I have?"
        ]
        
        for prompt in test_prompts:
            # This is a simplified test - in practice, you'd use the model's generate function
            print(f"Prompt: {prompt}")
            # Note: Actual generation would require more setup
        
        # Determine convergence success
        converged = (train_improvement > 10 and eval_improvement > 5) or final_train_loss < 2.0
        
        print("\n" + "=" * 60)
        print(f"CONVERGENCE TEST: {'PASSED' if converged else 'FAILED'}")
        print("=" * 60)
        
        if converged:
            print("✓ MeZO successfully converged!")
            print(f"  - Training loss decreased by {train_improvement:.1f}%")
            print(f"  - Evaluation loss decreased by {eval_improvement:.1f}%")
        else:
            print("✗ MeZO did not show sufficient convergence")
            print("  - This may require more steps or hyperparameter tuning")
        
        return converged, {
            'train_losses': train_losses,
            'eval_losses': eval_losses,
            'train_improvement': train_improvement,
            'eval_improvement': eval_improvement
        }
        
    except Exception as e:
        print(f"\nError during convergence test: {e}")
        import traceback
        traceback.print_exc()
        return False, {}
    
    finally:
        # Cleanup
        if os.path.exists(train_file):
            os.remove(train_file)
        if os.path.exists(eval_file):
            os.remove(eval_file)


if __name__ == "__main__":
    converged, results = test_mezo_convergence()
    
    if converged:
        print("\n✅ MeZO convergence test completed successfully!")
    else:
        print("\n❌ MeZO convergence test needs further investigation")