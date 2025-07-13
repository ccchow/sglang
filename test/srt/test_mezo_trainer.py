"""
Unit tests for MeZO (Memory-efficient Zeroth-order) trainer implementation.

Tests cover:
- Gradient estimation correctness
- Dataset handling and tokenization
- KV cache optimization
- Integration with SGLang runtime
- Edge cases and error handling
"""

import unittest
import torch
import numpy as np
import tempfile
import json
from unittest.mock import Mock, MagicMock, patch

# Import the modules to test
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../python"))

from sglang.srt.mezo_trainer import MeZOTrainer, MeZODataset, create_dataloader, mezo_finetune


class TestMeZOGradientEstimation(unittest.TestCase):
    """Test MeZO gradient estimation accuracy."""
    
    def setUp(self):
        # Create a simple linear model for testing
        self.input_dim = 10
        self.output_dim = 5
        self.weight = torch.randn(self.output_dim, self.input_dim)
        self.bias = torch.randn(self.output_dim)
        
    def test_gradient_estimation_accuracy(self):
        """Test that MeZO gradient estimation is accurate for a linear model."""
        # Create a simple loss function
        x = torch.randn(1, self.input_dim)
        target = torch.randn(1, self.output_dim)
        
        def loss_fn(w, b):
            output = torch.matmul(x, w.t()) + b
            return torch.nn.functional.mse_loss(output, target)
        
        # Compute analytical gradient
        w_param = torch.nn.Parameter(self.weight.clone())
        b_param = torch.nn.Parameter(self.bias.clone())
        loss = loss_fn(w_param, b_param)
        loss.backward()
        true_grad_w = w_param.grad.clone()
        true_grad_b = b_param.grad.clone()
        
        # Compute MeZO gradient estimate
        epsilon = 1e-3
        z_w = torch.randn_like(self.weight)
        z_b = torch.randn_like(self.bias)
        
        # Positive perturbation
        loss_plus = loss_fn(self.weight + epsilon * z_w, self.bias + epsilon * z_b)
        # Negative perturbation
        loss_minus = loss_fn(self.weight - epsilon * z_w, self.bias - epsilon * z_b)
        
        # MeZO gradient estimate
        grad_scale = (loss_plus - loss_minus) / (2 * epsilon)
        mezo_grad_w = z_w * grad_scale
        mezo_grad_b = z_b * grad_scale
        
        # Check that gradients are in the same direction
        # (exact match is not expected due to approximation)
        cosine_sim_w = torch.nn.functional.cosine_similarity(
            true_grad_w.flatten(), mezo_grad_w.flatten(), dim=0
        )
        cosine_sim_b = torch.nn.functional.cosine_similarity(
            true_grad_b.flatten(), mezo_grad_b.flatten(), dim=0
        )
        
        # Gradient estimates should be roughly aligned
        self.assertGreater(cosine_sim_w.item(), 0.5)
        self.assertGreater(cosine_sim_b.item(), 0.5)


class TestMeZODataset(unittest.TestCase):
    """Test MeZODataset functionality."""
    
    def setUp(self):
        # Mock tokenizer
        self.tokenizer = Mock()
        self.tokenizer.encode = Mock(side_effect=lambda text, add_special_tokens=True: 
                                     [101] + list(range(len(text.split()))) + ([102] if add_special_tokens else []))
        self.tokenizer.pad_token_id = 0
        
    def test_dataset_from_list(self):
        """Test dataset creation from a list of examples."""
        examples = [
            {"prompt": "Hello world", "completion": "Hi there"},
            {"prompt": "How are you", "completion": "I'm fine"},
        ]
        
        dataset = MeZODataset(examples, self.tokenizer, max_length=10)
        self.assertEqual(len(dataset), 2)
        
        # Test first item
        item = dataset[0]
        self.assertIn('input_ids', item)
        self.assertIn('attention_mask', item)
        self.assertIn('prompt', item)
        self.assertIn('completion', item)
        self.assertIn('prompt_length', item)
        
        # Check shapes
        self.assertEqual(item['input_ids'].shape, (10,))
        self.assertEqual(item['attention_mask'].shape, (10,))
        
    def test_dataset_from_jsonl(self):
        """Test dataset loading from JSONL file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"prompt": "Test prompt", "completion": "Test completion"}\n')
            f.write('{"prompt": "Another prompt", "completion": "Another completion"}\n')
            temp_path = f.name
        
        try:
            dataset = MeZODataset(temp_path, self.tokenizer)
            self.assertEqual(len(dataset), 2)
            
            item = dataset[0]
            self.assertEqual(item['prompt'], "Test prompt")
            self.assertEqual(item['completion'], "Test completion")
        finally:
            os.unlink(temp_path)
    
    def test_dataset_validation(self):
        """Test that dataset validates required fields."""
        invalid_examples = [
            {"prompt": "Missing completion"},
            {"text": "Wrong field name", "response": "Wrong"}
        ]
        
        with self.assertRaises(ValueError):
            MeZODataset(invalid_examples, self.tokenizer)


class TestKVCacheOptimization(unittest.TestCase):
    """Test KV cache optimization features."""
    
    def test_in_place_perturbation(self):
        """Test that in-place perturbations work correctly."""
        # Create dummy parameters
        params = [torch.randn(10, 10) for _ in range(3)]
        original_values = [p.clone() for p in params]
        
        epsilon = 0.1
        z_list = [torch.randn_like(p) for p in params]
        
        # Apply positive perturbation
        for i, p in enumerate(params):
            p.data.add_(epsilon * z_list[i])
        
        # Check perturbation was applied
        for i, (p, orig) in enumerate(zip(params, original_values)):
            expected = orig + epsilon * z_list[i]
            torch.testing.assert_close(p, expected)
        
        # Apply negative perturbation (from +ε to -ε)
        for i, p in enumerate(params):
            p.data.add_(-2 * epsilon * z_list[i])
        
        # Check we're at -ε
        for i, (p, orig) in enumerate(zip(params, original_values)):
            expected = orig - epsilon * z_list[i]
            torch.testing.assert_close(p, expected)
        
        # Restore to original
        for i, p in enumerate(params):
            p.data.add_(epsilon * z_list[i])
        
        # Check restoration
        for p, orig in zip(params, original_values):
            torch.testing.assert_close(p, orig)


class TestMeZOIntegration(unittest.TestCase):
    """Integration tests for MeZO training."""
    
    @patch('sglang.srt.mezo_trainer.ModelRunner')
    @patch('sglang.srt.mezo_trainer.get_tokenizer')
    @patch('sglang.srt.mezo_trainer.ModelConfig')
    def test_mezo_finetune_basic(self, mock_model_config, mock_get_tokenizer, mock_model_runner):
        """Test basic mezo_finetune functionality."""
        # Setup mocks
        mock_tokenizer = Mock()
        mock_tokenizer.encode = Mock(return_value=[1, 2, 3])
        mock_tokenizer.pad_token_id = 0
        mock_get_tokenizer.return_value = mock_tokenizer
        
        mock_runner_instance = Mock()
        mock_lora_manager = Mock()
        mock_lora_adapter = Mock()
        mock_lora_adapter.layers = []
        mock_lora_manager.loras = {'mezo_lora': mock_lora_adapter}
        mock_lora_manager.load_lora_adapter = Mock()
        
        mock_runner_instance.lora_manager = mock_lora_manager
        mock_model_runner.return_value = mock_runner_instance
        
        # Test data
        train_data = [
            {"prompt": "Test", "completion": "Response"}
        ]
        
        # Run training
        result = mezo_finetune(
            model_path="test-model",
            train_dataset=train_data,
            num_steps=1,
            batch_size=1
        )
        
        # Verify result structure
        self.assertIn('weights', result)
        self.assertIn('config', result)
        self.assertEqual(result['config']['model_path'], "test-model")
        self.assertEqual(result['config']['num_steps'], 1)


class TestMeZOEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def test_empty_dataset(self):
        """Test handling of empty dataset."""
        tokenizer = Mock()
        
        with self.assertRaises(ValueError):
            dataset = MeZODataset([], tokenizer)
            # Should not allow empty dataset
    
    def test_very_small_epsilon(self):
        """Test numerical stability with very small epsilon."""
        epsilon = 1e-10
        params = [torch.randn(5, 5)]
        z_list = [torch.randn_like(p) for p in params]
        
        # Apply tiny perturbation
        original = params[0].clone()
        params[0].data.add_(epsilon * z_list[0])
        
        # Should still be numerically different
        self.assertFalse(torch.allclose(params[0], original))
        
        # But difference should be tiny
        diff = torch.abs(params[0] - original).max()
        self.assertLess(diff, 1e-8)


if __name__ == "__main__":
    unittest.main()