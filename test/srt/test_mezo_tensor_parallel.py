#!/usr/bin/env python3
"""
Test MeZO training with tensor parallelism.
This test verifies that MeZO works correctly with tensor parallel configurations.
"""

import torch
import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

class TestMeZOTensorParallel(unittest.TestCase):
    """Test MeZO trainer with tensor parallelism."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        
    @patch('sglang.srt.mezo_trainer.get_tensor_model_parallel_world_size')
    @patch('sglang.srt.mezo_trainer.get_tensor_model_parallel_rank')
    @patch('sglang.srt.mezo_trainer.get_tp_group')
    @patch('sglang.srt.mezo_trainer.tensor_model_parallel_all_reduce')
    def test_synchronized_perturbations(self, mock_all_reduce, mock_get_tp_group, 
                                      mock_get_rank, mock_get_size):
        """Test that perturbations are synchronized across TP ranks."""
        # Mock TP configuration
        mock_get_size.return_value = 2
        mock_get_rank.return_value = 0
        
        # Mock TP group
        mock_tp_group = Mock()
        mock_tp_group.device_group = Mock()
        mock_get_tp_group.return_value = mock_tp_group
        
        # Mock broadcast to verify it's called
        with patch('torch.distributed.broadcast') as mock_broadcast:
            from sglang.srt.mezo_trainer import MeZOTrainer
            
            # Create mock components
            model_runner = Mock()
            lora_manager = Mock()
            lora_manager.device = torch.device('cuda')
            lora_manager.loras = {'test_lora': Mock()}
            tokenizer = Mock()
            
            trainer = MeZOTrainer(model_runner, lora_manager, 'test_lora', tokenizer)
            
            # Test perturbation generation
            lora_params = [torch.randn(10, 10, device='cuda')]
            z_list = trainer._generate_synchronized_perturbations(lora_params)
            
            # Verify broadcast was called
            mock_broadcast.assert_called_once()
            broadcast_args = mock_broadcast.call_args[0]
            self.assertIsInstance(broadcast_args[0], torch.Tensor)  # seed tensor
            
            # Verify z_list has correct shape
            self.assertEqual(len(z_list), 1)
            self.assertEqual(z_list[0].shape, lora_params[0].shape)
    
    @patch('sglang.srt.mezo_trainer.get_tensor_model_parallel_world_size')
    @patch('sglang.srt.mezo_trainer.get_tensor_model_parallel_rank')
    @patch('sglang.srt.mezo_trainer.get_tp_group')
    def test_loss_aggregation(self, mock_get_tp_group, mock_get_rank, mock_get_size):
        """Test loss aggregation across TP ranks."""
        # Mock TP configuration
        mock_get_size.return_value = 2
        mock_get_rank.return_value = 0
        mock_tp_group = Mock()
        mock_get_tp_group.return_value = mock_tp_group
        
        from sglang.srt.mezo_trainer import MeZOTrainer
        
        # Create trainer
        model_runner = Mock()
        lora_manager = Mock()
        lora_manager.device = torch.device('cuda')
        tokenizer = Mock()
        
        trainer = MeZOTrainer(model_runner, lora_manager, 'test_lora', tokenizer)
        
        # Test loss aggregation
        with patch('sglang.srt.mezo_trainer.tensor_model_parallel_all_reduce') as mock_all_reduce:
            # Mock all_reduce to simulate averaging
            def simulate_all_reduce(tensor):
                tensor.mul_(2)  # Simulate sum of 2 ranks
            
            mock_all_reduce.side_effect = simulate_all_reduce
            
            # Test with scalar loss
            loss = 4.0
            aggregated_loss = trainer._aggregate_loss_across_tp(loss)
            
            # Should be averaged: (4.0 * 2) / 2 = 4.0
            self.assertAlmostEqual(aggregated_loss, 4.0, places=4)
            
            # Test with tensor loss
            loss_tensor = torch.tensor(6.0, device='cuda')
            aggregated_loss = trainer._aggregate_loss_across_tp(loss_tensor)
            
            # Should be averaged: (6.0 * 2) / 2 = 6.0
            self.assertAlmostEqual(aggregated_loss, 6.0, places=4)
    
    @patch('sglang.srt.mezo_trainer.get_tensor_model_parallel_world_size')
    @patch('sglang.srt.mezo_trainer.get_tensor_model_parallel_rank')
    def test_single_gpu_fallback(self, mock_get_rank, mock_get_size):
        """Test that single GPU configuration works without TP."""
        # Mock single GPU
        mock_get_size.return_value = 1
        mock_get_rank.return_value = 0
        
        from sglang.srt.mezo_trainer import MeZOTrainer
        
        # Create trainer
        model_runner = Mock()
        lora_manager = Mock()
        lora_manager.device = torch.device('cuda')
        lora_manager.loras = {'test_lora': Mock(layers=[])}
        tokenizer = Mock()
        
        trainer = MeZOTrainer(model_runner, lora_manager, 'test_lora', tokenizer)
        
        # Verify TP is not enabled
        self.assertEqual(trainer.tp_size, 1)
        self.assertIsNone(trainer.tp_group)
        
        # Test that perturbations are generated without synchronization
        lora_params = [torch.randn(5, 5)]
        
        # Mock the forward pass
        trainer._forward_pass = Mock(return_value=1.0)
        
        # Create mock optimizer
        optimizer = Mock()
        optimizer.zero_grad = Mock()
        optimizer.step = Mock()
        
        # Run a MeZO step
        batch = {'input_ids': torch.tensor([[1, 2, 3]])}
        loss = trainer._mezo_step(batch, lora_params, optimizer, epsilon=1e-3)
        
        # Verify it runs without errors
        self.assertIsInstance(loss, float)
        optimizer.step.assert_called_once()
    
    def test_tp_info_logging(self):
        """Test that TP information is logged correctly."""
        with patch('sglang.srt.mezo_trainer.get_tensor_model_parallel_world_size') as mock_size:
            with patch('sglang.srt.mezo_trainer.get_tensor_model_parallel_rank') as mock_rank:
                with patch('sglang.srt.mezo_trainer.get_tp_group') as mock_group:
                    mock_size.return_value = 4
                    mock_rank.return_value = 2
                    mock_group.return_value = Mock()
                    
                    from sglang.srt.mezo_trainer import MeZOTrainer
                    
                    # Create trainer with mocked logger
                    model_runner = Mock()
                    lora_manager = Mock()
                    lora_manager.device = torch.device('cuda')
                    tokenizer = Mock()
                    
                    with patch('logging.getLogger') as mock_logger:
                        logger_instance = Mock()
                        mock_logger.return_value = logger_instance
                        
                        trainer = MeZOTrainer(model_runner, lora_manager, 'test_lora', tokenizer)
                        
                        # Check that TP info was logged
                        logger_instance.info.assert_any_call(
                            "Tensor parallelism enabled: size=4, rank=2"
                        )


if __name__ == "__main__":
    unittest.main(verbosity=2)