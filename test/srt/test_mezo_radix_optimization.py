#!/usr/bin/env python3
"""
Test MeZO RadixAttention optimization for KV cache reuse.
"""

import torch
import unittest
from unittest.mock import Mock, patch
import numpy as np

from sglang.srt.mezo_radix_optimizer import MeZORadixOptimizer, MeZOCacheStats


class TestMeZORadixOptimization(unittest.TestCase):
    """Test RadixAttention optimization for MeZO."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.optimizer = MeZORadixOptimizer(epsilon=1e-3)
        
    def test_request_preparation(self):
        """Test request preparation for cache optimization."""
        # Create mock batch
        batch = {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]),
            'prompt_length': torch.tensor([4, 5]),
            'prompt': ['Test prompt 1', 'Test prompt 2']
        }
        
        # Prepare requests for +εz
        plus_requests, plus_metadata = self.optimizer.prepare_mezo_requests(
            batch, perturbation_sign=1, request_prefix="test"
        )
        
        # Prepare requests for -εz
        minus_requests, minus_metadata = self.optimizer.prepare_mezo_requests(
            batch, perturbation_sign=-1, request_prefix="test"
        )
        
        # Check request creation
        self.assertEqual(len(plus_requests), 2)
        self.assertEqual(len(minus_requests), 2)
        
        # Check request IDs enable cache sharing
        self.assertEqual(plus_requests[0].rid, "test_batch0_plus")
        self.assertEqual(minus_requests[0].rid, "test_batch0_minus")
        
        # Check metadata
        self.assertEqual(plus_metadata["test_batch0_plus"]["base_rid"], "test_batch0")
        self.assertEqual(plus_metadata["test_batch0_plus"]["perturbation_sign"], 1)
        self.assertEqual(minus_metadata["test_batch0_minus"]["perturbation_sign"], -1)
    
    def test_cache_potential_analysis(self):
        """Test cache potential analysis for different configurations."""
        # Mock model config
        model_config = Mock()
        model_config.num_hidden_layers = 32
        model_config.hidden_size = 4096
        model_config.num_attention_heads = 32
        
        # Mock LoRA config
        lora_config = Mock()
        lora_config.target_modules = ['q_proj', 'v_proj']
        
        # Analyze cache potential
        analysis = self.optimizer.analyze_cache_potential(
            model_config, lora_config, epsilon=1e-3
        )
        
        # Check analysis results
        self.assertIn('base_reuse_rate', analysis)
        self.assertIn('epsilon_adjusted_rate', analysis)
        self.assertIn('estimated_speedup', analysis)
        
        # For small epsilon, should have higher reuse rate than large epsilon
        small_epsilon_analysis = self.optimizer.analyze_cache_potential(
            model_config, lora_config, epsilon=1e-5
        )
        self.assertGreater(small_epsilon_analysis['epsilon_adjusted_rate'], 0.25)
        
        # For large epsilon, should have lower reuse rate
        large_epsilon_analysis = self.optimizer.analyze_cache_potential(
            model_config, lora_config, epsilon=1e-1
        )
        self.assertLess(large_epsilon_analysis['epsilon_adjusted_rate'], 0.2)
    
    def test_cache_state_tracking(self):
        """Test cache state tracking and hit rate calculation."""
        # Create mock requests
        plus_requests = [
            Mock(rid="test_batch0_plus", origin_input_ids=[1, 2, 3, 4]),
            Mock(rid="test_batch1_plus", origin_input_ids=[5, 6, 7, 8])
        ]
        minus_requests = [
            Mock(rid="test_batch0_minus", origin_input_ids=[1, 2, 3, 4]),
            Mock(rid="test_batch1_minus", origin_input_ids=[5, 6, 7, 8])
        ]
        
        # Update cache state
        self.optimizer._update_cache_state(plus_requests, minus_requests)
        
        # Check statistics
        stats = self.optimizer.get_optimization_stats()
        self.assertEqual(stats['total_forward_passes'], 4)
        self.assertEqual(stats['cache_hits'], 2)  # minus requests can reuse plus cache
        self.assertEqual(stats['tokens_reused'], 8)  # 4 tokens × 2 requests
        self.assertEqual(stats['cache_hit_rate'], 0.5)  # 2 hits / 4 passes
    
    def test_memory_savings_estimation(self):
        """Test memory savings estimation."""
        # Mock model config
        model_config = Mock()
        model_config.hidden_size = 4096
        model_config.num_hidden_layers = 32
        model_config.num_attention_heads = 32
        
        # Set up some cache statistics
        self.optimizer.stats.tokens_reused = 1000
        self.optimizer.stats.tokens_computed = 200
        
        # Estimate memory savings
        savings = self.optimizer.estimate_memory_savings(
            model_config, batch_size=4, sequence_length=512
        )
        
        # Check results
        self.assertIn('memory_no_optimization_gb', savings)
        self.assertIn('memory_with_optimization_gb', savings)
        self.assertIn('memory_savings_gb', savings)
        self.assertIn('memory_reduction_percent', savings)
        
        # Should have some memory savings
        self.assertGreater(savings['memory_savings_gb'], 0)
        self.assertGreater(savings['memory_reduction_percent'], 0)
    
    def test_forward_schedule_optimization(self):
        """Test forward pass scheduling optimization."""
        # Create mock requests with different prompts
        plus_requests = [
            Mock(rid="req0_plus", origin_input_ids=[1, 2, 3, 4, 5]),
            Mock(rid="req1_plus", origin_input_ids=[1, 2, 3, 6, 7]),
            Mock(rid="req2_plus", origin_input_ids=[8, 9, 10, 11, 12])
        ]
        minus_requests = [
            Mock(rid="req0_minus", origin_input_ids=[1, 2, 3, 4, 5]),
            Mock(rid="req1_minus", origin_input_ids=[1, 2, 3, 6, 7]),
            Mock(rid="req2_minus", origin_input_ids=[8, 9, 10, 11, 12])
        ]
        
        # Optimize scheduling
        tree_cache = Mock()
        plus_sorted, minus_sorted = self.optimizer.optimize_forward_schedule(
            plus_requests, minus_requests, tree_cache
        )
        
        # Check that requests with similar prefixes are grouped
        # Requests 0 and 1 share prefix [1, 2, 3], so should be adjacent
        similar_prefixes = (
            plus_sorted[0].origin_input_ids[:3] == plus_sorted[1].origin_input_ids[:3]
            or plus_sorted[1].origin_input_ids[:3] == plus_sorted[2].origin_input_ids[:3]
        )
        self.assertTrue(similar_prefixes)


class TestRadixCacheIntegration(unittest.TestCase):
    """Test integration with SGLang's RadixCache."""
    
    @patch('sglang.srt.mezo_trainer.get_tensor_model_parallel_world_size', return_value=1)
    @patch('sglang.srt.mezo_trainer.get_tensor_model_parallel_rank', return_value=0)
    @patch('sglang.srt.mezo_trainer.get_tp_group', return_value=None)
    @patch('sglang.srt.mezo_trainer.ScheduleBatch')
    @patch('sglang.srt.mezo_trainer.ModelWorkerBatch')
    def test_radix_optimized_forward_pass(self, mock_model_batch, mock_schedule_batch, 
                                        mock_tp_group, mock_tp_rank, mock_tp_size):
        """Test RadixAttention-optimized forward pass integration."""
        from sglang.srt.mezo_trainer import MeZOTrainer
        
        # Create mocked components
        model_runner = Mock()
        model_runner.req_to_token_pool = Mock()
        model_runner.token_to_kv_pool = Mock()
        model_runner.tree_cache = Mock()
        model_runner.forward = Mock(return_value=(torch.randn(2, 5, 100), None))
        
        lora_manager = Mock()
        lora_manager.device = torch.device('cpu')
        lora_manager.loras = {'test': Mock(layers=[])}
        
        tokenizer = Mock()
        
        # Create trainer with RadixAttention enabled
        trainer = MeZOTrainer(model_runner, lora_manager, 'test', tokenizer)
        trainer.enable_kv_cache_optimization = True
        trainer.current_step = 0
        
        # Create test batch
        batch = {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]),
            'prompt_length': torch.tensor([5, 5]),
            'prompt': ['Test 1', 'Test 2']
        }
        
        # Test optimized forward pass
        lora_params = [torch.randn(10, 10)]
        z_list = [torch.randn_like(p) for p in lora_params]
        
        # Mock schedule batch initialization
        mock_schedule_batch.init_new.return_value = Mock(
            forward_mode=Mock(),
            req_pool_indices=[0, 1],
            seq_lens=[5, 5],
            prefix_lens=[0, 0],
            extend_lens=[5, 5]
        )
        
        # Run optimized forward passes
        loss_plus, loss_minus = trainer._forward_pass_radix_optimized(
            batch, lora_params, epsilon=1e-3, z_list=z_list
        )
        
        # Verify forward passes were called
        self.assertEqual(model_runner.forward.call_count, 2)
        
        # Check that requests were created for cache optimization
        self.assertEqual(mock_schedule_batch.init_new.call_count, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)