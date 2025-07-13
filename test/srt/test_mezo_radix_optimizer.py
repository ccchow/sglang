#!/usr/bin/env python3
"""
Comprehensive test suite for MeZORadixOptimizer.
Tests all methods and validates cache optimization functionality.
"""

import unittest
import torch
import numpy as np
from typing import Dict, List
from dataclasses import dataclass

# Add parent directory to path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sglang.srt.mezo_radix_optimizer import MeZORadixOptimizer, MeZOCacheStats
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams


class TestMeZOCacheStats(unittest.TestCase):
    """Test MeZOCacheStats dataclass and properties."""
    
    def test_initialization(self):
        """Test default initialization."""
        stats = MeZOCacheStats()
        self.assertEqual(stats.total_forward_passes, 0)
        self.assertEqual(stats.cache_hits, 0)
        self.assertEqual(stats.tokens_reused, 0)
        self.assertEqual(stats.tokens_computed, 0)
    
    def test_cache_hit_rate(self):
        """Test cache hit rate calculation."""
        stats = MeZOCacheStats()
        
        # Test with zero passes
        self.assertEqual(stats.cache_hit_rate, 0.0)
        
        # Test with some hits and misses
        stats.total_forward_passes = 10
        stats.cache_hits = 7
        self.assertAlmostEqual(stats.cache_hit_rate, 0.7)
        
        # Test perfect hit rate
        stats.cache_hits = 10
        self.assertEqual(stats.cache_hit_rate, 1.0)
    
    def test_token_reuse_rate(self):
        """Test token reuse rate calculation."""
        stats = MeZOCacheStats()
        
        # Test with zero tokens
        self.assertEqual(stats.token_reuse_rate, 0.0)
        
        # Test with some reused and computed tokens
        stats.tokens_reused = 800
        stats.tokens_computed = 200
        self.assertAlmostEqual(stats.token_reuse_rate, 0.8)
        
        # Test perfect reuse
        stats.tokens_computed = 0
        self.assertEqual(stats.token_reuse_rate, 1.0)


class TestMeZORadixOptimizer(unittest.TestCase):
    """Test MeZORadixOptimizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.optimizer = MeZORadixOptimizer(epsilon=1e-3)
        
        # Create mock batch data
        self.batch_size = 4
        self.seq_length = 128
        self.base_batch = {
            'input_ids': torch.randint(0, 50000, (self.batch_size, self.seq_length)),
            'prompt_length': torch.tensor([64, 80, 96, 128]),
            'prompt': [f"Test prompt {i}" for i in range(self.batch_size)]
        }
        
        # Create mock model config
        self.model_config = type('ModelConfig', (), {
            'hidden_size': 768,
            'num_hidden_layers': 12,
            'num_attention_heads': 12
        })()
        
        # Create mock LoRA config
        self.lora_config = type('LoRAConfig', (), {
            'target_modules': ['q_proj', 'v_proj']
        })()
    
    def test_initialization(self):
        """Test optimizer initialization."""
        self.assertEqual(self.optimizer.epsilon, 1e-3)
        self.assertIsInstance(self.optimizer.stats, MeZOCacheStats)
        self.assertEqual(len(self.optimizer._cache_state), 0)
    
    def test_prepare_mezo_requests(self):
        """Test request preparation for MeZO forward passes."""
        # Test positive perturbation requests
        plus_requests, plus_metadata = self.optimizer.prepare_mezo_requests(
            self.base_batch,
            perturbation_sign=1,
            request_prefix="test_step0"
        )
        
        self.assertEqual(len(plus_requests), self.batch_size)
        self.assertEqual(len(plus_metadata), self.batch_size)
        
        # Check request IDs
        for i, req in enumerate(plus_requests):
            expected_rid = f"test_step0_batch{i}_plus"
            self.assertEqual(req.rid, expected_rid)
            self.assertIn(expected_rid, plus_metadata)
            self.assertEqual(plus_metadata[expected_rid]['perturbation_sign'], 1)
        
        # Test negative perturbation requests
        minus_requests, minus_metadata = self.optimizer.prepare_mezo_requests(
            self.base_batch,
            perturbation_sign=-1,
            request_prefix="test_step0"
        )
        
        self.assertEqual(len(minus_requests), self.batch_size)
        
        # Check that request IDs enable cache sharing
        for i in range(self.batch_size):
            plus_rid = plus_requests[i].rid
            minus_rid = minus_requests[i].rid
            self.assertTrue(plus_rid.endswith("_plus"))
            self.assertTrue(minus_rid.endswith("_minus"))
            # Base RID should be the same
            self.assertEqual(plus_rid.replace("_plus", ""), minus_rid.replace("_minus", ""))
    
    def test_analyze_cache_potential(self):
        """Test cache potential analysis."""
        # Test with default epsilon
        analysis = self.optimizer.analyze_cache_potential(
            self.model_config,
            self.lora_config,
            epsilon=1e-3
        )
        
        self.assertIn('base_reuse_rate', analysis)
        self.assertIn('epsilon_adjusted_rate', analysis)
        self.assertIn('estimated_speedup', analysis)
        
        # With attention modules targeted, all layers affected
        self.assertEqual(analysis['base_reuse_rate'], 0.3)  # Minimum reuse
        
        # Test different epsilon values
        epsilons = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        speedups = []
        
        for eps in epsilons:
            analysis = self.optimizer.analyze_cache_potential(
                self.model_config,
                self.lora_config,
                epsilon=eps
            )
            speedups.append(analysis['estimated_speedup'])
        
        # Smaller epsilon should give higher speedup
        for i in range(len(speedups) - 1):
            self.assertGreaterEqual(speedups[i], speedups[i + 1])
    
    def test_optimize_forward_schedule(self):
        """Test forward pass scheduling optimization."""
        # Create test requests
        plus_requests = []
        minus_requests = []
        
        for i in range(4):
            # Create requests with different input patterns
            input_ids = [i] * 20  # Simple pattern for testing
            
            plus_req = Req(
                rid=f"test_batch{i}_plus",
                origin_input_text=f"Test {i}",
                origin_input_ids=input_ids,
                sampling_params=SamplingParams(temperature=0, max_new_tokens=0)
            )
            minus_req = Req(
                rid=f"test_batch{i}_minus",
                origin_input_text=f"Test {i}",
                origin_input_ids=input_ids,
                sampling_params=SamplingParams(temperature=0, max_new_tokens=0)
            )
            
            plus_requests.append(plus_req)
            minus_requests.append(minus_req)
        
        # Test scheduling optimization
        plus_sorted, minus_sorted = self.optimizer.optimize_forward_schedule(
            plus_requests,
            minus_requests,
            tree_cache=None  # Mock cache
        )
        
        self.assertEqual(len(plus_sorted), len(plus_requests))
        self.assertEqual(len(minus_sorted), len(minus_requests))
        
        # Check that cache state was updated
        self.assertGreater(len(self.optimizer._cache_state), 0)
    
    def test_update_cache_state(self):
        """Test cache state tracking."""
        # Reset optimizer
        self.optimizer = MeZORadixOptimizer(epsilon=1e-3)
        
        # Create paired requests
        plus_req = Req(
            rid="test_batch0_plus",
            origin_input_text="Test",
            origin_input_ids=[1, 2, 3, 4, 5],
            sampling_params=SamplingParams(temperature=0, max_new_tokens=0)
        )
        minus_req = Req(
            rid="test_batch0_minus",
            origin_input_text="Test",
            origin_input_ids=[1, 2, 3, 4, 5],
            sampling_params=SamplingParams(temperature=0, max_new_tokens=0)
        )
        
        # Update cache state
        self.optimizer._update_cache_state([plus_req], [minus_req])
        
        # Check statistics
        stats = self.optimizer.get_optimization_stats()
        self.assertEqual(stats['total_forward_passes'], 2)
        self.assertEqual(stats['cache_hits'], 1)  # Minus request hits cache
        self.assertEqual(stats['tokens_reused'], 5)  # All tokens reused
        self.assertEqual(stats['cache_hit_rate'], 0.5)  # 1 hit out of 2 passes
    
    def test_memory_savings_estimation(self):
        """Test memory savings calculation."""
        # Simulate some cache usage
        self.optimizer.stats.tokens_reused = 1000
        self.optimizer.stats.tokens_computed = 1000
        
        memory_stats = self.optimizer.estimate_memory_savings(
            self.model_config,
            batch_size=4,
            sequence_length=512
        )
        
        self.assertIn('memory_no_optimization_gb', memory_stats)
        self.assertIn('memory_with_optimization_gb', memory_stats)
        self.assertIn('memory_savings_gb', memory_stats)
        self.assertIn('memory_reduction_percent', memory_stats)
        
        # With 50% token reuse, memory reduction depends on model architecture
        # The actual reduction is less than 50% due to other memory overhead
        self.assertGreater(memory_stats['memory_reduction_percent'], 20.0)
        self.assertLess(memory_stats['memory_reduction_percent'], 60.0)
    
    def test_full_mezo_simulation(self):
        """Test a complete MeZO training simulation."""
        optimizer = MeZORadixOptimizer(epsilon=1e-3)
        
        num_steps = 10
        for step in range(num_steps):
            # Prepare requests for both perturbations
            plus_requests, _ = optimizer.prepare_mezo_requests(
                self.base_batch,
                perturbation_sign=1,
                request_prefix=f"step{step}"
            )
            minus_requests, _ = optimizer.prepare_mezo_requests(
                self.base_batch,
                perturbation_sign=-1,
                request_prefix=f"step{step}"
            )
            
            # Simulate forward passes
            optimizer._update_cache_state(plus_requests, minus_requests)
        
        # Check final statistics
        final_stats = optimizer.get_optimization_stats()
        
        # Should have 2 * batch_size * num_steps forward passes
        expected_passes = 2 * self.batch_size * num_steps
        self.assertEqual(final_stats['total_forward_passes'], expected_passes)
        
        # Cache hits should be half (minus passes hit cache)
        self.assertEqual(final_stats['cache_hits'], expected_passes // 2)
        
        # Cache hit rate should be 50%
        self.assertAlmostEqual(final_stats['cache_hit_rate'], 0.5)
        
        # All tokens from minus passes should be reused
        self.assertGreater(final_stats['tokens_reused'], 0)
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with empty batch
        empty_batch = {
            'input_ids': torch.empty((0, 128)),
            'prompt_length': torch.empty((0,), dtype=torch.long),
            'prompt': []
        }
        
        requests, metadata = self.optimizer.prepare_mezo_requests(
            empty_batch,
            perturbation_sign=1
        )
        
        self.assertEqual(len(requests), 0)
        self.assertEqual(len(metadata), 0)
        
        # Test with zero epsilon
        zero_eps_optimizer = MeZORadixOptimizer(epsilon=0.0)
        self.assertEqual(zero_eps_optimizer.epsilon, 0.0)
        
        # Test memory estimation with no reuse
        # Cannot set property directly, so simulate by setting underlying values
        zero_eps_optimizer.stats.tokens_reused = 0
        zero_eps_optimizer.stats.tokens_computed = 1000
        memory_stats = zero_eps_optimizer.estimate_memory_savings(
            self.model_config,
            batch_size=1,
            sequence_length=128
        )
        
        # No savings expected
        self.assertEqual(memory_stats['memory_reduction_percent'], 0.0)


class TestIntegration(unittest.TestCase):
    """Integration tests for MeZORadixOptimizer."""
    
    def test_realistic_training_scenario(self):
        """Test a realistic training scenario."""
        optimizer = MeZORadixOptimizer(epsilon=1e-3)
        
        # Simulate 50 training steps
        batch_size = 8
        seq_length = 512
        num_steps = 50
        
        for step in range(num_steps):
            # Create batch
            batch = {
                'input_ids': torch.randint(0, 50000, (batch_size, seq_length)),
                'prompt_length': torch.full((batch_size,), seq_length),
                'prompt': [f"Training sample {i}" for i in range(batch_size)]
            }
            
            # Prepare and process requests
            plus_reqs, _ = optimizer.prepare_mezo_requests(
                batch, 1, f"train_step{step}"
            )
            minus_reqs, _ = optimizer.prepare_mezo_requests(
                batch, -1, f"train_step{step}"
            )
            
            # Update cache state
            optimizer._update_cache_state(plus_reqs, minus_reqs)
        
        # Verify statistics
        stats = optimizer.get_optimization_stats()
        
        # Should achieve 50% cache hit rate
        self.assertAlmostEqual(stats['cache_hit_rate'], 0.5, delta=0.01)
        
        # Token reuse should be high
        self.assertGreater(stats['token_reuse_rate'], 0.0)
        
        # Check memory savings
        model_config = type('Config', (), {
            'hidden_size': 4096,
            'num_hidden_layers': 32,
            'num_attention_heads': 32
        })()
        
        memory_stats = optimizer.estimate_memory_savings(
            model_config, batch_size, seq_length
        )
        
        # Should have significant memory savings
        self.assertGreater(memory_stats['memory_savings_gb'], 0)
        self.assertGreater(memory_stats['memory_reduction_percent'], 40)


if __name__ == '__main__':
    unittest.main(verbosity=2)