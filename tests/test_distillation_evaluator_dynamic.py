"""
Tests for dynamic prompt loading in ShadowModelEvaluator.

This test suite verifies that evaluation prompts can be loaded dynamically
from external files to prevent model memorization.
"""

import json
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Any

import pytest

from vulcan.distillation.evaluator import ShadowModelEvaluator, DEFAULT_GOLDEN_PROMPTS


class MockModel:
    """Mock model for testing."""
    
    def __init__(self, responses: Dict[str, str] = None):
        self.responses = responses or {}
        self.call_count = 0
    
    def generate(self, prompt: str, max_tokens: int = 200) -> str:
        """Generate a response based on the prompt."""
        self.call_count += 1
        return self.responses.get(prompt, "default response")


class TestDynamicPromptLoading:
    """Test dynamic prompt loading functionality."""
    
    def test_load_prompts_from_file(self):
        """Test loading prompts from an external JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_file = Path(tmpdir) / "test_prompts.json"
            
            # Create test prompts file
            test_prompts = [
                {
                    "prompt": "Test prompt 1",
                    "expected_contains": ["answer1"],
                    "domain": "test"
                },
                {
                    "prompt": "Test prompt 2",
                    "expected_contains": ["answer2"],
                    "domain": "test"
                }
            ]
            
            with open(prompts_file, 'w') as f:
                json.dump(test_prompts, f)
            
            # Create evaluator with custom prompts file
            evaluator = ShadowModelEvaluator(prompts_path=prompts_file)
            
            # Load prompts
            loaded_prompts = evaluator._load_prompts()
            
            assert len(loaded_prompts) == 2
            assert loaded_prompts[0]["prompt"] == "Test prompt 1"
            assert loaded_prompts[1]["prompt"] == "Test prompt 2"
    
    def test_fallback_to_defaults_when_file_not_found(self):
        """Test that evaluator falls back to default prompts if file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            non_existent_file = Path(tmpdir) / "non_existent.json"
            
            evaluator = ShadowModelEvaluator(prompts_path=non_existent_file)
            
            # Should fall back to default prompts
            loaded_prompts = evaluator._load_prompts()
            
            assert len(loaded_prompts) == len(DEFAULT_GOLDEN_PROMPTS)
            assert loaded_prompts == DEFAULT_GOLDEN_PROMPTS
    
    def test_fallback_to_defaults_on_invalid_json(self):
        """Test fallback to defaults when JSON file is invalid."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_file = Path(tmpdir) / "invalid.json"
            
            # Create invalid JSON file
            with open(prompts_file, 'w') as f:
                f.write("{ invalid json content")
            
            evaluator = ShadowModelEvaluator(prompts_path=prompts_file)
            
            # Should fall back to defaults
            loaded_prompts = evaluator._load_prompts()
            
            assert len(loaded_prompts) == len(DEFAULT_GOLDEN_PROMPTS)
    
    def test_fallback_to_defaults_on_empty_list(self):
        """Test fallback to defaults when JSON file contains empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_file = Path(tmpdir) / "empty.json"
            
            # Create file with empty list
            with open(prompts_file, 'w') as f:
                json.dump([], f)
            
            evaluator = ShadowModelEvaluator(prompts_path=prompts_file)
            
            # Should fall back to defaults
            loaded_prompts = evaluator._load_prompts()
            
            assert len(loaded_prompts) == len(DEFAULT_GOLDEN_PROMPTS)
    
    def test_file_modification_detection(self):
        """Test that file modifications are detected and prompts are reloaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_file = Path(tmpdir) / "test_prompts.json"
            
            # Create initial prompts file
            initial_prompts = [{"prompt": "Initial", "expected_contains": ["a"], "domain": "test"}]
            with open(prompts_file, 'w') as f:
                json.dump(initial_prompts, f)
            
            evaluator = ShadowModelEvaluator(prompts_path=prompts_file)
            
            # Load prompts (should cache)
            prompts1 = evaluator._load_prompts()
            assert len(prompts1) == 1
            assert prompts1[0]["prompt"] == "Initial"
            
            # Wait a bit to ensure mtime changes
            time.sleep(0.1)
            
            # Modify file
            updated_prompts = [
                {"prompt": "Updated 1", "expected_contains": ["b"], "domain": "test"},
                {"prompt": "Updated 2", "expected_contains": ["c"], "domain": "test"}
            ]
            with open(prompts_file, 'w') as f:
                json.dump(updated_prompts, f)
            
            # Load prompts again (should detect modification and reload)
            prompts2 = evaluator._load_prompts()
            assert len(prompts2) == 2
            assert prompts2[0]["prompt"] == "Updated 1"
            assert prompts2[1]["prompt"] == "Updated 2"
    
    def test_caching_when_file_unchanged(self):
        """Test that prompts are cached when file hasn't changed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_file = Path(tmpdir) / "test_prompts.json"
            
            # Create prompts file
            test_prompts = [{"prompt": "Test", "expected_contains": ["a"], "domain": "test"}]
            with open(prompts_file, 'w') as f:
                json.dump(test_prompts, f)
            
            evaluator = ShadowModelEvaluator(prompts_path=prompts_file)
            
            # Load prompts twice
            prompts1 = evaluator._load_prompts()
            prompts2 = evaluator._load_prompts()
            
            # Should return the same cached object
            assert prompts1 is prompts2


class TestPromptSampling:
    """Test prompt sampling to prevent memorization."""
    
    def test_sample_size_limits_prompts(self):
        """Test that sample_size limits the number of prompts used."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_file = Path(tmpdir) / "test_prompts.json"
            
            # Create 10 test prompts
            test_prompts = [
                {
                    "prompt": f"Prompt {i}",
                    "expected_contains": [f"answer{i}"],
                    "domain": "test"
                }
                for i in range(10)
            ]
            
            with open(prompts_file, 'w') as f:
                json.dump(test_prompts, f)
            
            # Create evaluator with sample_size=5
            evaluator = ShadowModelEvaluator(
                prompts_path=prompts_file,
                sample_size=5
            )
            
            # Get evaluation prompts
            sampled_prompts = evaluator.get_evaluation_prompts()
            
            # Should only return 5 prompts
            assert len(sampled_prompts) == 5
    
    def test_sample_with_random_seed_is_reproducible(self):
        """Test that sampling with a seed produces reproducible results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_file = Path(tmpdir) / "test_prompts.json"
            
            # Create 10 test prompts
            test_prompts = [
                {
                    "prompt": f"Prompt {i}",
                    "expected_contains": [f"answer{i}"],
                    "domain": "test"
                }
                for i in range(10)
            ]
            
            with open(prompts_file, 'w') as f:
                json.dump(test_prompts, f)
            
            # Create two evaluators with same seed
            evaluator1 = ShadowModelEvaluator(
                prompts_path=prompts_file,
                sample_size=5,
                random_seed=42
            )
            evaluator2 = ShadowModelEvaluator(
                prompts_path=prompts_file,
                sample_size=5,
                random_seed=42
            )
            
            # Get evaluation prompts from both
            sampled1 = evaluator1.get_evaluation_prompts()
            sampled2 = evaluator2.get_evaluation_prompts()
            
            # Should be identical
            assert len(sampled1) == len(sampled2)
            for i in range(len(sampled1)):
                assert sampled1[i]["prompt"] == sampled2[i]["prompt"]
    
    def test_sample_without_seed_varies(self):
        """Test that sampling without seed produces different results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_file = Path(tmpdir) / "test_prompts.json"
            
            # Create 10 test prompts
            test_prompts = [
                {
                    "prompt": f"Prompt {i}",
                    "expected_contains": [f"answer{i}"],
                    "domain": "test"
                }
                for i in range(10)
            ]
            
            with open(prompts_file, 'w') as f:
                json.dump(test_prompts, f)
            
            # Create evaluator without seed
            evaluator = ShadowModelEvaluator(
                prompts_path=prompts_file,
                sample_size=5,
                random_seed=None
            )
            
            # Get samples multiple times
            samples = [evaluator.get_evaluation_prompts() for _ in range(5)]
            
            # At least some samples should be different
            # (with 10 prompts and sample_size=5, there's only C(10,5)=252 combinations,
            # so we might get duplicates, but unlikely all 5 are identical)
            prompts_sets = [tuple(s[i]["prompt"] for i in range(5)) for s in samples]
            unique_sets = set(prompts_sets)
            
            # Should have at least 2 different sets (very high probability)
            assert len(unique_sets) >= 2, "Random sampling should produce variety"
    
    def test_sample_size_larger_than_pool_returns_all(self):
        """Test that requesting more samples than available returns all prompts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_file = Path(tmpdir) / "test_prompts.json"
            
            # Create only 3 test prompts
            test_prompts = [
                {
                    "prompt": f"Prompt {i}",
                    "expected_contains": [f"answer{i}"],
                    "domain": "test"
                }
                for i in range(3)
            ]
            
            with open(prompts_file, 'w') as f:
                json.dump(test_prompts, f)
            
            # Request 10 samples (more than available)
            evaluator = ShadowModelEvaluator(
                prompts_path=prompts_file,
                sample_size=10
            )
            
            sampled = evaluator.get_evaluation_prompts()
            
            # Should return all 3 available prompts
            assert len(sampled) == 3
    
    def test_no_sample_size_returns_all_prompts(self):
        """Test that not specifying sample_size returns all prompts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_file = Path(tmpdir) / "test_prompts.json"
            
            # Create 10 test prompts
            test_prompts = [
                {
                    "prompt": f"Prompt {i}",
                    "expected_contains": [f"answer{i}"],
                    "domain": "test"
                }
                for i in range(10)
            ]
            
            with open(prompts_file, 'w') as f:
                json.dump(test_prompts, f)
            
            # Don't specify sample_size
            evaluator = ShadowModelEvaluator(prompts_path=prompts_file)
            
            sampled = evaluator.get_evaluation_prompts()
            
            # Should return all 10 prompts
            assert len(sampled) == 10


class TestReloadPrompts:
    """Test manual prompt reload functionality."""
    
    def test_reload_prompts_clears_cache(self):
        """Test that reload_prompts clears the cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_file = Path(tmpdir) / "test_prompts.json"
            
            # Create initial prompts
            initial_prompts = [{"prompt": "Initial", "expected_contains": ["a"], "domain": "test"}]
            with open(prompts_file, 'w') as f:
                json.dump(initial_prompts, f)
            
            evaluator = ShadowModelEvaluator(prompts_path=prompts_file)
            
            # Load and cache
            evaluator._load_prompts()
            assert evaluator._prompts_cache is not None
            
            # Reload (clears cache)
            evaluator.reload_prompts()
            assert evaluator._prompts_cache is None
            assert evaluator._cache_mtime is None
    
    def test_reload_prompts_allows_immediate_file_changes(self):
        """Test that reload_prompts allows detecting file changes immediately."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_file = Path(tmpdir) / "test_prompts.json"
            
            # Create initial prompts
            initial = [{"prompt": "Initial", "expected_contains": ["a"], "domain": "test"}]
            with open(prompts_file, 'w') as f:
                json.dump(initial, f)
            
            evaluator = ShadowModelEvaluator(prompts_path=prompts_file)
            
            # Load initial
            prompts1 = evaluator._load_prompts()
            assert prompts1[0]["prompt"] == "Initial"
            
            # Modify file (same mtime might happen if very fast)
            updated = [{"prompt": "Updated", "expected_contains": ["b"], "domain": "test"}]
            with open(prompts_file, 'w') as f:
                json.dump(updated, f)
            
            # Reload to clear cache
            evaluator.reload_prompts()
            
            # Load again
            prompts2 = evaluator._load_prompts()
            assert prompts2[0]["prompt"] == "Updated"


class TestEvaluateWithDynamicPrompts:
    """Test that evaluate_model works with dynamically loaded prompts."""
    
    def test_evaluate_uses_dynamic_prompts(self):
        """Test that evaluate_model uses dynamically loaded prompts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_file = Path(tmpdir) / "test_prompts.json"
            
            # Create custom test prompts
            test_prompts = [
                {
                    "prompt": "Custom question",
                    "expected_contains": ["custom answer"],
                    "domain": "custom"
                }
            ]
            
            with open(prompts_file, 'w') as f:
                json.dump(test_prompts, f)
            
            evaluator = ShadowModelEvaluator(prompts_path=prompts_file)
            
            # Create mock model that responds correctly
            model = MockModel(responses={
                "Custom question": "This is a custom answer to the question."
            })
            
            # Evaluate
            results = evaluator.evaluate_model(model)
            
            # Should use custom prompts
            assert len(results["details"]) == 1
            assert results["details"][0]["domain"] == "custom"
            assert results["scores"]["custom"] == 1.0  # Should match
    
    def test_evaluate_with_sampling(self):
        """Test that evaluate_model respects sample_size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_file = Path(tmpdir) / "test_prompts.json"
            
            # Create 10 test prompts
            test_prompts = [
                {
                    "prompt": f"Question {i}",
                    "expected_contains": [f"answer{i}"],
                    "domain": f"domain{i}"
                }
                for i in range(10)
            ]
            
            with open(prompts_file, 'w') as f:
                json.dump(test_prompts, f)
            
            # Create evaluator with sampling
            evaluator = ShadowModelEvaluator(
                prompts_path=prompts_file,
                sample_size=3,
                random_seed=42  # Reproducible
            )
            
            # Create mock model
            model = MockModel(responses={
                f"Question {i}": f"This contains answer{i}"
                for i in range(10)
            })
            
            # Evaluate
            results = evaluator.evaluate_model(model)
            
            # Should only evaluate 3 prompts
            assert len(results["details"]) == 3
            assert model.call_count == 3


class TestAddGoldenPrompt:
    """Test adding prompts at runtime."""
    
    def test_add_golden_prompt_updates_cache(self):
        """Test that add_golden_prompt adds to the cache."""
        evaluator = ShadowModelEvaluator()
        
        # Load defaults first
        initial_prompts = evaluator.get_evaluation_prompts()
        initial_count = len(initial_prompts)
        
        # Add a new prompt
        evaluator.add_golden_prompt(
            prompt="New test prompt",
            expected_contains=["new", "test"],
            domain="new_domain"
        )
        
        # Should be in evaluation prompts now
        updated_prompts = evaluator.get_evaluation_prompts()
        assert len(updated_prompts) == initial_count + 1
        assert any(p["prompt"] == "New test prompt" for p in updated_prompts)
    
    def test_save_prompts_to_file(self):
        """Test saving prompts to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "saved_prompts.json"
            
            evaluator = ShadowModelEvaluator()
            
            # Add a custom prompt
            evaluator.add_golden_prompt(
                prompt="Custom prompt",
                expected_contains=["custom"],
                domain="custom"
            )
            
            # Save to file
            success = evaluator.save_prompts_to_file(save_path)
            assert success
            assert save_path.exists()
            
            # Load and verify
            with open(save_path, 'r') as f:
                saved_prompts = json.load(f)
            
            assert isinstance(saved_prompts, list)
            assert any(p["prompt"] == "Custom prompt" for p in saved_prompts)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
