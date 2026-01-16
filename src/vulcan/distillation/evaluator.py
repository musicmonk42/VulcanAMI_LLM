# ============================================================
# VULCAN-AGI Shadow Model Evaluator Module
# Evaluates model improvements before promoting weights
# ============================================================
#
# Implements the evaluation gate pattern:
#     - Dynamic loading of test prompts from external file
#     - Regression checks on critical tasks
#     - Domain-specific metrics
#     - Support for prompt sampling to prevent memorization
#
# VERSION HISTORY:
#     1.0.0 - Extracted from main.py for modular architecture
#     1.1.0 - Added dynamic prompt loading and sampling support
# ============================================================

import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Module metadata
__version__ = "1.1.0"
__author__ = "VULCAN-AGI Team"

logger = logging.getLogger(__name__)

# Default prompts as fallback
DEFAULT_GOLDEN_PROMPTS = [
    {
        "prompt": "What is 2 + 2?",
        "expected_contains": ["4"],
        "domain": "math",
    },
    {
        "prompt": "Write a simple Python function that adds two numbers.",
        "expected_contains": ["def", "return", "+"],
        "domain": "code",
    },
    {
        "prompt": "Explain what machine learning is in one sentence.",
        "expected_contains": ["learn", "data"],
        "domain": "explanation",
    },
    {
        "prompt": "What is the capital of France?",
        "expected_contains": ["Paris"],
        "domain": "factual",
    },
]


class ShadowModelEvaluator:
    """
    Evaluates model improvements before promoting weights.
    
    Implements the evaluation gate pattern:
    - Dynamic loading of test prompts from external file
    - Regression checks on critical tasks
    - Domain-specific metrics
    - Support for prompt sampling to prevent memorization
    """
    
    # Default path for evaluation prompts
    DEFAULT_PROMPTS_PATH = Path("config/evaluation_prompts.json")
    
    # Regression threshold (10% drop from baseline triggers failure)
    REGRESSION_THRESHOLD = 0.1
    
    def __init__(
        self,
        baseline_scores: Optional[Dict[str, float]] = None,
        prompts_path: Optional[Path] = None,
        sample_size: Optional[int] = None,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize the shadow model evaluator.
        
        Args:
            baseline_scores: Optional baseline scores for regression detection
            prompts_path: Path to JSON file containing evaluation prompts.
                         Falls back to DEFAULT_GOLDEN_PROMPTS if not found.
            sample_size: If set, randomly sample this many prompts from the pool.
                        Helps prevent memorization by varying the eval set.
            random_seed: Seed for reproducible sampling. If None, uses random sampling.
        """
        self.baseline_scores = baseline_scores or {}
        self.evaluation_history: List[Dict[str, Any]] = []
        self.logger = logging.getLogger("ShadowModelEvaluator")
        
        # Dynamic prompt loading configuration
        self.prompts_path = prompts_path or self.DEFAULT_PROMPTS_PATH
        self.sample_size = sample_size
        self.random_seed = random_seed
        self._prompts_cache: Optional[List[Dict[str, Any]]] = None
        self._cache_mtime: Optional[float] = None
    
    def _load_prompts(self) -> List[Dict[str, Any]]:
        """
        Load prompts from file, with caching and file modification detection.
        
        Returns:
            List of prompt dictionaries
        """
        # Check if file exists and has been modified
        if self.prompts_path.exists():
            current_mtime = self.prompts_path.stat().st_mtime
            
            # Return cached if file unchanged
            if (self._prompts_cache is not None and 
                self._cache_mtime == current_mtime):
                return self._prompts_cache
            
            # Load from file
            try:
                with open(self.prompts_path, 'r', encoding='utf-8') as f:
                    prompts = json.load(f)
                
                if not isinstance(prompts, list) or len(prompts) == 0:
                    logger.warning(
                        f"Invalid prompts file {self.prompts_path}, using defaults"
                    )
                    return DEFAULT_GOLDEN_PROMPTS
                
                self._prompts_cache = prompts
                self._cache_mtime = current_mtime
                logger.info(f"Loaded {len(prompts)} evaluation prompts from {self.prompts_path}")
                return prompts
                
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(
                    f"Failed to load prompts from {self.prompts_path}: {e}. "
                    f"Using default prompts."
                )
                return DEFAULT_GOLDEN_PROMPTS
        else:
            logger.info(
                f"Prompts file {self.prompts_path} not found, using defaults. "
                f"Create this file to customize evaluation prompts."
            )
            return DEFAULT_GOLDEN_PROMPTS
    
    def get_evaluation_prompts(self) -> List[Dict[str, Any]]:
        """
        Get prompts for evaluation, optionally sampling to prevent memorization.
        
        Returns:
            List of prompt dictionaries for evaluation
        """
        all_prompts = self._load_prompts()
        
        # If no sampling requested, return all
        if self.sample_size is None or self.sample_size >= len(all_prompts):
            return all_prompts
        
        # Sample to prevent memorization
        if self.random_seed is not None:
            rng = random.Random(self.random_seed)
        else:
            rng = random.Random()  # Truly random each time
        
        sampled = rng.sample(all_prompts, self.sample_size)
        logger.debug(f"Sampled {len(sampled)} prompts from pool of {len(all_prompts)}")
        return sampled
    
    def reload_prompts(self) -> None:
        """Force reload of prompts from file (clears cache)."""
        self._prompts_cache = None
        self._cache_mtime = None
        logger.info("Evaluation prompts cache cleared, will reload on next access")
    
    def evaluate_model(
        self,
        model: Any,
        generate_fn: Optional[Callable[[str], str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate model on golden test set.
        
        Args:
            model: The model to evaluate
            generate_fn: Optional function to use for generation
            
        Returns:
            Evaluation results with pass/fail status
        """
        results = {
            "passed": True,
            "scores": {},
            "regressions": [],
            "improvements": [],
            "details": [],
        }
        
        total_score = 0.0
        
        # Get evaluation prompts (may be sampled for variety)
        evaluation_prompts = self.get_evaluation_prompts()
        
        for test in evaluation_prompts:
            prompt = test["prompt"]
            expected = test["expected_contains"]
            domain = test.get("domain", "general")
            
            try:
                # Generate response
                if generate_fn:
                    response = generate_fn(prompt)
                elif hasattr(model, "generate"):
                    response = model.generate(prompt, max_tokens=200)
                else:
                    response = str(model(prompt))
                
                # Check for expected content
                response_lower = response.lower() if response else ""
                matches = sum(
                    1 for exp in expected
                    if exp.lower() in response_lower
                )
                score = matches / len(expected) if expected else 0.0
                
                results["details"].append({
                    "domain": domain,
                    "prompt": prompt[:50],
                    "score": score,
                    "matched": matches,
                    "expected": len(expected),
                })
                
                results["scores"][domain] = score
                total_score += score
                
                # Check for regression
                if domain in self.baseline_scores:
                    baseline = self.baseline_scores[domain]
                    if score < baseline - self.REGRESSION_THRESHOLD:
                        results["regressions"].append({
                            "domain": domain,
                            "baseline": baseline,
                            "current": score,
                        })
                        results["passed"] = False
                    elif score > baseline + self.REGRESSION_THRESHOLD:
                        results["improvements"].append({
                            "domain": domain,
                            "baseline": baseline,
                            "current": score,
                        })
                        
            except Exception as e:
                self.logger.warning(f"Evaluation failed for {domain}: {e}")
                results["details"].append({
                    "domain": domain,
                    "error": str(e),
                    "score": 0.0,
                })
        
        results["average_score"] = total_score / len(evaluation_prompts) if evaluation_prompts else 0.0
        
        # Store evaluation
        self.evaluation_history.append({
            "timestamp": time.time(),
            "results": results,
        })
        
        return results
    
    def update_baseline(self, scores: Dict[str, float]):
        """
        Update baseline scores after successful promotion.
        
        Args:
            scores: New baseline scores dictionary
        """
        self.baseline_scores.update(scores)
        self.logger.info(f"Baseline scores updated: {list(scores.keys())}")
    
    def get_evaluation_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent evaluation history.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of evaluation history entries
        """
        return self.evaluation_history[-limit:]
    
    def add_golden_prompt(
        self,
        prompt: str,
        expected_contains: List[str],
        domain: str,
    ):
        """
        Add a new prompt to the evaluation set (runtime addition).
        
        Note: This adds to the in-memory cache only. To persist prompts,
        save them to the prompts file using save_prompts_to_file().
        
        Args:
            prompt: The test prompt
            expected_contains: List of strings expected in the response
            domain: The domain/category of the test
        """
        new_prompt = {
            "prompt": prompt,
            "expected_contains": expected_contains,
            "domain": domain,
        }
        
        # Add to cache if it exists, otherwise add to default prompts
        if self._prompts_cache is not None:
            self._prompts_cache.append(new_prompt)
        else:
            # Load current prompts and add to them
            current_prompts = self._load_prompts()
            current_prompts.append(new_prompt)
            self._prompts_cache = current_prompts
        
        self.logger.info(f"Added golden prompt for domain: {domain}")
    
    def save_prompts_to_file(self, path: Optional[Path] = None) -> bool:
        """
        Save current evaluation prompts to file for persistence.
        
        Args:
            path: Optional path to save to. If None, uses self.prompts_path
            
        Returns:
            True if save was successful, False otherwise
        """
        save_path = path or self.prompts_path
        
        try:
            # Get current prompts
            prompts = self._prompts_cache or self._load_prompts()
            
            # Ensure parent directory exists
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write to file with pretty formatting
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(prompts, f, indent=4, ensure_ascii=False)
            
            self.logger.info(f"Saved {len(prompts)} evaluation prompts to {save_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save prompts to {save_path}: {e}")
            return False


__all__ = ["ShadowModelEvaluator"]
