# ============================================================
# VULCAN-AGI Shadow Model Evaluator Module
# Evaluates model improvements before promoting weights
# ============================================================
#
# Implements the evaluation gate pattern:
#     - Golden set of frozen test prompts
#     - Regression checks on critical tasks
#     - Domain-specific metrics
#
# VERSION HISTORY:
#     1.0.0 - Extracted from main.py for modular architecture
# ============================================================

import logging
import time
from typing import Any, Callable, Dict, List, Optional

# Module metadata
__version__ = "1.0.0"
__author__ = "VULCAN-AGI Team"

logger = logging.getLogger(__name__)


class ShadowModelEvaluator:
    """
    Evaluates model improvements before promoting weights.
    
    Implements the evaluation gate pattern:
    - Golden set of frozen test prompts
    - Regression checks on critical tasks
    - Domain-specific metrics
    """
    
    # Golden test set (frozen prompts for consistent evaluation)
    GOLDEN_PROMPTS = [
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
    
    # Regression threshold (10% drop from baseline triggers failure)
    REGRESSION_THRESHOLD = 0.1
    
    def __init__(self, baseline_scores: Optional[Dict[str, float]] = None):
        """
        Initialize the shadow model evaluator.
        
        Args:
            baseline_scores: Optional baseline scores for regression detection
        """
        self.baseline_scores = baseline_scores or {}
        self.evaluation_history: List[Dict[str, Any]] = []
        self.logger = logging.getLogger("ShadowModelEvaluator")
    
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
        
        for test in self.GOLDEN_PROMPTS:
            prompt = test["prompt"]
            expected = test["expected_contains"]
            domain = test["domain"]
            
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
        
        results["average_score"] = total_score / len(self.GOLDEN_PROMPTS)
        
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
        Add a new prompt to the golden test set.
        
        Args:
            prompt: The test prompt
            expected_contains: List of strings expected in the response
            domain: The domain/category of the test
        """
        self.GOLDEN_PROMPTS.append({
            "prompt": prompt,
            "expected_contains": expected_contains,
            "domain": domain,
        })
        self.logger.info(f"Added golden prompt for domain: {domain}")


__all__ = ["ShadowModelEvaluator"]
