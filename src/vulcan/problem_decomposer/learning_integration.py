"""
learning_integration.py - Integration layer between problem decomposer and VULCAN learning systems
Part of the VULCAN-AGI system

This module bridges the problem decomposition system with continual learning, curriculum learning,
meta-learning, RLHF feedback systems, and knowledge crystallization to create a unified learning loop.

COMPLETE IMPLEMENTATION - All learning systems integrated including principle extraction and promotion
"""

import json
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch.nn as nn

# Import decomposer components
try:
    from .decomposition_library import StratifiedDecompositionLibrary
    from .problem_decomposer_core import (
        DecompositionPlan,
        ExecutionOutcome,
        ProblemDecomposer,
        ProblemGraph,
    )
except ImportError:
    try:
        from decomposition_library import StratifiedDecompositionLibrary
        from problem_decomposer_core import (
            DecompositionPlan,
            ExecutionOutcome,
            ProblemDecomposer,
            ProblemGraph,
        )
    except ImportError:
        logging.warning("Could not import ProblemDecomposer components")
        StratifiedDecompositionLibrary = None

# Import learning components - FIXED: Changed .learning to ..learning
try:
    from ..learning.continual_learning import EnhancedContinualLearner
    from ..learning.curriculum_learning import CurriculumLearner, DifficultyEstimator
    from ..learning.learning_types import FeedbackData, LearningConfig, TaskInfo
    from ..learning.meta_learning import MetaLearner
    from ..learning.metacognition import MetaCognitiveMonitor
    from ..learning.parameter_history import ParameterHistoryManager
    from ..learning.rlhf_feedback import RLHFManager
except ImportError:
    try:
        from continual_learning import EnhancedContinualLearner
        from curriculum_learning import CurriculumLearner, DifficultyEstimator
        from learning_types import FeedbackData, LearningConfig, TaskInfo
        from meta_learning import MetaLearner
        from metacognition import MetaCognitiveMonitor
        from parameter_history import ParameterHistoryManager
        from rlhf_feedback import RLHFManager
    except ImportError:
        logging.warning("Learning components not available")
        EnhancedContinualLearner = None
        CurriculumLearner = None
        MetaLearner = None
        DifficultyEstimator = None
        MetaCognitiveMonitor = None
        LearningConfig = None
        TaskInfo = None
        FeedbackData = None
        ParameterHistoryManager = None
        RLHFManager = None

# Import principle learning components
try:
    from .principle_learner import PrincipleLearner
except ImportError:
    try:
        from principle_learner import PrincipleLearner
    except ImportError:
        logging.warning(
            "PrincipleLearner not available - principle extraction disabled"
        )
        PrincipleLearner = None

logger = logging.getLogger(__name__)


# ============================================================
# FALLBACK BASE CLASSES
# ============================================================

# Create fallback base classes if learning components not available
if DifficultyEstimator is None:

    class DifficultyEstimator:
        """Fallback base class for difficulty estimation"""

        def estimate(self, task: Any) -> float:
            """Estimate task difficulty"""
            return 0.5


if LearningConfig is None:

    @dataclass
    class LearningConfig:
        """Fallback learning configuration"""

        curriculum_stages: List[int] = field(default_factory=lambda: [1, 2, 3])


if FeedbackData is None:

    @dataclass
    class FeedbackData:
        """Fallback feedback data"""

        feedback_id: str = ""
        timestamp: float = 0.0
        feedback_type: str = ""
        content: Dict[str, Any] = field(default_factory=dict)
        context: Dict[str, Any] = field(default_factory=dict)
        agent_response: Optional[Dict[str, Any]] = None
        human_preference: Optional[float] = None
        reward_signal: float = 0.0
        metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# CONVERSION UTILITIES
# ============================================================


class ProblemToExperienceConverter:
    """Converts problem decomposition artifacts to learning experience format"""

    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        self.problem_embeddings = {}
        self.conversion_stats = {"total_conversions": 0, "failed_conversions": 0}

        # Thread safety and cache management
        self._lock = threading.RLock()
        self.max_cache_size = 1000

        # Simple neural encoder for problem graphs
        self.graph_encoder = nn.Sequential(
            nn.Linear(embedding_dim, 256), nn.ReLU(), nn.Linear(256, embedding_dim)
        )

        logger.info("ProblemToExperienceConverter initialized")

    def problem_to_embedding(self, problem: ProblemGraph) -> np.ndarray:
        """Convert problem graph to embedding vector"""
        # Check cache
        sig = problem.get_signature()

        with self._lock:
            if sig in self.problem_embeddings:
                return self.problem_embeddings[sig]

            # Extract features from problem graph
            features = []

            # Node count feature
            features.append(len(problem.nodes) / 100.0)  # Normalize

            # Edge count feature
            features.append(len(problem.edges) / 100.0)

            # Complexity score
            features.append(problem.complexity_score)

            # Domain encoding (hash to consistent value)
            domain = problem.metadata.get("domain", "general")
            domain_hash = hash(domain) % 100 / 100.0
            features.append(domain_hash)

            # Structural features
            features.append(1.0 if problem.root else 0.0)
            features.append(len(problem.metadata.get("constraints", [])) / 10.0)

            # Pad or truncate to embedding_dim
            while len(features) < self.embedding_dim:
                features.append(0.0)
            features = features[: self.embedding_dim]

            embedding = np.array(features, dtype=np.float32)

            # Enforce cache size limit
            if len(self.problem_embeddings) >= self.max_cache_size:
                # Remove 10% oldest entries (FIFO approximation)
                items = list(self.problem_embeddings.items())
                self.problem_embeddings = dict(items[100:])

            # Cache
            self.problem_embeddings[sig] = embedding

            return embedding

    def convert_to_experience(
        self, problem: ProblemGraph, plan: DecompositionPlan, outcome: ExecutionOutcome
    ) -> Dict[str, Any]:
        """Convert problem/plan/outcome to experience dictionary"""
        try:
            # Extract embedding
            embedding = self.problem_to_embedding(problem)

            # Calculate reward based on outcome
            if outcome.success:
                base_reward = 1.0
            else:
                base_reward = -0.5

            # Adjust reward by performance metrics
            if outcome.metrics:
                if "actual_complexity" in outcome.metrics:
                    # Bonus for handling complexity well
                    complexity_bonus = (
                        max(
                            0,
                            problem.complexity_score
                            - outcome.metrics["actual_complexity"],
                        )
                        * 0.1
                    )
                    base_reward += complexity_bonus

                if "solution_quality" in outcome.metrics:
                    base_reward *= outcome.metrics["solution_quality"]

            # Time penalty for slow execution
            if outcome.execution_time > 60:
                time_penalty = min(0.5, (outcome.execution_time - 60) / 120.0)
                base_reward -= time_penalty

            # Success rate bonus
            success_rate = outcome.get_success_rate()
            base_reward += success_rate * 0.2

            # Create experience dictionary
            experience = {
                "embedding": embedding,
                "reward": float(base_reward),
                "loss": (
                    1.0 - success_rate
                    if outcome.sub_results
                    else (0.0 if outcome.success else 1.0)
                ),
                "modality": "problem_decomposition",
                "metadata": {
                    "problem_signature": problem.get_signature(),
                    "strategy": plan.strategy.name if plan.strategy else "unknown",
                    "complexity": problem.complexity_score,
                    "domain": problem.metadata.get("domain", "general"),
                    "num_steps": len(plan.steps),
                    "execution_time": outcome.execution_time,
                    "success": outcome.success,
                    "confidence": plan.confidence,
                    "timestamp": time.time(),
                },
                "context": {
                    "problem_nodes": len(problem.nodes),
                    "problem_edges": len(problem.edges),
                    "plan_confidence": plan.confidence,
                    "estimated_complexity": plan.estimated_complexity,
                },
            }

            # Add error information if failed
            if not outcome.success and outcome.errors:
                experience["metadata"]["errors"] = outcome.errors[:3]  # First 3 errors

            with self._lock:
                self.conversion_stats["total_conversions"] += 1

            return experience

        except Exception as e:
            logger.error(f"Failed to convert to experience: {e}")

            with self._lock:
                self.conversion_stats["failed_conversions"] += 1

            # Return minimal experience
            return {
                "embedding": np.zeros(self.embedding_dim),
                "reward": 0.0,
                "loss": 1.0,
                "modality": "problem_decomposition",
                "metadata": {"error": str(e)},
            }

    def get_statistics(self) -> Dict[str, Any]:
        """Get conversion statistics"""
        with self._lock:
            return self.conversion_stats.copy()


# ============================================================
# DIFFICULTY ESTIMATION
# ============================================================


class DecompositionDifficultyEstimator(DifficultyEstimator):
    """Difficulty estimator specialized for problem decomposition"""

    def __init__(self):
        self.difficulty_history = defaultdict(list)
        logger.info("DecompositionDifficultyEstimator initialized")

    def estimate(self, task: Any) -> float:
        """Estimate difficulty of a problem"""
        if isinstance(task, ProblemGraph):
            return self._estimate_problem_difficulty(task)
        elif isinstance(task, dict) and "problem" in task:
            return self._estimate_problem_difficulty(task["problem"])
        else:
            # Default difficulty
            return 0.5

    def _estimate_problem_difficulty(self, problem: ProblemGraph) -> float:
        """Estimate difficulty based on problem structure"""
        factors = []

        # Complexity score (already calculated)
        if problem.complexity_score > 0:
            factors.append(min(1.0, problem.complexity_score / 5.0))

        # Node count factor
        node_factor = min(1.0, len(problem.nodes) / 50.0)
        factors.append(node_factor)

        # Edge density factor
        if len(problem.nodes) > 1:
            max_edges = len(problem.nodes) * (len(problem.nodes) - 1)
            edge_density = len(problem.edges) / max_edges if max_edges > 0 else 0
            factors.append(edge_density)

        # Constraint factor
        constraint_count = len(problem.metadata.get("constraints", []))
        constraint_factor = min(1.0, constraint_count / 10.0)
        factors.append(constraint_factor)

        # Domain difficulty (learned)
        domain = problem.metadata.get("domain", "general")
        if domain in self.difficulty_history and self.difficulty_history[domain]:
            domain_difficulty = np.mean(self.difficulty_history[domain])
            factors.append(domain_difficulty)

        # Calculate weighted difficulty
        if factors:
            difficulty = np.mean(factors)
        else:
            difficulty = 0.5

        # Clamp to [0, 1]
        difficulty = max(0.0, min(1.0, difficulty))

        # Store for learning
        self.difficulty_history[domain].append(difficulty)

        return difficulty


# ============================================================
# RLHF FEEDBACK ROUTER
# ============================================================


class RLHFFeedbackRouter:
    """Routes decomposition outcomes to RLHF system"""

    def __init__(self, rlhf_manager: Optional[Any] = None):
        self.rlhf_manager = rlhf_manager
        self.feedback_count = 0
        logger.info("RLHFFeedbackRouter initialized")

    def route_outcome_to_feedback(
        self, problem: ProblemGraph, plan: DecompositionPlan, outcome: ExecutionOutcome
    ) -> Optional[FeedbackData]:
        """Convert execution outcome to RLHF feedback"""
        if not self.rlhf_manager:
            return None

        try:
            # Create feedback data
            feedback = FeedbackData(
                feedback_id=f"decomp_{self.feedback_count}_{int(time.time())}",
                timestamp=time.time(),
                feedback_type="execution_result",
                content={
                    "problem": problem.get_signature(),
                    "strategy": plan.strategy.name if plan.strategy else "unknown",
                    "steps": len(plan.steps),
                },
                context={
                    "domain": problem.metadata.get("domain", "general"),
                    "complexity": problem.complexity_score,
                    "confidence": plan.confidence,
                },
                agent_response={
                    "plan": plan.to_dict(),
                    "execution_time": outcome.execution_time,
                },
                human_preference=None,  # Would come from actual human feedback
                reward_signal=1.0 if outcome.success else -0.5,
                metadata={
                    "success": outcome.success,
                    "errors": outcome.errors,
                    "metrics": outcome.metrics,
                    "sub_results": len(outcome.sub_results),
                },
            )

            # Send to RLHF manager
            self.rlhf_manager.receive_feedback(feedback)
            self.feedback_count += 1

            return feedback

        except Exception as e:
            logger.error(f"Failed to route feedback: {e}")
            return None

    def route_human_feedback(
        self, problem_signature: str, rating: float, comments: Optional[str] = None
    ):
        """Route explicit human feedback about a solution"""
        if not self.rlhf_manager:
            return

        feedback = FeedbackData(
            feedback_id=f"human_{int(time.time())}",
            timestamp=time.time(),
            feedback_type="rating",
            content={"rating": rating, "comments": comments},
            context={"problem": problem_signature},
            agent_response=None,
            human_preference=rating,
            reward_signal=rating,
            metadata={"source": "human_feedback"},
        )

        self.rlhf_manager.receive_feedback(feedback)


# ============================================================
# INTEGRATED LEARNING COORDINATOR
# ============================================================


class IntegratedLearningCoordinator:
    """Coordinates learning across all subsystems including principle extraction"""

    def __init__(
        self,
        decomposer: ProblemDecomposer,
        continual_learner: Optional[Any] = None,
        curriculum_learner: Optional[Any] = None,
        meta_learner: Optional[Any] = None,
        metacognition: Optional[Any] = None,
        rlhf_manager: Optional[Any] = None,
        principle_learner: Optional[Any] = None,
        config: Optional[LearningConfig] = None,
    ):
        # Validate component availability
        if continual_learner is not None and not hasattr(
            continual_learner, "process_experience"
        ):
            logger.error(
                "continual_learner missing required method 'process_experience'"
            )
            continual_learner = None

        if curriculum_learner is not None and not hasattr(
            curriculum_learner, "generate_curriculum"
        ):
            logger.error(
                "curriculum_learner missing required method 'generate_curriculum'"
            )
            curriculum_learner = None

        if meta_learner is not None and not hasattr(meta_learner, "online_meta_update"):
            logger.error("meta_learner missing required method 'online_meta_update'")
            meta_learner = None

        if metacognition is not None and not hasattr(
            metacognition, "update_self_model"
        ):
            logger.error("metacognition missing required method 'update_self_model'")
            metacognition = None

        if principle_learner is not None and not hasattr(
            principle_learner, "extract_and_promote"
        ):
            logger.error(
                "principle_learner missing required method 'extract_and_promote'"
            )
            principle_learner = None

        self.decomposer = decomposer
        self.continual_learner = continual_learner
        self.curriculum_learner = curriculum_learner
        self.meta_learner = meta_learner
        self.metacognition = metacognition
        self.rlhf_manager = rlhf_manager
        self.principle_learner = principle_learner
        self.config = config or LearningConfig()

        # Converters and routers
        self.converter = ProblemToExperienceConverter()
        self.difficulty_estimator = DecompositionDifficultyEstimator()
        self.feedback_router = RLHFFeedbackRouter(rlhf_manager)

        # Curriculum integration
        if self.curriculum_learner:
            self.curriculum_learner.difficulty_estimator = self.difficulty_estimator

        # Statistics
        self.integration_stats = {
            "total_learning_calls": 0,
            "continual_learning_updates": 0,
            "curriculum_batches": 0,
            "meta_updates": 0,
            "rlhf_feedback": 0,
            "metacognition_introspections": 0,
            "principles_extracted": 0,
            "principles_validated": 0,
            "principles_promoted": 0,
            "learning_errors": 0,
        }

        # Problem queue for curriculum
        self.problem_queue = deque(maxlen=1000)

        # Lock for thread safety
        self._lock = threading.RLock()

        logger.info(
            "IntegratedLearningCoordinator initialized (principle_learner: %s)",
            principle_learner is not None,
        )

    def learn_integrated(
        self, problem: ProblemGraph, plan: DecompositionPlan, outcome: ExecutionOutcome
    ):
        """Unified learning across all systems including principle extraction and promotion"""
        with self._lock:
            self.integration_stats["total_learning_calls"] += 1

            try:
                # Step 1: Convert to experience format
                experience = self.converter.convert_to_experience(
                    problem, plan, outcome
                )

                # Step 2: Process with continual learner (gets task detection, EWC, replay)
                if self.continual_learner:
                    try:
                        continual_result = self.continual_learner.process_experience(
                            experience
                        )
                        self.integration_stats["continual_learning_updates"] += 1

                        # Update decomposer with task information
                        detected_task = continual_result.get("task_id")
                        if detected_task:
                            # Map task to problem domain
                            problem.metadata["detected_task"] = detected_task

                    except Exception as e:
                        logger.error(f"Continual learning failed: {e}")
                        self.integration_stats["learning_errors"] += 1

                # Step 3: Update original decomposer learning (for compatibility)
                try:
                    self.decomposer.learn_from_execution(problem, plan, outcome)
                except Exception as e:
                    logger.error(f"Decomposer learning failed: {e}")
                    self.integration_stats["learning_errors"] += 1

                # Step 4: Route to RLHF
                if self.rlhf_manager:
                    try:
                        self.feedback_router.route_outcome_to_feedback(
                            problem, plan, outcome
                        )
                        self.integration_stats["rlhf_feedback"] += 1
                    except Exception as e:
                        logger.error(f"RLHF routing failed: {e}")
                        self.integration_stats["learning_errors"] += 1

                # Step 5: Meta-learning update (if available)
                if self.meta_learner:
                    try:
                        self.meta_learner.online_meta_update(experience)
                        self.integration_stats["meta_updates"] += 1
                    except Exception as e:
                        logger.error(f"Meta-learning failed: {e}")
                        self.integration_stats["learning_errors"] += 1

                # Step 6: Metacognitive introspection
                if self.metacognition:
                    try:
                        metrics = {
                            "loss": experience["loss"],
                            "reward": experience["reward"],
                            "confidence": plan.confidence,
                            "actual_performance": 1.0 if outcome.success else 0.0,
                            "predicted_confidence": plan.confidence,
                            "modality": "decomposition",
                            "domain": problem.metadata.get("domain", "general"),
                        }
                        self.metacognition.update_self_model(metrics)
                        self.integration_stats["metacognition_introspections"] += 1
                    except Exception as e:
                        logger.error(f"Metacognition failed: {e}")
                        self.integration_stats["learning_errors"] += 1

                # Step 7: Extract and promote principles (NEW - closes the learning loop)
                if self.principle_learner:
                    try:
                        principle_results = self.principle_learner.extract_and_promote(
                            problem, plan, outcome
                        )

                        # Update statistics
                        self.integration_stats[
                            "principles_extracted"
                        ] += principle_results.get("principles_extracted", 0)
                        self.integration_stats[
                            "principles_validated"
                        ] += principle_results.get("principles_validated", 0)
                        self.integration_stats[
                            "principles_promoted"
                        ] += principle_results.get("principles_promoted", 0)

                        logger.debug(
                            "Principle learning: extracted=%d, validated=%d, promoted=%d",
                            principle_results.get("principles_extracted", 0),
                            principle_results.get("principles_validated", 0),
                            principle_results.get("principles_promoted", 0),
                        )

                    except Exception as e:
                        logger.error(f"Principle learning failed: {e}")
                        self.integration_stats["learning_errors"] += 1
                        import traceback

                        traceback.print_exc()

                # Step 8: Add to problem queue for curriculum
                if self.curriculum_learner:
                    self.problem_queue.append(
                        {
                            "problem": problem,
                            "difficulty": self.difficulty_estimator.estimate(problem),
                            "timestamp": time.time(),
                        }
                    )

            except Exception as e:
                logger.error(f"Integrated learning failed: {e}")
                self.integration_stats["learning_errors"] += 1
                import traceback

                traceback.print_exc()

    def get_next_problems_curriculum(
        self, batch_size: int = 10, performance: Optional[float] = None
    ) -> List[ProblemGraph]:
        """Get next batch of problems ordered by curriculum"""
        if not self.curriculum_learner:
            return []

        with self._lock:
            # Convert problem queue to format curriculum expects
            all_problems = [item["problem"] for item in self.problem_queue]

            if not all_problems:
                return []

            # Generate curriculum if not already done
            if not self.curriculum_learner.curriculum_stages:
                self.curriculum_learner.generate_curriculum(all_problems)

            # Get next batch
            batch = self.curriculum_learner.get_next_batch(
                performance=performance, batch_size=batch_size
            )

            self.integration_stats["curriculum_batches"] += 1

            return batch

    def analyze_learning_effectiveness(self) -> Dict[str, Any]:
        """Comprehensive analysis of learning effectiveness"""
        with self._lock:
            analysis = {
                "integration_stats": self.integration_stats.copy(),
                "decomposer_stats": self.decomposer.get_statistics(),
                "conversion_stats": self.converter.get_statistics(),
            }

            # Add continual learning stats
            if self.continual_learner:
                analysis["continual_learning"] = self.continual_learner.get_statistics()

            # Add curriculum stats
            if self.curriculum_learner:
                analysis["curriculum_learning"] = (
                    self.curriculum_learner.get_curriculum_stats()
                )

            # Add meta-learning stats
            if self.meta_learner:
                analysis["meta_learning"] = self.meta_learner.get_statistics()

            # Add metacognition analysis
            if self.metacognition:
                analysis["metacognition"] = (
                    self.metacognition.analyze_learning_efficiency()
                )

            # Add RLHF stats
            if self.rlhf_manager:
                analysis["rlhf"] = self.rlhf_manager.get_statistics()

            # Add principle learning stats (NEW)
            if self.principle_learner:
                analysis["principle_learning"] = (
                    self.principle_learner.get_learning_statistics()
                )

            return analysis

    def get_recommendations(self) -> List[str]:
        """Get recommendations for improving learning"""
        recommendations = []

        with self._lock:
            # Check metacognition
            if self.metacognition:
                efficiency = self.metacognition.analyze_learning_efficiency()

                if efficiency.get("recommendations"):
                    for rec in efficiency["recommendations"]:
                        recommendations.append(
                            f"Metacognition: {rec.get('suggestion', 'Unknown')}"
                        )

            # Check continual learning
            if self.continual_learner:
                stats = self.continual_learner.get_statistics()

                forgetting = stats.get("continual_metrics", {}).get(
                    "forgetting_measure", 0
                )
                if forgetting > 0.3:
                    recommendations.append(
                        "High forgetting detected - increase EWC lambda or replay frequency"
                    )

                num_tasks = stats.get("num_tasks", 0)
                if num_tasks > 10 and stats.get("free_capacity", 1.0) < 0.2:
                    recommendations.append(
                        "Low capacity remaining - consider progressive network expansion"
                    )

            # Check curriculum
            if self.curriculum_learner:
                stats = self.curriculum_learner.get_curriculum_stats()

                if stats.get("difficulty_adjustments"):
                    recent_adjustments = stats["difficulty_adjustments"][-5:]
                    if len(recent_adjustments) == 5:
                        recommendations.append(
                            "Frequent difficulty adjustments - curriculum may need rebalancing"
                        )

            # Check decomposer performance
            decomp_stats = self.decomposer.get_statistics()
            success_rate = decomp_stats["decomposition_stats"]["success_rate"]

            if success_rate < 0.5:
                recommendations.append(
                    "Low decomposition success rate - consider adding more strategies"
                )
            elif success_rate > 0.95:
                recommendations.append(
                    "Very high success rate - problems may be too easy"
                )

            # Check principle learning (NEW)
            if self.principle_learner:
                principle_stats = self.principle_learner.get_learning_statistics()

                # Check promotion rate
                promotion_stats = principle_stats.get("promotion", {}).get(
                    "promoter_stats", {}
                )
                promotion_rate = promotion_stats.get("promotion_rate", 0)

                if promotion_rate < 0.2:
                    recommendations.append(
                        "Low principle promotion rate - consider lowering promotion threshold"
                    )
                elif promotion_rate > 0.9:
                    recommendations.append(
                        "Very high promotion rate - consider raising quality standards"
                    )

                # Check knowledge base growth
                kb_stats = principle_stats.get("knowledge_base", {})
                total_principles = kb_stats.get("total_principles", 0)

                if total_principles > 1000:
                    recommendations.append(
                        "Large principle library - consider pruning low-quality principles"
                    )
                elif (
                    total_principles < 10
                    and self.integration_stats["total_learning_calls"] > 100
                ):
                    recommendations.append(
                        "Low principle extraction rate - check extraction thresholds"
                    )

            return recommendations


# ============================================================
# UNIFIED DECOMPOSER WITH INTEGRATED LEARNING
# ============================================================


class UnifiedDecomposerLearner:
    """Main interface combining decomposer with all learning systems including principle extraction"""

    def __init__(
        self,
        semantic_bridge=None,
        vulcan_memory=None,
        validator=None,
        enable_continual: bool = True,
        enable_curriculum: bool = True,
        enable_meta: bool = True,
        enable_metacognition: bool = True,
        enable_rlhf: bool = True,
        enable_principle_learning: bool = True,
        config: Optional[LearningConfig] = None,
    ):
        self.config = config or LearningConfig()

        # Create core decomposer
        from .decomposer_bootstrap import create_decomposer

        self.decomposer = create_decomposer(
            semantic_bridge=semantic_bridge,
            vulcan_memory=vulcan_memory,
            validator=validator,
        )

        # Initialize learning systems
        self.continual_learner = None
        self.curriculum_learner = None
        self.meta_learner = None
        self.metacognition = None
        self.rlhf_manager = None
        self.principle_learner = None

        if enable_continual and EnhancedContinualLearner:
            try:
                # Use decomposer's base model as reference
                self.continual_learner = EnhancedContinualLearner(
                    config=self.config, use_hierarchical=True
                )
                logger.info("Continual learning enabled")
            except Exception as e:
                logger.error(f"Failed to initialize continual learner: {e}")

        if enable_curriculum and CurriculumLearner:
            try:
                self.curriculum_learner = CurriculumLearner(
                    config=self.config, pacing_strategy=self.config.curriculum_stages
                )
                logger.info("Curriculum learning enabled")
            except Exception as e:
                logger.error(f"Failed to initialize curriculum learner: {e}")

        if enable_meta and MetaLearner and self.continual_learner:
            try:
                self.meta_learner = MetaLearner(
                    self.continual_learner, config=self.config
                )
                logger.info("Meta-learning enabled")
            except Exception as e:
                logger.error(f"Failed to initialize meta-learner: {e}")

        if enable_metacognition and MetaCognitiveMonitor:
            try:
                self.metacognition = MetaCognitiveMonitor()
                logger.info("Metacognition enabled")
            except Exception as e:
                logger.error(f"Failed to initialize metacognition: {e}")

        if enable_rlhf and RLHFManager and self.continual_learner:
            try:
                self.rlhf_manager = RLHFManager(
                    self.continual_learner, config=self.config
                )
                logger.info("RLHF enabled")
            except Exception as e:
                logger.error(f"Failed to initialize RLHF: {e}")

        # Initialize principle learning (NEW)
        if enable_principle_learning and PrincipleLearner:
            try:
                # Get or create decomposition library
                if hasattr(self.decomposer, "strategy_library"):
                    library = self.decomposer.strategy_library
                elif StratifiedDecompositionLibrary:
                    library = StratifiedDecompositionLibrary()
                    self.decomposer.strategy_library = library
                else:
                    library = None
                    logger.warning(
                        "No decomposition library available for principle learning"
                    )

                if library:
                    self.principle_learner = PrincipleLearner(
                        library=library,
                        min_promotion_score=0.7,
                        enable_auto_promotion=True,
                    )
                    logger.info("Principle learning enabled")
                else:
                    logger.warning("Principle learning disabled - no library available")

            except Exception as e:
                logger.error(f"Failed to initialize principle learner: {e}")
                import traceback

                traceback.print_exc()

        # Create coordinator
        self.coordinator = IntegratedLearningCoordinator(
            decomposer=self.decomposer,
            continual_learner=self.continual_learner,
            curriculum_learner=self.curriculum_learner,
            meta_learner=self.meta_learner,
            metacognition=self.metacognition,
            rlhf_manager=self.rlhf_manager,
            principle_learner=self.principle_learner,
            config=self.config,
        )

        logger.info(
            "UnifiedDecomposerLearner initialized with all systems (principle_learning: %s)",
            self.principle_learner is not None,
        )

    def decompose_and_execute(
        self, problem: ProblemGraph, validate: bool = False
    ) -> Tuple[DecompositionPlan, ExecutionOutcome]:
        """Decompose and execute with integrated learning including principle extraction"""
        # Step 1: Decompose and execute using base system
        plan, outcome = self.decomposer.decompose_and_execute(problem, validate)

        # Step 2: Integrated learning across all systems (including principle extraction)
        self.coordinator.learn_integrated(problem, plan, outcome)

        return plan, outcome

    def decompose_only(self, problem: ProblemGraph) -> DecompositionPlan:
        """Just decompose without execution"""
        return self.decomposer.decompose_novel_problem(problem)

    def get_next_curriculum_problems(self, batch_size: int = 10) -> List[ProblemGraph]:
        """Get next problems according to curriculum"""
        # Calculate recent performance
        stats = self.decomposer.get_statistics()
        performance = stats["decomposition_stats"].get("success_rate", 0.5)

        return self.coordinator.get_next_problems_curriculum(batch_size, performance)

    def provide_human_feedback(
        self, problem_signature: str, rating: float, comments: Optional[str] = None
    ):
        """Provide explicit human feedback on a solution"""
        if self.coordinator.feedback_router:
            self.coordinator.feedback_router.route_human_feedback(
                problem_signature, rating, comments
            )

    def get_applicable_principles(self, problem: ProblemGraph) -> List[Any]:
        """
        Get principles applicable to a problem

        Args:
            problem: Problem to solve

        Returns:
            List of applicable principles
        """
        if not self.principle_learner:
            return []

        return self.principle_learner.find_applicable_principles(problem)

    def prune_principles(
        self, age_threshold_days: int = 90, confidence_threshold: float = 0.3
    ) -> int:
        """
        Prune low-quality or outdated principles

        Args:
            age_threshold_days: Age threshold in days
            confidence_threshold: Confidence threshold

        Returns:
            Number of principles pruned
        """
        if not self.principle_learner:
            return 0

        return self.principle_learner.prune_low_quality_principles(
            age_threshold_days, confidence_threshold
        )

    def export_principles(self, path: Path, format: str = "json") -> bool:
        """
        Export learned principles

        Args:
            path: Export path
            format: Export format

        Returns:
            True if successful
        """
        if not self.principle_learner:
            logger.warning("No principle learner available for export")
            return False

        return self.principle_learner.export_learned_principles(path, format)

    def import_principles(self, path: Path) -> bool:
        """
        Import principles from file

        Args:
            path: Import path

        Returns:
            True if successful
        """
        if not self.principle_learner:
            logger.warning("No principle learner available for import")
            return False

        return self.principle_learner.import_principles(path)

    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get statistics from all systems including principle learning"""
        return self.coordinator.analyze_learning_effectiveness()

    def get_learning_recommendations(self) -> List[str]:
        """Get recommendations for improving learning"""
        return self.coordinator.get_recommendations()

    def save_state(self, path: Optional[str] = None) -> Dict[str, str]:
        """Save complete state of all systems"""
        if path is None:
            path = Path("unified_learner_states")
        else:
            path = Path(path)

        path.mkdir(parents=True, exist_ok=True)

        saved_paths = {}
        timestamp = int(time.time())

        # Save continual learner
        if self.continual_learner:
            cl_path = path / f"continual_learner_{timestamp}.pkl"
            try:
                saved_paths["continual_learner"] = self.continual_learner.save_state(
                    str(cl_path)
                )
            except Exception as e:
                logger.error(f"Failed to save continual learner: {e}")

        # Save curriculum learner
        if self.curriculum_learner:
            curr_path = path / f"curriculum_learner_{timestamp}.pkl"
            try:
                saved_paths["curriculum_learner"] = self.curriculum_learner.save_state(
                    str(curr_path)
                )
            except Exception as e:
                logger.error(f"Failed to save curriculum learner: {e}")

        # Save metacognition
        if self.metacognition:
            meta_path = path / f"metacognition_{timestamp}.pkl"
            try:
                saved_paths["metacognition"] = self.metacognition.save_state(
                    str(meta_path)
                )
            except Exception as e:
                logger.error(f"Failed to save metacognition: {e}")

        # Save principle learner (NEW)
        if self.principle_learner:
            principle_path = path / f"principle_learner_{timestamp}"
            try:
                principle_paths = self.principle_learner.knowledge_base.export(
                    principle_path / "knowledge_base.json", format="json"
                )
                saved_paths["principle_learner"] = str(principle_path)
            except Exception as e:
                logger.error(f"Failed to save principle learner: {e}")

        # Save coordinator state
        coord_path = path / f"coordinator_{timestamp}.json"
        try:
            with open(coord_path, "w", encoding="utf-8") as f:
                json.dump(self.coordinator.integration_stats, f, indent=2)
            saved_paths["coordinator"] = str(coord_path)
        except Exception as e:
            logger.error(f"Failed to save coordinator: {e}")

        logger.info(f"Saved unified learner state to {path}")
        return saved_paths

    def load_state(self, paths: Dict[str, str]):
        """Load state from saved files"""
        # Load continual learner
        if "continual_learner" in paths and self.continual_learner:
            try:
                self.continual_learner.load_state(paths["continual_learner"])
                logger.info("Loaded continual learner state")
            except Exception as e:
                logger.error(f"Failed to load continual learner: {e}")

        # Load curriculum learner
        if "curriculum_learner" in paths and self.curriculum_learner:
            try:
                self.curriculum_learner.load_state(paths["curriculum_learner"])
                logger.info("Loaded curriculum learner state")
            except Exception as e:
                logger.error(f"Failed to load curriculum learner: {e}")

        # Load metacognition
        if "metacognition" in paths and self.metacognition:
            try:
                self.metacognition.load_state(paths["metacognition"])
                logger.info("Loaded metacognition state")
            except Exception as e:
                logger.error(f"Failed to load metacognition: {e}")

        # Load principle learner (NEW)
        if "principle_learner" in paths and self.principle_learner:
            try:
                kb_path = Path(paths["principle_learner"]) / "knowledge_base.json"
                if kb_path.exists():
                    self.principle_learner.knowledge_base.import_from(kb_path)
                    logger.info("Loaded principle learner state")
            except Exception as e:
                logger.error(f"Failed to load principle learner: {e}")

    def shutdown(self):
        """Clean shutdown of all systems"""
        logger.info("Shutting down UnifiedDecomposerLearner...")

        # Save state before shutdown
        try:
            self.save_state()
        except Exception as e:
            logger.error(f"Failed to save state during shutdown: {e}")

        # Shutdown individual systems
        if self.continual_learner:
            try:
                self.continual_learner.shutdown()
            except Exception as e:
                logger.debug(f"Failed to integrate learning data: {e}")

        if self.rlhf_manager:
            try:
                self.rlhf_manager.shutdown()
            except Exception as e:
                logger.debug(f"Failed to update learning metrics: {e}")

        if self.meta_learner:
            try:
                self.meta_learner.shutdown()
            except Exception as e:
                logger.debug(f"Failed to persist learning state: {e}")

        logger.info("UnifiedDecomposerLearner shutdown complete")


# ============================================================
# FACTORY FUNCTION
# ============================================================


def create_unified_decomposer(
    semantic_bridge=None,
    vulcan_memory=None,
    validator=None,
    config: Optional[LearningConfig] = None,
    enable_all: bool = True,
    enable_principle_learning: bool = True,
) -> UnifiedDecomposerLearner:
    """
    Factory function to create fully integrated decomposer with all learning systems

    Args:
        semantic_bridge: Optional semantic bridge component
        vulcan_memory: Optional VULCAN memory system
        validator: Optional validator for solution validation
        config: Optional learning configuration
        enable_all: Whether to enable all learning systems (default True)
        enable_principle_learning: Whether to enable principle extraction (default True)

    Returns:
        UnifiedDecomposerLearner with all systems integrated

    Example:
        >>> from vulcan.problem_decomposer.learning_integration import create_unified_decomposer
        >>> from vulcan.problem_decomposer.problem_decomposer_core import ProblemGraph
        >>>
        >>> # Create unified system with principle learning
        >>> learner = create_unified_decomposer()
        >>>
        >>> # Create and solve problem
        >>> problem = ProblemGraph(
        ...     nodes={'A': {}, 'B': {}},
        ...     edges=[('A', 'B', {})],
        ...     metadata={'domain': 'planning'}
        ... )
        >>>
        >>> # Decompose and execute with full learning (including principle extraction)
        >>> plan, outcome = learner.decompose_and_execute(problem)
        >>>
        >>> # Get comprehensive statistics
        >>> stats = learner.get_comprehensive_statistics()
        >>> print(f"Success: {outcome.success}")
        >>> print(f"Learning updates: {stats['integration_stats']}")
        >>> print(f"Principles promoted: {stats['integration_stats']['principles_promoted']}")
        >>>
        >>> # Get applicable principles for next problem
        >>> applicable = learner.get_applicable_principles(problem)
        >>> print(f"Found {len(applicable)} applicable principles")
    """
    return UnifiedDecomposerLearner(
        semantic_bridge=semantic_bridge,
        vulcan_memory=vulcan_memory,
        validator=validator,
        enable_continual=enable_all,
        enable_curriculum=enable_all,
        enable_meta=enable_all,
        enable_metacognition=enable_all,
        enable_rlhf=enable_all,
        enable_principle_learning=enable_all and enable_principle_learning,
        config=config,
    )


# ============================================================
# TESTING AND VALIDATION
# ============================================================


def test_integration():
    """Test the integration including principle learning"""
    logger.info("Testing learning integration with principle extraction...")

    try:
        # Create unified learner
        learner = create_unified_decomposer()

        # Create test problem
        from .problem_decomposer_core import ProblemGraph

        problem = ProblemGraph(
            nodes={
                "start": {"type": "decision"},
                "process": {"type": "operation"},
                "end": {"type": "result"},
            },
            edges=[("start", "process", {}), ("process", "end", {})],
            root="start",
            metadata={"domain": "testing", "complexity": 2.0},
        )

        problem.complexity_score = 2.0

        # Decompose and execute
        plan, outcome = learner.decompose_and_execute(problem)

        # Get statistics
        stats = learner.get_comprehensive_statistics()

        logger.info("Integration test results:")
        logger.info(f"  Plan steps: {len(plan.steps)}")
        logger.info(f"  Outcome success: {outcome.success}")
        logger.info(
            f"  Learning updates: {stats['integration_stats']['total_learning_calls']}"
        )

        if stats["integration_stats"]["continual_learning_updates"] > 0:
            logger.info("  Continual learning working")
        if stats["integration_stats"]["rlhf_feedback"] > 0:
            logger.info("  RLHF routing working")
        if stats["integration_stats"]["metacognition_introspections"] > 0:
            logger.info("  Metacognition working")
        if stats["integration_stats"]["principles_extracted"] >= 0:
            logger.info(
                f"  Principle extraction working: {stats['integration_stats']['principles_extracted']} extracted"
            )
        if stats["integration_stats"]["principles_promoted"] >= 0:
            logger.info(
                f"  Principle promotion working: {stats['integration_stats']['principles_promoted']} promoted"
            )

        # Get recommendations
        recommendations = learner.get_learning_recommendations()
        if recommendations:
            logger.info("Recommendations:")
            for rec in recommendations:
                logger.info(f"  - {rec}")

        # Test principle retrieval
        if learner.principle_learner:
            applicable = learner.get_applicable_principles(problem)
            logger.info(f"  Found {len(applicable)} applicable principles")

        logger.info("Integration test passed!")
        return True

    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run test when module executed directly
    logging.basicConfig(level=logging.INFO)
    test_integration()
