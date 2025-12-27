"""
VULCAN-AGI Learning Module - Unified Integration
Coordinates: Continual + Curriculum + Meta + RLHF + World Model + Metacognition
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# =============================================================================
# Learning Weight Adjustment Constants
# =============================================================================
# These constants control how much tool weights are adjusted based on outcomes.
# Positive adjustment for success encourages using successful tools.
# Smaller negative adjustment for failure prevents over-penalizing tools.
WEIGHT_ADJUSTMENT_SUCCESS = 0.01
WEIGHT_ADJUSTMENT_FAILURE = -0.005

# Make torch import conditional to allow module import even without torch
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    logger.warning("torch not available - continual learning will be disabled")

# List of torch-dependent components
TORCH_DEPENDENT_COMPONENTS = [
    "ContinualLearner",
    "EnhancedContinualLearner",
    "MetaLearner",
    "MetaLearningAlgorithm",
    "TaskDetector",
    "CompositionalUnderstanding",
    "MetaCognitiveMonitor",
    "LiveFeedbackProcessor",
    "RLHFManager",
]

# Import torch-free components
from .curriculum_learning import (
    CurriculumLearner,
    LearnedDifficultyEstimator,
    PacingStrategy,
)
from .learning_types import FeedbackData, LearningConfig, LearningMode, TaskInfo
from .parameter_history import ParameterHistoryManager

# Import torch-dependent components conditionally
if TORCH_AVAILABLE:
    try:
        from .continual_learning import ContinualLearner, EnhancedContinualLearner
        from .meta_learning import MetaLearner, MetaLearningAlgorithm, TaskDetector
        from .metacognition import CompositionalUnderstanding, MetaCognitiveMonitor
        from .rlhf_feedback import LiveFeedbackProcessor, RLHFManager
    except Exception as e:
        logger.error(f"Failed to import torch-dependent learning components: {e}")
        # Set all to None if import fails
        for component in TORCH_DEPENDENT_COMPONENTS:
            globals()[component] = None
else:
    # Set all to None if torch not available
    for component in TORCH_DEPENDENT_COMPONENTS:
        globals()[component] = None

# Import world model (no torch dependency)
try:
    from .world_model import PlanningAlgorithm, UnifiedWorldModel
except Exception as e:
    logger.error(f"Failed to import world model: {e}")
    PlanningAlgorithm = None
    UnifiedWorldModel = None


class UnifiedLearningSystem:
    """
    Production-ready unified learning system integrating all components:
    - Continual Learning (EWC, replay, progressive networks)
    - Curriculum Learning (adaptive difficulty)
    - Meta-Learning (MAML, task adaptation)
    - RLHF (human feedback, PPO)
    - World Model (planning, dynamics prediction)
    - Metacognition (self-monitoring, improvement)
    """

    def __init__(
        self,
        config: Optional[LearningConfig] = None,
        embedding_dim: int = 384,
        enable_world_model: bool = True,
        enable_curriculum: bool = True,
        enable_metacognition: bool = True,
    ):
        self.config = config or LearningConfig()
        self.embedding_dim = embedding_dim

        logger.info("Initializing UnifiedLearningSystem...")

        # Core continual learner (only if torch is available)
        if TORCH_AVAILABLE and EnhancedContinualLearner:
            self.continual_learner = EnhancedContinualLearner(
                embedding_dim=embedding_dim,
                config=self.config,
                use_hierarchical=True,
                use_progressive=True,
            )
        else:
            logger.warning(
                "EnhancedContinualLearner not available (torch not installed or import failed)"
            )
            self.continual_learner = None

        # CRITICAL: Connect metacognitive monitor AFTER continual learner is initialized
        if enable_metacognition and TORCH_AVAILABLE and MetaCognitiveMonitor:
            self._connect_metacognition()
        else:
            self.meta_monitor = None

        # Initialize curriculum attributes first (FIXED)
        self.curriculum_active = False
        self.current_curriculum = []

        # Curriculum learner with integrated difficulty estimator
        self.curriculum_learner = None
        if enable_curriculum:
            self.curriculum_learner = CurriculumLearner(
                difficulty_estimator=self._create_integrated_difficulty_estimator(),
                config=self.config,
                pacing_strategy=PacingStrategy.ADAPTIVE,
            )

        # World model for planning (optional but powerful)
        self.world_model = None
        if enable_world_model:
            self.world_model = UnifiedWorldModel(
                state_dim=embedding_dim, ensemble_size=3, use_attention=True
            )
            logger.info("World model initialized with MCTS/CEM/MPPI planning")

        # Event loop for async operations (RLHF API calls, monitoring)
        self._loop = None
        self._loop_thread = None
        self._loop_ready = threading.Event()
        self._start_event_loop()

        # System state
        self.learning_mode = LearningMode.CONTINUAL
        self.stats_history = []
        self._shutdown_requested = threading.Event()
        self._shutdown_event = threading.Event()  # ADDED: For external threads
        
        # Learning from outcomes
        self.slow_routing_threshold_ms = 5000
        self.tool_weight_adjustments: Dict[str, float] = {}
        
        # MetaLearner reference (if available)
        self.meta_learner = None
        if self.continual_learner and hasattr(self.continual_learner, 'meta_learner'):
            self.meta_learner = self.continual_learner.meta_learner

        logger.info(
            f"UnifiedLearningSystem initialized with {self._count_components()} active components"
        )

    def _connect_metacognition(self):
        """CRITICAL: Connect metacognitive monitor to actual model and optimizer"""
        if not hasattr(self.continual_learner, "meta_cognitive"):
            from .metacognition import MetaCognitiveMonitor

            self.continual_learner.meta_cognitive = MetaCognitiveMonitor()

        # Connect to general model and optimizer
        if "general" in self.continual_learner.optimizers:
            self.continual_learner.meta_cognitive.set_model_optimizer(
                self.continual_learner,  # Pass the entire continual learner
                self.continual_learner.optimizers["general"],
            )
            logger.info("Metacognitive monitor connected to model and optimizer")
        else:
            logger.warning(
                "Could not connect metacognitive monitor - no general optimizer found"
            )

    async def process_outcome(self, outcome: Dict[str, Any]) -> None:
        """
        Main entry point for learning from query outcomes.
        
        This method is called by OutcomeBridge when a query outcome is recorded.
        It coordinates learning across all subsystems: ContinualLearner, MetaLearner,
        and tool weight adjustments.
        
        Args:
            outcome: Dictionary containing query outcome data:
                - query_id: Unique query identifier
                - status: Query status ("success", "error", "timeout")
                - routing_ms: Time spent routing
                - total_ms: Total processing time
                - complexity: Query complexity score
                - query_type: Type of query
                - tools: List of tools used
                - timestamp: Time of the outcome
        """
        query_id = outcome.get('query_id', 'unknown')
        routing_ms = outcome.get('routing_ms', 0)
        status = outcome.get('status', 'unknown')
        tools = outcome.get('tools', [])
        query_type = outcome.get('query_type', 'unknown')
        
        logger.info(f"[Learning] Processing outcome: {query_id}, type={query_type}, tools={tools}")
        
        # 1. Feed to ContinualLearner if available
        if self.continual_learner:
            try:
                # Use the basic ContinualLearner's learn_from_outcome if available
                if hasattr(self.continual_learner, 'learn_from_outcome'):
                    await self.continual_learner.learn_from_outcome(outcome)
            except Exception as e:
                logger.error(f"[Learning] ContinualLearner error: {e}")
        
        # 2. Detect and log slow routing
        if routing_ms > self.slow_routing_threshold_ms:
            logger.warning(f"[Learning] SLOW ROUTING DETECTED: {routing_ms}ms (threshold: {self.slow_routing_threshold_ms}ms)")
            logger.warning(f"[Learning] Slow query details: type={query_type}, tools={tools}")
            
            # Feed to MetaLearner for strategy adjustment
            if self.meta_learner:
                try:
                    if hasattr(self.meta_learner, 'record_slow_routing'):
                        await self.meta_learner.record_slow_routing(query_type, tools, routing_ms)
                except Exception as e:
                    logger.error(f"[Learning] MetaLearner slow routing error: {e}")
        
        # 3. Update tool weights based on success/failure
        weight_delta = WEIGHT_ADJUSTMENT_SUCCESS if status == 'success' else WEIGHT_ADJUSTMENT_FAILURE
        for tool in tools:
            if tool not in self.tool_weight_adjustments:
                self.tool_weight_adjustments[tool] = 0.0
            self.tool_weight_adjustments[tool] += weight_delta
            logger.info(f"[Learning] Tool '{tool}' weight adjustment: {weight_delta:+.3f} (cumulative: {self.tool_weight_adjustments[tool]:+.3f})")
        
        # 4. Feed to MetaLearner for pattern detection
        if self.meta_learner:
            try:
                if hasattr(self.meta_learner, 'update_from_outcome'):
                    await self.meta_learner.update_from_outcome(outcome)
                    logger.info(f"[MetaLearner] Processed outcome {query_id}")
            except Exception as e:
                logger.error(f"[Learning] MetaLearner error: {e}")
        
        logger.info(f"[Learning] Outcome processing complete for {query_id}")

    def get_tool_weight_adjustment(self, tool: str) -> float:
        """
        Get cumulative weight adjustment for a tool.
        
        This is used by ToolSelector to apply learned weight adjustments
        to tool selection probabilities.
        
        Args:
            tool: Tool name
            
        Returns:
            Cumulative weight adjustment (positive = more successful, negative = less successful)
        """
        return self.tool_weight_adjustments.get(tool, 0.0)

    def _create_integrated_difficulty_estimator(self):
        """Create difficulty estimator that uses continual learner's knowledge"""

        class IntegratedDifficultyEstimator(LearnedDifficultyEstimator):
            def __init__(self, continual_learner):
                super().__init__()
                self.continual_learner = continual_learner

            def estimate(self, task: Any) -> float:
                # Try to use continual learner's task detection
                if isinstance(task, dict):
                    task_id = self.continual_learner.task_detector.detect_task(task)

                    # Use historical performance as difficulty indicator
                    if task_id in self.continual_learner.task_info:
                        info = self.continual_learner.task_info[task_id]
                        # Inverse of performance = difficulty
                        difficulty = 1.0 - info.performance

                        # Factor in task's own difficulty if available
                        if hasattr(info, "difficulty"):
                            difficulty = max(difficulty, info.difficulty)

                        return float(np.clip(difficulty, 0, 1))

                # Fallback to parent estimator
                return super().estimate(task)

            def update(self, task: Any, performance: float):
                """Update both estimators"""
                super().update(task, performance)

                # Also update continual learner's task info
                if isinstance(task, dict):
                    task_id = self.continual_learner.task_detector.detect_task(task)
                    if task_id in self.continual_learner.task_info:
                        self.continual_learner.task_info[task_id].difficulty = (
                            1.0 - performance
                        )

        return IntegratedDifficultyEstimator(self.continual_learner)

    def _start_event_loop(self):
        """FIXED: Start event loop with proper checks for existing loops"""

        def run_loop():
            try:
                # CRITICAL FIX: Check if there's already a running loop
                try:
                    existing_loop = asyncio.get_running_loop()
                    logger.info("Using existing event loop")
                    self._loop = existing_loop
                    self._loop_ready.set()
                    return  # Use existing loop, don't create new one
                except RuntimeError:
                    # No running loop, safe to create one
                    pass

                # Create new event loop
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)

                # Create monitoring tasks if available
                if hasattr(self.continual_learner, "live_feedback"):
                    self._loop.create_task(
                        self.continual_learner.live_feedback.start_monitoring()
                    )

                # Signal that loop is ready
                self._loop_ready.set()

                # Run the loop until stopped
                self._loop.run_forever()

            except Exception as e:
                logger.error(f"Event loop error: {e}")
                self._loop_ready.set()  # Set anyway to prevent deadlock
            finally:
                # Cleanup
                if self._loop:
                    try:
                        self._loop.close()
                    except Exception as e:
                        logger.debug(
                            f"{self.__class__.__name__ if hasattr(self, '__class__') else 'Operation'} error: {e}"
                        )

        # Check if we're already in an async context
        try:
            asyncio.get_running_loop()
            logger.info("Already in async context, using existing loop")
            self._loop_ready.set()
            return  # Don't start thread if we're in async context
        except RuntimeError as e:
            logger.debug(f"Operation failed: {e}")

        # Start loop in background thread
        self._loop_thread = threading.Thread(target=run_loop, daemon=True)
        self._loop_thread.start()

        # Wait for loop to be ready with timeout
        if not self._loop_ready.wait(timeout=5):
            logger.error("Event loop failed to start within 5 seconds")
        else:
            logger.info("Background event loop started for async operations")

    def start_curriculum(self, all_tasks: List[Any], auto_cluster: bool = True):
        """Start curriculum-based learning"""
        if not self.curriculum_learner:
            logger.warning("Curriculum learning not enabled")
            return

        logger.info(f"Starting curriculum with {len(all_tasks)} tasks")
        self.current_curriculum = self.curriculum_learner.generate_curriculum(
            all_tasks, auto_cluster=auto_cluster
        )
        self.curriculum_active = True
        self.learning_mode = LearningMode.CURRICULUM

        logger.info(f"Generated curriculum with {len(self.current_curriculum)} stages")

    def process_experience(self, experience: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point: Process experience through unified system
        Coordinates: continual + curriculum + meta + RLHF + world model + metacognition
        """

        # Add curriculum context if active
        if (
            hasattr(self, "curriculum_active")
            and self.curriculum_active
            and self.curriculum_learner
        ):
            current_task = self.continual_learner.task_detector.current_task
            performance = None

            if current_task and current_task in self.continual_learner.task_info:
                performance = self.continual_learner.task_info[current_task].performance

            # Get next curriculum batch
            curriculum_batch = self.curriculum_learner.get_next_batch(
                performance=performance, batch_size=1
            )

            if curriculum_batch:
                experience["curriculum_stage"] = self.curriculum_learner.current_stage
                experience["curriculum_batch"] = (
                    curriculum_batch[0] if curriculum_batch else None
                )

        # World model: Use for planning if enabled (FIXED: Better None handling)
        if self.world_model and "embedding" in experience:
            embedding = experience["embedding"]

            # Handle None embedding gracefully
            if embedding is not None:
                if isinstance(embedding, np.ndarray):
                    embedding = torch.tensor(embedding, dtype=torch.float32)
                elif not isinstance(embedding, torch.Tensor):
                    # Try to convert to tensor
                    try:
                        embedding = torch.tensor(embedding, dtype=torch.float32)
                    except Exception:
                        embedding = None

                if embedding is not None:
                    # Update world model with transition
                    if "action" in experience and "reward" in experience:
                        self.world_model.update_state(
                            embedding,
                            experience.get("action"),
                            experience.get("reward", 0.0),
                            experience.get("next_embedding", embedding),
                        )

                    # Add world model predictions to experience
                    try:
                        if embedding.dim() == 0:  # Scalar
                            embedding = embedding.unsqueeze(0)
                        elif embedding.dim() == 1:  # 1D tensor
                            embedding = embedding.unsqueeze(0)

                        experience["world_model_value"] = (
                            self.world_model.predict_value(embedding).item()
                        )
                    except Exception as e:
                        logger.debug(f"Could not compute world model value: {e}")
                        experience["world_model_value"] = 0.0

        # Process through continual learner (handles EWC, replay, task detection, RLHF, meta-learning)
        result = self.continual_learner.process_experience(experience)

        # Metacognitive monitoring
        if hasattr(self.continual_learner, "meta_cognitive"):
            # Update self-model with performance
            self.continual_learner.meta_cognitive.update_self_model(
                {
                    "loss": result.get("loss", 0),
                    "modality": experience.get("modality"),
                    "predicted_confidence": result.get("confidence", 0.5),
                    "actual_performance": 1.0 / (1.0 + result.get("loss", 1.0)),
                }
            )

            # Analyze learning efficiency periodically
            if self.continual_learner.consolidation_counter % 50 == 0:
                efficiency_analysis = (
                    self.continual_learner.meta_cognitive.analyze_learning_efficiency()
                )
                result["metacognitive_analysis"] = efficiency_analysis

        # Add unified metadata
        result["learning_mode"] = self.learning_mode.value
        result["curriculum_active"] = getattr(self, "curriculum_active", False)
        result["world_model_enabled"] = self.world_model is not None

        # Store in stats history
        self.stats_history.append(
            {
                "timestamp": time.time(),
                "result": result,
                "experience_type": experience.get("type", "unknown"),
            }
        )

        return result

    def plan_with_world_model(
        self,
        current_state: torch.Tensor,
        candidate_actions: List[torch.Tensor],
        algorithm: PlanningAlgorithm = PlanningAlgorithm.MCTS,
        horizon: int = 5,
    ) -> Tuple[torch.Tensor, Dict]:
        """Use world model for planning (if enabled)"""
        if not self.world_model:
            raise RuntimeError("World model not enabled. Set enable_world_model=True")

        return self.world_model.plan_actions(
            current_state, candidate_actions, horizon=horizon, algorithm=algorithm
        )

    def train_world_model(self, num_steps: int = 100):
        """Train world model on collected transitions"""
        if not self.world_model:
            logger.warning("World model not enabled")
            return

        if len(self.world_model.transition_buffer) < 32:
            logger.warning("Insufficient transitions for world model training")
            return

        logger.info(f"Training world model for {num_steps} steps...")

        for step in range(num_steps):
            # Sample batch from transition buffer
            batch_size = min(32, len(self.world_model.transition_buffer))
            indices = np.random.choice(
                len(self.world_model.transition_buffer), batch_size
            )

            batch = {"state": [], "action": [], "next_state": [], "reward": []}

            for idx in indices:
                transition = self.world_model.transition_buffer[idx]
                for key in batch:
                    value = transition.get(key)
                    if value is not None:
                        if isinstance(value, np.ndarray):
                            value = torch.tensor(value, dtype=torch.float32)
                        elif not isinstance(value, torch.Tensor):
                            value = torch.tensor(value, dtype=torch.float32)
                        batch[key].append(value)

            # Stack tensors
            for key in batch:
                if batch[key]:
                    batch[key] = torch.stack(batch[key])

            # Train step
            if all(len(batch[k]) > 0 for k in batch):
                losses = self.world_model.train_step(batch)

                if step % 20 == 0:
                    logger.info(f"Step {step}: {losses}")

    def get_unified_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all subsystems"""
        stats = {
            "timestamp": time.time(),
            "learning_mode": self.learning_mode.value,
            "continual": self.continual_learner.get_statistics(),
        }

        # Curriculum stats
        if self.curriculum_learner and getattr(self, "curriculum_active", False):
            stats["curriculum"] = self.curriculum_learner.get_curriculum_stats()

        # Meta-learning stats
        if hasattr(self.continual_learner, "meta_learner"):
            stats["meta_learning"] = (
                self.continual_learner.meta_learner.get_statistics()
            )

        # Metacognition stats
        if hasattr(self.continual_learner, "meta_cognitive"):
            stats["metacognition"] = {
                "self_model": self.continual_learner.meta_cognitive.self_model,
                "improvements_applied": len(
                    self.continual_learner.meta_cognitive.applied_improvements
                ),
                "improvement_audit": self.continual_learner.meta_cognitive.get_improvement_history()[
                    -10:
                ],
            }

        # RLHF stats
        if (
            hasattr(self.continual_learner, "rlhf_manager")
            and self.continual_learner.rlhf_manager
        ):
            stats["rlhf"] = self.continual_learner.rlhf_manager.get_statistics()

        # World model stats
        if self.world_model:
            stats["world_model"] = self.world_model.get_training_stats()

        # Integration metrics
        stats["integration"] = self._compute_integration_metrics()

        return stats

    def _compute_integration_metrics(self) -> Dict[str, Any]:
        """Measure how well subsystems are integrated"""
        metrics = {
            "components_active": self._count_components(),
            "curriculum_continual_alignment": 0.0,
            "meta_learning_effectiveness": 0.0,
            "overall_coherence": 0.0,
        }

        # Curriculum-continual alignment
        if (
            getattr(self, "curriculum_active", False)
            and self.curriculum_learner
            and self.continual_learner.task_info
        ):
            curr_progress = self.curriculum_learner.current_stage / max(
                1, len(self.current_curriculum)
            )
            avg_task_perf = np.mean(
                [info.performance for info in self.continual_learner.task_info.values()]
            )
            metrics["curriculum_continual_alignment"] = 1.0 - abs(
                curr_progress - avg_task_perf
            )

        # Meta-learning effectiveness
        if hasattr(self.continual_learner, "meta_learner"):
            meta_stats = self.continual_learner.meta_learner.get_statistics()
            if meta_stats.get("num_adaptations", 0) > 0:
                avg_loss = meta_stats.get("avg_task_loss", 1.0)
                metrics["meta_learning_effectiveness"] = 1.0 / (1.0 + avg_loss)

        # Overall coherence
        metrics["overall_coherence"] = (
            metrics["curriculum_continual_alignment"]
            + metrics["meta_learning_effectiveness"]
        ) / 2

        return metrics

    def _count_components(self) -> int:
        """Count active components"""
        count = 1  # Continual learner always active

        if self.curriculum_learner and getattr(self, "curriculum_active", False):
            count += 1
        if self.world_model:
            count += 1
        if hasattr(self.continual_learner, "meta_cognitive"):
            count += 1
        if (
            hasattr(self.continual_learner, "rlhf_manager")
            and self.continual_learner.rlhf_manager
        ):
            count += 1

        return count

    def save_complete_state(self, base_path: str = "unified_learning_state"):
        """Save complete state of all subsystems"""
        save_dir = Path(base_path) / f"state_{int(time.time())}"
        save_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving unified learning state to {save_dir}...")

        # Save continual learner
        continual_path = self.continual_learner.save_state(
            str(save_dir / "continual_state.pkl")
        )

        # Save curriculum if active
        if self.curriculum_learner and getattr(self, "curriculum_active", False):
            curriculum_path = self.curriculum_learner.save_state(
                str(save_dir / "curriculum_state.pkl")
            )

        # Save world model
        if self.world_model:
            self.world_model.save_model(str(save_dir / "world_model.pt"))

        # Save metacognition
        if hasattr(self.continual_learner, "meta_cognitive"):
            self.continual_learner.meta_cognitive.save_state(
                str(save_dir / "metacog_state.pkl")
            )

        # Save unified stats
        with open(save_dir / "unified_stats.json", "w", encoding="utf-8") as f:
            import json

            json.dump(self.get_unified_stats(), f, indent=2, default=str)

        logger.info(f"Complete state saved to: {save_dir}")
        return str(save_dir)

    def shutdown(self, timeout: float = 15.0):
        """FIXED: Comprehensive shutdown with proper ordering and timeout protection"""
        if self._shutdown_requested.is_set():
            logger.info("Already shut down")
            return

        logger.info("=" * 60)
        logger.info("Shutting down UnifiedLearningSystem...")
        logger.info("=" * 60)

        # Signal shutdown immediately to stop accepting new work
        self._shutdown_requested.set()
        self._shutdown_event.set()  # ADDED: Signal external threads

        # Step 1: Stop curriculum to prevent new task assignments
        if self.curriculum_learner:
            self.curriculum_active = False
            logger.info("Stopped curriculum learning")

        # Step 2: Stop event loop BEFORE shutting down components
        # This prevents new async operations from starting
        if self._loop and self._loop.is_running():
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
                logger.info("Stopping event loop...")

                # Wait for loop to stop
                if self._loop_thread and self._loop_thread.is_alive():
                    self._loop_thread.join(timeout=2)

                logger.info("Event loop stopped")
            except Exception as e:
                logger.error(f"Failed to stop event loop gracefully: {e}")

        # Step 3: Shutdown components in dependency order with timeout
        # RLHF first (might have pending API calls), then others
        components = []

        # Add RLHF manager if present (shutdown first - has API connections)
        if (
            hasattr(self.continual_learner, "rlhf_manager")
            and self.continual_learner.rlhf_manager
        ):
            components.append(("rlhf_manager", self.continual_learner.rlhf_manager))

        # World model param history (has background thread)
        if self.world_model and hasattr(self.world_model, "param_history"):
            components.append(
                ("world_model.param_history", self.world_model.param_history)
            )

        # Continual learner last (coordinates everything)
        components.append(("continual_learner", self.continual_learner))

        # Shutdown with thread pool for timeout control (cross-platform)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(components)
        ) as executor:
            futures = {}

            for name, component in components:
                if component and hasattr(component, "shutdown"):
                    future = executor.submit(
                        self._safe_component_shutdown, component, name
                    )
                    futures[name] = future
                    logger.info(f"Submitted shutdown for: {name}")

            # Wait with timeout
            done, not_done = concurrent.futures.wait(
                futures.values(),
                timeout=timeout,
                return_when=concurrent.futures.ALL_COMPLETED,
            )

            # Process results
            for name, future in futures.items():
                if future in not_done:
                    logger.error(
                        f"Component '{name}' shutdown timed out after {timeout}s"
                    )
                    future.cancel()
                elif future.done():
                    try:
                        success = future.result()
                        if success:
                            logger.info(f"✔ Component '{name}' shut down successfully")
                        else:
                            logger.warning(
                                f"⚠ Component '{name}' shutdown completed with warnings"
                            )
                    except Exception as e:
                        logger.error(f"✗ Component '{name}' shutdown error: {e}")

        # IMPROVED: Wait for external background threads with parallel joining and shared timeout
        logger.info("Waiting for background service threads to terminate...")
        external_threads = [
            "HardwareHealthCheck",
            "MetricsCleanup",
            "GovernanceCleanup",
            "AuditRotation",
            "RollbackCleanup",
        ]

        # Find matching threads
        threads_to_join = [
            thread
            for thread in threading.enumerate()
            if thread is not threading.current_thread()
            and thread.name in external_threads
        ]

        if threads_to_join:
            # Use a shared timeout for all thread joins (max 3 seconds total)
            shared_timeout = min(3.0, timeout / 3)

            def join_thread(thread):
                try:
                    thread.join(timeout=shared_timeout)
                    return thread.name, not thread.is_alive()
                except Exception as e:
                    logger.error(f"Error joining thread {thread.name}: {e}")
                    return thread.name, False

            # Join all threads in parallel with timeout
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(threads_to_join)
            ) as executor:
                futures = {executor.submit(join_thread, t): t for t in threads_to_join}
                done, not_done = concurrent.futures.wait(
                    futures.keys(),
                    timeout=shared_timeout
                    + 1,  # Allow a little extra for thread overhead
                )

                for future in done:
                    try:
                        name, success = future.result(timeout=0.1)
                        if success:
                            logger.debug(f"Thread {name} terminated")
                        else:
                            logger.warning(f"Thread {name} did not terminate in time")
                    except Exception as e:
                        logger.error(f"Error getting thread join result: {e}")

                for future in not_done:
                    future.cancel()
                    logger.warning(f"Thread join timed out")

        logger.info("Background service thread join complete.")

        # Step 4: Save state AFTER components are stable
        try:
            logger.info("Saving final state...")
            save_path = self.save_complete_state()
            logger.info(f"✔ Saved final state to: {save_path}")
        except Exception as e:
            logger.error(f"✗ Failed to save final state: {e}")

        logger.info("=" * 60)
        logger.info("UnifiedLearningSystem shutdown complete")
        logger.info("=" * 60)

    def _safe_component_shutdown(self, component: Any, name: str) -> bool:
        """Safely shutdown a component with error handling and logging"""
        try:
            logger.debug(f"Shutting down {name}...")
            component.shutdown()
            logger.debug(f"{name} shutdown complete")
            return True
        except Exception as e:
            logger.error(f"Error shutting down {name}: {e}", exc_info=True)
            return False

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure clean shutdown"""
        self.shutdown()
        return False


# Export all main classes
__all__ = [
    "UnifiedLearningSystem",
    "ContinualLearner",
    "EnhancedContinualLearner",
    "CurriculumLearner",
    "MetaLearner",
    "TaskDetector",
    "MetaCognitiveMonitor",
    "CompositionalUnderstanding",
    "ParameterHistoryManager",
    "RLHFManager",
    "LiveFeedbackProcessor",
    "UnifiedWorldModel",
    "LearningConfig",
    "TaskInfo",
    "FeedbackData",
    "LearningMode",
    "PlanningAlgorithm",
    "MetaLearningAlgorithm",
    "PacingStrategy",
]
