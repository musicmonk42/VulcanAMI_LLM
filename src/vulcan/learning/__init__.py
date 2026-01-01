"""
VULCAN-AGI Learning Module - Unified Integration
Coordinates: Continual + Curriculum + Meta + RLHF + World Model + Metacognition
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import gc
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

# ISSUE #5 FIX: Tool Weight Death Spiral Prevention
# These constants prevent tools from accumulating unbounded negative weights.
# Without bounds, tools can become unusable due to accumulated failures.
MIN_TOOL_WEIGHT = -0.1  # Minimum weight bound (prevents tools becoming unusable)
MAX_TOOL_WEIGHT = 0.2   # Maximum weight bound (prevents runaway positive weights)
WEIGHT_DECAY_FACTOR = 0.95  # Decay factor applied to weights periodically (moves towards 0)
WEIGHT_DECAY_INTERVAL_SECONDS = 3600  # Apply decay every hour
WEIGHT_DECAY_EPSILON = 0.001  # Weights below this threshold are considered zero
MAX_DECAY_INTERVALS = 168  # Cap decay at 1 week (168 hours) to prevent precision issues

# ISSUE P0.3 FIX: Corrupted Weight Detection Constants
# These constants control when tool weights are considered "corrupted" and need reset.
# Corruption typically occurs from the "death spiral" bug (P0.1) where LLM failures
# were incorrectly penalizing tools, causing all weights to degrade.
WEIGHT_CORRUPTION_THRESHOLD = 0.01  # Weights below this (when not zero) indicate corruption
WEIGHT_CORRUPTION_NEGATIVE_THRESHOLD = -0.05  # Any weight below this is definitely corrupted
WEIGHT_CORRUPTION_MIN_COUNT = 3  # Minimum corrupted weights to trigger reset
WEIGHT_CORRUPTION_PERCENTAGE = 0.5  # If this percentage of weights are corrupted, reset all

# ISSUE P0.1 FIX: LLM Fallback Detection Constants
# Used to detect when a response came from OpenAI fallback (LLM failure, not tool failure)
LLM_FALLBACK_LOW_CONFIDENCE_THRESHOLD = 0.1  # Default low confidence indicates fallback

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

# Import learning state persistence
try:
    from ..memory.learning_persistence import LearningStatePersistence
    LEARNING_PERSISTENCE_AVAILABLE = True
except ImportError:
    LEARNING_PERSISTENCE_AVAILABLE = False
    LearningStatePersistence = None
    logger.warning("LearningStatePersistence not available - learning state will not persist")

# BUG FIX: Import shared weight manager for propagating learned weights to Ensemble
# Without this, the learning system updates weights in its own dictionary,
# but the reasoning ensemble reads from a separate ToolWeightManager instance.
try:
    from ..reasoning.unified_reasoning import get_weight_manager
    WEIGHT_MANAGER_AVAILABLE = True
except ImportError:
    WEIGHT_MANAGER_AVAILABLE = False
    get_weight_manager = None
    logger.warning("ToolWeightManager not available - learned weights won't propagate to ensemble")

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
        # PERFORMANCE FIX: Increased threshold to reduce noise from expected slow routing
        # Production logs showed routing times of 20-70s during complex operations
        # Set to 10s to flag truly abnormal routing while allowing normal operation
        self.slow_routing_threshold_ms = 10000
        self.tool_weight_adjustments: Dict[str, float] = {}
        # ISSUE #15 FIX: Lock for thread-safe weight adjustments
        # Prevents race conditions when async process_outcome calls update weights concurrently
        self._weight_lock = threading.Lock()
        
        # PERSISTENCE FIX: Initialize persistence layer and load previous state
        # This ensures learning state persists across queries and server restarts
        self._learning_persistence = None
        if LEARNING_PERSISTENCE_AVAILABLE:
            try:
                self._learning_persistence = LearningStatePersistence()
                # Load previous tool weights from disk
                persisted_state = self._learning_persistence.load_state()
                persisted_weights = persisted_state.get("tool_weights", {})
                if persisted_weights:
                    self.tool_weight_adjustments = persisted_weights.copy()
                    logger.info(
                        f"[Learning] Loaded {len(persisted_weights)} tool weights from disk: "
                        f"{persisted_weights}"
                    )
                    
                    # ISSUE P0.3 FIX: Detect and fix corrupted tool weights
                    # Check if weights are severely degraded (near-zero or negative)
                    # This indicates a "death spiral" has occurred and weights need reset
                    corrupted_weights = []
                    for tool, weight in persisted_weights.items():
                        # Use defined constants for corruption detection thresholds
                        if weight < WEIGHT_CORRUPTION_NEGATIVE_THRESHOLD or (
                            weight < WEIGHT_CORRUPTION_THRESHOLD and weight != 0.0
                        ):
                            corrupted_weights.append((tool, weight))
                    
                    if len(corrupted_weights) >= WEIGHT_CORRUPTION_MIN_COUNT or (
                        len(corrupted_weights) > 0 and 
                        len(corrupted_weights) >= len(persisted_weights) * WEIGHT_CORRUPTION_PERCENTAGE
                    ):
                        # More than half of weights are corrupted, or 3+ corrupted - reset all
                        logger.warning(
                            f"[Learning] CORRUPTED WEIGHTS DETECTED: {len(corrupted_weights)} of "
                            f"{len(persisted_weights)} weights are severely degraded. "
                            f"Corrupted: {corrupted_weights}. RESETTING ALL WEIGHTS."
                        )
                        self.tool_weight_adjustments.clear()
                        if self._learning_persistence:
                            self._learning_persistence.clear_state()
                        logger.info("[Learning] Tool weights reset to defaults (0.0) due to corruption")
                    else:
                        # BUG FIX: Propagate persisted weights to shared ToolWeightManager at startup
                        # This ensures the ensemble uses learned weights from previous sessions
                        if WEIGHT_MANAGER_AVAILABLE and get_weight_manager:
                            try:
                                for tool, adjustment in persisted_weights.items():
                                    get_weight_manager().set_weight(tool, 1.0 + adjustment)
                                logger.info(f"[Learning] Propagated {len(persisted_weights)} persisted weights to ToolWeightManager")
                            except Exception as e:
                                logger.warning(f"[Learning] Failed to propagate persisted weights: {e}")
            except Exception as e:
                logger.warning(f"[Learning] Failed to initialize persistence: {e}")
                self._learning_persistence = None
        
        # ISSUE #5 FIX: Track last decay time for periodic weight decay
        self._last_weight_decay_time = time.time()
        
        # ISSUE #10 FIX: Slow routing recovery mechanism
        # Track consecutive slow routing events to trigger automatic recovery
        self._slow_routing_count = 0
        self._slow_routing_threshold_count = 3  # Trigger recovery after 3 consecutive slow events
        self._total_recoveries_attempted = 0
        self._total_recoveries_successful = 0
        self._last_recovery_time = 0.0
        self._recovery_cooldown_seconds = 60.0  # Don't attempt recovery more often than once per minute
        
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
        # Check for 'tools' first, then fall back to 'capabilities_used' (from QueryOutcome)
        # Use explicit None check to avoid unexpected fallback when tools is empty list
        tools_value = outcome.get('tools')
        tools = tools_value if tools_value is not None else outcome.get('capabilities_used', [])
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
        
        # 2. Detect and log slow routing with automatic recovery (ISSUE #10 FIX)
        if routing_ms > self.slow_routing_threshold_ms:
            logger.warning(f"[Learning] SLOW ROUTING DETECTED: {routing_ms}ms (threshold: {self.slow_routing_threshold_ms}ms)")
            logger.warning(f"[Learning] Slow query details: type={query_type}, tools={tools}")
            
            # Increment slow routing counter
            self._slow_routing_count += 1
            
            # Feed to MetaLearner for strategy adjustment
            if self.meta_learner:
                try:
                    if hasattr(self.meta_learner, 'record_slow_routing'):
                        await self.meta_learner.record_slow_routing(query_type, tools, routing_ms)
                except Exception as e:
                    logger.error(f"[Learning] MetaLearner slow routing error: {e}")
            
            # ISSUE #10 FIX: Trigger automatic recovery after consecutive slow routing events
            if self._slow_routing_count >= self._slow_routing_threshold_count:
                current_time = time.time()
                if current_time - self._last_recovery_time >= self._recovery_cooldown_seconds:
                    logger.warning(
                        f"[Learning] Triggering slow routing RECOVERY: "
                        f"{self._slow_routing_count} consecutive slow events detected"
                    )
                    recovery_success = await self._attempt_slow_routing_recovery()
                    self._total_recoveries_attempted += 1
                    if recovery_success:
                        self._total_recoveries_successful += 1
                        logger.info("[Learning] Slow routing recovery SUCCESSFUL")
                    else:
                        logger.warning("[Learning] Slow routing recovery FAILED")
                    self._last_recovery_time = current_time
                    self._slow_routing_count = 0  # Reset counter after recovery attempt
                else:
                    remaining_cooldown = self._recovery_cooldown_seconds - (current_time - self._last_recovery_time)
                    logger.warning(
                        f"[Learning] Recovery on cooldown, waiting {remaining_cooldown:.1f}s before next attempt"
                    )
        else:
            # Reset slow routing counter on successful (fast) routing
            self._slow_routing_count = 0
        
        # 3. Update tool weights based on success/failure
        # ISSUE #15 FIX: Use lock to prevent race conditions in concurrent async calls
        # ISSUE #5 FIX: Apply weight bounds to prevent death spiral
        # ISSUE #9 FIX: Skip weight adjustment for system/code bugs (AttributeError, TypeError, etc.)
        #   These errors indicate code bugs, not tool performance issues. Penalizing tools for
        #   code bugs causes incorrect weight drift that deprioritizes working tools.
        # ISSUE P0.1 FIX: Skip weight adjustment when LLM (not tool) fails
        #   The root cause of the "death spiral" is penalizing tools when the INTERNAL LLM
        #   returns None and falls back to OpenAI. This is an LLM failure, not a tool failure.
        #   Tools were selected correctly but the LLM timed out or failed to generate.
        if tools:  # Only if tools were recorded
            # Check if this is a system/code bug vs tool performance failure
            error_type = outcome.get('error_type', '')
            error_message = outcome.get('error', '') or outcome.get('error_message', '')
            
            # System error patterns that indicate code bugs (not tool performance issues)
            system_error_patterns = [
                'AttributeError', 'TypeError', 'KeyError', 'NameError',
                'ImportError', 'ModuleNotFoundError', 'SyntaxError',
                "has no attribute", "object is not", "is not defined",
                "unexpected keyword argument", "missing required"
            ]
            
            is_system_error = any(
                pattern in error_type or pattern in error_message 
                for pattern in system_error_patterns
            )
            
            # ISSUE P0.1 FIX: Check if this was an LLM fallback (not a tool failure)
            # When the internal LLM fails and falls back to OpenAI, we should NOT penalize
            # tools. The tools were selected correctly; the LLM just failed to generate.
            # Penalizing tools for LLM failures causes the "death spiral" where all tool
            # weights become near-zero or negative.
            source = outcome.get('source', '')
            systems_used = outcome.get('systems_used', [])
            metadata = outcome.get('metadata', {})
            
            # Check various indicators of LLM fallback
            is_llm_fallback = (
                # Direct source indicators
                source in ('openai_full_reasoning_fallback', 'openai_fallback', 'openai') or
                # System used indicators
                'openai_full_reasoning_fallback' in systems_used or
                'openai_fallback' in systems_used or
                # Metadata indicators
                metadata.get('openai_role') == 'full_reasoning_fallback' or
                metadata.get('vulcan_llm_failed', False) or
                # Low confidence from fallback (default 10% indicates fallback)
                (outcome.get('confidence', 1.0) <= LLM_FALLBACK_LOW_CONFIDENCE_THRESHOLD and source != 'local')
            )
            
            if is_system_error and status != 'success':
                logger.info(
                    f"[Learning] SKIPPING weight adjustment for system error: "
                    f"type={error_type}, message={error_message[:100] if error_message else 'N/A'}. "
                    f"Tools {tools} not penalized for code bug."
                )
                # Don't adjust weights - this was a code bug, not tool performance
                weight_delta = 0.0
            elif is_llm_fallback and status != 'success':
                # ISSUE P0.1 FIX: Don't penalize tools when LLM failed
                logger.info(
                    f"[Learning] SKIPPING weight adjustment for LLM fallback: "
                    f"source={source}, systems={systems_used}. "
                    f"Tools {tools} not penalized - internal LLM failed, not tool selection."
                )
                weight_delta = 0.0
            else:
                weight_delta = WEIGHT_ADJUSTMENT_SUCCESS if status == 'success' else WEIGHT_ADJUSTMENT_FAILURE
            
            if weight_delta != 0.0:  # Only process if there's a weight change
                with self._weight_lock:
                    # Apply periodic weight decay first (moves weights towards 0)
                    self._apply_weight_decay_if_needed()
                    
                    for tool in tools:
                        if tool not in self.tool_weight_adjustments:
                            self.tool_weight_adjustments[tool] = 0.0
                        
                        new_weight = self.tool_weight_adjustments[tool] + weight_delta
                        
                        # ISSUE #5 FIX: Clamp weight to bounds to prevent death spiral
                        old_weight = self.tool_weight_adjustments[tool]
                        self.tool_weight_adjustments[tool] = max(MIN_TOOL_WEIGHT, min(MAX_TOOL_WEIGHT, new_weight))
                        
                        # Log if weight was clamped
                        if self.tool_weight_adjustments[tool] != new_weight:
                            logger.info(
                                f"[Learning] Tool '{tool}' weight CLAMPED: {old_weight:+.3f} + {weight_delta:+.3f} = "
                                f"{new_weight:+.3f} -> {self.tool_weight_adjustments[tool]:+.3f} (bounds: [{MIN_TOOL_WEIGHT}, {MAX_TOOL_WEIGHT}])"
                            )
                        else:
                            logger.info(f"[Learning] Tool '{tool}' weight adjustment: {weight_delta:+.3f} (cumulative: {self.tool_weight_adjustments[tool]:+.3f})")
                        
                        # BUG FIX: Propagate weight to shared ToolWeightManager so Ensemble can use it
                        # Previously, learning updated its own dictionary but Ensemble read from a separate
                        # ToolWeightManager instance, so learned weights were never applied.
                        if WEIGHT_MANAGER_AVAILABLE and get_weight_manager:
                            try:
                                # Use absolute weight (1.0 + adjustment) since ToolWeightManager expects base weight
                                get_weight_manager().set_weight(tool, 1.0 + self.tool_weight_adjustments[tool])
                                logger.debug(f"[Learning] Propagated weight to ToolWeightManager: {tool} = {1.0 + self.tool_weight_adjustments[tool]:.4f}")
                            except Exception as e:
                                logger.warning(f"[Learning] Failed to propagate weight to ToolWeightManager: {e}")
                    
                    # PERSISTENCE FIX: Save tool weights to disk after every update
                    # This ensures learning state persists across queries and server restarts
                    if self._learning_persistence:
                        try:
                            self._learning_persistence.update_tool_weights(self.tool_weight_adjustments)
                        except Exception as e:
                            logger.warning(f"[Learning] Failed to persist tool weights: {e}")
        else:
            logger.warning(f"[Learning] No tools recorded for {query_id} - cannot learn from selection")
        
        # 4. Store interaction in learning persistence for query/answer history
        # This enables learning from past interactions and semantic search
        if self._learning_persistence:
            try:
                # Extract query and answer from outcome if available
                # Outcome may contain: 'query'/'prompt' for input, 'answer'/'response' for output
                # This supports multiple outcome formats from different callers
                query_text = outcome.get('query') or outcome.get('prompt', '')
                answer_text = outcome.get('answer') or outcome.get('response', '')
                
                if query_text or answer_text:
                    self._learning_persistence.add_interaction(
                        query_id=query_id,
                        query=str(query_text),
                        answer=str(answer_text),
                        tools_used=tools,
                        success=(status == 'success'),
                        latency_ms=float(outcome.get('total_ms') or routing_ms or 0),
                        metadata={
                            'query_type': query_type,
                            'complexity': float(outcome.get('complexity') or 0.0),
                            'routing_ms': float(routing_ms or 0),
                        }
                    )
            except Exception as e:
                logger.debug(f"[Learning] Failed to store interaction: {e}")
        
        # 5. Feed to MetaLearner for pattern detection
        if self.meta_learner:
            try:
                if hasattr(self.meta_learner, 'update_from_outcome'):
                    await self.meta_learner.update_from_outcome(outcome)
                    logger.info(f"[MetaLearner] Processed outcome {query_id}")
            except Exception as e:
                logger.error(f"[Learning] MetaLearner error: {e}")
        
        logger.info(f"[Learning] Outcome processing complete for {query_id}")

    def _apply_weight_decay_if_needed(self) -> None:
        """
        Apply periodic weight decay to all tool weights.
        
        ISSUE #5 FIX: This method prevents the tool weight death spiral by
        periodically decaying weights towards zero. Without decay, weights
        can accumulate indefinitely in either direction.
        
        This should be called while holding self._weight_lock.
        """
        current_time = time.time()
        time_since_decay = current_time - self._last_weight_decay_time
        
        if time_since_decay >= WEIGHT_DECAY_INTERVAL_SECONDS:
            # Calculate how many decay intervals have passed, capped to prevent precision issues
            intervals = min(
                int(time_since_decay / WEIGHT_DECAY_INTERVAL_SECONDS),
                MAX_DECAY_INTERVALS
            )
            decay_multiplier = WEIGHT_DECAY_FACTOR ** intervals
            
            decayed_count = 0
            for tool in self.tool_weight_adjustments:
                old_weight = self.tool_weight_adjustments[tool]
                if abs(old_weight) > WEIGHT_DECAY_EPSILON:  # Only decay non-zero weights
                    self.tool_weight_adjustments[tool] = old_weight * decay_multiplier
                    decayed_count += 1
            
            if decayed_count > 0:
                logger.info(
                    f"[Learning] Applied weight decay (factor={decay_multiplier:.3f}) "
                    f"to {decayed_count} tool weights after {time_since_decay:.0f}s"
                )
            
            self._last_weight_decay_time = current_time

    def get_tool_weight_adjustment(self, tool: str) -> float:
        """
        Get cumulative weight adjustment for a tool.
        
        This is used by ToolSelector to apply learned weight adjustments
        to tool selection probabilities.
        
        Args:
            tool: Tool name
            
        Returns:
            Cumulative weight adjustment (positive = more successful, negative = less successful)
            Value is bounded by [MIN_TOOL_WEIGHT, MAX_TOOL_WEIGHT]
        """
        # ISSUE #15 FIX: Use lock for thread-safe read
        with self._weight_lock:
            return self.tool_weight_adjustments.get(tool, 0.0)

    def reset_tool_weights(self) -> None:
        """
        Reset all tool weights to zero.
        
        ISSUE #5 FIX: This method allows manual reset of accumulated weights
        if the system gets into a bad state. Should be called during system
        recovery or after major configuration changes.
        """
        with self._weight_lock:
            old_weights = dict(self.tool_weight_adjustments)
            self.tool_weight_adjustments.clear()
            self._last_weight_decay_time = time.time()
            logger.info(f"[Learning] Reset tool weights: {old_weights} -> cleared")
            
            # PERSISTENCE FIX: Clear persisted state when weights are reset
            if self._learning_persistence:
                try:
                    self._learning_persistence.clear_state()
                    logger.info("[Learning] Cleared persisted learning state")
                except Exception as e:
                    logger.warning(f"[Learning] Failed to clear persisted state: {e}")

    async def _attempt_slow_routing_recovery(self) -> bool:
        """
        Attempt to recover from slow routing performance degradation.
        
        ISSUE #10 FIX: This method implements automatic recovery when the system
        detects persistent slow routing. Recovery actions include:
        1. Clearing embedding caches to remove potentially stale data
        2. Resetting circuit breakers that may be in degraded state
        3. Triggering garbage collection to reclaim memory
        
        Returns:
            True if recovery was successful, False otherwise
        """
        logger.info("[Learning] Starting slow routing recovery process...")
        recovery_actions_taken = []
        
        try:
            # 1. Clear embedding cache to reset any stale or accumulated entries
            try:
                from vulcan.routing.embedding_cache import clear_cache, get_cache_stats
                stats_before = get_cache_stats()
                clear_cache()
                stats_after = get_cache_stats()
                recovery_actions_taken.append(
                    f"Embedding cache cleared: {stats_before.get('size', 0)} entries removed"
                )
                logger.info(
                    f"[Learning] Recovery: Embedding cache cleared "
                    f"(was {stats_before.get('size', 0)} entries)"
                )
            except ImportError:
                logger.debug("[Learning] Recovery: Embedding cache module not available")
            except Exception as e:
                logger.warning(f"[Learning] Recovery: Failed to clear embedding cache: {e}")
            
            # 2. Reset embedding circuit breaker to allow fresh embedding attempts
            try:
                from vulcan.reasoning.selection.embedding_circuit_breaker import (
                    reset_circuit_breaker,
                    get_circuit_breaker_stats,
                )
                stats_before = get_circuit_breaker_stats()
                reset_circuit_breaker()
                recovery_actions_taken.append(
                    f"Circuit breaker reset: was in {stats_before.get('state', 'unknown')} state"
                )
                logger.info(
                    f"[Learning] Recovery: Circuit breaker reset "
                    f"(was {stats_before.get('state', 'unknown')})"
                )
            except ImportError:
                logger.debug("[Learning] Recovery: Circuit breaker module not available")
            except Exception as e:
                logger.warning(f"[Learning] Recovery: Failed to reset circuit breaker: {e}")
            
            # 3. ISSUE #2 FIX: Clear SemanticToolMatcher query embedding cache
            try:
                from vulcan.reasoning.selection.semantic_tool_matcher import SemanticToolMatcher
                cache_stats = SemanticToolMatcher.get_cache_stats()
                SemanticToolMatcher.clear_query_cache()
                recovery_actions_taken.append(
                    f"SemanticToolMatcher query cache cleared: {cache_stats.get('size', 0)} entries"
                )
                logger.info(
                    f"[Learning] Recovery: SemanticToolMatcher query cache cleared "
                    f"(was {cache_stats.get('size', 0)} entries, hit_rate={cache_stats.get('hit_rate', 0):.2%})"
                )
            except ImportError:
                logger.debug("[Learning] Recovery: SemanticToolMatcher not available")
            except Exception as e:
                logger.warning(f"[Learning] Recovery: Failed to clear SemanticToolMatcher cache: {e}")
            
            # 4. Trigger garbage collection to reclaim memory
            gc.collect()
            recovery_actions_taken.append("Garbage collection triggered")
            logger.info("[Learning] Recovery: Garbage collection complete")
            
            # Log recovery summary
            logger.info(
                f"[Learning] Slow routing recovery complete: {len(recovery_actions_taken)} actions taken"
            )
            for action in recovery_actions_taken:
                logger.info(f"[Learning] Recovery action: {action}")
            
            return len(recovery_actions_taken) > 0
            
        except Exception as e:
            logger.error(f"[Learning] Slow routing recovery failed with error: {e}")
            return False

    def get_recovery_stats(self) -> Dict[str, Any]:
        """
        Get statistics about slow routing recovery attempts.
        
        ISSUE #10 FIX: This method provides visibility into recovery attempts
        to address the "recovery mechanisms not triggered" issue.
        
        Returns:
            Dictionary with recovery statistics
        """
        return {
            "slow_routing_count": self._slow_routing_count,
            "slow_routing_threshold_count": self._slow_routing_threshold_count,
            "total_recoveries_attempted": self._total_recoveries_attempted,
            "total_recoveries_successful": self._total_recoveries_successful,
            "last_recovery_time": self._last_recovery_time,
            "recovery_cooldown_seconds": self._recovery_cooldown_seconds,
        }

    def get_persistence_stats(self) -> Dict[str, Any]:
        """
        Get statistics about learning state persistence.
        
        PERSISTENCE FIX: Provides visibility into the persistence layer state.
        
        Returns:
            Dictionary with persistence statistics
        """
        if self._learning_persistence:
            return self._learning_persistence.get_stats()
        return {
            "persistence_available": False,
            "reason": "LearningStatePersistence not initialized"
        }

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
    # Weight adjustment constants
    "WEIGHT_ADJUSTMENT_SUCCESS",
    "WEIGHT_ADJUSTMENT_FAILURE",
    "MIN_TOOL_WEIGHT",
    "MAX_TOOL_WEIGHT",
]

# Mathematical Accuracy Integration
try:
    from .mathematical_accuracy_integration import (
        MathematicalAccuracyIntegration,
        MathematicalFeedback,
        create_math_learning_integration,
        MATH_ERROR_PENALTIES,
        MATH_VERIFICATION_REWARD,
    )
    
    MATH_LEARNING_AVAILABLE = True
    __all__.extend([
        "MathematicalAccuracyIntegration",
        "MathematicalFeedback",
        "create_math_learning_integration",
        "MATH_ERROR_PENALTIES",
        "MATH_VERIFICATION_REWARD",
        "MATH_LEARNING_AVAILABLE",
    ])
except ImportError as e:
    logger.warning(f"Mathematical accuracy integration not available: {e}")
    MATH_LEARNING_AVAILABLE = False
    MathematicalAccuracyIntegration = None
    MathematicalFeedback = None
    create_math_learning_integration = None
    MATH_ERROR_PENALTIES = {}
    MATH_VERIFICATION_REWARD = 0.015
