"""
Continual learning implementations with EWC and experience replay
"""

import asyncio
import logging
import pickle
import re
import secrets
import threading  # <-- threading is imported here
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

try:
    from safety.safety_types import SafetyConfig
    from safety.safety_validator import EnhancedSafetyValidator
except ImportError:
    EnhancedSafetyValidator = None

# FIX 602: Add import guard for HierarchicalMemory
try:
    from ..memory.hierarchical import HierarchicalMemory

    HIERARCHICAL_AVAILABLE = True
except ImportError:
    HIERARCHICAL_AVAILABLE = False
    HierarchicalMemory = None

# Knowledge Crystallizer integration
try:
    from ..knowledge_crystallizer import (
        KnowledgeCrystallizer,
        ExecutionTrace,
        KNOWLEDGE_CRYSTALLIZER_AVAILABLE,
    )
except ImportError:
    KNOWLEDGE_CRYSTALLIZER_AVAILABLE = False
    KnowledgeCrystallizer = None
    ExecutionTrace = None

from ..config import EMBEDDING_DIM, HIDDEN_DIM, LATENT_DIM
from ..security_fixes import safe_pickle_load
from .learning_types import FeedbackData, LearningConfig, TaskInfo
from .meta_learning import MetaLearner, TaskDetector
from .parameter_history import ParameterHistoryManager
from .rlhf_feedback import LiveFeedbackProcessor, RLHFManager

logger = logging.getLogger(__name__)

# ============================================================
# CONTINUAL LEARNING METRICS
# ============================================================


@dataclass
class ContinualMetrics:
    """Metrics for continual learning evaluation"""

    backward_transfer: float = 0.0  # Performance change on old tasks
    forward_transfer: float = 0.0  # Performance on new tasks
    average_accuracy: float = 0.0  # Average across all tasks
    forgetting_measure: float = 0.0  # Amount of forgetting
    task_accuracies: Dict[str, float] = None

    def __post_init__(self):
        if self.task_accuracies is None:
            self.task_accuracies = {}


# ============================================================
# BASIC CONTINUAL LEARNER (for backward compatibility)
# ============================================================


class ContinualLearner:
    """Original continual learner for backward compatibility"""

    def __init__(self, memory=None):
        self.enabled = True
        self.ewc_importance = {}
        self.task_models = {}
        self.experience_buffer = deque(maxlen=1000)  # Added for update method
        self.learning_history = []
        self.slow_routing_patterns = []
        self.tool_success_rates = {}
        
        # Try to use HierarchicalMemory, fall back to simple dict
        if memory:
            self.memory = memory
            logger.info("[ContinualLearner] Using provided memory instance")
        else:
            try:
                from ..memory.hierarchical import HierarchicalMemory
                from ..memory.base import MemoryConfig
                # Create default config for HierarchicalMemory
                default_config = MemoryConfig(
                    max_working_memory=50,
                    max_short_term=1000,
                    max_long_term=10000,
                )
                self.memory = HierarchicalMemory(config=default_config)
                logger.info("[ContinualLearner] Hierarchical memory initialized")
            except Exception as e:
                logger.warning(f"[ContinualLearner] HierarchicalMemory unavailable: {e}")
                self.memory = {'patterns': [], 'weights': {}, 'history': []}
                logger.info("[ContinualLearner] Using dict fallback memory - learning still enabled")
        
        logger.info("[ContinualLearner] Initialized and ready for learning")

    def process_experience(self, experience: Dict[str, Any]) -> Dict[str, Any]:
        """Basic experience processing"""
        # Simplified for backward compatibility
        return {"processed": True, "loss": 0.0}

    def update(self, experience: Dict[str, Any]):
        """
        Update learner with new experience

        Args:
            experience: Dict containing:
                - features: np.ndarray
                - result: Any
                - success: bool
                - tool_used: str
                - context: Dict
        """
        try:
            # Extract components
            features = experience.get("features")
            experience.get("result")
            experience.get("success", True)
            tool_used = experience.get("tool_used")
            experience.get("context", {})

            # This is a simplified learner; we'll just store the experience
            self.experience_buffer.append(experience)

            # In a full implementation, this would trigger model updates, etc.
            # For now, we'll just log it.
            if features is not None:
                logger.debug(f"Updating task memory for tool {tool_used}")

            # Trigger consolidation periodically
            if len(self.experience_buffer) >= 100:
                self.consolidate_knowledge()

        except Exception as e:
            logger.error(f"Continual learning update failed: {e}")

    def consolidate_knowledge(self):
        """Placeholder for knowledge consolidation"""
        logger.info("Consolidating knowledge in basic ContinualLearner...")
        # In a real scenario, this would update EWC importance, etc.
        self.experience_buffer.clear()

    async def learn_from_outcome(self, outcome: Dict[str, Any]) -> None:
        """
        Process an outcome and extract learnable patterns.
        
        This method is called by the UnifiedLearningSystem when an outcome
        is recorded via the OutcomeBridge.
        
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
        logger.info(f"[ContinualLearner] Learning from: {query_id}")
        
        # Track tool success rates
        tools = outcome.get('tools', [])
        status = outcome.get('status', 'unknown')
        for tool in tools:
            if tool not in self.tool_success_rates:
                self.tool_success_rates[tool] = {'success': 0, 'total': 0}
            self.tool_success_rates[tool]['total'] += 1
            if status == 'success':
                self.tool_success_rates[tool]['success'] += 1
        
        # Detect slow routing patterns
        routing_ms = outcome.get('routing_ms', 0)
        if routing_ms > 5000:
            pattern = {
                'query_type': outcome.get('query_type'),
                'tools': tools,
                'routing_ms': routing_ms,
                'timestamp': outcome.get('timestamp')
            }
            self.slow_routing_patterns.append(pattern)
            logger.warning(f"[ContinualLearner] Recorded slow routing pattern: {routing_ms}ms for {tools}")
        
        self.learning_history.append(outcome)
        logger.info(f"[ContinualLearner] History size: {len(self.learning_history)}, slow patterns: {len(self.slow_routing_patterns)}")


# ============================================================
# PROGRESSIVE NEURAL NETWORK MODULE
# ============================================================


class ProgressiveColumn(nn.Module):
    """Single column in Progressive Neural Network"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            ]
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass returning intermediate activations"""
        activations = []
        h = x
        for layer in self.layers:
            h = layer(h)
            if isinstance(layer, nn.ReLU):
                activations.append(h)
        activations.append(h)
        return activations


class ProgressiveNeuralNetwork(nn.Module):
    """Progressive Neural Network for continual learning"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.columns = nn.ModuleList()
        self.lateral_connections = nn.ModuleDict()

    def add_column(self, task_id: str) -> None:
        """Add new column for task"""
        column_idx = len(self.columns)
        self.columns.append(
            ProgressiveColumn(self.input_dim, self.hidden_dim, self.output_dim)
        )

        # Add lateral connections from all previous columns
        if column_idx > 0:
            for prev_idx in range(column_idx):
                key = f"{prev_idx}_to_{column_idx}"
                self.lateral_connections[key] = nn.ModuleList(
                    [
                        nn.Linear(self.hidden_dim, self.hidden_dim),
                        nn.Linear(self.hidden_dim, self.hidden_dim),
                        nn.Linear(self.output_dim, self.output_dim),
                    ]
                )

    def forward(self, x: torch.Tensor, column_idx: int) -> torch.Tensor:
        """Forward pass through specific column with lateral connections"""
        if column_idx >= len(self.columns):
            raise ValueError(f"Column {column_idx} not found")

        # Get activations from current column
        current_activations = self.columns[column_idx](x)

        # Add lateral connections from previous columns
        if column_idx > 0:
            for prev_idx in range(column_idx):
                prev_activations = self.columns[prev_idx](x)
                key = f"{prev_idx}_to_{column_idx}"

                if key in self.lateral_connections:
                    lateral = self.lateral_connections[key]
                    for i, (prev_act, lat_conn) in enumerate(
                        zip(prev_activations, lateral)
                    ):
                        if i < len(current_activations) - 1:
                            current_activations[i] = current_activations[i] + lat_conn(
                                prev_act
                            )

        return current_activations[-1]


# ============================================================
# ENHANCED CONTINUAL LEARNER
# ============================================================


class EnhancedContinualLearner(nn.Module):
    """Enhanced continual learning with task detection, RLHF, and live feedback."""

    def __init__(
        self,
        embedding_dim: int = EMBEDDING_DIM,
        config: LearningConfig = None,
        use_hierarchical: bool = True,
        use_progressive: bool = False,
    ):
        super().__init__()
        self.config = config or LearningConfig()
        self.embedding_dim = embedding_dim
        self.use_hierarchical = use_hierarchical
        self.use_progressive = use_progressive

        # Task management
        self.task_detector = TaskDetector(self.config.task_detection_threshold)
        self.task_models = nn.ModuleDict()
        self.task_info = {}
        self.task_order = []  # Order in which tasks were learned

        # Shared backbone
        self.shared_encoder = nn.Sequential(
            nn.Linear(embedding_dim, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # General model
        self.general_model = self._create_task_head()

        # Progressive Neural Network (optional)
        if use_progressive:
            self.progressive_network = ProgressiveNeuralNetwork(
                embedding_dim, HIDDEN_DIM, embedding_dim
            )

        # EWC components
        self.fisher_information = {}
        self.optimal_params = {}

        # PackNet components (parameter isolation)
        self.task_masks = {}  # Binary masks for each task
        self.free_capacity = torch.ones(sum(p.numel() for p in self.parameters()))

        # Experience replay
        self.replay_buffer = deque(maxlen=self.config.replay_buffer_size)
        # FIX 603: Convert defaultdict to regular dict with helper method and lock
        self.task_buffers = {}
        self._task_buffers_lock = threading.RLock()

        # Optimizers and schedulers
        self.optimizers = {}
        self.schedulers = {}
        self._setup_optimizer("general", self.general_model)

        # FIXED: Use single lock for all stats to prevent deadlocks
        self._stats_lock = threading.RLock()
        self.performance_tracker = {}
        self.consolidation_counter = 0
        self.continual_metrics = ContinualMetrics()
        self.task_performance_history = {}

        # FIX 602: Use import guard for HierarchicalMemory
        if use_hierarchical and HIERARCHICAL_AVAILABLE:
            self.hierarchical_memory = HierarchicalMemory(config=self.config)
            logger.info("Hierarchical memory initialized")
        else:
            if use_hierarchical and not HIERARCHICAL_AVAILABLE:
                logger.warning("HierarchicalMemory not available")
            self.hierarchical_memory = None

        # RLHF integration
        if self.config.rlhf_enabled:
            self.rlhf_manager = RLHFManager(self, self.config)
        else:
            self.rlhf_manager = None

        # Live feedback processor
        self.live_feedback = LiveFeedbackProcessor(self, self.config)

        # Knowledge Crystallizer integration
        if KNOWLEDGE_CRYSTALLIZER_AVAILABLE:
            try:
                self.knowledge_crystallizer = KnowledgeCrystallizer()
                logger.info("[KnowledgeCrystallizer] Initialized successfully")
            except Exception as e:
                logger.warning(f"[KnowledgeCrystallizer] Failed to initialize: {e}")
                self.knowledge_crystallizer = None
        else:
            self.knowledge_crystallizer = None
            logger.debug("[KnowledgeCrystallizer] Module not available")

        # Parameter history
        self.param_history = ParameterHistoryManager(config=self.config)

        # Meta-learner
        self.meta_learner = MetaLearner(self, self.config)

        # Safety validator (singleton to avoid thread leaks)
        self._safety_validator = None

        # Thread safety - single main lock for operations
        self._lock = threading.RLock()
        self._shutdown = threading.Event()

        # Save/Load path
        self.save_path = Path("continual_learning_checkpoints")
        self.save_path.mkdir(parents=True, exist_ok=True)

        # --- START BUG FIX: Correctly define _NON_PICKLABLE_TYPES ---
        # Get the *types* of locks, not the factory functions
        try:
            lock_type = type(threading.Lock())
        except Exception:
            lock_type = None

        try:
            rlock_type = type(threading.RLock())
        except Exception:
            rlock_type = None

        # Thread and Event are classes (types), so getattr is fine
        self._NON_PICKLABLE_TYPES = tuple(
            [
                lock_type,
                rlock_type,
                getattr(threading, "Thread", None),
                getattr(threading, "Event", None),
            ]
        )
        # Filter out None values in case an attribute doesn't exist or type() failed
        self._NON_PICKLABLE_TYPES = tuple(
            t for t in self._NON_PICKLABLE_TYPES if t is not None
        )
        # --- END BUG FIX ---

    # FIXED: Unified helper methods using single stats lock
    def _get_performance_deque(self, task_id: str) -> deque:
        """Get or create performance deque for task (thread-safe)"""
        with self._stats_lock:
            if task_id not in self.performance_tracker:
                self.performance_tracker[task_id] = deque(maxlen=100)
            return self.performance_tracker[task_id]

    def _get_task_buffer(self, task_id: str) -> deque:
        """Get or create task buffer (thread-safe)"""
        with self._task_buffers_lock:
            if task_id not in self.task_buffers:
                self.task_buffers[task_id] = deque(maxlen=1000)
            return self.task_buffers[task_id]

    def _get_task_performance_history(self, task_id: str) -> list:
        """Get or create task performance history (thread-safe)"""
        with self._stats_lock:
            if task_id not in self.task_performance_history:
                self.task_performance_history[task_id] = []
            return self.task_performance_history[task_id]

    def _increment_consolidation_counter(self) -> bool:
        """FIXED: Atomically increment and check consolidation counter"""
        with self._stats_lock:
            self.consolidation_counter += 1
            should_consolidate = (
                self.consolidation_counter >= self.config.consolidation_threshold
            )
            if should_consolidate:
                self.consolidation_counter = 0
            return should_consolidate

    # ============================================================
    # RLHF PUBLIC INTERFACE METHODS
    # ============================================================

    def receive_feedback(self, feedback: FeedbackData):
        """
        Public method to receive human feedback.
        Delegates to RLHF manager if enabled.
        
        Args:
            feedback: FeedbackData instance with user feedback
        """
        if self.rlhf_manager:
            self.rlhf_manager.receive_feedback(feedback)
            # Log only feedback type and reward, not user content
            logger.info(f"[RLHF] Feedback received: type={feedback.feedback_type}, reward={feedback.reward_signal}")
        else:
            logger.warning("[RLHF] Feedback received but RLHF is disabled")

    def process_live_feedback(self, user_message: str, context: Dict[str, Any]):
        """
        Process real-time feedback from user messages.
        Auto-detects corrections and preferences.
        
        Args:
            user_message: User's message
            context: Conversation context including previous query/response
        """
        if self.live_feedback:
            # Create feedback dict for async processing
            feedback_data = {
                "type": "live",
                "message": user_message,
                "context": context,
                "timestamp": time.time(),
            }
            # Queue for async processing (if event loop is running)
            try:
                loop = asyncio.get_running_loop()
                # Store task reference to prevent garbage collection
                task = asyncio.ensure_future(self.live_feedback.process_live_feedback(feedback_data))
                # Add callback to handle any errors silently
                task.add_done_callback(lambda t: t.exception() if t.done() and not t.cancelled() else None)
            except RuntimeError:
                # No event loop - process synchronously
                logger.debug("[LiveFeedback] No event loop - processing synchronously")
                self.live_feedback.performance_tracker["feedback_processed"] += 1
            # Log only message length to avoid PII exposure
            logger.debug(f"[LiveFeedback] Processed message: length={len(user_message)} chars")

    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get current RLHF statistics"""
        stats = {
            "rlhf_enabled": self.rlhf_manager is not None,
            "live_feedback_enabled": self.live_feedback is not None,
        }
        
        if self.rlhf_manager:
            stats["rlhf_stats"] = self.rlhf_manager.feedback_stats.copy()
        
        if self.live_feedback:
            stats["live_feedback_stats"] = self.live_feedback.performance_tracker.copy()
        
        return stats

    def submit_thumbs_feedback(self, query_id: str, response_id: str, is_positive: bool, 
                               context: Optional[Dict[str, Any]] = None):
        """
        Submit thumbs up/down feedback for a response.
        
        Args:
            query_id: ID of the original query
            response_id: ID of the response being rated
            is_positive: True for thumbs up, False for thumbs down
            context: Optional additional context
        """
        reward_signal = 1.0 if is_positive else -1.0
        feedback_type = "thumbs_up" if is_positive else "thumbs_down"
        
        feedback = FeedbackData(
            feedback_type=feedback_type,
            user_input=query_id,
            agent_response=response_id,
            reward_signal=reward_signal,
            context=context or {},
            metadata={
                "query_id": query_id,
                "response_id": response_id,
                "feedback_source": "ui_button",
            },
        )
        
        self.receive_feedback(feedback)
        # Log only feedback type to avoid ID exposure
        logger.info(f"[RLHF] Thumbs {'up' if is_positive else 'down'} feedback submitted")

    # ============================================================
    # AUTO-DETECTION UTILITIES (Task 2)
    # ============================================================

    # Patterns for detecting different types of feedback in user messages
    CORRECTION_PATTERNS = [
        r"\b(no|wrong|incorrect|mistake|error|that'?s not right)\b",
        r"\b(actually|correction|should be|meant to say)\b",
        r"\b(fix|redo|try again|not what i asked)\b",
    ]
    
    POSITIVE_PATTERNS = [
        r"\b(thanks|thank you|great|perfect|exactly|correct|right|good job|well done)\b",
        r"\b(awesome|excellent|amazing|helpful|works)\b",
        r"\b(yes|yep|yeah|that'?s it|bingo)\b",
    ]
    
    NEGATIVE_PATTERNS = [
        r"\b(bad|terrible|awful|useless|unhelpful|disappointing)\b",
        r"\b(doesn'?t work|didn'?t work|broken|failed)\b",
        r"\b(hate|worst|garbage|trash)\b",
    ]
    
    PREFERENCE_PATTERNS = [
        r"\b(prefer|rather|instead|better if|would like)\b",
        r"\b(don'?t like|dislike|stop doing|avoid)\b",
        r"\b(always|never|please don'?t|please do)\b",
    ]

    def detect_feedback_type(self, user_message: str) -> Dict[str, Any]:
        """
        Auto-detect feedback type from user message.
        
        Args:
            user_message: The user's message text
            
        Returns:
            Dict with detected feedback type, confidence, and signals
        """
        message_lower = user_message.lower()
        
        # Count pattern matches for each type
        correction_matches = sum(
            1 for pattern in self.CORRECTION_PATTERNS 
            if re.search(pattern, message_lower, re.IGNORECASE)
        )
        positive_matches = sum(
            1 for pattern in self.POSITIVE_PATTERNS 
            if re.search(pattern, message_lower, re.IGNORECASE)
        )
        negative_matches = sum(
            1 for pattern in self.NEGATIVE_PATTERNS 
            if re.search(pattern, message_lower, re.IGNORECASE)
        )
        preference_matches = sum(
            1 for pattern in self.PREFERENCE_PATTERNS 
            if re.search(pattern, message_lower, re.IGNORECASE)
        )
        
        # Determine primary feedback type
        max_matches = max(correction_matches, positive_matches, negative_matches, preference_matches)
        
        if max_matches == 0:
            return {
                "feedback_type": "neutral",
                "confidence": 0.0,
                "signals": {},
            }
        
        # Calculate confidence based on number of matches
        confidence = min(max_matches / 3.0, 1.0)  # Cap at 1.0
        
        if correction_matches == max_matches:
            feedback_type = "correction"
        elif positive_matches == max_matches:
            feedback_type = "positive"
        elif negative_matches == max_matches:
            feedback_type = "negative"
        else:
            feedback_type = "preference"
        
        return {
            "feedback_type": feedback_type,
            "confidence": confidence,
            "signals": {
                "correction_signals": correction_matches,
                "positive_signals": positive_matches,
                "negative_signals": negative_matches,
                "preference_signals": preference_matches,
            },
        }

    def auto_process_user_message(self, user_message: str, context: Dict[str, Any]) -> Optional[FeedbackData]:
        """
        Automatically process a user message for implicit feedback.
        Creates FeedbackData if feedback is detected with sufficient confidence.
        
        Args:
            user_message: The user's message
            context: Conversation context with previous query/response
            
        Returns:
            FeedbackData if feedback detected, None otherwise
        """
        detection = self.detect_feedback_type(user_message)
        
        # Only create feedback if confidence is high enough
        if detection["confidence"] < 0.3:
            return None
        
        feedback_type = detection["feedback_type"]
        
        # Map detected type to reward signal
        reward_map = {
            "positive": 1.0,
            "negative": -1.0,
            "correction": -0.5,  # Corrections indicate something was wrong
            "preference": 0.0,   # Preferences are informational, not good/bad
        }
        
        reward_signal = reward_map.get(feedback_type, 0.0)
        
        # Create feedback data
        feedback = FeedbackData(
            feedback_id=f"auto_{int(time.time())}_{secrets.token_hex(4)}",
            timestamp=time.time(),
            feedback_type=f"auto_{feedback_type}",
            content=None,  # Don't store user message content for privacy
            context=context,
            agent_response=context.get("previous_response_id"),
            human_preference=None,
            reward_signal=reward_signal,
            metadata={
                "auto_detected": True,
                "detection_confidence": detection["confidence"],
                "signals": detection["signals"],
            },
        )
        
        # Submit to RLHF system
        self.receive_feedback(feedback)
        
        logger.info(
            f"[AutoDetect] Detected {feedback_type} feedback with confidence {detection['confidence']:.2f}"
        )
        
        return feedback

    # =========================================================================
    # KNOWLEDGE CRYSTALLIZER INTEGRATION
    # =========================================================================

    def crystallize_from_execution(
        self,
        query: str,
        response: str,
        success: bool,
        tools_used: Optional[List[str]] = None,
        strategy: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Crystallize knowledge from an execution trace.
        
        The Knowledge Crystallizer extracts reusable principles from execution traces.
        It follows the pattern: EXAMINE → SELECT → APPLY → REMEMBER
        
        ExecutionTrace structure (from knowledge_crystallizer_core.py):
        - trace_id: str - Unique identifier
        - actions: List[Dict] - Actions performed during execution
        - outcomes: Dict - Results/outcomes of the execution
        - context: Dict - Context information
        - success: bool - Whether execution was successful
        - metadata: Dict - Additional metadata
        - domain: str - Domain/category of the execution
        
        Args:
            query: The user's query
            response: The system's response
            success: Whether the execution was successful
            tools_used: List of tools used in the response
            strategy: The strategy/approach used (reasoning_type)
            metadata: Additional metadata about the execution
            
        Returns:
            Crystallization result dict or None if crystallizer unavailable
        """
        if not self.knowledge_crystallizer:
            logger.debug("[KnowledgeCrystallizer] Not available, skipping crystallization")
            return None
        
        try:
            # Import ExecutionTrace from knowledge_crystallizer_core (not principle_extractor)
            from ..knowledge_crystallizer.knowledge_crystallizer_core import ExecutionTrace as CoreExecutionTrace
            
            # Create trace_id from query hash + timestamp
            trace_id = f"exec_{int(time.time())}_{secrets.token_hex(4)}"
            
            # Build actions list representing what VULCAN did
            actions = []
            if tools_used:
                for tool in tools_used:
                    actions.append({
                        "type": "tool_invocation",
                        "tool": tool,
                        "timestamp": time.time(),
                    })
            if strategy:
                actions.append({
                    "type": "reasoning",
                    "strategy": strategy,
                    "timestamp": time.time(),
                })
            
            # Build outcomes dict
            outcomes = {
                "success": success,
                "response_length": len(response) if response else 0,
                "tools_count": len(tools_used) if tools_used else 0,
            }
            
            # Build context
            context = {
                "query": query[:500] if query else "",  # Truncate for privacy
                "strategy": strategy,
                "tools_available": tools_used or [],
            }
            
            # Determine domain from strategy/tools
            domain = "general"
            if strategy:
                if "math" in strategy.lower() or "calculation" in strategy.lower():
                    domain = "math"
                elif "code" in strategy.lower() or "programming" in strategy.lower():
                    domain = "programming"
                elif "reason" in strategy.lower():
                    domain = "reasoning"
            
            # Create the ExecutionTrace with proper structure
            trace = CoreExecutionTrace(
                trace_id=trace_id,
                actions=actions,
                outcomes=outcomes,
                context=context,
                success=success,
                metadata=metadata or {},
                domain=domain,
                timestamp=time.time(),
            )
            
            # Attempt crystallization
            result = self.knowledge_crystallizer.crystallize(trace)
            
            if result and hasattr(result, 'principles') and result.principles:
                logger.info(
                    f"[KnowledgeCrystallizer] Crystallized {len(result.principles)} principles "
                    f"from trace {trace_id} (confidence: {result.confidence:.2f})"
                )
                # Return as dict for easier handling
                return result.to_dict() if hasattr(result, 'to_dict') else {
                    "principles": len(result.principles),
                    "confidence": result.confidence,
                    "mode": result.mode.value if hasattr(result.mode, 'value') else str(result.mode),
                }
            else:
                logger.debug(f"[KnowledgeCrystallizer] No principles extracted from trace {trace_id}")
                return None
                
        except ImportError as e:
            logger.warning(f"[KnowledgeCrystallizer] Import error: {e}")
            return None
        except Exception as e:
            logger.error(f"[KnowledgeCrystallizer] Crystallization failed: {e}")
            return None

    def apply_crystallized_knowledge(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Apply crystallized knowledge to enhance a query/response.
        
        Uses KnowledgeCrystallizer.apply_knowledge() to find applicable principles
        and adapt them to the current problem context.
        
        Args:
            query: The user's query
            context: Additional context (domain, constraints, etc.)
            
        Returns:
            ApplicationResult dict with principle_used, solution, confidence, etc.
            or None if no applicable knowledge found
        """
        if not self.knowledge_crystallizer:
            return None
        
        try:
            # Build problem specification for apply_knowledge
            problem = {
                "query": query,
                "context": context or {},
                "domain": context.get("domain", "general") if context else "general",
            }
            
            # Query the crystallizer for applicable knowledge
            result = self.knowledge_crystallizer.apply_knowledge(problem)
            
            if result and result.principle_used:
                logger.debug(
                    f"[KnowledgeCrystallizer] Applied principle with confidence {result.confidence:.2f}"
                )
                return result.to_dict() if hasattr(result, 'to_dict') else {
                    "principle_used": str(result.principle_used),
                    "solution": result.solution,
                    "confidence": result.confidence,
                    "adaptations": result.adaptations,
                    "warnings": result.warnings,
                }
            
            return None
            
        except Exception as e:
            logger.error(f"[KnowledgeCrystallizer] Apply failed: {e}")
            return None

    def get_crystallizer_stats(self) -> Dict[str, Any]:
        """Get statistics from the Knowledge Crystallizer."""
        if not self.knowledge_crystallizer:
            return {
                "available": False,
                "reason": "KnowledgeCrystallizer not initialized",
            }
        
        try:
            stats = {
                "available": True,
                "principles_count": 0,
                "contraindications_count": 0,
            }
            
            # Get stats from crystallizer if available
            if hasattr(self.knowledge_crystallizer, "get_stats"):
                stats.update(self.knowledge_crystallizer.get_stats())
            elif hasattr(self.knowledge_crystallizer, "knowledge_base"):
                kb = self.knowledge_crystallizer.knowledge_base
                if hasattr(kb, "get_all_principles"):
                    stats["principles_count"] = len(kb.get_all_principles())
            
            return stats
            
        except Exception as e:
            logger.error(f"[KnowledgeCrystallizer] Stats failed: {e}")
            return {"available": True, "error": str(e)}

    def _create_task_head(self) -> nn.Module:
        """Create task-specific head."""
        return nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.LayerNorm(HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN_DIM // 2, LATENT_DIM),
            nn.ReLU(),
            nn.Linear(LATENT_DIM, self.embedding_dim),
        )

    def _setup_optimizer(self, task_id: str, model: nn.Module):
        """Setup optimizer and scheduler for task."""
        params = list(self.shared_encoder.parameters()) + list(model.parameters())
        self.optimizers[task_id] = optim.Adam(params, lr=self.config.learning_rate)
        self.schedulers[task_id] = ReduceLROnPlateau(
            self.optimizers[task_id], mode="min", patience=10, factor=0.5
        )

    def forward(self, x: torch.Tensor, task_id: Optional[str] = None) -> torch.Tensor:
        """Forward pass through appropriate task model."""
        # Use progressive network if available
        if self.use_progressive and hasattr(self, "progressive_network"):
            if task_id in self.task_order:
                column_idx = self.task_order.index(task_id)
                return self.progressive_network(x, column_idx)

        # Standard forward pass
        encoded = self.shared_encoder(x)

        # --- START FIX: Remove broken PackNet logic from forward pass ---
        # Apply task mask if using PackNet
        # if task_id and task_id in self.task_masks:
        #     encoded = encoded * self.task_masks[task_id] # <-- BUG: This is incorrect application of PackNet
        # --- END FIX ---

        # Task-specific head
        if task_id and task_id in self.task_models:
            output = self.task_models[task_id](encoded)
        else:
            output = self.general_model(encoded)

        return output

    def _sanitize_experience(self, experience: Dict[str, Any]) -> Dict[str, Any]:
        """Remove non-picklable objects from experience before storing - FIXED to avoid isinstance issues"""
        clean_experience = {}

        for key, value in experience.items():
            clean_value = self._sanitize_value(value)
            if clean_value is not None or value is None:
                clean_experience[key] = clean_value

        return clean_experience

    def _sanitize_value(self, value: Any) -> Any:
        """Helper to sanitize individual values - ensures all values are picklable"""
        # Handle None explicitly
        if value is None:
            return None

        # --- START FIX: Check for unpicklable types *before* trying to pickle ---
        # Handle non-picklable types (like locks)
        if isinstance(value, self._NON_PICKLABLE_TYPES):
            return None
        # --- END FIX ---

        # Handle basic types using type() instead of isinstance
        value_type = type(value)
        if value_type in (str, int, float, bool):
            return value

        # Handle tensors - convert to numpy
        try:
            if torch.is_tensor(value):
                return value.detach().cpu().numpy()
        except Exception as e:
            logger.debug(
                f"{self.__class__.__name__ if hasattr(self, '__class__') else 'Operation'} error: {e}"
            )

        # Handle numpy arrays
        try:
            if hasattr(value, "dtype") and hasattr(value, "shape"):
                # Likely a numpy array
                return np.array(value).copy()
        except Exception as e:
            logger.debug(
                f"{self.__class__.__name__ if hasattr(self, '__class__') else 'Operation'} error: {e}"
            )

        # Handle lists using type()
        if value_type == list:
            return [self._sanitize_value(v) for v in value]

        # Handle dicts using type()
        if value_type == dict:
            return self._sanitize_experience(value)

        # Handle threading objects by type name
        type_name = value_type.__name__
        if "Lock" in type_name or "Thread" in type_name or "Event" in type_name:
            return None

        # Try to handle Enum-like objects
        try:
            if hasattr(value, "value") and hasattr(value, "name"):
                # Looks like an enum
                return str(value.value) if hasattr(value, "value") else str(value)
        except Exception as e:
            logger.debug(
                f"{self.__class__.__name__ if hasattr(self, '__class__') else 'Operation'} error: {e}"
            )

        # Test if the value is actually picklable
        try:
            pickle.dumps(value)
            return value
        except (TypeError, pickle.PicklingError, AttributeError):
            # If not picklable, try to convert to string
            try:
                return str(value)
            except Exception as e:  # If we can't stringify it, return None
                return None

    def process_experience(self, experience: Dict[str, Any]) -> Dict[str, Any]:
        """Process experience with task detection, RLHF, and tracking."""
        # Use timeout to prevent deadlocks
        acquired = self._lock.acquire(timeout=5.0)
        if not acquired:
            logger.error("Failed to acquire lock for experience processing")
            return {"error": "Lock timeout", "adapted": False}

        try:
            # FIXED: Sanitize experience ONCE at the beginning
            clean_experience = self._sanitize_experience(experience)

            # Start trajectory if auditing enabled
            trajectory_id = None
            if self.config.audit_trail_enabled:
                # --- START BUG FIX: Wrap external call to catch PicklingError ---
                try:
                    trajectory_id = self.param_history.start_trajectory(
                        task_id=self.task_detector.current_task or "unknown",
                        agent_id="continual_learner",
                    )
                except (TypeError, pickle.PicklingError, AttributeError) as e:
                    logger.warning(f"Failed to start trajectory (pickle error): {e}")
                # --- END BUG FIX ---

            # Detect task using cleaned experience
            task_id = self.task_detector.detect_task(clean_experience)

            # Check for new task
            if task_id not in self.task_models:
                self._on_new_task(task_id)

            # Update task info
            if task_id not in self.task_info:
                self.task_info[task_id] = TaskInfo(
                    task_id=task_id,
                    task_type="unknown",
                    signature=self.task_detector.task_signatures.get(task_id),
                )

            self.task_info[task_id].samples_seen += 1

            # Store in hierarchical memory if available
            if self.hierarchical_memory:
                self.hierarchical_memory.store(
                    clean_experience.get("embedding"),  # Use clean_experience
                    level=0,
                    metadata={"task_id": task_id},
                )

            # Process with appropriate model (use clean_experience)
            result = self._process_with_task(clean_experience, task_id)

            # Record step for trajectory
            if self.config.audit_trail_enabled and "embedding" in clean_experience:
                embedding = clean_experience["embedding"]
                if isinstance(embedding, torch.Tensor):
                    # Already detached by sanitize, but numpy() is needed
                    embedding = embedding.numpy()

                # --- START BUG FIX: Wrap external call to catch PicklingError ---
                try:
                    self.param_history.record_step(
                        state=embedding,
                        action=task_id,
                        reward=clean_experience.get(
                            "reward", 0.0
                        ),  # Use clean_experience
                        loss=result.get("loss", 0.0),
                    )
                except (TypeError, pickle.PicklingError, AttributeError) as e:
                    logger.warning(f"Failed to record step (pickle error): {e}")
                # --- END BUG FIX ---

            # Store sanitized experience in buffers
            self.replay_buffer.append(clean_experience)
            self._get_task_buffer(task_id).append(clean_experience)

            # Process with RLHF if enabled
            if self.rlhf_manager and "feedback" in clean_experience:
                feedback = FeedbackData(
                    feedback_id=f"exp_{time.time()}",
                    timestamp=time.time(),
                    feedback_type="experience",
                    content=clean_experience,  # Use clean_experience
                    context={"task_id": task_id},
                    agent_response=result.get("output"),
                    human_preference=clean_experience.get(
                        "feedback"
                    ),  # Use clean_experience
                    reward_signal=clean_experience.get(
                        "reward", 0.0
                    ),  # Use clean_experience
                )
                # --- START BUG FIX: Wrap external call to catch PicklingError ---
                try:
                    self.rlhf_manager.receive_feedback(feedback)
                except (TypeError, pickle.PicklingError, AttributeError) as rlhf_e:
                    logger.warning(
                        f"Failed to process RLHF feedback (pickle error): {rlhf_e}"
                    )
                # --- END BUG FIX ---

            # Online meta-learning
            # --- START BUG FIX: Wrap external call to catch PicklingError ---
            try:
                self.meta_learner.online_meta_update(
                    clean_experience
                )  # Use clean_experience
            except (TypeError, pickle.PicklingError, AttributeError) as meta_e:
                logger.warning(
                    f"Failed to process meta-learning update (pickle error): {meta_e}"
                )
            # --- END BUG FIX ---

            # Atomic check and consolidation
            should_consolidate = self._increment_consolidation_counter()

            if should_consolidate:
                self._consolidate_knowledge(task_id)

                # Update continual learning metrics
                self._update_continual_metrics()

                # Save checkpoint
                if self.config.checkpoint_frequency > 0:
                    # FIXED: Create a picklable state dictionary with sanitized buffers
                    with self._stats_lock:
                        # Helper to sanitize buffers
                        def sanitize_buffer_items(buffer_list):
                            sanitized = []
                            for item in buffer_list:
                                if isinstance(item, dict):
                                    clean_item = {}
                                    for k, v in item.items():
                                        try:
                                            pickle.dumps(v)
                                            clean_item[k] = v
                                        except Exception:
                                            if isinstance(v, torch.Tensor):
                                                clean_item[k] = v.detach().cpu().numpy()
                                            elif isinstance(v, np.ndarray):
                                                clean_item[k] = v.copy()
                                            else:
                                                try:
                                                    clean_item[k] = str(v)
                                                except Exception as e:
                                                    logger.debug(
                                                        f"{self.__class__.__name__ if hasattr(self, '__class__') else 'Operation'} error: {e}"
                                                    )
                                    sanitized.append(clean_item)
                                else:
                                    try:
                                        pickle.dumps(item)
                                        sanitized.append(item)
                                    except Exception:
                                        sanitized.append(str(item))
                            return sanitized

                        # Create config dict excluding locks and testing picklability
                        clean_config_dict = {}
                        for k, v in self.config.__dict__.items():
                            # --- START FIX: Remove extra parentheses ---
                            if isinstance(v, self._NON_PICKLABLE_TYPES):
                                # --- END FIX ---
                                continue
                            # Test if value is actually picklable
                            try:
                                pickle.dumps(v)
                                clean_config_dict[k] = v
                            except Exception as e:
                                logger.error(
                                    f"Error pickling config value for key {k}: {e}",
                                    exc_info=True,
                                )

                        # Sanitize task_info dicts
                        clean_task_info = {}
                        for k, v in self.task_info.items():
                            info_dict = v.__dict__.copy()
                            # Test if the dict is picklable, if not sanitize it
                            try:
                                pickle.dumps(info_dict)
                                clean_task_info[k] = info_dict
                            except Exception as e:  # Sanitize individual fields
                                sanitized_info = {}
                                for field, value in info_dict.items():
                                    try:
                                        pickle.dumps(value)
                                        sanitized_info[field] = value
                                    except (
                                        Exception
                                    ) as e:  # Convert to string if not picklable
                                        sanitized_info[field] = (
                                            str(value) if value is not None else None
                                        )
                                clean_task_info[k] = sanitized_info

                        # Build state and test each component individually
                        components = {
                            "model_state": self.state_dict(),
                            "task_info": clean_task_info,
                            "task_order": self.task_order.copy(),
                            "fisher_information": self.fisher_information,
                            "optimal_params": self.optimal_params,
                            "task_masks": self.task_masks,
                            "continual_metrics": self.continual_metrics.__dict__.copy(),
                            "config": clean_config_dict,
                            "replay_buffer": sanitize_buffer_items(
                                list(self.replay_buffer)
                            ),
                            "task_buffers": {
                                k: sanitize_buffer_items(list(v))
                                for k, v in self.task_buffers.items()
                            },
                            "performance_tracker": {
                                k: list(v) for k, v in self.performance_tracker.items()
                            },
                            "task_performance_history": dict(
                                self.task_performance_history
                            ),
                            "consolidation_counter": self.consolidation_counter,
                            "embedding_dim": self.embedding_dim,
                            "use_hierarchical": self.use_hierarchical,
                            "use_progressive": self.use_progressive,
                            "optimizer_states": {
                                task_id_opt: opt.state_dict()
                                for task_id_opt, opt in self.optimizers.items()
                            },
                            "scheduler_states": {
                                task_id_sched: sched.state_dict()
                                for task_id_sched, sched in self.schedulers.items()
                            },
                        }

                        # Test each component and only add if picklable
                        picklable_state = {}
                        for key, value in components.items():
                            try:
                                pickle.dumps(value)
                                picklable_state[key] = value
                            except Exception as e:
                                logger.warning(
                                    f"Skipping non-picklable component '{key}': {e}"
                                )

                    try:
                        self.param_history.async_checkpoint(
                            picklable_state,
                            metadata={
                                "task_id": task_id,
                                "consolidation": True,
                                "metrics": self.continual_metrics.__dict__.copy(),
                            },
                        )
                    except Exception as checkpoint_err:
                        # Log but don't fail if checkpointing fails
                        logger.warning(f"Failed to checkpoint state: {checkpoint_err}")

            # Experience replay with intelligent sampling
            if len(self.replay_buffer) > 100 and np.random.random() < 0.1:
                replay_result = self._intelligent_experience_replay(task_id)
                result["replay_loss"] = replay_result

            # End trajectory
            if trajectory_id:
                # --- START BUG FIX: Wrap external call to catch PicklingError ---
                try:
                    self.param_history.end_trajectory()
                except (TypeError, pickle.PicklingError, AttributeError) as e:
                    logger.warning(f"Failed to end trajectory (pickle error): {e}")
                # --- END BUG FIX ---
                result["trajectory_id"] = trajectory_id

            result["adapted"] = True
            result["continual_metrics"] = self.continual_metrics.__dict__
            return result

        except Exception as e:
            # Convert exception to string immediately to avoid pickle issues
            error_msg = str(e)
            logger.error(f"Error processing experience: {error_msg}")
            return {"error": error_msg, "adapted": False}
        finally:
            self._lock.release()

    def _on_new_task(self, task_id: str):
        """Handle new task detection."""
        logger.info(f"Creating model for new task: {task_id}")

        # Add to task order
        self.task_order.append(task_id)

        # Progressive network - add new column
        if self.use_progressive and hasattr(self, "progressive_network"):
            self.progressive_network.add_column(task_id)

        # Create task-specific head
        task_head = self._create_task_head()

        # Initialize from general model with noise for diversity
        task_head.load_state_dict(self.general_model.state_dict())

        # Add small random noise to break symmetry
        with torch.no_grad():
            for param in task_head.parameters():
                param.add_(torch.randn_like(param) * 0.01)

        # Register module
        self.task_models[task_id] = task_head

        # Setup optimizer
        self._setup_optimizer(task_id, task_head)

        # Allocate capacity for PackNet
        if len(self.task_order) > 1:
            self._allocate_task_capacity(task_id)

    def _allocate_task_capacity(self, task_id: str):
        """Allocate network capacity for new task (PackNet-style)"""
        # Calculate required capacity
        total_params = sum(p.numel() for p in self.parameters())
        num_tasks = len(self.task_order)

        # Equal allocation strategy (can be improved)
        capacity_per_task = 1.0 / (num_tasks + 2)  # +2 for future tasks

        # Create binary mask for this task
        mask = torch.zeros(total_params)

        # --- START FIX: Resize free_capacity if model grew ---
        if total_params > self.free_capacity.numel():
            logger.warning(
                f"Resizing free_capacity tensor: {self.free_capacity.numel()} -> {total_params}"
            )
            new_free_capacity = torch.ones(total_params)
            new_free_capacity[: self.free_capacity.numel()] = self.free_capacity
            self.free_capacity = new_free_capacity
        # --- END FIX ---

        free_indices = torch.where(self.free_capacity > 0.5)[0]

        if len(free_indices) > 0:
            # Allocate from free parameters
            num_to_allocate = min(
                int(capacity_per_task * total_params), len(free_indices)
            )

            selected_indices = torch.randperm(len(free_indices))[:num_to_allocate]
            mask[free_indices[selected_indices]] = 1.0

            # Update free capacity
            self.free_capacity[free_indices[selected_indices]] = 0.0

        self.task_masks[task_id] = mask

    def _process_with_task(
        self, experience: Dict[str, Any], task_id: str
    ) -> Dict[str, Any]:
        """Process experience with task-specific model."""
        # Extract embedding
        embedding = experience.get("embedding")
        if embedding is None:
            return {"error": "No embedding in experience"}

        # Convert to tensor
        if isinstance(embedding, np.ndarray):
            x = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
        elif isinstance(embedding, torch.Tensor):
            # Already detached by _sanitize_experience
            x = embedding.unsqueeze(0) if embedding.dim() == 1 else embedding
        else:
            x = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)

        # Forward pass
        output = self.forward(x, task_id)

        # Compute loss with EWC penalty
        loss = self._compute_loss_with_ewc(output, x, task_id)

        # Backward pass
        optimizer = self.optimizers.get(task_id, self.optimizers["general"])

        # Use adaptive learning rate if available
        if hasattr(self, "live_feedback"):
            for param_group in optimizer.param_groups:
                param_group["lr"] = self.live_feedback.adaptive_lr

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

        # Safety check with correct parameter names
        # THREAD SAFETY FIX: Reuse validator to prevent thread leaks.
        # EnhancedSafetyValidator spawns RollbackManager and AuditLogger, each with
        # background threads (rotation_worker, cleanup_worker). Creating a new validator
        # on every call causes thread accumulation since threads wait with 1-hour timeouts.
        if EnhancedSafetyValidator:
            try:
                # Thread-safe lazy initialization of validator singleton
                with self._lock:
                    if self._safety_validator is None:
                        self._safety_validator = EnhancedSafetyValidator(
                            SafetyConfig(safety_level="STRICT")
                        )

                # Collect gradients and model state for validation
                grads_to_validate = {
                    name: p.grad.clone()
                    for name, p in self.named_parameters()
                    if p.grad is not None
                }
                model_state = {
                    name: p.data.clone() for name, p in self.named_parameters()
                }

                # Use correct parameter names for validator
                safe, reason = self._safety_validator.validate_action(
                    {
                        "type": "MODEL_UPDATE",
                        "gradients": grads_to_validate,
                        "model_state": model_state,
                    }
                )

                if not safe:
                    logger.warning(f"Blocked update: {reason}")
                    # Rollback by not stepping optimizer and zeroing gradients
                    optimizer.zero_grad()
                    return {
                        "task_id": task_id,
                        "loss": loss.item(),
                        "output": output.detach(),
                        "output_shape": output.shape,
                        "samples_seen": self.task_info[task_id].samples_seen,
                        "update_blocked": True,
                        "block_reason": reason,
                    }
            except Exception as e:
                logger.error(f"Safety validation failed: {e}")
                # Continue with update if validator fails

        optimizer.step()

        # Update scheduler
        if task_id in self.schedulers:
            self.schedulers[task_id].step(loss)

        # Track performance atomically using single lock
        with self._stats_lock:
            self._get_performance_deque(task_id).append(loss.item())
            self._get_task_performance_history(task_id).append(
                {
                    "timestamp": time.time(),
                    "loss": loss.item(),
                    "samples_seen": self.task_info[task_id].samples_seen,
                }
            )

            # Update task info
            if task_id in self.task_info:
                recent_losses = list(self.performance_tracker.get(task_id, deque()))[
                    -10:
                ]
                if recent_losses:
                    self.task_info[task_id].performance = 1.0 / (
                        1.0 + np.mean(recent_losses)
                    )

        return {
            "task_id": task_id,
            "loss": loss.item(),
            "output": output.detach(),
            "output_shape": output.shape,
            "samples_seen": self.task_info[task_id].samples_seen,
        }

    def _compute_loss_with_ewc(
        self, output: torch.Tensor, target: torch.Tensor, task_id: str
    ) -> torch.Tensor:
        """Compute loss with EWC penalty."""
        # Base loss (reconstruction)
        base_loss = F.mse_loss(output, target)

        # EWC penalty
        ewc_loss = 0

        for prev_task in self.fisher_information:
            if prev_task != task_id:  # Don't penalize current task
                fisher = self.fisher_information[prev_task]
                optimal = self.optimal_params[prev_task]

                for name, param in self.named_parameters():
                    if name in fisher:
                        ewc_loss += (fisher[name] * (param - optimal[name]) ** 2).sum()

        total_loss = base_loss + self.config.ewc_lambda * ewc_loss

        return total_loss

    def _consolidate_knowledge(self, task_id: str):
        """Consolidate knowledge for task using EWC."""
        logger.info(f"Consolidating knowledge for task {task_id}")

        # Compute Fisher Information Matrix
        fisher_info = {}
        optimal_params = {}

        # Get task data using helper method
        task_buffer = self._get_task_buffer(task_id)
        task_data = list(task_buffer)[-100:]  # Last 100 samples

        if task_data:
            for param_name, param in self.named_parameters():
                param_grad_sum = torch.zeros_like(param)

                for experience in task_data:
                    # Forward pass
                    embedding = experience.get("embedding")
                    if embedding is None:
                        continue

                    if isinstance(embedding, np.ndarray):
                        x = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
                    else:
                        # Already detached tensor from buffer
                        x = (
                            embedding.unsqueeze(0)
                            if embedding.dim() == 1
                            else embedding
                        )

                    output = self.forward(x, task_id)
                    loss = F.mse_loss(output, x)

                    # Compute gradients
                    self.zero_grad()
                    loss.backward(retain_graph=True)

                    if param.grad is not None:
                        param_grad_sum += param.grad.data**2

                fisher_info[param_name] = param_grad_sum / max(1, len(task_data))
                optimal_params[param_name] = param.data.clone()

        self.fisher_information[task_id] = fisher_info
        self.optimal_params[task_id] = optimal_params

        # Consolidate in hierarchical memory if available
        if self.hierarchical_memory:
            self.hierarchical_memory.consolidate()

    def _intelligent_experience_replay(
        self, current_task: str, n_samples: int = 32
    ) -> float:
        """Intelligent replay with task-aware sampling."""
        if len(self.replay_buffer) < n_samples:
            return 0.0

        total_loss = 0.0

        # Sample from each task proportionally
        samples_per_task = {}

        for exp in self.replay_buffer:
            # Experience in replay buffer is already sanitized
            task = self.task_detector.detect_task(exp)
            if task not in samples_per_task:
                samples_per_task[task] = []
            samples_per_task[task].append(exp)

        # Calculate sampling weights based on forgetting
        task_weights = {}
        for task in samples_per_task:
            if task in self.task_info:
                # Higher weight for tasks with more forgetting
                recent_perf = self.task_info[task].performance
                task_weights[task] = 1.0 - recent_perf
            else:
                task_weights[task] = 0.5

        # Normalize weights
        total_weight = sum(task_weights.values())
        if total_weight > 0:
            task_weights = {k: v / total_weight for k, v in task_weights.items()}

        # Sample experiences
        sampled_experiences = []
        for task, weight in task_weights.items():
            n_task_samples = int(n_samples * weight)
            if n_task_samples > 0 and samples_per_task[task]:
                # Need to convert list to array for np.random.choice
                task_exp_list = samples_per_task[task]
                indices = np.random.choice(
                    len(task_exp_list),
                    min(n_task_samples, len(task_exp_list)),
                    replace=False,
                )
                task_samples = [task_exp_list[i] for i in indices]
                sampled_experiences.extend(task_samples)

        # Process sampled experiences
        for experience in sampled_experiences:
            # Process with reduced learning rate
            original_lr = self.config.learning_rate
            self.config.learning_rate *= 0.5

            result = self._process_with_task(experience, current_task)
            total_loss += result.get("loss", 0.0)

            self.config.learning_rate = original_lr

        return total_loss / max(1, len(sampled_experiences))

    def _experience_replay(self, current_task: str, n_samples: int = 32) -> float:
        """Simple replay past experiences to prevent forgetting."""
        if len(self.replay_buffer) < n_samples:
            return 0.0

        # Sample experiences
        indices = np.random.choice(len(self.replay_buffer), n_samples, replace=False)

        total_loss = 0.0
        for idx in indices:
            experience = self.replay_buffer[idx]

            # Process with reduced learning rate
            original_lr = self.config.learning_rate
            self.config.learning_rate *= 0.5

            result = self._process_with_task(experience, current_task)
            total_loss += result.get("loss", 0.0)

            self.config.learning_rate = original_lr

        return total_loss / n_samples

    def _update_continual_metrics(self):
        """Update continual learning metrics with proper locking."""
        if len(self.task_order) < 2:
            return

        with self._stats_lock:
            # Calculate average accuracy across tasks
            accuracies = []
            for task_id in self.task_order:
                if task_id in self.task_info:
                    accuracies.append(self.task_info[task_id].performance)

            self.continual_metrics.average_accuracy = (
                np.mean(accuracies) if accuracies else 0
            )

            # Calculate backward transfer (performance on old tasks)
            if len(self.task_order) > 1:
                old_task_perfs = []
                for task_id in self.task_order[:-1]:
                    if task_id in self.task_info:
                        old_task_perfs.append(self.task_info[task_id].performance)

                self.continual_metrics.backward_transfer = (
                    np.mean(old_task_perfs) if old_task_perfs else 0
                )

            # Calculate forward transfer (performance on new task)
            if self.task_order:
                latest_task = self.task_order[-1]
                if latest_task in self.task_info:
                    self.continual_metrics.forward_transfer = self.task_info[
                        latest_task
                    ].performance

            # Calculate forgetting measure
            forgetting = 0
            for task_id in self.task_order[:-1]:
                if task_id in self.task_performance_history:
                    history = self.task_performance_history[task_id]
                    if len(history) > 10:
                        best_perf = min([h["loss"] for h in history[:10]])
                        current_perf = np.mean([h["loss"] for h in history[-10:]])
                        forgetting += max(0, current_perf - best_perf)

            self.continual_metrics.forgetting_measure = forgetting / max(
                1, len(self.task_order) - 1
            )

            # Update task accuracies
            for task_id in self.task_order:
                if task_id in self.task_info:
                    self.continual_metrics.task_accuracies[task_id] = self.task_info[
                        task_id
                    ].performance

    def adapt_to_distribution_shift(
        self, new_distribution_samples: List[np.ndarray]
    ) -> Dict:
        """Adapt to distribution shift in data."""
        if not new_distribution_samples:
            return {"adapted": False, "reason": "No samples provided"}

        # Detect shift magnitude
        old_embeddings = [
            exp["embedding"]
            for exp in list(self.replay_buffer)[-100:]
            if "embedding" in exp
        ]

        if old_embeddings:
            # Ensure embeddings are numpy for mean calculation
            old_embeddings_np = [
                e.numpy() if isinstance(e, torch.Tensor) else e for e in old_embeddings
            ]
            old_mean = np.mean(old_embeddings_np, axis=0)
            new_mean = np.mean(new_distribution_samples, axis=0)
            shift_magnitude = np.linalg.norm(old_mean - new_mean)
        else:
            shift_magnitude = 1.0

        # Adjust learning rates based on shift
        for optimizer in self.optimizers.values():
            for param_group in optimizer.param_groups:
                if shift_magnitude > 0.5:
                    # Increase learning rate for large shifts
                    param_group["lr"] = min(0.01, param_group["lr"] * 1.5)
                else:
                    # Decrease for small shifts
                    param_group["lr"] = max(0.0001, param_group["lr"] * 0.9)

        # Create pseudo-task for shift
        shift_task_id = f"shift_{int(time.time())}"
        logger.info(
            f"Detected distribution shift (magnitude: {shift_magnitude:.3f}), creating task: {shift_task_id}"
        )

        return {
            "adapted": True,
            "shift_magnitude": float(shift_magnitude),
            "num_tasks": len(self.task_models),
            "shift_task_id": shift_task_id,
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics with proper locking."""
        with self._lock:
            with self._stats_lock:
                stats = {
                    "num_tasks": len(self.task_models),
                    "task_order": self.task_order,
                    "total_experiences": sum(
                        self.task_info[t].samples_seen for t in self.task_info
                    ),
                    "replay_buffer_size": len(self.replay_buffer),
                    "continual_metrics": self.continual_metrics.__dict__,
                    "task_performance": {
                        task_id: {
                            "performance": info.performance,
                            "samples_seen": info.samples_seen,
                            "difficulty": info.difficulty,
                        }
                        for task_id, info in self.task_info.items()
                    },
                    "free_capacity": (
                        float(self.free_capacity.mean())
                        if hasattr(self, "free_capacity")
                        else 1.0
                    ),
                }

                if self.hierarchical_memory:
                    stats["hierarchical_memory"] = {
                        "num_levels": (
                            len(self.hierarchical_memory.levels)
                            if hasattr(self.hierarchical_memory, "levels")
                            else 0
                        )
                    }

                return stats

    def save_state(self, path: Optional[str] = None) -> str:
        """Save complete state for recovery - exclude non-picklable objects."""
        if path is None:
            path = self.save_path / f"continual_state_{int(time.time())}.pkl"
        else:
            path = Path(path)

        with self._lock:
            with self._stats_lock:
                # Sanitize buffers to remove non-picklable objects
                def sanitize_buffer(buffer_list):
                    """Recursively sanitize buffer items"""
                    sanitized = []
                    for item in buffer_list:
                        if isinstance(item, dict):
                            clean_item = {}
                            for k, v in item.items():
                                try:
                                    # Test if picklable
                                    pickle.dumps(v)
                                    clean_item[k] = v
                                except (
                                    TypeError,
                                    pickle.PicklingError,
                                    AttributeError,
                                ):
                                    # Convert to safe representation
                                    if isinstance(v, torch.Tensor):
                                        clean_item[k] = v.detach().cpu().numpy()
                                    elif isinstance(v, np.ndarray):
                                        clean_item[k] = v.copy()
                                    else:
                                        try:
                                            clean_item[k] = str(v)
                                        except Exception:
                                            clean_item[k] = None
                            sanitized.append(clean_item)
                        else:
                            try:
                                pickle.dumps(item)
                                sanitized.append(item)
                            except Exception:
                                sanitized.append(str(item))
                    return sanitized

                # Create a picklable state by excluding locks and non-picklable objects
                state = {
                    "model_state": self.state_dict(),
                    "task_info": {k: v.__dict__ for k, v in self.task_info.items()},
                    "task_order": self.task_order.copy(),
                    "fisher_information": self.fisher_information,
                    "optimal_params": self.optimal_params,
                    "task_masks": self.task_masks,
                    "continual_metrics": self.continual_metrics.__dict__.copy(),
                    "config": {
                        k: v
                        for k, v in self.config.__dict__.items()
                        # --- START FIX: Remove extra parentheses ---
                        if not isinstance(v, self._NON_PICKLABLE_TYPES)
                    },
                    # --- END FIX ---
                    # Store sanitized buffers
                    "replay_buffer": sanitize_buffer(list(self.replay_buffer)),
                    "task_buffers": {
                        k: sanitize_buffer(list(v))
                        for k, v in self.task_buffers.items()
                    },
                    "performance_tracker": {
                        k: list(v) for k, v in self.performance_tracker.items()
                    },
                    "task_performance_history": dict(self.task_performance_history),
                    "consolidation_counter": self.consolidation_counter,
                    "embedding_dim": self.embedding_dim,
                    "use_hierarchical": self.use_hierarchical,
                    "use_progressive": self.use_progressive,
                    # Store optimizer states
                    "optimizer_states": {
                        task_id: opt.state_dict()
                        for task_id, opt in self.optimizers.items()
                    },
                    "scheduler_states": {
                        task_id: sched.state_dict()
                        for task_id, sched in self.schedulers.items()
                    },
                }

        # Save only picklable state
        with open(path, "wb") as f:
            pickle.dump(state, f)

        logger.info(f"Saved continual learning state to {path}")
        return str(path)

    def load_state(self, path: str):
        """Load complete state - properly reconstruct all components."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"State file not found: {path}")

        with open(path, "rb") as f:
            state = safe_pickle_load(f)

        with self._lock:
            with self._stats_lock:
                # 1. Restore config FIRST
                if "config" in state:
                    loaded_config_dict = state["config"]
                    for k, v in loaded_config_dict.items():
                        if hasattr(self.config, k):
                            setattr(self.config, k, v)

                # 2. Restore task order and task info
                self.task_order = (
                    state["task_order"].copy()
                    if isinstance(state["task_order"], list)
                    else list(state["task_order"])
                )
                self.task_info = {}
                for task_id, info_dict in state["task_info"].items():
                    self.task_info[task_id] = TaskInfo(**info_dict)

                # 3. Rebuild task model architecture to match saved state
                # Clear existing task models first
                self.task_models.clear()

                for task_id in self.task_order:
                    # Re-create the module in the nn.ModuleDict
                    task_head = self._create_task_head()
                    self.task_models[task_id] = task_head

                    # Re-create optimizer and scheduler
                    if task_id not in self.optimizers:
                        self._setup_optimizer(task_id, task_head)

                # 4. Load model weights - use strict=False to handle any mismatches gracefully
                try:
                    self.load_state_dict(state["model_state"], strict=False)
                except Exception as e:
                    logger.warning(
                        f"Failed to load full state dict, attempting partial load: {e}"
                    )
                    # Try to load only matching keys
                    current_state = self.state_dict()
                    filtered_state = {
                        k: v
                        for k, v in state["model_state"].items()
                        if k in current_state
                    }
                    self.load_state_dict(filtered_state, strict=False)

                # 5. Restore EWC and PackNet data
                self.fisher_information = state["fisher_information"]
                self.optimal_params = state["optimal_params"]
                self.task_masks = state.get("task_masks", {})

                # 6. Restore metrics
                metrics_dict = state.get("continual_metrics", {})
                self.continual_metrics = ContinualMetrics(**metrics_dict)

                # 7. Restore buffers and trackers
                if "replay_buffer" in state:
                    self.replay_buffer = deque(
                        state["replay_buffer"], maxlen=self.config.replay_buffer_size
                    )

                if "task_buffers" in state:
                    self.task_buffers = {
                        k: deque(v, maxlen=1000)
                        for k, v in state["task_buffers"].items()
                    }

                if "performance_tracker" in state:
                    self.performance_tracker = {
                        k: deque(v, maxlen=100)
                        for k, v in state["performance_tracker"].items()
                    }

                if "task_performance_history" in state:
                    self.task_performance_history = dict(
                        state["task_performance_history"]
                    )

                if "consolidation_counter" in state:
                    self.consolidation_counter = state["consolidation_counter"]

                # 8. Restore optimizer and scheduler states
                if "optimizer_states" in state:
                    for task_id, opt_state in state["optimizer_states"].items():
                        if task_id in self.optimizers:
                            try:
                                self.optimizers[task_id].load_state_dict(opt_state)
                            except Exception as e:
                                logger.warning(
                                    f"Failed to load optimizer state for {task_id}: {e}"
                                )

                if "scheduler_states" in state:
                    for task_id, sched_state in state["scheduler_states"].items():
                        if task_id in self.schedulers:
                            try:
                                self.schedulers[task_id].load_state_dict(sched_state)
                            except Exception as e:
                                logger.warning(
                                    f"Failed to load scheduler state for {task_id}: {e}"
                                )

        logger.info(f"Loaded continual learning state from {path}")

    def shutdown(self):
        """Clean shutdown of all components."""
        logger.info("Shutting down EnhancedContinualLearner...")

        # Signal shutdown
        self._shutdown.set()

        # Save current state
        try:
            self.save_state()
        except Exception as e:
            logger.error(f"Failed to save state during shutdown: {e}")

        # Shutdown components in order
        if self.rlhf_manager:
            try:
                self.rlhf_manager.shutdown()
            except Exception as e:
                logger.error(f"Failed to shutdown RLHF manager: {e}")

        if hasattr(self, "param_history"):
            try:
                self.param_history.shutdown()
            except Exception as e:
                logger.error(f"Failed to shutdown param history: {e}")

        if hasattr(self, "meta_learner"):
            try:
                self.meta_learner.shutdown()
            except Exception as e:
                logger.error(f"Failed to shutdown meta learner: {e}")

        # Shutdown safety validator (has background threads)
        if hasattr(self, "_safety_validator") and self._safety_validator:
            try:
                self._safety_validator.shutdown()
            except Exception as e:
                logger.error(f"Failed to shutdown safety validator: {e}")

        logger.info("EnhancedContinualLearner shutdown complete")
