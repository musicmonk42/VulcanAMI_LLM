# ============================================================
# VULCAN-AGI Telemetry Recorder - Dual-Mode Learning Telemetry
# ============================================================
# Enterprise-grade telemetry recording with dual-mode learning support:
# - MODE 1: User interaction telemetry (queries, feedback, quality)
# - MODE 2: AI-to-AI interaction telemetry (collaborations, tournaments)
#
# PRODUCTION-READY: Thread-safe, buffered writes, auto-flush
# META-LEARNING: Populates llm_meta_state.json for experiment triggers
# MEMORY SYSTEMS: Updates success/risk/utility/cost memories
#
# PERFORMANCE FIX: Memory updates are now buffered and flushed periodically
# instead of writing to disk on every request, preventing progressive slowdown.
# ============================================================

"""
VULCAN Telemetry Recorder with Dual-Mode Learning Support

Records interaction telemetry for the meta-learning system supporting:

MODE 1: User Interactions
    - Query patterns and types
    - Response quality metrics
    - User feedback (implicit/explicit)
    - Utility memory population

MODE 2: AI Interactions
    - Agent collaborations
    - Tournament results
    - Evolution outcomes
    - Inter-agent communication patterns
    - Success/risk memory population

Storage:
    - data/llm_meta_state.json: Main telemetry storage
    - memory.success_memory: AI performance data
    - memory.risk_memory: Error and risk tracking
    - memory.utility_memory: User interaction patterns
    - memory.cost_memory: Resource usage tracking

Thread Safety:
    All public methods are thread-safe. The TelemetryRecorder uses
    buffered writes with automatic flushing to minimize I/O overhead.

Usage:
    from vulcan.routing import record_telemetry, record_ai_interaction

    # Record user interaction
    record_telemetry(query, response, metadata, source="user")

    # Record AI-to-AI interaction
    record_ai_interaction(
        interaction_type="agent_communication",
        sender="perception",
        receiver="reasoning"
    )
"""

from __future__ import annotations

import atexit
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

# Initialize logger immediately after imports
logger = logging.getLogger(__name__)

# ============================================================
# CONSTANTS
# ============================================================

# Default paths for telemetry storage
DEFAULT_META_STATE_PATH = Path("data/llm_meta_state.json")

# Buffer configuration
TELEMETRY_BUFFER_SIZE = 100
TELEMETRY_FLUSH_INTERVAL = 60  # seconds
AI_INTERACTION_BUFFER_SIZE = 100

# PERFORMANCE FIX: Memory update buffer size
# Memory updates are now buffered instead of written on every request
MEMORY_UPDATE_BUFFER_SIZE = 50

# Thresholds for experiment triggers
USER_INTERACTION_EXPERIMENT_THRESHOLD = 100
AI_INTERACTION_EXPERIMENT_THRESHOLD = 50

# Memory limits
MAX_MEMORY_PATTERNS = 100
MAX_TELEMETRY_ENTRIES = 10000
MAX_AI_INTERACTION_ENTRIES = 10000


# ============================================================
# ENUMS
# ============================================================


class InteractionSource(str, Enum):
    """Source of interaction for dual-mode learning."""

    USER = "user"
    AGENT = "agent"
    ARENA = "arena"


# ============================================================
# DATA CLASSES
# ============================================================


@dataclass
class TelemetryEntry:
    """
    Single telemetry entry for meta-learning with dual-mode support.

    Attributes:
        timestamp: Entry creation timestamp
        query_id: Unique query identifier
        session_id: Session identifier
        source: Interaction source (user/agent/arena)
        query_type: Classified query type
        query_length: Length of query in characters
        response_length: Length of response in characters
        latency_ms: Response latency in milliseconds
        agent_tasks_submitted: Number of tasks submitted to agent pool
        agent_tasks_completed: Number of tasks completed
        governance_triggered: Whether governance was triggered
        experiment_triggered: Whether an experiment was triggered
        user_feedback: User feedback if provided
        response_quality_score: Quality score if available (0.0-1.0)
        error_occurred: Whether an error occurred
        error_type: Type of error if any
        collaboration_session_id: ID of collaboration session if applicable
        agents_involved: List of agents involved
        interaction_type: Type of interaction
        metadata: Additional metadata
    """

    timestamp: float
    query_id: str
    session_id: Optional[str]
    source: str
    query_type: str
    query_length: int
    response_length: int
    latency_ms: float
    agent_tasks_submitted: int
    agent_tasks_completed: int
    governance_triggered: bool
    experiment_triggered: bool
    user_feedback: Optional[str] = None
    response_quality_score: Optional[float] = None
    error_occurred: bool = False
    error_type: Optional[str] = None
    collaboration_session_id: Optional[str] = None
    agents_involved: List[str] = field(default_factory=list)
    interaction_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "query_id": self.query_id,
            "session_id": self.session_id,
            "source": self.source,
            "query_type": self.query_type,
            "query_length": self.query_length,
            "response_length": self.response_length,
            "latency_ms": self.latency_ms,
            "agent_tasks": {
                "submitted": self.agent_tasks_submitted,
                "completed": self.agent_tasks_completed,
            },
            "governance_triggered": self.governance_triggered,
            "experiment_triggered": self.experiment_triggered,
            "user_feedback": self.user_feedback,
            "response_quality_score": self.response_quality_score,
            "error_occurred": self.error_occurred,
            "error_type": self.error_type,
            "collaboration_session_id": self.collaboration_session_id,
            "agents_involved": self.agents_involved,
            "interaction_type": self.interaction_type,
            "meta": self.metadata,
        }


@dataclass
class AIInteractionEntry:
    """
    Entry specifically for AI-to-AI interactions.

    Attributes:
        timestamp: Entry creation timestamp
        interaction_id: Unique interaction identifier
        interaction_type: Type of interaction
        sender: Sending agent/system
        receiver: Receiving agent/system
        session_id: Collaboration session ID
        message_type: Type of message if applicable
        query: Original query context
        result: Interaction result if available
        metadata: Additional metadata
    """

    timestamp: float
    interaction_id: str
    interaction_type: str
    sender: str
    receiver: str
    session_id: Optional[str]
    message_type: Optional[str]
    query: str
    result: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "interaction_id": self.interaction_id,
            "interaction_type": self.interaction_type,
            "sender": self.sender,
            "receiver": self.receiver,
            "session_id": self.session_id,
            "message_type": self.message_type,
            "query": self.query[:500] if self.query else None,
            "result": self.result,
            "meta": self.metadata,
        }


@dataclass
class MemoryUpdateEntry:
    """
    PERFORMANCE FIX: Buffered memory update entry.
    
    Instead of writing to disk on every request, memory updates are
    buffered and flushed periodically with other telemetry data.
    
    Attributes:
        timestamp: Update timestamp
        source: Interaction source (user/agent/arena)
        query_type: Type of query
        quality_score: Response quality score if available
        error_occurred: Whether an error occurred
        error_type: Type of error if any
        interaction_type: Type of AI interaction if applicable
    """
    
    timestamp: float
    source: str
    query_type: str
    quality_score: Optional[float] = None
    error_occurred: bool = False
    error_type: Optional[str] = None
    interaction_type: Optional[str] = None


# ============================================================
# TELEMETRY RECORDER CLASS
# ============================================================


class TelemetryRecorder:
    """
    Records interaction telemetry for the meta-learning system with dual-mode support.

    Supports both user interactions and AI-to-AI interactions for comprehensive
    meta-learning data collection. Uses buffered writes with automatic flushing
    to minimize I/O overhead.

    Thread-safe implementation suitable for production use.

    Usage:
        recorder = TelemetryRecorder()

        # Record user interaction
        recorder.record(
            query="Analyze this pattern",
            response="The pattern shows...",
            metadata={"query_type": "perception"},
            source="user"
        )

        # Record AI interaction
        recorder.record_ai_interaction(
            interaction_type="agent_communication",
            sender="perception",
            receiver="reasoning"
        )

        # Get statistics
        print(recorder.get_stats())
    """

    def __init__(
        self,
        meta_state_path: Optional[Path] = None,
        buffer_size: int = TELEMETRY_BUFFER_SIZE,
        ai_buffer_size: int = AI_INTERACTION_BUFFER_SIZE,
        memory_update_buffer_size: int = MEMORY_UPDATE_BUFFER_SIZE,
        flush_interval: float = TELEMETRY_FLUSH_INTERVAL,
        auto_flush: bool = True,
    ):
        """
        Initialize the telemetry recorder.

        Args:
            meta_state_path: Path to llm_meta_state.json
            buffer_size: Number of telemetry entries to buffer before flush
            ai_buffer_size: Number of AI interaction entries to buffer
            memory_update_buffer_size: Number of memory updates to buffer before flush
            flush_interval: Seconds between automatic flushes
            auto_flush: Whether to automatically flush on interval
        """
        self._meta_state_path = meta_state_path or DEFAULT_META_STATE_PATH
        self._buffer_size = buffer_size
        self._ai_buffer_size = ai_buffer_size
        self._memory_update_buffer_size = memory_update_buffer_size
        self._flush_interval = flush_interval

        # Buffers for batched writes
        self._buffer: List[TelemetryEntry] = []
        self._ai_interaction_buffer: List[AIInteractionEntry] = []
        
        # PERFORMANCE FIX: Buffer for memory updates
        # Memory updates are now buffered instead of written on every request
        self._memory_update_buffer: List[MemoryUpdateEntry] = []
        self._memory_update_lock = threading.RLock()

        # Thread safety
        self._lock = threading.RLock()

        # Counters
        self._total_entries = 0
        self._session_entries = 0

        # Dual-mode statistics
        self._stats = {
            # General stats
            "total_queries": 0,
            "total_latency_ms": 0.0,
            "governance_triggers": 0,
            "experiment_triggers": 0,
            "errors": 0,
            "positive_feedback": 0,
            "negative_feedback": 0,
            # User interaction stats
            "user_interactions": 0,
            "user_queries": 0,
            # AI interaction stats
            "ai_interactions": 0,
            "agent_collaborations": 0,
            "tournaments": 0,
            "debates": 0,
            "votes": 0,
            # PERFORMANCE FIX: Memory update stats
            "memory_updates_queued": 0,
            "memory_updates_flushed": 0,
        }

        # Auto-flush configuration
        self._auto_flush = auto_flush
        self._shutdown_event = threading.Event()
        self._flush_thread: Optional[threading.Thread] = None

        if auto_flush:
            self._start_auto_flush()

        # Register shutdown handler
        atexit.register(self._atexit_handler)

        logger.debug(f"TelemetryRecorder initialized, path: {self._meta_state_path}")

    def _atexit_handler(self) -> None:
        """Handle graceful shutdown on process exit."""
        try:
            self.shutdown()
        except Exception:
            pass  # Ignore errors during shutdown

    def _start_auto_flush(self) -> None:
        """Start the auto-flush background thread."""
        self._flush_thread = threading.Thread(
            target=self._auto_flush_loop, daemon=True, name="TelemetryFlush"
        )
        self._flush_thread.start()

    def _auto_flush_loop(self) -> None:
        """Background loop for automatic flushing."""
        while not self._shutdown_event.wait(timeout=self._flush_interval):
            try:
                self.flush()
            except Exception as e:
                logger.error(f"[TelemetryRecorder] Auto-flush error: {e}")

    def record(
        self,
        query: str,
        response: Optional[str],
        metadata: Dict[str, Any],
        query_id: Optional[str] = None,
        session_id: Optional[str] = None,
        source: Literal["user", "agent", "arena"] = "user",
        latency_ms: float = 0.0,
        agent_tasks_submitted: int = 0,
        agent_tasks_completed: int = 0,
        governance_triggered: bool = False,
        experiment_triggered: bool = False,
        error_occurred: bool = False,
        error_type: Optional[str] = None,
        collaboration_session_id: Optional[str] = None,
        agents_involved: Optional[List[str]] = None,
        interaction_type: Optional[str] = None,
    ) -> None:
        """
        Record interaction telemetry for meta-learning (dual-mode).

        Args:
            query: The query text
            response: The response text (can be None for errors)
            metadata: Additional metadata from QueryPlan
            query_id: Unique query identifier
            session_id: Session identifier
            source: "user", "agent", or "arena" (determines learning mode)
            latency_ms: Response latency in milliseconds
            agent_tasks_submitted: Number of tasks submitted to agent pool
            agent_tasks_completed: Number of tasks completed
            governance_triggered: Whether governance was triggered
            experiment_triggered: Whether an experiment was triggered
            error_occurred: Whether an error occurred
            error_type: Type of error if any
            collaboration_session_id: ID of collaboration session if applicable
            agents_involved: List of agents involved if collaboration
            interaction_type: Type of AI interaction if applicable
        """
        import uuid

        entry = TelemetryEntry(
            timestamp=time.time(),
            query_id=query_id or f"q_{uuid.uuid4().hex[:12]}",
            session_id=session_id,
            source=source,
            query_type=metadata.get("query_type", "unknown"),
            query_length=len(query) if query else 0,
            response_length=len(response) if response else 0,
            latency_ms=latency_ms,
            agent_tasks_submitted=agent_tasks_submitted,
            agent_tasks_completed=agent_tasks_completed,
            governance_triggered=governance_triggered,
            experiment_triggered=experiment_triggered,
            error_occurred=error_occurred,
            error_type=error_type,
            collaboration_session_id=collaboration_session_id,
            agents_involved=agents_involved or [],
            interaction_type=interaction_type,
            metadata=metadata,
        )

        with self._lock:
            self._buffer.append(entry)
            self._total_entries += 1
            self._session_entries += 1

            # Update stats based on source (dual-mode)
            self._stats["total_queries"] += 1
            self._stats["total_latency_ms"] += latency_ms

            if source == "user":
                self._stats["user_interactions"] += 1
                self._stats["user_queries"] += 1
            else:
                self._stats["ai_interactions"] += 1
                if interaction_type == "collaboration":
                    self._stats["agent_collaborations"] += 1
                elif interaction_type == "tournament":
                    self._stats["tournaments"] += 1

            if governance_triggered:
                self._stats["governance_triggers"] += 1
            if experiment_triggered:
                self._stats["experiment_triggers"] += 1
            if error_occurred:
                self._stats["errors"] += 1

            # Flush if buffer is full
            if len(self._buffer) >= self._buffer_size:
                self._flush_buffer_locked()

        logger.debug(f"[TelemetryRecorder] Recorded {source} entry {entry.query_id}")

    def record_ai_interaction(
        self,
        interaction_type: str,
        sender: str,
        receiver: str,
        session_id: Optional[str] = None,
        message_type: Optional[str] = None,
        query: str = "",
        result: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record an AI-to-AI interaction for meta-learning.

        Args:
            interaction_type: "agent_communication", "tournament", "debate", "vote"
            sender: Agent or system sending
            receiver: Agent or system receiving
            session_id: Collaboration session ID
            message_type: Type of message
            query: Original query context
            result: Result of the interaction if available
            metadata: Additional metadata
        """
        import uuid

        entry = AIInteractionEntry(
            timestamp=time.time(),
            interaction_id=f"ai_{uuid.uuid4().hex[:12]}",
            interaction_type=interaction_type,
            sender=sender,
            receiver=receiver,
            session_id=session_id,
            message_type=message_type,
            query=query,
            result=result,
            metadata=metadata or {},
        )

        with self._lock:
            self._ai_interaction_buffer.append(entry)
            self._stats["ai_interactions"] += 1

            # Track specific interaction types
            if interaction_type == "agent_communication":
                self._stats["agent_collaborations"] += 1
            elif interaction_type == "tournament":
                self._stats["tournaments"] += 1
            elif interaction_type == "debate":
                self._stats["debates"] += 1
            elif interaction_type == "vote":
                self._stats["votes"] += 1

            # Flush AI interactions if buffer is full
            if len(self._ai_interaction_buffer) >= self._ai_buffer_size:
                self._flush_ai_interactions_locked()

        logger.debug(
            f"[TelemetryRecorder] Recorded AI interaction: {sender} -> {receiver} ({interaction_type})"
        )

    def record_feedback(
        self, query_id: str, feedback: str, quality_score: Optional[float] = None
    ) -> None:
        """
        Record user feedback for a previous query.

        Args:
            query_id: The query ID to attach feedback to
            feedback: User feedback (positive/negative/neutral)
            quality_score: Optional quality score (0.0 to 1.0)
        """
        with self._lock:
            # Try to find the entry in buffer
            for entry in self._buffer:
                if entry.query_id == query_id:
                    entry.user_feedback = feedback
                    entry.response_quality_score = quality_score
                    break

            # Update feedback stats
            feedback_lower = feedback.lower()
            if feedback_lower in ("positive", "good", "helpful", "yes", "thumbs_up"):
                self._stats["positive_feedback"] += 1
            elif feedback_lower in (
                "negative",
                "bad",
                "unhelpful",
                "no",
                "thumbs_down",
            ):
                self._stats["negative_feedback"] += 1

        logger.info(f"[TelemetryRecorder] Recorded feedback for {query_id}: {feedback}")

    def queue_memory_update(
        self,
        source: str,
        query_type: str,
        quality_score: Optional[float] = None,
        error_occurred: bool = False,
        error_type: Optional[str] = None,
        interaction_type: Optional[str] = None,
    ) -> None:
        """
        PERFORMANCE FIX: Queue a memory update for batch processing.
        
        Instead of writing to disk on every request (which caused progressive
        slowdown), memory updates are now buffered and flushed periodically.
        
        Args:
            source: Interaction source (user/agent/arena)
            query_type: Type of query
            quality_score: Response quality score if available
            error_occurred: Whether an error occurred
            error_type: Type of error if any
            interaction_type: Type of AI interaction if applicable
        """
        entry = MemoryUpdateEntry(
            timestamp=time.time(),
            source=source,
            query_type=query_type,
            quality_score=quality_score,
            error_occurred=error_occurred,
            error_type=error_type,
            interaction_type=interaction_type,
        )
        
        with self._memory_update_lock:
            self._memory_update_buffer.append(entry)
            self._stats["memory_updates_queued"] += 1
            
            # Flush if buffer is full
            if len(self._memory_update_buffer) >= self._memory_update_buffer_size:
                self._flush_memory_updates_locked()

    def flush(self) -> None:
        """Flush all buffered telemetry to storage."""
        with self._lock:
            self._flush_buffer_locked()
            self._flush_ai_interactions_locked()
        
        # PERFORMANCE FIX: Also flush memory updates
        with self._memory_update_lock:
            self._flush_memory_updates_locked()

    def _flush_buffer_locked(self) -> None:
        """Flush telemetry buffer while holding lock."""
        if not self._buffer:
            return

        entries_to_write = list(self._buffer)
        self._buffer.clear()

        try:
            self._write_to_meta_state(entries_to_write)
            logger.info(
                f"[TelemetryRecorder] Flushed {len(entries_to_write)} telemetry entries"
            )
        except Exception as e:
            logger.error(f"[TelemetryRecorder] Flush failed: {e}", exc_info=True)
            # Re-add entries to buffer on failure (partial recovery)
            if len(self._buffer) + len(entries_to_write) <= self._buffer_size * 2:
                self._buffer.extend(entries_to_write)

    def _flush_ai_interactions_locked(self) -> None:
        """Flush AI interaction buffer while holding lock."""
        if not self._ai_interaction_buffer:
            return

        entries_to_write = list(self._ai_interaction_buffer)
        self._ai_interaction_buffer.clear()

        try:
            self._write_ai_interactions_to_meta_state(entries_to_write)
            logger.info(
                f"[TelemetryRecorder] Flushed {len(entries_to_write)} AI interactions"
            )
        except Exception as e:
            logger.error(
                f"[TelemetryRecorder] AI interaction flush failed: {e}", exc_info=True
            )
            # Re-add entries to buffer on failure
            if (
                len(self._ai_interaction_buffer) + len(entries_to_write)
                <= self._ai_buffer_size * 2
            ):
                self._ai_interaction_buffer.extend(entries_to_write)

    def _flush_memory_updates_locked(self) -> None:
        """
        PERFORMANCE FIX: Flush memory update buffer while holding lock.
        
        This batches all memory updates and writes them in a single disk I/O
        operation, instead of the previous behavior of writing on every request.
        """
        if not self._memory_update_buffer:
            return

        updates_to_write = list(self._memory_update_buffer)
        self._memory_update_buffer.clear()

        try:
            self._write_memory_updates_to_meta_state(updates_to_write)
            self._stats["memory_updates_flushed"] += len(updates_to_write)
            logger.debug(
                f"[TelemetryRecorder] Flushed {len(updates_to_write)} memory updates"
            )
        except Exception as e:
            logger.error(
                f"[TelemetryRecorder] Memory update flush failed: {e}", exc_info=True
            )
            # Re-add entries to buffer on failure (with limit to prevent unbounded growth)
            if (
                len(self._memory_update_buffer) + len(updates_to_write)
                <= self._memory_update_buffer_size * 2
            ):
                self._memory_update_buffer.extend(updates_to_write)

    def _write_memory_updates_to_meta_state(
        self, updates: List[MemoryUpdateEntry]
    ) -> None:
        """
        PERFORMANCE FIX: Write batched memory updates to llm_meta_state.json.
        
        This consolidates multiple memory updates into a single disk write,
        dramatically reducing I/O overhead compared to the previous per-request writes.
        
        Args:
            updates: List of MemoryUpdateEntry objects to write
        """
        if not updates:
            return
            
        self._meta_state_path.parent.mkdir(parents=True, exist_ok=True)
        
        state = self._load_or_create_state()
        
        if "memory" not in state:
            state["memory"] = {
                "success_memory": {},
                "risk_memory": {},
                "utility_memory": {},
                "cost_memory": {},
            }
        
        # Process all buffered updates
        for update in updates:
            if update.source == "user":
                # Update utility_memory with user interaction data
                if update.query_type not in state["memory"]["utility_memory"]:
                    state["memory"]["utility_memory"][update.query_type] = {
                        "count": 0,
                        "last_updated": update.timestamp,
                        "patterns": [],
                    }
                
                mem = state["memory"]["utility_memory"][update.query_type]
                mem["count"] += 1
                mem["last_updated"] = update.timestamp
                
                # Store pattern if response quality is known
                if update.quality_score is not None:
                    mem["patterns"].append({
                        "timestamp": update.timestamp,
                        "quality": update.quality_score
                    })
                    # Keep only recent patterns
                    mem["patterns"] = mem["patterns"][-MAX_MEMORY_PATTERNS:]
            
            elif update.source in ("agent", "arena"):
                # Update success_memory with AI performance data
                interaction_type = update.interaction_type or "general"
                if interaction_type not in state["memory"]["success_memory"]:
                    state["memory"]["success_memory"][interaction_type] = {
                        "count": 0,
                        "successes": 0,
                        "last_updated": update.timestamp,
                    }
                
                mem = state["memory"]["success_memory"][interaction_type]
                mem["count"] += 1
                mem["last_updated"] = update.timestamp
                
                if not update.error_occurred:
                    mem["successes"] += 1
                
                # Update risk_memory if errors occurred
                if update.error_occurred:
                    error_type = update.error_type or "unknown"
                    if error_type not in state["memory"]["risk_memory"]:
                        state["memory"]["risk_memory"][error_type] = {
                            "count": 0,
                            "last_occurred": update.timestamp,
                        }
                    state["memory"]["risk_memory"][error_type]["count"] += 1
                    state["memory"]["risk_memory"][error_type]["last_occurred"] = update.timestamp
        
        # Update state timestamp
        if "state" not in state:
            state["state"] = {}
        state["state"]["last_memory_update"] = time.time()
        
        # Write back atomically
        self._write_state_atomic(state)

    def _write_to_meta_state(self, entries: List[TelemetryEntry]) -> None:
        """
        Write telemetry entries to llm_meta_state.json.

        Args:
            entries: List of TelemetryEntry objects to write
        """
        # Ensure parent directory exists
        self._meta_state_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing state or create new
        state = self._load_or_create_state()

        # Ensure objects.telemetry exists
        if "objects" not in state:
            state["objects"] = {}
        if "telemetry" not in state["objects"]:
            state["objects"]["telemetry"] = []

        # Add new entries
        for entry in entries:
            telemetry_record = entry.to_dict()
            telemetry_record["step"] = len(state["objects"]["telemetry"])
            state["objects"]["telemetry"].append(telemetry_record)

        # Enforce maximum entries limit
        if len(state["objects"]["telemetry"]) > MAX_TELEMETRY_ENTRIES:
            state["objects"]["telemetry"] = state["objects"]["telemetry"][
                -MAX_TELEMETRY_ENTRIES:
            ]

        # Update state counters
        if "state" not in state:
            state["state"] = {}
        state["state"]["telemetry_entries"] = len(state["objects"]["telemetry"])
        state["state"]["last_telemetry_update"] = time.time()

        # Write back atomically
        self._write_state_atomic(state)

    def _write_ai_interactions_to_meta_state(
        self, entries: List[AIInteractionEntry]
    ) -> None:
        """
        Write AI interaction entries to llm_meta_state.json.

        Args:
            entries: List of AIInteractionEntry objects to write
        """
        self._meta_state_path.parent.mkdir(parents=True, exist_ok=True)

        state = self._load_or_create_state()

        if "objects" not in state:
            state["objects"] = {}
        if "ai_interactions" not in state["objects"]:
            state["objects"]["ai_interactions"] = []

        # Add new entries
        for entry in entries:
            ai_record = entry.to_dict()
            ai_record["step"] = len(state["objects"]["ai_interactions"])
            state["objects"]["ai_interactions"].append(ai_record)

        # Enforce maximum entries limit
        if len(state["objects"]["ai_interactions"]) > MAX_AI_INTERACTION_ENTRIES:
            state["objects"]["ai_interactions"] = state["objects"]["ai_interactions"][
                -MAX_AI_INTERACTION_ENTRIES:
            ]

        # Update counters
        if "state" not in state:
            state["state"] = {}
        state["state"]["ai_interaction_entries"] = len(
            state["objects"]["ai_interactions"]
        )
        state["state"]["last_ai_interaction_update"] = time.time()

        self._write_state_atomic(state)

    def _load_or_create_state(self) -> Dict[str, Any]:
        """Load existing state or create new default state."""
        if self._meta_state_path.exists():
            try:
                with open(self._meta_state_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(
                    f"[TelemetryRecorder] Could not load state, creating new: {e}"
                )

        return self._create_default_state()

    def _write_state_atomic(self, state: Dict[str, Any]) -> None:
        """Write state atomically using temporary file."""
        temp_path = self._meta_state_path.with_suffix(".json.tmp")
        try:
            with open(temp_path, "w") as f:
                json.dump(state, f, indent=2)
            temp_path.replace(self._meta_state_path)
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise e

    def _create_default_state(self) -> Dict[str, Any]:
        """Create default llm_meta_state.json structure."""
        return {
            "config": {
                "max_history": 5000,
                "max_concurrent_experiments": 3,
                "user_experiment_threshold": USER_INTERACTION_EXPERIMENT_THRESHOLD,
                "ai_experiment_threshold": AI_INTERACTION_EXPERIMENT_THRESHOLD,
            },
            "state": {
                "telemetry_entries": 0,
                "ai_interaction_entries": 0,
                "last_telemetry_update": time.time(),
                "last_ai_interaction_update": time.time(),
                "last_memory_update": time.time(),
            },
            "memory": {
                "success_memory": {},
                "risk_memory": {},
                "utility_memory": {},
                "cost_memory": {},
            },
            "objects": {
                "telemetry": [],
                "ai_interactions": [],
                "issues": [],
                "experiments": [],
                "outcomes": [],
            },
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive telemetry statistics.

        Returns:
            Dictionary with counts and metrics
        """
        with self._lock:
            stats = dict(self._stats)
            stats["buffer_size"] = len(self._buffer)
            stats["ai_buffer_size"] = len(self._ai_interaction_buffer)
            stats["total_entries"] = self._total_entries
            stats["session_entries"] = self._session_entries

            if stats["total_queries"] > 0:
                stats["avg_latency_ms"] = (
                    stats["total_latency_ms"] / stats["total_queries"]
                )
            else:
                stats["avg_latency_ms"] = 0.0

        # PERFORMANCE FIX: Include memory update buffer stats
        with self._memory_update_lock:
            stats["memory_update_buffer_size"] = len(self._memory_update_buffer)

        return stats

    def get_telemetry_count(self) -> int:
        """Get total telemetry entries (including those flushed to disk)."""
        with self._lock:
            return self._total_entries

    def get_ai_interaction_count(self) -> int:
        """Get total AI interaction entries recorded."""
        with self._lock:
            return self._stats["ai_interactions"]

    def shutdown(self) -> None:
        """Shutdown the recorder, flushing remaining entries."""
        logger.debug("[TelemetryRecorder] Shutting down...")
        self._shutdown_event.set()

        if self._flush_thread and self._flush_thread.is_alive():
            self._flush_thread.join(timeout=5.0)

        # Final flush
        try:
            self.flush()
        except Exception as e:
            logger.error(f"[TelemetryRecorder] Error during shutdown flush: {e}")

        logger.debug("[TelemetryRecorder] Shutdown complete")


# ============================================================
# SINGLETON PATTERN
# ============================================================

_global_recorder: Optional[TelemetryRecorder] = None
_recorder_lock = threading.Lock()


def get_telemetry_recorder() -> TelemetryRecorder:
    """
    Get or create the global telemetry recorder (thread-safe singleton).

    Returns:
        TelemetryRecorder instance
    """
    global _global_recorder

    if _global_recorder is None:
        with _recorder_lock:
            if _global_recorder is None:
                _global_recorder = TelemetryRecorder()
                logger.debug("Global TelemetryRecorder instance created")

    return _global_recorder


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================


def record_telemetry(
    query: str, response: Optional[str], metadata: Dict[str, Any], **kwargs
) -> None:
    """
    Record interaction telemetry for meta-learning system.

    Convenience function using global recorder.

    Args:
        query: The query text
        response: The response text
        metadata: Additional metadata
        **kwargs: Additional arguments passed to TelemetryRecorder.record()
    """
    recorder = get_telemetry_recorder()
    recorder.record(query, response, metadata, **kwargs)


def record_interaction(
    source: Literal["user", "agent", "arena"],
    query: str,
    response: str,
    metadata: Dict[str, Any],
) -> None:
    """
    Record ALL interactions for meta-learning (dual-mode).

    User interactions:
        - Query patterns
        - Response quality
        - User feedback (implicit/explicit)

    AI interactions:
        - Agent collaborations
        - Tournament results
        - Evolution outcomes
        - Inter-agent communication patterns

    Args:
        source: "user", "agent", or "arena"
        query: The query
        response: The response
        metadata: Additional metadata
    """
    recorder = get_telemetry_recorder()

    # Extract known kwargs from metadata
    record_kwargs = {
        "source": source,
        "query_id": metadata.get("query_id"),
        "session_id": metadata.get("session_id"),
        "latency_ms": metadata.get("latency_ms", 0.0),
        "agent_tasks_submitted": metadata.get("agent_tasks_submitted", 0),
        "agent_tasks_completed": metadata.get("agent_tasks_completed", 0),
        "governance_triggered": metadata.get("governance_triggered", False),
        "experiment_triggered": metadata.get("experiment_triggered", False),
        "error_occurred": metadata.get("error_occurred", False),
        "error_type": metadata.get("error_type"),
        "collaboration_session_id": metadata.get("collaboration_session_id"),
        "agents_involved": metadata.get("agents_involved"),
        "interaction_type": metadata.get("interaction_type"),
    }

    recorder.record(query, response, metadata, **record_kwargs)

    # PERFORMANCE FIX: Queue memory update instead of writing to disk immediately
    # This prevents progressive slowdown by batching disk writes
    recorder.queue_memory_update(
        source=source,
        query_type=metadata.get("query_type", "general"),
        quality_score=metadata.get("response_quality_score"),
        error_occurred=metadata.get("error_occurred", False),
        error_type=metadata.get("error_type"),
        interaction_type=metadata.get("interaction_type"),
    )


def record_ai_interaction(
    interaction_type: str,
    sender: str,
    receiver: str,
    session_id: Optional[str] = None,
    message_type: Optional[str] = None,
    query: str = "",
    result: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Record an AI-to-AI interaction.

    This records:
        - Agent-to-agent communication
        - Tournament participation
        - Debate contributions
        - Voting actions

    All AI interactions are recorded as learning data for meta-learning.

    Args:
        interaction_type: "agent_communication", "tournament", "debate", "vote"
        sender: Agent or system sending
        receiver: Agent or system receiving
        session_id: Collaboration session ID
        message_type: Type of message
        query: Original query context
        result: Result of the interaction if available
        metadata: Additional metadata
    """
    recorder = get_telemetry_recorder()
    recorder.record_ai_interaction(
        interaction_type=interaction_type,
        sender=sender,
        receiver=receiver,
        session_id=session_id,
        message_type=message_type,
        query=query,
        result=result,
        metadata=metadata,
    )


def _update_memory_systems(
    source: str, query: str, response: str, metadata: Dict[str, Any]
) -> None:
    """
    DEPRECATED: Update VULCAN memory systems based on interaction source.
    
    This function is kept for backwards compatibility but now uses the
    buffered queue_memory_update() method instead of direct disk writes.
    
    PERFORMANCE FIX: Memory updates are now buffered and flushed periodically
    instead of writing to disk on every request. This prevents the progressive
    slowdown that was occurring when the JSON file grew large.

    User interactions -> utility_memory (learn from real-world problems)
    AI interactions -> success_memory, risk_memory (learn from AI performance)

    Args:
        source: Interaction source
        query: The query
        response: The response
        metadata: Interaction metadata
    """
    # PERFORMANCE FIX: Use buffered updates instead of direct disk I/O
    recorder = get_telemetry_recorder()
    
    recorder.queue_memory_update(
        source=source,
        query_type=metadata.get("query_type", "general"),
        quality_score=metadata.get("response_quality_score"),
        error_occurred=metadata.get("error_occurred", False),
        error_type=metadata.get("error_type"),
        interaction_type=metadata.get("interaction_type"),
    )
