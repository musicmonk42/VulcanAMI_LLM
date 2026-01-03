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
# PERFORMANCE FIX: All disk I/O now runs in a background ThreadPoolExecutor
# to prevent blocking the asyncio event loop and causing request delays.
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

Performance Fix:
    All disk I/O operations now run in a background ThreadPoolExecutor
    to prevent blocking the asyncio event loop. This eliminates the
    progressive slowdown issue that occurred when the JSON file grew large.

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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TYPE_CHECKING

# Initialize logger immediately after imports
logger = logging.getLogger(__name__)

# ============================================================
# SOPHISTICATED MEMORY SYSTEM INTEGRATION
# ============================================================
# Import the advanced persistent memory system from persistant_memory_v46
# This provides GraphRAG, MerkleLSM, S3 packfile storage, and ZK proofs
# for production-grade memory management instead of just JSON files.

GRAPH_RAG_AVAILABLE = False
PERSISTENT_MEMORY_AVAILABLE = False

try:
    from persistant_memory_v46 import GraphRAG, create_memory_system, get_system_info
    GRAPH_RAG_AVAILABLE = True
    logger.info("GraphRAG from persistant_memory_v46 available for advanced memory storage")
except ImportError:
    logger.debug("persistant_memory_v46 not available, using JSON-based memory storage")

try:
    from vulcan.memory.hierarchical import HierarchicalMemory
    PERSISTENT_MEMORY_AVAILABLE = True
    logger.info("HierarchicalMemory available for persistent memory storage")
except ImportError:
    try:
        from src.vulcan.memory.hierarchical import HierarchicalMemory
        PERSISTENT_MEMORY_AVAILABLE = True
        logger.info("HierarchicalMemory available for persistent memory storage (src prefix)")
    except ImportError:
        logger.debug("HierarchicalMemory not available, using JSON-based memory storage")

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

# PERFORMANCE FIX: Thread pool for non-blocking I/O
IO_THREAD_POOL_SIZE = 2

# Thresholds for experiment triggers
USER_INTERACTION_EXPERIMENT_THRESHOLD = 100
AI_INTERACTION_EXPERIMENT_THRESHOLD = 50

# Memory limits
MAX_MEMORY_PATTERNS = 100
MAX_TELEMETRY_ENTRIES = 10000
MAX_AI_INTERACTION_ENTRIES = 10000

# GraphRAG storage limits
GRAPHRAG_QUERY_TRUNCATE_LENGTH = 2000  # Max chars for query text in GraphRAG
GRAPHRAG_RESPONSE_TRUNCATE_LENGTH = 2000  # Max chars for response text in GraphRAG
GRAPHRAG_RESULT_TRUNCATE_LENGTH = 500  # Max chars for AI result JSON in GraphRAG


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

    PERFORMANCE FIX: All disk I/O operations now run in a background
    ThreadPoolExecutor to prevent blocking the asyncio event loop.

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
        io_thread_pool_size: int = IO_THREAD_POOL_SIZE,
        use_graph_rag: bool = True,
        graph_rag_config: Optional[Dict[str, Any]] = None,
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
            io_thread_pool_size: Number of threads for background I/O operations
            use_graph_rag: Whether to use the sophisticated GraphRAG memory system
            graph_rag_config: Optional configuration for GraphRAG
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
        
        # PERFORMANCE FIX: Thread pool for non-blocking I/O
        # All disk operations run here instead of blocking the main thread
        self._io_executor = ThreadPoolExecutor(
            max_workers=io_thread_pool_size,
            thread_name_prefix="TelemetryIO"
        )
        self._flush_in_progress = threading.Event()
        self._executor_shutdown = False  # Track if executor has been shut down

        # ============================================================
        # SOPHISTICATED MEMORY SYSTEM INTEGRATION
        # ============================================================
        # Initialize GraphRAG for advanced memory storage if available
        self._graph_rag: Optional[Any] = None
        self._use_graph_rag = use_graph_rag and GRAPH_RAG_AVAILABLE
        
        if self._use_graph_rag:
            try:
                rag_config = graph_rag_config or {}
                self._graph_rag = GraphRAG(
                    embedding_dim=rag_config.get("embedding_dim", 384),
                    cache_capacity=rag_config.get("cache_capacity", 1000),
                    **{k: v for k, v in rag_config.items() if k not in ["embedding_dim", "cache_capacity"]}
                )
                logger.info(
                    "[TelemetryRecorder] GraphRAG memory system initialized - "
                    "using advanced persistent memory with graph-based retrieval"
                )
            except Exception as e:
                logger.warning(f"[TelemetryRecorder] Failed to initialize GraphRAG: {e}, falling back to JSON")
                self._graph_rag = None
                self._use_graph_rag = False
        else:
            logger.info(
                "[TelemetryRecorder] Using JSON-based memory storage "
                f"(GraphRAG available: {GRAPH_RAG_AVAILABLE}, enabled: {use_graph_rag})"
            )

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
            # I/O stats
            "async_flushes_started": 0,
            "async_flushes_completed": 0,
            # GraphRAG stats
            "graph_rag_enabled": self._use_graph_rag,
            "graph_rag_documents_added": 0,
        }

        # Auto-flush configuration
        self._auto_flush = auto_flush
        self._shutdown_event = threading.Event()
        self._flush_thread: Optional[threading.Thread] = None

        if auto_flush:
            self._start_auto_flush()

        # Register shutdown handler
        atexit.register(self._atexit_handler)

        logger.debug(f"TelemetryRecorder initialized with non-blocking I/O, path: {self._meta_state_path}")

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
                # PERFORMANCE FIX: Use non-blocking flush
                self.flush_async()
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

        This method is non-blocking - it only appends to an in-memory buffer.
        Actual disk I/O happens asynchronously in a background thread.

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

        # Include query/response in metadata for GraphRAG storage
        # This enables semantic search over past interactions
        enhanced_metadata = dict(metadata)
        if self._use_graph_rag:
            # Store truncated query/response for GraphRAG indexing
            enhanced_metadata["query"] = query[:GRAPHRAG_QUERY_TRUNCATE_LENGTH] if query else ""
            enhanced_metadata["response"] = response[:GRAPHRAG_RESPONSE_TRUNCATE_LENGTH] if response else ""

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
            metadata=enhanced_metadata,
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

            # PERFORMANCE FIX: Trigger async flush if buffer is full
            if len(self._buffer) >= self._buffer_size:
                self._schedule_flush_locked()

        logger.debug(f"[TelemetryRecorder] Recorded {source} entry {entry.query_id}")
        
        # FIX Issue #1: Queue memory update so conversation context is stored
        # Previously, only convenience functions (record_telemetry, record_interaction)
        # queued memory updates. Now record() does too, ensuring all interactions
        # contribute to learning data regardless of how they're recorded.
        self.queue_memory_update(
            source=source,
            query_type=metadata.get("query_type", "general"),
            quality_score=metadata.get("response_quality_score"),
            error_occurred=error_occurred,
            error_type=error_type,
            interaction_type=interaction_type,
        )

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

        This method is non-blocking - it only appends to an in-memory buffer.

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

            # PERFORMANCE FIX: Trigger async flush if buffer is full
            if len(self._ai_interaction_buffer) >= self._ai_buffer_size:
                self._schedule_flush_locked()

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
        
        This method is non-blocking.
        
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
            
            # Trigger async flush if buffer is full
            if len(self._memory_update_buffer) >= self._memory_update_buffer_size:
                self._schedule_memory_flush_locked()

    def _submit_flush_to_executor(self) -> None:
        """
        Submit a flush task to the background executor.
        
        This is the core implementation that handles executor shutdown gracefully.
        Does nothing if the executor has been shut down or a flush is already in progress.
        
        Returns:
            None
        """
        if self._executor_shutdown:
            return
        if not self._flush_in_progress.is_set():
            self._flush_in_progress.set()
            self._stats["async_flushes_started"] += 1
            try:
                self._io_executor.submit(self._do_flush_in_background)
            except RuntimeError:
                # Executor is shutting down - ignore
                self._flush_in_progress.clear()

    def _schedule_flush_locked(self) -> None:
        """
        Schedule a non-blocking flush (must hold self._lock).
        
        PERFORMANCE FIX: This submits the flush to a background thread pool
        instead of blocking the current thread.
        
        Does nothing if the executor has been shut down.
        """
        self._submit_flush_to_executor()

    def _schedule_memory_flush_locked(self) -> None:
        """
        Schedule a non-blocking memory flush (must hold self._memory_update_lock).
        
        Does nothing if the executor has been shut down.
        """
        self._submit_flush_to_executor()

    def flush_async(self) -> None:
        """
        PERFORMANCE FIX: Trigger a non-blocking flush.
        
        The actual I/O happens in a background thread, so this method
        returns immediately without blocking the caller.
        
        Does nothing if the executor has been shut down.
        """
        self._submit_flush_to_executor()

    def _do_flush_in_background(self) -> None:
        """
        PERFORMANCE FIX: Perform the actual flush in a background thread.
        
        This runs in the ThreadPoolExecutor, NOT the main event loop.
        The main thread is never blocked by disk I/O.
        """
        try:
            # Grab data from buffers quickly while holding locks
            telemetry_entries = []
            ai_entries = []
            memory_updates = []
            
            with self._lock:
                if self._buffer:
                    telemetry_entries = list(self._buffer)
                    self._buffer.clear()
                if self._ai_interaction_buffer:
                    ai_entries = list(self._ai_interaction_buffer)
                    self._ai_interaction_buffer.clear()
            
            with self._memory_update_lock:
                if self._memory_update_buffer:
                    memory_updates = list(self._memory_update_buffer)
                    self._memory_update_buffer.clear()
            
            # Now do the slow disk I/O WITHOUT holding any locks
            # This doesn't block the main thread
            if telemetry_entries or ai_entries or memory_updates:
                self._write_all_to_meta_state(telemetry_entries, ai_entries, memory_updates)
                
                total_entries = len(telemetry_entries)
                total_ai = len(ai_entries)
                total_memory = len(memory_updates)
                
                logger.info(
                    f"[TelemetryRecorder] Flushed {total_entries} telemetry entries, "
                    f"{total_ai} AI interactions, {total_memory} memory updates (background)"
                )
                
                with self._lock:
                    self._stats["async_flushes_completed"] += 1
                    self._stats["memory_updates_flushed"] += total_memory
                    
        except Exception as e:
            logger.error(f"[TelemetryRecorder] Background flush failed: {e}", exc_info=True)
        finally:
            self._flush_in_progress.clear()

    def _write_all_to_meta_state(
        self,
        telemetry_entries: List[TelemetryEntry],
        ai_entries: List[AIInteractionEntry],
        memory_updates: List[MemoryUpdateEntry],
    ) -> None:
        """
        PERFORMANCE FIX: Write all buffered data to llm_meta_state.json in a single operation.
        
        This runs in a background thread and does NOT block the event loop.
        All data is written atomically to prevent corruption.
        
        Args:
            telemetry_entries: List of TelemetryEntry objects to write
            ai_entries: List of AIInteractionEntry objects to write
            memory_updates: List of MemoryUpdateEntry objects to write
        """
        # Ensure parent directory exists
        self._meta_state_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing state or create new
        state = self._load_or_create_state()
        
        # Ensure structure exists
        if "objects" not in state:
            state["objects"] = {}
        if "telemetry" not in state["objects"]:
            state["objects"]["telemetry"] = []
        if "ai_interactions" not in state["objects"]:
            state["objects"]["ai_interactions"] = []
        if "memory" not in state:
            state["memory"] = {
                "success_memory": {},
                "risk_memory": {},
                "utility_memory": {},
                "cost_memory": {},
            }
        if "state" not in state:
            state["state"] = {}
        
        # Add telemetry entries
        for entry in telemetry_entries:
            telemetry_record = entry.to_dict()
            telemetry_record["step"] = len(state["objects"]["telemetry"])
            state["objects"]["telemetry"].append(telemetry_record)
        
        # Enforce maximum telemetry entries limit
        if len(state["objects"]["telemetry"]) > MAX_TELEMETRY_ENTRIES:
            state["objects"]["telemetry"] = state["objects"]["telemetry"][
                -MAX_TELEMETRY_ENTRIES:
            ]
        
        # Add AI interaction entries
        for entry in ai_entries:
            ai_record = entry.to_dict()
            ai_record["step"] = len(state["objects"]["ai_interactions"])
            state["objects"]["ai_interactions"].append(ai_record)
        
        # Enforce maximum AI interaction entries limit
        if len(state["objects"]["ai_interactions"]) > MAX_AI_INTERACTION_ENTRIES:
            state["objects"]["ai_interactions"] = state["objects"]["ai_interactions"][
                -MAX_AI_INTERACTION_ENTRIES:
            ]
        
        # Process memory updates
        for update in memory_updates:
            self._apply_memory_update(state, update)
        
        # Update state counters
        state["state"]["telemetry_entries"] = len(state["objects"]["telemetry"])
        state["state"]["ai_interaction_entries"] = len(
            state["objects"]["ai_interactions"]
        )
        state["state"]["last_telemetry_update"] = time.time()
        state["state"]["last_ai_interaction_update"] = time.time()
        state["state"]["last_memory_update"] = time.time()
        
        # Write back atomically
        self._write_state_atomic(state)
        
        # ============================================================
        # SOPHISTICATED MEMORY SYSTEM INTEGRATION
        # ============================================================
        # Also store telemetry entries in GraphRAG for advanced retrieval
        if self._use_graph_rag and self._graph_rag is not None:
            self._store_to_graph_rag(telemetry_entries, ai_entries, memory_updates)

    def _store_to_graph_rag(
        self,
        telemetry_entries: List[TelemetryEntry],
        ai_entries: List[AIInteractionEntry],
        memory_updates: List[MemoryUpdateEntry],
    ) -> None:
        """
        Store telemetry data to the sophisticated GraphRAG memory system.
        
        This enables:
        - Semantic search over telemetry history
        - Graph-based relationship discovery between interactions
        - Hybrid retrieval (vector + BM25) for finding similar past interactions
        - Cross-encoder reranking for high-quality retrieval
        
        Args:
            telemetry_entries: List of TelemetryEntry objects
            ai_entries: List of AIInteractionEntry objects
            memory_updates: List of MemoryUpdateEntry objects
        """
        if not self._graph_rag:
            return
            
        try:
            # Store telemetry entries as documents
            for entry in telemetry_entries:
                doc_id = f"telemetry_{entry.query_id}_{int(entry.timestamp * 1000)}"
                # Get query/response from metadata if available (they're not stored in entry directly)
                query_text = entry.metadata.get("query", f"Query (length: {entry.query_length})")
                response_text = entry.metadata.get("response", f"Response (length: {entry.response_length})")
                content = f"Query: {query_text}\nResponse: {response_text}"
                metadata = {
                    "type": "telemetry",
                    "source": entry.source,
                    "query_type": entry.metadata.get("query_type", "unknown"),
                    "timestamp": entry.timestamp,
                    "session_id": entry.session_id,
                    "latency_ms": entry.latency_ms,
                    "governance_triggered": entry.governance_triggered,
                    "experiment_triggered": entry.experiment_triggered,
                }
                
                self._graph_rag.add_document(
                    doc_id=doc_id,
                    content=content,
                    metadata=metadata,
                    auto_chunk=True,
                )
                self._stats["graph_rag_documents_added"] += 1
            
            # Store AI interactions as documents
            for entry in ai_entries:
                doc_id = f"ai_{entry.interaction_id}_{int(entry.timestamp * 1000)}"
                content = f"AI Interaction: {entry.interaction_type}\nFrom: {entry.sender}\nTo: {entry.receiver}\nQuery: {entry.query}"
                if entry.result:
                    content += f"\nResult: {json.dumps(entry.result)[:GRAPHRAG_RESULT_TRUNCATE_LENGTH]}"
                    
                metadata = {
                    "type": "ai_interaction",
                    "interaction_type": entry.interaction_type,
                    "sender": entry.sender,
                    "receiver": entry.receiver,
                    "timestamp": entry.timestamp,
                    "session_id": entry.session_id,
                    "message_type": entry.message_type,
                }
                
                self._graph_rag.add_document(
                    doc_id=doc_id,
                    content=content,
                    metadata=metadata,
                    auto_chunk=False,  # AI interactions are typically shorter
                )
                self._stats["graph_rag_documents_added"] += 1
                
            logger.debug(
                f"[TelemetryRecorder] Stored {len(telemetry_entries)} telemetry + "
                f"{len(ai_entries)} AI entries to GraphRAG"
            )
            
        except Exception as e:
            logger.warning(f"[TelemetryRecorder] Failed to store to GraphRAG: {e}")

    def retrieve_similar_interactions(
        self,
        query: str,
        k: int = 10,
        interaction_type: Optional[str] = None,
        use_rerank: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve similar past interactions using the sophisticated GraphRAG system.
        
        This method leverages the advanced persistent memory capabilities:
        - Semantic vector search for similarity matching
        - BM25 keyword search for exact matches
        - Graph-based expansion for related context
        - Cross-encoder reranking for quality results
        
        Args:
            query: The query to search for similar interactions
            k: Number of results to return
            interaction_type: Optional filter for interaction type ("telemetry" or "ai_interaction")
            use_rerank: Whether to use cross-encoder reranking
            
        Returns:
            List of similar interactions with scores and metadata
        """
        if not self._use_graph_rag or not self._graph_rag:
            logger.debug("[TelemetryRecorder] GraphRAG not available for retrieval")
            return []
            
        try:
            # Build filters
            filters = {}
            if interaction_type:
                filters["type"] = interaction_type
                
            # Retrieve from GraphRAG
            results = self._graph_rag.retrieve(
                query=query,
                k=k,
                use_rerank=use_rerank,
                use_hybrid=True,
                filters=filters if filters else None,
            )
            
            # Convert to dict format
            return [
                {
                    "node_id": r.node_id,
                    "content": r.content,
                    "score": r.score,
                    "metadata": r.metadata,
                    "source": r.source,
                }
                for r in results
            ]
            
        except Exception as e:
            logger.warning(f"[TelemetryRecorder] GraphRAG retrieval failed: {e}")
            return []

    def get_graph_rag_stats(self) -> Dict[str, Any]:
        """
        Get statistics from the GraphRAG memory system.
        
        Returns:
            Dictionary with GraphRAG statistics or empty dict if not available
        """
        if not self._use_graph_rag or not self._graph_rag:
            return {"enabled": False, "reason": "GraphRAG not initialized"}
            
        try:
            return {
                "enabled": True,
                **self._graph_rag.get_statistics(),
            }
        except Exception as e:
            return {"enabled": True, "error": str(e)}

    def _apply_memory_update(self, state: Dict[str, Any], update: MemoryUpdateEntry) -> None:
        """
        Apply a single memory update to the state dict.
        
        User interactions -> utility_memory (learn from real-world problems)
        AI interactions -> success_memory, risk_memory (learn from AI performance)
        
        Args:
            state: The state dictionary to update
            update: The memory update entry to apply
        """
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

    def flush(self) -> None:
        """
        Flush all buffered telemetry to storage.
        
        This is a synchronous method that waits for the flush to complete.
        Use flush_async() for non-blocking behavior.
        """
        # Trigger async flush
        self.flush_async()
        
        # Wait for flush to complete (with timeout)
        timeout = 30  # seconds
        start = time.time()
        while self._flush_in_progress.is_set() and (time.time() - start) < timeout:
            time.sleep(0.1)

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

        stats["flush_in_progress"] = self._flush_in_progress.is_set()
        
        # Include GraphRAG statistics if available
        if self._use_graph_rag:
            stats["graph_rag"] = self.get_graph_rag_stats()

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

        # Final synchronous flush
        try:
            self.flush()
        except Exception as e:
            logger.error(f"[TelemetryRecorder] Error during shutdown flush: {e}")

        # Mark executor as shut down BEFORE shutting it down
        # This prevents flush_async() from trying to submit new work
        self._executor_shutdown = True
        
        # Shutdown thread pool
        self._io_executor.shutdown(wait=True, cancel_futures=False)

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

    Convenience function using global recorder. Non-blocking.
    
    Memory updates are automatically queued by the record() method, so
    conversation context and learning data is stored.

    Args:
        query: The query text
        response: The response text
        metadata: Additional metadata
        **kwargs: Additional arguments passed to TelemetryRecorder.record()
    """
    recorder = get_telemetry_recorder()
    # record() now automatically queues memory updates (FIX Issue #1)
    recorder.record(query, response, metadata, **kwargs)


def record_interaction(
    source: Literal["user", "agent", "arena"],
    query: str,
    response: str,
    metadata: Dict[str, Any],
) -> None:
    """
    Record ALL interactions for meta-learning (dual-mode).

    This is completely non-blocking - all disk I/O happens in background threads.
    Memory updates are automatically queued by the record() method.

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

    # record() now automatically queues memory updates (FIX Issue #1)
    recorder.record(query, response, metadata, **record_kwargs)


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

    This is non-blocking - all disk I/O happens in background threads.

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
    Update VULCAN memory systems based on interaction source.

    DEPRECATED: This function is kept for backwards compatibility but now uses
    the buffered queue_memory_update() method instead of direct disk writes.

    PERFORMANCE FIX: Memory updates are now buffered and flushed periodically
    in a background thread, instead of writing to disk on every request.
    This prevents the progressive slowdown that was occurring when the JSON
    file grew large.

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
