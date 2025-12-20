# ============================================================
# VULCAN-AGI Agent Collaboration System
# ============================================================
# Enterprise-grade multi-agent collaboration for complex queries:
# - Agent-to-agent communication protocol
# - Multi-agent deliberation and debate
# - Collaborative problem solving
# - Inter-agent voting and consensus
#
# PRODUCTION-READY: Thread-safe, graceful degradation, comprehensive logging
# AI LEARNING: All interactions recorded for meta-learning system
# ============================================================

"""
VULCAN Agent Collaboration System

Enables multi-agent collaboration for complex queries with full support
for recording AI-to-AI interactions as learning data.

Features:
    - Agent-to-agent communication protocol
    - Multi-agent deliberation sessions
    - Inter-agent debates and voting
    - Collaborative problem solving
    - Comprehensive telemetry recording

Collaboration Types:
    - Sequential: Agents process in order, each building on previous results
    - Parallel: Agents process simultaneously, results synthesized
    - Debate: Agents present opposing viewpoints, reasoning agent synthesizes
    - Vote: Agents vote on options, majority or weighted decision

Thread Safety:
    All public methods are thread-safe. The AgentCollaborationManager
    maintains internal state using proper locking mechanisms.

Usage:
    from vulcan.routing import trigger_agent_collaboration

    # Start a collaboration session
    session = trigger_agent_collaboration(
        query="Analyze and plan approach",
        agent_types=["perception", "reasoning", "planning"]
    )

    # Check session status
    print(session.status, session.interaction_count)
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

# Initialize logger immediately after imports
logger = logging.getLogger(__name__)

# ============================================================
# TYPE CHECKING IMPORTS
# ============================================================

if TYPE_CHECKING:
    from ..orchestrator.agent_pool import AgentPool


# ============================================================
# CONSTANTS
# ============================================================

# Maximum concurrent collaboration sessions
MAX_CONCURRENT_SESSIONS = 50

# Session timeout in seconds
DEFAULT_SESSION_TIMEOUT = 300.0

# Maximum messages per session
MAX_MESSAGES_PER_SESSION = 1000

# Maximum agents per collaboration
MAX_AGENTS_PER_COLLABORATION = 10


# ============================================================
# ENUMS
# ============================================================


class MessageType(str, Enum):
    """Types of inter-agent messages."""

    RESULT = "result"
    QUESTION = "question"
    DEBATE = "debate"
    VOTE = "vote"
    REQUEST = "request"
    RESPONSE = "response"
    SYNTHESIS = "synthesis"
    ERROR = "error"


class CollaborationStatus(str, Enum):
    """Status of a collaboration session."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class CollaborationType(str, Enum):
    """Types of collaboration patterns."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    DEBATE = "debate"
    VOTE = "vote"
    SYNTHESIS = "synthesis"


# ============================================================
# DATA CLASSES
# ============================================================


@dataclass
class AgentMessage:
    """
    Protocol for agents to communicate with each other.

    All inter-agent messages use this format to ensure consistent
    tracking and telemetry recording.

    Attributes:
        message_id: Unique message identifier
        sender_agent: Agent sending the message
        receiver_agent: Agent receiving the message (or "all" for broadcast)
        message_type: Type of message
        content: Message content as dictionary
        context: Original query context
        timestamp: Message creation timestamp
        parent_message_id: ID of message this responds to (for threading)
    """

    message_id: str
    sender_agent: str
    receiver_agent: str
    message_type: MessageType
    content: Dict[str, Any]
    context: str
    timestamp: float = field(default_factory=time.time)
    parent_message_id: Optional[str] = None

    def to_task(self) -> Dict[str, Any]:
        """
        Convert message to agent pool task format.

        Returns:
            Dictionary suitable for submission to agent pool
        """
        return {
            "id": f"msg_{self.message_id}",
            "type": f"agent_message_{self.message_type.value}",
            "capability": (
                self.receiver_agent if self.receiver_agent != "all" else "general"
            ),
            "nodes": [
                {
                    "id": "receive",
                    "type": "message_receive",
                    "params": {"message": self.content},
                },
                {
                    "id": "process",
                    "type": (
                        self.receiver_agent
                        if self.receiver_agent != "all"
                        else "general"
                    ),
                    "params": {"context": self.context},
                },
                {
                    "id": "respond",
                    "type": "message_send",
                    "params": {"sender": self.receiver_agent},
                },
            ],
            "edges": [
                {"from": "receive", "to": "process"},
                {"from": "process", "to": "respond"},
            ],
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "message_id": self.message_id,
            "sender_agent": self.sender_agent,
            "receiver_agent": self.receiver_agent,
            "message_type": self.message_type.value,
            "content": self.content,
            "context": self.context[:200] if self.context else None,
            "timestamp": self.timestamp,
            "parent_message_id": self.parent_message_id,
        }


@dataclass
class CollaborationSession:
    """
    Represents a multi-agent collaboration session.

    Tracks all messages, results, and statistics for a collaboration.

    Attributes:
        session_id: Unique session identifier
        original_query: The query being processed
        agents_involved: List of agent types participating
        collaboration_type: Type of collaboration pattern
        status: Current session status
        messages: List of inter-agent messages
        results: Results from each agent
        start_time: Session start timestamp
        end_time: Session end timestamp (when completed)
        error: Error message if session failed
    """

    session_id: str
    original_query: str
    agents_involved: List[str]
    collaboration_type: CollaborationType = CollaborationType.PARALLEL
    status: CollaborationStatus = CollaborationStatus.PENDING
    messages: List[AgentMessage] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    error: Optional[str] = None

    @property
    def duration_seconds(self) -> float:
        """Get session duration in seconds."""
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def interaction_count(self) -> int:
        """Get number of inter-agent interactions."""
        return len(self.messages)

    @property
    def is_active(self) -> bool:
        """Check if session is still active."""
        return self.status in (
            CollaborationStatus.PENDING,
            CollaborationStatus.IN_PROGRESS,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "original_query": self.original_query[:200],
            "agents_involved": self.agents_involved,
            "collaboration_type": self.collaboration_type.value,
            "status": self.status.value,
            "interaction_count": self.interaction_count,
            "results_count": len(self.results),
            "duration_seconds": self.duration_seconds,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "error": self.error,
        }


# ============================================================
# AGENT COLLABORATION MANAGER
# ============================================================


class AgentCollaborationManager:
    """
    Manages multi-agent collaboration sessions.

    When a query needs multiple perspectives:
    1. Creates collaboration session
    2. Submits subtasks to different agents
    3. Tracks inter-agent communication
    4. Synthesizes results
    5. Records all interactions as AI learning data

    Thread-safe implementation with comprehensive statistics tracking.

    Usage:
        manager = AgentCollaborationManager()
        session = manager.start_collaboration(
            query="Complex analysis request",
            agent_types=["perception", "reasoning", "planning"]
        )

        # Monitor session
        while session.is_active:
            time.sleep(1)

        print(session.results)
    """

    def __init__(
        self,
        agent_pool: Optional[AgentPool] = None,
        telemetry_recorder: Optional[Any] = None,
        max_concurrent_sessions: int = MAX_CONCURRENT_SESSIONS,
        session_timeout: float = DEFAULT_SESSION_TIMEOUT,
    ):
        """
        Initialize the collaboration manager.

        Args:
            agent_pool: Reference to the agent pool for submitting tasks
            telemetry_recorder: Reference to telemetry recorder for logging
            max_concurrent_sessions: Maximum concurrent collaboration sessions
            session_timeout: Default session timeout in seconds
        """
        self._agent_pool = agent_pool
        self._telemetry_recorder = telemetry_recorder
        self._max_concurrent_sessions = max_concurrent_sessions
        self._session_timeout = session_timeout

        # Thread safety
        self._lock = threading.RLock()

        # Active collaboration sessions
        self._sessions: Dict[str, CollaborationSession] = {}

        # Statistics tracking
        self._stats = {
            "total_collaborations": 0,
            "total_messages": 0,
            "total_interactions": 0,
            "successful_collaborations": 0,
            "failed_collaborations": 0,
            "timeout_collaborations": 0,
            "avg_agents_per_collaboration": 0.0,
            "avg_duration_seconds": 0.0,
        }

        # Callbacks for extensibility
        self._on_message_callback: Optional[Callable[[AgentMessage], None]] = None
        self._on_complete_callback: Optional[Callable[[CollaborationSession], None]] = (
            None
        )

        # Background cleanup thread
        self._shutdown_event = threading.Event()
        self._cleanup_thread: Optional[threading.Thread] = None
        self._start_cleanup_thread()

        logger.debug("AgentCollaborationManager initialized")

    def _start_cleanup_thread(self) -> None:
        """Start background thread for session cleanup."""
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop, daemon=True, name="CollaborationCleanup"
        )
        self._cleanup_thread.start()

    def _cleanup_loop(self) -> None:
        """Background loop for cleaning up timed-out sessions."""
        while not self._shutdown_event.wait(timeout=60.0):
            try:
                self._cleanup_stale_sessions()
            except Exception as e:
                logger.error(f"[Collaboration] Cleanup error: {e}")

    def _cleanup_stale_sessions(self) -> None:
        """Clean up timed-out and completed sessions."""
        current_time = time.time()
        sessions_to_remove = []

        with self._lock:
            for session_id, session in self._sessions.items():
                # Check for timeout
                if (
                    session.is_active
                    and (current_time - session.start_time) > self._session_timeout
                ):
                    session.status = CollaborationStatus.TIMEOUT
                    session.end_time = current_time
                    session.error = "Session timed out"
                    self._stats["timeout_collaborations"] += 1
                    logger.warning(f"[Collaboration] Session {session_id} timed out")

                # Remove completed sessions older than 1 hour
                if not session.is_active and session.end_time:
                    if (current_time - session.end_time) > 3600:
                        sessions_to_remove.append(session_id)

            for session_id in sessions_to_remove:
                del self._sessions[session_id]

        if sessions_to_remove:
            logger.debug(
                f"[Collaboration] Cleaned up {len(sessions_to_remove)} stale sessions"
            )

    def set_agent_pool(self, agent_pool: AgentPool) -> None:
        """
        Set the agent pool reference.

        Args:
            agent_pool: Agent pool instance for task submission
        """
        self._agent_pool = agent_pool
        logger.debug("Agent pool reference set")

    def set_telemetry_recorder(self, recorder: Any) -> None:
        """
        Set the telemetry recorder reference.

        Args:
            recorder: Telemetry recorder for AI interaction logging
        """
        self._telemetry_recorder = recorder
        logger.debug("Telemetry recorder reference set")

    def set_on_message_callback(self, callback: Callable[[AgentMessage], None]) -> None:
        """
        Set callback for when messages are sent.

        Args:
            callback: Function to call with each AgentMessage
        """
        self._on_message_callback = callback

    def set_on_complete_callback(
        self, callback: Callable[[CollaborationSession], None]
    ) -> None:
        """
        Set callback for when collaboration completes.

        Args:
            callback: Function to call with completed CollaborationSession
        """
        self._on_complete_callback = callback

    def start_collaboration(
        self,
        query: str,
        agent_types: List[str],
        collaboration_type: CollaborationType = CollaborationType.PARALLEL,
        context: Optional[Dict[str, Any]] = None,
    ) -> CollaborationSession:
        """
        Start a multi-agent collaboration session.

        Creates a new session, submits initial tasks to agents, and
        begins tracking inter-agent communication.

        Args:
            query: The original query to process
            agent_types: List of agent types to involve
            collaboration_type: Type of collaboration pattern
            context: Additional context for the collaboration

        Returns:
            CollaborationSession for tracking progress

        Raises:
            ValueError: If too many agents requested or session limit reached
        """
        # Validate inputs
        if not query:
            raise ValueError("Query cannot be empty")

        if not agent_types:
            raise ValueError("At least one agent type must be specified")

        if len(agent_types) > MAX_AGENTS_PER_COLLABORATION:
            raise ValueError(
                f"Maximum {MAX_AGENTS_PER_COLLABORATION} agents per collaboration"
            )

        with self._lock:
            # Check session limit
            active_count = sum(1 for s in self._sessions.values() if s.is_active)
            if active_count >= self._max_concurrent_sessions:
                raise ValueError(
                    f"Maximum concurrent sessions ({self._max_concurrent_sessions}) reached"
                )

            # Create session
            session_id = f"collab_{uuid.uuid4().hex[:12]}"

            session = CollaborationSession(
                session_id=session_id,
                original_query=query,
                agents_involved=list(agent_types),
                collaboration_type=collaboration_type,
                status=CollaborationStatus.IN_PROGRESS,
            )

            self._sessions[session_id] = session
            self._stats["total_collaborations"] += 1

            # Update average agents per collaboration
            total = self._stats["total_collaborations"]
            current_avg = self._stats["avg_agents_per_collaboration"]
            self._stats["avg_agents_per_collaboration"] = (
                current_avg * (total - 1) + len(agent_types)
            ) / total

        logger.info(
            f"[Collaboration] Started session {session_id} with agents: {agent_types}, "
            f"type: {collaboration_type.value}"
        )

        # Submit initial tasks to each agent
        self._submit_initial_tasks(session, query, context)

        return session

    def _submit_initial_tasks(
        self,
        session: CollaborationSession,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Submit initial tasks to all agents in the collaboration.

        Args:
            session: The collaboration session
            query: The query to process
            context: Additional context
        """
        for agent_type in session.agents_involved:
            # Create initial message to each agent
            msg = AgentMessage(
                message_id=f"{session.session_id}_{agent_type}_init",
                sender_agent="orchestrator",
                receiver_agent=agent_type,
                message_type=MessageType.REQUEST,
                content={
                    "query": query,
                    "task": f"Process query from {agent_type} perspective",
                    "context": context or {},
                    "collaboration_type": session.collaboration_type.value,
                },
                context=query,
            )

            # Record message
            with self._lock:
                if len(session.messages) < MAX_MESSAGES_PER_SESSION:
                    session.messages.append(msg)

            # Submit to agent pool
            self._submit_message_to_pool(msg, session.session_id)

            # Record as AI interaction
            self._record_ai_interaction(msg, session)

    def _submit_message_to_pool(self, msg: AgentMessage, session_id: str) -> None:
        """
        Submit a message as a task to the agent pool.

        Args:
            msg: The agent message to submit
            session_id: The collaboration session ID
        """
        if not self._agent_pool:
            logger.warning(
                "[Collaboration] Agent pool not available for task submission"
            )
            return

        try:
            # Import here to avoid circular import
            try:
                from ..orchestrator.agent_lifecycle import AgentCapability

                capability_map = {
                    "perception": AgentCapability.PERCEPTION,
                    "reasoning": AgentCapability.REASONING,
                    "planning": AgentCapability.PLANNING,
                    "execution": AgentCapability.EXECUTION,
                    "learning": AgentCapability.LEARNING,
                    "general": AgentCapability.GENERAL,
                }

                capability = capability_map.get(
                    msg.receiver_agent, AgentCapability.GENERAL
                )
            except ImportError:
                logger.debug("AgentCapability not available, using string capability")
                capability = msg.receiver_agent

            # Convert message to task graph
            task_graph = msg.to_task()

            # Submit to agent pool
            job_id = self._agent_pool.submit_job(
                graph=task_graph,
                parameters={
                    "message_id": msg.message_id,
                    "session_id": session_id,
                    "sender": msg.sender_agent,
                    "receiver": msg.receiver_agent,
                    "message_type": msg.message_type.value,
                },
                priority=2,
                capability_required=capability,
                timeout_seconds=15.0,
            )

            logger.debug(
                f"[Collaboration] Submitted message {msg.message_id} as job {job_id}"
            )

            with self._lock:
                self._stats["total_messages"] += 1

        except Exception as e:
            logger.error(
                f"[Collaboration] Failed to submit message: {e}", exc_info=True
            )

    def _record_ai_interaction(
        self, msg: AgentMessage, session: CollaborationSession
    ) -> None:
        """
        Record an inter-agent message as AI interaction telemetry.

        Args:
            msg: The agent message
            session: The collaboration session
        """
        with self._lock:
            self._stats["total_interactions"] += 1

        if self._telemetry_recorder:
            try:
                # Try to use the recorder's method
                if hasattr(self._telemetry_recorder, "record_ai_interaction"):
                    self._telemetry_recorder.record_ai_interaction(
                        interaction_type="agent_communication",
                        sender=msg.sender_agent,
                        receiver=msg.receiver_agent,
                        session_id=session.session_id,
                        message_type=msg.message_type.value,
                        query=session.original_query,
                    )
            except Exception as e:
                logger.debug(f"[Collaboration] Telemetry recording failed: {e}")

        # Call message callback if set
        if self._on_message_callback:
            try:
                self._on_message_callback(msg)
            except Exception as e:
                logger.error(f"[Collaboration] Message callback failed: {e}")

    def send_message(
        self,
        session_id: str,
        sender_agent: str,
        receiver_agent: str,
        message_type: MessageType,
        content: Dict[str, Any],
        parent_message_id: Optional[str] = None,
    ) -> Optional[AgentMessage]:
        """
        Send a message from one agent to another within a collaboration.

        Args:
            session_id: The collaboration session ID
            sender_agent: Agent sending the message
            receiver_agent: Agent receiving the message
            message_type: Type of message
            content: Message content
            parent_message_id: ID of message this is responding to

        Returns:
            The created AgentMessage, or None if session not found
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                logger.warning(f"[Collaboration] Session {session_id} not found")
                return None

            if not session.is_active:
                logger.warning(f"[Collaboration] Session {session_id} is not active")
                return None

            if len(session.messages) >= MAX_MESSAGES_PER_SESSION:
                logger.warning(
                    f"[Collaboration] Session {session_id} message limit reached"
                )
                return None

        msg = AgentMessage(
            message_id=f"{session_id}_{uuid.uuid4().hex[:8]}",
            sender_agent=sender_agent,
            receiver_agent=receiver_agent,
            message_type=message_type,
            content=content,
            context=session.original_query,
            parent_message_id=parent_message_id,
        )

        with self._lock:
            session.messages.append(msg)

        # Submit to agent pool
        self._submit_message_to_pool(msg, session_id)

        # Record as AI interaction
        self._record_ai_interaction(msg, session)

        return msg

    def record_agent_result(
        self, session_id: str, agent_type: str, result: Any
    ) -> None:
        """
        Record a result from an agent in the collaboration.

        Args:
            session_id: The collaboration session ID
            agent_type: The agent that produced the result
            result: The result from the agent
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return

            session.results[agent_type] = result

            # Check if all agents have responded
            if len(session.results) >= len(session.agents_involved):
                self._complete_collaboration(session)

    def _complete_collaboration(self, session: CollaborationSession) -> None:
        """
        Complete a collaboration session and synthesize results.

        Args:
            session: The collaboration session to complete
        """
        session.status = CollaborationStatus.COMPLETED
        session.end_time = time.time()

        with self._lock:
            self._stats["successful_collaborations"] += 1

            # Update average duration
            total = self._stats["successful_collaborations"]
            current_avg = self._stats["avg_duration_seconds"]
            self._stats["avg_duration_seconds"] = (
                current_avg * (total - 1) + session.duration_seconds
            ) / total

        logger.info(
            f"[Collaboration] Session {session.session_id} completed. "
            f"Duration: {session.duration_seconds:.2f}s, "
            f"Interactions: {session.interaction_count}"
        )

        # Request synthesis from reasoning agent if applicable
        if session.collaboration_type in (
            CollaborationType.PARALLEL,
            CollaborationType.DEBATE,
            CollaborationType.SYNTHESIS,
        ):
            synthesis_msg = AgentMessage(
                message_id=f"{session.session_id}_synthesis",
                sender_agent="orchestrator",
                receiver_agent="reasoning",
                message_type=MessageType.SYNTHESIS,
                content={
                    "results": session.results,
                    "query": session.original_query,
                    "task": "Synthesize results from all agents",
                    "agents": session.agents_involved,
                },
                context=session.original_query,
            )

            with self._lock:
                if len(session.messages) < MAX_MESSAGES_PER_SESSION:
                    session.messages.append(synthesis_msg)

            self._record_ai_interaction(synthesis_msg, session)

        # Call completion callback
        if self._on_complete_callback:
            try:
                self._on_complete_callback(session)
            except Exception as e:
                logger.error(f"[Collaboration] Completion callback failed: {e}")

    def fail_collaboration(self, session_id: str, reason: str) -> None:
        """
        Mark a collaboration session as failed.

        Args:
            session_id: The collaboration session ID
            reason: Reason for failure
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.status = CollaborationStatus.FAILED
                session.end_time = time.time()
                session.error = reason
                self._stats["failed_collaborations"] += 1

        logger.error(f"[Collaboration] Session {session_id} failed: {reason}")

    def trigger_debate(
        self, session_id: str, topic: str, agents: List[str]
    ) -> List[AgentMessage]:
        """
        Trigger a debate between agents on a topic.

        Each agent presents their perspective, generating AI-to-AI
        interaction data.

        Args:
            session_id: The collaboration session ID
            topic: Topic to debate
            agents: Agents to participate in debate

        Returns:
            List of debate messages created
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session or not session.is_active:
                return []

        messages = []

        # Each agent presents their perspective
        for agent in agents:
            msg = AgentMessage(
                message_id=f"{session_id}_debate_{agent}_{uuid.uuid4().hex[:4]}",
                sender_agent=agent,
                receiver_agent="all",
                message_type=MessageType.DEBATE,
                content={
                    "topic": topic,
                    "perspective": f"{agent} perspective on: {topic}",
                    "debate_round": len(
                        [
                            m
                            for m in session.messages
                            if m.message_type == MessageType.DEBATE
                        ]
                    )
                    + 1,
                },
                context=session.original_query,
            )

            with self._lock:
                if len(session.messages) < MAX_MESSAGES_PER_SESSION:
                    session.messages.append(msg)

            messages.append(msg)
            self._record_ai_interaction(msg, session)

        logger.info(
            f"[Collaboration] Triggered debate with {len(agents)} agents "
            f"on topic: {topic[:50]}..."
        )

        return messages

    def trigger_vote(
        self, session_id: str, options: List[str], agents: List[str]
    ) -> List[AgentMessage]:
        """
        Trigger a vote among agents.

        Each agent casts a vote, generating AI-to-AI interaction data.

        Args:
            session_id: The collaboration session ID
            options: Options to vote on
            agents: Agents to participate in voting

        Returns:
            List of vote messages created
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session or not session.is_active:
                return []

        messages = []

        # Each agent casts a vote
        for agent in agents:
            msg = AgentMessage(
                message_id=f"{session_id}_vote_{agent}_{uuid.uuid4().hex[:4]}",
                sender_agent=agent,
                receiver_agent="orchestrator",
                message_type=MessageType.VOTE,
                content={
                    "options": options,
                    "voter": agent,
                    "vote_round": len(
                        [
                            m
                            for m in session.messages
                            if m.message_type == MessageType.VOTE
                        ]
                    )
                    + 1,
                },
                context=session.original_query,
            )

            with self._lock:
                if len(session.messages) < MAX_MESSAGES_PER_SESSION:
                    session.messages.append(msg)

            messages.append(msg)
            self._record_ai_interaction(msg, session)

        logger.info(
            f"[Collaboration] Triggered vote with {len(agents)} agents "
            f"on {len(options)} options"
        )

        return messages

    def get_session(self, session_id: str) -> Optional[CollaborationSession]:
        """
        Get a collaboration session by ID.

        Args:
            session_id: The session ID to retrieve

        Returns:
            CollaborationSession or None if not found
        """
        with self._lock:
            return self._sessions.get(session_id)

    def get_active_sessions(self) -> List[CollaborationSession]:
        """
        Get all active collaboration sessions.

        Returns:
            List of active CollaborationSession objects
        """
        with self._lock:
            return [s for s in self._sessions.values() if s.is_active]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive collaboration statistics.

        Returns:
            Dictionary with collaboration counts and metrics
        """
        with self._lock:
            stats = dict(self._stats)
            stats["active_sessions"] = len(
                [s for s in self._sessions.values() if s.is_active]
            )
            stats["total_sessions"] = len(self._sessions)
            return stats

    def shutdown(self) -> None:
        """Shutdown the collaboration manager."""
        self._shutdown_event.set()
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5.0)
        logger.debug("AgentCollaborationManager shutdown complete")


# ============================================================
# SINGLETON PATTERN
# ============================================================

_global_manager: Optional[AgentCollaborationManager] = None
_manager_lock = threading.Lock()


def get_collaboration_manager() -> AgentCollaborationManager:
    """
    Get or create the global collaboration manager (thread-safe singleton).

    Returns:
        AgentCollaborationManager instance
    """
    global _global_manager

    if _global_manager is None:
        with _manager_lock:
            if _global_manager is None:
                _global_manager = AgentCollaborationManager()
                logger.debug("Global AgentCollaborationManager instance created")

    return _global_manager


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================


def trigger_agent_collaboration(
    query: str,
    agent_types: List[str],
    collaboration_type: CollaborationType = CollaborationType.PARALLEL,
    context: Optional[Dict[str, Any]] = None,
) -> CollaborationSession:
    """
    Trigger multi-agent collaboration for a query.

    When a query needs multiple perspectives:
    1. Creates collaboration session
    2. Submits subtasks to different agents
    3. Agents process and return results
    4. Reasoning agent synthesizes
    5. Records inter-agent communication as AI learning data

    Args:
        query: The query to process collaboratively
        agent_types: List of agent types to involve
        collaboration_type: Type of collaboration pattern
        context: Additional context

    Returns:
        CollaborationSession for tracking

    Example:
        session = trigger_agent_collaboration(
            query="Analyze and plan approach",
            agent_types=["perception", "reasoning", "planning"]
        )

        # Wait for completion
        while session.is_active:
            time.sleep(0.5)

        print(session.results)
    """
    manager = get_collaboration_manager()
    return manager.start_collaboration(query, agent_types, collaboration_type, context)
