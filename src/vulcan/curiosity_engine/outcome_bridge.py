"""
SQLite Bridge for Cross-Process Query Outcome Sharing.

Part of the VULCAN-AGI system.

This module provides a SQLite-based communication channel between the main
query processing pipeline and the CuriosityEngine subprocess. The main process
writes query outcomes, and the subprocess reads them for analysis and gap
detection.

Key Features:
    - Thread-safe SQLite operations with proper connection management
    - Automatic database initialization with idempotent schema creation
    - TTL-based cleanup to prevent unbounded database growth
    - Analysis functions for gap detection from outcomes
    - Comprehensive statistics tracking for monitoring
    - Batch operations for efficient bulk processing

Performance Characteristics:
    - Uses WAL mode for concurrent read/write access
    - Bounded queries with LIMIT to prevent unbounded result sets
    - Connection pooling via context manager pattern
    - Index-optimized queries for timestamp and status lookups

Database Schema:
    query_outcomes table:
        - id: INTEGER PRIMARY KEY AUTOINCREMENT
        - query_id: TEXT UNIQUE (e.g., "q_abc123")
        - timestamp: TEXT NOT NULL (ISO 8601 format)
        - status: TEXT NOT NULL ("success", "error", "timeout")
        - routing_time_ms: REAL (time spent routing)
        - total_time_ms: REAL (total processing time)
        - complexity: REAL (0.0 to 1.0)
        - query_type: TEXT (reasoning, perception, planning, etc.)
        - tasks: INTEGER DEFAULT 1 (agent tasks created)
        - error_type: TEXT (error classification if failed)
        - processed: INTEGER DEFAULT 0 (learning system flag)
        - created_at: REAL (Unix timestamp)

Usage:
    # In main query handler (after query completion):
    from vulcan.curiosity_engine.outcome_bridge import record_query_outcome

    success = record_query_outcome(
        query_id="q_abc123",
        status="success",
        routing_time_ms=150.0,
        total_time_ms=2500.0,
        complexity=0.45,
        query_type="reasoning",
    )

    # In CuriosityEngine subprocess (during learning cycle):
    from vulcan.curiosity_engine.outcome_bridge import (
        get_recent_outcomes,
        analyze_outcomes_for_gaps,
        get_outcome_statistics,
    )

    outcomes = get_recent_outcomes(minutes=60, limit=500)
    gaps = analyze_outcomes_for_gaps(outcomes)
    stats = get_outcome_statistics()

    # Periodic cleanup to prevent unbounded growth:
    from vulcan.curiosity_engine.outcome_bridge import cleanup_old_outcomes
    deleted = cleanup_old_outcomes(days=7)

Thread Safety:
    All functions are thread-safe. Database connections are managed per-call
    with proper cleanup. Module-level initialization uses double-checked
    locking for thread-safe lazy initialization.

Error Handling:
    All functions return sensible defaults on error (empty lists, False,
    default statistics) and log warnings. Exceptions are caught and
    handled gracefully to prevent cascading failures.
"""

import asyncio
import logging
import os
import re
import sqlite3
import statistics as stats_module
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration Constants
# =============================================================================

# Logging prefix for consistent output
LOG_PREFIX = "[OutcomeBridge]"

# Default database path - uses tempdir for cross-platform compatibility
# Can be overridden via VULCAN_OUTCOME_DB_PATH environment variable
DEFAULT_DB_PATH = Path(
    os.environ.get(
        "VULCAN_OUTCOME_DB_PATH",
        os.path.join(tempfile.gettempdir(), "vulcan_query_outcomes.db")
    )
)

# Database connection timeout in seconds
DB_TIMEOUT_SECONDS = 30.0

# Gap detection thresholds
SLOW_ROUTING_THRESHOLD_MS = 10000  # 10 seconds
COMPLEX_QUERY_THRESHOLD = 0.5  # Complexity score
COMPLEX_QUERY_TIME_THRESHOLD_MS = 30000  # 30 seconds
HIGH_ERROR_RATE_THRESHOLD = 0.1  # 10%
HIGH_VARIANCE_CV_THRESHOLD = 1.0  # Coefficient of variation
MIN_SAMPLES_FOR_VARIANCE = 10  # Minimum samples for variance analysis
MIN_SLOW_QUERIES_FOR_GAP = 3  # Minimum slow queries to report gap

# =============================================================================
# AUDIT FIX: Outcome Classification Constants
# =============================================================================
# These statuses represent legitimate outcomes that should NOT be counted as errors:
# - "success": Query completed successfully
# - "slow": Query completed but took longer than expected (still successful)
# - "low_confidence": Query completed but reasoning paradigm had low certainty (still successful)
# - "partial": Query partially completed (may be successful for complex queries)
#
# Only these statuses should be counted as TRUE errors:
# - "error": Actual error during processing
# - "timeout": Query exceeded time limit (hard failure)
# - "cancelled": Query was cancelled externally
# - "no_result": No result was produced (hard failure)
#
# CRITICAL: Different reasoning paradigms (causal, symbolic, probabilistic, analogical, 
# multimodal) are COMPLEMENTARY, not redundant. They SHOULD produce different outputs.
# A low consensus between paradigms is NOT an error - it's expected behavior.

# Statuses that are NOT errors (query completed in some form)
NON_ERROR_STATUSES = frozenset({
    "success",
    "slow",           # Completed but slow - not an error
    "low_confidence", # Completed with low confidence - not an error  
    "partial",        # Partially completed - may be acceptable for complex queries
})

# Statuses that ARE true errors (query failed to complete)
TRUE_ERROR_STATUSES = frozenset({
    "error",
    "timeout",
    "cancelled",
    "no_result",
    "execution_failed",
})

# Cleanup defaults
DEFAULT_RETENTION_DAYS = 7
DEFAULT_QUERY_LIMIT = 500

# =============================================================================
# ANSWER QUALITY ASSESSMENT
# =============================================================================
# The curiosity engine was blind to quality issues - it measured "completion"
# (did the query return something) but not "quality" (was the answer useful).
# 
# This fix adds quality assessment so the engine can detect:
# - Engine rejection patterns ("does not involve probability")
# - Raw data dumps instead of user-facing answers
# - Misclassified queries (e.g., creative queries as introspection)
# - Low-confidence responses that aren't useful
#
# Quality values:
#   "good"    - Answer appears useful and user-facing
#   "partial" - Answer was produced but may not be complete
#   "failed"  - Answer contains known failure patterns
#   None      - Quality not assessed (no response_text provided)

# Patterns that indicate a failed/useless answer
ANSWER_QUALITY_FAILURE_PATTERNS = frozenset({
    "does not involve probability",
    "does not appear to be",
    "not applicable",
    "unable to",
    "cannot process",
    "not supported",
    "query type not recognized",
    "no matching handler",
    "fallback response",
    # Parse error patterns - symbolic engine failures
    "parse error",
    "parser failed",
    "unexpected token",
    "advanced parser failed",
    "standard parser failed",
    # NL to Logic conversion failures
    "nl conversion failed",
    "nl to logic conversion",
    "conversion still failed",
    # General parsing failures
    "failed to parse",
    "syntax error",
    "malformed",
    # Proof failure patterns (from tool_selector diagnostic strings)
    "failed to prove",
    "error:",
})

# Patterns indicating raw data dumps (not user-facing)
# NOTE: We removed "confidence: 0." patterns because they cause false positives
# (e.g. "confidence: 0.85" contains "confidence: 0." as substring).
# Low confidence is now detected via the explicit proven: false check instead.
# NOTE: "error:" is in ANSWER_QUALITY_FAILURE_PATTERNS (immediate fail)
# so we don't need it here for the 2+ pattern threshold check.
RAW_DATA_DUMP_PATTERNS = frozenset({
    "proven:",
    "proof:",
    "method:",
    "applicable:",
    "result_dict:",
    "'error':",  # Dict key format (different from "error:" failure pattern)
    "validation:",
})

# Number of raw data patterns required to classify as a dump
RAW_DATA_PATTERN_THRESHOLD = 2

# Patterns indicating misclassification
MISCLASSIFICATION_PATTERNS = frozenset({
    "i recognize this as",
    "introspective query",
    "self-referential",
})

# Patterns in queries that indicate legitimate introspective questions
INTROSPECTION_QUERY_PATTERNS = frozenset({
    "yourself",
    "you are",
    "your capabilities",
    "what are you",
    "who are you",
    "about you",
    "tell me about yourself",
})

# Quality thresholds
MIN_USEFUL_RESPONSE_LENGTH = 50  # Minimum chars for a useful response
LOW_CONFIDENCE_THRESHOLD = 0.3  # Confidence below this suggests failure

# Gap detection thresholds for answer quality
LOW_QUALITY_SUCCESS_THRESHOLD = 0.5  # < 50% good answers triggers gap
HIGH_QUALITY_FAILURE_THRESHOLD = 0.2  # > 20% failed answers triggers gap


def assess_answer_quality(
    response_text: Optional[str],
    confidence: Optional[float] = None,
    query_text: Optional[str] = None,
) -> str:
    """
    Assess the quality of an answer to determine if it was actually useful.
    
    This function detects failure patterns in responses that would otherwise
    be counted as "successful" because they returned something. The curiosity
    engine uses this to identify knowledge gaps based on answer quality, not
    just completion status.
    
    Args:
        response_text: The response text to assess. If None, returns "unknown".
        confidence: Optional confidence score (0.0 to 1.0) from the reasoning engine.
        query_text: Optional original query text for context-aware assessment.
    
    Returns:
        Quality classification:
        - "good": Answer appears useful and user-facing
        - "partial": Answer was produced but may be incomplete
        - "failed": Answer contains known failure patterns
        - "unknown": Quality could not be assessed (no response_text)
    
    Example:
        >>> assess_answer_quality("The answer is 42")
        'good'
        >>> assess_answer_quality("does not involve probability concepts")
        'failed'
        >>> assess_answer_quality("confidence: 0.1, proven: false")
        'failed'
    """
    if response_text is None:
        return "unknown"
    
    response_lower = response_text.lower().strip()
    
    # Check for empty or very short responses
    if len(response_lower) < 10:
        return "failed"
    
    # Check for known failure patterns
    for pattern in ANSWER_QUALITY_FAILURE_PATTERNS:
        if pattern in response_lower:
            return "failed"
    
    # Check for raw data dumps (not user-facing responses)
    raw_pattern_count = sum(
        1 for pattern in RAW_DATA_DUMP_PATTERNS
        if pattern in response_lower
    )
    if raw_pattern_count >= RAW_DATA_PATTERN_THRESHOLD:
        # Multiple raw data patterns suggest a dump, not a user-facing answer
        return "failed"
    
    # Check for explicit failure indicators in structured responses
    # "proven: false" with low confidence is a failed response
    if "proven: false" in response_lower or "proven:false" in response_lower:
        # If confidence is provided and low, it's clearly a failure
        if confidence is not None and confidence < 0.5:
            return "failed"
        # Try to extract confidence from response text if not provided
        # Pattern: "confidence: X.XX" or "confidence:X.XX"
        # Use \d+(?:\.\d+)? to match valid decimal numbers only
        conf_match = re.search(r'confidence[:\s]+(\d+(?:\.\d+)?)', response_lower)
        if conf_match:
            try:
                extracted_conf = float(conf_match.group(1))
                if extracted_conf < 0.5:
                    return "failed"
            except ValueError:
                pass
        # Even without explicit confidence, proven: false is a partial success at best
        return "partial"
    
    # Check for misclassification patterns
    for pattern in MISCLASSIFICATION_PATTERNS:
        if pattern in response_lower:
            # Check if query was actually asking about introspection
            if query_text:
                query_lower = query_text.lower()
                is_introspective_query = any(
                    indicator in query_lower 
                    for indicator in INTROSPECTION_QUERY_PATTERNS
                )
                if not is_introspective_query:
                    return "failed"
    
    # Check for low confidence
    if confidence is not None and confidence < LOW_CONFIDENCE_THRESHOLD:
        return "partial"
    
    # Check for minimal length (might be an error message)
    if len(response_lower) < MIN_USEFUL_RESPONSE_LENGTH:
        return "partial"
    
    return "good"


# =============================================================================
# Enums and Data Classes
# =============================================================================

class QueryStatus(Enum):
    """
    Enumeration of query outcome statuses.
    
    Used for type-safe status handling and validation.
    """
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class GapType(Enum):
    """
    Enumeration of detectable knowledge gap types.
    
    Each type represents a different category of performance issue
    that the learning system should address.
    """
    SLOW_ROUTING = "slow_routing"
    COMPLEX_QUERY_HANDLING = "complex_query_handling"
    HIGH_ERROR_RATE = "high_error_rate"
    ROUTING_VARIANCE = "routing_variance"
    LOW_ANSWER_QUALITY = "low_answer_quality"
    ANSWER_FAILURE_PATTERNS = "answer_failure_patterns"


@dataclass
class OutcomeStatistics:
    """
    Statistics computed from query outcomes for monitoring.
    
    This dataclass provides aggregate metrics for monitoring system
    health and identifying performance trends.
    
    Attributes:
        total: Total number of outcomes in database
        unprocessed: Number of outcomes not yet processed by learning system
        avg_routing_ms: Average routing time in milliseconds
        max_routing_ms: Maximum routing time observed
        slow_routing_count: Count of queries exceeding slow threshold
        error_count: Total error count in recent window
        success_rate: Success rate as ratio (0.0 to 1.0)
        quality_good_count: Count of answers marked as "good" quality
        quality_failed_count: Count of answers marked as "failed" quality
        quality_success_rate: Ratio of good quality answers to total with quality
        
    Example:
        >>> stats = get_outcome_statistics()
        >>> print(f"Success rate: {stats.success_rate:.1%}")
        >>> print(f"Quality success rate: {stats.quality_success_rate:.1%}")
        >>> print(f"Slow queries: {stats.slow_routing_count}")
    """
    
    total: int
    unprocessed: int
    avg_routing_ms: float
    max_routing_ms: float
    slow_routing_count: int
    error_count: int
    success_rate: float
    quality_good_count: int = 0
    quality_failed_count: int = 0
    quality_success_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert statistics to dictionary for serialization.
        
        Returns:
            Dictionary representation of statistics.
        """
        return {
            "total": self.total,
            "unprocessed": self.unprocessed,
            "avg_routing_ms": self.avg_routing_ms,
            "max_routing_ms": self.max_routing_ms,
            "slow_routing_count": self.slow_routing_count,
            "error_count": self.error_count,
            "success_rate": self.success_rate,
            "quality_good_count": self.quality_good_count,
            "quality_failed_count": self.quality_failed_count,
            "quality_success_rate": self.quality_success_rate,
        }


@dataclass
class DetectedGap:
    """
    Represents a detected knowledge gap from outcome analysis.
    
    Attributes:
        gap_type: Type of gap detected
        description: Human-readable description
        severity: Severity score (0.0 to 1.0)
        evidence: List of supporting evidence (query IDs, metrics)
        suggested_action: Recommended action to address the gap
    """
    
    gap_type: str
    description: str
    severity: float
    evidence: List[str] = field(default_factory=list)
    suggested_action: str = ""
    
    def __post_init__(self) -> None:
        """Validate and clamp severity to valid range."""
        self.severity = max(0.0, min(1.0, self.severity))
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert gap to dictionary for serialization.
        
        Returns:
            Dictionary representation of the gap.
        """
        return {
            "gap_type": self.gap_type,
            "description": self.description,
            "severity": self.severity,
            "evidence": self.evidence,
            "suggested_action": self.suggested_action,
        }


# =============================================================================
# OutcomeBridge Class for Learning System Integration
# =============================================================================

class OutcomeBridge:
    """
    Bridge between query outcomes and the learning system.
    
    This class provides the connection between recorded query outcomes
    and the UnifiedLearningSystem for feedback-driven learning.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern for OutcomeBridge."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.learning_system = None
        self._outcome_queue: List[Dict[str, Any]] = []
        self._initialized = True
        logger.info(f"{LOG_PREFIX} OutcomeBridge initialized")
    
    def set_learning_system(self, learning_system: Any) -> None:
        """
        Connect to learning system for feedback loop.
        
        Args:
            learning_system: The UnifiedLearningSystem instance
        """
        self.learning_system = learning_system
        logger.info(f"{LOG_PREFIX} Connected to UnifiedLearningSystem")
        
        # Process any queued outcomes
        if self._outcome_queue:
            logger.info(f"{LOG_PREFIX} Processing {len(self._outcome_queue)} queued outcomes")
            for outcome in self._outcome_queue:
                self._send_to_learning_sync(outcome)
            self._outcome_queue.clear()
    
    def record(
        self,
        query_id: str,
        status: str,
        routing_ms: float,
        total_ms: float,
        complexity: float,
        query_type: str,
        tools: Optional[List[str]] = None,
    ) -> bool:
        """
        Record a query outcome and send to learning system.
        
        Args:
            query_id: Unique query identifier
            status: Query status ("success", "error", "timeout")
            routing_ms: Time spent routing in milliseconds
            total_ms: Total processing time in milliseconds
            complexity: Query complexity score (0.0 to 1.0)
            query_type: Type of query (reasoning, perception, etc.)
            tools: List of tools used for the query
            
        Returns:
            True if recording and learning system update succeeded
        """
        tools = tools if tools is not None else []
        
        # FIX MINOR-1: Use DEBUG level here since record_query_outcome already logs at INFO
        # This prevents duplicate "[QueryOutcome] Recorded:" log entries
        logger.debug(
            f"[QueryOutcome] Bridge record: {query_id}, status={status}, "
            f"routing={routing_ms:.0f}ms, total={total_ms:.0f}ms, "
            f"complexity={complexity:.2f}, type={query_type}, tools={tools}"
        )
        
        # Create outcome object
        outcome = {
            'query_id': query_id,
            'status': status,
            'routing_ms': routing_ms,
            'total_ms': total_ms,
            'complexity': complexity,
            'query_type': query_type,
            'tools': tools,
            'timestamp': time.time(),
        }
        
        # Send to learning system
        if self.learning_system:
            try:
                # Use asyncio if in async context, otherwise call sync
                try:
                    loop = asyncio.get_running_loop()
                    # We're in an async context, schedule the task
                    asyncio.create_task(self._send_to_learning_async(outcome))
                    logger.info(f"[QueryOutcome] Sent to learning system for processing")
                except RuntimeError:
                    # No running loop, use sync method
                    self._send_to_learning_sync(outcome)
                    logger.info(f"[QueryOutcome] Sent to learning system for processing")
                return True
            except Exception as e:
                logger.warning(f"[QueryOutcome] Failed to send to learning: {e}")
                self._outcome_queue.append(outcome)  # Queue for retry
                return False
        else:
            logger.debug(f"[QueryOutcome] No learning system connected - outcome not processed for learning")
            self._outcome_queue.append(outcome)  # Queue for later
            return True  # Still return True since SQLite recording worked
    
    async def _send_to_learning_async(self, outcome: Dict[str, Any]) -> None:
        """Async send outcome to learning system."""
        try:
            if hasattr(self.learning_system, 'process_outcome'):
                await self.learning_system.process_outcome(outcome)
        except Exception as e:
            logger.error(f"{LOG_PREFIX} Learning processing failed: {e}")
    
    def _send_to_learning_sync(self, outcome: Dict[str, Any]) -> None:
        """Sync send outcome to learning system (when not in async context).
        
        Note: This method handles the case where we need to call an async
        method from a sync context. It uses asyncio.run() when possible
        for proper event loop lifecycle management.
        """
        try:
            if hasattr(self.learning_system, 'process_outcome'):
                # Check if process_outcome is a coroutine function before calling
                if asyncio.iscoroutinefunction(self.learning_system.process_outcome):
                    coro = self.learning_system.process_outcome(outcome)
                    # Use asyncio.run() for proper event loop lifecycle management
                    # This is cleaner than manually creating/managing event loops
                    try:
                        asyncio.run(coro)
                    except RuntimeError as e:
                        # asyncio.run() can fail if called from inside a running loop
                        # In that case, fall back to manual loop management
                        if "cannot be called from a running event loop" in str(e):
                            logger.debug(f"{LOG_PREFIX} asyncio.run() failed, using loop.run_until_complete()")
                            loop = asyncio.new_event_loop()
                            try:
                                coro = self.learning_system.process_outcome(outcome)
                                loop.run_until_complete(coro)
                            finally:
                                loop.close()
                        else:
                            raise
                else:
                    # Not a coroutine, call directly
                    self.learning_system.process_outcome(outcome)
        except Exception as e:
            logger.error(f"{LOG_PREFIX} Learning processing failed: {e}")


def get_outcome_bridge() -> OutcomeBridge:
    """Get the singleton OutcomeBridge instance."""
    return OutcomeBridge()


# =============================================================================
# Module-Level State
# =============================================================================

# Thread-local storage for database connections (not currently used but available)
_local = threading.local()

# Module-level lock for database initialization
_init_lock = threading.Lock()
_db_initialized = False


# =============================================================================
# Database Connection Management
# =============================================================================

@contextmanager
def _get_db(db_path: Optional[Path] = None) -> Generator[sqlite3.Connection, None, None]:
    """
    Get database connection with proper cleanup.
    
    Uses WAL mode for better concurrent access and sets a reasonable
    timeout for busy waiting. Connections are created fresh for each
    context to ensure thread safety.
    
    Args:
        db_path: Optional path to database file. Defaults to DEFAULT_DB_PATH.
    
    Yields:
        SQLite connection with row factory set to sqlite3.Row
    
    Example:
        >>> with _get_db() as conn:
        ...     cursor = conn.execute("SELECT COUNT(*) FROM query_outcomes")
        ...     count = cursor.fetchone()[0]
    """
    path = db_path or DEFAULT_DB_PATH
    conn = sqlite3.connect(str(path), timeout=DB_TIMEOUT_SECONDS)
    conn.row_factory = sqlite3.Row
    
    # Enable WAL mode for better concurrent access
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    except sqlite3.Error:
        pass  # Ignore if already set or not supported
    
    try:
        yield conn
    finally:
        conn.close()


def _init_db(db_path: Optional[Path] = None) -> bool:
    """
    Initialize database schema (idempotent).
    
    Creates the query_outcomes table and necessary indexes if they don't exist.
    Thread-safe and can be called multiple times without side effects.
    Uses double-checked locking for efficient thread-safe initialization.
    
    Args:
        db_path: Optional path to database file. Defaults to DEFAULT_DB_PATH.
    
    Returns:
        True if initialization succeeded, False otherwise.
    
    Example:
        >>> success = _init_db()
        >>> if success:
        ...     print("Database ready")
    """
    global _db_initialized
    
    # If a custom db_path is provided, always initialize it (test scenario)
    # Only use the global flag for the default path
    if db_path is None:
        # Fast path if already initialized for default path
        if _db_initialized:
            return True
    
    with _init_lock:
        # Double-checked locking (only for default path)
        if db_path is None and _db_initialized:
            return True
        
        try:
            with _get_db(db_path) as conn:
                # Create main table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS query_outcomes (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        query_id TEXT UNIQUE,
                        timestamp TEXT NOT NULL,
                        status TEXT NOT NULL,
                        routing_time_ms REAL,
                        total_time_ms REAL,
                        complexity REAL,
                        query_type TEXT,
                        tasks INTEGER DEFAULT 1,
                        error_type TEXT,
                        processed INTEGER DEFAULT 0,
                        created_at REAL DEFAULT (strftime('%s', 'now')),
                        answer_quality TEXT DEFAULT NULL
                    )
                """)
                
                # Add answer_quality column if it doesn't exist (migration)
                try:
                    conn.execute(
                        "ALTER TABLE query_outcomes ADD COLUMN answer_quality TEXT DEFAULT NULL"
                    )
                except sqlite3.OperationalError as e:
                    # Only ignore duplicate column error, re-raise other schema issues
                    if "duplicate column name" not in str(e).lower():
                        raise
                
                # Create indexes for common query patterns
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_timestamp "
                    "ON query_outcomes(timestamp)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_processed "
                    "ON query_outcomes(processed)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_status "
                    "ON query_outcomes(status)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_query_type "
                    "ON query_outcomes(query_type)"
                )
                
                conn.commit()
            
            # Only set the global flag if using default path
            if db_path is None:
                _db_initialized = True
            logger.info(f"{LOG_PREFIX} Database initialized successfully")
            return True
            
        except sqlite3.Error as e:
            logger.warning(f"{LOG_PREFIX} Database initialization failed: {e}")
            return False


# =============================================================================
# Core CRUD Operations
# =============================================================================

def record_query_outcome(
    query_id: str,
    status: str,
    routing_time_ms: float,
    total_time_ms: float,
    complexity: float,
    query_type: str,
    tasks: int = 1,
    error_type: Optional[str] = None,
    db_path: Optional[Path] = None,
    tools: Optional[List[str]] = None,
    response_text: Optional[str] = None,
    answer_quality: Optional[str] = None,
    confidence: Optional[float] = None,
    query_text: Optional[str] = None,
) -> bool:
    """
    Record a query outcome to shared storage.
    
    Call this from the main process after each query completes. The outcome
    will be available for the CuriosityEngine subprocess to analyze during
    its learning cycles.
    
    Args:
        query_id: Unique query identifier (e.g., "q_abc123").
            Should be globally unique to prevent collisions.
        status: Query status. Valid values:
            - "success": Query completed successfully
            - "error": Query failed with an error
            - "timeout": Query exceeded time limit
            - "cancelled": Query was cancelled
        routing_time_ms: Time spent in query routing phase in milliseconds.
            This is the time to determine how to process the query.
        total_time_ms: Total query processing time in milliseconds.
            Includes routing, processing, and response generation.
        complexity: Query complexity score (0.0 to 1.0).
            Higher values indicate more complex queries.
        query_type: Type of query. Common values:
            - "reasoning": Logical reasoning queries
            - "perception": Pattern recognition queries
            - "planning": Multi-step planning queries
            - "execution": Action execution queries
            - "general": General knowledge queries
        tasks: Number of agent tasks created (default: 1).
            Useful for tracking query decomposition.
        error_type: Error classification if status is "error".
            Helps categorize and analyze failure patterns.
        db_path: Optional path to database file.
            Defaults to DEFAULT_DB_PATH.
        tools: List of tools used for the query.
            Enables learning system to track tool selection patterns.
        response_text: Optional response text for quality assessment.
            If provided and answer_quality is not set, quality will be computed.
        answer_quality: Optional pre-computed answer quality assessment.
            Values: "good", "partial", "failed", "unknown".
            If not provided but response_text is, quality will be computed.
        confidence: Optional confidence score (0.0 to 1.0) from reasoning engine.
            Used for quality assessment when response_text is provided.
        query_text: Optional original query text for context-aware quality assessment.
    
    Returns:
        True if recording succeeded, False otherwise.
    
    Example:
        >>> success = record_query_outcome(
        ...     query_id="q_abc123",
        ...     status="success",
        ...     routing_time_ms=150.0,
        ...     total_time_ms=2500.0,
        ...     complexity=0.45,
        ...     query_type="reasoning",
        ...     tools=["probabilistic", "causal"],
        ...     response_text="The answer is 42.",
        ... )
        >>> if success:
        ...     print("Outcome recorded")
    """
    # Validate inputs
    if not query_id:
        logger.warning(f"{LOG_PREFIX} Cannot record outcome with empty query_id")
        return False
    
    # Ensure database is initialized
    if not _init_db(db_path):
        return False
    
    # Compute answer quality if response_text provided but quality not set
    computed_quality = answer_quality
    if computed_quality is None and response_text is not None:
        computed_quality = assess_answer_quality(
            response_text=response_text,
            confidence=confidence,
            query_text=query_text,
        )
    
    try:
        with _get_db(db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO query_outcomes 
                (query_id, timestamp, status, routing_time_ms, total_time_ms, 
                 complexity, query_type, tasks, error_type, processed, answer_quality)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
                """,
                (
                    query_id,
                    datetime.utcnow().isoformat(),
                    status,
                    routing_time_ms,
                    total_time_ms,
                    complexity,
                    query_type,
                    tasks,
                    error_type,
                    computed_quality,
                ),
            )
            conn.commit()
        
        quality_str = f", quality={computed_quality}" if computed_quality else ""
        logger.info(
            f"[QueryOutcome] Recorded: {query_id}, status={status}, "
            f"routing={routing_time_ms:.0f}ms, total={total_time_ms:.0f}ms, "
            f"complexity={complexity:.2f}, type={query_type}, tools={tools or []}{quality_str}"
        )
        
        # Also send to learning system via OutcomeBridge
        try:
            bridge = get_outcome_bridge()
            bridge.record(
                query_id=query_id,
                status=status,
                routing_ms=routing_time_ms,
                total_ms=total_time_ms,
                complexity=complexity,
                query_type=query_type,
                tools=tools,
            )
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} OutcomeBridge record failed (non-critical): {e}")
        
        return True
        
    except sqlite3.Error as e:
        logger.warning(f"{LOG_PREFIX} Failed to record outcome: {e}")
        return False


def get_recent_outcomes(
    minutes: int = 60,
    limit: int = DEFAULT_QUERY_LIMIT,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Get recent outcomes for analysis.
    
    Call this from the CuriosityEngine subprocess during learning cycles
    to retrieve query outcomes for gap analysis. Results are ordered by
    timestamp descending (most recent first).
    
    Args:
        minutes: Look back window in minutes (default: 60).
            Only outcomes within this window are returned.
        limit: Maximum number of results (default: 500).
            Prevents unbounded result sets.
        db_path: Optional path to database file.
    
    Returns:
        List of outcome dictionaries with all fields. Each dictionary
        contains: id, query_id, timestamp, status, routing_time_ms,
        total_time_ms, complexity, query_type, tasks, error_type,
        processed, created_at.
    
    Example:
        >>> outcomes = get_recent_outcomes(minutes=60)
        >>> print(f"Found {len(outcomes)} outcomes")
        >>> for o in outcomes[:3]:
        ...     print(f"  {o['query_id']}: {o['status']}")
    """
    path = db_path or DEFAULT_DB_PATH
    
    # Check if database exists
    if not path.exists():
        logger.debug(f"{LOG_PREFIX} Database does not exist: {path}")
        return []
    
    cutoff = (datetime.utcnow() - timedelta(minutes=minutes)).isoformat()
    
    try:
        with _get_db(db_path) as conn:
            cursor = conn.execute(
                """
                SELECT * FROM query_outcomes 
                WHERE timestamp > ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (cutoff, limit),
            )
            rows = cursor.fetchall()
        
        outcomes = [dict(row) for row in rows]
        
        if outcomes:
            logger.info(
                f"{LOG_PREFIX} Loaded {len(outcomes)} recent outcomes "
                f"(last {minutes} min)"
            )
        
        return outcomes
        
    except sqlite3.Error as e:
        logger.warning(f"{LOG_PREFIX} Failed to load outcomes: {e}")
        return []


def get_recent_count(
    minutes: int = 5,
    db_path: Optional[Path] = None,
) -> int:
    """
    Get count of recent outcomes for wake trigger detection.
    
    This is a lightweight function used by CuriosityEngine to detect
    if new query outcomes have arrived since the last check, which
    serves as a wake trigger to exit backoff mode.
    
    Args:
        minutes: Look back window in minutes (default: 5).
        db_path: Optional path to database file.
    
    Returns:
        Count of outcomes in the specified time window.
    
    Example:
        >>> count = get_recent_count(minutes=5)
        >>> if count > last_count:
        ...     print("New outcomes detected!")
    """
    path = db_path or DEFAULT_DB_PATH
    
    # Check if database exists
    if not path.exists():
        return 0
    
    cutoff = (datetime.utcnow() - timedelta(minutes=minutes)).isoformat()
    
    try:
        with _get_db(db_path) as conn:
            cursor = conn.execute(
                """
                SELECT COUNT(*) FROM query_outcomes 
                WHERE timestamp > ?
                """,
                (cutoff,),
            )
            row = cursor.fetchone()
            return row[0] if row else 0
        
    except sqlite3.Error as e:
        logger.debug(f"{LOG_PREFIX} Failed to count recent outcomes: {e}")
        return 0


def get_unprocessed_outcomes(
    limit: int = 100,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Get outcomes that haven't been processed by the learning system.
    
    Returns outcomes in chronological order (oldest first) to ensure
    FIFO processing of the learning queue.
    
    Args:
        limit: Maximum number of results (default: 100).
        db_path: Optional path to database file.
    
    Returns:
        List of unprocessed outcome dictionaries.
    
    Example:
        >>> unprocessed = get_unprocessed_outcomes(limit=50)
        >>> print(f"Found {len(unprocessed)} unprocessed outcomes")
    """
    path = db_path or DEFAULT_DB_PATH
    
    if not path.exists():
        return []
    
    try:
        with _get_db(db_path) as conn:
            cursor = conn.execute(
                """
                SELECT * FROM query_outcomes 
                WHERE processed = 0
                ORDER BY timestamp ASC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()
        
        outcomes = [dict(row) for row in rows]
        
        if outcomes:
            logger.debug(
                f"{LOG_PREFIX} Found {len(outcomes)} unprocessed outcomes"
            )
        
        return outcomes
        
    except sqlite3.Error as e:
        logger.warning(f"{LOG_PREFIX} Failed to load unprocessed outcomes: {e}")
        return []


def mark_outcomes_processed(
    outcome_ids: List[int],
    db_path: Optional[Path] = None,
) -> bool:
    """
    Mark outcomes as processed by the learning system.
    
    Call this after the learning system has analyzed outcomes to prevent
    reprocessing. Uses parameterized queries to prevent SQL injection.
    
    Args:
        outcome_ids: List of outcome IDs (integers) to mark as processed.
            IDs are validated to ensure they are integers.
        db_path: Optional path to database file.
    
    Returns:
        True if successful, False otherwise.
    
    Example:
        >>> outcomes = get_unprocessed_outcomes()
        >>> ids = [o['id'] for o in outcomes]
        >>> success = mark_outcomes_processed(ids)
    """
    if not outcome_ids:
        return True  # Nothing to do
    
    # Validate all IDs are integers to prevent injection
    try:
        validated_ids = [int(id_) for id_ in outcome_ids]
    except (ValueError, TypeError) as e:
        logger.warning(f"{LOG_PREFIX} Invalid outcome IDs provided: {e}")
        return False
    
    try:
        with _get_db(db_path) as conn:
            # Use parameterized placeholders - count is derived from validated list
            placeholders = ",".join("?" for _ in validated_ids)
            conn.execute(
                f"UPDATE query_outcomes SET processed = 1 WHERE id IN ({placeholders})",
                validated_ids,
            )
            conn.commit()
        
        logger.debug(
            f"{LOG_PREFIX} Marked {len(validated_ids)} outcomes as processed"
        )
        return True
        
    except sqlite3.Error as e:
        logger.warning(f"{LOG_PREFIX} Failed to mark outcomes processed: {e}")
        return False


# =============================================================================
# Statistics and Analysis
# =============================================================================

def get_outcome_statistics(
    db_path: Optional[Path] = None,
) -> OutcomeStatistics:
    """
    Get statistics for monitoring and gap detection.
    
    Computes aggregate statistics from the outcomes database including
    success rates, average routing times, and error counts. Statistics
    are computed from the last hour of data for relevance.
    
    AUDIT FIX: The success_rate now correctly distinguishes between:
    - TRUE errors: "error", "timeout", "cancelled", "no_result"
    - NON-errors: "success", "slow", "low_confidence", "partial"
    
    The error_count only includes TRUE errors, not slow/low_confidence outcomes.
    
    Args:
        db_path: Optional path to database file.
    
    Returns:
        OutcomeStatistics dataclass with computed metrics.
        Returns default (zero) statistics if database is unavailable.
    
    Example:
        >>> stats = get_outcome_statistics()
        >>> print(f"Success rate: {stats.success_rate:.1%}")
        >>> print(f"Avg routing: {stats.avg_routing_ms:.0f}ms")
        >>> print(f"Slow queries: {stats.slow_routing_count}")
    """
    path = db_path or DEFAULT_DB_PATH
    
    default_stats = OutcomeStatistics(
        total=0,
        unprocessed=0,
        avg_routing_ms=0.0,
        max_routing_ms=0.0,
        slow_routing_count=0,
        error_count=0,
        success_rate=0.0,
    )
    
    if not path.exists():
        return default_stats
    
    try:
        with _get_db(db_path) as conn:
            # Total and unprocessed counts
            total = conn.execute(
                "SELECT COUNT(*) FROM query_outcomes"
            ).fetchone()[0]
            
            unprocessed = conn.execute(
                "SELECT COUNT(*) FROM query_outcomes WHERE processed = 0"
            ).fetchone()[0]
            
            if total == 0:
                return default_stats
            
            # Compute statistics from last hour
            hour_ago = (datetime.utcnow() - timedelta(hours=1)).isoformat()
            
            # AUDIT FIX: Count TRUE errors only, not slow/low_confidence
            # Build a SQL IN clause for true error statuses
            true_error_list = list(TRUE_ERROR_STATUSES)
            true_error_placeholders = ",".join("?" for _ in true_error_list)
            
            # AUDIT FIX: Count non-error statuses (success + slow + low_confidence + partial)
            non_error_list = list(NON_ERROR_STATUSES)
            non_error_placeholders = ",".join("?" for _ in non_error_list)
            
            stats_row = conn.execute(
                f"""
                SELECT 
                    AVG(routing_time_ms),
                    MAX(routing_time_ms),
                    COUNT(CASE WHEN routing_time_ms > ? THEN 1 END),
                    COUNT(CASE WHEN status IN ({true_error_placeholders}) THEN 1 END),
                    COUNT(CASE WHEN status IN ({non_error_placeholders}) THEN 1 END),
                    COUNT(*)
                FROM query_outcomes WHERE timestamp > ?
                """,
                (SLOW_ROUTING_THRESHOLD_MS, *true_error_list, *non_error_list, hour_ago),
            ).fetchone()
            
            # AUDIT FIX: success_count now includes all non-error statuses
            success_count = stats_row[4] or 0
            error_count = stats_row[3] or 0
            total_recent = stats_row[5] or 0
            
            # Success rate = non-errors / total (more accurate than old method)
            success_rate = success_count / total_recent if total_recent > 0 else 0.0
            
            # Compute quality metrics
            quality_row = conn.execute(
                """
                SELECT 
                    COUNT(CASE WHEN answer_quality = 'good' THEN 1 END),
                    COUNT(CASE WHEN answer_quality = 'failed' THEN 1 END),
                    COUNT(CASE WHEN answer_quality IS NOT NULL THEN 1 END)
                FROM query_outcomes WHERE timestamp > ?
                """,
                (hour_ago,),
            ).fetchone()
            
            quality_good_count = quality_row[0] or 0
            quality_failed_count = quality_row[1] or 0
            total_with_quality = quality_row[2] or 0
            
            quality_success_rate = (
                quality_good_count / total_with_quality 
                if total_with_quality > 0 else 0.0
            )
        
        return OutcomeStatistics(
            total=total,
            unprocessed=unprocessed,
            avg_routing_ms=stats_row[0] or 0.0,
            max_routing_ms=stats_row[1] or 0.0,
            slow_routing_count=stats_row[2] or 0,
            error_count=error_count,
            success_rate=success_rate,
            quality_good_count=quality_good_count,
            quality_failed_count=quality_failed_count,
            quality_success_rate=quality_success_rate,
        )
        
    except sqlite3.Error as e:
        logger.warning(f"{LOG_PREFIX} Failed to get statistics: {e}")
        return default_stats


def _deduplicate_outcomes(
    outcomes: List[Dict[str, Any]],
    log_level: str = "info"
) -> List[Dict[str, Any]]:
    """
    Deduplicate outcomes by query_id.
    
    AUDIT FIX: Shared helper function to ensure consistent deduplication
    behavior across all gap detection functions. This prevents the same
    query from being counted multiple times.
    
    Behavior:
    - Outcomes with unique query_id are kept (first occurrence wins)
    - Outcomes with duplicate query_id are filtered out
    - Outcomes with no query_id are included (legacy data compatibility)
    
    Args:
        outcomes: List of outcome dictionaries to deduplicate
        log_level: Logging level for deduplication messages ("info" or "debug")
    
    Returns:
        List of unique outcomes (deduplicated by query_id)
    """
    seen_query_ids: set = set()
    unique_outcomes: List[Dict[str, Any]] = []
    duplicate_count = 0
    
    for o in outcomes:
        query_id = o.get("query_id")
        if query_id and query_id not in seen_query_ids:
            seen_query_ids.add(query_id)
            unique_outcomes.append(o)
        elif query_id:
            duplicate_count += 1
        else:
            # If no query_id, include it (legacy data compatibility)
            # This allows backwards compatibility with old data that may
            # not have query_ids while still deduplicating new data
            unique_outcomes.append(o)
    
    if duplicate_count > 0:
        message = (
            f"{LOG_PREFIX} Deduplicated {duplicate_count} duplicate outcomes "
            f"({len(unique_outcomes)} unique from {len(outcomes)} total)"
        )
        if log_level == "debug":
            logger.debug(message)
        else:
            logger.info(message)
    
    return unique_outcomes


def analyze_outcomes_for_gaps(
    outcomes: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Analyze outcomes and return detected knowledge gaps.
    
    Examines patterns in query outcomes to identify systematic issues
    that should be addressed by the learning system. Uses configurable
    thresholds for gap detection.
    
    AUDIT FIX: Added deduplication by query_id to prevent counting the same
    failure multiple times. This fixes the issue where phantom resolutions
    were occurring because the same query outcome was being processed
    multiple times through different code paths.
    
    Gap Types Detected:
        - slow_routing: Multiple queries taking >10s to route.
            Suggests embedding cache issues or model loading problems.
        - complex_query_handling: High complexity queries averaging >30s.
            Suggests reasoning pipeline optimization needed.
        - high_error_rate: TRUE error rate exceeding 10%.
            Only counts actual errors (timeout, error, cancelled, no_result).
            Does NOT count slow/low_confidence as errors (these are legitimate).
        - routing_variance: High variance in routing times (CV > 1.0).
            Suggests inconsistent cache behavior.
    
    Args:
        outcomes: List of outcome dictionaries from get_recent_outcomes().
            Each dictionary should contain routing_time_ms, total_time_ms,
            complexity, status, error_type, and query_id fields.
    
    Returns:
        List of gap dictionaries with:
            - gap_type: Type of gap detected
            - description: Human-readable description
            - severity: Severity score (0.0 to 1.0)
            - evidence: Supporting query IDs or metrics
            - suggested_action: Recommended remediation
    
    Example:
        >>> outcomes = get_recent_outcomes(minutes=60)
        >>> gaps = analyze_outcomes_for_gaps(outcomes)
        >>> for gap in gaps:
        ...     print(f"{gap['gap_type']}: {gap['description']}")
        ...     print(f"  Severity: {gap['severity']:.1%}")
    """
    if not outcomes:
        return []
    
    # AUDIT FIX: Deduplicate outcomes at the top level using shared helper
    # All downstream functions receive already-deduplicated data
    unique_outcomes = _deduplicate_outcomes(outcomes, log_level="info")
    
    gaps: List[Dict[str, Any]] = []
    
    # Gap: Slow routing (>10s)
    gaps.extend(_detect_slow_routing_gaps(unique_outcomes))
    
    # Gap: Complex query handling issues
    gaps.extend(_detect_complex_query_gaps(unique_outcomes))
    
    # Gap: High error rate (TRUE errors only, excludes slow/low_confidence)
    gaps.extend(_detect_error_rate_gaps(unique_outcomes))
    
    # Gap: High routing time variance
    gaps.extend(_detect_variance_gaps(unique_outcomes))
    
    # Gap: Low answer quality (answers marked as "failed" or "partial")
    gaps.extend(_detect_answer_quality_gaps(unique_outcomes))
    
    return gaps


def _detect_slow_routing_gaps(
    outcomes: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Detect slow routing gaps from outcomes.
    
    Args:
        outcomes: List of outcome dictionaries
    
    Returns:
        List of detected slow routing gaps
    """
    slow = [
        o for o in outcomes 
        if (o.get("routing_time_ms") or 0) > SLOW_ROUTING_THRESHOLD_MS
    ]
    
    if len(slow) < MIN_SLOW_QUERIES_FOR_GAP:
        return []
    
    avg_time = sum(o.get("routing_time_ms", 0) for o in slow) / len(slow)
    
    return [
        DetectedGap(
            gap_type=GapType.SLOW_ROUTING.value,
            description=(
                f"{len(slow)} queries with routing > "
                f"{SLOW_ROUTING_THRESHOLD_MS / 1000:.0f}s "
                f"(avg: {avg_time / 1000:.1f}s)"
            ),
            severity=min(1.0, len(slow) / 10),
            evidence=[o.get("query_id", "unknown") for o in slow[:5]],
            suggested_action="Implement or verify embedding cache",
        ).to_dict()
    ]


def _detect_complex_query_gaps(
    outcomes: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Detect complex query handling gaps from outcomes.
    
    Args:
        outcomes: List of outcome dictionaries
    
    Returns:
        List of detected complex query gaps
    """
    complex_q = [
        o for o in outcomes 
        if (o.get("complexity") or 0) > COMPLEX_QUERY_THRESHOLD
    ]
    
    if not complex_q:
        return []
    
    avg_time = sum(o.get("total_time_ms", 0) for o in complex_q) / len(complex_q)
    
    if avg_time <= COMPLEX_QUERY_TIME_THRESHOLD_MS:
        return []
    
    return [
        DetectedGap(
            gap_type=GapType.COMPLEX_QUERY_HANDLING.value,
            description=f"Complex queries averaging {avg_time / 1000:.1f}s",
            severity=min(1.0, avg_time / 60000),
            evidence=[o.get("query_id", "unknown") for o in complex_q[:5]],
            suggested_action="Optimize reasoning pipeline",
        ).to_dict()
    ]


def _detect_error_rate_gaps(
    outcomes: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Detect high error rate gaps from outcomes.
    
    AUDIT FIX: This function was counting ALL non-success outcomes as errors,
    including legitimate statuses like "slow" and "low_confidence". This caused
    successful complex reasoning (which may be slow or have lower confidence)
    to be misclassified as failures.
    
    The fix distinguishes between:
    - TRUE errors: "error", "timeout", "cancelled", "no_result", "execution_failed"
    - NON-errors: "success", "slow", "low_confidence", "partial"
    
    CRITICAL: Different reasoning paradigms (causal, symbolic, probabilistic, 
    analogical, multimodal) are COMPLEMENTARY and produce different outputs BY DESIGN.
    Low consensus between paradigms is NOT an error.
    
    Note: Deduplication is handled by the caller (analyze_outcomes_for_gaps),
    so this function assumes outcomes are already deduplicated.
    
    Args:
        outcomes: List of outcome dictionaries (assumed to be pre-deduplicated)
    
    Returns:
        List of detected error rate gaps
    """
    if not outcomes:
        return []
    
    # AUDIT FIX: Only count TRUE errors, not slow/low_confidence outcomes
    # "slow" and "low_confidence" are legitimate outcomes, not errors
    # Note: Deduplication is handled by the caller (analyze_outcomes_for_gaps)
    true_errors = [
        o for o in outcomes 
        if o.get("status") in TRUE_ERROR_STATUSES
    ]
    
    error_rate = len(true_errors) / len(outcomes) if outcomes else 0.0
    
    # Log classification breakdown for debugging
    status_counts: Dict[str, int] = {}
    for o in outcomes:
        status = o.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    
    logger.debug(
        f"{LOG_PREFIX} Outcome classification: total={len(outcomes)}, "
        f"true_errors={len(true_errors)}, error_rate={error_rate:.1%}, "
        f"breakdown={status_counts}"
    )
    
    if error_rate <= HIGH_ERROR_RATE_THRESHOLD:
        return []
    
    # Group by error type for evidence
    error_types: Dict[str, int] = {}
    for e in true_errors:
        error_type = e.get("error_type", "unknown")
        error_types[error_type] = error_types.get(error_type, 0) + 1
    
    return [
        DetectedGap(
            gap_type=GapType.HIGH_ERROR_RATE.value,
            description=f"{error_rate * 100:.1f}% true error rate ({len(true_errors)}/{len(outcomes)} queries)",
            severity=min(1.0, error_rate * 5),
            evidence=list(error_types.keys())[:5],
            suggested_action="Investigate error patterns (excludes slow/low_confidence outcomes)",
        ).to_dict()
    ]


def _detect_variance_gaps(
    outcomes: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Detect high routing time variance gaps from outcomes.
    
    Args:
        outcomes: List of outcome dictionaries
    
    Returns:
        List of detected variance gaps
    """
    if len(outcomes) < MIN_SAMPLES_FOR_VARIANCE:
        return []
    
    routing_times = [
        o.get("routing_time_ms", 0)
        for o in outcomes
        if o.get("routing_time_ms") is not None
    ]
    
    if not routing_times:
        return []
    
    try:
        mean_time = stats_module.mean(routing_times)
        std_time = stats_module.stdev(routing_times)
        cv = std_time / mean_time if mean_time > 0 else 0
        
        # Only report if CV > threshold and mean is significant
        if cv <= HIGH_VARIANCE_CV_THRESHOLD or mean_time <= 1000:
            return []
        
        return [
            DetectedGap(
                gap_type=GapType.ROUTING_VARIANCE.value,
                description=(
                    f"High routing time variance "
                    f"(CV={cv:.2f}, mean={mean_time:.0f}ms)"
                ),
                severity=min(1.0, cv / 2),
                evidence=[
                    f"mean={mean_time:.0f}ms",
                    f"std={std_time:.0f}ms",
                ],
                suggested_action="Investigate cache effectiveness",
            ).to_dict()
        ]
        
    except stats_module.StatisticsError:
        return []  # Not enough data


def _detect_answer_quality_gaps(
    outcomes: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Detect gaps based on answer quality metrics.
    
    This function addresses the critical issue where the curiosity engine
    was blind to answer quality - it measured completion (did the query return
    something) but not quality (was the answer actually useful).
    
    Quality-based gaps are detected when:
    - Many answers are marked as "failed" (contain failure patterns)
    - Many answers are marked as "partial" (incomplete/low confidence)
    - The quality success rate (good answers / total) is below threshold
    
    Args:
        outcomes: List of outcome dictionaries with optional answer_quality field
    
    Returns:
        List of detected answer quality gaps
    """
    # Count outcomes by quality
    quality_counts: Dict[str, int] = {
        "good": 0,
        "partial": 0,
        "failed": 0,
        "unknown": 0,
    }
    
    total_with_quality = 0
    failed_query_ids: List[str] = []
    
    for o in outcomes:
        quality = o.get("answer_quality")
        if quality in quality_counts:
            quality_counts[quality] += 1
            total_with_quality += 1
            
            if quality == "failed":
                query_id = o.get("query_id", "unknown")
                if len(failed_query_ids) < 10:
                    failed_query_ids.append(query_id)
        elif quality is None:
            quality_counts["unknown"] += 1
    
    # If no quality data available, can't detect quality gaps
    if total_with_quality == 0:
        return []
    
    # Calculate quality success rate
    good_count = quality_counts["good"]
    failed_count = quality_counts["failed"]
    partial_count = quality_counts["partial"]
    
    quality_success_rate = good_count / total_with_quality if total_with_quality > 0 else 0.0
    quality_failure_rate = failed_count / total_with_quality if total_with_quality > 0 else 0.0
    
    gaps: List[Dict[str, Any]] = []
    
    # Log quality metrics for visibility
    logger.info(
        f"{LOG_PREFIX} Quality metrics: {good_count}/{total_with_quality} good answers "
        f"({quality_success_rate:.1%}), failed={failed_count}, partial={partial_count}"
    )
    
    # Gap: Low answer quality rate (uses module constant LOW_QUALITY_SUCCESS_THRESHOLD)
    if total_with_quality >= 5 and quality_success_rate < LOW_QUALITY_SUCCESS_THRESHOLD:
        gaps.append(
            DetectedGap(
                gap_type="low_answer_quality",
                description=(
                    f"Low answer quality: only {quality_success_rate:.1%} "
                    f"({good_count}/{total_with_quality}) of answers are good quality"
                ),
                severity=min(1.0, (1.0 - quality_success_rate) * 1.5),
                evidence=failed_query_ids[:5],
                suggested_action="Investigate failure patterns in responses (engine rejections, raw data dumps, misclassifications)",
            ).to_dict()
        )
    
    # Gap: High failure rate in answers (uses module constant HIGH_QUALITY_FAILURE_THRESHOLD)
    if total_with_quality >= 5 and quality_failure_rate > HIGH_QUALITY_FAILURE_THRESHOLD:
        gaps.append(
            DetectedGap(
                gap_type="answer_failure_patterns",
                description=(
                    f"High answer failure rate: {quality_failure_rate:.1%} "
                    f"({failed_count}/{total_with_quality}) answers contain failure patterns"
                ),
                severity=min(1.0, quality_failure_rate * 2),
                evidence=failed_query_ids[:5],
                suggested_action="Fix engine gate checks, output formatting, and query classification",
            ).to_dict()
        )
    
    return gaps


# =============================================================================
# Maintenance Operations
# =============================================================================

def cleanup_old_outcomes(
    days: int = DEFAULT_RETENTION_DAYS,
    db_path: Optional[Path] = None,
) -> int:
    """
    Remove outcomes older than specified days.
    
    Should be called periodically (e.g., daily) to prevent unbounded
    database growth. This is a maintenance operation that doesn't affect
    normal operation.
    
    Args:
        days: Number of days to keep (default: 7).
            Outcomes older than this will be deleted.
        db_path: Optional path to database file.
    
    Returns:
        Number of rows deleted.
    
    Example:
        >>> deleted = cleanup_old_outcomes(days=7)
        >>> print(f"Cleaned up {deleted} old outcomes")
    """
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
    
    try:
        with _get_db(db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM query_outcomes WHERE timestamp < ?",
                (cutoff,),
            )
            deleted = cursor.rowcount
            conn.commit()
        
        if deleted > 0:
            logger.info(f"{LOG_PREFIX} Cleaned up {deleted} old outcomes")
        
        return deleted
        
    except sqlite3.Error as e:
        logger.warning(f"{LOG_PREFIX} Cleanup failed: {e}")
        return 0


def get_database_path() -> Path:
    """
    Get the current database path.
    
    Returns:
        Path to the outcomes database file.
    """
    return DEFAULT_DB_PATH


def reset_database(db_path: Optional[Path] = None) -> bool:
    """
    Reset the database by dropping and recreating tables.
    
    WARNING: This will delete all stored outcomes. Use only for testing
    or when a fresh start is needed.
    
    Args:
        db_path: Optional path to database file.
    
    Returns:
        True if reset succeeded, False otherwise.
    """
    global _db_initialized
    
    path = db_path or DEFAULT_DB_PATH
    
    try:
        with _get_db(db_path) as conn:
            conn.execute("DROP TABLE IF EXISTS query_outcomes")
            conn.commit()
        
        # Reset initialization flag
        with _init_lock:
            _db_initialized = False
        
        # Reinitialize
        return _init_db(db_path)
        
    except sqlite3.Error as e:
        logger.error(f"{LOG_PREFIX} Database reset failed: {e}")
        return False


# =============================================================================
# Module Initialization
# =============================================================================

# Initialize on import (non-blocking, logs debug if deferred)
try:
    _init_db()
except Exception as e:
    logger.debug(f"{LOG_PREFIX} Deferred initialization: {e}")
