"""
query_outcome.py - Query outcome data structure for Curiosity Engine learning
Part of the VULCAN-AGI system

This module defines the QueryOutcome dataclass that represents the outcome
of a processed query. It enables data flow from the main query processing
pipeline to the curiosity engine for learning and gap analysis.

BUG #3 FIX: This solves the problem of the curiosity engine finding 0 knowledge
gaps because it had no data. QueryOutcome captures all relevant metrics from
query processing that the curiosity engine needs for pattern recognition.

Architecture:
    Main Process:
        Query Router → Agent Pool → LLM Response
              ↓              ↓           ↓
        QueryOutcome (captures metrics from all stages)
              ↓
        OutcomeBuffer (thread-safe queue)
              ↓
    Curiosity Engine Subprocess:
        GapAnalyzer → DependencyGraph → ExperimentGenerator

Thread Safety:
    QueryOutcome is immutable after creation. The OutcomeBuffer handles
    thread-safe access when passing outcomes between processes.
"""

import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# OUTCOME STATUS ENUM
# =============================================================================


class OutcomeStatus(Enum):
    """
    Status of query outcome for learning classification.
    
    Used by GapAnalyzer to categorize outcomes and identify patterns
    in successful vs failed queries.
    
    Values:
        SUCCESS: Query completed successfully with valid response
        PARTIAL: Query partially completed (some systems failed)
        FAILURE: Query failed to produce a response
        TIMEOUT: Query timed out before completion
        REJECTED: Query rejected by safety validation
    """
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    REJECTED = "rejected"


# =============================================================================
# QUERY OUTCOME DATACLASS
# =============================================================================


@dataclass
class QueryOutcome:
    """
    Represents the outcome of a processed query for curiosity-driven learning.
    
    This structure captures all relevant information about a query's processing
    that the curiosity engine needs to identify patterns, gaps, and improvement
    opportunities. It follows the EXAMINE → SELECT → APPLY → REMEMBER pattern
    by providing data for each stage.
    
    EXAMINE: query_text, query_type, complexity, uncertainty
    SELECT: capabilities_used, agents_used, tasks_generated
    APPLY: execution_time_ms, status, error_type
    REMEMBER: confidence, user_feedback, features (for ML)
    
    Thread Safety:
        This dataclass is effectively immutable after creation.
        The compute_features() method modifies self.features but this
        is expected to be called once immediately after creation.
    
    Attributes:
        query_id: Unique identifier for the query (e.g., q_2c4878b85ba4)
        timestamp: Unix timestamp when the outcome was recorded
        query_text: The original query text (should be truncated for storage)
        query_type: Classification of query type (general, perception, reasoning, etc.)
        complexity: Complexity score from routing (0.0-1.0)
        uncertainty: Uncertainty score from routing (0.0-1.0)
        routing_time_ms: Time spent in query routing
        tasks_generated: Number of agent tasks created
        was_creative: Whether this was identified as a creative task
        agents_used: List of agent IDs that processed the query
        capabilities_used: List of capabilities engaged (perception, reasoning, etc.)
        execution_time_ms: Total execution time in milliseconds
        status: Final outcome status (success, failure, timeout, etc.)
        confidence: Confidence score of the response (0.0-1.0)
        llm_tokens_used: Number of tokens used for LLM generation
        user_feedback: Optional user feedback (thumbs_up, thumbs_down, none)
        error_type: Type of error if failed (exception class name)
        features: Computed feature vector for ML algorithms
        
    Example:
        >>> outcome = QueryOutcome(
        ...     query_id="q_abc123",
        ...     query_text="Explain quantum entanglement",
        ...     query_type="reasoning",
        ...     complexity=0.7,
        ...     status=OutcomeStatus.SUCCESS,
        ...     execution_time_ms=4500.0,
        ... )
        >>> outcome.compute_features()
        >>> record_outcome(outcome)
    """
    
    # Identity - Required fields
    query_id: str
    timestamp: float = field(default_factory=time.time)
    
    # Query characteristics - From routing layer
    query_text: str = ""
    query_type: str = "general"  # general, perception, reasoning, planning, execution, learning
    complexity: float = 0.0
    uncertainty: float = 0.0
    
    # Routing info - From query router
    routing_time_ms: float = 0.0
    tasks_generated: int = 0
    was_creative: bool = False
    
    # Execution info - From agent pool
    agents_used: List[str] = field(default_factory=list)
    capabilities_used: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    
    # Outcome - Final results
    status: OutcomeStatus = OutcomeStatus.SUCCESS
    confidence: float = 0.0
    llm_tokens_used: int = 0
    
    # Learning feedback
    user_feedback: Optional[str] = None  # thumbs_up, thumbs_down, none
    error_type: Optional[str] = None
    
    # Feature vector for ML (computed lazily)
    features: Optional[List[float]] = None
    
    # =============================================================================
    # SERIALIZATION METHODS
    # =============================================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization and logging.
        
        Truncates query_text to 200 characters to prevent log bloat
        while preserving enough context for analysis.
        
        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return {
            "query_id": self.query_id,
            "timestamp": self.timestamp,
            "query_text": self._safe_truncate(self.query_text, 200),
            "query_type": self.query_type,
            "complexity": self.complexity,
            "uncertainty": self.uncertainty,
            "routing_time_ms": self.routing_time_ms,
            "tasks_generated": self.tasks_generated,
            "was_creative": self.was_creative,
            "agents_used": self.agents_used.copy() if self.agents_used else [],
            "capabilities_used": self.capabilities_used.copy() if self.capabilities_used else [],
            "execution_time_ms": self.execution_time_ms,
            "status": self.status.value,
            "confidence": self.confidence,
            "llm_tokens_used": self.llm_tokens_used,
            "user_feedback": self.user_feedback,
            "error_type": self.error_type,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueryOutcome":
        """
        Create QueryOutcome from dictionary.
        
        Handles status enum conversion and provides defaults for missing fields.
        
        Args:
            data: Dictionary with outcome data
            
        Returns:
            New QueryOutcome instance
        """
        # Handle status enum conversion
        status_val = data.get("status", "success")
        if isinstance(status_val, str):
            try:
                status = OutcomeStatus(status_val)
            except ValueError:
                logger.warning(f"Unknown status value: {status_val}, defaulting to SUCCESS")
                status = OutcomeStatus.SUCCESS
        elif isinstance(status_val, OutcomeStatus):
            status = status_val
        else:
            status = OutcomeStatus.SUCCESS
        
        return cls(
            query_id=data.get("query_id", ""),
            timestamp=data.get("timestamp", time.time()),
            query_text=data.get("query_text", ""),
            query_type=data.get("query_type", "general"),
            complexity=float(data.get("complexity", 0.0)),
            uncertainty=float(data.get("uncertainty", 0.0)),
            routing_time_ms=float(data.get("routing_time_ms", 0.0)),
            tasks_generated=int(data.get("tasks_generated", 0)),
            was_creative=bool(data.get("was_creative", False)),
            agents_used=list(data.get("agents_used", [])),
            capabilities_used=list(data.get("capabilities_used", [])),
            execution_time_ms=float(data.get("execution_time_ms", 0.0)),
            status=status,
            confidence=float(data.get("confidence", 0.0)),
            llm_tokens_used=int(data.get("llm_tokens_used", 0)),
            user_feedback=data.get("user_feedback"),
            error_type=data.get("error_type"),
        )
    
    # =============================================================================
    # FEATURE EXTRACTION FOR ML
    # =============================================================================
    
    def compute_features(self) -> List[float]:
        """
        Compute feature vector for learning algorithms.
        
        Creates a normalized feature vector that can be used by the curiosity
        engine's ML components for pattern recognition and gap analysis.
        
        EXAMINE: Extracts key metrics from the outcome
        SELECT: Chooses features most relevant for learning
        APPLY: Normalizes values to [0, 1] range where possible
        REMEMBER: Stores in self.features for later use
        
        Returns:
            List of 9 float features representing the query outcome:
            - complexity (0-1)
            - uncertainty (0-1)
            - routing_time_seconds (normalized)
            - execution_time_seconds (normalized)
            - tasks_generated (count)
            - capabilities_count (count)
            - was_creative (0 or 1)
            - confidence (0-1)
            - success (0 or 1)
        """
        try:
            self.features = [
                float(self.complexity),
                float(self.uncertainty),
                self.routing_time_ms / 1000.0,  # Normalize to seconds
                self.execution_time_ms / 1000.0,  # Normalize to seconds
                float(self.tasks_generated),
                float(len(self.capabilities_used)),
                1.0 if self.was_creative else 0.0,
                float(self.confidence),
                1.0 if self.status == OutcomeStatus.SUCCESS else 0.0,
            ]
        except Exception as e:
            logger.error(f"Error computing features: {e}")
            # Return zero vector on error
            self.features = [0.0] * 9
        
        return self.features
    
    def get_feature_names(self) -> List[str]:
        """
        Get names of features for interpretability.
        
        Returns:
            List of feature names corresponding to compute_features() output
        """
        return [
            "complexity",
            "uncertainty",
            "routing_time_s",
            "execution_time_s",
            "tasks_generated",
            "capabilities_count",
            "was_creative",
            "confidence",
            "success",
        ]
    
    # =============================================================================
    # UTILITY METHODS
    # =============================================================================
    
    @staticmethod
    def _safe_truncate(text: str, max_length: int) -> str:
        """
        Safely truncate text to max_length, respecting unicode boundaries.
        
        Uses encode/decode to ensure we don't cut in the middle of
        multi-byte characters or surrogate pairs.
        
        Args:
            text: The text to truncate
            max_length: Maximum length in characters
            
        Returns:
            Truncated text that doesn't cut unicode characters
        """
        if not text or len(text) <= max_length:
            return text
        
        truncated = text[:max_length]
        
        # Validate UTF-8 encoding to catch broken unicode
        try:
            truncated.encode('utf-8')
            return truncated
        except UnicodeEncodeError:
            # Back up until we get valid UTF-8
            while truncated:
                truncated = truncated[:-1]
                try:
                    truncated.encode('utf-8')
                    return truncated
                except UnicodeEncodeError:
                    continue
            return ""
    
    def is_success(self) -> bool:
        """Check if this outcome represents a successful query."""
        return self.status == OutcomeStatus.SUCCESS
    
    def is_failure(self) -> bool:
        """Check if this outcome represents a failed query."""
        return self.status in (OutcomeStatus.FAILURE, OutcomeStatus.TIMEOUT, OutcomeStatus.REJECTED)
    
    def get_domain(self) -> str:
        """
        Get domain for gap analysis categorization.
        
        Returns query_type as the primary domain, or "unknown" if not set.
        """
        return self.query_type if self.query_type else "unknown"
    
    def __hash__(self) -> int:
        """Make outcome hashable by query_id for set operations."""
        return hash(self.query_id)
    
    def __eq__(self, other: object) -> bool:
        """Equality comparison based on query_id."""
        if not isinstance(other, QueryOutcome):
            return False
        return self.query_id == other.query_id

