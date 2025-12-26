# ============================================================
# VULCAN-AGI Student Router Module
# Knowledge Distillation-based Hybrid Routing for CPU Optimization
# ============================================================
#
# ARCHITECTURAL NOTE:
# This module implements the "Master and Apprentice" pattern for CPU optimization.
# The Student (lightweight classifier) handles easy queries with low latency,
# while the Teacher (full Vulcan pipeline) handles complex queries requiring
# deep reasoning. This prevents using a "Supercomputer to answer 'What time is it?'"
#
# ROUTING FLOW:
#
#     Query Received
#         │
#         ▼
#     StudentRouter.route()
#         │
#         ├─ Check Cache ─────────────────────────► Cache Hit → FAST_PATH
#         │
#         ▼
#     QueryTypeClassifier.classify()  (<1ms, CPU L2/L3 cache)
#         │
#         ├─ confidence >= threshold ────────────► FAST_PATH  (50ms target)
#         │
#         ├─ confidence <= escalation_threshold ─► ESCALATE   (60s Teacher)
#         │
#         └─ moderate confidence ────────────────► HYBRID     (try fast first)
#
# PERFORMANCE TARGETS:
#     - Student inference: <50ms (fits in CPU L2/L3 cache)
#     - Teacher inference: ~60s (full reasoning pipeline)
#     - Expected CPU reduction: 80% for typical workloads (80% easy queries)
#
# USAGE:
#     from strategies.student_router import StudentRouter, RoutingDecision
#
#     router = StudentRouter(config={
#         "distillation_config": {
#             "enabled": True,
#             "mode": "active",
#             "confidence_threshold": 0.85,
#         }
#     })
#
#     result = router.route("Hello, how are you?")
#     if result.decision == RoutingDecision.FAST_PATH:
#         # Handle with lightweight response
#     else:
#         # Escalate to full Vulcan pipeline
#
# VERSION HISTORY:
#     1.0.0 - Initial implementation with pattern-based classification
#     1.0.1 - Added SHA-256 hashing, configurable patterns
#     1.1.0 - Added persistence, thread safety, comprehensive statistics
# ============================================================

import hashlib
import json
import logging
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Module metadata
__version__ = "1.1.0"
__author__ = "VULCAN-AGI Team"

logger = logging.getLogger(__name__)


class QueryComplexity(Enum):
    """Query complexity classification."""
    TRIVIAL = "trivial"       # Greetings, simple lookups
    SIMPLE = "simple"         # Factual questions
    MODERATE = "moderate"     # Multi-step reasoning
    COMPLEX = "complex"       # Deep analysis, code generation
    EXPERT = "expert"         # Novel problems, creative synthesis


class RoutingDecision(Enum):
    """Routing decision for query handling."""
    FAST_PATH = "fast_path"           # Use Student/cache
    ESCALATE = "escalate"             # Send to Teacher
    HYBRID = "hybrid"                 # Try Student first, escalate if needed


@dataclass
class RoutingResult:
    """Result of routing decision with full serialization support."""
    decision: RoutingDecision
    confidence: float
    complexity: QueryComplexity
    estimated_latency_ms: float
    reasoning: str
    query_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "decision": self.decision.value,
            "confidence": self.confidence,
            "complexity": self.complexity.value,
            "estimated_latency_ms": self.estimated_latency_ms,
            "reasoning": self.reasoning,
            "query_type": self.query_type,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


@dataclass
class StudentPrediction:
    """Prediction from the Student model with inference metrics."""
    label: str
    confidence: float
    soft_labels: Dict[str, float]
    inference_time_ms: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "label": self.label,
            "confidence": self.confidence,
            "soft_labels": self.soft_labels,
            "inference_time_ms": self.inference_time_ms,
            "timestamp": self.timestamp,
        }


@dataclass
class RouterStatistics:
    """Comprehensive routing statistics with serialization support."""
    total_queries: int = 0
    fast_path_count: int = 0
    escalation_count: int = 0
    hybrid_count: int = 0
    cache_hits: int = 0
    avg_student_latency_ms: float = 0.0
    avg_teacher_latency_ms: float = 0.0
    cpu_savings_percent: float = 0.0
    min_routing_time_ms: float = float("inf")
    max_routing_time_ms: float = 0.0
    p50_routing_time_ms: float = 0.0
    p95_routing_time_ms: float = 0.0
    p99_routing_time_ms: float = 0.0
    last_query_timestamp: Optional[float] = None
    start_time: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        total = max(1, self.total_queries)
        return {
            "total_queries": self.total_queries,
            "fast_path_count": self.fast_path_count,
            "escalation_count": self.escalation_count,
            "hybrid_count": self.hybrid_count,
            "cache_hits": self.cache_hits,
            "fast_path_rate": self.fast_path_count / total,
            "escalation_rate": self.escalation_count / total,
            "hybrid_rate": self.hybrid_count / total,
            "cache_hit_rate": self.cache_hits / total,
            "avg_student_latency_ms": self.avg_student_latency_ms,
            "avg_teacher_latency_ms": self.avg_teacher_latency_ms,
            "cpu_savings_percent": self.cpu_savings_percent,
            "min_routing_time_ms": self.min_routing_time_ms if self.min_routing_time_ms != float("inf") else 0.0,
            "max_routing_time_ms": self.max_routing_time_ms,
            "p50_routing_time_ms": self.p50_routing_time_ms,
            "p95_routing_time_ms": self.p95_routing_time_ms,
            "p99_routing_time_ms": self.p99_routing_time_ms,
            "last_query_timestamp": self.last_query_timestamp,
            "uptime_seconds": time.time() - self.start_time,
        }

    def update_latency_percentiles(self, latencies: List[float]) -> None:
        """Update latency percentiles from list of observations."""
        if latencies:
            import numpy as np
            self.p50_routing_time_ms = float(np.percentile(latencies, 50))
            self.p95_routing_time_ms = float(np.percentile(latencies, 95))
            self.p99_routing_time_ms = float(np.percentile(latencies, 99))


class QueryTypeClassifier:
    """
    Lightweight query classifier that runs entirely in CPU cache.
    
    This is the "Student" in the Master-Apprentice pattern.
    It doesn't understand the "why", but memorizes the "what" - 
    quickly guessing query types without heavy computation.
    
    Default patterns can be extended or overridden by passing custom patterns
    in the config dictionary with keys: greeting_patterns, simple_factual_patterns,
    complex_reasoning_patterns, code_patterns, creative_patterns.
    """
    
    # Default pattern-based classification (ultra-fast, regex-based)
    # Can be overridden via config
    DEFAULT_GREETING_PATTERNS = [
        r"^(hi|hello|hey|good morning|good afternoon|good evening|howdy|greetings)\b",
        r"^(how are you|how's it going|what's up)\??$",
        r"^(thanks|thank you|cheers|bye|goodbye|see you)\b",
    ]
    
    DEFAULT_SIMPLE_FACTUAL_PATTERNS = [
        r"^(what is|what's|who is|who's|when was|where is|where's)\s",
        r"^(define|meaning of|definition of)\s",
        r"^(how many|how much|how old|how tall|how far)\s",
    ]
    
    DEFAULT_COMPLEX_REASONING_PATTERNS = [
        r"\b(explain|analyze|compare|contrast|evaluate|discuss)\b",
        r"\b(why does|how does|what causes|what happens if)\b",
        r"\b(pros and cons|advantages|disadvantages)\b",
        r"\b(relationship between|difference between)\b",
    ]
    
    DEFAULT_CODE_PATTERNS = [
        r"\b(write|create|implement|code|function|class|script|program)\b.*\b(python|javascript|java|c\+\+|rust|go|sql)\b",
        r"\b(debug|fix|refactor|optimize)\b.*\b(code|function|method|class)\b",
        r"```",  # Code blocks
        r"\b(api|endpoint|database|query|algorithm)\b",
    ]
    
    DEFAULT_CREATIVE_PATTERNS = [
        r"\b(write|create|compose|draft)\b.*\b(story|poem|essay|article|blog)\b",
        r"\b(creative|imaginative|fictional|narrative)\b",
        r"\b(brainstorm|ideas for|suggestions for)\b",
    ]
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the classifier with optional config."""
        config = config or {}
        
        # Allow custom patterns via config, fallback to defaults
        greeting_patterns = config.get("greeting_patterns", self.DEFAULT_GREETING_PATTERNS)
        simple_patterns = config.get("simple_factual_patterns", self.DEFAULT_SIMPLE_FACTUAL_PATTERNS)
        complex_patterns = config.get("complex_reasoning_patterns", self.DEFAULT_COMPLEX_REASONING_PATTERNS)
        code_patterns = config.get("code_patterns", self.DEFAULT_CODE_PATTERNS)
        creative_patterns = config.get("creative_patterns", self.DEFAULT_CREATIVE_PATTERNS)
        
        # Compile patterns for speed
        self._greeting_re = [re.compile(p, re.IGNORECASE) for p in greeting_patterns]
        self._simple_re = [re.compile(p, re.IGNORECASE) for p in simple_patterns]
        self._complex_re = [re.compile(p, re.IGNORECASE) for p in complex_patterns]
        self._code_re = [re.compile(p, re.IGNORECASE) for p in code_patterns]
        self._creative_re = [re.compile(p, re.IGNORECASE) for p in creative_patterns]
        
        # Configurable confidence thresholds
        self.confidence_thresholds = config.get("query_type_classifier", {
            "simple_greetings": 0.99,
            "factual_lookup": 0.90,
            "complex_reasoning": 0.30,
            "code_generation": 0.40,
            "creative_writing": 0.50,
        })
        
        # Statistics
        self.stats = {
            "classifications": 0,
            "by_type": {},
            "avg_inference_time_ms": 0.0,
        }
    
    def classify(self, query: str) -> StudentPrediction:
        """
        Classify a query into a type with confidence scores.
        
        This is the fast "Student" inference - pattern matching that
        runs entirely in CPU cache without touching slow RAM.
        
        Args:
            query: The input query string
            
        Returns:
            StudentPrediction with classification results
        """
        start_time = time.perf_counter()
        
        query = query.strip()
        query_lower = query.lower()
        query_len = len(query)
        
        # Initialize soft labels (probability distribution)
        soft_labels = {
            "greeting": 0.0,
            "simple_factual": 0.0,
            "complex_reasoning": 0.0,
            "code": 0.0,
            "creative": 0.0,
            "general": 0.2,  # Base probability for unclassified
        }
        
        # Pattern matching (ultra-fast)
        for pattern in self._greeting_re:
            if pattern.search(query_lower):
                soft_labels["greeting"] = 0.95
                break
        
        for pattern in self._simple_re:
            if pattern.search(query_lower):
                soft_labels["simple_factual"] = max(soft_labels["simple_factual"], 0.85)
        
        for pattern in self._complex_re:
            if pattern.search(query_lower):
                soft_labels["complex_reasoning"] = max(soft_labels["complex_reasoning"], 0.75)
        
        for pattern in self._code_re:
            if pattern.search(query_lower):
                soft_labels["code"] = max(soft_labels["code"], 0.80)
        
        for pattern in self._creative_re:
            if pattern.search(query_lower):
                soft_labels["creative"] = max(soft_labels["creative"], 0.70)
        
        # Length-based adjustments
        if query_len < 20:
            soft_labels["greeting"] = min(1.0, soft_labels["greeting"] * 1.2)
            soft_labels["simple_factual"] = min(1.0, soft_labels["simple_factual"] * 1.1)
        elif query_len > 200:
            soft_labels["complex_reasoning"] = min(1.0, soft_labels["complex_reasoning"] * 1.3)
            soft_labels["code"] = min(1.0, soft_labels["code"] * 1.2)
        
        # Normalize to probabilities
        total = sum(soft_labels.values())
        if total > 0:
            soft_labels = {k: v / total for k, v in soft_labels.items()}
        
        # Select top prediction
        top_label = max(soft_labels, key=soft_labels.get)
        top_confidence = soft_labels[top_label]
        
        inference_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Update stats
        self.stats["classifications"] += 1
        self.stats["by_type"][top_label] = self.stats["by_type"].get(top_label, 0) + 1
        n = self.stats["classifications"]
        self.stats["avg_inference_time_ms"] = (
            (self.stats["avg_inference_time_ms"] * (n - 1) + inference_time_ms) / n
        )
        
        return StudentPrediction(
            label=top_label,
            confidence=top_confidence,
            soft_labels=soft_labels,
            inference_time_ms=inference_time_ms,
        )


class StudentRouter:
    """
    Hybrid routing system using Knowledge Distillation pattern.
    
    Routes queries between fast Student (lightweight classifier) and
    slow Teacher (full Vulcan pipeline) based on confidence thresholds.
    
    This implements the "triage" pattern:
    1. Query comes in
    2. Student makes quick prediction with confidence
    3. If confident (>threshold): Fast path
    4. If uncertain (<threshold): Escalate to Teacher
    
    Expected results:
    - 80% of queries use fast path (simple greetings, factual lookups)
    - Average latency drops from 60s to <5s
    - CPU usage drops by ~80%
    
    Features:
    - Thread-safe operations with RLock
    - Persistent state save/load for model continuity
    - Comprehensive statistics with percentile tracking
    - LRU-style response caching with TTL
    - Full audit trail via routing history
    
    Attributes:
        enabled: Whether routing is enabled
        mode: Operating mode ("active" or "passive")
        confidence_threshold: Threshold for fast path routing
        escalation_threshold: Threshold for Teacher escalation
        classifier: The QueryTypeClassifier instance
        stats: RouterStatistics instance tracking all metrics
    """
    
    # Configuration defaults
    DEFAULT_CACHE_SIZE = 1000
    DEFAULT_CACHE_TTL_SECONDS = 300
    DEFAULT_HISTORY_SIZE = 10000
    DEFAULT_CONFIDENCE_THRESHOLD = 0.85
    DEFAULT_ESCALATION_THRESHOLD = 0.40
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Student Router with configuration.
        
        Args:
            config: Configuration dictionary with optional keys:
                - distillation_config: Dict with routing parameters
                - cache_max_size: Maximum cache entries (default: 1000)
                - cache_ttl_seconds: Cache entry TTL (default: 300)
                - history_size: Max routing history entries (default: 10000)
        """
        config = config or {}
        
        # Get distillation config
        distillation_config = config.get("distillation_config", {})
        
        # Core settings
        self.enabled = distillation_config.get("enabled", True)
        self.mode = distillation_config.get("mode", "active")  # "active" or "passive"
        self.confidence_threshold = distillation_config.get(
            "confidence_threshold", self.DEFAULT_CONFIDENCE_THRESHOLD
        )
        self.escalation_threshold = distillation_config.get(
            "escalation_threshold", self.DEFAULT_ESCALATION_THRESHOLD
        )
        self.fast_path_latency_target_ms = distillation_config.get("fast_path_latency_target_ms", 50)
        self.teacher_timeout_seconds = distillation_config.get("teacher_timeout_seconds", 60)
        self.student_timeout_seconds = distillation_config.get("student_timeout_seconds", 5)
        self.log_routing_decisions = distillation_config.get("log_routing_decisions", True)
        
        # Initialize classifier
        self.classifier = QueryTypeClassifier(distillation_config)
        
        # Response cache for instant answers (thread-safe)
        self._response_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_max_size = config.get("cache_max_size", self.DEFAULT_CACHE_SIZE)
        self._cache_ttl_seconds = config.get("cache_ttl_seconds", self.DEFAULT_CACHE_TTL_SECONDS)
        
        # Routing history for learning and audit
        history_size = config.get("history_size", self.DEFAULT_HISTORY_SIZE)
        self._routing_history: deque = deque(maxlen=history_size)
        
        # Routing time observations for percentile calculations
        self._routing_times: deque = deque(maxlen=1000)
        
        # Statistics - using dataclass for comprehensive tracking
        self.stats = RouterStatistics()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Start time for uptime tracking
        self._start_time = time.time()
        
        logger.info(
            f"StudentRouter initialized: mode={self.mode}, "
            f"confidence_threshold={self.confidence_threshold}, "
            f"escalation_threshold={self.escalation_threshold}, "
            f"cache_size={self._cache_max_size}"
        )
    
    def _get_cache_key(self, query: str) -> str:
        """
        Generate cache key for query using SHA-256.
        
        Uses cryptographically secure hashing to prevent collision attacks
        and ensure unique cache entries.
        
        Args:
            query: The input query string
            
        Returns:
            SHA-256 hex digest of normalized query
        """
        return hashlib.sha256(query.strip().lower().encode()).hexdigest()
    
    def _check_cache(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Check if query has a cached response (thread-safe).
        
        Args:
            query: The input query string
            
        Returns:
            Cached response if found and not expired, None otherwise
        """
        with self._lock:
            key = self._get_cache_key(query)
            if key in self._response_cache:
                entry = self._response_cache[key]
                if time.time() - entry["timestamp"] < self._cache_ttl_seconds:
                    self.stats.cache_hits += 1
                    return entry["response"]
                else:
                    # Expired entry - remove it
                    del self._response_cache[key]
            return None
    
    def _cache_response(self, query: str, response: Dict[str, Any]) -> None:
        """
        Cache a response for future use (thread-safe, LRU eviction).
        
        Args:
            query: The original query string
            response: The response to cache
        """
        with self._lock:
            # LRU eviction when cache is full
            if len(self._response_cache) >= self._cache_max_size:
                # Remove oldest 10% of entries
                eviction_count = max(1, self._cache_max_size // 10)
                oldest_keys = sorted(
                    self._response_cache.keys(),
                    key=lambda k: self._response_cache[k]["timestamp"]
                )[:eviction_count]
                for k in oldest_keys:
                    del self._response_cache[k]
            
            key = self._get_cache_key(query)
            self._response_cache[key] = {
                "response": response,
                "timestamp": time.time(),
            }
    
    def route(self, query: str) -> RoutingResult:
        """
        Route a query to the appropriate handler (thread-safe).
        
        This is the main entry point for the triage system. It performs:
        1. Cache lookup for instant responses
        2. Student classification (ultra-fast pattern matching)
        3. Confidence-based routing decision
        
        Args:
            query: The input query string
            
        Returns:
            RoutingResult with decision, confidence, and metadata
            
        Example:
            >>> router = StudentRouter()
            >>> result = router.route("Hello!")
            >>> if result.decision == RoutingDecision.FAST_PATH:
            ...     # Handle with lightweight response
            ...     pass
        """
        start_time = time.perf_counter()
        
        with self._lock:
            self.stats.total_queries += 1
            self.stats.last_query_timestamp = time.time()
        
        # Check if routing is disabled
        if not self.enabled or self.mode == "passive":
            return RoutingResult(
                decision=RoutingDecision.ESCALATE,
                confidence=0.0,
                complexity=QueryComplexity.MODERATE,
                estimated_latency_ms=self.teacher_timeout_seconds * 1000,
                reasoning="Routing disabled or in passive mode",
                query_type="unknown",
            )
        
        # Step 1: Check cache for instant response
        cached = self._check_cache(query)
        if cached:
            return RoutingResult(
                decision=RoutingDecision.FAST_PATH,
                confidence=1.0,
                complexity=QueryComplexity.TRIVIAL,
                estimated_latency_ms=1.0,
                reasoning="Cache hit - instant response",
                query_type="cached",
                metadata={"cache_hit": True, "cached_response": cached},
            )
        
        # Step 2: Student classification (ultra-fast)
        prediction = self.classifier.classify(query)
        
        # Step 3: Determine complexity from query type
        complexity_map = {
            "greeting": QueryComplexity.TRIVIAL,
            "simple_factual": QueryComplexity.SIMPLE,
            "general": QueryComplexity.MODERATE,
            "creative": QueryComplexity.MODERATE,
            "complex_reasoning": QueryComplexity.COMPLEX,
            "code": QueryComplexity.COMPLEX,
        }
        complexity = complexity_map.get(prediction.label, QueryComplexity.MODERATE)
        
        # Step 4: Make routing decision based on confidence
        with self._lock:
            if prediction.confidence >= self.confidence_threshold:
                # High confidence - use fast path
                decision = RoutingDecision.FAST_PATH
                self.stats.fast_path_count += 1
                estimated_latency = self.fast_path_latency_target_ms
                reasoning = (
                    f"Student confident ({prediction.confidence:.0%}) that this is a "
                    f"{prediction.label} query - using fast path"
                )
            elif prediction.confidence <= self.escalation_threshold:
                # Low confidence - escalate to Teacher
                decision = RoutingDecision.ESCALATE
                self.stats.escalation_count += 1
                estimated_latency = self.teacher_timeout_seconds * 1000
                reasoning = (
                    f"Student uncertain ({prediction.confidence:.0%}) - "
                    f"escalating to Teacher for full analysis"
                )
            else:
                # Moderate confidence - try hybrid approach
                decision = RoutingDecision.HYBRID
                self.stats.hybrid_count += 1
                estimated_latency = self.student_timeout_seconds * 1000
                reasoning = (
                    f"Student moderately confident ({prediction.confidence:.0%}) - "
                    f"will try fast path first, escalate if needed"
                )
        
        routing_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Update routing time statistics (thread-safe)
        with self._lock:
            self._routing_times.append(routing_time_ms)
            
            # Update min/max
            if routing_time_ms < self.stats.min_routing_time_ms:
                self.stats.min_routing_time_ms = routing_time_ms
            if routing_time_ms > self.stats.max_routing_time_ms:
                self.stats.max_routing_time_ms = routing_time_ms
            
            # Update percentiles periodically (every 100 queries)
            if self.stats.total_queries % 100 == 0 and self._routing_times:
                self.stats.update_latency_percentiles(list(self._routing_times))
            
            # Update CPU savings estimate
            fast_path_rate = self.stats.fast_path_count / max(1, self.stats.total_queries)
            self.stats.cpu_savings_percent = fast_path_rate * 80  # Assume 80% CPU saved per fast path
        
        result = RoutingResult(
            decision=decision,
            confidence=prediction.confidence,
            complexity=complexity,
            estimated_latency_ms=estimated_latency,
            reasoning=reasoning,
            query_type=prediction.label,
            metadata={
                "soft_labels": prediction.soft_labels,
                "student_inference_ms": prediction.inference_time_ms,
                "routing_time_ms": routing_time_ms,
                "total_queries": self.stats.total_queries,
                "fast_path_rate": fast_path_rate,
            },
        )
        
        # Log routing decision
        if self.log_routing_decisions:
            logger.debug(
                f"[StudentRouter] Query routed: {decision.value} "
                f"(confidence={prediction.confidence:.2f}, type={prediction.label}, "
                f"latency={routing_time_ms:.2f}ms)"
            )
        
        # Record in history (thread-safe)
        with self._lock:
            self._routing_history.append({
                "timestamp": time.time(),
                "query_hash": self._get_cache_key(query),
                "decision": decision.value,
                "confidence": prediction.confidence,
                "query_type": prediction.label,
                "routing_time_ms": routing_time_ms,
            })
        
        return result
    
    def record_outcome(
        self,
        query: str,
        decision: RoutingDecision,
        actual_latency_ms: float,
        success: bool,
        response: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record the outcome of a routing decision for learning (thread-safe).
        
        This allows the system to learn from its routing decisions
        and improve over time. Successful fast path responses are cached
        for future instant retrieval.
        
        Args:
            query: The original query
            decision: The routing decision that was made
            actual_latency_ms: Actual observed latency in milliseconds
            success: Whether the request succeeded
            response: The response (for caching successful fast path responses)
        """
        with self._lock:
            # Update latency stats using incremental mean calculation
            if decision == RoutingDecision.FAST_PATH:
                n = self.stats.fast_path_count
                if n > 0:
                    self.stats.avg_student_latency_ms = (
                        (self.stats.avg_student_latency_ms * (n - 1) + actual_latency_ms) / n
                    )
                
                # Cache successful fast path responses
                if success and response:
                    self._cache_response(query, response)
                    
            elif decision == RoutingDecision.ESCALATE:
                n = self.stats.escalation_count
                if n > 0:
                    self.stats.avg_teacher_latency_ms = (
                        (self.stats.avg_teacher_latency_ms * (n - 1) + actual_latency_ms) / n
                    )
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive routing statistics (thread-safe).
        
        Returns:
            Dictionary containing all routing metrics including:
            - Basic counts (total, fast_path, escalation, hybrid)
            - Rates (fast_path_rate, escalation_rate, cache_hit_rate)
            - Latency metrics (avg, min, max, percentiles)
            - Classifier statistics
            - Cache statistics
        """
        with self._lock:
            stats_dict = self.stats.to_dict()
            stats_dict["classifier_stats"] = self.classifier.stats.copy()
            stats_dict["history_size"] = len(self._routing_history)
            stats_dict["cache_size"] = len(self._response_cache)
            stats_dict["cache_max_size"] = self._cache_max_size
            stats_dict["enabled"] = self.enabled
            stats_dict["mode"] = self.mode
            return stats_dict
    
    def reset_statistics(self) -> None:
        """Reset all statistics (thread-safe)."""
        with self._lock:
            self.stats = RouterStatistics()
            self.classifier.stats = {
                "classifications": 0,
                "by_type": {},
                "avg_inference_time_ms": 0.0,
            }
            self._routing_times.clear()
            logger.info("[StudentRouter] Statistics reset")
    
    def clear_cache(self) -> int:
        """
        Clear the response cache (thread-safe).
        
        Returns:
            Number of entries cleared
        """
        with self._lock:
            count = len(self._response_cache)
            self._response_cache.clear()
            logger.info(f"[StudentRouter] Cache cleared ({count} entries)")
            return count
    
    def save_state(self, path: Union[str, Path]) -> None:
        """
        Save router state to disk for persistence.
        
        Saves statistics, cache, and configuration to enable
        state recovery after restart.
        
        Args:
            path: Directory path to save state files
        """
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        with self._lock:
            # Save statistics
            stats_file = save_path / "router_stats.json"
            with open(stats_file, "w", encoding="utf-8") as f:
                json.dump(self.stats.to_dict(), f, indent=2)
            
            # Save classifier stats
            classifier_file = save_path / "classifier_stats.json"
            with open(classifier_file, "w", encoding="utf-8") as f:
                json.dump(self.classifier.stats, f, indent=2)
            
            # Save cache (only responses, not internal metadata)
            cache_file = save_path / "response_cache.json"
            cache_data = {
                k: {"response": v["response"], "timestamp": v["timestamp"]}
                for k, v in self._response_cache.items()
            }
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2)
            
            # Save configuration
            config_file = save_path / "router_config.json"
            config_data = {
                "enabled": self.enabled,
                "mode": self.mode,
                "confidence_threshold": self.confidence_threshold,
                "escalation_threshold": self.escalation_threshold,
                "fast_path_latency_target_ms": self.fast_path_latency_target_ms,
                "teacher_timeout_seconds": self.teacher_timeout_seconds,
                "student_timeout_seconds": self.student_timeout_seconds,
                "cache_max_size": self._cache_max_size,
                "cache_ttl_seconds": self._cache_ttl_seconds,
            }
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(config_data, f, indent=2)
        
        logger.info(f"[StudentRouter] State saved to {save_path}")
    
    def load_state(self, path: Union[str, Path]) -> bool:
        """
        Load router state from disk.
        
        Restores statistics, cache, and verifies configuration
        compatibility after restart.
        
        Args:
            path: Directory path containing state files
            
        Returns:
            True if state was loaded successfully, False otherwise
        """
        load_path = Path(path)
        
        if not load_path.exists():
            logger.warning(f"[StudentRouter] State path not found: {load_path}")
            return False
        
        try:
            with self._lock:
                # Load statistics
                stats_file = load_path / "router_stats.json"
                if stats_file.exists():
                    with open(stats_file, "r", encoding="utf-8") as f:
                        stats_data = json.load(f)
                    # Restore key statistics
                    self.stats.total_queries = stats_data.get("total_queries", 0)
                    self.stats.fast_path_count = stats_data.get("fast_path_count", 0)
                    self.stats.escalation_count = stats_data.get("escalation_count", 0)
                    self.stats.hybrid_count = stats_data.get("hybrid_count", 0)
                    self.stats.cache_hits = stats_data.get("cache_hits", 0)
                    self.stats.avg_student_latency_ms = stats_data.get("avg_student_latency_ms", 0.0)
                    self.stats.avg_teacher_latency_ms = stats_data.get("avg_teacher_latency_ms", 0.0)
                
                # Load classifier stats
                classifier_file = load_path / "classifier_stats.json"
                if classifier_file.exists():
                    with open(classifier_file, "r", encoding="utf-8") as f:
                        self.classifier.stats = json.load(f)
                
                # Load cache
                cache_file = load_path / "response_cache.json"
                if cache_file.exists():
                    with open(cache_file, "r", encoding="utf-8") as f:
                        cache_data = json.load(f)
                    # Only load non-expired entries
                    current_time = time.time()
                    self._response_cache = {
                        k: v for k, v in cache_data.items()
                        if current_time - v.get("timestamp", 0) < self._cache_ttl_seconds
                    }
            
            logger.info(f"[StudentRouter] State loaded from {load_path}")
            return True
            
        except Exception as e:
            logger.error(f"[StudentRouter] Failed to load state: {e}")
            return False
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get router health status for monitoring.
        
        Returns:
            Dictionary with health indicators including:
            - operational: Whether router is working
            - mode: Current operating mode
            - uptime_seconds: Time since initialization
            - cache_utilization: Cache usage percentage
            - recent_error_rate: Error rate in recent queries
        """
        with self._lock:
            total = max(1, self.stats.total_queries)
            return {
                "operational": self.enabled,
                "mode": self.mode,
                "uptime_seconds": time.time() - self._start_time,
                "total_queries": self.stats.total_queries,
                "cache_utilization": len(self._response_cache) / self._cache_max_size,
                "cache_hit_rate": self.stats.cache_hits / total,
                "fast_path_rate": self.stats.fast_path_count / total,
                "avg_routing_time_ms": self.stats.p50_routing_time_ms,
                "cpu_savings_percent": self.stats.cpu_savings_percent,
            }


# Module exports
__all__ = [
    "StudentRouter",
    "QueryTypeClassifier",
    "QueryComplexity",
    "RoutingDecision",
    "RoutingResult",
    "StudentPrediction",
    "RouterStatistics",
]
