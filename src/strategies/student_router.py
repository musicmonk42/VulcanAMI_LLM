"""
Student Router for Knowledge Distillation-based Hybrid Routing

This module implements the "Master and Apprentice" pattern for CPU optimization:
- The Student (lightweight) handles easy queries with low latency
- The Teacher (Vulcan/heavy) handles complex queries requiring deep reasoning

The Student router performs fast triage on incoming queries:
1. If confidence > threshold: Use fast path (Student answer or cached response)
2. If confidence < threshold: Escalate to Teacher (full Vulcan pipeline)

This prevents using a "Supercomputer to answer 'What time is it?'" by reserving
heavy CPU cycles for questions that actually require them.

PERFORMANCE TARGETS:
- Student inference: <50ms (fits in CPU L2/L3 cache)
- Teacher inference: ~60s (full reasoning pipeline)
- Expected CPU reduction: 80% for typical workloads (80% easy queries)
"""

import hashlib
import json
import logging
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

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
    """Result of routing decision."""
    decision: RoutingDecision
    confidence: float
    complexity: QueryComplexity
    estimated_latency_ms: float
    reasoning: str
    query_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StudentPrediction:
    """Prediction from the Student model."""
    label: str
    confidence: float
    soft_labels: Dict[str, float]
    inference_time_ms: float


class QueryTypeClassifier:
    """
    Lightweight query classifier that runs entirely in CPU cache.
    
    This is the "Student" in the Master-Apprentice pattern.
    It doesn't understand the "why", but memorizes the "what" - 
    quickly guessing query types without heavy computation.
    """
    
    # Pattern-based classification (ultra-fast, regex-based)
    GREETING_PATTERNS = [
        r"^(hi|hello|hey|good morning|good afternoon|good evening|howdy|greetings)\b",
        r"^(how are you|how's it going|what's up)\??$",
        r"^(thanks|thank you|cheers|bye|goodbye|see you)\b",
    ]
    
    SIMPLE_FACTUAL_PATTERNS = [
        r"^(what is|what's|who is|who's|when was|where is|where's)\s",
        r"^(define|meaning of|definition of)\s",
        r"^(how many|how much|how old|how tall|how far)\s",
    ]
    
    COMPLEX_REASONING_PATTERNS = [
        r"\b(explain|analyze|compare|contrast|evaluate|discuss)\b",
        r"\b(why does|how does|what causes|what happens if)\b",
        r"\b(pros and cons|advantages|disadvantages)\b",
        r"\b(relationship between|difference between)\b",
    ]
    
    CODE_PATTERNS = [
        r"\b(write|create|implement|code|function|class|script|program)\b.*\b(python|javascript|java|c\+\+|rust|go|sql)\b",
        r"\b(debug|fix|refactor|optimize)\b.*\b(code|function|method|class)\b",
        r"```",  # Code blocks
        r"\b(api|endpoint|database|query|algorithm)\b",
    ]
    
    CREATIVE_PATTERNS = [
        r"\b(write|create|compose|draft)\b.*\b(story|poem|essay|article|blog)\b",
        r"\b(creative|imaginative|fictional|narrative)\b",
        r"\b(brainstorm|ideas for|suggestions for)\b",
    ]
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the classifier with optional config."""
        config = config or {}
        
        # Compile patterns for speed
        self._greeting_re = [re.compile(p, re.IGNORECASE) for p in self.GREETING_PATTERNS]
        self._simple_re = [re.compile(p, re.IGNORECASE) for p in self.SIMPLE_FACTUAL_PATTERNS]
        self._complex_re = [re.compile(p, re.IGNORECASE) for p in self.COMPLEX_REASONING_PATTERNS]
        self._code_re = [re.compile(p, re.IGNORECASE) for p in self.CODE_PATTERNS]
        self._creative_re = [re.compile(p, re.IGNORECASE) for p in self.CREATIVE_PATTERNS]
        
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
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Student Router."""
        config = config or {}
        
        # Get distillation config
        distillation_config = config.get("distillation_config", {})
        
        # Core settings
        self.enabled = distillation_config.get("enabled", True)
        self.mode = distillation_config.get("mode", "active")  # "active" or "passive"
        self.confidence_threshold = distillation_config.get("confidence_threshold", 0.85)
        self.escalation_threshold = distillation_config.get("escalation_threshold", 0.40)
        self.fast_path_latency_target_ms = distillation_config.get("fast_path_latency_target_ms", 50)
        self.teacher_timeout_seconds = distillation_config.get("teacher_timeout_seconds", 60)
        self.student_timeout_seconds = distillation_config.get("student_timeout_seconds", 5)
        self.log_routing_decisions = distillation_config.get("log_routing_decisions", True)
        
        # Initialize classifier
        self.classifier = QueryTypeClassifier(distillation_config)
        
        # Response cache for instant answers
        self._response_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_max_size = 1000
        self._cache_ttl_seconds = 300
        
        # Routing history for learning
        self._routing_history: deque = deque(maxlen=10000)
        
        # Statistics
        self.stats = {
            "total_queries": 0,
            "fast_path_count": 0,
            "escalation_count": 0,
            "hybrid_count": 0,
            "cache_hits": 0,
            "avg_student_latency_ms": 0.0,
            "avg_teacher_latency_ms": 0.0,
            "cpu_savings_percent": 0.0,
        }
        
        logger.info(
            f"StudentRouter initialized: mode={self.mode}, "
            f"confidence_threshold={self.confidence_threshold}, "
            f"escalation_threshold={self.escalation_threshold}"
        )
    
    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for query."""
        return hashlib.md5(query.strip().lower().encode()).hexdigest()
    
    def _check_cache(self, query: str) -> Optional[Dict[str, Any]]:
        """Check if query has a cached response."""
        key = self._get_cache_key(query)
        if key in self._response_cache:
            entry = self._response_cache[key]
            if time.time() - entry["timestamp"] < self._cache_ttl_seconds:
                self.stats["cache_hits"] += 1
                return entry["response"]
            else:
                del self._response_cache[key]
        return None
    
    def _cache_response(self, query: str, response: Dict[str, Any]):
        """Cache a response for future use."""
        if len(self._response_cache) >= self._cache_max_size:
            # Remove oldest entries
            oldest_keys = sorted(
                self._response_cache.keys(),
                key=lambda k: self._response_cache[k]["timestamp"]
            )[:100]
            for k in oldest_keys:
                del self._response_cache[k]
        
        key = self._get_cache_key(query)
        self._response_cache[key] = {
            "response": response,
            "timestamp": time.time(),
        }
    
    def route(self, query: str) -> RoutingResult:
        """
        Route a query to the appropriate handler.
        
        This is the main entry point for the triage system.
        
        Args:
            query: The input query string
            
        Returns:
            RoutingResult with decision and metadata
        """
        start_time = time.perf_counter()
        self.stats["total_queries"] += 1
        
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
        
        # Step 4: Make routing decision
        if prediction.confidence >= self.confidence_threshold:
            # High confidence - use fast path
            decision = RoutingDecision.FAST_PATH
            self.stats["fast_path_count"] += 1
            estimated_latency = self.fast_path_latency_target_ms
            reasoning = (
                f"Student confident ({prediction.confidence:.0%}) that this is a "
                f"{prediction.label} query - using fast path"
            )
        elif prediction.confidence <= self.escalation_threshold:
            # Low confidence - escalate to Teacher
            decision = RoutingDecision.ESCALATE
            self.stats["escalation_count"] += 1
            estimated_latency = self.teacher_timeout_seconds * 1000
            reasoning = (
                f"Student uncertain ({prediction.confidence:.0%}) - "
                f"escalating to Teacher for full analysis"
            )
        else:
            # Moderate confidence - try hybrid approach
            decision = RoutingDecision.HYBRID
            self.stats["hybrid_count"] += 1
            estimated_latency = self.student_timeout_seconds * 1000
            reasoning = (
                f"Student moderately confident ({prediction.confidence:.0%}) - "
                f"will try fast path first, escalate if needed"
            )
        
        routing_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Update CPU savings estimate
        fast_path_rate = self.stats["fast_path_count"] / max(1, self.stats["total_queries"])
        self.stats["cpu_savings_percent"] = fast_path_rate * 80  # Assume 80% CPU saved per fast path
        
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
                "total_queries": self.stats["total_queries"],
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
        
        # Record in history
        self._routing_history.append({
            "timestamp": time.time(),
            "query_hash": self._get_cache_key(query),
            "decision": decision.value,
            "confidence": prediction.confidence,
            "query_type": prediction.label,
        })
        
        return result
    
    def record_outcome(
        self,
        query: str,
        decision: RoutingDecision,
        actual_latency_ms: float,
        success: bool,
        response: Optional[Dict[str, Any]] = None,
    ):
        """
        Record the outcome of a routing decision for learning.
        
        This allows the system to learn from its routing decisions
        and improve over time.
        
        Args:
            query: The original query
            decision: The routing decision made
            actual_latency_ms: Actual latency observed
            success: Whether the request succeeded
            response: The response (for caching successful fast path responses)
        """
        # Update latency stats
        if decision == RoutingDecision.FAST_PATH:
            n = self.stats["fast_path_count"]
            self.stats["avg_student_latency_ms"] = (
                (self.stats["avg_student_latency_ms"] * (n - 1) + actual_latency_ms) / max(1, n)
            )
            
            # Cache successful fast path responses
            if success and response:
                self._cache_response(query, response)
                
        elif decision == RoutingDecision.ESCALATE:
            n = self.stats["escalation_count"]
            self.stats["avg_teacher_latency_ms"] = (
                (self.stats["avg_teacher_latency_ms"] * (n - 1) + actual_latency_ms) / max(1, n)
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get routing statistics."""
        total = self.stats["total_queries"]
        return {
            **self.stats,
            "fast_path_rate": self.stats["fast_path_count"] / max(1, total),
            "escalation_rate": self.stats["escalation_count"] / max(1, total),
            "hybrid_rate": self.stats["hybrid_count"] / max(1, total),
            "cache_hit_rate": self.stats["cache_hits"] / max(1, total),
            "classifier_stats": self.classifier.stats,
            "history_size": len(self._routing_history),
        }
    
    def reset_statistics(self):
        """Reset all statistics."""
        self.stats = {
            "total_queries": 0,
            "fast_path_count": 0,
            "escalation_count": 0,
            "hybrid_count": 0,
            "cache_hits": 0,
            "avg_student_latency_ms": 0.0,
            "avg_teacher_latency_ms": 0.0,
            "cpu_savings_percent": 0.0,
        }
        self.classifier.stats = {
            "classifications": 0,
            "by_type": {},
            "avg_inference_time_ms": 0.0,
        }
        logger.info("[StudentRouter] Statistics reset")


# Module exports
__all__ = [
    "StudentRouter",
    "QueryTypeClassifier",
    "QueryComplexity",
    "RoutingDecision",
    "RoutingResult",
    "StudentPrediction",
]
