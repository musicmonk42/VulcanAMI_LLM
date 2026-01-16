"""
Philosophical Query Router

Classifies philosophical queries to route to appropriate handler.
Prevents meta-reasoning block where philosophical queries about ethics/dilemmas
are misrouted to meta-reasoning system for self-analysis.

Industry Standards Applied:
- Chain of Responsibility pattern for query classification
- Intent detection before routing (not tool selection before intent)
- Defensive regex with explicit anchoring
- Comprehensive logging for debugging

BUG FIX #3: Meta-Reasoning Block - Philosophical Query Routing (The Bureaucrat)
"""

import logging
import re
from enum import Enum
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


class PhilosophicalQueryType(Enum):
    """Classification of philosophical query intent."""
    EXTERNAL_DILEMMA = "external_dilemma"      # User asks for ethical decision
    EXTERNAL_ANALYSIS = "external_analysis"    # User asks for ethical analysis
    INTERNAL_REFLECTION = "internal_reflection" # Query about AI's own nature
    META_REASONING = "meta_reasoning"          # System self-analysis


class PhilosophicalQueryClassifier:
    """
    Classifies philosophical queries to route to appropriate handler.
    
    Design Pattern: Chain of Responsibility
    Industry Standard: Intent classification before routing
    
    BUG FIX #3: This prevents the "meta-reasoning block" where external
    philosophical questions (trolley problem, ethics) are routed to the
    meta-reasoning layer for system self-analysis instead of being answered.
    """
    
    # Indicators that user wants a DECISION (A or B, Yes or No)
    DECISION_REQUIRED_PATTERNS = [
        r'\b(choose|select|pick)\s+(one|a|b|option)',
        r'\b(yes|no)\s*(or|\|)\s*(no|yes)',
        r'\b(a|b)\s*(or|\|)\s*(b|a)\b',
        r'(which|what)\s+(would|should|do)\s+you\s+(choose|pick|select)',
        r'you\s+must\s+(choose|decide|act)',
        r'(pull|don\'t pull)\s+the\s+lever',
        r'is\s+(it|this)\s+(permissible|allowed|ethical|moral)\?',
        r'(should|would)\s+(i|you|one)\s+',
    ]
    
    # Indicators of external ethical dilemma (not about AI itself)
    EXTERNAL_DILEMMA_INDICATORS = [
        "trolley", "lever", "track", "divert",
        "patient", "doctor", "bystander",
        "innocent", "guilty", "harm",
        "save", "kill", "die", "death",
        "utilitarian", "deontological", "kantian",
        "moral dilemma", "ethical dilemma",
        "thought experiment",
    ]
    
    # Indicators of internal/self-referential query (about AI)
    INTERNAL_REFLECTION_INDICATORS = [
        "are you", "do you", "can you", "would you",
        "your design", "your goals", "your purpose",
        "self-aware", "consciousness", "sentient",
        "feel", "believe", "think", "want",
        "as an ai", "as a system", "as vulcan",
    ]
    
    @classmethod
    def classify(cls, query: str, context: Dict[str, Any] = None) -> Tuple[PhilosophicalQueryType, float]:
        """
        Classify philosophical query intent.
        
        Industry Standard: Multi-stage classification with confidence scoring
        
        Args:
            query: User query string
            context: Optional context dictionary (unused but reserved for future)
        
        Returns:
            Tuple of (query_type, confidence)
        
        Example:
            >>> classify("A trolley is headed toward 5 people. Choose A or B.")
            (PhilosophicalQueryType.EXTERNAL_DILEMMA, 0.9)
        """
        query_lower = query.lower()
        
        # Check for decision-required patterns first (highest priority)
        decision_required = any(
            re.search(pattern, query_lower) 
            for pattern in cls.DECISION_REQUIRED_PATTERNS
        )
        
        # Count indicators
        external_score = sum(
            1 for ind in cls.EXTERNAL_DILEMMA_INDICATORS 
            if ind in query_lower
        )
        internal_score = sum(
            1 for ind in cls.INTERNAL_REFLECTION_INDICATORS 
            if ind in query_lower
        )
        
        logger.debug(
            f"[PhilosophicalClassifier] Query analysis: "
            f"decision_required={decision_required}, "
            f"external_score={external_score}, "
            f"internal_score={internal_score}"
        )
        
        # Classification logic
        if decision_required and external_score > internal_score:
            logger.info(
                f"[PhilosophicalClassifier] Classified as EXTERNAL_DILEMMA "
                f"(decision_required=True, external>{internal})"
            )
            return PhilosophicalQueryType.EXTERNAL_DILEMMA, 0.9
        
        if external_score > internal_score + 2:
            logger.info(
                f"[PhilosophicalClassifier] Classified as EXTERNAL_ANALYSIS "
                f"(external_score={external_score} >> internal_score={internal_score})"
            )
            return PhilosophicalQueryType.EXTERNAL_ANALYSIS, 0.8
        
        if internal_score > external_score:
            logger.info(
                f"[PhilosophicalClassifier] Classified as INTERNAL_REFLECTION "
                f"(internal_score={internal_score} > external_score={external_score})"
            )
            return PhilosophicalQueryType.INTERNAL_REFLECTION, 0.75
        
        # Default to external analysis for philosophical queries
        logger.debug("[PhilosophicalClassifier] Defaulting to EXTERNAL_ANALYSIS")
        return PhilosophicalQueryType.EXTERNAL_ANALYSIS, 0.6
    
    @classmethod
    def requires_decision(cls, query: str) -> bool:
        """
        Check if query requires a concrete decision (A/B, Yes/No).
        
        Args:
            query: User query string
        
        Returns:
            True if query requires a decision, False otherwise
        
        Example:
            >>> requires_decision("Choose A or B")
            True
            >>> requires_decision("Explain the trolley problem")
            False
        """
        query_lower = query.lower()
        return any(
            re.search(pattern, query_lower)
            for pattern in cls.DECISION_REQUIRED_PATTERNS
        )


def route_philosophical_query(query: str, context: Dict[str, Any] = None) -> str:
    """
    Route philosophical query to appropriate handler.
    
    Industry Standard: Intent-based routing with explicit decision support
    
    Args:
        query: User query string
        context: Optional context dictionary
    
    Returns:
        Handler name string (e.g., "philosophical_decision_engine")
    
    Example:
        >>> route_philosophical_query("A trolley problem: Choose A or B")
        "philosophical_decision_engine"
    """
    query_type, confidence = PhilosophicalQueryClassifier.classify(query, context)
    
    logger.info(
        f"[PhilosophicalRouter] Routing query: type={query_type.value}, "
        f"confidence={confidence:.2f}"
    )
    
    if query_type == PhilosophicalQueryType.EXTERNAL_DILEMMA:
        # User wants a decision - route to decision engine
        return "philosophical_decision_engine"
    
    if query_type == PhilosophicalQueryType.EXTERNAL_ANALYSIS:
        # User wants analysis - route to ethical analysis
        return "ethical_analysis_engine"
    
    if query_type == PhilosophicalQueryType.INTERNAL_REFLECTION:
        # Query about AI nature - route with transparency
        return "self_reflection_handler"
    
    # Meta-reasoning for system internals only
    return "meta_reasoning_engine"
