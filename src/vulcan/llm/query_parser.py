# ============================================================
# VULCAN-AGI Query Parser Module
# LLM-based parsing of natural language queries to structured format
# ============================================================
#
# PURPOSE:
#     This module implements the "Language Interface IN" layer of VULCAN's
#     architecture. It uses LLMs to parse natural language user queries into
#     structured formats that VULCAN's reasoning engines can process.
#
# ARCHITECTURE:
#     User (natural language)
#         ↓
#     LLM (parse intent, extract parameters) ← LANGUAGE INTERFACE IN
#         ↓
#     VULCAN reasoning engines (actual computation)
#         ↓
#     Structured result {answer: 42, proof: [...]}
#         ↓
#     LLM (format as natural language) ← LANGUAGE INTERFACE OUT
#         ↓
#     User (natural language response)
#
# KEY INSIGHT:
#     The LLM does NOT reason or compute answers. It ONLY understands what
#     the user wants and extracts the relevant parameters. VULCAN's reasoning
#     systems do ALL the actual thinking.
#
# VERSION HISTORY:
#     1.0.0 - Initial implementation for language interface architecture
# ============================================================

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict

logger = logging.getLogger(__name__)

__version__ = "1.0.0"
__author__ = "VULCAN-AGI Team"


class QueryIntent(Enum):
    """
    The user's intent - what they want to do.
    
    This is NOT about HOW to do it (that's VULCAN's job), but WHAT the user wants.
    """
    COMPUTE = "compute"      # Calculate, solve, evaluate
    EXPLAIN = "explain"      # Describe, clarify, elaborate
    SEARCH = "search"        # Find, lookup, retrieve
    ANALYZE = "analyze"      # Examine, investigate, assess
    PLAN = "plan"            # Design, strategize, organize
    COMPARE = "compare"      # Contrast, differentiate, evaluate differences
    UNKNOWN = "unknown"      # Cannot determine intent


class QueryDomain(Enum):
    """
    The domain or subject area of the query.
    
    This helps route to the appropriate VULCAN reasoning engine.
    """
    MATH = "math"            # Mathematical operations, calculations
    LOGIC = "logic"          # Logical reasoning, proofs
    CAUSAL = "causal"        # Cause-effect relationships
    GENERAL = "general"      # General knowledge, facts
    CODE = "code"            # Programming, software


@dataclass
class StructuredQuery:
    """
    Structured representation of a parsed user query.
    
    This is the output of the "Language Interface IN" step. It captures:
    - What the user wants (intent)
    - What domain it's in (domain)
    - Specific parameters needed for computation
    
    The LLM populates this structure. VULCAN's reasoning engines consume it.
    
    Attributes:
        intent: What the user wants to do (compute, explain, etc.)
        domain: Subject area (math, logic, causal, etc.)
        parameters: Extracted values needed for computation
        original_text: Original user query for reference
        confidence: LLM's confidence in the parsing (0.0-1.0)
    """
    intent: QueryIntent
    domain: QueryDomain
    parameters: Dict[str, Any] = field(default_factory=dict)
    original_text: str = ""
    confidence: float = 0.0
    
    def __post_init__(self):
        """Validate confidence is in valid range."""
        if not isinstance(self.confidence, (int, float)):
            raise TypeError(f"confidence must be a number, got {type(self.confidence).__name__}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {self.confidence}")
    
    def is_high_confidence(self, threshold: float = 0.7) -> bool:
        """
        Check if parsing confidence exceeds threshold.
        
        Args:
            threshold: Minimum confidence level (default: 0.7)
            
        Returns:
            True if confidence >= threshold, False otherwise
        """
        return self.confidence >= threshold
    
    def validate(self) -> bool:
        """
        Validate that the structured query is well-formed.
        
        Returns:
            True if valid, False otherwise
        """
        # Check enums are valid
        if not isinstance(self.intent, QueryIntent):
            return False
        if not isinstance(self.domain, QueryDomain):
            return False
        
        # Check parameters is a dict
        if not isinstance(self.parameters, dict):
            return False
        
        # Check confidence is valid
        if not isinstance(self.confidence, (int, float)):
            return False
        if not 0.0 <= self.confidence <= 1.0:
            return False
        
        return True
    
    @classmethod
    def from_json(cls, json_str: str, original_text: str = "") -> "StructuredQuery":
        """
        Parse JSON output from LLM into StructuredQuery.
        
        The LLM outputs JSON with intent, domain, parameters, and confidence.
        This method converts that JSON into a structured StructuredQuery object.
        
        Args:
            json_str: JSON string from LLM
            original_text: Original user query
            
        Returns:
            StructuredQuery instance
            
        Raises:
            json.JSONDecodeError: If JSON is invalid
            KeyError: If required fields are missing
            ValueError: If enum values are invalid
        """
        try:
            data = json.loads(json_str)
            
            # Parse intent enum (with fallback to UNKNOWN)
            intent_str = data.get("intent", "unknown")
            try:
                intent = QueryIntent(intent_str)
            except ValueError:
                logger.warning(f"Invalid intent '{intent_str}', defaulting to UNKNOWN")
                intent = QueryIntent.UNKNOWN
            
            # Parse domain enum (with fallback to GENERAL)
            domain_str = data.get("domain", "general")
            try:
                domain = QueryDomain(domain_str)
            except ValueError:
                logger.warning(f"Invalid domain '{domain_str}', defaulting to GENERAL")
                domain = QueryDomain.GENERAL
            
            return cls(
                intent=intent,
                domain=domain,
                parameters=data.get("parameters", {}),
                original_text=original_text,
                confidence=float(data.get("confidence", 0.5))
            )
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON output: {e}")
            logger.debug(f"Invalid JSON: {json_str[:200]}")
            raise
        except Exception as e:
            logger.error(f"Error creating StructuredQuery from JSON: {e}")
            raise
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.
        
        Returns:
            Dictionary representation
        """
        return {
            "intent": self.intent.value,
            "domain": self.domain.value,
            "parameters": self.parameters,
            "original_text": self.original_text,
            "confidence": self.confidence,
        }
    
    def __repr__(self) -> str:
        return (
            f"StructuredQuery(intent={self.intent.value}, domain={self.domain.value}, "
            f"confidence={self.confidence:.2f})"
        )


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    "QueryIntent",
    "QueryDomain",
    "StructuredQuery",
]
