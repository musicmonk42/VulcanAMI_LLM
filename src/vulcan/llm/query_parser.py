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
# P2 FIX: ROBUST ENUM PARSING
#     This module now gracefully handles unknown/malformed enum values from
#     LLM output. Instead of crashing, it falls back to safe defaults and
#     logs detailed warnings for debugging.
#
# VERSION HISTORY:
#     1.0.0 - Initial implementation for language interface architecture
#     1.1.0 - P2 FIX: Robust enum parsing with comprehensive fallback handling
# ============================================================

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

__version__ = "1.1.0"  # P2 FIX: Robust enum parsing
__author__ = "VULCAN-AGI Team"


# ============================================================
# ENUMS (defined first for use in helper functions)
# ============================================================

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


# ============================================================
# ENUM PARSING UTILITIES (P2 FIX)
# ============================================================

def _normalize_enum_value(value: Any) -> str:
    """
    Normalize a value for enum matching.
    
    P2 FIX: This handles various malformed inputs from LLM output:
    - Case insensitivity (COMPUTE, Compute, compute)
    - Extra whitespace
    - Common typos/variations
    
    Args:
        value: Raw value from LLM output
        
    Returns:
        Normalized string suitable for enum matching
    """
    if value is None:
        return ""
    
    # Convert to string and normalize
    str_value = str(value).strip().lower()
    
    # Remove common JSON artifacts
    str_value = str_value.strip('"\'')
    
    # Replace underscores/hyphens with nothing for fuzzy matching
    str_value = re.sub(r'[-_\s]+', '', str_value)
    
    return str_value


def _parse_intent_robust(value: Any) -> Tuple[QueryIntent, bool]:
    """
    Robustly parse a QueryIntent from LLM output.
    
    P2 FIX: Never crashes on invalid input. Always returns a valid enum.
    
    Args:
        value: Raw value from LLM output (may be malformed)
        
    Returns:
        Tuple of (parsed_intent, was_fallback_used)
    """
    if value is None:
        logger.warning(
            f"[P2 FIX] Intent value is None, falling back to UNKNOWN"
        )
        return QueryIntent.UNKNOWN, True
    
    normalized = _normalize_enum_value(value)
    original_value = str(value)
    
    # Direct match attempt
    try:
        # Try exact value match first
        return QueryIntent(str(value).lower().strip()), False
    except ValueError:
        pass
    
    # Fuzzy matching for common variations
    INTENT_ALIASES = {
        # Compute variations
        "compute": QueryIntent.COMPUTE,
        "calculate": QueryIntent.COMPUTE,
        "calc": QueryIntent.COMPUTE,
        "solve": QueryIntent.COMPUTE,
        "evaluate": QueryIntent.COMPUTE,
        "computation": QueryIntent.COMPUTE,
        # Explain variations
        "explain": QueryIntent.EXPLAIN,
        "describe": QueryIntent.EXPLAIN,
        "clarify": QueryIntent.EXPLAIN,
        "elaborate": QueryIntent.EXPLAIN,
        "explanation": QueryIntent.EXPLAIN,
        # Search variations
        "search": QueryIntent.SEARCH,
        "find": QueryIntent.SEARCH,
        "lookup": QueryIntent.SEARCH,
        "retrieve": QueryIntent.SEARCH,
        "query": QueryIntent.SEARCH,
        # Analyze variations
        "analyze": QueryIntent.ANALYZE,
        "analyse": QueryIntent.ANALYZE,  # British spelling
        "examine": QueryIntent.ANALYZE,
        "investigate": QueryIntent.ANALYZE,
        "assess": QueryIntent.ANALYZE,
        "analysis": QueryIntent.ANALYZE,
        # Plan variations
        "plan": QueryIntent.PLAN,
        "design": QueryIntent.PLAN,
        "strategize": QueryIntent.PLAN,
        "organize": QueryIntent.PLAN,
        "planning": QueryIntent.PLAN,
        # Compare variations
        "compare": QueryIntent.COMPARE,
        "contrast": QueryIntent.COMPARE,
        "differentiate": QueryIntent.COMPARE,
        "comparison": QueryIntent.COMPARE,
        # Unknown variations
        "unknown": QueryIntent.UNKNOWN,
        "other": QueryIntent.UNKNOWN,
        "none": QueryIntent.UNKNOWN,
    }
    
    # Try alias matching
    if normalized in INTENT_ALIASES:
        return INTENT_ALIASES[normalized], False
    
    # Try partial matching (for typos like "compue" or "explian")
    for alias, intent in INTENT_ALIASES.items():
        if alias in normalized or normalized in alias:
            logger.info(
                f"[P2 FIX] Fuzzy-matched intent '{original_value}' to {intent.value} via alias '{alias}'"
            )
            return intent, True
    
    # Final fallback
    logger.warning(
        f"[P2 FIX] Could not parse intent from '{original_value}' (normalized: '{normalized}'). "
        f"Falling back to UNKNOWN. Valid intents: {[e.value for e in QueryIntent]}"
    )
    return QueryIntent.UNKNOWN, True


def _parse_domain_robust(value: Any) -> Tuple[QueryDomain, bool]:
    """
    Robustly parse a QueryDomain from LLM output.
    
    P2 FIX: Never crashes on invalid input. Always returns a valid enum.
    
    Args:
        value: Raw value from LLM output (may be malformed)
        
    Returns:
        Tuple of (parsed_domain, was_fallback_used)
    """
    if value is None:
        logger.warning(
            f"[P2 FIX] Domain value is None, falling back to GENERAL"
        )
        return QueryDomain.GENERAL, True
    
    normalized = _normalize_enum_value(value)
    original_value = str(value)
    
    # Direct match attempt
    try:
        return QueryDomain(str(value).lower().strip()), False
    except ValueError:
        pass
    
    # Fuzzy matching for common variations
    DOMAIN_ALIASES = {
        # Math variations
        "math": QueryDomain.MATH,
        "mathematics": QueryDomain.MATH,
        "mathematical": QueryDomain.MATH,
        "arithmetic": QueryDomain.MATH,
        "algebra": QueryDomain.MATH,
        "calculus": QueryDomain.MATH,
        "numeric": QueryDomain.MATH,
        "numerical": QueryDomain.MATH,
        "maths": QueryDomain.MATH,  # British spelling
        # Logic variations
        "logic": QueryDomain.LOGIC,
        "logical": QueryDomain.LOGIC,
        "boolean": QueryDomain.LOGIC,
        "proof": QueryDomain.LOGIC,
        "proofs": QueryDomain.LOGIC,
        "reasoning": QueryDomain.LOGIC,
        "deduction": QueryDomain.LOGIC,
        # Causal variations
        "causal": QueryDomain.CAUSAL,
        "causality": QueryDomain.CAUSAL,
        "causeeffect": QueryDomain.CAUSAL,
        "cause": QueryDomain.CAUSAL,
        "effect": QueryDomain.CAUSAL,
        # General variations
        "general": QueryDomain.GENERAL,
        "other": QueryDomain.GENERAL,
        "misc": QueryDomain.GENERAL,
        "miscellaneous": QueryDomain.GENERAL,
        "unknown": QueryDomain.GENERAL,
        "common": QueryDomain.GENERAL,
        # Code variations
        "code": QueryDomain.CODE,
        "coding": QueryDomain.CODE,
        "programming": QueryDomain.CODE,
        "software": QueryDomain.CODE,
        "development": QueryDomain.CODE,
        "script": QueryDomain.CODE,
    }
    
    # Try alias matching
    if normalized in DOMAIN_ALIASES:
        return DOMAIN_ALIASES[normalized], False
    
    # Try partial matching
    for alias, domain in DOMAIN_ALIASES.items():
        if alias in normalized or normalized in alias:
            logger.info(
                f"[P2 FIX] Fuzzy-matched domain '{original_value}' to {domain.value} via alias '{alias}'"
            )
            return domain, True
    
    # Final fallback
    logger.warning(
        f"[P2 FIX] Could not parse domain from '{original_value}' (normalized: '{normalized}'). "
        f"Falling back to GENERAL. Valid domains: {[e.value for e in QueryDomain]}"
    )
    return QueryDomain.GENERAL, True


# ============================================================
# STRUCTURED QUERY CLASS
# ============================================================

@dataclass
class StructuredQuery:
    """
    Structured representation of a parsed user query.
    
    This is the output of the "Language Interface IN" step. It captures:
    - What the user wants (intent)
    - What domain it's in (domain)
    - Specific parameters needed for computation
    
    The LLM populates this structure. VULCAN's reasoning engines consume it.
    
    P2 FIX: This class now gracefully handles malformed LLM output with
    comprehensive fallback handling and detailed logging.
    
    Attributes:
        intent: What the user wants to do (compute, explain, etc.)
        domain: Subject area (math, logic, causal, etc.)
        parameters: Extracted values needed for computation
        original_text: Original user query for reference
        confidence: LLM's confidence in the parsing (0.0-1.0)
        parsing_warnings: List of warnings from parsing (P2 FIX)
    """
    intent: QueryIntent
    domain: QueryDomain
    parameters: Dict[str, Any] = field(default_factory=dict)
    original_text: str = ""
    confidence: float = 0.0
    parsing_warnings: List[str] = field(default_factory=list)  # P2 FIX
    
    def __post_init__(self):
        """Validate and normalize confidence value."""
        # P2 FIX: Handle non-numeric confidence gracefully
        if not isinstance(self.confidence, (int, float)):
            try:
                self.confidence = float(self.confidence)
                logger.warning(
                    f"[P2 FIX] Confidence was {type(self.confidence).__name__}, converted to float"
                )
            except (ValueError, TypeError):
                logger.warning(
                    f"[P2 FIX] Invalid confidence value '{self.confidence}', defaulting to 0.0"
                )
                self.confidence = 0.0
                self.parsing_warnings.append(f"Invalid confidence value, defaulted to 0.0")
        
        # Clamp confidence to valid range
        if self.confidence < 0.0:
            logger.warning(f"[P2 FIX] Confidence {self.confidence} < 0, clamping to 0.0")
            self.confidence = 0.0
            self.parsing_warnings.append(f"Confidence clamped from negative value")
        elif self.confidence > 1.0:
            logger.warning(f"[P2 FIX] Confidence {self.confidence} > 1, clamping to 1.0")
            self.confidence = 1.0
            self.parsing_warnings.append(f"Confidence clamped from value > 1.0")
    
    def is_high_confidence(self, threshold: float = 0.7) -> bool:
        """
        Check if parsing confidence exceeds threshold.
        
        Args:
            threshold: Minimum confidence level (default: 0.7)
            
        Returns:
            True if confidence >= threshold, False otherwise
        """
        return self.confidence >= threshold
    
    def had_parsing_issues(self) -> bool:
        """
        Check if there were any parsing issues/fallbacks.
        
        P2 FIX: Allows callers to detect when fallback values were used.
        
        Returns:
            True if any parsing warnings were recorded
        """
        return len(self.parsing_warnings) > 0
    
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
        
        P2 FIX: This method now NEVER raises exceptions for malformed data.
        It always returns a valid StructuredQuery, using fallback values
        when necessary and logging detailed warnings.
        
        Args:
            json_str: JSON string from LLM (may be malformed)
            original_text: Original user query
            
        Returns:
            StructuredQuery instance (always valid, may use fallback values)
        """
        parsing_warnings: List[str] = []
        
        # P2 FIX: Handle None/empty input gracefully
        if not json_str or not json_str.strip():
            logger.warning("[P2 FIX] Empty JSON string received, using all defaults")
            return cls(
                intent=QueryIntent.UNKNOWN,
                domain=QueryDomain.GENERAL,
                parameters={"raw_text": original_text},
                original_text=original_text,
                confidence=0.0,
                parsing_warnings=["Empty JSON input"]
            )
        
        # P2 FIX: Try to extract JSON from potentially messy LLM output
        json_str_clean = json_str.strip()
        
        # Handle markdown code blocks
        if "```json" in json_str_clean:
            match = re.search(r'```json\s*(.*?)\s*```', json_str_clean, re.DOTALL)
            if match:
                json_str_clean = match.group(1)
        elif "```" in json_str_clean:
            match = re.search(r'```\s*(.*?)\s*```', json_str_clean, re.DOTALL)
            if match:
                json_str_clean = match.group(1)
        
        # Try to parse JSON
        try:
            data = json.loads(json_str_clean)
        except json.JSONDecodeError as e:
            logger.warning(
                f"[P2 FIX] JSON parse failed: {e}. Input preview: '{json_str_clean[:100]}...'"
            )
            parsing_warnings.append(f"JSON parse error: {str(e)[:50]}")
            
            # P2 FIX: Try to extract key-value pairs with regex as last resort
            data = {}
            intent_match = re.search(r'"?intent"?\s*[:=]\s*"?(\w+)"?', json_str_clean, re.I)
            if intent_match:
                data["intent"] = intent_match.group(1)
            domain_match = re.search(r'"?domain"?\s*[:=]\s*"?(\w+)"?', json_str_clean, re.I)
            if domain_match:
                data["domain"] = domain_match.group(1)
            conf_match = re.search(r'"?confidence"?\s*[:=]\s*([0-9.]+)', json_str_clean, re.I)
            if conf_match:
                data["confidence"] = conf_match.group(1)
            
            if not data:
                # Complete fallback
                return cls(
                    intent=QueryIntent.UNKNOWN,
                    domain=QueryDomain.GENERAL,
                    parameters={"raw_text": original_text, "parse_error": str(e)},
                    original_text=original_text,
                    confidence=0.0,
                    parsing_warnings=parsing_warnings
                )
        
        # P2 FIX: Robust enum parsing with detailed logging
        intent, intent_fallback = _parse_intent_robust(data.get("intent"))
        if intent_fallback:
            parsing_warnings.append(f"Intent fallback used (raw: {data.get('intent')})")
        
        domain, domain_fallback = _parse_domain_robust(data.get("domain"))
        if domain_fallback:
            parsing_warnings.append(f"Domain fallback used (raw: {data.get('domain')})")
        
        # P2 FIX: Robust confidence parsing
        raw_confidence = data.get("confidence", 0.5)
        try:
            confidence = float(raw_confidence)
        except (ValueError, TypeError):
            logger.warning(f"[P2 FIX] Invalid confidence '{raw_confidence}', defaulting to 0.5")
            confidence = 0.5
            parsing_warnings.append(f"Confidence parse error (raw: {raw_confidence})")
        
        # P2 FIX: Robust parameters parsing
        parameters = data.get("parameters", {})
        if not isinstance(parameters, dict):
            logger.warning(f"[P2 FIX] Parameters is not a dict: {type(parameters).__name__}")
            parameters = {"raw_parameters": parameters}
            parsing_warnings.append(f"Parameters converted to dict wrapper")
        
        return cls(
            intent=intent,
            domain=domain,
            parameters=parameters,
            original_text=original_text,
            confidence=confidence,
            parsing_warnings=parsing_warnings
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.
        
        Returns:
            Dictionary representation
        """
        result = {
            "intent": self.intent.value,
            "domain": self.domain.value,
            "parameters": self.parameters,
            "original_text": self.original_text,
            "confidence": self.confidence,
        }
        
        # Include parsing warnings if any (for debugging)
        if self.parsing_warnings:
            result["_parsing_warnings"] = self.parsing_warnings
        
        return result
    
    def __repr__(self) -> str:
        warning_indicator = " [!]" if self.parsing_warnings else ""
        return (
            f"StructuredQuery(intent={self.intent.value}, domain={self.domain.value}, "
            f"confidence={self.confidence:.2f}{warning_indicator})"
        )


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    "QueryIntent",
    "QueryDomain",
    "StructuredQuery",
    # P2 FIX: Expose robust parsing utilities
    "_parse_intent_robust",
    "_parse_domain_robust",
    "_normalize_enum_value",
]
