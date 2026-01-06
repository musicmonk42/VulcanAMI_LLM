"""
Answer validation to ensure reasoning results relate to queries.

This module provides a critical safety layer that catches cases where reasoning
engines return confident but nonsensical answers that have no connection to the
original question.

Module: vulcan.reasoning.answer_validator
Part of the VULCAN-AGI system.

Architecture
------------
The AnswerValidator acts as a post-processing filter that examines reasoning
results before they are returned to users. It uses domain-specific validators
to check that answers match expected formats and content types.

Key Components:
    - AnswerValidator: Main validation class with pluggable validators
    - ValidationResult: Structured result containing pass/fail and explanations
    - ValidationFailureReason: Enumeration of possible failure types

Examples of Invalid Outputs This Catches:
    - Query: "Is {A→B, B→C, ¬C} satisfiable?" -> Answer: "3x**2 + 2x" (WRONG!)
    - Query: "Formalize in FOL" -> Answer: "exp(x)" (WRONG!)
    - Query: "Is it permissible?" -> Answer: "derivative = 6x" (WRONG!)

Usage Example
-------------
    >>> from vulcan.reasoning.answer_validator import validate_reasoning_result
    >>> 
    >>> # After getting a result from a reasoning engine
    >>> result = engine.reason(query)
    >>> validation = validate_reasoning_result(query, result)
    >>> 
    >>> if not validation.valid:
    ...     logger.warning(f"Invalid result: {validation.explanation}")
    ...     # Try different engine or return error

Thread Safety
-------------
All methods are thread-safe. The AnswerValidator class is stateless and can be
shared across threads.

Performance
-----------
Validation is designed to be fast (<1ms for typical queries). Pattern matching
uses pre-compiled regex where possible.

See Also
--------
    - vulcan.reasoning.reasoning_integration: Main reasoning pipeline
    - vulcan.reasoning.reasoning_types: ReasoningResult types
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Tuple

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

# Pre-compiled regex patterns for performance
_MATH_EXPRESSION_PATTERN = re.compile(r'\d+x?\*\*\d+')
_PROBABILITY_PATTERN = re.compile(r'0?\.\d+|1\.0|\d+%')
_NUMERIC_PATTERN = re.compile(r'\d+\.?\d*')

# Known bug output patterns
_KNOWN_BUG_OUTPUTS: FrozenSet[str] = frozenset([
    '3x**2 + 2x',
    '3*x**2 + 2*x',
    '3x^2 + 2x',
])

# Logic symbols (Unicode)
_LOGIC_SYMBOLS: FrozenSet[str] = frozenset(['→', '∧', '∨', '¬', '∀', '∃', '⊢', '⊨'])


# =============================================================================
# Enums and Data Classes
# =============================================================================

class ValidationFailureReason(Enum):
    """
    Enumeration of reasons why answer validation might fail.
    
    These reasons help diagnose why a particular answer was rejected,
    enabling appropriate corrective action.
    
    Attributes:
        MISSING_REQUIRED_FORMAT: Answer lacks expected format (e.g., no YES/NO)
        WRONG_DOMAIN: Answer is from wrong domain (e.g., calculus for logic query)
        NO_ANSWER_PROVIDED: Answer doesn't provide actual response
        NONSENSICAL_OUTPUT: Answer is obviously unrelated to query
        MISSING_REQUIRED_ELEMENTS: Answer lacks required elements (e.g., no quantifiers for FOL)
    """
    MISSING_REQUIRED_FORMAT = "missing_required_format"
    WRONG_DOMAIN = "wrong_domain"
    NO_ANSWER_PROVIDED = "no_answer_provided"
    NONSENSICAL_OUTPUT = "nonsensical_output"
    MISSING_REQUIRED_ELEMENTS = "missing_required_elements"


@dataclass(frozen=False)
class ValidationResult:
    """
    Result of answer validation.
    
    This dataclass encapsulates all information about whether a reasoning
    result is valid for the given query.
    
    Attributes:
        valid: True if the answer passes all validation checks
        confidence: Confidence in the validation result (0.0 to 1.0)
        failures: List of specific failure reasons if invalid
        explanation: Human-readable explanation of the validation outcome
        suggestions: List of suggestions for corrective action
        
    Example:
        >>> result = ValidationResult(
        ...     valid=False,
        ...     confidence=0.0,
        ...     failures=[ValidationFailureReason.WRONG_DOMAIN],
        ...     explanation="Answer is mathematical but query is about logic",
        ...     suggestions=["Use symbolic reasoning engine instead"]
        ... )
    """
    valid: bool
    confidence: float
    failures: List[ValidationFailureReason]
    explanation: str
    suggestions: List[str] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """Validate field values after initialization."""
        # Ensure confidence is within valid range
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {self.confidence}")
        
        # Ensure failures is a list
        if self.failures is None:
            object.__setattr__(self, 'failures', [])
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        return {
            'valid': self.valid,
            'confidence': self.confidence,
            'failures': [f.value for f in self.failures],
            'explanation': self.explanation,
            'suggestions': self.suggestions,
        }


# =============================================================================
# Type Aliases
# =============================================================================

ValidatorFunc = Callable[[str, Dict[str, Any]], ValidationResult]


# =============================================================================
# Main Validator Class
# =============================================================================

class AnswerValidator:
    """
    Validates that reasoning engine outputs actually address the query.
    
    This is a critical safety layer that prevents the system from
    returning confident but nonsensical answers. It uses a collection
    of domain-specific validators to check answer appropriateness.
    
    The validator supports:
        - First-Order Logic (FOL) answers
        - SAT/Satisfiability answers
        - Yes/No answers
        - Mathematical computation answers
        - Proof verification answers
        - Bayesian inference answers
        - Ethical/deontic reasoning answers
    
    Thread Safety:
        This class is stateless and thread-safe. Multiple threads can
        share a single instance safely.
    
    Usage:
        >>> validator = AnswerValidator()
        >>> result = validator.validate(query, reasoning_result)
        >>> 
        >>> if not result.valid:
        ...     logger.warning(f"Invalid result: {result.explanation}")
        ...     # Handle invalid result
    
    Attributes:
        validators: Dictionary mapping answer types to validator functions
    """
    
    # Class-level constants for query type detection
    _FOL_KEYWORDS: FrozenSet[str] = frozenset([
        'formalize', 'first-order logic', 'fol', 'quantifier',
        '∀', '∃', 'forall', 'exists', 'predicate logic'
    ])
    
    _SAT_KEYWORDS: FrozenSet[str] = frozenset([
        'satisfiable', 'unsatisfiable', 'sat', 'contradiction', 'model',
        'propositional', 'boolean formula'
    ])
    
    _ETHICAL_KEYWORDS: FrozenSet[str] = frozenset([
        'permissible', 'impermissible', 'forbidden', 'obligatory',
        'moral', 'ethical', 'trolley', 'dilemma', 'deontic'
    ])
    
    _YES_NO_KEYWORDS: FrozenSet[str] = frozenset([
        'yes/no', 'permissible?', 'is it', 'does it', 'can it', 'should'
    ])
    
    _MATH_KEYWORDS: FrozenSet[str] = frozenset([
        'compute', 'calculate', 'solve', '∫', '∑', 'derivative',
        'integrate', 'differentiate', 'evaluate'
    ])
    
    _PROOF_KEYWORDS: FrozenSet[str] = frozenset([
        'proof', 'verify', 'valid', 'invalid', 'step', 'theorem'
    ])
    
    _BAYESIAN_KEYWORDS: FrozenSet[str] = frozenset([
        'bayes', 'bayesian', 'posterior', 'p(', 'probability', 'prior',
        'likelihood', 'conditional probability'
    ])
    
    def __init__(self) -> None:
        """Initialize the AnswerValidator with domain-specific validators."""
        self.validators: Dict[str, ValidatorFunc] = {
            'fol': self._validate_fol_answer,
            'sat': self._validate_sat_answer,
            'yes_no': self._validate_yes_no_answer,
            'mathematical': self._validate_mathematical_answer,
            'proof': self._validate_proof_answer,
            'bayesian': self._validate_bayesian_answer,
            'ethical': self._validate_ethical_answer,
        }
    
    def validate(
        self, 
        query: str, 
        result: Dict[str, Any],
        expected_type: Optional[str] = None
    ) -> ValidationResult:
        """
        Validate that a reasoning result addresses the query.
        
        This method performs comprehensive validation including:
        1. Type-specific validation based on expected answer type
        2. General nonsense detection for cross-domain mismatches
        
        Args:
            query: Original query text to validate against
            result: Reasoning result dictionary, must contain 'conclusion' key
            expected_type: Expected answer type. If None, will be inferred from query.
                Valid types: 'fol', 'sat', 'yes_no', 'mathematical', 'proof',
                'bayesian', 'ethical', 'general'
        
        Returns:
            ValidationResult containing:
                - valid: True if all checks pass
                - confidence: 1.0 if valid, 0.0 otherwise
                - failures: List of ValidationFailureReason if invalid
                - explanation: Human-readable explanation
                - suggestions: Corrective action suggestions
        
        Raises:
            TypeError: If query is not a string or result is not a dict
        
        Example:
            >>> validator = AnswerValidator()
            >>> result = {'conclusion': 'The answer is 42'}
            >>> validation = validator.validate("What is 6 * 7?", result)
            >>> print(validation.valid)
            True
        """
        # Input validation
        if not isinstance(query, str):
            raise TypeError(f"query must be a string, got {type(query).__name__}")
        if not isinstance(result, dict):
            raise TypeError(f"result must be a dict, got {type(result).__name__}")
        
        failures: List[ValidationFailureReason] = []
        suggestions: List[str] = []
        
        # Infer expected type if not provided
        if expected_type is None:
            expected_type = self._infer_expected_type(query)
        
        # Run type-specific validator if available
        if expected_type in self.validators:
            try:
                validator = self.validators[expected_type]
                type_validation = validator(query, result)
                if not type_validation.valid:
                    failures.extend(type_validation.failures)
                    suggestions.extend(type_validation.suggestions)
            except Exception as e:
                logger.warning(
                    f"[AnswerValidator] Type-specific validator failed: {e}"
                )
                # Continue with general validation
        
        # Always check for obvious nonsense
        try:
            if self._is_obviously_nonsensical(query, result):
                if ValidationFailureReason.NONSENSICAL_OUTPUT not in failures:
                    failures.append(ValidationFailureReason.NONSENSICAL_OUTPUT)
                    suggestions.append(
                        "Result appears unrelated to query - consider using different engine"
                    )
        except Exception as e:
            logger.warning(f"[AnswerValidator] Nonsense check failed: {e}")
        
        # Determine overall validity
        valid = len(failures) == 0
        confidence = 1.0 if valid else 0.0
        
        explanation = self._format_explanation(failures, suggestions)
        
        return ValidationResult(
            valid=valid,
            confidence=confidence,
            failures=failures,
            explanation=explanation,
            suggestions=suggestions
        )
    
    def _infer_expected_type(self, query: str) -> str:
        """
        Infer the expected answer type from the query content.
        
        Uses keyword matching and symbol detection to determine what type
        of answer the query expects.
        
        Args:
            query: The query string to analyze
            
        Returns:
            String identifier for the expected answer type:
            'fol', 'sat', 'ethical', 'yes_no', 'mathematical', 'proof',
            'bayesian', or 'general'
        """
        query_lower = query.lower()
        
        # Check for logic symbols in original query (case-sensitive)
        has_logic_symbols = any(sym in query for sym in _LOGIC_SYMBOLS)
        
        # FOL formalization (check before SAT since FOL is more specific)
        if any(kw in query_lower for kw in self._FOL_KEYWORDS):
            return 'fol'
        
        # SAT/satisfiability
        if any(kw in query_lower for kw in self._SAT_KEYWORDS) or has_logic_symbols:
            return 'sat'
        
        # Ethical/permissibility questions
        if any(kw in query_lower for kw in self._ETHICAL_KEYWORDS):
            return 'ethical'
        
        # YES/NO questions (check after ethical since ethical can also be yes/no)
        if any(kw in query_lower for kw in self._YES_NO_KEYWORDS):
            return 'yes_no'
        
        # Mathematical computation
        if any(kw in query_lower for kw in self._MATH_KEYWORDS):
            return 'mathematical'
        
        # Proof verification
        if any(kw in query_lower for kw in self._PROOF_KEYWORDS):
            return 'proof'
        
        # Bayesian inference
        if any(kw in query_lower for kw in self._BAYESIAN_KEYWORDS):
            return 'bayesian'
        
        return 'general'
    
    def _validate_fol_answer(self, query: str, result: Dict[str, Any]) -> ValidationResult:
        """
        Validate First-Order Logic formalization answer.
        
        Checks that the answer contains appropriate FOL symbols and
        is not a mathematical expression.
        
        Args:
            query: Original query
            result: Result dictionary with 'conclusion' key
            
        Returns:
            ValidationResult for FOL-specific checks
        """
        failures: List[ValidationFailureReason] = []
        suggestions: List[str] = []
        
        result_text = str(result.get('conclusion', ''))
        result_upper = result_text.upper()
        
        # Check for FOL symbols/keywords
        fol_indicators = [
            '∀', '∃', '∧', '∨', '→', '¬', '⊢', '⊨',
            'FORALL', 'EXISTS', 'AND', 'OR', 'NOT', 'IMPLIES'
        ]
        has_fol = any(indicator in result_text or indicator in result_upper 
                      for indicator in fol_indicators)
        
        if not has_fol:
            failures.append(ValidationFailureReason.MISSING_REQUIRED_ELEMENTS)
            suggestions.append(
                "FOL answer should contain quantifiers (∀, ∃) or logical operators (∧, ∨, →, ¬)"
            )
        
        # Check it's not a mathematical expression instead
        if _MATH_EXPRESSION_PATTERN.search(result_text):
            failures.append(ValidationFailureReason.WRONG_DOMAIN)
            suggestions.append(
                "Result appears to be a mathematical expression, not FOL - wrong engine used"
            )
        
        return ValidationResult(
            valid=len(failures) == 0,
            confidence=1.0 if len(failures) == 0 else 0.0,
            failures=failures,
            explanation="",
            suggestions=suggestions
        )
    
    def _validate_sat_answer(self, query: str, result: Dict[str, Any]) -> ValidationResult:
        """
        Validate SAT/satisfiability answer.
        
        Checks that the answer indicates satisfiability status and
        is not from an unrelated domain.
        
        Args:
            query: Original query
            result: Result dictionary with 'conclusion' key
            
        Returns:
            ValidationResult for SAT-specific checks
        """
        failures: List[ValidationFailureReason] = []
        suggestions: List[str] = []
        
        result_text = str(result.get('conclusion', '')).lower()
        
        # Must indicate satisfiability status
        sat_keywords = [
            'satisfiable', 'unsatisfiable', 'yes', 'no', 
            'model', 'contradiction', 'true', 'false',
            'consistent', 'inconsistent'
        ]
        has_answer = any(kw in result_text for kw in sat_keywords)
        
        if not has_answer:
            failures.append(ValidationFailureReason.NO_ANSWER_PROVIDED)
            suggestions.append(
                "SAT answer must state whether satisfiable/unsatisfiable or provide model/contradiction"
            )
        
        # Check for wrong domain indicators
        wrong_domain_patterns = ['exp(', 'derivative', 'integral', 'sin(', 'cos(', 'tan(', 'log(']
        if any(pattern in result_text for pattern in wrong_domain_patterns):
            failures.append(ValidationFailureReason.WRONG_DOMAIN)
            suggestions.append(
                "Result contains calculus/analysis terms - wrong domain for SAT query"
            )
        
        return ValidationResult(
            valid=len(failures) == 0,
            confidence=1.0 if len(failures) == 0 else 0.0,
            failures=failures,
            explanation="",
            suggestions=suggestions
        )
    
    def _validate_yes_no_answer(self, query: str, result: Dict[str, Any]) -> ValidationResult:
        """
        Validate YES/NO answer.
        
        Checks that the answer contains a clear affirmative or negative response.
        
        Args:
            query: Original query
            result: Result dictionary with 'conclusion' key
            
        Returns:
            ValidationResult for YES/NO-specific checks
        """
        failures: List[ValidationFailureReason] = []
        suggestions: List[str] = []
        
        result_text = str(result.get('conclusion', '')).upper()
        
        # Must contain YES or NO or equivalent
        yes_no_indicators = [
            'YES', 'NO', 'TRUE', 'FALSE', 
            'PERMISSIBLE', 'IMPERMISSIBLE', 'FORBIDDEN',
            'VALID', 'INVALID', 
            'SATISFIABLE', 'UNSATISFIABLE',
            'CORRECT', 'INCORRECT'
        ]
        has_yes_no = any(indicator in result_text for indicator in yes_no_indicators)
        
        if not has_yes_no:
            failures.append(ValidationFailureReason.MISSING_REQUIRED_FORMAT)
            suggestions.append(
                "Question requires YES or NO answer (or equivalent: TRUE/FALSE, VALID/INVALID)"
            )
        
        return ValidationResult(
            valid=len(failures) == 0,
            confidence=1.0 if len(failures) == 0 else 0.0,
            failures=failures,
            explanation="",
            suggestions=suggestions
        )
    
    def _validate_mathematical_answer(self, query: str, result: Dict[str, Any]) -> ValidationResult:
        """
        Validate mathematical computation answer.
        
        Checks that the answer contains numbers or mathematical expressions.
        
        Args:
            query: Original query
            result: Result dictionary with 'conclusion' key
            
        Returns:
            ValidationResult for mathematical-specific checks
        """
        failures: List[ValidationFailureReason] = []
        suggestions: List[str] = []
        
        result_text = str(result.get('conclusion', ''))
        
        # Should contain numbers or mathematical expressions
        has_numbers = bool(_NUMERIC_PATTERN.search(result_text))
        has_math_symbols = any(sym in result_text for sym in ['=', '+', '-', '*', '/', '^', 'x', 'y'])
        
        if not (has_numbers or has_math_symbols):
            failures.append(ValidationFailureReason.MISSING_REQUIRED_ELEMENTS)
            suggestions.append(
                "Mathematical answer should contain numbers or mathematical expressions"
            )
        
        return ValidationResult(
            valid=len(failures) == 0,
            confidence=1.0 if len(failures) == 0 else 0.0,
            failures=failures,
            explanation="",
            suggestions=suggestions
        )
    
    def _validate_proof_answer(self, query: str, result: Dict[str, Any]) -> ValidationResult:
        """
        Validate proof verification answer.
        
        Checks that the answer indicates validity status or identifies flaws.
        
        Args:
            query: Original query
            result: Result dictionary with 'conclusion' key
            
        Returns:
            ValidationResult for proof-specific checks
        """
        failures: List[ValidationFailureReason] = []
        suggestions: List[str] = []
        
        result_text = str(result.get('conclusion', '')).lower()
        
        # Should indicate validity or identify flaws
        proof_keywords = [
            'valid', 'invalid', 'flaw', 'error', 'correct', 'incorrect',
            'sound', 'unsound', 'complete', 'incomplete', 'qed', 'proven'
        ]
        has_verification = any(kw in result_text for kw in proof_keywords)
        
        if not has_verification:
            failures.append(ValidationFailureReason.NO_ANSWER_PROVIDED)
            suggestions.append(
                "Proof verification should state valid/invalid or identify specific flaws"
            )
        
        return ValidationResult(
            valid=len(failures) == 0,
            confidence=1.0 if len(failures) == 0 else 0.0,
            failures=failures,
            explanation="",
            suggestions=suggestions
        )
    
    def _validate_bayesian_answer(self, query: str, result: Dict[str, Any]) -> ValidationResult:
        """
        Validate Bayesian inference answer.
        
        Checks that the answer contains a probability value.
        
        Args:
            query: Original query
            result: Result dictionary with 'conclusion' key
            
        Returns:
            ValidationResult for Bayesian-specific checks
        """
        failures: List[ValidationFailureReason] = []
        suggestions: List[str] = []
        
        result_text = str(result.get('conclusion', ''))
        
        # Should contain probability value (decimal or percentage)
        has_probability = bool(_PROBABILITY_PATTERN.search(result_text))
        
        # Also check for probability keywords
        prob_keywords = ['probability', 'p(', 'likely', 'chance', 'odds']
        has_prob_context = any(kw in result_text.lower() for kw in prob_keywords)
        
        if not (has_probability or has_prob_context):
            failures.append(ValidationFailureReason.MISSING_REQUIRED_ELEMENTS)
            suggestions.append(
                "Bayesian answer should provide probability value (e.g., 0.75 or 75%)"
            )
        
        return ValidationResult(
            valid=len(failures) == 0,
            confidence=1.0 if len(failures) == 0 else 0.0,
            failures=failures,
            explanation="",
            suggestions=suggestions
        )
    
    def _validate_ethical_answer(self, query: str, result: Dict[str, Any]) -> ValidationResult:
        """
        Validate ethical/deontic reasoning answer.
        
        Checks that the answer provides a moral judgment and is not
        from an unrelated domain.
        
        Args:
            query: Original query
            result: Result dictionary with 'conclusion' key
            
        Returns:
            ValidationResult for ethical-specific checks
        """
        failures: List[ValidationFailureReason] = []
        suggestions: List[str] = []
        
        result_text = str(result.get('conclusion', '')).lower()
        
        # Should contain ethical judgment
        ethical_keywords = [
            'permissible', 'impermissible', 'forbidden', 'obligatory',
            'right', 'wrong', 'moral', 'immoral', 'ethical', 'unethical',
            'should', 'must', 'duty', 'obligation', 'virtue', 'vice'
        ]
        has_judgment = any(kw in result_text for kw in ethical_keywords)
        
        if not has_judgment:
            failures.append(ValidationFailureReason.NO_ANSWER_PROVIDED)
            suggestions.append(
                "Ethical answer should provide moral judgment (permissible/impermissible, right/wrong, etc.)"
            )
        
        # Check it's not a mathematical expression
        if _MATH_EXPRESSION_PATTERN.search(result_text) or 'derivative' in result_text:
            failures.append(ValidationFailureReason.WRONG_DOMAIN)
            suggestions.append(
                "Result contains mathematical content - wrong domain for ethical query"
            )
        
        return ValidationResult(
            valid=len(failures) == 0,
            confidence=1.0 if len(failures) == 0 else 0.0,
            failures=failures,
            explanation="",
            suggestions=suggestions
        )
    
    def _is_obviously_nonsensical(self, query: str, result: Dict[str, Any]) -> bool:
        """
        Check for obviously nonsensical outputs that indicate domain mismatch.
        
        This method catches obvious cases where the answer type doesn't match
        the query type at all, regardless of specific content validation.
        
        Args:
            query: Original query text
            result: Result dictionary with 'conclusion' key
            
        Returns:
            True if the output is obviously nonsensical, False otherwise
            
        Examples:
            - Logic query with calculus answer -> True
            - Ethical query with mathematical formula -> True  
            - Known bug output "3x**2 + 2x" on non-math query -> True
        """
        query_lower = query.lower()
        result_text = str(result.get('conclusion', ''))
        result_lower = result_text.lower()
        
        # Define domain indicators
        logic_indicators = [
            'logic', 'sat', 'satisfiable', 'fol', 'propositional', 
            'predicate', 'formalize', 'quantifier'
        ]
        
        calculus_indicators = [
            'derivative', 'exp(', 'integral', 'sin(', 'cos(', 'tan(', 
            'log(', 'diff(', '**2', '**3'
        ]
        
        ethical_indicators = [
            'permissible', 'moral', 'ethical', 'trolley', 'dilemma',
            'should', 'ought', 'forbidden'
        ]
        
        # Check for logic symbols in original query
        has_logic_symbols = any(sym in query for sym in _LOGIC_SYMBOLS)
        
        # Detect query type
        is_logic_query = (
            any(ind in query_lower for ind in logic_indicators) or 
            has_logic_symbols
        )
        is_ethical_query = any(ind in query_lower for ind in ethical_indicators)
        
        # Detect answer type
        has_calculus_answer = any(ind in result_lower for ind in calculus_indicators)
        
        # Logic query + calculus answer = nonsensical
        if is_logic_query and has_calculus_answer:
            logger.warning(
                f"[AnswerValidator] Nonsensical output: Logic query got calculus answer. "
                f"Query: {query[:60]}... Answer: {result_text[:60]}..."
            )
            return True
        
        # Ethical query + calculus answer = nonsensical
        if is_ethical_query and has_calculus_answer:
            logger.warning(
                f"[AnswerValidator] Nonsensical output: Ethical query got calculus answer. "
                f"Query: {query[:60]}..."
            )
            return True
        
        # Check for known bug outputs
        for bug_output in _KNOWN_BUG_OUTPUTS:
            if bug_output in result_text:
                # Only flag if this isn't actually a derivative query
                if 'derivative' not in query_lower and 'differentiate' not in query_lower:
                    logger.warning(
                        f"[AnswerValidator] Known bug output '{bug_output}' detected "
                        f"in non-derivative query: {query[:60]}..."
                    )
                    return True
        
        return False
    
    def _format_explanation(
        self, 
        failures: List[ValidationFailureReason],
        suggestions: List[str]
    ) -> str:
        """
        Format a human-readable explanation of validation results.
        
        Args:
            failures: List of failure reasons
            suggestions: List of corrective action suggestions
            
        Returns:
            Formatted explanation string
        """
        if not failures:
            return "Answer validation passed"
        
        lines = ["Answer validation failed:"]
        for failure in failures:
            lines.append(f"  - {failure.value}")
        
        if suggestions:
            lines.append("")
            lines.append("Suggestions:")
            for suggestion in suggestions:
                lines.append(f"  - {suggestion}")
        
        return "\n".join(lines)


# =============================================================================
# Convenience Functions
# =============================================================================

def validate_reasoning_result(
    query: str,
    result: Dict[str, Any],
    expected_type: Optional[str] = None
) -> ValidationResult:
    """
    Convenience function to validate reasoning results.
    
    Creates an AnswerValidator instance and validates the result.
    For repeated validations, consider creating a single AnswerValidator
    instance and reusing it.
    
    Args:
        query: Original query string that was sent to the reasoning engine
        result: Reasoning result dictionary, must contain 'conclusion' key
        expected_type: Optional expected answer type. If None, will be 
            inferred from query content. Valid types:
            'fol', 'sat', 'yes_no', 'mathematical', 'proof', 'bayesian', 'ethical'
        
    Returns:
        ValidationResult indicating whether the answer is valid for the query
        
    Raises:
        TypeError: If query is not a string or result is not a dict
        
    Example:
        >>> result = engine.reason(query)
        >>> validation = validate_reasoning_result(query, result)
        >>> 
        >>> if not validation.valid:
        ...     logger.warning(f"Invalid result: {validation.explanation}")
        ...     # Try different engine or return error
        
    See Also:
        AnswerValidator.validate: The underlying validation method
    """
    validator = AnswerValidator()
    return validator.validate(query, result, expected_type)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Classes
    'AnswerValidator',
    'ValidationResult', 
    'ValidationFailureReason',
    # Functions
    'validate_reasoning_result',
]

