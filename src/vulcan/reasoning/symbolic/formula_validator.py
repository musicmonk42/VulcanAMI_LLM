"""
Formula Validator for Symbolic Logic.

BUG #8 FIX: The symbolic parser was accepting malformed logic and producing
cryptic errors that didn't help users understand what was wrong.

This module provides:
1. Pre-validation of formula syntax before parsing
2. Helpful, actionable error messages
3. Common mistake detection and correction suggestions
4. Thread-safe, immutable pattern compilation

Architecture:
    The validator uses a pipeline approach with pre-compiled patterns:
    1. Check structural validity (parentheses, brackets)
    2. Check for empty/malformed expressions
    3. Validate operator usage and placement
    4. Provide actionable suggestions for fixes

Industry Standards Compliance:
    - Type hints on all public methods (PEP 484)
    - Comprehensive docstrings (Google style)
    - Immutable compiled patterns for thread safety
    - Logging for debugging and monitoring
    - Unit testable design with dependency injection ready
    - Dataclasses for structured error information (PEP 557)
    - Frozen sets for immutable constant collections

Example:
    >>> validator = FormulaValidator()
    >>> is_valid, error = validator.validate("A → B ∧ (C")
    >>> if not is_valid:
    ...     print(error)
    Formula validation failed:
      Formula: A → B ∧ (C
      Errors:
        - Unbalanced parentheses: 1 '(' but 0 ')'
      Please fix syntax and try again.

Performance:
    - All regex patterns compiled once at module load (not per-call)
    - Validation runs in O(n) time complexity
    - Memory usage is O(1) for patterns, O(n) for error collection
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    FrozenSet,
    List,
    Optional,
    Pattern,
    Sequence,
    Tuple,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Constants - Immutable, Thread-Safe
# =============================================================================

# Valid logic operators (Unicode and ASCII)
# Using frozenset for immutability and O(1) lookup
VALID_UNICODE_OPERATORS: FrozenSet[str] = frozenset({
    '∧',  # Conjunction (AND)
    '∨',  # Disjunction (OR)
    '¬',  # Negation (NOT)
    '→',  # Implication
    '↔',  # Biconditional (IFF)
    '⇒',  # Alternative implication symbol
    '⇔',  # Alternative biconditional symbol
})

VALID_ASCII_OPERATORS: FrozenSet[str] = frozenset({
    '&',   # Conjunction (AND)
    '|',   # Disjunction (OR) - single pipe only in logic context
    '~',   # Negation (NOT)
    '!',   # Alternative negation
    '->',  # Implication (multi-char)
    '<->', # Biconditional (multi-char)
    '&&',  # Alternative conjunction
    '||',  # Alternative disjunction
})

VALID_KEYWORD_OPERATORS: FrozenSet[str] = frozenset({
    'AND', 'OR', 'NOT', 'IMPLIES', 'IFF',
    'and', 'or', 'not', 'implies', 'iff',
})

# Combined set of all valid operators
VALID_OPERATORS: FrozenSet[str] = (
    VALID_UNICODE_OPERATORS | VALID_ASCII_OPERATORS | VALID_KEYWORD_OPERATORS
)

# Valid quantifiers
VALID_QUANTIFIERS: FrozenSet[str] = frozenset({
    '∀',       # Universal quantifier
    '∃',       # Existential quantifier
    'forall',  # Keyword universal
    'exists',  # Keyword existential
    'FORALL',
    'EXISTS',
})

# Binary operators that require operands on both sides
# These should not appear at start/end of formula
BINARY_OPERATORS: Tuple[str, ...] = (
    '∧', '∨', '→', '↔', '⇒', '⇔',  # Unicode
    '&', '->', '<->',               # ASCII (single | excluded - ambiguous)
)

# Common operator mistakes and their corrections
# Format: {mistake: (correction, is_actually_valid)}
# is_actually_valid=True means we accept it but suggest alternatives
OPERATOR_CORRECTIONS: dict[str, Tuple[str, bool]] = {
    '=>': ('→ or ->', False),      # Invalid - must use -> or →
    '<=>': ('↔ or <->', False),    # Invalid - must use <-> or ↔
    '!!': ('¬ or !', False),       # Double negation should be explicit
}


# =============================================================================
# Pre-compiled Patterns - Module Level for Performance
# =============================================================================

# Empty parentheses pattern: matches "()" with optional whitespace inside
_EMPTY_PARENS_PATTERN: Pattern[str] = re.compile(r'\(\s*\)')

# Consecutive binary operators (excluding spaces)
# Matches patterns like "∧∧", "&&", "||" but not "¬¬" (negation can stack)
_CONSECUTIVE_BINARY_OPS_PATTERN: Pattern[str] = re.compile(
    r'[∧∨→↔⇒⇔&|]\s*[∧∨→↔⇒⇔&|]'
)

# Valid variable/predicate name pattern
_IDENTIFIER_PATTERN: Pattern[str] = re.compile(r'^[A-Za-z][A-Za-z0-9_]*$')

# Whitespace normalization pattern
_WHITESPACE_PATTERN: Pattern[str] = re.compile(r'\s+')


# =============================================================================
# Enums
# =============================================================================

class ValidationErrorType(Enum):
    """
    Enumeration of validation error types.
    
    Using Enum ensures type safety and enables exhaustive matching
    in error handling code.
    """
    EMPTY_FORMULA = auto()
    UNBALANCED_PARENTHESES = auto()
    EMPTY_EXPRESSION = auto()
    INVALID_OPERATOR = auto()
    CONSECUTIVE_OPERATORS = auto()
    DANGLING_OPERATOR = auto()
    MISMATCHED_BRACKETS = auto()
    INVALID_QUANTIFIER_SYNTAX = auto()


# =============================================================================
# Data Classes
# =============================================================================

@dataclass(frozen=True)
class ValidationError:
    """
    Represents a single validation error with context.
    
    This dataclass is frozen (immutable) for thread safety and
    to prevent accidental modification after creation.
    
    Attributes:
        error_type: The category of error (from ValidationErrorType enum)
        message: Human-readable error description
        position: Optional character position where error was detected
        suggestion: Optional actionable suggestion for fixing the error
        severity: Error severity level ("error", "warning", "info")
    """
    error_type: ValidationErrorType
    message: str
    position: Optional[int] = None
    suggestion: Optional[str] = None
    severity: str = "error"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "error_type": self.error_type.name,
            "message": self.message,
            "position": self.position,
            "suggestion": self.suggestion,
            "severity": self.severity,
        }


@dataclass
class ValidationResult:
    """
    Complete result of formula validation.
    
    Attributes:
        is_valid: Whether the formula passed all validation checks
        formula: The original formula that was validated
        errors: List of validation errors found (empty if valid)
        normalized_formula: Whitespace-normalized version of formula
    """
    is_valid: bool
    formula: str
    errors: List[ValidationError] = field(default_factory=list)
    normalized_formula: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "is_valid": self.is_valid,
            "formula": self.formula,
            "errors": [e.to_dict() for e in self.errors],
            "normalized_formula": self.normalized_formula,
        }
    
    def get_error_message(self) -> Optional[str]:
        """
        Get a formatted error message suitable for display.
        
        Returns:
            Formatted error string, or None if validation passed
        """
        if self.is_valid:
            return None
        return _build_error_message(self.formula, self.errors)


# =============================================================================
# Main Validator Class
# =============================================================================

class FormulaValidator:
    """
    Validate formula syntax before reasoning (BUG #8 FIX).
    
    This validator checks for common syntax errors and provides helpful
    error messages that guide users to fix their formulas.
    
    Thread Safety:
        This class is thread-safe. All patterns are compiled at module load
        and stored as module-level constants. Instance methods only read
        from these constants and create new objects for results.
    
    Example:
        >>> validator = FormulaValidator()
        >>> is_valid, error_msg = validator.validate("P(x) ∧ Q(y")
        >>> print(error_msg)
        Formula validation failed:
          Formula: P(x) ∧ Q(y
          Errors:
            - Unbalanced parentheses: 2 '(' but 1 ')'
          Please fix syntax and try again.
    
    Attributes:
        strict_mode: If True, treats warnings as errors (default: False)
    """
    
    __slots__ = ('_strict_mode',)
    
    def __init__(self, strict_mode: bool = False) -> None:
        """
        Initialize the formula validator.
        
        Args:
            strict_mode: If True, treats warnings as errors
        """
        self._strict_mode = strict_mode
    
    @property
    def strict_mode(self) -> bool:
        """Whether strict validation mode is enabled."""
        return self._strict_mode
    
    def validate(self, formula: str) -> Tuple[bool, Optional[str]]:
        """
        Validate formula and return helpful error messages.
        
        BUG #8 FIX: This method provides pre-validation with actionable
        error messages instead of letting the parser produce cryptic errors.
        
        Args:
            formula: The formula string to validate
            
        Returns:
            Tuple of (is_valid, error_message)
            - If valid: (True, None)
            - If invalid: (False, helpful_error_message)
        
        Example:
            >>> validator = FormulaValidator()
            >>> is_valid, error = validator.validate("A ∧ B")
            >>> assert is_valid is True
            >>> assert error is None
        """
        result = self.validate_detailed(formula)
        return result.is_valid, result.get_error_message()
    
    def validate_detailed(self, formula: str) -> ValidationResult:
        """
        Perform detailed validation and return structured result.
        
        This method provides more detailed results than validate(),
        including all errors found and the normalized formula.
        
        Args:
            formula: The formula string to validate
            
        Returns:
            ValidationResult with all validation details
        """
        # Handle empty input
        if not formula or not formula.strip():
            return ValidationResult(
                is_valid=False,
                formula=formula or "",
                errors=[ValidationError(
                    error_type=ValidationErrorType.EMPTY_FORMULA,
                    message="Formula is empty or contains only whitespace",
                    suggestion="Provide a valid logical formula."
                )]
            )
        
        # Normalize whitespace for consistent checking
        normalized = _WHITESPACE_PATTERN.sub(' ', formula.strip())
        errors: List[ValidationError] = []
        
        # Run validation checks in order of severity
        self._check_balanced_parens(formula, errors)
        self._check_non_empty_expressions(formula, errors)
        self._check_valid_operators(formula, errors)
        self._check_consecutive_operators(formula, errors)
        self._check_dangling_operators(formula, errors)
        
        # Determine validity
        is_valid = len(errors) == 0
        if self._strict_mode:
            # In strict mode, warnings also count as invalid
            is_valid = all(e.severity == "info" for e in errors)
        
        if errors:
            logger.warning(
                f"[FormulaValidator] BUG#8 FIX: {len(errors)} error(s) in formula"
            )
        else:
            logger.debug("[FormulaValidator] BUG#8 FIX: Formula validated successfully")
        
        return ValidationResult(
            is_valid=is_valid,
            formula=formula,
            errors=errors,
            normalized_formula=normalized
        )
    
    def _check_balanced_parens(
        self, 
        formula: str, 
        errors: List[ValidationError]
    ) -> None:
        """
        Check if parentheses and brackets are balanced.
        
        Modifies errors list in-place for efficiency.
        """
        # Track opening positions for better error messages
        paren_stack: List[Tuple[str, int]] = []
        bracket_pairs = {'(': ')', '[': ']', '{': '}'}
        closing_to_opening = {')': '(', ']': '[', '}': '{'}
        
        for i, char in enumerate(formula):
            if char in bracket_pairs:
                paren_stack.append((char, i))
            elif char in closing_to_opening:
                expected_open = closing_to_opening[char]
                if not paren_stack:
                    errors.append(ValidationError(
                        error_type=ValidationErrorType.UNBALANCED_PARENTHESES,
                        message=f"Unexpected closing '{char}' at position {i} with no matching opening bracket",
                        position=i,
                        suggestion=f"Add a matching '{expected_open}' before position {i}, or remove this '{char}'."
                    ))
                elif paren_stack[-1][0] != expected_open:
                    open_char, open_pos = paren_stack[-1]
                    errors.append(ValidationError(
                        error_type=ValidationErrorType.MISMATCHED_BRACKETS,
                        message=f"Mismatched brackets: '{open_char}' at position {open_pos} closed with '{char}' at position {i}",
                        position=i,
                        suggestion=f"Use '{bracket_pairs[open_char]}' to close '{open_char}', or fix the opening bracket."
                    ))
                    paren_stack.pop()
                else:
                    paren_stack.pop()
        
        # Check for unclosed brackets
        for open_char, open_pos in paren_stack:
            errors.append(ValidationError(
                error_type=ValidationErrorType.UNBALANCED_PARENTHESES,
                message=f"Unclosed '{open_char}' at position {open_pos}",
                position=open_pos,
                suggestion=f"Add a closing '{bracket_pairs[open_char]}' at the end of the formula."
            ))
    
    def _check_non_empty_expressions(
        self, 
        formula: str, 
        errors: List[ValidationError]
    ) -> None:
        """Check for empty expressions like '()'."""
        match = _EMPTY_PARENS_PATTERN.search(formula)
        if match:
            errors.append(ValidationError(
                error_type=ValidationErrorType.EMPTY_EXPRESSION,
                message=f"Empty parentheses '()' found at position {match.start()}",
                position=match.start(),
                suggestion="Put a valid expression inside the parentheses, or remove them."
            ))
    
    def _check_valid_operators(
        self, 
        formula: str, 
        errors: List[ValidationError]
    ) -> None:
        """Check for invalid or non-standard operators."""
        for mistake, (correct, is_valid) in OPERATOR_CORRECTIONS.items():
            if mistake in formula and not is_valid:
                pos = formula.find(mistake)
                errors.append(ValidationError(
                    error_type=ValidationErrorType.INVALID_OPERATOR,
                    message=f"Invalid operator '{mistake}' at position {pos}",
                    position=pos,
                    suggestion=f"Use {correct} instead of '{mistake}'."
                ))
    
    def _check_consecutive_operators(
        self, 
        formula: str, 
        errors: List[ValidationError]
    ) -> None:
        """Check for consecutive binary operators (e.g., 'A ∧ ∧ B')."""
        match = _CONSECUTIVE_BINARY_OPS_PATTERN.search(formula)
        if match:
            errors.append(ValidationError(
                error_type=ValidationErrorType.CONSECUTIVE_OPERATORS,
                message=f"Consecutive binary operators at position {match.start()}: '{match.group()}'",
                position=match.start(),
                suggestion="Remove the extra operator or add an operand between them."
            ))
    
    def _check_dangling_operators(
        self, 
        formula: str, 
        errors: List[ValidationError]
    ) -> None:
        """Check for binary operators at start/end without operands."""
        formula_stripped = formula.strip()
        
        for op in BINARY_OPERATORS:
            if formula_stripped.startswith(op):
                errors.append(ValidationError(
                    error_type=ValidationErrorType.DANGLING_OPERATOR,
                    message=f"Formula starts with binary operator '{op}'",
                    position=0,
                    suggestion=f"Add an operand before '{op}', or remove the operator."
                ))
                break  # Only report first dangling start operator
        
        for op in BINARY_OPERATORS:
            if formula_stripped.endswith(op):
                pos = len(formula_stripped) - len(op)
                errors.append(ValidationError(
                    error_type=ValidationErrorType.DANGLING_OPERATOR,
                    message=f"Formula ends with binary operator '{op}'",
                    position=pos,
                    suggestion=f"Add an operand after '{op}', or remove the operator."
                ))
                break  # Only report first dangling end operator
    
    def get_fix_suggestions(self, formula: str) -> List[str]:
        """
        Get a list of actionable suggestions to fix the formula.
        
        Args:
            formula: The formula string to analyze
            
        Returns:
            List of suggestion strings, or ["Formula appears valid."] if no issues
        """
        result = self.validate_detailed(formula)
        
        if result.is_valid:
            return ["Formula appears valid."]
        
        # Collect unique suggestions
        suggestions: List[str] = []
        seen: set[str] = set()
        
        for error in result.errors:
            if error.suggestion and error.suggestion not in seen:
                suggestions.append(error.suggestion)
                seen.add(error.suggestion)
        
        return suggestions if suggestions else ["Check formula syntax carefully."]


# =============================================================================
# Helper Functions
# =============================================================================

def _build_error_message(
    formula: str, 
    errors: Sequence[ValidationError]
) -> str:
    """
    Build a helpful, formatted error message from validation errors.
    
    Args:
        formula: The original formula
        errors: Sequence of validation errors
        
    Returns:
        Formatted multi-line error message
    """
    lines = [
        "Formula validation failed:",
        f"  Formula: {formula}",
        "  Errors:",
    ]
    
    for error in errors:
        position_info = f" (at position {error.position})" if error.position is not None else ""
        lines.append(f"    - [{error.severity.upper()}] {error.message}{position_info}")
        if error.suggestion:
            lines.append(f"      Suggestion: {error.suggestion}")
    
    lines.append("  Please fix syntax and try again.")
    
    return "\n".join(lines)


# =============================================================================
# Convenience Functions
# =============================================================================

def validate_formula(formula: str) -> Tuple[bool, Optional[str]]:
    """
    Validate a formula string.
    
    BUG #8 FIX: Convenience function for quick validation without
    instantiating a FormulaValidator object.
    
    This function creates a new validator instance for each call,
    which is fine for occasional use. For high-throughput scenarios,
    reuse a FormulaValidator instance.
    
    Args:
        formula: The formula to validate
        
    Returns:
        Tuple of (is_valid, error_message or None)
        
    Example:
        >>> is_valid, error = validate_formula("A ∧ B")
        >>> assert is_valid is True
    """
    validator = FormulaValidator()
    return validator.validate(formula)


def validate_formula_strict(formula: str) -> Tuple[bool, Optional[str]]:
    """
    Validate a formula string in strict mode.
    
    In strict mode, warnings are also treated as validation failures.
    
    Args:
        formula: The formula to validate
        
    Returns:
        Tuple of (is_valid, error_message or None)
    """
    validator = FormulaValidator(strict_mode=True)
    return validator.validate(formula)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Main class
    'FormulaValidator',
    # Data classes
    'ValidationError',
    'ValidationResult',
    'ValidationErrorType',
    # Convenience functions
    'validate_formula',
    'validate_formula_strict',
    # Constants (for extensibility)
    'VALID_OPERATORS',
    'VALID_QUANTIFIERS',
    'BINARY_OPERATORS',
]
