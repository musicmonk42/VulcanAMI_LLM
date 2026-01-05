"""
Formula Validator for Symbolic Logic.

BUG #8 FIX: The symbolic parser was accepting malformed logic and producing
cryptic errors that didn't help users understand what was wrong.

This module provides:
1. Pre-validation of formula syntax before parsing
2. Helpful, actionable error messages
3. Common mistake detection and correction suggestions

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
"""

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Valid logic operators (Unicode and ASCII)
VALID_OPERATORS = frozenset([
    '∧', '∨', '¬', '→', '↔', '⇒', '⇔',  # Unicode
    '&', '|', '~', '!', '->', '<->', '&&', '||',  # ASCII
    'AND', 'OR', 'NOT', 'IMPLIES', 'IFF',  # Keywords
    'and', 'or', 'not', 'implies', 'iff',  # Lowercase keywords
])

# Valid quantifiers
VALID_QUANTIFIERS = frozenset([
    '∀', '∃',  # Unicode
    'forall', 'exists', 'FORALL', 'EXISTS',  # Keywords
])


@dataclass
class ValidationError:
    """Represents a single validation error."""
    error_type: str
    message: str
    position: Optional[int] = None
    suggestion: Optional[str] = None


class FormulaValidator:
    """
    Validate formula syntax before reasoning (BUG #8 FIX).
    
    This validator checks for common syntax errors and provides helpful
    error messages that guide users to fix their formulas.
    
    Example:
        >>> validator = FormulaValidator()
        >>> is_valid, error_msg = validator.validate("P(x) ∧ Q(y")
        >>> print(error_msg)
        Formula validation failed:
          Formula: P(x) ∧ Q(y
          Errors:
            - Unbalanced parentheses: 2 '(' but 1 ')'
          Please fix syntax and try again.
    """
    
    def __init__(self):
        """Initialize the formula validator."""
        # Compile patterns once for efficiency
        self._empty_parens_pattern = re.compile(r'\(\s*\)')
        self._double_op_pattern = re.compile(r'[∧∨→↔&|]{2,}')
        self._var_pattern = re.compile(r'^[A-Za-z][A-Za-z0-9_]*$')
    
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
        """
        if not formula or not formula.strip():
            return False, "Formula is empty. Please provide a valid logical formula."
        
        formula = formula.strip()
        errors: List[ValidationError] = []
        
        # Check 1: Balanced parentheses
        paren_error = self._check_balanced_parens(formula)
        if paren_error:
            errors.append(paren_error)
        
        # Check 2: Empty expressions
        empty_error = self._check_non_empty_expressions(formula)
        if empty_error:
            errors.append(empty_error)
        
        # Check 3: Valid operators
        op_error = self._check_valid_operators(formula)
        if op_error:
            errors.append(op_error)
        
        # Check 4: Consecutive operators
        consec_error = self._check_consecutive_operators(formula)
        if consec_error:
            errors.append(consec_error)
        
        # Check 5: Dangling operators
        dangle_error = self._check_dangling_operators(formula)
        if dangle_error:
            errors.append(dangle_error)
        
        # Build error message if any errors found
        if errors:
            error_msg = self._build_error_message(formula, errors)
            logger.warning(f"[FormulaValidator] BUG#8 FIX: {len(errors)} error(s) in formula")
            return False, error_msg
        
        logger.debug(f"[FormulaValidator] BUG#8 FIX: Formula validated successfully")
        return True, None
    
    def _check_balanced_parens(self, formula: str) -> Optional[ValidationError]:
        """Check if parentheses are balanced."""
        open_count = 0
        close_count = 0
        unmatched_pos = None
        
        for i, char in enumerate(formula):
            if char == '(':
                open_count += 1
            elif char == ')':
                close_count += 1
                if close_count > open_count:
                    unmatched_pos = i
        
        if open_count != close_count:
            return ValidationError(
                error_type="unbalanced_parentheses",
                message=f"Unbalanced parentheses: {open_count} '(' but {close_count} ')'",
                position=unmatched_pos,
                suggestion="Add or remove parentheses to balance them."
            )
        
        return None
    
    def _check_non_empty_expressions(self, formula: str) -> Optional[ValidationError]:
        """Check for empty expressions like '()'."""
        if self._empty_parens_pattern.search(formula):
            return ValidationError(
                error_type="empty_expression",
                message="Empty parentheses '()' found",
                suggestion="Put a valid expression inside the parentheses."
            )
        
        if not formula.strip():
            return ValidationError(
                error_type="empty_formula",
                message="Formula is empty or contains only whitespace",
                suggestion="Provide a valid logical formula."
            )
        
        return None
    
    def _check_valid_operators(self, formula: str) -> Optional[ValidationError]:
        """Check if all operators are valid."""
        # Find potential operators (symbols that aren't variables or parens)
        # This is a basic check - more sophisticated parsing would be better
        
        # Check for obviously invalid operators
        invalid_ops = []
        
        # Common mistakes
        common_mistakes = {
            '=>': '→ or ->',
            '<=>': '↔ or <->',
            '&&': '∧ or &',
            '||': '∨ or |',
            '!!': '¬ or !',
        }
        
        for mistake, correct in common_mistakes.items():
            if mistake in formula and mistake not in ['&&', '||']:  # && and || are valid
                invalid_ops.append((mistake, correct))
        
        if invalid_ops:
            mistakes_str = ", ".join(f"'{m}' (use {c})" for m, c in invalid_ops)
            return ValidationError(
                error_type="invalid_operator",
                message=f"Invalid or unknown operators: {mistakes_str}",
                suggestion="Use standard logic operators: ∧ ∨ ¬ → ↔ or ASCII equivalents & | ~ -> <->"
            )
        
        return None
    
    def _check_consecutive_operators(self, formula: str) -> Optional[ValidationError]:
        """Check for consecutive binary operators (e.g., 'A ∧ ∧ B')."""
        # Find consecutive operators (excluding negation which can stack)
        binary_ops = ['∧', '∨', '→', '↔', '&', '|']
        
        prev_was_binary_op = False
        for i, char in enumerate(formula):
            is_binary_op = char in binary_ops
            if is_binary_op and prev_was_binary_op:
                return ValidationError(
                    error_type="consecutive_operators",
                    message=f"Consecutive binary operators at position {i}",
                    position=i,
                    suggestion="Remove extra operator or add operand between them."
                )
            prev_was_binary_op = is_binary_op
        
        return None
    
    def _check_dangling_operators(self, formula: str) -> Optional[ValidationError]:
        """Check for operators at start/end without operands."""
        formula_stripped = formula.strip()
        
        # Binary operators that shouldn't start a formula
        binary_ops = ['∧', '∨', '→', '↔', '&', '|', '->', '<->']
        
        for op in binary_ops:
            if formula_stripped.startswith(op):
                return ValidationError(
                    error_type="dangling_operator",
                    message=f"Formula starts with binary operator '{op}'",
                    suggestion=f"Add an operand before '{op}' or remove it."
                )
            if formula_stripped.endswith(op):
                return ValidationError(
                    error_type="dangling_operator",
                    message=f"Formula ends with binary operator '{op}'",
                    suggestion=f"Add an operand after '{op}' or remove it."
                )
        
        return None
    
    def _build_error_message(
        self, 
        formula: str, 
        errors: List[ValidationError]
    ) -> str:
        """Build a helpful error message from validation errors."""
        lines = [
            "Formula validation failed:",
            f"  Formula: {formula}",
            "  Errors:",
        ]
        
        for error in errors:
            lines.append(f"    - {error.message}")
            if error.suggestion:
                lines.append(f"      Suggestion: {error.suggestion}")
        
        lines.append("  Please fix syntax and try again.")
        
        return "\n".join(lines)
    
    def get_fix_suggestions(self, formula: str) -> List[str]:
        """
        Get a list of suggestions to fix the formula.
        
        Args:
            formula: The formula string
            
        Returns:
            List of suggestion strings
        """
        is_valid, error_msg = self.validate(formula)
        
        if is_valid:
            return ["Formula appears valid."]
        
        suggestions = []
        
        # Count parens
        open_count = formula.count('(')
        close_count = formula.count(')')
        if open_count > close_count:
            suggestions.append(f"Add {open_count - close_count} closing parenthesis ')'")
        elif close_count > open_count:
            suggestions.append(f"Remove {close_count - open_count} extra closing parenthesis ')'")
        
        # Check for common ASCII alternatives
        if '=>' in formula:
            suggestions.append("Replace '=>' with '→' or '->'")
        
        if '()' in formula:
            suggestions.append("Fill in empty parentheses '()' with an expression")
        
        return suggestions if suggestions else ["Check formula syntax carefully."]


# Convenience function
def validate_formula(formula: str) -> Tuple[bool, Optional[str]]:
    """
    Validate a formula string.
    
    BUG #8 FIX: Convenience function for quick validation.
    
    Args:
        formula: The formula to validate
        
    Returns:
        Tuple of (is_valid, error_message or None)
    """
    validator = FormulaValidator()
    return validator.validate(formula)


__all__ = [
    'FormulaValidator',
    'ValidationError',
    'validate_formula',
]
