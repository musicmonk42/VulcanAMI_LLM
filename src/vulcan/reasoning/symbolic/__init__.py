"""
Symbolic Reasoning Submodule

A comprehensive symbolic reasoning system for first-order logic, constraint solving,
probabilistic reasoning, and advanced reasoning methods.

Components:
- Core: Terms, literals, clauses, unification, proof trees
- Parsing: Lexical analysis, syntactic parsing, and CNF conversion for FOL formulas
- Provers: Multiple theorem proving methods (tableau, resolution, etc.)
- Solvers: CSP, Bayesian networks
- Advanced: Fuzzy logic, temporal reasoning, meta-reasoning, learning
- Reasoner: Main interface integrating all components

Example Usage:
    >>> from vulcan.reasoning.symbolic import SymbolicReasoner
    >>> reasoner = SymbolicReasoner()
    >>>
    >>> # Add knowledge from strings
    >>> reasoner.add_rule("forall X (Human(X) -> Mortal(X))")
    >>> reasoner.add_fact("Human(socrates)")
    >>>
    >>> # Prove theorem
    >>> result = reasoner.query("Mortal(socrates)")
    >>> print(f"Proven: {result['proven']}, Confidence: {result['confidence']:.2f}")

Features:
- First-order logic theorem proving with multiple methods
- Constraint satisfaction with AC-3 and heuristics
- Bayesian network inference with variable elimination and learning
- Fuzzy logic inference with multiple membership functions
- Temporal reasoning with Allen's interval algebra
- Meta-level reasoning for strategy selection
- Proof learning and pattern extraction
"""

from typing import List

# --- Advanced Reasoning Systems ---
from .advanced import FuzzyLogicReasoner, MetaReasoner, ProofLearner, TemporalReasoner

# --- Core Data Structures ---
from .core import (
    Clause,
    Constant,
    Function,
    Literal,
    ProofNode,
    Term,
    Unifier,
    Variable,
)

# --- Parsing Pipeline Components ---
# All parsing logic is now in `parsing.py`
from .parsing import (
    ASTConverter,
    ASTNode,
    ClauseExtractor,
    CNFConverter,
    FormulaBuilder,
    FormulaParser,
    FormulaUtils,
    Lexer,
    NodeType,
    Parser,
    PrenexConverter,
    SkolemFunction,
    Skolemizer,
    Token,
    TokenType,
    VariableRenamer,
)

# BUG #5 FIX: Natural Language to Logic Converter
from .nl_converter import NaturalLanguageToLogicConverter, convert_nl_to_logic

# BUG #8 FIX: Formula Validator for pre-validation with helpful error messages
from .formula_validator import (
    FormulaValidator,
    ValidationError,
    ValidationResult,
    ValidationErrorType,
    validate_formula,
    validate_formula_strict,
)

# --- Theorem Provers ---
from .provers import (
    BaseProver,
    ConnectionMethodProver,
    ModelEliminationProver,
    NaturalDeductionProver,
    ParallelProver,
    ResolutionProver,
    TableauProver,
)

# --- Main Reasoner Interface ---
# This class integrates all the above components.
from .reasoner import HybridReasoner, ProbabilisticReasoner, SymbolicReasoner

# --- Other Solvers ---
from .solvers import BayesianNetworkReasoner, CSPSolver, VariableType

# Version
__version__ = "1.0.0"

# Package metadata
__author__ = "Vulcan Reasoning Team"
__description__ = "Comprehensive symbolic reasoning system"

# Export all public components for `from .symbolic import *`
__all__ = [
    # ========================================================================
    # MAIN INTERFACES
    # ========================================================================
    "SymbolicReasoner",
    "ProbabilisticReasoner",
    "HybridReasoner",
    # ========================================================================
    # CORE COMPONENTS
    # ========================================================================
    "Term",
    "Variable",
    "Constant",
    "Function",
    "Literal",
    "Clause",
    "Unifier",
    "ProofNode",
    # ========================================================================
    # PARSING & AST
    # ========================================================================
    "TokenType",
    "Token",
    "Lexer",
    "Parser",
    "ASTConverter",
    "NodeType",
    "ASTNode",
    "FormulaUtils",
    "VariableRenamer",
    "PrenexConverter",
    "SkolemFunction",
    "Skolemizer",
    "CNFConverter",
    "ClauseExtractor",
    "FormulaParser",
    "FormulaBuilder",
    # BUG #5 FIX: NL to Logic Converter
    "NaturalLanguageToLogicConverter",
    "convert_nl_to_logic",
    # BUG #8 FIX: Formula Validator
    "FormulaValidator",
    "ValidationError",
    "ValidationResult",
    "ValidationErrorType",
    "validate_formula",
    "validate_formula_strict",
    # ========================================================================
    # THEOREM PROVERS
    # ========================================================================
    "BaseProver",
    "TableauProver",
    "ResolutionProver",
    "ModelEliminationProver",
    "ConnectionMethodProver",
    "NaturalDeductionProver",
    "ParallelProver",
    # ========================================================================
    # SOLVERS
    # ========================================================================
    "CSPSolver",
    "BayesianNetworkReasoner",
    "VariableType",
    # ========================================================================
    # ADVANCED REASONING
    # ========================================================================
    "FuzzyLogicReasoner",
    "TemporalReasoner",
    "MetaReasoner",
    "ProofLearner",
    # ========================================================================
    # METADATA & CONVENIENCE
    # ========================================================================
    "__version__",
    "create_reasoner",
    "quick_prove",
    "check_consistency",
]


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def create_reasoner(prover_type: str = "parallel") -> SymbolicReasoner:
    """
    Create a symbolic reasoner with default settings.

    Convenience function for quick setup.

    Args:
        prover_type: Proving method to use ('parallel', 'resolution', etc.)

    Returns:
        Configured SymbolicReasoner instance
    """
    return SymbolicReasoner(prover_type=prover_type)


def quick_prove(
    goal: str, facts: List[str], method: str = "parallel", timeout: float = 5.0
) -> bool:
    """
    Quick theorem proving without creating a persistent reasoner.

    Convenience function for one-off proofs.

    Args:
        goal: Goal formula to prove
        facts: List of fact and rule formulas
        method: Proving method
        timeout: Timeout in seconds

    Returns:
        True if proven, False otherwise
    """
    reasoner = SymbolicReasoner(prover_type=method)
    for fact in facts:
        reasoner.add_rule(fact)

    result = reasoner.query(goal, timeout=timeout)
    return result.get("proven", False)


def check_consistency(formulas: List[str], timeout: float = 5.0) -> bool:
    """
    Check if a set of formulas is consistent (satisfiable).

    Args:
        formulas: List of formula strings
        timeout: Timeout in seconds

    Returns:
        True if consistent, False if contradictory
    """
    # To check consistency, we try to prove a contradiction (False).
    # A common way to represent False is an empty goal.
    reasoner = SymbolicReasoner(prover_type="resolution")  # Resolution is good for this

    for formula in formulas:
        reasoner.add_rule(formula)

    # If we can prove an empty goal, it means the KB is inconsistent.
    result = reasoner.query("", timeout=timeout)

    # The set is consistent if a contradiction is NOT proven.
    return not result.get("proven", False)
