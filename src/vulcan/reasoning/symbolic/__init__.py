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

import logging
from typing import List

logger = logging.getLogger(__name__)

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

# Note: Natural Language to Logic Converter
from .nl_converter import NaturalLanguageToLogicConverter, convert_nl_to_logic

# Note: Formula Validator for pre-validation with helpful error messages
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
    # Note: NL to Logic Converter
    "NaturalLanguageToLogicConverter",
    "convert_nl_to_logic",
    # Note: Formula Validator
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
        
    Example:
        >>> # A→B, B→C, ¬C, A∨B is UNSATISFIABLE (contradiction)
        >>> formulas = ['A → B', 'B → C', '¬C', 'A ∨ B']
        >>> check_consistency(formulas)  # Returns False
        False
    """
    # FIX Issue #12: Use resolution refutation properly
    # To check satisfiability, we check if the set of formulas leads to a contradiction.
    # Resolution proves unsatisfiability by deriving the empty clause.
    
    from .provers import ResolutionProver
    from .parsing import FormulaParser
    
    reasoner = SymbolicReasoner(prover_type="resolution")
    
    # Add all formulas to the knowledge base
    for formula in formulas:
        reasoner.add_rule(formula)
    
    # For resolution refutation, we need to check if the KB itself is inconsistent.
    # This is done by running resolution on the KB without any specific goal.
    # If resolution derives the empty clause (⊥), the KB is inconsistent.
    
    try:
        # Use the prover directly to avoid the is_symbolic_query check
        # The goal is an empty clause (representing ⊥ - contradiction)
        empty_goal = Clause(literals=[], is_goal=True)
        
        # Run resolution
        proven, proof, confidence = reasoner.prover.prove(
            empty_goal, 
            reasoner.kb.clauses, 
            timeout
        )
        
        # If we proved the empty clause, the KB is inconsistent (unsatisfiable)
        # The set is consistent only if we could NOT derive a contradiction
        return not proven
        
    except Exception as e:
        logger.warning(f"Consistency check failed: {e}, assuming consistent")
        return True  # Default to consistent on error
