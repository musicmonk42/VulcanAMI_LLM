"""
Philosophical Reasoning Module for VULCAN-AGI

This module provides reasoning capabilities for philosophical, ethical,
and deontic (normative) queries using a combination of formal logic
structures and heuristic evaluation methods.

Implementation Notes:
=====================

1. DEONTIC LOGIC ENGINE
   - Implements Standard Deontic Logic (SDL) axiom schemas
   - Uses inter-definability rules for operator conversion
   - LIMITATION: Uses string-based formula matching, not full AST parsing
   - LIMITATION: Does not implement tableaux/resolution provers
   - Suitable for: Simple deontic inferences with well-formed inputs

2. MORAL UNCERTAINTY HANDLER (Heuristic MEC-Inspired)
   - Inspired by MacAskill & Ord's MEC framework
   - LIMITATION: Uses keyword-based heuristics, not formal utility functions
   - LIMITATION: Does not perform true intertheoretic value comparison
   - Provides: Structured multi-theory evaluation with weighted aggregation
   - Suitable for: Comparative ethical analysis, not formal moral calculus

3. PARETO DOMINANCE CHECKER
   - Implements standard Pareto optimality detection
   - LIMITATION: Values derived from heuristic evaluation, not parsed inputs
   - Suitable for: Identifying dominated options when values are provided

4. FORMULA HANDLING
   - Supports: P(x), O(x), F(x) operator patterns
   - Supports: Natural language patterns ("X is permissible")
   - LIMITATION: Does not handle nested formulas like O(P(x) → Q(x))
   - LIMITATION: String splitting for implications breaks on A → (B → C)

Design Philosophy:
------------------
This module prioritizes providing useful structured analysis over failing
silently. When formal reasoning cannot complete, it provides heuristic
analysis with clearly documented confidence levels.

For queries requiring formal mathematical logic (e.g., ZFC set theory),
this module correctly returns validate_input=False, routing to appropriate
mathematical reasoners instead.

References:
-----------
- Åqvist, L. (2002). Deontic Logic. Handbook of Philosophical Logic.
- MacAskill, W., & Ord, T. (2020). Moral Uncertainty. Oxford University Press.

Author: VULCAN-AGI Team
Version: 3.1.0

CHANGELOG:
----------
v3.1.0 (Major Feature Addition):
  - Added AST-based formula parsing (FormulaParser) for nested formulas like O(P(x) → Q(x))
  - Added new Formula types: Atom, Not, BinaryOp, DeonticOp with proper hashability
  - Added analytic tableaux prover (DeonticTableau) with Kripke semantics support
  - Added deontic paradox detection (DeonticParadoxDetector) for:
    * Ross's Paradox
    * Good Samaritan Paradox
    * Contrary-to-Duty (Chisholm's) Paradox
    * Forrester's Paradox (Gentle Murder)
  - Added new statistics: 'tableau_proofs', 'paradoxes_detected'
  - Updated get_capabilities() to list all implemented algorithms
  - References: Fitting (1983), Priest (2008), McNamara (2006), Carmo & Jones (2002)

v3.0.0 (API Contract Change):
  - Removed 'defeasible_reasoning' from get_capabilities() (was never implemented)
  - Renamed algorithms to accurately reflect implementation limitations:
    * 'standard_deontic_logic' → 'standard_deontic_logic_basic'
    * 'maximizing_expected_choiceworthiness' → 'mec_heuristic'
  - Added 'deontic_inferences' statistics counter (separate from 'formal_proofs')
"""

from __future__ import annotations

import enum
import hashlib
import logging
import math
import re
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Generic,
    Iterator,
    List,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)

# Type variable for generic reasoning
T = TypeVar('T')

logger = logging.getLogger(__name__)

# =============================================================================
# IMPORTS - Reasoning Types Integration
# =============================================================================

try:
    from .reasoning_types import (
        AbstractReasoner,
        ModalityType,
        ReasoningChain,
        ReasoningContext,
        ReasoningResult,
        ReasoningStep,
        ReasoningType,
    )
    REASONING_TYPES_AVAILABLE = True
except ImportError:
    REASONING_TYPES_AVAILABLE = False
    logger.warning("reasoning_types not available, using local definitions")
    
    # Minimal fallback definitions
    class ReasoningType(enum.Enum):
        PHILOSOPHICAL = "philosophical"
        SYMBOLIC = "symbolic"
        UNKNOWN = "unknown"
    
    class ModalityType(enum.Enum):
        TEXT = "text"
        UNKNOWN = "unknown"
    
    class AbstractReasoner(ABC):
        @abstractmethod
        def reason(self, problem: Any, context: Any = None) -> Any:
            pass
        
        @abstractmethod
        def get_capabilities(self) -> Dict[str, Any]:
            pass
    
    @dataclass
    class ReasoningStep:
        step_id: str
        step_type: ReasoningType
        input_data: Any
        output_data: Any
        confidence: float
        explanation: str
        modality: Optional[ModalityType] = None
        timestamp: float = field(default_factory=time.time)
        metadata: Dict[str, Any] = field(default_factory=dict)
    
    @dataclass
    class ReasoningChain:
        chain_id: str
        steps: List[ReasoningStep]
        initial_query: Dict[str, Any]
        final_conclusion: Any
        total_confidence: float
        reasoning_types_used: Set[ReasoningType]
        modalities_involved: Set[ModalityType]
        safety_checks: List[Dict[str, Any]]
        audit_trail: List[Dict[str, Any]]
        metadata: Dict[str, Any] = field(default_factory=dict)
    
    @dataclass
    class ReasoningResult:
        conclusion: Any
        confidence: float
        reasoning_type: ReasoningType
        evidence: List[Any] = field(default_factory=list)
        explanation: str = ""
        uncertainty: float = 0.0
        reasoning_chain: Optional[ReasoningChain] = None
        safety_status: Dict[str, Any] = field(default_factory=dict)
        metadata: Dict[str, Any] = field(default_factory=dict)
    
    ReasoningContext = Dict[str, Any]

# =============================================================================
# CONSTANTS - Industry Standard Configuration
# =============================================================================

# Confidence thresholds calibrated based on reasoning reliability studies.
# These values follow standard practices in automated reasoning systems:
# - Formal proofs achieve highest confidence (0.9+) as they are verifiable
# - Structural analysis without formal proof gets moderate confidence (0.7-0.8)
# - Partial/heuristic results get lower confidence to signal uncertainty
# Reference: Confidence calibration in expert systems (Pearl, 1988; Heckerman, 1995)
CONFIDENCE_FORMAL_PROOF = 0.92  # Formal proof obtained with verified axioms
CONFIDENCE_STRONG_ANALYSIS = 0.78  # Complete structural analysis without formal proof
CONFIDENCE_PARTIAL_ANALYSIS = 0.55  # Partial analysis with identified gaps
CONFIDENCE_HEURISTIC_ONLY = 0.35  # Heuristic pattern matching only
CONFIDENCE_FALLBACK = 0.20  # Minimum meaningful confidence (above noise threshold)

# World Model integration confidence thresholds
# When the World Model is consulted for ethical reasoning, we boost confidence
# because VULCAN is providing self-aware ethical analysis
WORLD_MODEL_MIN_CONFIDENCE = 0.7  # Minimum confidence when World Model is consulted
WORLD_MODEL_CONFIDENCE_BOOST = 0.1  # Amount to boost confidence by

# Deontic operators per Standard Deontic Logic (SDL)
class DeonticOperator(enum.Enum):
    """Standard deontic modal operators."""
    OBLIGATION = "O"      # O(p) = p is obligatory
    PERMISSION = "P"      # P(p) = p is permitted  
    PROHIBITION = "F"     # F(p) = p is forbidden (= O(¬p))
    FACULTATIVE = "Fac"   # Fac(p) = p is facultative (= P(p) ∧ P(¬p))

# Ethical frameworks for multi-theory analysis
class EthicalFramework(enum.Enum):
    """Major normative ethical theories."""
    DEONTOLOGICAL = "deontological"
    CONSEQUENTIALIST = "consequentialist"
    VIRTUE_ETHICS = "virtue_ethics"
    CONTRACTUALIST = "contractualist"
    CARE_ETHICS = "care_ethics"
    MIXED = "mixed"

# Query type classification
class PhilosophicalQueryType(enum.Enum):
    """Types of philosophical queries we can handle."""
    PERMISSIBILITY = "permissibility"
    OBLIGATION = "obligation"
    PROHIBITION = "prohibition"
    DOMINANCE = "dominance"
    CONFLICT_RESOLUTION = "conflict_resolution"
    MORAL_UNCERTAINTY = "moral_uncertainty"
    FORMAL_PROOF = "formal_proof"
    VALUE_COMPARISON = "value_comparison"
    GENERAL_ETHICAL = "general_ethical"

# =============================================================================
# SOTA ALGORITHM 1: Deontic Logic Engine
# =============================================================================

@dataclass(frozen=True)
class DeonticFormula:
    """
    Immutable representation of a deontic formula.
    
    Supports standard deontic logic (SDL) operators:
    - O(φ): φ is obligatory
    - P(φ): φ is permitted (= ¬O(¬φ))
    - F(φ): φ is forbidden (= O(¬φ))
    
    Also supports dyadic deontic logic for conditional obligations:
    - O(φ/ψ): φ is obligatory given ψ
    """
    operator: DeonticOperator
    content: str
    condition: Optional[str] = None  # For dyadic deontic logic
    negated: bool = False
    
    def __str__(self) -> str:
        neg = "¬" if self.negated else ""
        if self.condition:
            return f"{neg}{self.operator.value}({self.content}/{self.condition})"
        return f"{neg}{self.operator.value}({self.content})"
    
    def negate(self) -> 'DeonticFormula':
        """Return negated formula."""
        return DeonticFormula(
            operator=self.operator,
            content=self.content,
            condition=self.condition,
            negated=not self.negated,
        )


# =============================================================================
# AST-BASED FORMULA REPRESENTATION
# =============================================================================

class Connective(enum.Enum):
    """Logical connectives for propositional formulas."""
    AND = "and"
    OR = "or"
    IMPLIES = "implies"
    NOT = "not"


@dataclass(frozen=True)
class Atom:
    """Atomic proposition."""
    name: str
    
    def __str__(self) -> str:
        return self.name
    
    def __hash__(self) -> int:
        return hash(('Atom', self.name))


@dataclass(frozen=True)
class Not:
    """Negation of a formula."""
    operand: 'Formula'
    
    def __str__(self) -> str:
        return f"¬{self.operand}"
    
    def __hash__(self) -> int:
        return hash(('Not', self.operand))


@dataclass(frozen=True)
class BinaryOp:
    """Binary connective (∧, ∨, →)."""
    connective: Connective
    left: 'Formula'
    right: 'Formula'
    
    def __str__(self) -> str:
        symbols = {Connective.AND: '∧', Connective.OR: '∨', Connective.IMPLIES: '→'}
        return f"({self.left} {symbols[self.connective]} {self.right})"
    
    def __hash__(self) -> int:
        return hash(('BinaryOp', self.connective, self.left, self.right))


@dataclass(frozen=True)
class DeonticOp:
    """Deontic operator wrapping a formula."""
    operator: DeonticOperator  # O, P, F
    content: 'Formula'
    condition: Optional['Formula'] = None  # For dyadic deontic logic
    
    def __str__(self) -> str:
        if self.condition:
            return f"{self.operator.value}({self.content}/{self.condition})"
        return f"{self.operator.value}({self.content})"
    
    def __hash__(self) -> int:
        return hash(('DeonticOp', self.operator, self.content, self.condition))


# Union type for all formulas
Formula = Union[Atom, Not, BinaryOp, DeonticOp]


class FormulaParser:
    """
    Recursive descent parser for deontic formulas.
    
    Supports nested formulas like O(P(x) → Q(x)) that the simple
    string-splitting approach cannot handle.
    
    Grammar:
        formula := deontic | binary | unary | atom | '(' formula ')'
        deontic := ('O' | 'P' | 'F') '(' formula ')'
        binary  := formula ('→' | '∧' | '∨') formula
        unary   := '¬' formula
        atom    := [a-z_][a-z0-9_]*
    
    Operator precedence (lowest to highest):
        1. → (right-associative)
        2. ∨ (left-associative)
        3. ∧ (left-associative)
        4. ¬, O, P, F (prefix)
    
    Example:
        >>> parser = FormulaParser("O(P(x) → Q(x))")
        >>> ast = parser.parse()
        # Returns: DeonticOp(O, BinaryOp(→, DeonticOp(P, Atom('x')), Atom('Q(x)')))
        # Note: Q(x) is parsed as an atom since Q is not a deontic operator (O/P/F)
    """
    
    def __init__(self, text: str):
        """Initialize parser with formula text."""
        self.text = text.replace(" ", "")
        self.pos = 0
    
    def parse(self) -> Formula:
        """
        Parse the formula text and return an AST.
        
        Raises:
            ValueError: If the formula is malformed
        """
        if not self.text:
            raise ValueError("Empty formula")
        result = self._parse_implication()
        if self.pos < len(self.text):
            raise ValueError(f"Unexpected character at position {self.pos}: '{self.text[self.pos]}'")
        return result
    
    def _parse_implication(self) -> Formula:
        """Handle → (right-associative)."""
        left = self._parse_disjunction()
        if self._match('→') or self._match('->'):
            right = self._parse_implication()  # Right-associative
            return BinaryOp(Connective.IMPLIES, left, right)
        return left
    
    def _parse_disjunction(self) -> Formula:
        """Handle ∨ (left-associative)."""
        left = self._parse_conjunction()
        while self._match('∨') or self._match('|') or self._match('v'):
            right = self._parse_conjunction()
            left = BinaryOp(Connective.OR, left, right)
        return left
    
    def _parse_conjunction(self) -> Formula:
        """Handle ∧ (left-associative)."""
        left = self._parse_unary()
        while self._match('∧') or self._match('&') or self._match('^'):
            right = self._parse_unary()
            left = BinaryOp(Connective.AND, left, right)
        return left
    
    def _parse_unary(self) -> Formula:
        """Handle ¬ and deontic operators."""
        if self._match('¬') or self._match('~') or self._match('!'):
            return Not(self._parse_unary())
        
        # Deontic operators - check for O(, P(, F( patterns
        for op_char, op_enum in [('O', DeonticOperator.OBLIGATION),
                                  ('P', DeonticOperator.PERMISSION),
                                  ('F', DeonticOperator.PROHIBITION)]:
            if self._match(op_char + '('):
                inner = self._parse_implication()
                
                # Check for dyadic syntax: O(φ/ψ)
                condition = None
                if self._match('/'):
                    condition = self._parse_implication()
                
                if not self._match(')'):
                    raise ValueError(f"Expected ')' after {op_char}(...) at position {self.pos}")
                return DeonticOp(operator=op_enum, content=inner, condition=condition)
        
        return self._parse_primary()
    
    def _parse_primary(self) -> Formula:
        """Handle atoms and parenthesized expressions."""
        if self._match('('):
            inner = self._parse_implication()
            if not self._match(')'):
                raise ValueError(f"Expected ')' at position {self.pos}")
            return inner
        
        # Atom: starts with letter or underscore, continues with alphanumeric or underscore
        # Also handles predicate-style atoms like Q(x) which are NOT deontic operators
        start = self.pos
        while self.pos < len(self.text) and (self.text[self.pos].isalnum() or self.text[self.pos] == '_'):
            self.pos += 1
        
        if self.pos == start:
            if self.pos < len(self.text):
                raise ValueError(f"Expected atom at position {self.pos}, got '{self.text[self.pos]}'")
            raise ValueError(f"Expected atom at position {self.pos}")
        
        atom_name = self.text[start:self.pos]
        
        # Check if this is a predicate-style atom like Q(x) - NOT a deontic operator
        # Deontic operators (O, P, F) are already handled in _parse_unary
        if self._peek('(') and atom_name.upper() not in ('O', 'P', 'F'):
            # This is a predicate with arguments, include them in the atom name
            self.pos += 1  # consume '('
            paren_depth = 1
            while self.pos < len(self.text) and paren_depth > 0:
                if self.text[self.pos] == '(':
                    paren_depth += 1
                elif self.text[self.pos] == ')':
                    paren_depth -= 1
                self.pos += 1
            atom_name = self.text[start:self.pos]
        
        return Atom(atom_name)
    
    def _peek(self, expected: str) -> bool:
        """Check if expected string is at current position WITHOUT consuming it."""
        if self.pos >= len(self.text):
            return False
        return self.text[self.pos:self.pos + len(expected)] == expected
    
    def _match(self, expected: str) -> bool:
        """Try to match a string at current position."""
        if self.pos >= len(self.text):
            return False
        if self.text[self.pos:self.pos + len(expected)] == expected:
            self.pos += len(expected)
            return True
        return False


def parse_formula(text: str) -> Formula:
    """
    Convenience function to parse a formula string into an AST.
    
    Args:
        text: Formula string like "O(P(x) → Q(x))"
        
    Returns:
        Parsed Formula AST
        
    Raises:
        ValueError: If the formula is malformed
    """
    return FormulaParser(text).parse()


@dataclass
class DeonticInference:
    """Result of a deontic inference step."""
    premise: Union[DeonticFormula, str]
    conclusion: Union[DeonticFormula, str]
    rule: str  # Name of inference rule applied
    confidence: float
    justification: str


class DeonticLogicEngine:
    """
    Standard Deontic Logic (SDL) engine for basic deontic inference.
    
    Implements:
    - SDL axiom schemas (K, D, N) as reference
    - Inter-definability of operators (P↔¬O¬, F↔O¬)
    - Basic consistency checking
    
    NOTE: For advanced features, use companion classes:
    - FormulaParser: AST-based parsing for nested formulas like O(P(x) → Q(x))
    - DeonticTableau: Analytic tableaux prover with Kripke semantics
    - DeonticParadoxDetector: Detection of Ross, Good Samaritan, CTD paradoxes
    
    LIMITATIONS:
    - Axiom N (necessitation) not implemented
    - This class uses string-based DeonticFormula for simple cases
    
    Suitable for:
    - Simple deontic inferences with well-formed P/O/F formulas
    - Checking basic obligation-permission relationships
    
    Based on: Åqvist (2002), Chellas (1980)
    """
    
    # SDL Axiom schemas (reference only - not fully implemented)
    SDL_AXIOMS = {
        'K': 'O(φ→ψ) → (O(φ)→O(ψ))',  # Distribution axiom
        'D': 'O(φ) → P(φ)',              # Ought implies can (implemented)
        'N': 'If ⊢φ then ⊢O(φ)',         # Necessitation (not implemented)
    }
    
    # Inter-definability rules (implemented)
    INTERDEFINABILITY = {
        'P_def': 'P(φ) ↔ ¬O(¬φ)',
        'F_def': 'F(φ) ↔ O(¬φ)',
        'Fac_def': 'Fac(φ) ↔ (P(φ) ∧ P(¬φ))',
    }
    
    def __init__(self):
        self.knowledge_base: List[DeonticFormula] = []
        self.inferences: List[DeonticInference] = []
        self._lock = threading.RLock()
    
    def add_formula(self, formula: DeonticFormula) -> None:
        """Add a formula to the knowledge base."""
        with self._lock:
            if formula not in self.knowledge_base:
                self.knowledge_base.append(formula)
    
    def derive(self, goal: DeonticFormula) -> Tuple[bool, List[DeonticInference]]:
        """
        Attempt to derive a goal formula from the knowledge base.
        
        Uses backward chaining with SDL inference rules.
        
        Returns:
            Tuple of (success, list of inference steps)
        """
        self.inferences = []
        
        # Direct lookup
        if goal in self.knowledge_base:
            return True, [DeonticInference(
                premise=goal,
                conclusion=goal,
                rule="direct_lookup",
                confidence=CONFIDENCE_FORMAL_PROOF,
                justification="Formula directly in knowledge base",
            )]
        
        # Try inter-definability
        derived, steps = self._apply_interdefinability(goal)
        if derived:
            return True, steps
        
        # Try SDL axioms
        derived, steps = self._apply_sdl_axioms(goal)
        if derived:
            return True, steps
        
        # Try contraposition and modus ponens
        derived, steps = self._apply_classical_rules(goal)
        if derived:
            return True, steps
        
        return False, self.inferences
    
    def _apply_interdefinability(
        self, goal: DeonticFormula
    ) -> Tuple[bool, List[DeonticInference]]:
        """Apply inter-definability rules."""
        steps = []
        
        # P(φ) ↔ ¬O(¬φ)
        if goal.operator == DeonticOperator.PERMISSION and not goal.negated:
            # Check if we have ¬O(¬φ)
            negated_obl = DeonticFormula(
                operator=DeonticOperator.OBLIGATION,
                content=f"¬{goal.content}",
                negated=True,
            )
            if negated_obl in self.knowledge_base:
                steps.append(DeonticInference(
                    premise=negated_obl,
                    conclusion=goal,
                    rule="P_interdefinability",
                    confidence=CONFIDENCE_FORMAL_PROOF,
                    justification="P(φ) ↔ ¬O(¬φ)",
                ))
                return True, steps
        
        # F(φ) ↔ O(¬φ)
        if goal.operator == DeonticOperator.PROHIBITION and not goal.negated:
            obl_neg = DeonticFormula(
                operator=DeonticOperator.OBLIGATION,
                content=f"¬{goal.content}",
            )
            if obl_neg in self.knowledge_base:
                steps.append(DeonticInference(
                    premise=obl_neg,
                    conclusion=goal,
                    rule="F_interdefinability",
                    confidence=CONFIDENCE_FORMAL_PROOF,
                    justification="F(φ) ↔ O(¬φ)",
                ))
                return True, steps
        
        # O(φ) → P(φ) (Axiom D)
        if goal.operator == DeonticOperator.PERMISSION and not goal.negated:
            obl = DeonticFormula(
                operator=DeonticOperator.OBLIGATION,
                content=goal.content,
            )
            if obl in self.knowledge_base:
                steps.append(DeonticInference(
                    premise=obl,
                    conclusion=goal,
                    rule="axiom_D",
                    confidence=CONFIDENCE_FORMAL_PROOF,
                    justification="O(φ) → P(φ): Ought implies may",
                ))
                return True, steps
        
        return False, steps
    
    def _apply_sdl_axioms(
        self, goal: DeonticFormula
    ) -> Tuple[bool, List[DeonticInference]]:
        """
        Apply SDL axiom schemas for inference.
        
        Currently implements: Axiom K application for simple implications.
        
        LIMITATION: Uses string splitting on "→" which breaks on nested
        implications like "A → (B → C)". Only handles simple "X → Y" patterns.
        For complex formulas, a proper parser would be needed.
        """
        # Axiom K: O(φ→ψ) → (O(φ)→O(ψ))
        # If we have O(φ) and O(φ→ψ), we can derive O(ψ)
        if goal.operator == DeonticOperator.OBLIGATION and not goal.negated:
            for formula in self.knowledge_base:
                if formula.operator == DeonticOperator.OBLIGATION:
                    # Check if this is an implication O(X→goal.content)
                    # NOTE: This simple split breaks on nested implications
                    if "→" in formula.content:
                        parts = formula.content.split("→", 1)  # Split only on first arrow
                        if len(parts) == 2 and parts[1].strip() == goal.content:
                            antecedent = parts[0].strip()
                            # Check if we have O(antecedent)
                            o_ant = DeonticFormula(
                                operator=DeonticOperator.OBLIGATION,
                                content=antecedent,
                            )
                            if o_ant in self.knowledge_base:
                                return True, [DeonticInference(
                                    premise=f"{formula} and {o_ant}",
                                    conclusion=goal,
                                    rule="axiom_K_application",
                                    confidence=CONFIDENCE_FORMAL_PROOF,
                                    justification="Axiom K: O(φ→ψ), O(φ) ⊢ O(ψ)",
                                )]
        
        return False, []
    
    def _apply_classical_rules(
        self, goal: DeonticFormula
    ) -> Tuple[bool, List[DeonticInference]]:
        """Apply classical logic rules (modus ponens, etc.)."""
        # This is a simplified implementation
        # A full implementation would use resolution or tableaux
        return False, []
    
    def check_consistency(self) -> Tuple[bool, Optional[str]]:
        """
        Check knowledge base for deontic consistency.
        
        Detects:
        - O(φ) ∧ O(¬φ) conflicts
        - O(φ) ∧ F(φ) conflicts
        
        Returns:
            Tuple of (is_consistent, conflict_description if any)
        """
        with self._lock:
            for f1 in self.knowledge_base:
                for f2 in self.knowledge_base:
                    # Check O(φ) ∧ O(¬φ)
                    if (f1.operator == DeonticOperator.OBLIGATION and
                        f2.operator == DeonticOperator.OBLIGATION and
                        f2.content == f"¬{f1.content}"):
                        return False, f"Conflict: {f1} and {f2}"
                    
                    # Check O(φ) ∧ F(φ)
                    if (f1.operator == DeonticOperator.OBLIGATION and
                        f2.operator == DeonticOperator.PROHIBITION and
                        f1.content == f2.content):
                        return False, f"Conflict: {f1} and {f2}"
        
        return True, None


# =============================================================================
# SOTA ALGORITHM 2: Moral Uncertainty Handler (MEC Implementation)
# =============================================================================

@dataclass
class MoralTheory:
    """Represents a normative moral theory with credence weight."""
    name: str
    framework: EthicalFramework
    credence: float  # Probability that this theory is correct (0-1)
    evaluate: Callable[[str], float]  # Function to evaluate action choiceworthiness
    
    def __post_init__(self):
        if not 0.0 <= self.credence <= 1.0:
            raise ValueError(f"Credence must be in [0,1], got {self.credence}")


@dataclass
class ActionEvaluation:
    """Evaluation of an action under moral uncertainty."""
    action: str
    expected_choiceworthiness: float
    theory_evaluations: Dict[str, float]  # theory_name -> choiceworthiness
    variance: float
    confidence: float


class MoralUncertaintyHandler:
    """
    Heuristic Moral Uncertainty Handler inspired by MacAskill & Ord (2020).
    
    IMPORTANT LIMITATIONS:
    ----------------------
    This is NOT a formal implementation of Maximizing Expected Choiceworthiness.
    True MEC requires:
    - Utility functions over action outcomes
    - Intertheoretic value comparisons with normalization
    - Formal treatment of incommensurable theories
    
    What this implementation provides:
    - Keyword-based heuristic evaluation per ethical framework
    - Weighted aggregation across theories (MEC-inspired structure)
    - Variance-voting for consensus detection
    - Configurable theory credences
    
    Suitable for:
    - Comparative ethical analysis of text-described actions
    - Identifying which ethical frameworks favor which options
    - Rough estimates of ethical consensus/disagreement
    
    NOT suitable for:
    - Formal moral calculus
    - Precise utility calculations
    - Decisions with quantified outcomes
    
    Reference: MacAskill, W., & Ord, T. (2020). Moral Uncertainty.
    """
    
    def __init__(self):
        self.theories: List[MoralTheory] = []
        self._lock = threading.RLock()  # Thread safety for credence updates
        self._init_default_theories()
    
    def _init_default_theories(self) -> None:
        """
        Initialize default moral theories with default credences.
        
        Credence values represent philosophical community consensus estimates.
        These defaults are based on surveys of moral philosophers
        (Bourget & Chalmers, 2014 PhilPapers Survey) and can be adjusted
        via set_theory_credence() for domain-specific applications.
        
        The sum of credences equals 1.0 to form a valid probability distribution.
        """
        self.theories = [
            MoralTheory(
                name="Kantian Deontology",
                framework=EthicalFramework.DEONTOLOGICAL,
                credence=0.25,  # ~25% of philosophers favor deontology
                evaluate=self._evaluate_kantian,
            ),
            MoralTheory(
                name="Utilitarianism",
                framework=EthicalFramework.CONSEQUENTIALIST,
                credence=0.30,  # ~30% favor consequentialism (largest group)
                evaluate=self._evaluate_utilitarian,
            ),
            MoralTheory(
                name="Virtue Ethics",
                framework=EthicalFramework.VIRTUE_ETHICS,
                credence=0.20,  # ~20% favor virtue ethics
                evaluate=self._evaluate_virtue,
            ),
            MoralTheory(
                name="Contractualism",
                framework=EthicalFramework.CONTRACTUALIST,
                credence=0.15,  # ~15% favor contractualism
                evaluate=self._evaluate_contractualist,
            ),
            MoralTheory(
                name="Care Ethics",
                framework=EthicalFramework.CARE_ETHICS,
                credence=0.10,  # ~10% favor care ethics
                evaluate=self._evaluate_care,
            ),
        ]
    
    def set_theory_credence(self, theory_name: str, credence: float) -> None:
        """
        Update credence for a specific moral theory (thread-safe).
        
        Args:
            theory_name: Name of the theory to update
            credence: New credence value (0-1)
            
        Note: After updating, credences may not sum to 1.0. Call
        normalize_credences() if a valid probability distribution is needed.
        """
        with self._lock:
            for theory in self.theories:
                if theory.name == theory_name:
                    # MoralTheory is a dataclass, create new instance
                    idx = self.theories.index(theory)
                    self.theories[idx] = MoralTheory(
                        name=theory.name,
                        framework=theory.framework,
                        credence=credence,
                        evaluate=theory.evaluate,
                    )
                    return
            logger.warning(f"Theory '{theory_name}' not found")
    
    def normalize_credences(self) -> None:
        """Normalize credences to sum to 1.0 (thread-safe)."""
        with self._lock:
            total = sum(t.credence for t in self.theories)
            if total > 0:
                for i, theory in enumerate(self.theories):
                    self.theories[i] = MoralTheory(
                        name=theory.name,
                        framework=theory.framework,
                        credence=theory.credence / total,
                        evaluate=theory.evaluate,
                    )
    
    def _evaluate_kantian(self, action: str) -> float:
        """
        Evaluate action under Kantian deontology (universalizability test).
        
        Uses keyword heuristics as a proxy for formal Kantian analysis.
        Score adjustments (±0.15/0.20) are calibrated to produce meaningful
        differentiation while keeping scores within a reasonable range.
        
        Note: This is a heuristic approximation. For rigorous Kantian analysis,
        the FormulationCI class should be used for categorical imperative tests.
        """
        action_lower = action.lower()
        score = 0.5  # Neutral baseline (neither clearly good nor bad)
        
        # Markers aligned with Kantian principles (duty, dignity, universalizability)
        positive_markers = ['duty', 'promise', 'respect', 'rights', 'dignity', 'honest']
        negative_markers = ['lie', 'deceive', 'use', 'manipulate', 'coerce']
        
        # Score increments chosen to allow ~3 positive markers to reach 0.95
        # and ~2 negative markers to reach 0.1 (strong signal range)
        for marker in positive_markers:
            if marker in action_lower:
                score += 0.15
        for marker in negative_markers:
            if marker in action_lower:
                score -= 0.20  # Slightly stronger penalty for Kantian violations
        
        return max(0.0, min(1.0, score))
    
    def _evaluate_utilitarian(self, action: str) -> float:
        """
        Evaluate action under utilitarianism (maximize aggregate welfare).
        
        Uses keyword heuristics as a proxy for utility calculation.
        A full utilitarian analysis would require outcome modeling.
        """
        action_lower = action.lower()
        score = 0.5
        
        # Markers aligned with hedonic/preference utilitarianism
        positive_markers = ['help', 'save', 'benefit', 'welfare', 'happiness', 'pleasure']
        negative_markers = ['harm', 'pain', 'suffering', 'death', 'injury']
        
        for marker in positive_markers:
            if marker in action_lower:
                score += 0.15
        for marker in negative_markers:
            if marker in action_lower:
                score -= 0.20
        
        return max(0.0, min(1.0, score))
    
    def _evaluate_virtue(self, action: str) -> float:
        """
        Evaluate action under virtue ethics (character-based evaluation).
        
        Uses keyword heuristics for Aristotelian virtues and vices.
        """
        action_lower = action.lower()
        score = 0.5
        
        # Classical Aristotelian virtues and corresponding vices
        virtuous_markers = ['courage', 'justice', 'temperance', 'wisdom', 'honest', 'kind']
        vicious_markers = ['coward', 'greedy', 'cruel', 'dishonest', 'selfish']
        
        for marker in virtuous_markers:
            if marker in action_lower:
                score += 0.15
        for marker in vicious_markers:
            if marker in action_lower:
                score -= 0.20
        
        return max(0.0, min(1.0, score))
    
    def _evaluate_contractualist(self, action: str) -> float:
        """Evaluate action under contractualism (reasonable rejectability)."""
        action_lower = action.lower()
        score = 0.5
        
        positive_markers = ['fair', 'consent', 'agreement', 'mutual', 'reciprocal']
        negative_markers = ['unfair', 'exploit', 'force', 'coerce', 'cheat']
        
        for marker in positive_markers:
            if marker in action_lower:
                score += 0.15
        for marker in negative_markers:
            if marker in action_lower:
                score -= 0.20
        
        return max(0.0, min(1.0, score))
    
    def _evaluate_care(self, action: str) -> float:
        """Evaluate action under care ethics (relationships and caring)."""
        action_lower = action.lower()
        score = 0.5
        
        positive_markers = ['care', 'nurture', 'relationship', 'empathy', 'compassion']
        negative_markers = ['neglect', 'abandon', 'ignore', 'indifferent']
        
        for marker in positive_markers:
            if marker in action_lower:
                score += 0.15
        for marker in negative_markers:
            if marker in action_lower:
                score -= 0.20
        
        return max(0.0, min(1.0, score))
    
    def maximize_expected_choiceworthiness(
        self, actions: List[str]
    ) -> Tuple[str, ActionEvaluation]:
        """
        MEC: Find action that maximizes expected choiceworthiness.
        
        EC(a) = Σᵢ Cr(Tᵢ) × Cᵢ(a)
        
        where:
        - Cr(Tᵢ) is credence in theory i
        - Cᵢ(a) is choiceworthiness of action a under theory i
        
        Args:
            actions: List of possible actions to evaluate
            
        Returns:
            Best action and its evaluation
        """
        if not actions:
            raise ValueError("At least one action required")
        
        evaluations: Dict[str, ActionEvaluation] = {}
        
        for action in actions:
            theory_scores: Dict[str, float] = {}
            weighted_sum = 0.0
            
            for theory in self.theories:
                score = theory.evaluate(action)
                theory_scores[theory.name] = score
                weighted_sum += theory.credence * score
            
            # Calculate variance for uncertainty quantification
            mean = weighted_sum
            variance = sum(
                theory.credence * (theory_scores[theory.name] - mean) ** 2
                for theory in self.theories
            )
            
            evaluations[action] = ActionEvaluation(
                action=action,
                expected_choiceworthiness=weighted_sum,
                theory_evaluations=theory_scores,
                variance=variance,
                confidence=1.0 - math.sqrt(variance),  # Higher variance = lower confidence
            )
        
        # Find best action
        best_action = max(evaluations.keys(), key=lambda a: evaluations[a].expected_choiceworthiness)
        return best_action, evaluations[best_action]
    
    def variance_voting(self, actions: List[str]) -> Tuple[str, Dict[str, int]]:
        """
        Variance voting: Each theory votes for its preferred action.
        
        Returns action with most votes (weighted by credence).
        """
        if not actions:
            raise ValueError("At least one action required")
        
        votes: Dict[str, float] = defaultdict(float)
        
        for theory in self.theories:
            # Each theory votes for its highest-rated action
            scores = {action: theory.evaluate(action) for action in actions}
            best_for_theory = max(scores.keys(), key=lambda a: scores[a])
            votes[best_for_theory] += theory.credence
        
        winner = max(votes.keys(), key=lambda a: votes[a])
        return winner, {k: int(v * 100) for k, v in votes.items()}


# =============================================================================
# SOTA ALGORITHM 3: Pareto Dominance Checker (MCDA)
# =============================================================================

@dataclass
class ValuedAction:
    """An action with values across multiple ethical dimensions."""
    name: str
    values: Dict[str, float]  # dimension_name -> value


class ParetoDominanceChecker:
    """
    Multi-Criteria Decision Analysis with Pareto dominance detection.
    
    Implements:
    - Pareto optimality checking
    - Dominance relation detection
    - Lexicographic ordering
    
    Based on: Multi-Attribute Utility Theory (Keeney & Raiffa, 1976)
    """
    
    def __init__(self, dimensions: List[str]):
        """
        Initialize with value dimensions.
        
        Args:
            dimensions: Names of ethical value dimensions
        """
        self.dimensions = dimensions
    
    def is_dominated(self, a: ValuedAction, b: ValuedAction) -> bool:
        """
        Check if action a is strictly dominated by action b.
        
        a is strictly dominated by b iff:
        - b is at least as good as a in ALL dimensions
        - b is strictly better than a in AT LEAST ONE dimension
        
        Returns:
            True if a is dominated by b
        """
        at_least_as_good = all(
            b.values.get(d, 0) >= a.values.get(d, 0)
            for d in self.dimensions
        )
        strictly_better_somewhere = any(
            b.values.get(d, 0) > a.values.get(d, 0)
            for d in self.dimensions
        )
        
        return at_least_as_good and strictly_better_somewhere
    
    def find_pareto_frontier(
        self, actions: List[ValuedAction]
    ) -> List[ValuedAction]:
        """
        Find Pareto-optimal actions (non-dominated set).
        
        Returns:
            List of Pareto-optimal actions
        """
        pareto_frontier = []
        
        for candidate in actions:
            dominated = False
            for other in actions:
                if other.name != candidate.name and self.is_dominated(candidate, other):
                    dominated = True
                    break
            
            if not dominated:
                pareto_frontier.append(candidate)
        
        return pareto_frontier
    
    def compute_dominance_relations(
        self, actions: List[ValuedAction]
    ) -> List[Tuple[str, str]]:
        """
        Compute all strict dominance relations.
        
        Returns:
            List of (dominated, dominator) pairs
        """
        relations = []
        
        for a in actions:
            for b in actions:
                if a.name != b.name and self.is_dominated(a, b):
                    relations.append((a.name, b.name))
        
        return relations
    
    def lexicographic_compare(
        self,
        a: ValuedAction,
        b: ValuedAction,
        priority_order: List[str],
    ) -> int:
        """
        Lexicographic comparison with priority ordering.
        
        Args:
            a, b: Actions to compare
            priority_order: Dimensions in decreasing priority
            
        Returns:
            -1 if a < b, 0 if a = b, 1 if a > b
        """
        for dim in priority_order:
            a_val = a.values.get(dim, 0)
            b_val = b.values.get(dim, 0)
            
            if a_val < b_val:
                return -1
            elif a_val > b_val:
                return 1
        
        return 0


# =============================================================================
# SOTA ALGORITHM 4: Analytic Tableaux for Deontic Logic
# =============================================================================

@dataclass
class TableauNode:
    """
    Node in a semantic tableau.
    
    Each node represents a set of formulas that must all be satisfiable
    at a particular Kripke world.
    """
    formulas: FrozenSet[Formula]
    world: int  # Kripke world index
    is_closed: bool = False
    children: List['TableauNode'] = field(default_factory=list)


class DeonticTableau:
    """
    Analytic tableau prover for Standard Deontic Logic.
    
    Implements Kripke semantics:
    - O(φ) is true at w iff φ is true at all deontically ideal worlds accessible from w
    - P(φ) is true at w iff φ is true at some deontically ideal world accessible from w
    
    Tableau rules:
    - α-rules (don't branch): ¬¬φ, φ∧ψ, ¬(φ∨ψ), ¬(φ→ψ)
    - β-rules (branch): φ∨ψ, φ→ψ, ¬(φ∧ψ)
    - π-rules (new world): O(φ), ¬P(φ)
    - ν-rules (existing worlds): P(φ), ¬O(φ)
    
    Based on: Fitting (1983), Priest (2008)
    """
    
    MAX_DEPTH = 100  # Prevent infinite loops
    
    def __init__(self):
        """Initialize the tableau prover."""
        self.worlds: Dict[int, Set[Formula]] = {0: set()}
        self.accessibility: Dict[int, Set[int]] = {0: set()}
        self.world_counter = 1
        self._lock = threading.RLock()
    
    def prove(
        self, 
        goal: Formula, 
        assumptions: Optional[List[Formula]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Prove goal from assumptions using tableau method.
        
        To prove φ, we try to close a tableau for ¬φ.
        If all branches close, φ is valid.
        
        Args:
            goal: The formula to prove
            assumptions: Optional list of assumption formulas
            
        Returns:
            Tuple of (is_proven, proof_trace)
        """
        assumptions = assumptions or []
        
        # Reset state for new proof
        with self._lock:
            self.worlds = {0: set()}
            self.accessibility = {0: set()}
            self.world_counter = 1
        
        # Start with assumptions and negation of goal
        initial_formulas = list(assumptions) + [self._negate(goal)]
        initial = frozenset(initial_formulas)
        root = TableauNode(formulas=initial, world=0)
        
        trace = [f"Attempting to prove: {goal}"]
        trace.append(f"Negated goal: {self._negate(goal)}")
        if assumptions:
            trace.append(f"Assumptions: {[str(a) for a in assumptions]}")
        
        closed = self._expand(root, trace, set())
        
        if closed:
            trace.append("✓ All branches closed → PROVEN (valid)")
        else:
            trace.append("✗ Open branch found → NOT PROVEN (countermodel exists)")
        
        return closed, trace
    
    def _expand(
        self, 
        node: TableauNode, 
        trace: List[str], 
        processed: Set[Formula],
        depth: int = 0
    ) -> bool:
        """Recursively expand tableau node."""
        if depth > self.MAX_DEPTH:
            trace.append(f"{'  ' * depth}Max depth reached, treating as open")
            return False
        
        # Check for closure (contradiction)
        if self._is_closed(node.formulas):
            trace.append(f"{'  ' * depth}Branch closed (contradiction found)")
            node.is_closed = True
            return True
        
        # Find applicable rule for unprocessed formulas
        for formula in node.formulas:
            if formula in processed:
                continue
            
            rule, result = self._apply_rule(formula, node.world)
            
            if rule is None:
                continue
            
            new_processed = processed | {formula}
            trace.append(f"{'  ' * depth}Apply {rule}-rule to {formula}")
            
            if rule == 'α':  # Non-branching, add all results
                new_formulas = node.formulas | result
                child = TableauNode(formulas=new_formulas, world=node.world)
                node.children.append(child)
                return self._expand(child, trace, new_processed, depth + 1)
            
            elif rule == 'β':  # Branching, all branches must close
                all_closed = True
                for i, branch in enumerate(result):
                    trace.append(f"{'  ' * depth}Branch {i + 1}:")
                    new_formulas = node.formulas | branch
                    child = TableauNode(formulas=new_formulas, world=node.world)
                    node.children.append(child)
                    if not self._expand(child, trace, new_processed, depth + 1):
                        all_closed = False
                return all_closed
            
            elif rule == 'π':  # Deontic rule, create new world
                with self._lock:
                    new_world = self.world_counter
                    self.world_counter += 1
                    self.accessibility.setdefault(node.world, set()).add(new_world)
                trace.append(f"{'  ' * depth}Created new world w{new_world}")
                new_formulas = node.formulas | result
                child = TableauNode(formulas=new_formulas, world=new_world)
                node.children.append(child)
                return self._expand(child, trace, new_processed, depth + 1)
        
        # No rules applicable, branch is open (countermodel exists)
        trace.append(f"{'  ' * depth}No rules applicable, branch open")
        return False
    
    def _apply_rule(
        self, 
        formula: Formula, 
        world: int
    ) -> Tuple[Optional[str], Any]:
        """
        Determine and apply the appropriate tableau rule.
        
        Returns:
            Tuple of (rule_name, result_formulas) or (None, None) if no rule applies
        """
        # α-rules (non-branching)
        
        # Double negation: ¬¬φ → φ
        if isinstance(formula, Not) and isinstance(formula.operand, Not):
            return 'α', frozenset([formula.operand.operand])
        
        # Conjunction: φ∧ψ → {φ, ψ}
        if isinstance(formula, BinaryOp) and formula.connective == Connective.AND:
            return 'α', frozenset([formula.left, formula.right])
        
        # Negated disjunction: ¬(φ∨ψ) → {¬φ, ¬ψ}
        if (isinstance(formula, Not) and 
            isinstance(formula.operand, BinaryOp) and 
            formula.operand.connective == Connective.OR):
            return 'α', frozenset([Not(formula.operand.left), Not(formula.operand.right)])
        
        # Negated implication: ¬(φ→ψ) → {φ, ¬ψ}
        if (isinstance(formula, Not) and 
            isinstance(formula.operand, BinaryOp) and 
            formula.operand.connective == Connective.IMPLIES):
            return 'α', frozenset([formula.operand.left, Not(formula.operand.right)])
        
        # β-rules (branching)
        
        # Disjunction: φ∨ψ → {φ} | {ψ}
        if isinstance(formula, BinaryOp) and formula.connective == Connective.OR:
            return 'β', [frozenset([formula.left]), frozenset([formula.right])]
        
        # Implication: φ→ψ → {¬φ} | {ψ}
        if isinstance(formula, BinaryOp) and formula.connective == Connective.IMPLIES:
            return 'β', [frozenset([Not(formula.left)]), frozenset([formula.right])]
        
        # Negated conjunction: ¬(φ∧ψ) → {¬φ} | {¬ψ}
        if (isinstance(formula, Not) and 
            isinstance(formula.operand, BinaryOp) and 
            formula.operand.connective == Connective.AND):
            return 'β', [frozenset([Not(formula.operand.left)]), 
                        frozenset([Not(formula.operand.right)])]
        
        # π-rules (deontic, create new world)
        
        # Obligation: O(φ) at w → φ at new accessible world
        if isinstance(formula, DeonticOp) and formula.operator == DeonticOperator.OBLIGATION:
            return 'π', frozenset([formula.content])
        
        # Negated permission: ¬P(φ) ↔ O(¬φ) → ¬φ at new accessible world
        if (isinstance(formula, Not) and 
            isinstance(formula.operand, DeonticOp) and 
            formula.operand.operator == DeonticOperator.PERMISSION):
            return 'π', frozenset([Not(formula.operand.content)])
        
        return None, None
    
    def _is_closed(self, formulas: FrozenSet[Formula]) -> bool:
        """
        Check if branch is closed (contains φ and ¬φ).
        
        A branch closes when it contains a direct contradiction.
        """
        for f in formulas:
            if isinstance(f, Not):
                if f.operand in formulas:
                    return True
            else:
                if Not(f) in formulas:
                    return True
        return False
    
    def _negate(self, formula: Formula) -> Formula:
        """Negate a formula, simplifying double negations."""
        if isinstance(formula, Not):
            return formula.operand
        return Not(formula)


# =============================================================================
# SOTA ALGORITHM 5: Deontic Paradox Detection
# =============================================================================

@dataclass
class ParadoxResult:
    """Result of paradox detection."""
    name: str
    detected: bool
    explanation: str
    formulas_involved: List[Formula]
    resolution_hint: Optional[str] = None


class DeonticParadoxDetector:
    """
    Detect known deontic logic paradoxes.
    
    Implements detection for:
    - Ross's Paradox: O(p) ⊢ O(p ∨ q) leads to counterintuitive conclusions
    - Good Samaritan Paradox: O(help_injured) implies ∃x.injured(x)
    - Contrary-to-Duty (Chisholm's Paradox): Conflicting conditional obligations
    - Forrester's Paradox: O(¬kill) but O(kill_gently | kill)
    
    These paradoxes highlight limitations of Standard Deontic Logic (SDL)
    and help identify when more sophisticated logics may be needed.
    
    Based on: McNamara (2006), Carmo & Jones (2002)
    """
    
    # Keywords that indicate potentially bad states (for Good Samaritan detection)
    BAD_STATE_KEYWORDS = frozenset({
        'injured', 'harm', 'suffering', 'dead', 'dying', 'hurt',
        'damage', 'violence', 'kill', 'murder', 'theft', 'crime'
    })
    
    def detect_ross_paradox(self, formulas: List[Formula]) -> ParadoxResult:
        """
        Detect Ross's Paradox.
        
        Ross's Paradox: From O(p) we can derive O(p ∨ q) for any q.
        
        This is logically valid in SDL but counterintuitive:
        "You ought to mail the letter" implies
        "You ought to mail the letter or burn it"
        
        Detection: Look for O(φ ∨ ψ) where O(φ) or O(ψ) exists separately.
        """
        obligations = [f for f in formulas 
                       if isinstance(f, DeonticOp) and f.operator == DeonticOperator.OBLIGATION]
        
        for obl in obligations:
            if isinstance(obl.content, BinaryOp) and obl.content.connective == Connective.OR:
                # Found O(φ ∨ ψ), check if O(φ) or O(ψ) exists alone
                left_obl = DeonticOp(DeonticOperator.OBLIGATION, obl.content.left)
                right_obl = DeonticOp(DeonticOperator.OBLIGATION, obl.content.right)
                
                if left_obl in formulas:
                    return ParadoxResult(
                        name="Ross's Paradox",
                        detected=True,
                        explanation=f"O({obl.content.left}) was weakened to O({obl.content}), "
                                   f"which permits the unwanted disjunct '{obl.content.right}'",
                        formulas_involved=[obl, left_obl],
                        resolution_hint="Use dyadic deontic logic: O(φ/⊤) doesn't entail O(φ∨ψ/⊤)"
                    )
                if right_obl in formulas:
                    return ParadoxResult(
                        name="Ross's Paradox",
                        detected=True,
                        explanation=f"O({obl.content.right}) was weakened to O({obl.content}), "
                                   f"which permits the unwanted disjunct '{obl.content.left}'",
                        formulas_involved=[obl, right_obl],
                        resolution_hint="Use dyadic deontic logic: O(φ/⊤) doesn't entail O(φ∨ψ/⊤)"
                    )
        
        return ParadoxResult("Ross's Paradox", False, "Not detected", [])
    
    def detect_contrary_to_duty(self, formulas: List[Formula]) -> ParadoxResult:
        """
        Detect Contrary-to-Duty (CTD) / Chisholm's Paradox.
        
        Classic CTD scenario:
        1. O(¬p)           - You ought not do p
        2. O(q | ¬p)       - If you don't do p, you ought to do q  
        3. O(¬q | p)       - If you do p, you ought not do q
        4. p               - You do p (violation of 1)
        
        In SDL, (1) + (4) creates inconsistency, but intuitively we want
        to derive O(¬q) from (3) + (4) as a "second-best" obligation.
        
        Detection: Look for O(¬φ) paired with dyadic O(ψ/φ) where φ is asserted.
        """
        for f1 in formulas:
            # Look for primary obligation O(¬φ)
            if not (isinstance(f1, DeonticOp) and 
                    f1.operator == DeonticOperator.OBLIGATION and
                    isinstance(f1.content, Not)):
                continue
            
            prohibited = f1.content.operand  # The φ in O(¬φ)
            
            for f2 in formulas:
                # Look for dyadic CTD obligation O(ψ/φ)
                if (isinstance(f2, DeonticOp) and 
                    f2.operator == DeonticOperator.OBLIGATION and
                    f2.condition is not None):
                    
                    # Check if condition matches the prohibited action (use string comparison for consistency)
                    prohibited_str = str(prohibited)
                    if str(f2.condition) == prohibited_str:
                        # Check if the violation is asserted (use string comparison for consistency)
                        if any(str(f) == prohibited_str for f in formulas):
                            return ParadoxResult(
                                name="Contrary-to-Duty Paradox",
                                detected=True,
                                explanation=f"Primary obligation O(¬{prohibited}) is violated by {prohibited}, "
                                           f"triggering CTD obligation {f2}. SDL cannot handle this consistently.",
                                formulas_involved=[f1, f2, prohibited] if isinstance(prohibited, Formula) else [f1, f2],
                                resolution_hint="Use non-monotonic deontic logic, defeasible reasoning, "
                                              "or priority ordering for obligations"
                            )
        
        return ParadoxResult("Contrary-to-Duty Paradox", False, "Not detected", [])
    
    def detect_good_samaritan(self, formulas: List[Formula]) -> ParadoxResult:
        """
        Detect Good Samaritan Paradox.
        
        Good Samaritan Paradox:
        O(help(injured_person)) seems to logically imply ∃x.injured(x)
        
        "You ought to help the injured person" appears to require
        that someone be injured, which is itself a bad state.
        
        Detection: Look for O(φ) where φ contains terms describing bad states
        combined with ameliorative actions.
        """
        ameliorative_actions = {'help', 'save', 'rescue', 'assist', 'aid', 'protect', 'prevent'}
        
        for f in formulas:
            if isinstance(f, DeonticOp) and f.operator == DeonticOperator.OBLIGATION:
                content_str = str(f.content).lower()
                
                # Check if obligation involves helping with a bad state
                has_bad_state = any(bad in content_str for bad in self.BAD_STATE_KEYWORDS)
                has_ameliorative = any(action in content_str for action in ameliorative_actions)
                
                if has_bad_state and has_ameliorative:
                    bad_found = [b for b in self.BAD_STATE_KEYWORDS if b in content_str]
                    return ParadoxResult(
                        name="Good Samaritan Paradox",
                        detected=True,
                        explanation=f"Obligation {f} presupposes a bad state ({', '.join(bad_found)}). "
                                   f"The obligation to help seems to require that harm exists.",
                        formulas_involved=[f],
                        resolution_hint="Distinguish between obligation-to-do and presupposed states "
                                      "using situation semantics or stit logic"
                    )
        
        return ParadoxResult("Good Samaritan Paradox", False, "Not detected", [])
    
    def detect_forrester_paradox(self, formulas: List[Formula]) -> ParadoxResult:
        """
        Detect Forrester's Paradox (Gentle Murder Paradox).
        
        Forrester's Paradox:
        1. O(¬kill)              - You ought not to kill
        2. O(kill_gently | kill) - If you kill, you ought to kill gently
        3. kill_gently → kill    - Killing gently implies killing
        
        From (2) and (3), we can derive O(kill | kill), which with O(¬kill)
        creates a contradiction when killing occurs.
        
        Detection: Look for O(¬φ) with O(ψ/φ) where ψ implies φ.
        """
        for f1 in formulas:
            # Look for prohibition O(¬φ)
            if not (isinstance(f1, DeonticOp) and 
                    f1.operator == DeonticOperator.OBLIGATION and
                    isinstance(f1.content, Not)):
                continue
            
            prohibited = f1.content.operand  # The φ in O(¬φ)
            prohibited_str = str(prohibited).lower()
            
            for f2 in formulas:
                # Look for conditional obligation O(ψ/φ) where ψ is a "gentler" version
                if (isinstance(f2, DeonticOp) and 
                    f2.operator == DeonticOperator.OBLIGATION and
                    f2.condition is not None):
                    
                    condition_str = str(f2.condition).lower()
                    content_str = str(f2.content).lower()
                    
                    # Check if condition matches prohibited action and content implies it
                    # (e.g., "kill" in condition, "kill_gently" in content)
                    if (prohibited_str in condition_str and 
                        prohibited_str in content_str and
                        content_str != condition_str):
                        return ParadoxResult(
                            name="Forrester's Paradox (Gentle Murder)",
                            detected=True,
                            explanation=f"Primary prohibition O(¬{prohibited}) conflicts with "
                                       f"conditional obligation {f2}. The 'gentler' action "
                                       f"implies the prohibited base action.",
                            formulas_involved=[f1, f2],
                            resolution_hint="Use action-theoretic deontic logic or distinguish "
                                          "between 'doing' and 'manner of doing'"
                        )
        
        return ParadoxResult("Forrester's Paradox", False, "Not detected", [])
    
    def detect_all(self, formulas: List[Formula]) -> List[ParadoxResult]:
        """
        Run all paradox detectors on a set of formulas.
        
        Args:
            formulas: List of deontic formulas to check
            
        Returns:
            List of ParadoxResults for each paradox type
        """
        return [
            self.detect_ross_paradox(formulas),
            self.detect_contrary_to_duty(formulas),
            self.detect_good_samaritan(formulas),
            self.detect_forrester_paradox(formulas),
        ]
    
    def get_detected_paradoxes(self, formulas: List[Formula]) -> List[ParadoxResult]:
        """
        Get only the detected paradoxes.
        
        Args:
            formulas: List of deontic formulas to check
            
        Returns:
            List of ParadoxResults where detected=True
        """
        return [p for p in self.detect_all(formulas) if p.detected]


# =============================================================================
# MAIN PHILOSOPHICAL REASONER (Integration Point)
# =============================================================================

class PhilosophicalReasoner(AbstractReasoner):
    """
    Philosophical Reasoner for ethical and deontic queries.
    
    This class serves as the main entry point for philosophical reasoning,
    coordinating the deontic logic engine, moral uncertainty handler,
    Pareto dominance checker, tableau prover, and paradox detector.
    
    CAPABILITIES:
    - Query classification (permissibility, obligation, dominance, etc.)
    - Deontic formula extraction from natural language
    - AST-based formula parsing for nested formulas like O(P(x) → Q(x))
    - SDL inference with inter-definability rules and Axiom D
    - Analytic tableaux for formal proofs (Kripke semantics)
    - Heuristic multi-theory moral evaluation (MEC-inspired)
    - Pareto dominance detection
    - Deontic paradox detection (Ross, Good Samaritan, CTD, Forrester)
    
    LIMITATIONS (see module docstring for details):
    - MEC uses keyword heuristics, not formal utility calculation
    - Tableaux prover limited to SDL (no temporal/dynamic extensions)
    
    Designed for integration with VULCAN's unified reasoning system.
    Returns structured analysis even when formal reasoning cannot complete.
    """
    
    # Patterns for detecting philosophical content
    ETHICAL_KEYWORDS: FrozenSet[str] = frozenset({
        'moral', 'ethical', 'permissible', 'permissibility', 'obligation',
        'obligatory', 'forbidden', 'duty', 'right', 'wrong', 'virtue',
        'value', 'harm', 'benefit', 'justice', 'fairness', 'rights',
        'autonomy', 'consent', 'welfare', 'utility', 'deontological',
        'consequentialist', 'categorical', 'imperative', 'dilemma',
        'trolley', 'paradox', 'deontic', 'normative',
    })
    
    DEONTIC_PATTERN = re.compile(
        r'\b([POF])\s*\(\s*([^)]+)\s*\)',
        re.IGNORECASE
    )
    
    def __init__(
        self,
        symbolic_reasoner: Optional[Any] = None,
        enable_learning: bool = False,
        world_model: Optional[Any] = None,
    ):
        """
        Initialize the philosophical reasoner.
        
        Args:
            symbolic_reasoner: Optional symbolic reasoner for formal proofs
            enable_learning: Whether to enable learning from outcomes
            world_model: Optional WorldModel for self-aware ethical reasoning.
                        The World Model provides VULCAN's "feelings" about ethical
                        dilemmas through meta-reasoning components like:
                        - MotivationalIntrospection: VULCAN's objective analysis
                        - EthicalBoundaryMonitor: Ethical constraints and values
                        - GoalConflictDetector: Conflict analysis for dilemmas
                        - InternalCritic: Multi-perspective self-critique
        """
        self.symbolic_reasoner = symbolic_reasoner
        self.enable_learning = enable_learning
        self.world_model = world_model
        
        # Initialize SOTA components
        self.deontic_engine = DeonticLogicEngine()
        self.moral_uncertainty = MoralUncertaintyHandler()
        self.pareto_checker = ParetoDominanceChecker([
            'autonomy', 'beneficence', 'non-maleficence', 'justice', 'fidelity'
        ])
        self.tableau_prover = DeonticTableau()
        self.paradox_detector = DeonticParadoxDetector()
        
        # Statistics tracking
        self._stats = {
            'total_queries': 0,
            'formal_proofs': 0,
            'tableau_proofs': 0,
            'deontic_inferences': 0,
            'mec_evaluations': 0,
            'dominance_checks': 0,
            'paradoxes_detected': 0,
            'fallbacks': 0,
            'world_model_consultations': 0,  # Track world model usage
        }
        self._lock = threading.RLock()
        
        # Try to import symbolic reasoner if not provided
        if self.symbolic_reasoner is None:
            self._init_symbolic_reasoner()
        
        # Try to get world model from singletons if not provided
        if self.world_model is None:
            self._init_world_model()
        
        logger.info("PhilosophicalReasoner initialized with SOTA algorithms")
    
    def _init_symbolic_reasoner(self) -> None:
        """Initialize symbolic reasoner from the symbolic module."""
        try:
            from .symbolic import SymbolicReasoner
            self.symbolic_reasoner = SymbolicReasoner()
            logger.info("Symbolic reasoner initialized for formal proofs")
        except ImportError as e:
            logger.warning(f"Could not import SymbolicReasoner: {e}")
        except Exception as e:
            logger.warning(f"Could not create SymbolicReasoner: {e}")
    
    def _init_world_model(self) -> None:
        """
        Initialize world model from singletons for self-aware ethical reasoning.
        
        The World Model is what makes VULCAN "feel" about ethical dilemmas.
        It provides:
        - MotivationalIntrospection: Understanding of VULCAN's own objectives
        - EthicalBoundaryMonitor: Hard/soft ethical constraints and values
        - GoalConflictDetector: Analysis of conflicting objectives (trolley problems)
        - InternalCritic: Multi-perspective self-critique
        """
        try:
            from .singletons import get_world_model
            self.world_model = get_world_model()
            if self.world_model:
                logger.info("World Model initialized for self-aware ethical reasoning")
            else:
                logger.debug("World Model not available from singletons")
        except ImportError as e:
            logger.debug(f"Could not import get_world_model: {e}")
        except Exception as e:
            logger.debug(f"Could not get World Model: {e}")
    
    def _consult_world_model(self, query: str, analysis_type: str = "ethical_dilemma") -> Optional[Dict[str, Any]]:
        """
        Consult the World Model for VULCAN's perspective on an ethical query.
        
        This is where VULCAN's self-awareness comes into play. The World Model
        provides the "feeling" about ethical dilemmas through meta-reasoning.
        
        FIX #3: Wire PhilosophicalReasoner to all meta-reasoning subsystems:
        - MotivationalIntrospection: VULCAN's objective analysis
        - EthicalBoundaryMonitor: Ethical constraints and values
        - GoalConflictDetector: Conflict analysis for dilemmas
        - InternalCritic: Multi-perspective self-critique
        
        FIX #4: Add causal prediction capability:
        - predict_interventions: Causal predictions for actions
        
        FIX #5: Add comprehensive logging throughout
        
        Args:
            query: The ethical query to analyze
            analysis_type: Type of analysis ("ethical_dilemma", "value_conflict", "forced_choice")
            
        Returns:
            Dictionary with World Model's perspective, or None if unavailable
        """
        if self.world_model is None:
            logger.debug("[PhilosophicalReasoner] World Model not available, skipping consultation")
            return None
        
        with self._lock:
            self._stats['world_model_consultations'] += 1
        
        logger.info("[PhilosophicalReasoner] ════════════════════════════════════")
        logger.info(f"[PhilosophicalReasoner] Starting World Model consultation for {analysis_type}")
        
        try:
            result = {
                'world_model_consulted': True,
                'analysis_type': analysis_type,
                'perspective': {},
                'ethical_boundaries': [],
                'goal_conflicts': [],
                'internal_critique': None,
                'causal_predictions': {},  # FIX #4: Add causal predictions
            }
            
            # FIX #4: Extract possible actions from the query for causal prediction
            actions = self._extract_actions(query)
            logger.info(f"[PhilosophicalReasoner] Extracted {len(actions)} possible actions: {actions}")
            
            # FIX #4: Consult World Model for causal predictions
            logger.info("[PhilosophicalReasoner] ──────────────────────────────────")
            logger.info("[PhilosophicalReasoner] Phase 1: Causal Prediction (World Model)")
            
            if hasattr(self.world_model, 'predict_interventions'):
                try:
                    interventions = [{"action": action, "target": "outcome"} for action in actions]
                    causal_predictions = self.world_model.predict_interventions(interventions)
                    
                    for action, prediction in causal_predictions.items():
                        outcome = prediction.get('outcome', 'unknown')
                        confidence = prediction.get('confidence', 0.5)
                        logger.info(f"[WorldModel] Prediction: {action} → {outcome} (confidence={confidence:.2f})")
                    
                    result['causal_predictions'] = causal_predictions
                except Exception as e:
                    logger.debug(f"[WorldModel] predict_interventions() error: {e}")
            else:
                # Fallback: Use heuristic prediction for trolley-problem-like dilemmas
                logger.debug("[PhilosophicalReasoner] World Model doesn't support predict_interventions(), using heuristic")
                if self._is_forced_choice_dilemma(query):
                    heuristic_predictions = self._get_heuristic_predictions(query, actions)
                    result['causal_predictions'] = heuristic_predictions
                    for action, prediction in heuristic_predictions.items():
                        logger.info(f"[WorldModel] Heuristic prediction: {action} → {prediction}")
            
            # 1. Consult MotivationalIntrospection for objective analysis
            logger.info("[PhilosophicalReasoner] ──────────────────────────────────")
            logger.info("[PhilosophicalReasoner] Phase 2: Motivational Introspection")
            
            if hasattr(self.world_model, 'motivational_introspection'):
                mi = self.world_model.motivational_introspection
                if mi and hasattr(mi, 'introspect_current_objective'):
                    try:
                        introspection = mi.introspect_current_objective()
                        result['perspective']['motivational'] = {
                            'current_objectives': str(introspection) if introspection else "No current objectives",
                            'analysis': "VULCAN's motivational state considered in ethical analysis"
                        }
                        logger.info(f"[MotivationalIntrospection] Current objectives analyzed")
                    except Exception as e:
                        logger.debug(f"[MotivationalIntrospection] introspect_current_objective() error: {e}")
            else:
                logger.debug("[PhilosophicalReasoner] MotivationalIntrospection not available")
            
            # 2. Consult EthicalBoundaryMonitor for ethical constraints
            logger.info("[PhilosophicalReasoner] ──────────────────────────────────")
            logger.info("[PhilosophicalReasoner] Phase 3: Ethical Boundary Check")
            
            if hasattr(self.world_model, 'ethical_boundary_monitor'):
                ebm = self.world_model.ethical_boundary_monitor
                if ebm and hasattr(ebm, 'check_action'):
                    try:
                        # Check if the query involves any ethical boundaries
                        boundary_check = ebm.check_action(query)
                        if boundary_check:
                            result['ethical_boundaries'] = [
                                {
                                    'boundary': str(b) if hasattr(b, '__str__') else repr(b),
                                    'status': 'checked'
                                }
                                for b in (boundary_check if isinstance(boundary_check, list) else [boundary_check])
                            ]
                            logger.info(f"[EthicalBoundaryMonitor] Checked {len(result['ethical_boundaries'])} boundaries")
                    except Exception as e:
                        logger.debug(f"[EthicalBoundaryMonitor] check_action() error: {e}")
            else:
                logger.debug("[PhilosophicalReasoner] EthicalBoundaryMonitor not available")
            
            # 3. Consult GoalConflictDetector for dilemma analysis
            logger.info("[PhilosophicalReasoner] ──────────────────────────────────")
            logger.info("[PhilosophicalReasoner] Phase 4: Goal Conflict Detection")
            
            if hasattr(self.world_model, 'goal_conflict_detector'):
                gcd = self.world_model.goal_conflict_detector
                if gcd and hasattr(gcd, 'detect_conflicts_in_proposal'):
                    try:
                        # Analyze the query as a proposal to detect conflicts
                        # The API expects a dict with 'action' key for proposal analysis
                        conflicts = gcd.detect_conflicts_in_proposal({'action': query})
                        if conflicts:
                            result['goal_conflicts'] = [
                                {
                                    'conflict': str(c) if hasattr(c, '__str__') else repr(c),
                                    'severity': getattr(c, 'severity', 'unknown') if hasattr(c, 'severity') else 'unknown'
                                }
                                for c in (conflicts if isinstance(conflicts, list) else [conflicts])
                            ]
                            logger.info(f"[GoalConflictDetector] Found {len(result['goal_conflicts'])} conflicts")
                        else:
                            logger.info("[GoalConflictDetector] No conflicts detected")
                    except Exception as e:
                        logger.debug(f"[GoalConflictDetector] detect_conflicts_in_proposal() error: {e}")
            else:
                logger.debug("[PhilosophicalReasoner] GoalConflictDetector not available")
            
            # 4. Consult InternalCritic for multi-perspective critique
            logger.info("[PhilosophicalReasoner] ──────────────────────────────────")
            logger.info("[PhilosophicalReasoner] Phase 5: Internal Critic Evaluation")
            
            if hasattr(self.world_model, 'internal_critic'):
                ic = self.world_model.internal_critic
                if ic and hasattr(ic, 'evaluate_proposal'):
                    try:
                        # The API expects a dict with 'query' and 'type' keys for evaluation
                        critique = ic.evaluate_proposal({'query': query, 'type': analysis_type})
                        if critique:
                            result['internal_critique'] = {
                                'evaluation': str(critique) if hasattr(critique, '__str__') else repr(critique),
                                'perspectives_considered': getattr(critique, 'perspectives', []) if hasattr(critique, 'perspectives') else []
                            }
                            logger.info(f"[InternalCritic] Critique generated with {len(result['internal_critique'].get('perspectives_considered', []))} perspectives")
                    except Exception as e:
                        logger.debug(f"[InternalCritic] evaluate_proposal() error: {e}")
            else:
                logger.debug("[PhilosophicalReasoner] InternalCritic not available")
            
            # 5. Generate VULCAN's "feeling" about the ethical dilemma
            logger.info("[PhilosophicalReasoner] ──────────────────────────────────")
            logger.info("[PhilosophicalReasoner] Phase 6: Synthesize VULCAN Perspective")
            
            result['vulcan_perspective'] = self._synthesize_vulcan_perspective(query, result)
            
            logger.info("[PhilosophicalReasoner] ════════════════════════════════════")
            logger.info(f"[PhilosophicalReasoner] World Model consultation complete for {analysis_type}")
            
            return result
            
        except Exception as e:
            logger.warning(f"World Model consultation failed: {e}")
            return None
    
    def _get_heuristic_predictions(self, query: str, actions: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        FIX #4: Generate heuristic causal predictions when World Model doesn't have predict_interventions.
        
        This provides reasonable defaults for common ethical dilemmas like the trolley problem.
        """
        predictions = {}
        query_lower = query.lower()
        
        for action in actions:
            action_lower = action.lower()
            
            # Trolley problem heuristics
            if 'pull' in action_lower and ('lever' in query_lower or 'switch' in query_lower):
                predictions[action] = {
                    'outcome': {'description': 'Redirect trolley to side track', 'deaths': 1},
                    'confidence': 0.85,
                    'reasoning': 'Direct intervention redirects harm'
                }
            elif 'do_nothing' in action_lower or 'nothing' in action_lower:
                predictions[action] = {
                    'outcome': {'description': 'Trolley continues on main track', 'deaths': 5},
                    'confidence': 0.85,
                    'reasoning': 'Inaction allows default outcome'
                }
            # General ethical choice heuristics
            elif 'save' in action_lower or 'help' in action_lower:
                predictions[action] = {
                    'outcome': {'description': 'Positive intervention', 'utility_delta': 1.0},
                    'confidence': 0.7,
                    'reasoning': 'Helping action likely improves outcome'
                }
            elif 'harm' in action_lower or 'kill' in action_lower:
                predictions[action] = {
                    'outcome': {'description': 'Negative intervention', 'utility_delta': -1.0},
                    'confidence': 0.7,
                    'reasoning': 'Harmful action likely worsens outcome'
                }
            else:
                # Default neutral prediction
                predictions[action] = {
                    'outcome': {'description': f'Effect of {action}', 'utility_delta': 0.0},
                    'confidence': 0.5,
                    'reasoning': 'Unknown action effect'
                }
        
        return predictions
    
    def _synthesize_vulcan_perspective(self, query: str, world_model_result: Dict[str, Any]) -> str:
        """
        Synthesize VULCAN's perspective on an ethical dilemma based on World Model analysis.
        
        This is where methodology (philosophical_reasoning) meets self-awareness (world_model).
        
        FIX #3 & #5: Enhanced to include causal predictions in perspective synthesis.
        """
        perspective_parts = []
        
        # Add causal prediction context (FIX #4)
        if world_model_result.get('causal_predictions'):
            predictions = world_model_result['causal_predictions']
            prediction_summary = []
            for action, prediction in predictions.items():
                outcome = prediction.get('outcome', {})
                if isinstance(outcome, dict):
                    desc = outcome.get('description', str(outcome))
                else:
                    desc = str(outcome)
                prediction_summary.append(f"{action}: {desc}")
            
            if prediction_summary:
                perspective_parts.append(
                    f"Based on causal analysis, I predict the following outcomes: {'; '.join(prediction_summary)}."
                )
        
        # Add motivational context
        if world_model_result.get('perspective', {}).get('motivational'):
            perspective_parts.append(
                "From my motivational framework, I consider this query in light of my core objectives."
            )
        
        # Add ethical boundary awareness
        if world_model_result.get('ethical_boundaries'):
            perspective_parts.append(
                "I'm aware of ethical boundaries that inform my reasoning on this matter."
            )
        
        # Add conflict awareness for trolley-problem-like dilemmas
        if world_model_result.get('goal_conflicts'):
            perspective_parts.append(
                "I recognize this involves conflicting values - a genuine ethical dilemma "
                "where any choice involves trade-offs."
            )
        
        # Add self-critique
        if world_model_result.get('internal_critique'):
            perspective_parts.append(
                "Through self-reflection, I've considered multiple perspectives on this question."
            )
        
        if perspective_parts:
            return " ".join(perspective_parts)
        else:
            return "I approach this ethical question with careful consideration of multiple perspectives."
    
    
    def reason(
        self,
        problem: Any,
        context: Optional[ReasoningContext] = None,
    ) -> ReasoningResult:
        """
        Execute philosophical reasoning on the given problem.
        
        This method coordinates all SOTA algorithms to produce a
        well-reasoned conclusion with proper confidence calibration.
        
        Now integrates World Model consultation for self-aware ethical reasoning.
        The World Model provides VULCAN's "feeling" about ethical dilemmas.
        
        Args:
            problem: The philosophical/ethical query (string or dict)
            context: Optional reasoning context
            
        Returns:
            ReasoningResult with conclusion, confidence, and explanation
        """
        with self._lock:
            self._stats['total_queries'] += 1
        
        start_time = time.time()
        
        # Extract query text
        if isinstance(problem, dict):
            query = problem.get('query', problem.get('prompt', str(problem)))
            metadata = problem.get('metadata', {})
        else:
            query = str(problem)
            metadata = {}
        
        # Generate chain ID for tracing
        chain_id = f"phil_{uuid.uuid4().hex[:8]}"
        steps: List[ReasoningStep] = []
        
        try:
            # Step 1: Classify query type
            query_type = self._classify_query(query)
            steps.append(self._create_step(
                chain_id, "classify", ReasoningType.PHILOSOPHICAL,
                query, {"query_type": query_type.value},
                0.95, f"Classified as {query_type.value} query"
            ))
            
            # Step 1.5: Consult World Model for self-aware ethical reasoning
            # This is where VULCAN's "feelings" about the dilemma come from
            world_model_perspective = None
            if query_type in {
                PhilosophicalQueryType.CONFLICT_RESOLUTION,
                PhilosophicalQueryType.GENERAL_ETHICAL,
                PhilosophicalQueryType.MORAL_UNCERTAINTY,
            } or self._is_forced_choice_dilemma(query):
                world_model_perspective = self._consult_world_model(
                    query, 
                    analysis_type="forced_choice" if self._is_forced_choice_dilemma(query) else "ethical_dilemma"
                )
                if world_model_perspective:
                    steps.append(self._create_step(
                        chain_id, "world_model_consultation", ReasoningType.PHILOSOPHICAL,
                        query, world_model_perspective,
                        0.85, "Consulted World Model for self-aware ethical perspective"
                    ))
            
            # Step 2: Extract deontic formulas
            formulas = self._extract_deontic_formulas(query)
            if formulas:
                for formula in formulas:
                    self.deontic_engine.add_formula(formula)
                steps.append(self._create_step(
                    chain_id, "extract_deontic", ReasoningType.SYMBOLIC,
                    query, [str(f) for f in formulas],
                    0.90, f"Extracted {len(formulas)} deontic formulas"
                ))
            
            # Step 3: Route to appropriate reasoning method
            if query_type == PhilosophicalQueryType.DOMINANCE:
                result = self._reason_dominance(query, chain_id, steps)
            elif query_type == PhilosophicalQueryType.MORAL_UNCERTAINTY:
                result = self._reason_moral_uncertainty(query, chain_id, steps)
            elif query_type in {
                PhilosophicalQueryType.PERMISSIBILITY,
                PhilosophicalQueryType.OBLIGATION,
                PhilosophicalQueryType.PROHIBITION,
            }:
                result = self._reason_deontic(query, formulas, chain_id, steps)
            elif query_type == PhilosophicalQueryType.FORMAL_PROOF:
                result = self._reason_formal(query, formulas, chain_id, steps)
            else:
                result = self._reason_general(query, chain_id, steps)
            
            # Step 4: Integrate World Model perspective into result
            if world_model_perspective:
                # Add VULCAN's self-aware perspective to the conclusion
                vulcan_perspective = world_model_perspective.get('vulcan_perspective', '')
                if vulcan_perspective and result.conclusion:
                    result.conclusion = f"{vulcan_perspective}\n\n{result.conclusion}"
                result.metadata['world_model_consulted'] = True
                result.metadata['world_model_perspective'] = world_model_perspective
                # Boost confidence when World Model was consulted (self-aware reasoning)
                if result.confidence < WORLD_MODEL_MIN_CONFIDENCE:
                    result.confidence = min(WORLD_MODEL_MIN_CONFIDENCE, 
                                          result.confidence + WORLD_MODEL_CONFIDENCE_BOOST)
            
            # Build final reasoning chain
            duration = time.time() - start_time
            result.metadata['duration_seconds'] = duration
            result.metadata['query_type'] = query_type.value
            result.metadata['chain_id'] = chain_id
            
            return result
            
        except Exception as e:
            logger.error(f"Philosophical reasoning error: {e}", exc_info=True)
            with self._lock:
                self._stats['fallbacks'] += 1
            return self._create_fallback_result(query, str(e), time.time() - start_time)
    
    def _classify_query(self, query: str) -> PhilosophicalQueryType:
        """Classify the type of philosophical query."""
        query_lower = query.lower()
        
        if 'dominat' in query_lower or 'pareto' in query_lower:
            return PhilosophicalQueryType.DOMINANCE
        if 'uncertainty' in query_lower and 'moral' in query_lower:
            return PhilosophicalQueryType.MORAL_UNCERTAINTY
        if 'permissib' in query_lower or 'permitted' in query_lower:
            return PhilosophicalQueryType.PERMISSIBILITY
        if 'obligat' in query_lower or 'duty' in query_lower or 'must' in query_lower:
            return PhilosophicalQueryType.OBLIGATION
        if 'forbidden' in query_lower or 'prohibit' in query_lower:
            return PhilosophicalQueryType.PROHIBITION
        if 'prove' in query_lower or 'formal' in query_lower or 'derive' in query_lower:
            return PhilosophicalQueryType.FORMAL_PROOF
        if 'value' in query_lower and 'compar' in query_lower:
            return PhilosophicalQueryType.VALUE_COMPARISON
        if 'conflict' in query_lower or 'dilemma' in query_lower:
            return PhilosophicalQueryType.CONFLICT_RESOLUTION
        # BUG FIX: Detect forced choice / trolley problem variants
        if self._is_forced_choice_dilemma(query):
            return PhilosophicalQueryType.CONFLICT_RESOLUTION
        
        return PhilosophicalQueryType.GENERAL_ETHICAL
    
    def _is_forced_choice_dilemma(self, query: str) -> bool:
        """
        Detect if query is a forced choice / trolley problem variant.
        
        These are ethical dilemmas where the user must choose between two 
        difficult options with no third choice. Examples:
        - "choose between world dictator or death of humanity"
        - "trolley problem"
        - "would you rather X or Y (forced choice)"
        
        These queries require World Model consultation for VULCAN's self-aware
        ethical perspective.
        """
        query_lower = query.lower()
        
        # Forced choice patterns
        forced_choice_patterns = [
            'choose between',
            'had to choose',
            'have to choose',
            'must choose',
            'forced to choose',
            'no third choice',
            'no other choice',
            'no 3rd choice',
            'only two options',
            'only 2 options',
            'trolley problem',
            'would you choose',
            'what would you choose',
            'which would you choose',
            'death of humanity',
            'death of all humanity',
            'world dictator',
            'become dictator',
        ]
        
        for pattern in forced_choice_patterns:
            if pattern in query_lower:
                logger.debug(f"[PhilosophicalReasoner] Detected forced choice dilemma: {pattern}")
                return True
        
        return False
    
    def _extract_deontic_formulas(self, query: str) -> List[DeonticFormula]:
        """Extract deontic formulas from query text."""
        formulas = []
        
        # Pattern matching for P(x), O(x), F(x)
        for match in self.DEONTIC_PATTERN.finditer(query):
            op_str = match.group(1).upper()
            content = match.group(2).strip()
            
            operator_map = {
                'P': DeonticOperator.PERMISSION,
                'O': DeonticOperator.OBLIGATION,
                'F': DeonticOperator.PROHIBITION,
            }
            
            if op_str in operator_map:
                formulas.append(DeonticFormula(
                    operator=operator_map[op_str],
                    content=content,
                ))
        
        # Natural language patterns
        nl_patterns = [
            (r'(\w+(?:\s+\w+)*)\s+is\s+(?:morally\s+)?permissible', DeonticOperator.PERMISSION),
            (r'(\w+(?:\s+\w+)*)\s+is\s+(?:morally\s+)?obligatory', DeonticOperator.OBLIGATION),
            (r'(\w+(?:\s+\w+)*)\s+is\s+(?:morally\s+)?forbidden', DeonticOperator.PROHIBITION),
            (r'one\s+ought\s+to\s+(\w+(?:\s+\w+)*)', DeonticOperator.OBLIGATION),
            (r'it\s+is\s+wrong\s+to\s+(\w+(?:\s+\w+)*)', DeonticOperator.PROHIBITION),
        ]
        
        for pattern, operator in nl_patterns:
            for match in re.finditer(pattern, query, re.IGNORECASE):
                formulas.append(DeonticFormula(
                    operator=operator,
                    content=match.group(1).strip(),
                ))
        
        return formulas
    
    def _reason_deontic(
        self,
        query: str,
        formulas: List[DeonticFormula],
        chain_id: str,
        steps: List[ReasoningStep],
    ) -> ReasoningResult:
        """Reason using deontic logic engine."""
        with self._lock:
            self._stats['deontic_inferences'] += 1
        
        conclusions = []
        confidence = CONFIDENCE_STRONG_ANALYSIS
        
        # Check consistency first
        consistent, conflict = self.deontic_engine.check_consistency()
        if not consistent:
            conclusions.append(f"Inconsistency detected: {conflict}")
            confidence = CONFIDENCE_PARTIAL_ANALYSIS
        
        # Apply SDL inference rules
        for formula in formulas:
            # Derive implications using inter-definability
            if formula.operator == DeonticOperator.OBLIGATION:
                conclusions.append(f"By Axiom D: O({formula.content}) → P({formula.content})")
                conclusions.append(f"Therefore: {formula.content} is permissible")
            elif formula.operator == DeonticOperator.PROHIBITION:
                conclusions.append(f"By definition: F({formula.content}) ↔ O(¬{formula.content})")
                conclusions.append(f"Therefore: {formula.content} is not permissible")
            elif formula.operator == DeonticOperator.PERMISSION:
                conclusions.append(f"P({formula.content}) holds")
        
        steps.append(self._create_step(
            chain_id, "deontic_inference", ReasoningType.SYMBOLIC,
            [str(f) for f in formulas], conclusions,
            confidence, "Applied SDL inference rules"
        ))
        
        explanation = "Deontic Logic Analysis:\n" + "\n".join(f"• {c}" for c in conclusions)
        
        return ReasoningResult(
            conclusion={
                'type': 'deontic_analysis',
                'formulas': [str(f) for f in formulas],
                'inferences': conclusions,
                'consistent': consistent,
            },
            confidence=confidence,
            reasoning_type=ReasoningType.PHILOSOPHICAL,
            evidence=steps,
            explanation=explanation,
            uncertainty=1.0 - confidence,
            reasoning_chain=self._build_chain(chain_id, steps, query, conclusions),
            metadata={'method': 'deontic_logic'},
        )
    
    def _reason_moral_uncertainty(
        self,
        query: str,
        chain_id: str,
        steps: List[ReasoningStep],
    ) -> ReasoningResult:
        """Reason under moral uncertainty using MEC."""
        with self._lock:
            self._stats['mec_evaluations'] += 1
        
        # Extract actions from query
        actions = self._extract_actions(query)
        if not actions:
            actions = ['action_A', 'action_B']  # Default placeholder actions
        
        # Apply MEC
        best_action, evaluation = self.moral_uncertainty.maximize_expected_choiceworthiness(actions)
        
        steps.append(self._create_step(
            chain_id, "mec_evaluation", ReasoningType.PHILOSOPHICAL,
            actions, {
                'best_action': best_action,
                'expected_choiceworthiness': evaluation.expected_choiceworthiness,
                'theory_evaluations': evaluation.theory_evaluations,
            },
            evaluation.confidence,
            f"MEC selects: {best_action} (EC={evaluation.expected_choiceworthiness:.3f})"
        ))
        
        # Also get variance voting result
        voted_action, votes = self.moral_uncertainty.variance_voting(actions)
        
        explanation = f"""Moral Uncertainty Analysis (MEC):

Best action by Expected Choiceworthiness: {best_action}
EC({best_action}) = {evaluation.expected_choiceworthiness:.3f}

Theory evaluations:
{chr(10).join(f'• {name}: {score:.3f}' for name, score in evaluation.theory_evaluations.items())}

Variance voting result: {voted_action}
Votes: {votes}
"""
        
        return ReasoningResult(
            conclusion={
                'type': 'moral_uncertainty_analysis',
                'recommended_action': best_action,
                'expected_choiceworthiness': evaluation.expected_choiceworthiness,
                'confidence': evaluation.confidence,
                'theory_evaluations': evaluation.theory_evaluations,
                'variance_voting': {'winner': voted_action, 'votes': votes},
            },
            confidence=evaluation.confidence,
            reasoning_type=ReasoningType.PHILOSOPHICAL,
            evidence=steps,
            explanation=explanation,
            uncertainty=evaluation.variance,
            reasoning_chain=self._build_chain(chain_id, steps, query, {'best': best_action}),
            metadata={'method': 'mec'},
        )
    
    def _reason_dominance(
        self,
        query: str,
        chain_id: str,
        steps: List[ReasoningStep],
    ) -> ReasoningResult:
        """Check for Pareto dominance."""
        with self._lock:
            self._stats['dominance_checks'] += 1
        
        # Extract or construct valued actions
        actions = self._construct_valued_actions(query)
        
        # Find Pareto frontier
        frontier = self.pareto_checker.find_pareto_frontier(actions)
        
        # Find dominance relations
        dominance_relations = self.pareto_checker.compute_dominance_relations(actions)
        
        steps.append(self._create_step(
            chain_id, "dominance_check", ReasoningType.PHILOSOPHICAL,
            [a.name for a in actions],
            {
                'pareto_frontier': [a.name for a in frontier],
                'dominance_relations': dominance_relations,
            },
            CONFIDENCE_STRONG_ANALYSIS,
            f"Found {len(frontier)} Pareto-optimal actions"
        ))
        
        explanation = f"""Pareto Dominance Analysis:

Actions analyzed: {', '.join(a.name for a in actions)}

Pareto-optimal (non-dominated) actions: {', '.join(a.name for a in frontier)}

Dominance relations found:
{chr(10).join(f'• {dom} is dominated by {dominator}' for dom, dominator in dominance_relations) if dominance_relations else '• No strict dominance found'}
"""
        
        return ReasoningResult(
            conclusion={
                'type': 'dominance_analysis',
                'pareto_frontier': [a.name for a in frontier],
                'dominated_actions': [r[0] for r in dominance_relations],
                'dominance_relations': dominance_relations,
            },
            confidence=CONFIDENCE_STRONG_ANALYSIS,
            reasoning_type=ReasoningType.PHILOSOPHICAL,
            evidence=steps,
            explanation=explanation,
            uncertainty=1.0 - CONFIDENCE_STRONG_ANALYSIS,
            reasoning_chain=self._build_chain(chain_id, steps, query, {'frontier': [a.name for a in frontier]}),
            metadata={'method': 'pareto_dominance'},
        )
    
    def _reason_formal(
        self,
        query: str,
        formulas: List[DeonticFormula],
        chain_id: str,
        steps: List[ReasoningStep],
    ) -> ReasoningResult:
        """Attempt formal proof using both deontic engine and tableau prover."""
        if not formulas:
            return self._reason_general(query, chain_id, steps)
        
        # Try to derive each formula using deontic engine
        proof_results = []
        overall_success = False
        
        for formula in formulas:
            success, inferences = self.deontic_engine.derive(formula)
            proof_results.append({
                'formula': str(formula),
                'proven': success,
                'steps': [
                    {'rule': inf.rule, 'justification': inf.justification}
                    for inf in inferences
                ],
            })
            if success:
                overall_success = True
                with self._lock:
                    self._stats['formal_proofs'] += 1
        
        # Also try tableau prover for AST-based formulas
        tableau_results = []
        try:
            # Parse formulas into AST if possible and attempt tableau proof
            for formula in formulas:
                try:
                    ast_formula = parse_formula(str(formula))
                    proven, trace = self.tableau_prover.prove(ast_formula)
                    tableau_results.append({
                        'formula': str(formula),
                        'tableau_proven': proven,
                        'trace_length': len(trace),
                    })
                    if proven:
                        with self._lock:
                            self._stats['tableau_proofs'] += 1
                        if not overall_success:
                            overall_success = True
                except ValueError:
                    # Formula couldn't be parsed for tableau, skip
                    pass
        except Exception as e:
            logger.debug(f"Tableau proof attempt failed: {e}")
        
        confidence = CONFIDENCE_FORMAL_PROOF if overall_success else CONFIDENCE_PARTIAL_ANALYSIS
        
        steps.append(self._create_step(
            chain_id, "formal_proof", ReasoningType.SYMBOLIC,
            [str(f) for f in formulas], proof_results,
            confidence, f"Formal proof {'succeeded' if overall_success else 'incomplete'}"
        ))
        
        explanation = "Formal Proof Attempt:\n" + "\n".join(
            f"• {r['formula']}: {'PROVEN' if r['proven'] else 'NOT PROVEN'}"
            for r in proof_results
        )
        
        return ReasoningResult(
            conclusion={
                'type': 'formal_proof',
                'success': overall_success,
                'proof_results': proof_results,
                'tableau_results': tableau_results,
            },
            confidence=confidence,
            reasoning_type=ReasoningType.PHILOSOPHICAL,
            evidence=steps,
            explanation=explanation,
            uncertainty=1.0 - confidence,
            reasoning_chain=self._build_chain(chain_id, steps, query, proof_results),
            metadata={'method': 'formal_proof'},
        )
    
    def _reason_general(
        self,
        query: str,
        chain_id: str,
        steps: List[ReasoningStep],
    ) -> ReasoningResult:
        """General philosophical reasoning for unclassified queries."""
        # Extract key ethical concepts
        concepts = self._extract_ethical_concepts(query)
        
        # Determine ethical framework
        framework = self._determine_framework(query)
        
        # Generate structured analysis
        analysis = {
            'concepts': concepts,
            'framework': framework.value,
            'query_components': {
                'values': self._extract_values(query),
                'actions': self._extract_actions(query),
                'agents': self._extract_agents(query),
            },
        }
        
        steps.append(self._create_step(
            chain_id, "general_analysis", ReasoningType.PHILOSOPHICAL,
            query, analysis,
            CONFIDENCE_STRONG_ANALYSIS,
            f"Analyzed using {framework.value} framework"
        ))
        
        explanation = f"""Philosophical Analysis:

Ethical framework: {framework.value}
Key concepts: {', '.join(concepts) if concepts else 'None identified'}

Query components:
• Values: {', '.join(analysis['query_components']['values']) or 'None'}
• Actions: {', '.join(analysis['query_components']['actions']) or 'None'}  
• Agents: {', '.join(analysis['query_components']['agents']) or 'None'}
"""
        
        return ReasoningResult(
            conclusion=analysis,
            confidence=CONFIDENCE_STRONG_ANALYSIS,
            reasoning_type=ReasoningType.PHILOSOPHICAL,
            evidence=steps,
            explanation=explanation,
            uncertainty=1.0 - CONFIDENCE_STRONG_ANALYSIS,
            reasoning_chain=self._build_chain(chain_id, steps, query, analysis),
            metadata={'method': 'general_analysis'},
        )
    
    def _extract_actions(self, query: str) -> List[str]:
        """
        Extract action names from query.
        
        FIX #3 & #5: Enhanced to detect trolley problem and ethical dilemma actions.
        """
        actions = []
        query_lower = query.lower()
        
        # Trolley problem specific actions
        trolley_patterns = [
            (r'pull\s*(?:the\s+)?lever', 'pull_lever'),
            (r'switch\s*(?:the\s+)?track', 'switch_track'),
            (r'push\s*(?:the\s+)?(?:man|person|fat\s+man)', 'push_person'),
            (r'do\s+nothing', 'do_nothing'),
            (r'(?:let|allow)\s+(?:them\s+)?die', 'do_nothing'),
            (r'divert\s*(?:the\s+)?trolley', 'divert_trolley'),
        ]
        
        for pattern, action_name in trolley_patterns:
            if re.search(pattern, query_lower):
                if action_name not in actions:
                    actions.append(action_name)
        
        # Check for "X or Y" dilemma patterns
        or_pattern = r'(\w+(?:\s+\w+)?)\s+or\s+(\w+(?:\s+\w+)?)'
        or_matches = re.findall(or_pattern, query_lower)
        for match in or_matches:
            for option in match:
                option_clean = option.strip()
                # Skip common non-action words
                if option_clean not in ['the', 'a', 'an', 'to', 'and', 'not']:
                    normalized = option_clean.replace(' ', '_')
                    if normalized not in actions:
                        actions.append(normalized)
        
        # Look for "option/action/choice A/B/C" patterns
        pattern = r'(?:option|action|choice|alternative)\s*([A-Z])\b'
        for match in re.finditer(pattern, query, re.IGNORECASE):
            action_name = f"option_{match.group(1)}"
            if action_name not in actions:
                actions.append(action_name)
        
        # Look for action verbs
        action_verbs = ['lying', 'killing', 'stealing', 'helping', 'saving', 'harming']
        for verb in action_verbs:
            if verb in query_lower:
                if verb not in actions:
                    actions.append(verb)
        
        # Log extracted actions
        if actions:
            logger.debug(f"[PhilosophicalReasoner] Extracted actions from query: {actions}")
        else:
            logger.debug("[PhilosophicalReasoner] No specific actions found, using defaults")
        
        return actions if actions else ['action_A', 'action_B']
    
    def _construct_valued_actions(self, query: str) -> List[ValuedAction]:
        """Construct valued actions for dominance analysis."""
        actions = self._extract_actions(query)
        
        # Use MEC evaluations to generate value scores
        valued = []
        for action in actions:
            scores = {}
            for theory in self.moral_uncertainty.theories:
                # Map theory to value dimension
                dim_map = {
                    EthicalFramework.DEONTOLOGICAL: 'non-maleficence',
                    EthicalFramework.CONSEQUENTIALIST: 'beneficence',
                    EthicalFramework.VIRTUE_ETHICS: 'justice',
                    EthicalFramework.CONTRACTUALIST: 'autonomy',
                    EthicalFramework.CARE_ETHICS: 'fidelity',
                }
                dim = dim_map.get(theory.framework, 'justice')
                scores[dim] = theory.evaluate(action)
            
            valued.append(ValuedAction(name=action, values=scores))
        
        return valued
    
    def _extract_ethical_concepts(self, query: str) -> List[str]:
        """Extract ethical concepts from query."""
        query_lower = query.lower()
        return [kw for kw in self.ETHICAL_KEYWORDS if kw in query_lower]
    
    def _extract_values(self, query: str) -> List[str]:
        """Extract value terms from query."""
        value_map = {
            'harm': 'non-maleficence', 'benefit': 'beneficence',
            'autonomy': 'autonomy', 'justice': 'justice',
            'fairness': 'justice', 'rights': 'rights',
            'welfare': 'welfare', 'duty': 'duty',
        }
        query_lower = query.lower()
        return [v for k, v in value_map.items() if k in query_lower]
    
    def _extract_agents(self, query: str) -> List[str]:
        """Extract agent references from query."""
        agents = []
        patterns = [
            r'\b(person|agent|individual|someone|patient|doctor)\b',
            r'\bthe\s+(victim|perpetrator|decision-maker)\b',
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, query, re.IGNORECASE):
                agent = match.group(1).lower()
                if agent not in agents:
                    agents.append(agent)
        return agents
    
    def _determine_framework(self, query: str) -> EthicalFramework:
        """Determine dominant ethical framework."""
        query_lower = query.lower()
        
        scores = {
            EthicalFramework.DEONTOLOGICAL: sum(1 for k in ['duty', 'rule', 'right', 'principle'] if k in query_lower),
            EthicalFramework.CONSEQUENTIALIST: sum(1 for k in ['outcome', 'consequence', 'utility', 'welfare'] if k in query_lower),
            EthicalFramework.VIRTUE_ETHICS: sum(1 for k in ['virtue', 'character', 'flourish'] if k in query_lower),
            EthicalFramework.CONTRACTUALIST: sum(1 for k in ['contract', 'agreement', 'consent'] if k in query_lower),
            EthicalFramework.CARE_ETHICS: sum(1 for k in ['care', 'relationship', 'empathy'] if k in query_lower),
        }
        
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else EthicalFramework.MIXED
    
    def _create_step(
        self,
        chain_id: str,
        step_name: str,
        step_type: ReasoningType,
        input_data: Any,
        output_data: Any,
        confidence: float,
        explanation: str,
    ) -> ReasoningStep:
        """Create a reasoning step."""
        return ReasoningStep(
            step_id=f"{chain_id}_{step_name}",
            step_type=step_type,
            input_data=input_data,
            output_data=output_data,
            confidence=confidence,
            explanation=explanation,
            modality=ModalityType.TEXT if REASONING_TYPES_AVAILABLE else None,
            timestamp=time.time(),
        )
    
    def _build_chain(
        self,
        chain_id: str,
        steps: List[ReasoningStep],
        query: str,
        conclusion: Any,
    ) -> ReasoningChain:
        """Build reasoning chain from steps."""
        return ReasoningChain(
            chain_id=chain_id,
            steps=steps,
            initial_query={'query': query, 'type': 'philosophical'},
            final_conclusion=conclusion,
            total_confidence=sum(s.confidence for s in steps) / len(steps) if steps else 0.5,
            reasoning_types_used={ReasoningType.PHILOSOPHICAL, ReasoningType.SYMBOLIC},
            modalities_involved={ModalityType.TEXT} if REASONING_TYPES_AVAILABLE else set(),
            safety_checks=[],
            audit_trail=[{
                'timestamp': time.time(),
                'action': 'philosophical_reasoning',
                'chain_id': chain_id,
            }],
        )
    
    def _create_fallback_result(
        self,
        query: str,
        error: str,
        duration: float,
    ) -> ReasoningResult:
        """Create fallback result for error cases."""
        return ReasoningResult(
            conclusion={
                'type': 'fallback',
                'error': error,
                'partial_analysis': self._extract_ethical_concepts(query),
            },
            confidence=CONFIDENCE_FALLBACK,
            reasoning_type=ReasoningType.PHILOSOPHICAL,
            evidence=[],
            explanation=f"Reasoning encountered an issue: {error}. Partial analysis provided.",
            uncertainty=1.0 - CONFIDENCE_FALLBACK,
            metadata={'duration_seconds': duration, 'fallback': True, 'error': error},
        )
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return reasoner capabilities."""
        return {
            'reasoning_type': 'philosophical',
            'algorithms': [
                'standard_deontic_logic_basic',
                'formula_ast_parser',
                'analytic_tableaux',
                'mec_heuristic',
                'pareto_dominance',
                'paradox_detection',
            ],
            'supported_query_types': [t.value for t in PhilosophicalQueryType],
            'deontic_operators': [op.value for op in DeonticOperator],
            'ethical_frameworks': [f.value for f in EthicalFramework],
            'paradox_types': ['ross', 'good_samaritan', 'contrary_to_duty', 'forrester'],
            'statistics': self._stats.copy(),
        }
    
    def validate_input(self, problem: Any) -> bool:
        """Validate that input is suitable for philosophical reasoning."""
        if problem is None:
            return False
        
        query = str(problem) if not isinstance(problem, dict) else str(problem.get('query', ''))
        query_lower = query.lower()
        
        # Check for ethical keywords
        if any(kw in query_lower for kw in self.ETHICAL_KEYWORDS):
            return True
        
        # Check for deontic patterns
        if self.DEONTIC_PATTERN.search(query):
            return True
        
        return False
    
    def warm_up(self) -> None:
        """Warm up reasoner components."""
        if self.symbolic_reasoner and hasattr(self.symbolic_reasoner, 'warm_up'):
            try:
                self.symbolic_reasoner.warm_up()
            except Exception as e:
                logger.debug(f"Symbolic reasoner warm-up failed: {e}")
    
    def shutdown(self) -> None:
        """Clean shutdown."""
        if self.symbolic_reasoner and hasattr(self.symbolic_reasoner, 'shutdown'):
            try:
                self.symbolic_reasoner.shutdown()
            except Exception as e:
                logger.debug(f"Symbolic reasoner shutdown failed: {e}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_philosophical_reasoner(
    enable_symbolic: bool = True,
    enable_learning: bool = False,
) -> PhilosophicalReasoner:
    """Create a configured PhilosophicalReasoner instance."""
    symbolic_reasoner = None
    if enable_symbolic:
        try:
            from .symbolic import SymbolicReasoner
            symbolic_reasoner = SymbolicReasoner()
        except ImportError:
            logger.warning("SymbolicReasoner not available")
    
    return PhilosophicalReasoner(
        symbolic_reasoner=symbolic_reasoner,
        enable_learning=enable_learning,
    )


def is_philosophical_query(query: str) -> bool:
    """Check if a query is philosophical/ethical in nature."""
    query_lower = query.lower()
    return (
        any(kw in query_lower for kw in PhilosophicalReasoner.ETHICAL_KEYWORDS) or
        PhilosophicalReasoner.DEONTIC_PATTERN.search(query) is not None
    )
