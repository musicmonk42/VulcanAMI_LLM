"""
Core data structures for symbolic reasoning.

Includes terms, literals, clauses, unification, and proof nodes.
All core functionality extracted from symbolic_reasoning.py.
"""

from __future__ import annotations

import copy
import logging
from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

logger = logging.getLogger(__name__)


# ============================================================================
# TERMS
# ============================================================================


@dataclass
class Term(ABC):
    """Base class for terms in first-order logic"""

    pass


@dataclass
class Variable(Term):
    """
    Variable term (e.g., X, Y, ?x)

    Variables start with uppercase letters or '?' prefix.
    Used in unification and quantification.
    """

    name: str

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other) -> bool:
        return isinstance(other, Variable) and self.name == other.name

    def __hash__(self) -> int:
        return hash(("var", self.name))


@dataclass
class Constant(Term):
    """
    Constant term (e.g., a, b, socrates)

    Constants are ground terms with fixed values.
    Used to represent specific objects in the domain.
    """

    name: str

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other) -> bool:
        return isinstance(other, Constant) and self.name == other.name

    def __hash__(self) -> int:
        return hash(("const", self.name))


@dataclass
class Function(Term):
    """
    Function term (e.g., f(x), father(john))

    Functions map terms to terms.
    Can be nested: f(g(x), h(a, b))
    """

    name: str
    args: List[Term]

    def __str__(self) -> str:
        args_str = ",".join(str(arg) for arg in self.args)
        return f"{self.name}({args_str})"

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, Function)
            and self.name == other.name
            and self.args == other.args
        )

    def __hash__(self) -> int:
        return hash(("func", self.name, tuple(self.args)))


# ============================================================================
# LITERALS AND CLAUSES
# ============================================================================


@dataclass
class Literal:
    """
    Represents a literal in first-order logic.

    A literal is an atomic formula or its negation:
    - P(x, y) - positive literal
    - ¬P(x, y) - negative literal

    Used in clauses for resolution and other proving methods.
    """

    predicate: str
    terms: List[Term]
    negated: bool = False

    def __str__(self) -> str:
        terms_str = ",".join(str(t) for t in self.terms)
        pred_str = f"{self.predicate}({terms_str})" if self.terms else self.predicate
        return f"¬{pred_str}" if self.negated else pred_str

    def __eq__(self, other) -> bool:
        if not isinstance(other, Literal):
            return False
        return (
            self.predicate == other.predicate
            and self.terms == other.terms
            and self.negated == other.negated
        )

    def __hash__(self) -> int:
        return hash((self.predicate, tuple(self.terms), self.negated))

    def negate(self) -> "Literal":
        """
        Return negated version of this literal.

        Returns:
            New Literal with opposite polarity
        """
        return Literal(
            predicate=self.predicate,
            terms=copy.deepcopy(self.terms),
            negated=not self.negated,
        )


@dataclass
class Clause:
    """
    Represents a clause (disjunction of literals).

    A clause is a disjunction: L₁ ∨ L₂ ∨ ... ∨ Lₙ

    Special cases:
    - Unit clause: single literal
    - Horn clause: at most one positive literal
    - Empty clause (□): represents contradiction

    Used in resolution, tableau, and other proving methods.
    """

    literals: List[Literal]
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_goal: bool = False  # Marks this clause as a goal/query clause

    def is_horn_clause(self) -> bool:
        """
        Check if this is a Horn clause.

        A Horn clause has at most one positive literal.
        Horn clauses are important for efficient reasoning.

        Returns:
            True if Horn clause, False otherwise
        """
        positive_count = sum(1 for lit in self.literals if not lit.negated)
        return positive_count <= 1

    def is_unit_clause(self) -> bool:
        """
        Check if this is a unit clause.

        A unit clause contains exactly one literal.
        Unit clauses are used in unit resolution and propagation.

        Returns:
            True if unit clause, False otherwise
        """
        return len(self.literals) == 1

    def is_empty(self) -> bool:
        """
        Check if this is the empty clause.

        The empty clause (□) represents a contradiction.
        Deriving the empty clause proves unsatisfiability.

        Returns:
            True if empty clause, False otherwise
        """
        return len(self.literals) == 0

    def __str__(self) -> str:
        if self.is_empty():
            return "□"
        return " ∨ ".join(str(lit) for lit in self.literals)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Clause):
            return False
        return set(self.literals) == set(other.literals)

    def __hash__(self) -> int:
        return hash(frozenset(self.literals))


# ============================================================================
# ISSUE #4 FIX: Annotated Clause for Derivation Tracking
# ============================================================================


@dataclass
class AnnotatedClause:
    """
    Clause with derivation history for contradiction proof tracking.
    
    This dataclass extends the basic Clause concept by tracking HOW each clause
    was derived, enabling construction of complete proof trees when a 
    contradiction (empty clause) is found.
    
    Industry-standard approach for proof logging in automated theorem provers.
    Based on proof logging techniques from:
    - OTTER/Prover9 proof logging
    - Vampire proof objects
    - E-prover derivation trees
    
    Attributes:
        clause: The underlying logical clause
        derived_from: List of AnnotatedClauses this was derived from (empty for premises)
        rule_used: Name of the inference rule used ("premise", "resolution", etc.)
        iteration: The iteration/step number when this clause was derived
        resolvent_literal: The literal that was resolved (for resolution steps)
        
    Examples:
        >>> # A premise clause
        >>> premise = AnnotatedClause(
        ...     clause=Clause([Literal("A", [])]),
        ...     derived_from=[],
        ...     rule_used="premise",
        ...     iteration=0
        ... )
        
        >>> # A derived clause from resolution
        >>> derived = AnnotatedClause(
        ...     clause=Clause([]),  # Empty clause = contradiction
        ...     derived_from=[clause1, clause2],
        ...     rule_used="resolution",
        ...     iteration=5,
        ...     resolvent_literal="A"
        ... )
    """
    
    clause: Clause
    derived_from: List["AnnotatedClause"] = field(default_factory=list)
    rule_used: str = "premise"
    iteration: int = 0
    resolvent_literal: Optional[str] = None
    
    def __str__(self) -> str:
        return str(self.clause)
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, AnnotatedClause):
            return False
        return self.clause == other.clause
    
    def __hash__(self) -> int:
        return hash(self.clause)
    
    def is_premise(self) -> bool:
        """Check if this clause is a premise (not derived)."""
        return self.rule_used == "premise"
    
    def is_empty(self) -> bool:
        """Check if this is the empty clause (contradiction)."""
        return self.clause.is_empty()


def build_proof_tree_from_annotated(
    empty_clause: AnnotatedClause,
    max_depth: int = 100
) -> "ProofNode":
    """
    Build a ProofNode tree from an AnnotatedClause derivation.
    
    Recursively traces back through the derivation history to construct
    a complete proof tree showing how the contradiction was reached.
    
    Args:
        empty_clause: The empty clause (contradiction) to trace back from
        max_depth: Maximum recursion depth to prevent infinite loops
        
    Returns:
        ProofNode representing the complete proof tree
        
    Examples:
        >>> proof_tree = build_proof_tree_from_annotated(empty_clause)
        >>> print(proof_tree.to_string())
    """
    def build_recursive(annotated: AnnotatedClause, depth: int) -> "ProofNode":
        if depth > max_depth:
            return ProofNode(
                conclusion=str(annotated.clause),
                premises=[],
                rule_used="max_depth_reached",
                confidence=0.5,
                depth=depth
            )
        
        # Build premise proof nodes recursively
        premise_proofs = []
        for parent in annotated.derived_from:
            premise_proofs.append(build_recursive(parent, depth + 1))
        
        # Create proof node for this clause
        conclusion = str(annotated.clause) if not annotated.is_empty() else "⊥ (contradiction)"
        
        metadata = {"iteration": annotated.iteration}
        if annotated.resolvent_literal:
            metadata["resolved_on"] = annotated.resolvent_literal
        
        return ProofNode(
            conclusion=conclusion,
            premises=premise_proofs,
            rule_used=annotated.rule_used,
            confidence=1.0 if annotated.is_premise() else 0.95,
            depth=depth,
            metadata=metadata
        )
    
    return build_recursive(empty_clause, 0)


def format_contradiction_proof(proof_tree: "ProofNode", max_steps: int = 20) -> str:
    """
    Format a proof tree as a human-readable contradiction explanation.
    
    Converts the recursive proof tree into a numbered list of steps,
    showing how premises lead to a contradiction through resolution.
    
    This follows the standard proof format used in logic textbooks and
    automated theorem provers like Prover9 and Vampire.
    
    Args:
        proof_tree: The ProofNode representing the proof
        max_steps: Maximum number of steps to include
        
    Returns:
        Human-readable string explaining the contradiction
        
    Examples:
        >>> explanation = format_contradiction_proof(proof_tree)
        >>> print(explanation)
        Step 1: A → B (premise)
        Step 2: B → C (premise)
        Step 3: ¬C (premise)
        Step 4: ¬B (from steps 2, 3 via resolution on C)
        Step 5: ¬A (from steps 1, 4 via resolution on B)
        Step 6: A ∨ B (premise)
        Step 7: ⊥ (from steps 5, 6 via resolution - CONTRADICTION)
    """
    steps = []
    step_map = {}  # Map proof nodes to step numbers
    
    def collect_steps(node: "ProofNode", visited: set):
        """Collect all steps in bottom-up order (premises first)."""
        if id(node) in visited:
            return
        visited.add(id(node))
        
        # Process premises first (bottom-up)
        for premise in node.premises:
            collect_steps(premise, visited)
        
        # Add this step
        if len(steps) < max_steps:
            step_num = len(steps) + 1
            step_map[id(node)] = step_num
            steps.append(node)
    
    # Collect all steps
    collect_steps(proof_tree, set())
    
    # Format output
    lines = ["Contradiction Proof:", "=" * 60]
    
    for i, step in enumerate(steps, 1):
        conclusion = step.conclusion
        rule = step.rule_used
        
        # Build premise references
        premise_refs = []
        for premise in step.premises:
            premise_id = id(premise)
            if premise_id in step_map:
                premise_refs.append(str(step_map[premise_id]))
        
        # Format the step
        if rule == "premise":
            lines.append(f"Step {i}: {conclusion} (premise)")
        elif rule == "resolution":
            resolved_on = step.metadata.get("resolved_on", "")
            if premise_refs:
                refs = ", ".join(premise_refs)
                if resolved_on:
                    lines.append(f"Step {i}: {conclusion} (from steps {refs} via resolution on {resolved_on})")
                else:
                    lines.append(f"Step {i}: {conclusion} (from steps {refs} via resolution)")
            else:
                lines.append(f"Step {i}: {conclusion} ({rule})")
        else:
            if premise_refs:
                refs = ", ".join(premise_refs)
                lines.append(f"Step {i}: {conclusion} (from steps {refs} via {rule})")
            else:
                lines.append(f"Step {i}: {conclusion} ({rule})")
    
    # Add final conclusion
    lines.append("=" * 60)
    if proof_tree.conclusion == "⊥ (contradiction)":
        lines.append("CONCLUSION: The formula is UNSATISFIABLE (contradiction derived)")
    else:
        lines.append(f"CONCLUSION: {proof_tree.conclusion}")
    
    return "\n".join(lines)


# ============================================================================
# KNOWLEDGE BASE
# ============================================================================


class KnowledgeBase:
    """
    A simple container for managing a collection of clauses.

    This class holds the set of logical statements (in clause form)
    that form the basis for reasoning.
    """

    def __init__(self):
        """Initializes an empty knowledge base."""
        self.clauses: List[Clause] = []

    def add_clause(self, clause: Clause):
        """
        Adds a clause to the knowledge base.

        Args:
            clause: The Clause object to add.
        """
        if clause not in self.clauses:
            self.clauses.append(clause)

    def get_clauses(self) -> List[Clause]:
        """
        Returns all clauses in the knowledge base.

        Returns:
            A list of all Clause objects.
        """
        return self.clauses

    def __str__(self) -> str:
        return "\n".join(str(c) for c in self.clauses)


# ============================================================================
# UNIFICATION
# ============================================================================


class Unifier:
    """
    Advanced unification with occurs check.

    Unification is the process of finding substitutions that make
    terms equal. Essential for theorem proving in first-order logic.

    Features:
    - Occurs check to prevent infinite structures
    - Substitution composition
    - Application to terms, literals, and clauses
    - Caching for performance

    Example:
        unifier = Unifier()
        t1 = Function('f', [Variable('X')])
        t2 = Function('f', [Constant('a')])
        subst = unifier.unify(t1, t2)
        # subst = {'X': Constant('a')}
    """

    def __init__(self):
        self.substitution_cache = {}

    def unify(
        self, term1: Term, term2: Term, subst: Optional[Dict[str, Term]] = None
    ) -> Optional[Dict[str, Term]]:
        """
        Unify two terms with occurs check.

        The occurs check prevents infinite structures like X = f(X).
        This is the Robinson unification algorithm with occurs check.

        Args:
            term1: First term to unify
            term2: Second term to unify
            subst: Existing substitution (or None for empty)

        Returns:
            Updated substitution if unification succeeds, None otherwise

        Examples:
            >>> unifier = Unifier()
            >>> # Unify variable with constant
            >>> unifier.unify(Variable('X'), Constant('a'))
            {'X': Constant('a')}

            >>> # Unify functions
            >>> f1 = Function('f', [Variable('X'), Constant('b')])
            >>> f2 = Function('f', [Constant('a'), Variable('Y')])
            >>> unifier.unify(f1, f2)
            {'X': Constant('a'), 'Y': Constant('b')}

            >>> # Occurs check prevents infinite structure
            >>> unifier.unify(Variable('X'), Function('f', [Variable('X')]))
            None
        """
        if subst is None:
            subst = {}

        # Apply existing substitution (dereference)
        term1 = self.deref(term1, subst)
        term2 = self.deref(term2, subst)

        # Same term - success
        if term1 == term2:
            return subst

        # Variable unification (term1 is variable)
        if isinstance(term1, Variable):
            if self.occurs_check(term1, term2, subst):
                return None  # Occurs check failed
            new_subst = subst.copy()
            new_subst[term1.name] = term2
            return new_subst

        # Variable unification (term2 is variable)
        if isinstance(term2, Variable):
            if self.occurs_check(term2, term1, subst):
                return None  # Occurs check failed
            new_subst = subst.copy()
            new_subst[term2.name] = term1
            return new_subst

        # Function unification
        if isinstance(term1, Function) and isinstance(term2, Function):
            # Function symbols must match
            if term1.name != term2.name or len(term1.args) != len(term2.args):
                return None

            # Recursively unify arguments
            current_subst = subst
            for arg1, arg2 in zip(term1.args, term2.args):
                current_subst = self.unify(arg1, arg2, current_subst)
                if current_subst is None:
                    return None

            return current_subst

        # Constant unification
        if isinstance(term1, Constant) and isinstance(term2, Constant):
            return subst if term1.name == term2.name else None

        # Different types - unification fails
        return None

    def occurs_check(self, var: Variable, term: Term, subst: Dict[str, Term]) -> bool:
        """
        Check if variable occurs in term.

        The occurs check prevents creating infinite structures like X = f(X).
        This is essential for sound unification.

        Args:
            var: Variable to check
            term: Term to check within
            subst: Current substitution

        Returns:
            True if variable occurs in term (failure), False otherwise

        Examples:
            >>> unifier = Unifier()
            >>> var = Variable('X')
            >>> term = Function('f', [Variable('X')])
            >>> unifier.occurs_check(var, term, {})
            True  # X occurs in f(X) - would create infinite structure
        """
        term = self.deref(term, subst)

        # Variable occurs in itself
        if var == term:
            return True

        # Check function arguments recursively
        if isinstance(term, Function):
            return any(self.occurs_check(var, arg, subst) for arg in term.args)

        return False

    def deref(self, term: Term, subst: Dict[str, Term]) -> Term:
        """
        Dereference a term through substitution.

        Follow the substitution chain until reaching a term
        that's not a variable or a variable with no binding.

        Args:
            term: Term to dereference
            subst: Substitution to apply

        Returns:
            Fully dereferenced term

        Examples:
            >>> unifier = Unifier()
            >>> subst = {'X': Variable('Y'), 'Y': Constant('a')}
            >>> unifier.deref(Variable('X'), subst)
            Constant('a')
        """
        if isinstance(term, Variable) and term.name in subst:
            return self.deref(subst[term.name], subst)
        return term

    def apply_substitution(self, term: Term, subst: Dict[str, Term]) -> Term:
        """
        Apply substitution to term.

        Replace all variables in the term according to the substitution.
        This is a deep operation that traverses the entire term structure.

        Args:
            term: Term to apply substitution to
            subst: Substitution mapping

        Returns:
            New term with substitution applied

        Examples:
            >>> unifier = Unifier()
            >>> term = Function('f', [Variable('X'), Variable('Y')])
            >>> subst = {'X': Constant('a'), 'Y': Constant('b')}
            >>> unifier.apply_substitution(term, subst)
            Function('f', [Constant('a'), Constant('b')])
        """
        term = self.deref(term, subst)

        if isinstance(term, Variable):
            return term
        elif isinstance(term, Constant):
            return term
        elif isinstance(term, Function):
            new_args = [self.apply_substitution(arg, subst) for arg in term.args]
            return Function(name=term.name, args=new_args)

        return term

    def apply_to_literal(self, literal: Literal, subst: Dict[str, Term]) -> Literal:
        """
        Apply substitution to literal.

        Creates a new literal with substitution applied to all terms.
        Preserves the predicate name and negation flag.

        Args:
            literal: Literal to apply substitution to
            subst: Substitution mapping

        Returns:
            New literal with substitution applied

        Examples:
            >>> unifier = Unifier()
            >>> lit = Literal('P', [Variable('X'), Constant('a')])
            >>> subst = {'X': Constant('b')}
            >>> unifier.apply_to_literal(lit, subst)
            Literal('P', [Constant('b'), Constant('a')])
        """
        new_terms = [self.apply_substitution(term, subst) for term in literal.terms]
        return Literal(
            predicate=literal.predicate, terms=new_terms, negated=literal.negated
        )

    def apply_to_clause(self, clause: Clause, subst: Dict[str, Term]) -> Clause:
        """
        Apply substitution to clause.

        Creates a new clause with substitution applied to all literals.
        Preserves confidence and metadata.

        Args:
            clause: Clause to apply substitution to
            subst: Substitution mapping

        Returns:
            New clause with substitution applied

        Examples:
            >>> unifier = Unifier()
            >>> clause = Clause([
            ...     Literal('P', [Variable('X')]),
            ...     Literal('Q', [Variable('Y')])
            ... ])
            >>> subst = {'X': Constant('a'), 'Y': Constant('b')}
            >>> unifier.apply_to_clause(clause, subst)
            Clause([Literal('P', [Constant('a')]), Literal('Q', [Constant('b')])])
        """
        new_literals = [self.apply_to_literal(lit, subst) for lit in clause.literals]
        return Clause(
            literals=new_literals,
            confidence=clause.confidence,
            metadata=clause.metadata.copy(),
        )


# ============================================================================
# PROOF REPRESENTATION
# ============================================================================


@dataclass
class ProofNode:
    """
    Node in a proof tree.

    Represents a step in a proof, including:
    - What was concluded
    - What premises were used
    - What inference rule was applied
    - Confidence in this step
    - Depth in the proof tree

    Used to track and explain proof structure.
    Can be converted to human-readable format.

    Attributes:
        conclusion: The formula/statement concluded
        premises: List of premise ProofNodes
        rule_used: Name of inference rule applied
        confidence: Confidence score (0.0 to 1.0)
        depth: Depth in proof tree (0 = root)
        metadata: Additional information (e.g., substitutions)

    Examples:
        >>> # Simple proof step
        >>> proof = ProofNode(
        ...     conclusion="P(a)",
        ...     premises=[],
        ...     rule_used="assumption",
        ...     confidence=1.0,
        ...     depth=0
        ... )

        >>> # Complex proof with premises
        >>> proof = ProofNode(
        ...     conclusion="Q(a)",
        ...     premises=[proof1, proof2],
        ...     rule_used="modus_ponens",
        ...     confidence=0.95,
        ...     depth=2
        ... )
    """

    conclusion: str
    premises: List["ProofNode"]
    rule_used: str
    confidence: float
    depth: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_string(self, indent: int = 0) -> str:
        """
        Convert proof tree to string representation.

        Creates an indented tree structure showing the proof steps.

        Args:
            indent: Indentation level (spaces)

        Returns:
            String representation of proof tree

        Examples:
            >>> proof.to_string()
            '├─ Q(a) (conf: 0.95, rule: modus_ponens)\\n'
            '  ├─ P(a) (conf: 1.0, rule: assumption)\\n'
            '  ├─ P(a) → Q(a) (conf: 1.0, rule: assumption)\\n'
        """
        result = " " * indent + f"├─ {self.conclusion} "
        result += f"(conf: {self.confidence:.2f}, rule: {self.rule_used})\n"

        for premise in self.premises:
            if isinstance(premise, ProofNode):
                result += premise.to_string(indent + 2)

        return result

    def get_all_rules_used(self) -> List[str]:
        """
        Get list of all rules used in proof tree.

        Returns:
            List of rule names used
        """
        rules = [self.rule_used]
        for premise in self.premises:
            if isinstance(premise, ProofNode):
                rules.extend(premise.get_all_rules_used())
        return rules

    def get_proof_depth(self) -> int:
        """
        Get maximum depth of proof tree.

        Returns:
            Maximum depth
        """
        if not self.premises:
            return self.depth

        max_child_depth = max(
            (p.get_proof_depth() for p in self.premises if isinstance(p, ProofNode)),
            default=self.depth,
        )
        return max(self.depth, max_child_depth)

    def count_steps(self) -> int:
        """
        Count total number of proof steps.

        Returns:
            Number of nodes in proof tree
        """
        count = 1
        for premise in self.premises:
            if isinstance(premise, ProofNode):
                count += premise.count_steps()
        return count


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Terms
    "Term",
    "Variable",
    "Constant",
    "Function",
    # Literals and Clauses
    "Literal",
    "Clause",
    # Knowledge Base
    "KnowledgeBase",
    # Unification
    "Unifier",
    # Proof
    "ProofNode",
]
