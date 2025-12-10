"""
Comprehensive tests for core symbolic reasoning data structures.

Tests cover:
- Terms (Variable, Constant, Function)
- Literals and Clauses
- KnowledgeBase management
- Unification algorithm with occurs check
- ProofNode structures

All tests validate the core functionality for symbolic reasoning.
"""

import pytest
from typing import Dict, Any

# Import the classes we're testing
from src.vulcan.reasoning.symbolic.core import (
    Term,
    Variable,
    Constant,
    Function,
    Literal,
    Clause,
    KnowledgeBase,
    Unifier,
    ProofNode,
)


# ============================================================================
# TERM TESTS
# ============================================================================


class TestVariable:
    """Tests for Variable terms."""

    def test_variable_creation(self):
        """Test creating a Variable."""
        var = Variable("X")
        assert var.name == "X"
        assert isinstance(var, Term)
        assert isinstance(var, Variable)

    def test_variable_string(self):
        """Test Variable string representation."""
        var = Variable("Y")
        assert str(var) == "Y"

    def test_variable_equality(self):
        """Test Variable equality."""
        var1 = Variable("X")
        var2 = Variable("X")
        var3 = Variable("Y")

        assert var1 == var2
        assert var1 != var3
        assert var2 != var3

    def test_variable_hash(self):
        """Test Variable hashing."""
        var1 = Variable("X")
        var2 = Variable("X")
        var3 = Variable("Y")

        # Same variables should have same hash
        assert hash(var1) == hash(var2)
        # Different variables likely have different hash
        assert hash(var1) != hash(var3)

        # Should be usable in sets
        var_set = {var1, var2, var3}
        assert len(var_set) == 2  # var1 and var2 are same

    def test_variable_inequality_with_other_types(self):
        """Test Variable inequality with other types."""
        var = Variable("X")
        const = Constant("X")

        assert var != const
        assert var != "X"
        assert var != None


class TestConstant:
    """Tests for Constant terms."""

    def test_constant_creation(self):
        """Test creating a Constant."""
        const = Constant("a")
        assert const.name == "a"
        assert isinstance(const, Term)
        assert isinstance(const, Constant)

    def test_constant_string(self):
        """Test Constant string representation."""
        const = Constant("socrates")
        assert str(const) == "socrates"

    def test_constant_equality(self):
        """Test Constant equality."""
        const1 = Constant("a")
        const2 = Constant("a")
        const3 = Constant("b")

        assert const1 == const2
        assert const1 != const3
        assert const2 != const3

    def test_constant_hash(self):
        """Test Constant hashing."""
        const1 = Constant("a")
        const2 = Constant("a")
        const3 = Constant("b")

        assert hash(const1) == hash(const2)
        assert hash(const1) != hash(const3)

        # Should be usable in sets
        const_set = {const1, const2, const3}
        assert len(const_set) == 2

    def test_constant_inequality_with_other_types(self):
        """Test Constant inequality with other types."""
        const = Constant("a")
        var = Variable("a")

        assert const != var
        assert const != "a"


class TestFunction:
    """Tests for Function terms."""

    def test_function_creation(self):
        """Test creating a Function."""
        func = Function("f", [Constant("a"), Variable("X")])
        assert func.name == "f"
        assert len(func.args) == 2
        assert isinstance(func, Term)
        assert isinstance(func, Function)

    def test_function_string_simple(self):
        """Test Function string representation."""
        func = Function("f", [Constant("a")])
        assert str(func) == "f(a)"

    def test_function_string_multiple_args(self):
        """Test Function with multiple arguments."""
        func = Function("g", [Constant("a"), Variable("X"), Constant("b")])
        assert str(func) == "g(a,X,b)"

    def test_function_string_nested(self):
        """Test nested Function."""
        inner = Function("g", [Constant("a")])
        outer = Function("f", [inner, Variable("X")])
        assert str(outer) == "f(g(a),X)"

    def test_function_equality(self):
        """Test Function equality."""
        func1 = Function("f", [Constant("a"), Variable("X")])
        func2 = Function("f", [Constant("a"), Variable("X")])
        func3 = Function("f", [Constant("b"), Variable("X")])
        func4 = Function("g", [Constant("a"), Variable("X")])

        assert func1 == func2
        assert func1 != func3  # Different arguments
        assert func1 != func4  # Different function name

    def test_function_hash(self):
        """Test Function hashing."""
        func1 = Function("f", [Constant("a")])
        func2 = Function("f", [Constant("a")])
        func3 = Function("f", [Constant("b")])

        assert hash(func1) == hash(func2)
        assert hash(func1) != hash(func3)

        # Should be usable in sets
        func_set = {func1, func2, func3}
        assert len(func_set) == 2

    def test_function_empty_args(self):
        """Test Function with no arguments."""
        func = Function("f", [])
        assert str(func) == "f()"
        assert len(func.args) == 0


# ============================================================================
# LITERAL TESTS
# ============================================================================


class TestLiteral:
    """Tests for Literal."""

    def test_literal_creation_positive(self):
        """Test creating positive literal."""
        lit = Literal("P", [Constant("a"), Variable("X")], negated=False)
        assert lit.predicate == "P"
        assert len(lit.terms) == 2
        assert not lit.negated

    def test_literal_creation_negative(self):
        """Test creating negative literal."""
        lit = Literal("P", [Constant("a")], negated=True)
        assert lit.predicate == "P"
        assert lit.negated

    def test_literal_string_positive(self):
        """Test positive literal string."""
        lit = Literal("P", [Constant("a"), Variable("X")], negated=False)
        assert str(lit) == "P(a,X)"

    def test_literal_string_negative(self):
        """Test negative literal string."""
        lit = Literal("P", [Constant("a")], negated=True)
        assert str(lit) == "¬P(a)"

    def test_literal_string_no_terms(self):
        """Test literal with no terms."""
        lit = Literal("Q", [], negated=False)
        assert str(lit) == "Q"

        lit_neg = Literal("Q", [], negated=True)
        assert str(lit_neg) == "¬Q"

    def test_literal_equality(self):
        """Test Literal equality."""
        lit1 = Literal("P", [Constant("a")], negated=False)
        lit2 = Literal("P", [Constant("a")], negated=False)
        lit3 = Literal("P", [Constant("a")], negated=True)
        lit4 = Literal("Q", [Constant("a")], negated=False)

        assert lit1 == lit2
        assert lit1 != lit3  # Different negation
        assert lit1 != lit4  # Different predicate

    def test_literal_hash(self):
        """Test Literal hashing."""
        lit1 = Literal("P", [Constant("a")], negated=False)
        lit2 = Literal("P", [Constant("a")], negated=False)
        lit3 = Literal("P", [Constant("a")], negated=True)

        assert hash(lit1) == hash(lit2)
        assert hash(lit1) != hash(lit3)

        # Should be usable in sets
        lit_set = {lit1, lit2, lit3}
        assert len(lit_set) == 2

    def test_literal_negate(self):
        """Test Literal negation."""
        lit_pos = Literal("P", [Constant("a")], negated=False)
        lit_neg = lit_pos.negate()

        assert lit_neg.predicate == "P"
        assert lit_neg.negated == True
        assert lit_neg.terms[0] == Constant("a")

        # Double negation
        lit_pos_again = lit_neg.negate()
        assert lit_pos_again.negated == False


# ============================================================================
# CLAUSE TESTS
# ============================================================================


class TestClause:
    """Tests for Clause."""

    def test_clause_creation(self):
        """Test creating a Clause."""
        lit1 = Literal("P", [Constant("a")], negated=False)
        lit2 = Literal("Q", [Variable("X")], negated=True)

        clause = Clause([lit1, lit2])
        assert len(clause.literals) == 2
        assert clause.confidence == 1.0
        assert clause.metadata == {}

    def test_clause_with_confidence(self):
        """Test Clause with custom confidence."""
        lit = Literal("P", [Constant("a")], negated=False)
        clause = Clause([lit], confidence=0.8)

        assert clause.confidence == 0.8

    def test_clause_with_metadata(self):
        """Test Clause with metadata."""
        lit = Literal("P", [Constant("a")], negated=False)
        clause = Clause([lit], metadata={"source": "axiom1"})

        assert clause.metadata["source"] == "axiom1"

    def test_clause_is_unit(self):
        """Test unit clause detection."""
        lit = Literal("P", [Constant("a")], negated=False)
        clause = Clause([lit])

        assert clause.is_unit_clause()

    def test_clause_is_not_unit(self):
        """Test non-unit clause."""
        lit1 = Literal("P", [Constant("a")], negated=False)
        lit2 = Literal("Q", [Variable("X")], negated=False)
        clause = Clause([lit1, lit2])

        assert not clause.is_unit_clause()

    def test_clause_is_horn(self):
        """Test Horn clause detection."""
        # At most one positive literal
        lit1 = Literal("P", [Constant("a")], negated=False)  # Positive
        lit2 = Literal("Q", [Variable("X")], negated=True)  # Negative
        lit3 = Literal("R", [Constant("b")], negated=True)  # Negative

        clause = Clause([lit1, lit2, lit3])
        assert clause.is_horn_clause()

    def test_clause_is_not_horn(self):
        """Test non-Horn clause."""
        # Two positive literals
        lit1 = Literal("P", [Constant("a")], negated=False)
        lit2 = Literal("Q", [Variable("X")], negated=False)
        lit3 = Literal("R", [Constant("b")], negated=True)

        clause = Clause([lit1, lit2, lit3])
        assert not clause.is_horn_clause()

    def test_clause_is_empty(self):
        """Test empty clause detection."""
        clause = Clause([])
        assert clause.is_empty()

    def test_clause_is_not_empty(self):
        """Test non-empty clause."""
        lit = Literal("P", [Constant("a")], negated=False)
        clause = Clause([lit])

        assert not clause.is_empty()

    def test_clause_string_single_literal(self):
        """Test Clause string with single literal."""
        lit = Literal("P", [Constant("a")], negated=False)
        clause = Clause([lit])

        assert str(clause) == "P(a)"

    def test_clause_string_multiple_literals(self):
        """Test Clause string with multiple literals."""
        lit1 = Literal("P", [Constant("a")], negated=False)
        lit2 = Literal("Q", [Variable("X")], negated=True)
        clause = Clause([lit1, lit2])

        result = str(clause)
        assert "P(a)" in result
        assert "¬Q(X)" in result
        assert "∨" in result

    def test_clause_string_empty(self):
        """Test empty clause string."""
        clause = Clause([])
        assert str(clause) == "□"

    def test_clause_equality(self):
        """Test Clause equality (order independent)."""
        lit1 = Literal("P", [Constant("a")], negated=False)
        lit2 = Literal("Q", [Variable("X")], negated=True)

        clause1 = Clause([lit1, lit2])
        clause2 = Clause([lit2, lit1])  # Different order
        clause3 = Clause([lit1])

        assert clause1 == clause2  # Order doesn't matter
        assert clause1 != clause3

    def test_clause_hash(self):
        """Test Clause hashing."""
        lit1 = Literal("P", [Constant("a")], negated=False)
        lit2 = Literal("Q", [Variable("X")], negated=True)

        clause1 = Clause([lit1, lit2])
        clause2 = Clause([lit2, lit1])  # Different order

        # Should have same hash regardless of order
        assert hash(clause1) == hash(clause2)

        # Should be usable in sets
        clause_set = {clause1, clause2}
        assert len(clause_set) == 1


# ============================================================================
# KNOWLEDGE BASE TESTS
# ============================================================================


class TestKnowledgeBase:
    """Tests for KnowledgeBase."""

    def test_kb_creation(self):
        """Test creating a KnowledgeBase."""
        kb = KnowledgeBase()
        assert len(kb.clauses) == 0

    def test_kb_add_clause(self):
        """Test adding clauses to KB."""
        kb = KnowledgeBase()

        lit = Literal("P", [Constant("a")], negated=False)
        clause = Clause([lit])

        kb.add_clause(clause)
        assert len(kb.clauses) == 1
        assert clause in kb.clauses

    def test_kb_add_multiple_clauses(self):
        """Test adding multiple clauses."""
        kb = KnowledgeBase()

        clause1 = Clause([Literal("P", [Constant("a")], negated=False)])
        clause2 = Clause([Literal("Q", [Variable("X")], negated=False)])

        kb.add_clause(clause1)
        kb.add_clause(clause2)

        assert len(kb.clauses) == 2

    def test_kb_no_duplicates(self):
        """Test that KB doesn't add duplicate clauses."""
        kb = KnowledgeBase()

        clause = Clause([Literal("P", [Constant("a")], negated=False)])

        kb.add_clause(clause)
        kb.add_clause(clause)  # Try to add again

        assert len(kb.clauses) == 1

    def test_kb_get_clauses(self):
        """Test getting clauses from KB."""
        kb = KnowledgeBase()

        clause1 = Clause([Literal("P", [Constant("a")], negated=False)])
        clause2 = Clause([Literal("Q", [Variable("X")], negated=False)])

        kb.add_clause(clause1)
        kb.add_clause(clause2)

        clauses = kb.get_clauses()
        assert len(clauses) == 2
        assert clause1 in clauses
        assert clause2 in clauses

    def test_kb_string(self):
        """Test KnowledgeBase string representation."""
        kb = KnowledgeBase()

        clause1 = Clause([Literal("P", [Constant("a")], negated=False)])
        clause2 = Clause([Literal("Q", [Variable("X")], negated=False)])

        kb.add_clause(clause1)
        kb.add_clause(clause2)

        kb_str = str(kb)
        assert "P(a)" in kb_str
        assert "Q(X)" in kb_str


# ============================================================================
# UNIFICATION TESTS
# ============================================================================


class TestUnifier:
    """Tests for Unifier."""

    def test_unify_variable_with_constant(self):
        """Test unifying variable with constant."""
        unifier = Unifier()

        var = Variable("X")
        const = Constant("a")

        subst = unifier.unify(var, const)

        assert subst is not None
        assert subst["X"] == const

    def test_unify_constant_with_variable(self):
        """Test unifying constant with variable (reverse)."""
        unifier = Unifier()

        const = Constant("a")
        var = Variable("X")

        subst = unifier.unify(const, var)

        assert subst is not None
        assert subst["X"] == const

    def test_unify_same_constants(self):
        """Test unifying identical constants."""
        unifier = Unifier()

        const1 = Constant("a")
        const2 = Constant("a")

        subst = unifier.unify(const1, const2)

        assert subst is not None
        assert subst == {}

    def test_unify_different_constants(self):
        """Test unifying different constants fails."""
        unifier = Unifier()

        const1 = Constant("a")
        const2 = Constant("b")

        subst = unifier.unify(const1, const2)

        assert subst is None

    def test_unify_same_variables(self):
        """Test unifying identical variables."""
        unifier = Unifier()

        var1 = Variable("X")
        var2 = Variable("X")

        subst = unifier.unify(var1, var2)

        assert subst is not None
        assert subst == {}

    def test_unify_different_variables(self):
        """Test unifying different variables."""
        unifier = Unifier()

        var1 = Variable("X")
        var2 = Variable("Y")

        subst = unifier.unify(var1, var2)

        assert subst is not None
        assert subst["X"] == var2 or subst["Y"] == var1

    def test_unify_functions_same_name(self):
        """Test unifying functions with same name and args."""
        unifier = Unifier()

        func1 = Function("f", [Variable("X"), Constant("b")])
        func2 = Function("f", [Constant("a"), Variable("Y")])

        subst = unifier.unify(func1, func2)

        assert subst is not None
        assert subst["X"] == Constant("a")
        assert subst["Y"] == Constant("b")

    def test_unify_functions_different_names(self):
        """Test unifying functions with different names fails."""
        unifier = Unifier()

        func1 = Function("f", [Variable("X")])
        func2 = Function("g", [Variable("Y")])

        subst = unifier.unify(func1, func2)

        assert subst is None

    def test_unify_functions_different_arity(self):
        """Test unifying functions with different arity fails."""
        unifier = Unifier()

        func1 = Function("f", [Variable("X")])
        func2 = Function("f", [Variable("Y"), Constant("a")])

        subst = unifier.unify(func1, func2)

        assert subst is None

    def test_unify_nested_functions(self):
        """Test unifying nested functions."""
        unifier = Unifier()

        inner1 = Function("g", [Variable("X")])
        func1 = Function("f", [inner1])

        inner2 = Function("g", [Constant("a")])
        func2 = Function("f", [inner2])

        subst = unifier.unify(func1, func2)

        assert subst is not None
        assert subst["X"] == Constant("a")

    def test_occurs_check_prevents_infinite_structure(self):
        """Test occurs check prevents X = f(X)."""
        unifier = Unifier()

        var = Variable("X")
        func = Function("f", [var])

        subst = unifier.unify(var, func)

        assert subst is None  # Should fail occurs check

    def test_occurs_check_nested(self):
        """Test occurs check with nested structure."""
        unifier = Unifier()

        var = Variable("X")
        inner = Function("g", [var])
        func = Function("f", [inner])

        subst = unifier.unify(var, func)

        assert subst is None  # Should fail occurs check

    def test_occurs_check_method(self):
        """Test occurs_check method directly."""
        unifier = Unifier()

        var = Variable("X")
        func = Function("f", [var])

        # Variable occurs in function containing it
        assert unifier.occurs_check(var, func, {}) == True

        # Variable doesn't occur in different term
        other_var = Variable("Y")
        assert unifier.occurs_check(var, other_var, {}) == False

    def test_deref_simple(self):
        """Test simple dereferencing."""
        unifier = Unifier()

        subst = {"X": Constant("a")}
        result = unifier.deref(Variable("X"), subst)

        assert result == Constant("a")

    def test_deref_chain(self):
        """Test dereferencing chain."""
        unifier = Unifier()

        subst = {"X": Variable("Y"), "Y": Constant("a")}
        result = unifier.deref(Variable("X"), subst)

        assert result == Constant("a")

    def test_deref_unbound_variable(self):
        """Test dereferencing unbound variable."""
        unifier = Unifier()

        subst = {}
        result = unifier.deref(Variable("X"), subst)

        assert result == Variable("X")

    def test_apply_substitution_to_variable(self):
        """Test applying substitution to variable."""
        unifier = Unifier()

        subst = {"X": Constant("a")}
        result = unifier.apply_substitution(Variable("X"), subst)

        assert result == Constant("a")

    def test_apply_substitution_to_constant(self):
        """Test applying substitution to constant."""
        unifier = Unifier()

        subst = {"X": Constant("a")}
        result = unifier.apply_substitution(Constant("b"), subst)

        assert result == Constant("b")

    def test_apply_substitution_to_function(self):
        """Test applying substitution to function."""
        unifier = Unifier()

        func = Function("f", [Variable("X"), Variable("Y")])
        subst = {"X": Constant("a"), "Y": Constant("b")}

        result = unifier.apply_substitution(func, subst)

        assert isinstance(result, Function)
        assert result.name == "f"
        assert result.args[0] == Constant("a")
        assert result.args[1] == Constant("b")

    def test_apply_to_literal(self):
        """Test applying substitution to literal."""
        unifier = Unifier()

        lit = Literal("P", [Variable("X"), Constant("a")], negated=False)
        subst = {"X": Constant("b")}

        result = unifier.apply_to_literal(lit, subst)

        assert result.predicate == "P"
        assert result.terms[0] == Constant("b")
        assert result.terms[1] == Constant("a")
        assert result.negated == False

    def test_apply_to_literal_preserves_negation(self):
        """Test applying substitution preserves negation."""
        unifier = Unifier()

        lit = Literal("P", [Variable("X")], negated=True)
        subst = {"X": Constant("a")}

        result = unifier.apply_to_literal(lit, subst)

        assert result.negated == True

    def test_apply_to_clause(self):
        """Test applying substitution to clause."""
        unifier = Unifier()

        lit1 = Literal("P", [Variable("X")], negated=False)
        lit2 = Literal("Q", [Variable("Y")], negated=True)
        clause = Clause([lit1, lit2], confidence=0.9)

        subst = {"X": Constant("a"), "Y": Constant("b")}

        result = unifier.apply_to_clause(clause, subst)

        assert len(result.literals) == 2
        assert result.literals[0].terms[0] == Constant("a")
        assert result.literals[1].terms[0] == Constant("b")
        assert result.confidence == 0.9

    def test_apply_to_clause_preserves_metadata(self):
        """Test applying substitution preserves metadata."""
        unifier = Unifier()

        lit = Literal("P", [Variable("X")], negated=False)
        clause = Clause([lit], metadata={"source": "axiom1"})

        subst = {"X": Constant("a")}
        result = unifier.apply_to_clause(clause, subst)

        assert result.metadata["source"] == "axiom1"

    def test_unify_with_existing_substitution(self):
        """Test unifying with existing substitution."""
        unifier = Unifier()

        # Existing substitution
        subst = {"Y": Constant("b")}

        # Unify X with a
        result = unifier.unify(Variable("X"), Constant("a"), subst)

        assert result is not None
        assert result["X"] == Constant("a")
        assert result["Y"] == Constant("b")


# ============================================================================
# PROOF NODE TESTS
# ============================================================================


class TestProofNode:
    """Tests for ProofNode."""

    def test_proof_node_creation(self):
        """Test creating a ProofNode."""
        proof = ProofNode(
            conclusion="P(a)", premises=[], rule_used="axiom", confidence=1.0, depth=0
        )

        assert proof.conclusion == "P(a)"
        assert len(proof.premises) == 0
        assert proof.rule_used == "axiom"
        assert proof.confidence == 1.0
        assert proof.depth == 0

    def test_proof_node_with_metadata(self):
        """Test ProofNode with metadata."""
        proof = ProofNode(
            conclusion="P(a)",
            premises=[],
            rule_used="axiom",
            confidence=1.0,
            depth=0,
            metadata={"source": "kb1"},
        )

        assert proof.metadata["source"] == "kb1"

    def test_proof_node_with_premises(self):
        """Test ProofNode with premises."""
        premise1 = ProofNode(
            conclusion="P(a)", premises=[], rule_used="axiom", confidence=1.0, depth=0
        )

        premise2 = ProofNode(
            conclusion="P(a) → Q(a)",
            premises=[],
            rule_used="axiom",
            confidence=1.0,
            depth=0,
        )

        proof = ProofNode(
            conclusion="Q(a)",
            premises=[premise1, premise2],
            rule_used="modus_ponens",
            confidence=0.95,
            depth=1,
        )

        assert len(proof.premises) == 2
        assert proof.premises[0] == premise1
        assert proof.premises[1] == premise2

    def test_proof_node_to_string(self):
        """Test ProofNode string representation."""
        proof = ProofNode(
            conclusion="P(a)", premises=[], rule_used="axiom", confidence=1.0, depth=0
        )

        result = proof.to_string()

        assert "P(a)" in result
        assert "axiom" in result
        assert "1.00" in result

    def test_proof_node_to_string_with_premises(self):
        """Test ProofNode string with premises."""
        premise = ProofNode(
            conclusion="P(a)", premises=[], rule_used="axiom", confidence=1.0, depth=0
        )

        proof = ProofNode(
            conclusion="Q(a)",
            premises=[premise],
            rule_used="inference",
            confidence=0.9,
            depth=1,
        )

        result = proof.to_string()

        assert "Q(a)" in result
        assert "P(a)" in result
        assert "inference" in result
        assert "axiom" in result

    def test_proof_node_get_all_rules_used(self):
        """Test getting all rules used in proof."""
        premise = ProofNode(
            conclusion="P(a)", premises=[], rule_used="axiom", confidence=1.0, depth=0
        )

        proof = ProofNode(
            conclusion="Q(a)",
            premises=[premise],
            rule_used="modus_ponens",
            confidence=0.9,
            depth=1,
        )

        rules = proof.get_all_rules_used()

        assert "modus_ponens" in rules
        assert "axiom" in rules
        assert len(rules) == 2

    def test_proof_node_get_proof_depth_no_premises(self):
        """Test getting proof depth with no premises."""
        proof = ProofNode(
            conclusion="P(a)", premises=[], rule_used="axiom", confidence=1.0, depth=0
        )

        assert proof.get_proof_depth() == 0

    def test_proof_node_get_proof_depth_with_premises(self):
        """Test getting proof depth with premises."""
        premise1 = ProofNode(
            conclusion="P(a)", premises=[], rule_used="axiom", confidence=1.0, depth=0
        )

        premise2 = ProofNode(
            conclusion="P(a) → Q(a)",
            premises=[],
            rule_used="axiom",
            confidence=1.0,
            depth=0,
        )

        proof = ProofNode(
            conclusion="Q(a)",
            premises=[premise1, premise2],
            rule_used="modus_ponens",
            confidence=0.95,
            depth=1,
        )

        assert proof.get_proof_depth() == 1

    def test_proof_node_get_proof_depth_nested(self):
        """Test getting proof depth with nested structure."""
        leaf = ProofNode(
            conclusion="A", premises=[], rule_used="axiom", confidence=1.0, depth=0
        )

        middle = ProofNode(
            conclusion="B", premises=[leaf], rule_used="rule1", confidence=1.0, depth=1
        )

        root = ProofNode(
            conclusion="C",
            premises=[middle],
            rule_used="rule2",
            confidence=1.0,
            depth=2,
        )

        assert root.get_proof_depth() == 2

    def test_proof_node_count_steps_single(self):
        """Test counting steps in single node."""
        proof = ProofNode(
            conclusion="P(a)", premises=[], rule_used="axiom", confidence=1.0, depth=0
        )

        assert proof.count_steps() == 1

    def test_proof_node_count_steps_with_premises(self):
        """Test counting steps with premises."""
        premise1 = ProofNode(
            conclusion="P(a)", premises=[], rule_used="axiom", confidence=1.0, depth=0
        )

        premise2 = ProofNode(
            conclusion="P(a) → Q(a)",
            premises=[],
            rule_used="axiom",
            confidence=1.0,
            depth=0,
        )

        proof = ProofNode(
            conclusion="Q(a)",
            premises=[premise1, premise2],
            rule_used="modus_ponens",
            confidence=0.95,
            depth=1,
        )

        assert proof.count_steps() == 3  # 1 root + 2 premises


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_kb_with_unification(self):
        """Test KnowledgeBase with unification."""
        kb = KnowledgeBase()
        unifier = Unifier()

        # Add clause: P(X) ∨ Q(a)
        lit1 = Literal("P", [Variable("X")], negated=False)
        lit2 = Literal("Q", [Constant("a")], negated=False)
        clause = Clause([lit1, lit2])

        kb.add_clause(clause)

        # Apply substitution
        subst = {"X": Constant("b")}
        result = unifier.apply_to_clause(clause, subst)

        assert result.literals[0].terms[0] == Constant("b")
        assert result.literals[1].terms[0] == Constant("a")

    def test_complex_unification_scenario(self):
        """Test complex unification scenario."""
        unifier = Unifier()

        # Unify: f(X, g(Y)) with f(a, g(b))
        func1 = Function("f", [Variable("X"), Function("g", [Variable("Y")])])

        func2 = Function("f", [Constant("a"), Function("g", [Constant("b")])])

        subst = unifier.unify(func1, func2)

        assert subst is not None
        assert subst["X"] == Constant("a")
        assert subst["Y"] == Constant("b")

    def test_proof_tree_construction(self):
        """Test constructing a proof tree."""
        # Axiom: P(a)
        axiom1 = ProofNode(
            conclusion="P(a)", premises=[], rule_used="axiom", confidence=1.0, depth=0
        )

        # Axiom: P(a) → Q(a)
        axiom2 = ProofNode(
            conclusion="P(a) → Q(a)",
            premises=[],
            rule_used="axiom",
            confidence=1.0,
            depth=0,
        )

        # Conclusion: Q(a) via modus ponens
        conclusion = ProofNode(
            conclusion="Q(a)",
            premises=[axiom1, axiom2],
            rule_used="modus_ponens",
            confidence=1.0,
            depth=1,
        )

        # Verify structure
        assert conclusion.count_steps() == 3
        assert conclusion.get_proof_depth() == 1

        rules = conclusion.get_all_rules_used()
        assert "axiom" in rules
        assert "modus_ponens" in rules


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
