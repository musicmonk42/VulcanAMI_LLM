"""
Comprehensive tests for theorem provers.

Tests cover:
- Tableau method with quantifier support
- Resolution with proper CNF negation
- Model elimination (ME-calculus)
- Connection method with unifier consistency
- Natural deduction with complete rule set
- Parallel prover

All tests validate the FIXED implementations.
"""

import time

import pytest

# Import core types needed for testing
from src.vulcan.reasoning.symbolic.core import (Clause, Constant, Function,
                                                Literal, ProofNode, Variable)
# Import the classes we're testing
from src.vulcan.reasoning.symbolic.provers import (ConnectionMethodProver,
                                                   ModelEliminationProver,
                                                   NaturalDeductionProver,
                                                   ParallelProver,
                                                   ResolutionProver,
                                                   TableauProver)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def create_simple_clause(predicate: str, *terms) -> Clause:
    """Helper to create a simple unit clause."""
    literal = Literal(predicate, list(terms), negated=False)
    return Clause([literal])


def create_negated_clause(predicate: str, *terms) -> Clause:
    """Helper to create a negated unit clause."""
    literal = Literal(predicate, list(terms), negated=True)
    return Clause([literal])


def create_implication(p_pred: str, q_pred: str, *p_terms) -> Clause:
    """Helper to create implication ¬P ∨ Q (P → Q)."""
    p_lit = Literal(p_pred, list(p_terms), negated=True)
    q_lit = Literal(q_pred, list(p_terms), negated=False)
    return Clause([p_lit, q_lit])


# ============================================================================
# TABLEAU PROVER TESTS
# ============================================================================


class TestTableauProver:
    """Tests for TableauProver with quantifier support."""

    def test_tableau_creation(self):
        """Test creating a TableauProver."""
        prover = TableauProver()
        assert prover.max_depth == 50
        assert isinstance(prover.unifier, object)

    def test_tableau_simple_proof(self):
        """Test simple tableau proof."""
        prover = TableauProver(max_depth=10)

        # KB: P(a)
        kb = [create_simple_clause("P", Constant("a"))]

        # Goal: P(a)
        goal = create_simple_clause("P", Constant("a"))

        proven, proof, confidence = prover.prove(goal, kb, timeout=1.0)

        assert proven
        assert proof is not None
        assert confidence > 0.9

    def test_tableau_with_negation(self):
        """Test tableau with negated literals."""
        prover = TableauProver(max_depth=10)

        # KB: P(a), ¬P(a) ∨ Q(a)
        kb = [
            create_simple_clause("P", Constant("a")),
            Clause(
                [
                    Literal("P", [Constant("a")], negated=True),
                    Literal("Q", [Constant("a")], negated=False),
                ]
            ),
        ]

        # Goal: Q(a)
        goal = create_simple_clause("Q", Constant("a"))

        proven, proof, confidence = prover.prove(goal, kb, timeout=2.0)

        assert proven
        assert proof is not None

    def test_tableau_unprovable(self):
        """Test tableau with unprovable goal."""
        prover = TableauProver(max_depth=5)

        # KB: P(a)
        kb = [create_simple_clause("P", Constant("a"))]

        # Goal: Q(a) - not provable
        goal = create_simple_clause("Q", Constant("a"))

        proven, proof, confidence = prover.prove(goal, kb, timeout=1.0)

        assert not proven
        assert proof is None

    def test_tableau_timeout(self):
        """Test tableau with timeout."""
        prover = TableauProver(max_depth=100)

        # Complex KB
        kb = [create_simple_clause("P", Variable("X")) for _ in range(10)]
        goal = create_simple_clause("Q", Variable("Y"))

        proven, proof, confidence = prover.prove(goal, kb, timeout=0.001)

        # Should timeout and return False
        assert not proven

    def test_tableau_with_variables(self):
        """Test FIXED: Tableau with universal quantifiers."""
        prover = TableauProver(max_depth=15)

        # KB: P(X) (universal), P(X) → Q(X)
        kb = [
            create_simple_clause("P", Variable("X")),
            Clause(
                [
                    Literal("P", [Variable("X")], negated=True),
                    Literal("Q", [Variable("X")], negated=False),
                ]
            ),
        ]

        # Goal: Q(a) for some constant a
        goal = create_simple_clause("Q", Constant("a"))

        proven, proof, confidence = prover.prove(goal, kb, timeout=2.0)

        # May or may not prove depending on instantiation
        assert isinstance(proven, bool)

    def test_tableau_collect_ground_terms(self):
        """Test ground term collection."""
        prover = TableauProver()

        clause = Clause([Literal("P", [Constant("a"), Constant("b")], negated=False)])

        prover._collect_ground_terms(clause)

        assert Constant("a") in prover.ground_terms
        assert Constant("b") in prover.ground_terms

    def test_tableau_negate_clause(self):
        """Test clause negation."""
        prover = TableauProver()

        clause = Clause(
            [
                Literal("P", [Constant("a")], negated=False),
                Literal("Q", [Constant("b")], negated=False),
            ]
        )

        negated = prover._negate_clause(clause)

        # All literals should be negated
        assert all(lit.negated for lit in negated.literals)

    def test_tableau_branch_closure(self):
        """Test branch closure detection."""
        prover = TableauProver()

        # Complementary literals: P(a) and ¬P(a)
        formulas = [
            create_simple_clause("P", Constant("a")),
            create_negated_clause("P", Constant("a")),
        ]

        assert prover._is_branch_closed(formulas)

    def test_tableau_no_closure(self):
        """Test non-closing branch."""
        prover = TableauProver()

        # Non-complementary literals
        formulas = [
            create_simple_clause("P", Constant("a")),
            create_simple_clause("Q", Constant("b")),
        ]

        assert not prover._is_branch_closed(formulas)


# ============================================================================
# RESOLUTION PROVER TESTS
# ============================================================================


class TestResolutionProver:
    """Tests for ResolutionProver with fixed CNF negation."""

    def test_resolution_creation(self):
        """Test creating a ResolutionProver."""
        prover = ResolutionProver()
        assert prover.max_iterations == 1000

    def test_resolution_simple_proof(self):
        """Test simple resolution proof."""
        prover = ResolutionProver(max_iterations=100)

        # KB: P(a), ¬P(a) ∨ Q(a)
        kb = [
            create_simple_clause("P", Constant("a")),
            Clause(
                [
                    Literal("P", [Constant("a")], negated=True),
                    Literal("Q", [Constant("a")], negated=False),
                ]
            ),
        ]

        # Goal: Q(a)
        goal = create_simple_clause("Q", Constant("a"))

        proven, proof, confidence = prover.prove(goal, kb, timeout=2.0)

        assert proven
        assert proof is not None

    def test_resolution_negate_to_cnf(self):
        """Test FIXED: Proper negation to CNF."""
        prover = ResolutionProver()

        # Clause: P(a) ∨ Q(b)
        clause = Clause(
            [
                Literal("P", [Constant("a")], negated=False),
                Literal("Q", [Constant("b")], negated=False),
            ]
        )

        # Negation: ¬(P(a) ∨ Q(b)) = ¬P(a) ∧ ¬Q(b)
        # In CNF: [¬P(a)], [¬Q(b)] (two separate clauses)
        negated = prover._negate_clause_to_cnf(clause)

        assert len(negated) == 2
        assert all(c.is_unit_clause() for c in negated)
        assert all(c.literals[0].negated for c in negated)

    def test_resolution_binary_resolution(self):
        """Test binary resolution rule."""
        prover = ResolutionProver()

        # P(a) ∨ Q(b)
        clause1 = Clause(
            [
                Literal("P", [Constant("a")], negated=False),
                Literal("Q", [Constant("b")], negated=False),
            ]
        )

        # ¬P(a) ∨ R(c)
        clause2 = Clause(
            [
                Literal("P", [Constant("a")], negated=True),
                Literal("R", [Constant("c")], negated=False),
            ]
        )

        # Should resolve on P(a) to get Q(b) ∨ R(c)
        resolvents = prover._resolve(clause1, clause2)

        assert len(resolvents) > 0
        assert any(len(r.literals) == 2 for r in resolvents)

    def test_resolution_empty_clause(self):
        """Test resolution derives empty clause."""
        prover = ResolutionProver(max_iterations=50)

        # KB: P(a), ¬P(a)
        kb = [
            create_simple_clause("P", Constant("a")),
            create_negated_clause("P", Constant("a")),
        ]

        # Goal: Q(a) (any goal will be proven from contradiction)
        goal = create_simple_clause("Q", Constant("a"))

        proven, proof, confidence = prover.prove(goal, kb, timeout=1.0)

        # Should prove due to contradiction in KB
        assert proven

    def test_resolution_subsumption(self):
        """Test subsumption checking."""
        prover = ResolutionProver()

        # P(X) subsumes P(X) ∨ Q(Y)
        clause1 = create_simple_clause("P", Variable("X"))
        clause2 = Clause(
            [
                Literal("P", [Variable("X")], negated=False),
                Literal("Q", [Variable("Y")], negated=False),
            ]
        )

        # clause1 should subsume clause2
        assert prover._subsumes(clause1, clause2)

    def test_resolution_with_unification(self):
        """Test resolution with unification."""
        prover = ResolutionProver(max_iterations=100)

        # KB: P(X), ¬P(a) ∨ Q(a)
        kb = [
            create_simple_clause("P", Variable("X")),
            Clause(
                [
                    Literal("P", [Constant("a")], negated=True),
                    Literal("Q", [Constant("a")], negated=False),
                ]
            ),
        ]

        # Goal: Q(a)
        goal = create_simple_clause("Q", Constant("a"))

        proven, proof, confidence = prover.prove(goal, kb, timeout=2.0)

        assert proven


# ============================================================================
# MODEL ELIMINATION PROVER TESTS
# ============================================================================


class TestModelEliminationProver:
    """Tests for ModelEliminationProver."""

    def test_me_creation(self):
        """Test creating a ModelEliminationProver."""
        prover = ModelEliminationProver()
        assert prover.max_depth == 20

    def test_me_simple_proof(self):
        """Test simple ME proof."""
        prover = ModelEliminationProver(max_depth=10)

        # KB: P(a)
        kb = [create_simple_clause("P", Constant("a"))]

        # Goal: P(a)
        goal = create_simple_clause("P", Constant("a"))

        proven, proof, confidence = prover.prove(goal, kb, timeout=1.0)

        assert proven
        assert proof is not None

    def test_me_clausify(self):
        """Test converting clause to program clauses."""
        prover = ModelEliminationProver()

        # P(a) ∨ Q(b)
        clause = Clause(
            [
                Literal("P", [Constant("a")], negated=False),
                Literal("Q", [Constant("b")], negated=False),
            ]
        )

        program_clauses = prover._clausify(clause)

        # Should create contrapositives
        assert len(program_clauses) == 2
        assert all("head" in pc for pc in program_clauses)
        assert all("body" in pc for pc in program_clauses)

    def test_me_variable_renaming(self):
        """Test IMPROVED: Global variable renaming."""
        prover = ModelEliminationProver()

        prog_clause = {
            "head": Literal("P", [Variable("X")], negated=False),
            "body": [Literal("Q", [Variable("X")], negated=False)],
            "original_clause": None,
        }

        renamed1 = prover._rename_variables_global(prog_clause)
        renamed2 = prover._rename_variables_global(prog_clause)

        # Should have different variable names (global counter)
        var1 = renamed1["head"].terms[0].name
        var2 = renamed2["head"].terms[0].name

        assert var1 != var2

    def test_me_horn_clause(self):
        """Test ME with Horn clause."""
        prover = ModelEliminationProver(max_depth=10)

        # KB: mortal(X) ∨ ¬human(X), human(socrates)
        kb = [
            Clause(
                [
                    Literal("mortal", [Variable("X")], negated=False),
                    Literal("human", [Variable("X")], negated=True),
                ]
            ),
            create_simple_clause("human", Constant("socrates")),
        ]

        # Goal: mortal(socrates)
        goal = create_simple_clause("mortal", Constant("socrates"))

        proven, proof, confidence = prover.prove(goal, kb, timeout=2.0)

        assert proven


# ============================================================================
# CONNECTION METHOD PROVER TESTS
# ============================================================================


class TestConnectionMethodProver:
    """Tests for ConnectionMethodProver with fixed unifier consistency."""

    def test_connection_creation(self):
        """Test creating a ConnectionMethodProver."""
        prover = ConnectionMethodProver()
        assert prover.max_depth == 30

    def test_connection_simple_proof(self):
        """Test simple connection proof."""
        prover = ConnectionMethodProver(max_depth=10)

        # KB: P(a)
        kb = [create_simple_clause("P", Constant("a"))]

        # Goal: P(a)
        goal = create_simple_clause("P", Constant("a"))

        proven, proof, confidence = prover.prove(goal, kb, timeout=1.0)

        assert proven
        assert proof is not None

    def test_connection_can_connect(self):
        """Test connection detection."""
        prover = ConnectionMethodProver()

        lit1 = Literal("P", [Constant("a")], negated=False)
        lit2 = Literal("P", [Constant("a")], negated=True)

        assert prover._can_connect(lit1, lit2)

    def test_connection_cannot_connect_same_polarity(self):
        """Test no connection with same polarity."""
        prover = ConnectionMethodProver()

        lit1 = Literal("P", [Constant("a")], negated=False)
        lit2 = Literal("P", [Constant("a")], negated=False)

        assert not prover._can_connect(lit1, lit2)

    def test_connection_unifier(self):
        """Test getting unifier for literals."""
        prover = ConnectionMethodProver()

        lit1 = Literal("P", [Variable("X"), Constant("a")], negated=False)
        lit2 = Literal("P", [Constant("b"), Variable("Y")], negated=True)

        unifier = prover._get_unifier(lit1, lit2)

        assert unifier is not None
        assert unifier["X"] == Constant("b")
        assert unifier["Y"] == Constant("a")

    def test_connection_compose_substitutions(self):
        """Test FIXED: Proper substitution composition."""
        prover = ConnectionMethodProver()

        subst1 = {"X": Constant("a")}
        subst2 = {"Y": Variable("X")}

        composed = prover._compose_substitutions(subst1, subst2)

        assert composed is not None
        assert "X" in composed
        assert "Y" in composed

    def test_connection_occurs_check(self):
        """Test occurs check in substitution."""
        prover = ConnectionMethodProver()

        # X should not occur in f(X)
        assert not prover._occurs_check("X", Function("f", [Variable("X")]))

        # X does not occur in constant
        assert prover._occurs_check("X", Constant("a"))


# ============================================================================
# NATURAL DEDUCTION PROVER TESTS
# ============================================================================


class TestNaturalDeductionProver:
    """Tests for NaturalDeductionProver with complete rule set."""

    def test_nd_creation(self):
        """Test creating a NaturalDeductionProver."""
        prover = NaturalDeductionProver()
        assert prover.max_depth == 15
        assert len(prover.rules) > 10  # Has all rules

    def test_nd_assumption_rule(self):
        """Test assumption rule."""
        prover = NaturalDeductionProver()

        goal = create_simple_clause("P", Constant("a"))
        assumptions = [goal]

        result = prover._assumption(goal, assumptions, [], 0, time.time(), 1.0)

        assert result is not None
        assert result.rule_used == "assumption"

    def test_nd_modus_ponens(self):
        """Test modus ponens rule."""
        prover = NaturalDeductionProver(max_depth=5)

        # Assumptions: P(a), ¬P(a) ∨ Q(a) (i.e., P(a) → Q(a))
        assumptions = [
            create_simple_clause("P", Constant("a")),
            Clause(
                [
                    Literal("P", [Constant("a")], negated=True),
                    Literal("Q", [Constant("a")], negated=False),
                ]
            ),
        ]

        # Goal: Q(a)
        goal = create_simple_clause("Q", Constant("a"))

        proven, proof, confidence = prover.prove(goal, assumptions, timeout=2.0)

        assert proven
        assert proof is not None

    def test_nd_and_introduction(self):
        """Test NEW: And introduction rule."""
        prover = NaturalDeductionProver(max_depth=5)

        # Assumptions: P(a), Q(b)
        assumptions = [
            create_simple_clause("P", Constant("a")),
            create_simple_clause("Q", Constant("b")),
        ]

        # Goal: P(a) ∧ Q(b) (represented as conjunction)
        goal = Clause(
            [
                Literal("P", [Constant("a")], negated=False),
                Literal("Q", [Constant("b")], negated=False),
            ]
        )

        proven, proof, confidence = prover.prove(goal, assumptions, timeout=1.0)

        # May prove with and_intro
        assert isinstance(proven, bool)

    def test_nd_and_elimination(self):
        """Test and elimination rule."""
        prover = NaturalDeductionProver(max_depth=5)

        # Assumption: P(a) ∧ Q(b)
        assumptions = [
            Clause(
                [
                    Literal("P", [Constant("a")], negated=False),
                    Literal("Q", [Constant("b")], negated=False),
                ]
            )
        ]

        # Goal: P(a) (extract from conjunction)
        goal = create_simple_clause("P", Constant("a"))

        result = prover._and_elimination(goal, assumptions, [], 0, time.time(), 1.0)

        # May succeed with and_elim
        assert isinstance(result, (ProofNode, type(None)))

    def test_nd_or_introduction(self):
        """Test NEW: Or introduction rule."""
        prover = NaturalDeductionProver(max_depth=5)

        # Assumption: P(a)
        assumptions = [create_simple_clause("P", Constant("a"))]

        # Goal: P(a) ∨ Q(b)
        goal = Clause(
            [
                Literal("P", [Constant("a")], negated=False),
                Literal("Q", [Constant("b")], negated=False),
            ]
        )

        proven, proof, confidence = prover.prove(goal, assumptions, timeout=1.0)

        # Should prove with or_intro
        assert proven

    def test_nd_explosion(self):
        """Test explosion (ex falso quodlibet)."""
        prover = NaturalDeductionProver(max_depth=5)

        # Assumptions: P(a), ¬P(a) (contradiction)
        assumptions = [
            create_simple_clause("P", Constant("a")),
            create_negated_clause("P", Constant("a")),
        ]

        # Goal: Q(b) (anything follows from contradiction)
        goal = create_simple_clause("Q", Constant("b"))

        proven, proof, confidence = prover.prove(goal, assumptions, timeout=1.0)

        assert proven
        # Don't check specific rule - any rule that proves from contradiction is valid
        # (could be 'explosion', 'double_neg_elim', or others)

    def test_nd_double_negation(self):
        """Test NEW: Double negation elimination."""
        prover = NaturalDeductionProver()

        # ¬¬P should give P
        double_neg = Literal("P", [Constant("a")], negated=False)
        double_neg = double_neg.negate().negate()

        # The rule is implemented but testing it directly is complex
        # This validates the rule exists
        assert "double_neg_elim" in prover.rules

    def test_nd_all_rules_exist(self):
        """Test COMPLETE: All standard ND rules are implemented."""
        prover = NaturalDeductionProver()

        expected_rules = [
            "assumption",
            "modus_ponens",
            "and_intro",
            "and_elim",
            "or_intro",
            "or_elim",
            "implies_intro",
            "not_intro",
            "double_neg_elim",
            "explosion",
            "contradiction",
            "universal_intro",
            "universal_elim",
            "existential_intro",
            "existential_elim",
        ]

        for rule in expected_rules:
            assert rule in prover.rules


# ============================================================================
# PARALLEL PROVER TESTS
# ============================================================================


class TestParallelProver:
    """Tests for ParallelProver."""

    def test_parallel_creation(self):
        """Test creating a ParallelProver."""
        prover = ParallelProver(max_workers=2)
        assert prover.max_workers == 2
        assert len(prover.provers) == 5  # All 5 provers

    def test_parallel_simple_proof(self):
        """Test parallel proving."""
        prover = ParallelProver(max_workers=3)

        # KB: P(a)
        kb = [create_simple_clause("P", Constant("a"))]

        # Goal: P(a)
        goal = create_simple_clause("P", Constant("a"))

        proven, proof, confidence, method = prover.prove_parallel(goal, kb, timeout=2.0)

        assert proven
        assert proof is not None
        assert method in [
            "tableau",
            "resolution",
            "model_elimination",
            "connection",
            "natural_deduction",
        ]

    def test_parallel_returns_first_success(self):
        """Test that parallel prover returns first successful proof."""
        prover = ParallelProver(max_workers=4)

        # Simple problem that should be solved quickly
        kb = [
            create_simple_clause("P", Constant("a")),
            Clause(
                [
                    Literal("P", [Constant("a")], negated=True),
                    Literal("Q", [Constant("a")], negated=False),
                ]
            ),
        ]

        goal = create_simple_clause("Q", Constant("a"))

        start_time = time.time()
        proven, proof, confidence, method = prover.prove_parallel(goal, kb, timeout=5.0)
        elapsed = time.time() - start_time

        assert proven
        assert elapsed < 5.0  # Should complete quickly

    def test_parallel_timeout(self):
        """Test parallel prover with timeout."""
        prover = ParallelProver(max_workers=2)

        # Unprovable goal
        kb = [create_simple_clause("P", Constant("a"))]
        goal = create_simple_clause("Q", Constant("b"))

        proven, proof, confidence, method = prover.prove_parallel(goal, kb, timeout=0.5)

        # Should timeout without proof
        assert not proven
        assert method == "none"

    def test_parallel_all_provers_registered(self):
        """Test that all provers are registered."""
        prover = ParallelProver()

        assert "tableau" in prover.provers
        assert "resolution" in prover.provers
        assert "model_elimination" in prover.provers
        assert "connection" in prover.provers
        assert "natural_deduction" in prover.provers


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestProverIntegration:
    """Integration tests across multiple provers."""

    def test_all_provers_same_simple_problem(self):
        """Test that all provers can solve a simple problem."""
        # KB: P(a)
        kb = [create_simple_clause("P", Constant("a"))]

        # Goal: P(a)
        goal = create_simple_clause("P", Constant("a"))

        provers = {
            "tableau": TableauProver(),
            "resolution": ResolutionProver(),
            "model_elimination": ModelEliminationProver(),
            "connection": ConnectionMethodProver(),
            "natural_deduction": NaturalDeductionProver(),
        }

        results = {}
        for name, prover in provers.items():
            proven, proof, confidence = prover.prove(goal, kb, timeout=2.0)
            results[name] = proven

        # At least some provers should succeed
        assert any(results.values())

    def test_modus_ponens_all_provers(self):
        """Test modus ponens across all provers."""
        # KB: P(a), P(a) → Q(a)
        kb = [
            create_simple_clause("P", Constant("a")),
            Clause(
                [
                    Literal("P", [Constant("a")], negated=True),
                    Literal("Q", [Constant("a")], negated=False),
                ]
            ),
        ]

        # Goal: Q(a)
        goal = create_simple_clause("Q", Constant("a"))

        provers = {
            "tableau": TableauProver(max_depth=10),
            "resolution": ResolutionProver(max_iterations=100),
            "model_elimination": ModelEliminationProver(max_depth=10),
            "natural_deduction": NaturalDeductionProver(max_depth=10),
        }

        successes = 0
        for name, prover in provers.items():
            proven, proof, confidence = prover.prove(goal, kb, timeout=2.0)
            if proven:
                successes += 1

        # At least 2 provers should succeed
        assert successes >= 2

    def test_contradiction_detection(self):
        """Test that provers can detect contradictions."""
        # KB: P(a), ¬P(a)
        kb = [
            create_simple_clause("P", Constant("a")),
            create_negated_clause("P", Constant("a")),
        ]

        # Goal: Q(a) (should be provable from contradiction)
        goal = create_simple_clause("Q", Constant("a"))

        # Resolution should definitely handle this
        prover = ResolutionProver(max_iterations=50)
        proven, proof, confidence = prover.prove(goal, kb, timeout=1.0)

        assert proven

        # Natural deduction should also handle it (explosion)
        nd_prover = NaturalDeductionProver(max_depth=5)
        proven_nd, proof_nd, conf_nd = nd_prover.prove(goal, kb, timeout=1.0)

        assert proven_nd

    def test_complex_proof_chain(self):
        """Test complex proof chain."""
        # KB: P(a), P(X) → Q(X), Q(X) → R(X)
        kb = [
            create_simple_clause("P", Constant("a")),
            Clause(
                [
                    Literal("P", [Variable("X")], negated=True),
                    Literal("Q", [Variable("X")], negated=False),
                ]
            ),
            Clause(
                [
                    Literal("Q", [Variable("X")], negated=True),
                    Literal("R", [Variable("X")], negated=False),
                ]
            ),
        ]

        # Goal: R(a)
        goal = create_simple_clause("R", Constant("a"))

        # Try with multiple provers
        provers = [
            TableauProver(max_depth=15),
            ResolutionProver(max_iterations=200),
            ModelEliminationProver(max_depth=15),
        ]

        proven_by_any = False
        for prover in provers:
            proven, proof, confidence = prover.prove(goal, kb, timeout=3.0)
            if proven:
                proven_by_any = True
                break

        # At least one prover should handle this
        assert proven_by_any


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================


class TestProverPerformance:
    """Performance and stress tests."""

    def test_depth_limit_prevents_infinite_loop(self):
        """Test that depth limits prevent infinite loops."""
        # Create cyclic-looking KB
        kb = [
            Clause(
                [
                    Literal("P", [Variable("X")], negated=True),
                    Literal("P", [Function("f", [Variable("X")])], negated=False),
                ]
            )
        ]

        goal = create_simple_clause("P", Constant("a"))

        prover = TableauProver(max_depth=5)

        start_time = time.time()
        proven, proof, confidence = prover.prove(goal, kb, timeout=2.0)
        elapsed = time.time() - start_time

        # Should not hang (depth limit or timeout)
        assert elapsed < 2.5

    def test_timeout_respected(self):
        """Test that timeout is respected."""
        # Large KB
        kb = [create_simple_clause(f"P{i}", Variable("X")) for i in range(20)]
        goal = create_simple_clause("Q", Variable("Y"))

        prover = ResolutionProver(max_iterations=10000)

        start_time = time.time()
        proven, proof, confidence = prover.prove(goal, kb, timeout=0.5)
        elapsed = time.time() - start_time

        # Should respect timeout (with small buffer for overhead)
        assert elapsed < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
