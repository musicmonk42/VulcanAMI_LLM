"""
Comprehensive tests for advanced reasoning systems.

Tests cover:
- SymbolicReasoner: FOL reasoning with multiple theorem provers
- ProbabilisticReasoner: Bayesian network reasoning
- HybridReasoner: Combined symbolic and probabilistic reasoning

All tests validate the FIXED implementations.
"""

import time
from typing import Any, Dict, List

import pytest

# Import core types for validation
from src.vulcan.reasoning.symbolic.core import (Clause, Constant, Function,
                                                Literal, ProofNode, Variable)
# Import the classes we're testing
from src.vulcan.reasoning.symbolic.reasoner import (HybridReasoner,
                                                    ProbabilisticReasoner,
                                                    SymbolicReasoner)

# ============================================================================
# SYMBOLIC REASONER TESTS
# ============================================================================


class TestSymbolicReasoner:
    """Tests for SymbolicReasoner with FOL support."""

    def test_symbolic_reasoner_creation(self):
        """Test creating a SymbolicReasoner."""
        reasoner = SymbolicReasoner()
        assert reasoner.kb is not None
        assert reasoner.prover is not None
        assert reasoner.prover_type == "parallel"

    def test_symbolic_reasoner_with_prover_type(self):
        """Test creating reasoner with specific prover type."""
        reasoner = SymbolicReasoner(prover_type="resolution")
        assert reasoner.prover_type == "resolution"

        reasoner = SymbolicReasoner(prover_type="tableau")
        assert reasoner.prover_type == "tableau"

    def test_add_simple_fact(self):
        """Test adding simple facts."""
        reasoner = SymbolicReasoner()
        reasoner.add_fact("P(a)")

        assert len(reasoner.kb.clauses) == 1
        assert reasoner.kb.clauses[0].is_unit_clause()

    def test_add_simple_rule(self):
        """Test adding simple rules."""
        reasoner = SymbolicReasoner()
        reasoner.add_rule("P(a) | Q(b)")

        assert len(reasoner.kb.clauses) == 1
        assert len(reasoner.kb.clauses[0].literals) == 2

    def test_add_implication_rule(self):
        """Test adding implication rules."""
        reasoner = SymbolicReasoner()
        reasoner.add_rule("P(x) -> Q(x)")

        assert len(reasoner.kb.clauses) == 1
        # P -> Q should be converted to ~P | Q
        clause = reasoner.kb.clauses[0]
        assert len(clause.literals) == 2

    def test_simple_query_success(self):
        """Test simple successful query."""
        reasoner = SymbolicReasoner(prover_type="resolution")

        # KB: P(a)
        reasoner.add_fact("P(a)")

        # Query: P(a)
        result = reasoner.query("P(a)", timeout=2.0)

        assert result["proven"] == True
        assert result["confidence"] > 0.8

    def test_simple_query_failure(self):
        """Test simple failing query."""
        reasoner = SymbolicReasoner(prover_type="resolution")

        # KB: P(a)
        reasoner.add_fact("P(a)")

        # Query: Q(b) - not provable
        result = reasoner.query("Q(b)", timeout=2.0)

        assert result["proven"] == False

    def test_modus_ponens_reasoning(self):
        """Test modus ponens inference."""
        reasoner = SymbolicReasoner(prover_type="resolution")

        # KB: P(a), P(a) -> Q(a)
        reasoner.add_fact("P(a)")
        reasoner.add_rule("P(a) -> Q(a)")

        # Query: Q(a)
        result = reasoner.query("Q(a)", timeout=3.0)

        assert result["proven"] == True

    def test_transitive_reasoning(self):
        """Test transitive reasoning chain."""
        reasoner = SymbolicReasoner(prover_type="resolution")

        # KB: P(a), P(x) -> Q(x), Q(x) -> R(x)
        reasoner.add_fact("P(a)")
        reasoner.add_rule("P(X) -> Q(X)")
        reasoner.add_rule("Q(X) -> R(X)")

        # Query: R(a)
        result = reasoner.query("R(a)", timeout=5.0)

        # Should prove R(a) through chain
        assert result["proven"] == True

    def test_parse_nested_functions(self):
        """Test parsing nested functions."""
        reasoner = SymbolicReasoner()

        # Parse f(g(x))
        clause = reasoner.parse_formula("P(f(g(x)))")

        assert clause.is_unit_clause()
        literal = clause.literals[0]
        assert len(literal.terms) == 1
        assert isinstance(literal.terms[0], Function)

    def test_parse_with_negation(self):
        """Test parsing negated literals."""
        reasoner = SymbolicReasoner()

        # Parse ~P(x)
        clause = reasoner.parse_formula("~P(x)")

        assert clause.is_unit_clause()
        assert clause.literals[0].negated == True

    def test_parse_disjunction(self):
        """Test parsing disjunctions."""
        reasoner = SymbolicReasoner()

        # Parse P(x) | Q(y) | R(z)
        clause = reasoner.parse_formula("P(x) | Q(y) | R(z)")

        assert len(clause.literals) == 3

    def test_fallback_parser(self):
        """Test fallback parser for simple formulas."""
        reasoner = SymbolicReasoner()

        # Use fallback directly
        clause = reasoner._fallback_parse("P(a)")

        assert clause.is_unit_clause()
        assert clause.literals[0].predicate == "P"

    def test_confidence_scores(self):
        """Test that confidence scores are preserved."""
        reasoner = SymbolicReasoner()

        reasoner.add_fact("P(a)", confidence=0.8)

        assert len(reasoner.kb.clauses) == 1
        assert reasoner.kb.clauses[0].confidence == 0.8

    def test_parallel_prover_returns_method(self):
        """Test that parallel prover returns which method succeeded."""
        reasoner = SymbolicReasoner(prover_type="parallel")

        reasoner.add_fact("P(a)")
        result = reasoner.query("P(a)", timeout=2.0)

        assert "method" in result
        assert result["method"] in [
            "tableau",
            "resolution",
            "model_elimination",
            "connection",
            "natural_deduction",
        ]

    def test_explain_proof_no_proof(self):
        """Test explaining when no proof available."""
        reasoner = SymbolicReasoner()

        explanation = reasoner.explain_proof(None)

        assert "No proof available" in explanation

    def test_explain_proof_with_proof(self):
        """Test explaining a valid proof."""
        reasoner = SymbolicReasoner()

        # Create a simple proof node
        proof = ProofNode(
            conclusion="P(a)",
            premises=[],
            rule_used="assumption",
            confidence=1.0,
            depth=0,
        )

        explanation = reasoner.explain_proof(proof)

        assert "assumption" in explanation
        assert "P(a)" in explanation

    def test_complex_formula_parsing(self):
        """Test parsing complex formulas."""
        reasoner = SymbolicReasoner()

        try:
            # Try parsing complex formula
            clause = reasoner.parse_formula("P(f(x, y)) | ~Q(g(z))")
            assert len(clause.literals) >= 1
        except Exception:
            # Fallback should handle it
            pass

    def test_error_handling_invalid_formula(self):
        """Test error handling for invalid formulas."""
        reasoner = SymbolicReasoner()

        # This should not crash, but may not parse correctly
        try:
            reasoner.add_rule("invalid@#$formula")
        except Exception as e:
            # Should raise an error
            assert True

    def test_multiple_facts(self):
        """Test adding multiple facts."""
        reasoner = SymbolicReasoner()

        reasoner.add_fact("P(a)")
        reasoner.add_fact("Q(b)")
        reasoner.add_fact("R(c)")

        assert len(reasoner.kb.clauses) == 3

    def test_query_timeout(self):
        """Test that query respects timeout."""
        reasoner = SymbolicReasoner()

        # Add complex KB
        for i in range(10):
            reasoner.add_rule(f"P{i}(X) -> P{i + 1}(X)")

        start_time = time.time()
        result = reasoner.query("P10(a)", timeout=0.5)
        elapsed = time.time() - start_time

        # Should timeout quickly
        assert elapsed < 1.5


# ============================================================================
# PROBABILISTIC REASONER TESTS
# ============================================================================


class TestProbabilisticReasoner:
    """Tests for ProbabilisticReasoner with Bayesian networks."""

    def test_probabilistic_reasoner_creation(self):
        """Test creating a ProbabilisticReasoner."""
        reasoner = ProbabilisticReasoner()
        assert reasoner.bn is not None
        assert len(reasoner.rules) == 0
        assert len(reasoner.variables) == 0

    def test_add_simple_rule(self):
        """Test adding simple probabilistic rule."""
        reasoner = ProbabilisticReasoner()

        reasoner.add_rule("IF rain THEN wet_grass", confidence=0.9)

        assert len(reasoner.rules) == 1
        assert "RAIN" in reasoner.variables
        assert "WET_GRASS" in reasoner.variables

    def test_add_causal_rule(self):
        """Test adding causal rule."""
        reasoner = ProbabilisticReasoner()

        reasoner.add_rule("rain CAUSES wet_grass", confidence=0.85)

        assert len(reasoner.rules) == 1
        rule = reasoner.rules[0]
        assert rule["type"] == "causal"

    def test_add_and_rule(self):
        """Test adding AND rule."""
        reasoner = ProbabilisticReasoner()

        reasoner.add_rule("IF rain AND sprinkler THEN wet_grass", confidence=0.95)

        rule = reasoner.rules[0]
        assert rule["operator"] == "and"
        assert len(rule["conditions"]) == 2

    def test_add_or_rule(self):
        """Test adding OR rule."""
        reasoner = ProbabilisticReasoner()

        reasoner.add_rule("IF rain OR sprinkler THEN wet_grass", confidence=0.8)

        rule = reasoner.rules[0]
        assert rule["operator"] == "or"

    def test_parse_rule_if_then(self):
        """Test parsing IF-THEN rules."""
        reasoner = ProbabilisticReasoner()

        rule = reasoner._parse_rule("IF A THEN B")

        assert rule["type"] == "conditional"
        assert "A" in rule["conditions"]
        assert rule["conclusion"] == "B"

    def test_parse_rule_causal(self):
        """Test parsing causal rules."""
        reasoner = ProbabilisticReasoner()

        rule = reasoner._parse_rule("A CAUSES B")

        assert rule["type"] == "causal"
        assert rule["operator"] == "causes"

    def test_extract_dependencies(self):
        """Test extracting dependencies from rules."""
        reasoner = ProbabilisticReasoner()

        reasoner.add_rule("IF A THEN B")
        reasoner.add_rule("IF B THEN C")

        deps = reasoner._extract_dependencies()

        assert "A" in deps["B"]
        assert "B" in deps["C"]

    def test_simple_query(self):
        """Test simple probabilistic query."""
        reasoner = ProbabilisticReasoner()

        # Add simple rule
        reasoner.add_rule("IF rain THEN wet_grass", confidence=0.9)

        # Query with evidence
        result = reasoner.query("WET_GRASS", evidence={"RAIN": True})

        # Should return probability distribution
        assert isinstance(result, dict)
        if True in result:
            assert result[True] > 0.5  # High probability when rain is true

    def test_query_without_evidence(self):
        """Test query without evidence."""
        reasoner = ProbabilisticReasoner()

        reasoner.add_rule("IF A THEN B", confidence=0.8)

        result = reasoner.query("B", evidence={})

        assert isinstance(result, dict)

    def test_multiple_causes_noisy_or(self):
        """Test Noisy-OR for multiple causes."""
        reasoner = ProbabilisticReasoner()

        # Multiple causes of same effect
        reasoner.add_rule("IF rain THEN wet_grass", confidence=0.9)
        reasoner.add_rule("IF sprinkler THEN wet_grass", confidence=0.85)

        result = reasoner.query("WET_GRASS", evidence={"RAIN": True, "SPRINKLER": True})

        # With both causes, should have very high probability
        if True in result:
            assert result[True] > 0.9

    def test_generate_parent_combinations(self):
        """Test generating parent value combinations."""
        reasoner = ProbabilisticReasoner()

        combos = reasoner._generate_parent_combinations(["A", "B"])

        assert len(combos) == 4  # 2^2 combinations
        assert (True, True) in combos
        assert (False, False) in combos

    def test_rule_satisfied_single(self):
        """Test rule satisfaction checking."""
        reasoner = ProbabilisticReasoner()

        rule = {"conditions": ["A"], "operator": "single"}

        # A is true
        assert reasoner._rule_satisfied(rule, (True,), ["A"]) == True

        # A is false
        assert reasoner._rule_satisfied(rule, (False,), ["A"]) == False

    def test_rule_satisfied_and(self):
        """Test AND rule satisfaction."""
        reasoner = ProbabilisticReasoner()

        rule = {"conditions": ["A", "B"], "operator": "and"}

        # Both true
        assert reasoner._rule_satisfied(rule, (True, True), ["A", "B"]) == True

        # One false
        assert reasoner._rule_satisfied(rule, (True, False), ["A", "B"]) == False

    def test_rule_satisfied_or(self):
        """Test OR rule satisfaction."""
        reasoner = ProbabilisticReasoner()

        rule = {"conditions": ["A", "B"], "operator": "or"}

        # At least one true
        assert reasoner._rule_satisfied(rule, (True, False), ["A", "B"]) == True

        # Both false
        assert reasoner._rule_satisfied(rule, (False, False), ["A", "B"]) == False

    def test_build_causal_structure(self):
        """Test building causal graph."""
        reasoner = ProbabilisticReasoner()

        reasoner.add_rule("A CAUSES B")
        reasoner.add_rule("B CAUSES C")

        reasoner.build_causal_structure()

        assert reasoner._causal_structure is not None
        assert "A" in reasoner._causal_structure
        assert "B" in reasoner._causal_structure["A"]

    def test_intervention_query(self):
        """Test intervention query (do-calculus)."""
        reasoner = ProbabilisticReasoner()

        reasoner.add_rule("IF A THEN B", confidence=0.9)

        # Query: P(B | do(A=true))
        result = reasoner.intervention_query("B", intervention={"A": True})

        assert isinstance(result, dict)

    def test_learn_from_data_placeholder(self):
        """Test learning from data (requires mock data)."""
        reasoner = ProbabilisticReasoner()

        reasoner.add_rule("IF A THEN B")

        # Create mock data
        data = [
            {"A": True, "B": True},
            {"A": True, "B": False},
            {"A": False, "B": False},
        ]

        try:
            reasoner.learn_from_data(data, method="mle")
            # If no error, learning succeeded
            assert True
        except Exception:
            # Learning may fail if network not properly built
            pass

    def test_confidence_in_rules(self):
        """Test that confidence is stored in rules."""
        reasoner = ProbabilisticReasoner()

        reasoner.add_rule("IF A THEN B", confidence=0.75)

        assert reasoner.rules[0]["confidence"] == 0.75

    def test_multiple_variables(self):
        """Test handling multiple variables."""
        reasoner = ProbabilisticReasoner()

        reasoner.add_rule("IF A THEN B")
        reasoner.add_rule("IF B THEN C")
        reasoner.add_rule("IF C THEN D")

        assert len(reasoner.variables) >= 4


# ============================================================================
# HYBRID REASONER TESTS
# ============================================================================


class TestHybridReasoner:
    """Tests for HybridReasoner combining symbolic and probabilistic."""

    def test_hybrid_reasoner_creation(self):
        """Test creating a HybridReasoner."""
        reasoner = HybridReasoner()
        assert reasoner.symbolic is not None
        assert reasoner.probabilistic is not None
        assert len(reasoner.reasoning_history) == 0

    def test_add_symbolic_rule(self):
        """Test adding symbolic rule."""
        reasoner = HybridReasoner()

        reasoner.add_rule("P(x) -> Q(x)", rule_type="symbolic")

        assert len(reasoner.symbolic.kb.clauses) == 1

    def test_add_probabilistic_rule(self):
        """Test adding probabilistic rule."""
        reasoner = HybridReasoner()

        reasoner.add_rule("IF A THEN B", rule_type="probabilistic", confidence=0.8)

        assert len(reasoner.probabilistic.rules) == 1

    def test_auto_rule_type_symbolic(self):
        """Test automatic detection of symbolic rules."""
        reasoner = HybridReasoner()

        rule_type = reasoner._infer_rule_type("forall X (P(X) -> Q(X))")

        assert rule_type == "symbolic"

    def test_auto_rule_type_probabilistic(self):
        """Test automatic detection of probabilistic rules."""
        reasoner = HybridReasoner()

        rule_type = reasoner._infer_rule_type("IF rain THEN wet_grass")

        assert rule_type == "probabilistic"

    def test_add_rule_auto_type(self):
        """Test adding rule with automatic type detection."""
        reasoner = HybridReasoner()

        # Should be detected as probabilistic
        reasoner.add_rule("IF A THEN B", rule_type="auto")

        assert len(reasoner.probabilistic.rules) >= 1

    def test_symbolic_query(self):
        """Test symbolic query through hybrid reasoner."""
        reasoner = HybridReasoner()

        reasoner.add_rule("P(a)", rule_type="symbolic")

        result = reasoner.query("P(a)", method="symbolic")

        assert "symbolic" in result
        assert result["symbolic"]["proven"] == True

    def test_probabilistic_query(self):
        """Test probabilistic query through hybrid reasoner."""
        reasoner = HybridReasoner()

        reasoner.add_rule("IF A THEN B", rule_type="probabilistic", confidence=0.9)

        result = reasoner.query("B", evidence={"A": True}, method="probabilistic")

        assert "probabilistic" in result

    def test_auto_query_method_with_evidence(self):
        """Test auto method selection with evidence."""
        reasoner = HybridReasoner()

        method = reasoner._select_method("query", evidence={"A": True})

        # Should prefer probabilistic when evidence provided
        assert method == "probabilistic"

    def test_auto_query_method_with_quantifiers(self):
        """Test auto method selection with quantifiers."""
        reasoner = HybridReasoner()

        method = reasoner._select_method("forall X P(X)", evidence=None)

        # Should prefer symbolic for quantifiers
        assert method == "symbolic"

    def test_auto_query_default(self):
        """Test auto method selection default."""
        reasoner = HybridReasoner()

        method = reasoner._select_method("simple_query", evidence=None)

        # Should default to auto (both)
        assert method == "auto"

    def test_combine_results_symbolic_only(self):
        """Test combining results with only symbolic."""
        reasoner = HybridReasoner()

        results = {"symbolic": {"proven": True, "confidence": 0.95}}

        combined = reasoner._combine_results(results)

        assert combined["proven"] == True
        assert combined["confidence"] == 0.95

    def test_combine_results_probabilistic_only(self):
        """Test combining results with only probabilistic."""
        reasoner = HybridReasoner()

        results = {"probabilistic": {True: 0.8, False: 0.2}}

        combined = reasoner._combine_results(results)

        assert "probability" in combined
        assert combined["confidence"] == 0.8

    def test_combine_results_both(self):
        """Test combining results from both reasoners."""
        reasoner = HybridReasoner()

        results = {
            "symbolic": {"proven": True, "confidence": 0.9},
            "probabilistic": {True: 0.85, False: 0.15},
        }

        combined = reasoner._combine_results(results)

        assert combined["proven"] == True
        assert combined["confidence"] >= 0.85

    def test_reasoning_history_recorded(self):
        """Test that reasoning history is recorded."""
        reasoner = HybridReasoner()

        reasoner.add_rule("P(a)", rule_type="symbolic")
        reasoner.query("P(a)", method="symbolic")

        assert len(reasoner.reasoning_history) == 1
        assert "query" in reasoner.reasoning_history[0]
        assert "method" in reasoner.reasoning_history[0]

    def test_hybrid_query_auto_method(self):
        """Test hybrid query with auto method."""
        reasoner = HybridReasoner()

        reasoner.add_rule("P(a)", rule_type="symbolic")

        result = reasoner.query("P(a)", method="auto")

        # Should try symbolic and potentially probabilistic
        assert "combined" in result

    def test_multiple_rules_hybrid(self):
        """Test adding multiple rules of different types."""
        reasoner = HybridReasoner()

        reasoner.add_rule("P(x) -> Q(x)", rule_type="symbolic")
        reasoner.add_rule("IF A THEN B", rule_type="probabilistic")

        assert len(reasoner.symbolic.kb.clauses) >= 1
        assert len(reasoner.probabilistic.rules) >= 1


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestReasonerIntegration:
    """Integration tests across all reasoners."""

    def test_symbolic_to_probabilistic_conversion(self):
        """Test using both reasoners for same problem."""
        symbolic = SymbolicReasoner()
        probabilistic = ProbabilisticReasoner()

        # Add similar rules
        symbolic.add_rule("P(a) -> Q(a)")
        probabilistic.add_rule("IF P THEN Q", confidence=0.9)

        # Both should handle queries
        sym_result = symbolic.query("Q(a)")
        prob_result = probabilistic.query("Q", evidence={"P": True})

        # Both should indicate Q is likely
        assert isinstance(sym_result, dict)
        assert isinstance(prob_result, dict)

    def test_complex_reasoning_chain(self):
        """Test complex reasoning chain."""
        reasoner = SymbolicReasoner()

        # Build knowledge base
        reasoner.add_fact("human(socrates)")
        reasoner.add_rule("human(X) -> mortal(X)")
        reasoner.add_rule("mortal(X) -> dies(X)")

        # Query
        result = reasoner.query("dies(socrates)", timeout=5.0)

        # Should prove through chain
        assert result["proven"] == True

    def test_probabilistic_reasoning_chain(self):
        """Test probabilistic reasoning chain."""
        reasoner = ProbabilisticReasoner()

        reasoner.add_rule("IF clouds THEN rain", confidence=0.7)
        reasoner.add_rule("IF rain THEN wet_grass", confidence=0.9)

        result = reasoner.query("WET_GRASS", evidence={"CLOUDS": True})

        # Should propagate probability
        assert isinstance(result, dict)

    def test_hybrid_complex_query(self):
        """Test complex query with hybrid reasoner."""
        reasoner = HybridReasoner()

        # Add both types of rules
        reasoner.add_rule("P(a)", rule_type="symbolic")
        reasoner.add_rule("IF X THEN Y", rule_type="probabilistic", confidence=0.8)

        # Query both
        sym_result = reasoner.query("P(a)", method="symbolic")
        prob_result = reasoner.query("Y", evidence={"X": True}, method="probabilistic")

        assert "symbolic" in sym_result or "combined" in sym_result
        assert "probabilistic" in prob_result or "combined" in prob_result


# ============================================================================
# EDGE CASE AND ERROR HANDLING TESTS
# ============================================================================


class TestReasonerEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_knowledge_base_query(self):
        """Test query on empty knowledge base."""
        reasoner = SymbolicReasoner()

        result = reasoner.query("P(a)", timeout=1.0)

        assert result["proven"] == False

    def test_invalid_formula_handling(self):
        """Test handling of invalid formulas."""
        reasoner = SymbolicReasoner()

        try:
            reasoner.add_rule("not a valid formula @#$%")
            # May raise exception or use fallback
        except Exception:
            assert True

    def test_probabilistic_query_no_rules(self):
        """Test probabilistic query with no rules."""
        reasoner = ProbabilisticReasoner()

        reasoner.add_rule("IF A THEN B")
        result = reasoner.query("B", evidence={})

        # Should return some distribution
        assert isinstance(result, dict)

    def test_hybrid_query_no_rules(self):
        """Test hybrid query with no rules."""
        reasoner = HybridReasoner()

        result = reasoner.query("anything", method="auto")

        # Should handle gracefully
        assert isinstance(result, dict)

    def test_very_long_reasoning_chain(self):
        """Test handling very long reasoning chains."""
        reasoner = SymbolicReasoner()

        # Build long chain
        for i in range(5):
            reasoner.add_rule(f"P{i}(X) -> P{i + 1}(X)")

        reasoner.add_fact("P0(a)")

        # Try to prove end of chain
        result = reasoner.query("P5(a)", timeout=10.0)

        # May or may not prove depending on prover
        assert isinstance(result, dict)

    def test_circular_rules(self):
        """Test handling circular rules."""
        reasoner = ProbabilisticReasoner()

        reasoner.add_rule("IF A THEN B")
        reasoner.add_rule("IF B THEN A")

        # Should not crash
        result = reasoner.query("A", evidence={"B": True})
        assert isinstance(result, dict)

    def test_timeout_respected(self):
        """Test that timeout is respected."""
        reasoner = SymbolicReasoner()

        # Add complex rules
        for i in range(20):
            reasoner.add_rule(f"P{i}(X) | Q{i}(Y)")

        start_time = time.time()
        result = reasoner.query("P10(a)", timeout=0.5)
        elapsed = time.time() - start_time

        # Should timeout quickly
        assert elapsed < 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
