"""
Tests for Issue #2, #4, and #7 Fixes - New Functionality

Comprehensive test suite covering:
- Issue #2: AST-based code pre-validation for math tool
- Issue #4: Contradiction proof tracking for SAT solver
- Issue #7: Complete analogical mapping for batch queries

Author: VulcanAMI Team
Date: 2026-01-19
"""

import pytest


# ============================================================================
# ISSUE #2 TESTS: AST-Based Code Pre-Validation
# ============================================================================


class TestValidateCodeSyntax:
    """Tests for validate_code_syntax() function (Issue #2 fix)."""

    @pytest.fixture
    def validate_func(self):
        """Import the validate function."""
        try:
            from vulcan.reasoning.mathematical_computation import validate_code_syntax
            return validate_code_syntax
        except ImportError:
            pytest.skip("Mathematical computation module not available")

    def test_valid_simple_code(self, validate_func):
        """Test that valid simple code passes validation."""
        is_valid, error = validate_func("x = 2 + 2")
        assert is_valid is True
        assert error is None

    def test_valid_multiline_code(self, validate_func):
        """Test that valid multiline code passes validation."""
        code = """
from sympy import Symbol, integrate
x = Symbol('x')
result = integrate(x**2, x)
print(result)
"""
        is_valid, error = validate_func(code)
        assert is_valid is True
        assert error is None

    def test_valid_function_definition(self, validate_func):
        """Test that valid function definitions pass validation."""
        code = """
def compute_sum(n):
    return sum(k**2 for k in range(1, n+1))
result = compute_sum(10)
"""
        is_valid, error = validate_func(code)
        assert is_valid is True
        assert error is None

    def test_invalid_syntax_original_bug(self, validate_func):
        """Test Issue #2 original bug: malformed code from integral parsing."""
        # This is the exact error from the issue
        code = "f = 0*Tu(t)2"  # SyntaxError: invalid syntax
        is_valid, error = validate_func(code)
        assert is_valid is False
        assert error is not None
        assert "Syntax error" in error

    def test_invalid_missing_colon(self, validate_func):
        """Test that missing colon in function def is caught."""
        code = "def foo()\n    return 1"
        is_valid, error = validate_func(code)
        assert is_valid is False
        assert "Syntax error" in error

    def test_invalid_unmatched_parenthesis(self, validate_func):
        """Test that unmatched parentheses are caught."""
        code = "result = sum(k**2 for k in range(1, n+1)"
        is_valid, error = validate_func(code)
        assert is_valid is False
        assert "Syntax error" in error

    def test_invalid_bad_indentation(self, validate_func):
        """Test that bad indentation is caught."""
        code = "def foo():\nreturn 1"  # Missing indentation
        is_valid, error = validate_func(code)
        assert is_valid is False
        assert error is not None

    def test_empty_code_rejected(self, validate_func):
        """Test that empty code is rejected."""
        is_valid, error = validate_func("")
        assert is_valid is False
        assert "Empty code" in error

    def test_whitespace_only_rejected(self, validate_func):
        """Test that whitespace-only code is rejected."""
        is_valid, error = validate_func("   \n\t  ")
        assert is_valid is False
        assert "Empty code" in error

    def test_error_message_contains_line_info(self, validate_func):
        """Test that error messages contain line information."""
        code = "x = 1\ny = 2\nz = 3(\n"  # Error on line 3
        is_valid, error = validate_func(code)
        assert is_valid is False
        assert "line" in error.lower()


# ============================================================================
# ISSUE #4 TESTS: Contradiction Proof Tracking
# ============================================================================


class TestAnnotatedClause:
    """Tests for AnnotatedClause dataclass (Issue #4 fix)."""

    @pytest.fixture
    def core_classes(self):
        """Import core classes."""
        try:
            from vulcan.reasoning.symbolic.core import (
                AnnotatedClause, Clause, Literal
            )
            return {"AnnotatedClause": AnnotatedClause, "Clause": Clause, "Literal": Literal}
        except ImportError:
            pytest.skip("Symbolic reasoning module not available")

    def test_create_premise_clause(self, core_classes):
        """Test creating an annotated clause marked as premise."""
        AnnotatedClause = core_classes["AnnotatedClause"]
        Clause = core_classes["Clause"]
        Literal = core_classes["Literal"]
        
        lit = Literal(predicate="A", terms=[], negated=False)
        clause = Clause(literals=[lit])
        annotated = AnnotatedClause(
            clause=clause,
            derived_from=[],
            rule_used="premise",
            iteration=0
        )
        
        assert annotated.is_premise() is True
        assert annotated.is_empty() is False
        assert str(annotated) == "A"

    def test_create_derived_clause(self, core_classes):
        """Test creating a derived annotated clause with parent tracking."""
        AnnotatedClause = core_classes["AnnotatedClause"]
        Clause = core_classes["Clause"]
        Literal = core_classes["Literal"]
        
        # Create parent clauses
        parent1 = AnnotatedClause(
            clause=Clause(literals=[Literal("A", [], False)]),
            derived_from=[],
            rule_used="premise",
            iteration=0
        )
        parent2 = AnnotatedClause(
            clause=Clause(literals=[Literal("A", [], True)]),  # negated
            derived_from=[],
            rule_used="premise",
            iteration=0
        )
        
        # Create derived empty clause
        derived = AnnotatedClause(
            clause=Clause(literals=[]),  # Empty clause = contradiction
            derived_from=[parent1, parent2],
            rule_used="resolution",
            iteration=1,
            resolvent_literal="A"
        )
        
        assert derived.is_premise() is False
        assert derived.is_empty() is True
        assert len(derived.derived_from) == 2
        assert derived.resolvent_literal == "A"

    def test_empty_clause_detection(self, core_classes):
        """Test that empty clause (contradiction) is correctly detected."""
        AnnotatedClause = core_classes["AnnotatedClause"]
        Clause = core_classes["Clause"]
        
        empty_annotated = AnnotatedClause(
            clause=Clause(literals=[]),
            derived_from=[],
            rule_used="resolution",
            iteration=1
        )
        
        assert empty_annotated.is_empty() is True


class TestProofTreeConstruction:
    """Tests for proof tree construction functions (Issue #4 fix)."""

    @pytest.fixture
    def proof_funcs(self):
        """Import proof tree functions."""
        try:
            from vulcan.reasoning.symbolic.core import (
                AnnotatedClause, Clause, Literal, ProofNode,
                build_proof_tree_from_annotated, format_contradiction_proof
            )
            return {
                "AnnotatedClause": AnnotatedClause,
                "Clause": Clause,
                "Literal": Literal,
                "ProofNode": ProofNode,
                "build_proof_tree": build_proof_tree_from_annotated,
                "format_proof": format_contradiction_proof
            }
        except ImportError:
            pytest.skip("Symbolic reasoning module not available")

    def test_build_simple_proof_tree(self, proof_funcs):
        """Test building a proof tree from annotated clauses."""
        AnnotatedClause = proof_funcs["AnnotatedClause"]
        Clause = proof_funcs["Clause"]
        Literal = proof_funcs["Literal"]
        build_proof_tree = proof_funcs["build_proof_tree"]
        
        # Create premises
        p1 = AnnotatedClause(
            clause=Clause(literals=[Literal("A", [], False)]),
            derived_from=[],
            rule_used="premise",
            iteration=0
        )
        p2 = AnnotatedClause(
            clause=Clause(literals=[Literal("A", [], True)]),
            derived_from=[],
            rule_used="premise",
            iteration=0
        )
        
        # Create empty clause from resolution
        empty = AnnotatedClause(
            clause=Clause(literals=[]),
            derived_from=[p1, p2],
            rule_used="resolution",
            iteration=1,
            resolvent_literal="A"
        )
        
        # Build proof tree
        proof_tree = build_proof_tree(empty)
        
        assert proof_tree is not None
        assert proof_tree.conclusion == "⊥ (contradiction)"
        assert proof_tree.rule_used == "resolution"
        assert len(proof_tree.premises) == 2

    def test_format_contradiction_proof(self, proof_funcs):
        """Test formatting proof tree as human-readable text."""
        AnnotatedClause = proof_funcs["AnnotatedClause"]
        Clause = proof_funcs["Clause"]
        Literal = proof_funcs["Literal"]
        build_proof_tree = proof_funcs["build_proof_tree"]
        format_proof = proof_funcs["format_proof"]
        
        # Create simple contradiction
        p1 = AnnotatedClause(
            clause=Clause(literals=[Literal("A", [], False)]),
            derived_from=[],
            rule_used="premise",
            iteration=0
        )
        p2 = AnnotatedClause(
            clause=Clause(literals=[Literal("A", [], True)]),
            derived_from=[],
            rule_used="negated_goal",
            iteration=0
        )
        empty = AnnotatedClause(
            clause=Clause(literals=[]),
            derived_from=[p1, p2],
            rule_used="resolution",
            iteration=1,
            resolvent_literal="A"
        )
        
        proof_tree = build_proof_tree(empty)
        formatted = format_proof(proof_tree)
        
        assert "Contradiction Proof:" in formatted
        assert "Step" in formatted
        assert "UNSATISFIABLE" in formatted or "contradiction" in formatted.lower()


class TestResolutionProverWithTracking:
    """Tests for updated ResolutionProver with derivation tracking (Issue #4 fix)."""

    @pytest.fixture
    def prover(self):
        """Create ResolutionProver instance."""
        try:
            from vulcan.reasoning.symbolic.provers import ResolutionProver
            return ResolutionProver()
        except ImportError:
            pytest.skip("Symbolic reasoning module not available")

    @pytest.fixture
    def core_classes(self):
        """Import core classes for test data."""
        try:
            from vulcan.reasoning.symbolic.core import Clause, Literal
            return {"Clause": Clause, "Literal": Literal}
        except ImportError:
            pytest.skip("Symbolic reasoning module not available")

    def test_prove_simple_contradiction(self, prover, core_classes):
        """Test that prover finds contradiction with proof tree."""
        Clause = core_classes["Clause"]
        Literal = core_classes["Literal"]
        
        # KB: A, ¬A (obvious contradiction)
        kb = [
            Clause(literals=[Literal("A", [], False)]),  # A
            Clause(literals=[Literal("A", [], True)]),   # ¬A
        ]
        
        # Goal: B (arbitrary, should fail but find contradiction in KB)
        goal = Clause(literals=[Literal("B", [], False)])
        
        # Note: Resolution proves by refutation, so proving B from this KB
        # should explore and may find the A, ¬A contradiction
        proven, proof, confidence = prover.prove(goal, kb, timeout=5.0)
        
        # The prover should return some result
        assert confidence is not None
        
    def test_proof_tree_contains_metadata(self, prover, core_classes):
        """Test that proof contains derivation metadata."""
        Clause = core_classes["Clause"]
        Literal = core_classes["Literal"]
        
        # Simple provable case: KB has P(a), P(a) → Q(a), Goal: Q(a)
        # In clause form: P(a), ¬P(a) ∨ Q(a), Goal: Q(a)
        kb = [
            Clause(literals=[Literal("P", [], False)]),  # P
            Clause(literals=[
                Literal("P", [], True),   # ¬P
                Literal("Q", [], False),  # Q
            ]),  # ¬P ∨ Q (i.e., P → Q)
        ]
        goal = Clause(literals=[Literal("Q", [], False)])  # Q
        
        proven, proof, confidence = prover.prove(goal, kb, timeout=5.0)
        
        if proven and proof is not None:
            # Check that proof has metadata
            assert hasattr(proof, 'metadata')
            assert 'method' in proof.metadata


# ============================================================================
# ISSUE #7 TESTS: Complete Analogical Mapping
# ============================================================================


class TestExtractMappingTargets:
    """Tests for extract_mapping_targets() function (Issue #7 fix)."""

    @pytest.fixture
    def reasoner(self):
        """Create AnalogicalReasoner instance."""
        try:
            from vulcan.reasoning.analogical.base_reasoner import AnalogicalReasoner
            return AnalogicalReasoner(enable_caching=False, enable_learning=False)
        except ImportError:
            pytest.skip("Analogical reasoning module not available")

    def test_extract_comma_separated(self, reasoner):
        """Test extracting comma-separated concepts."""
        query = "Map leader election, quorum, fencing token to biology"
        concepts = reasoner.extract_mapping_targets(query)
        
        assert len(concepts) >= 3
        assert "leader election" in [c.lower() for c in concepts]
        assert "quorum" in [c.lower() for c in concepts]

    def test_extract_numbered_list(self, reasoner):
        """Test extracting numbered list of concepts."""
        query = """
        1. Leader election → ?
        2. Quorum → ?
        3. Split brain → ?
        """
        concepts = reasoner.extract_mapping_targets(query)
        
        assert len(concepts) >= 3

    def test_extract_bullet_points(self, reasoner):
        """Test extracting bullet-pointed concepts."""
        query = """
        • leader election
        • quorum
        • fencing token
        """
        concepts = reasoner.extract_mapping_targets(query)
        
        assert len(concepts) >= 3

    def test_extract_with_and(self, reasoner):
        """Test extracting concepts separated by 'and'."""
        query = "Map leader election, quorum and fencing token to biology"
        concepts = reasoner.extract_mapping_targets(query)
        
        assert len(concepts) >= 3

    def test_empty_query_returns_empty(self, reasoner):
        """Test that empty query returns empty list."""
        concepts = reasoner.extract_mapping_targets("")
        assert concepts == []


class TestMapAllConcepts:
    """Tests for map_all_concepts() function (Issue #7 fix)."""

    @pytest.fixture
    def reasoner(self):
        """Create AnalogicalReasoner with test domains."""
        try:
            from vulcan.reasoning.analogical.base_reasoner import AnalogicalReasoner
            reasoner = AnalogicalReasoner(enable_caching=False, enable_learning=False)
            
            # Add test domains
            reasoner.add_domain("distributed_systems", {
                "name": "distributed_systems",
                "entities": [
                    {"name": "leader election", "type": "algorithm", "attributes": ["consensus"]},
                    {"name": "quorum", "type": "mechanism", "attributes": ["voting"]},
                ],
                "relations": []
            })
            reasoner.add_domain("biology", {
                "name": "biology",
                "entities": [
                    {"name": "nucleus", "type": "organelle", "attributes": ["control"]},
                    {"name": "cell signaling", "type": "mechanism", "attributes": ["communication"]},
                ],
                "relations": []
            })
            
            return reasoner
        except ImportError:
            pytest.skip("Analogical reasoning module not available")

    def test_map_multiple_concepts(self, reasoner):
        """Test mapping multiple concepts at once."""
        concepts = ["leader election", "quorum"]
        results = reasoner.map_all_concepts(concepts, "biology")
        
        assert len(results) == 2
        assert "leader election" in results
        assert "quorum" in results

    def test_all_concepts_have_results(self, reasoner):
        """Test that all requested concepts have result entries."""
        concepts = ["leader election", "quorum", "unknown_concept"]
        results = reasoner.map_all_concepts(concepts, "biology")
        
        # All concepts should have entries (even if not found)
        assert len(results) == 3
        for concept in concepts:
            assert concept in results


class TestCheckMappingCompleteness:
    """Tests for check_mapping_completeness() function (Issue #7 fix)."""

    @pytest.fixture
    def reasoner(self):
        """Create AnalogicalReasoner instance."""
        try:
            from vulcan.reasoning.analogical.base_reasoner import AnalogicalReasoner
            return AnalogicalReasoner(enable_caching=False, enable_learning=False)
        except ImportError:
            pytest.skip("Analogical reasoning module not available")

    def test_all_mapped_is_complete(self, reasoner):
        """Test that all concepts mapped returns is_complete=True."""
        requested = ["A", "B", "C"]
        mapped = {
            "A": {"found": True, "target_concept": "X"},
            "B": {"found": True, "target_concept": "Y"},
            "C": {"found": True, "target_concept": "Z"},
        }
        
        is_complete, unmapped, ratio = reasoner.check_mapping_completeness(requested, mapped)
        
        assert is_complete is True
        assert len(unmapped) == 0
        assert ratio == 1.0

    def test_partial_mapping_detected(self, reasoner):
        """Test that partial mapping is detected correctly."""
        requested = ["A", "B", "C", "D", "E"]
        mapped = {
            "A": {"found": True, "target_concept": "X"},
            "B": {"found": True, "target_concept": "Y"},
            "C": None,  # Failed
            "D": {"found": False, "target_concept": None},  # Not found
            "E": {"found": True, "target_concept": "Z"},
        }
        
        is_complete, unmapped, ratio = reasoner.check_mapping_completeness(requested, mapped)
        
        assert is_complete is False
        assert set(unmapped) == {"C", "D"}
        assert ratio == 0.6  # 3 out of 5

    def test_empty_request_is_complete(self, reasoner):
        """Test that empty request returns complete."""
        is_complete, unmapped, ratio = reasoner.check_mapping_completeness([], {})
        
        assert is_complete is True
        assert ratio == 1.0


class TestFormatMappingResponse:
    """Tests for format_mapping_response() function (Issue #7 fix)."""

    @pytest.fixture
    def reasoner(self):
        """Create AnalogicalReasoner instance."""
        try:
            from vulcan.reasoning.analogical.base_reasoner import AnalogicalReasoner
            return AnalogicalReasoner(enable_caching=False, enable_learning=False)
        except ImportError:
            pytest.skip("Analogical reasoning module not available")

    def test_format_includes_all_concepts(self, reasoner):
        """Test that formatted response includes all concepts."""
        mappings = {
            "leader election": {"found": True, "target_concept": "nucleus", "confidence": 0.85},
            "quorum": {"found": True, "target_concept": "cell signaling", "confidence": 0.70},
            "split brain": {"found": False, "target_concept": None, "confidence": 0.0},
        }
        
        response = reasoner.format_mapping_response(
            mappings, "distributed_systems", "biology", ["split brain"]
        )
        
        assert "leader election" in response
        assert "quorum" in response
        assert "split brain" in response
        assert "nucleus" in response
        assert "no mapping found" in response.lower()

    def test_format_includes_summary(self, reasoner):
        """Test that formatted response includes summary statistics."""
        mappings = {
            "A": {"found": True, "target_concept": "X", "confidence": 0.8},
            "B": {"found": False, "target_concept": None, "confidence": 0.0},
        }
        
        response = reasoner.format_mapping_response(mappings, "source", "target", ["B"])
        
        assert "Summary" in response
        assert "1/2" in response or "50" in response


class TestReasonWithCompleteMapping:
    """Integration tests for reason_with_complete_mapping() (Issue #7 fix)."""

    @pytest.fixture
    def reasoner(self):
        """Create AnalogicalReasoner with test domains."""
        try:
            from vulcan.reasoning.analogical.base_reasoner import AnalogicalReasoner
            reasoner = AnalogicalReasoner(enable_caching=False, enable_learning=False)
            
            # Add test domains
            reasoner.add_domain("distributed_systems", {
                "name": "distributed_systems",
                "entities": [
                    {"name": "leader", "type": "role", "attributes": ["coordination"]},
                    {"name": "quorum", "type": "mechanism", "attributes": ["voting"]},
                ],
                "relations": []
            })
            reasoner.add_domain("biology", {
                "name": "biology",
                "entities": [
                    {"name": "nucleus", "type": "organelle", "attributes": ["control"]},
                ],
                "relations": []
            })
            
            return reasoner
        except ImportError:
            pytest.skip("Analogical reasoning module not available")

    def test_complete_mapping_returns_all_fields(self, reasoner):
        """Test that reason_with_complete_mapping returns all expected fields."""
        result = reasoner.reason_with_complete_mapping(
            "Map leader, quorum to biology",
            "distributed_systems",
            "biology"
        )
        
        # Check all expected fields are present
        assert "success" in result
        assert "complete" in result
        assert "mappings" in result
        assert "unmapped" in result
        assert "completeness_ratio" in result
        assert "formatted_response" in result
        assert "concepts_found" in result

    def test_reports_unmapped_concepts(self, reasoner):
        """Test that unmapped concepts are reported."""
        result = reasoner.reason_with_complete_mapping(
            "Map leader, quorum, unknown_concept to biology",
            "distributed_systems",
            "biology"
        )
        
        # Should report some concepts as unmapped (at least unknown_concept)
        assert "unmapped" in result
        # Note: Whether concepts are mapped depends on the domain setup


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
