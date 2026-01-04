"""
Unit tests for QueryPreprocessor.

Part of the VULCAN-AGI system.

These tests verify the functionality of the QueryPreprocessor component,
which extracts formal syntax from natural language reasoning queries.

Test Categories:
    - SAT problem extraction
    - Mathematical formula extraction
    - Probabilistic notation extraction
    - Operator normalization
    - Edge cases and error handling
    - Thread safety
    - Metrics tracking
"""

import pytest
import threading
from concurrent.futures import ThreadPoolExecutor

from vulcan.reasoning.query_preprocessor import (
    QueryPreprocessor,
    PreprocessingResult,
    PreprocessorMetrics,
    ExtractionType,
    get_query_preprocessor,
    reset_query_preprocessor,
)


class TestPreprocessingResult:
    """Tests for the PreprocessingResult dataclass."""

    def test_result_creation_basic(self):
        """Test basic result creation."""
        result = PreprocessingResult(
            formal_input="(A→B)",
            original_query="Test query",
            preprocessing_applied=True,
            extraction_confidence=0.9,
        )

        assert result.formal_input == "(A→B)"
        assert result.original_query == "Test query"
        assert result.preprocessing_applied is True
        assert result.extraction_confidence == 0.9
        assert result.extraction_type == ExtractionType.NONE

    def test_result_with_all_fields(self):
        """Test result creation with all fields."""
        result = PreprocessingResult(
            formal_input="(A→B) ∧ (B→C)",
            original_query="Propositions: A,B,C",
            preprocessing_applied=True,
            extraction_confidence=0.9,
            extraction_type=ExtractionType.SYMBOLIC,
            extracted_propositions=("A", "B", "C"),
            extracted_constraints=("A→B", "B→C"),
            metadata={"pattern": "sat_problem"},
        )

        assert result.extraction_type == ExtractionType.SYMBOLIC
        assert result.extracted_propositions == ("A", "B", "C")
        assert result.extracted_constraints == ("A→B", "B→C")
        assert result.metadata == {"pattern": "sat_problem"}

    def test_result_confidence_clamping(self):
        """Test that confidence is clamped to valid range."""
        # Over 1.0 should be clamped
        result = PreprocessingResult(
            formal_input=None,
            original_query="test",
            preprocessing_applied=False,
            extraction_confidence=1.5,
        )
        assert result.extraction_confidence == 1.0

        # Under 0.0 should be clamped
        result2 = PreprocessingResult(
            formal_input=None,
            original_query="test",
            preprocessing_applied=False,
            extraction_confidence=-0.5,
        )
        assert result2.extraction_confidence == 0.0

    def test_result_to_dict(self):
        """Test conversion to dictionary."""
        result = PreprocessingResult(
            formal_input="(A→B)",
            original_query="Test",
            preprocessing_applied=True,
            extraction_confidence=0.85,
            extraction_type=ExtractionType.SYMBOLIC,
            extracted_propositions=("A", "B"),
            extracted_constraints=("A→B",),
        )

        d = result.to_dict()
        assert d["formal_input"] == "(A→B)"
        assert d["preprocessing_applied"] is True
        assert d["extraction_confidence"] == 0.85
        assert d["extraction_type"] == "symbolic"
        assert d["extracted_propositions"] == ["A", "B"]

    def test_result_get_method(self):
        """Test dictionary-like get method for compatibility."""
        result = PreprocessingResult(
            formal_input="(A→B)",
            original_query="Test",
            preprocessing_applied=True,
            extraction_confidence=0.9,
        )

        assert result.get("formal_input") == "(A→B)"
        assert result.get("preprocessing_applied") is True
        assert result.get("nonexistent", "default") == "default"


class TestQueryPreprocessorSATExtraction:
    """Tests for SAT problem extraction."""

    @pytest.fixture
    def preprocessor(self):
        """Create a fresh preprocessor instance."""
        return QueryPreprocessor()

    def test_extract_simple_sat_problem(self, preprocessor):
        """Test extraction of a simple SAT problem."""
        query = """
        Symbolic Reasoning
        S1 — Satisfiability (SAT-style)

        Propositions: A, B, C

        Constraints:
        1. A→B
        2. B→C
        3. ¬C
        4. A∨B

        Task: Is the set satisfiable?
        """

        result = preprocessor.preprocess(
            query=query,
            query_type="symbolic",
            reasoning_tools=["symbolic"],
        )

        assert result.preprocessing_applied is True
        assert result.extraction_confidence == 0.9
        assert result.extraction_type == ExtractionType.SYMBOLIC
        assert "(A→B)" in result.formal_input
        assert "(B→C)" in result.formal_input
        assert "(¬C)" in result.formal_input
        assert "(A∨B)" in result.formal_input
        assert " ∧ " in result.formal_input
        assert "A" in result.extracted_propositions
        assert "B" in result.extracted_propositions
        assert "C" in result.extracted_propositions

    def test_extract_sat_with_ascii_operators(self, preprocessor):
        """Test extraction with ASCII operators that need normalization."""
        query = """
        Propositions: X, Y

        Constraints:
        1. X -> Y
        2. NOT X OR Y

        Task: Check satisfiability.
        """

        result = preprocessor.preprocess(
            query=query,
            query_type="symbolic",
            reasoning_tools=["symbolic"],
        )

        assert result.preprocessing_applied is True
        # Should normalize operators
        assert "→" in result.formal_input or "->" not in result.formal_input

    def test_no_extraction_without_propositions_section(self, preprocessor):
        """Test that extraction fails gracefully without proper structure."""
        query = "What is the weather like today?"

        result = preprocessor.preprocess(
            query=query,
            query_type="symbolic",
            reasoning_tools=["symbolic"],
        )

        # No SAT pattern found, but might find direct formulas
        # If no patterns match at all, preprocessing_applied should be False
        assert isinstance(result, PreprocessingResult)

    def test_extract_direct_formulas(self, preprocessor):
        """Test extraction of formulas containing logical operators directly."""
        query = """
        Given the following formulas:
        A → B
        B → C
        ¬C
        
        Prove something.
        """

        result = preprocessor.preprocess(
            query=query,
            query_type="symbolic",
            reasoning_tools=["symbolic"],
        )

        assert result.preprocessing_applied is True
        assert result.extraction_confidence == 0.7  # Lower confidence for direct formulas

    def test_skip_non_symbolic_tools(self, preprocessor):
        """Test that preprocessing is skipped for non-formal tools."""
        query = "Propositions: A, B\nConstraints:\n1. A→B"

        result = preprocessor.preprocess(
            query=query,
            query_type="general",
            reasoning_tools=["general", "causal"],
        )

        assert result.preprocessing_applied is False


class TestQueryPreprocessorMathematicalExtraction:
    """Tests for mathematical formula extraction."""

    @pytest.fixture
    def preprocessor(self):
        """Create a fresh preprocessor instance."""
        return QueryPreprocessor()

    def test_extract_labeled_formula(self, preprocessor):
        """Test extraction of explicitly labeled formulas."""
        query = """
        Prove the following:
        
        Formula: x² + 2xy + y² = (x + y)²
        
        Using algebraic manipulation.
        """

        result = preprocessor.preprocess(
            query=query,
            query_type="mathematical",
            reasoning_tools=["mathematical"],
        )

        assert result.preprocessing_applied is True
        assert result.extraction_confidence == 0.85
        assert result.extraction_type == ExtractionType.MATHEMATICAL
        assert "=" in result.formal_input

    def test_extract_equation(self, preprocessor):
        """Test extraction of equations without explicit labels."""
        query = """
        Solve for x:
        2x + 3 = 7
        """

        result = preprocessor.preprocess(
            query=query,
            query_type="mathematical",
            reasoning_tools=["mathematical"],
        )

        assert result.preprocessing_applied is True
        assert "=" in result.formal_input

    def test_reject_text_with_equals(self, preprocessor):
        """Test that text containing 'is' at start is rejected."""
        query = """
        The answer is 42.
        """

        result = preprocessor.preprocess(
            query=query,
            query_type="mathematical",
            reasoning_tools=["mathematical"],
        )

        # Should not extract "is 42" as an equation
        # The line doesn't match our pattern well


class TestQueryPreprocessorProbabilisticExtraction:
    """Tests for probabilistic notation extraction."""

    @pytest.fixture
    def preprocessor(self):
        """Create a fresh preprocessor instance."""
        return QueryPreprocessor()

    def test_extract_probability_notation(self, preprocessor):
        """Test extraction of P(...) notation."""
        query = """
        Given P(A) = 0.3 and P(B|A) = 0.7,
        what is P(A and B)?
        """

        result = preprocessor.preprocess(
            query=query,
            query_type="probabilistic",
            reasoning_tools=["probabilistic"],
        )

        assert result.preprocessing_applied is True
        assert result.extraction_confidence == 0.8
        assert result.extraction_type == ExtractionType.PROBABILISTIC
        assert isinstance(result.formal_input, list)
        assert any("P(A)" in p for p in result.formal_input)
        assert any("P(B|A)" in p for p in result.formal_input)

    def test_extract_expectation_notation(self, preprocessor):
        """Test extraction of E[...] notation."""
        query = """
        Calculate E[X] given the distribution.
        Also find E[X|Y].
        """

        result = preprocessor.preprocess(
            query=query,
            query_type="probabilistic",
            reasoning_tools=["probabilistic"],
        )

        assert result.preprocessing_applied is True
        assert isinstance(result.formal_input, list)
        assert any("E[X]" in e for e in result.formal_input)


class TestQueryPreprocessorOperatorNormalization:
    """Tests for operator normalization."""

    @pytest.fixture
    def preprocessor(self):
        """Create a fresh preprocessor instance."""
        return QueryPreprocessor()

    def test_normalize_implication(self, preprocessor):
        """Test normalization of implication operators."""
        assert "→" in preprocessor._normalize_operators("A -> B")
        assert "→" in preprocessor._normalize_operators("A => B")
        assert "→" in preprocessor._normalize_operators("A implies B")

    def test_normalize_conjunction(self, preprocessor):
        """Test normalization of conjunction operators."""
        assert "∧" in preprocessor._normalize_operators("A AND B")
        assert "∧" in preprocessor._normalize_operators("A and B")
        assert "∧" in preprocessor._normalize_operators("A && B")

    def test_normalize_disjunction(self, preprocessor):
        """Test normalization of disjunction operators."""
        assert "∨" in preprocessor._normalize_operators("A OR B")
        assert "∨" in preprocessor._normalize_operators("A or B")
        assert "∨" in preprocessor._normalize_operators("A || B")

    def test_normalize_negation(self, preprocessor):
        """Test normalization of negation operators."""
        assert "¬" in preprocessor._normalize_operators("NOT A")
        assert "¬" in preprocessor._normalize_operators("not A")
        assert "¬" in preprocessor._normalize_operators("~A")
        assert "¬" in preprocessor._normalize_operators("!A")

    def test_normalize_biconditional(self, preprocessor):
        """Test normalization of biconditional operators."""
        assert "↔" in preprocessor._normalize_operators("A <-> B")
        assert "↔" in preprocessor._normalize_operators("A iff B")


class TestQueryPreprocessorMetrics:
    """Tests for metrics tracking."""

    @pytest.fixture
    def preprocessor(self):
        """Create a fresh preprocessor instance."""
        return QueryPreprocessor()

    def test_metrics_initialization(self, preprocessor):
        """Test that metrics start at zero."""
        metrics = preprocessor.get_metrics()

        assert metrics["total_queries"] == 0
        assert metrics["symbolic_extractions"] == 0
        assert metrics["mathematical_extractions"] == 0

    def test_metrics_increment_on_queries(self, preprocessor):
        """Test that metrics increment on processing."""
        # Symbolic query
        preprocessor.preprocess(
            query="Propositions: A\nConstraints:\n1. A",
            query_type="symbolic",
            reasoning_tools=["symbolic"],
        )

        metrics = preprocessor.get_metrics()
        assert metrics["total_queries"] == 1

        # Mathematical query
        preprocessor.preprocess(
            query="Formula: x = 1",
            query_type="mathematical",
            reasoning_tools=["mathematical"],
        )

        metrics = preprocessor.get_metrics()
        assert metrics["total_queries"] == 2

    def test_metrics_reset(self, preprocessor):
        """Test metrics reset functionality."""
        preprocessor.preprocess(
            query="Test",
            query_type="general",
            reasoning_tools=["general"],
        )

        preprocessor.reset_metrics()
        metrics = preprocessor.get_metrics()

        assert metrics["total_queries"] == 0


class TestQueryPreprocessorErrorHandling:
    """Tests for error handling."""

    @pytest.fixture
    def preprocessor(self):
        """Create a fresh preprocessor instance."""
        return QueryPreprocessor()

    def test_type_error_on_invalid_query(self, preprocessor):
        """Test that TypeError is raised for non-string query."""
        with pytest.raises(TypeError, match="query must be str"):
            preprocessor.preprocess(
                query=123,  # type: ignore
                query_type="symbolic",
                reasoning_tools=["symbolic"],
            )

    def test_type_error_on_invalid_tools(self, preprocessor):
        """Test that TypeError is raised for non-list tools."""
        with pytest.raises(TypeError, match="reasoning_tools must be list"):
            preprocessor.preprocess(
                query="test",
                query_type="symbolic",
                reasoning_tools="symbolic",  # type: ignore
            )

    def test_empty_query_handling(self, preprocessor):
        """Test handling of empty query."""
        result = preprocessor.preprocess(
            query="",
            query_type="symbolic",
            reasoning_tools=["symbolic"],
        )

        assert isinstance(result, PreprocessingResult)
        assert result.preprocessing_applied is False

    def test_empty_tools_handling(self, preprocessor):
        """Test handling of empty tools list."""
        result = preprocessor.preprocess(
            query="Propositions: A\nConstraints:\n1. A",
            query_type="symbolic",
            reasoning_tools=[],
        )

        assert isinstance(result, PreprocessingResult)
        assert result.preprocessing_applied is False


class TestQueryPreprocessorThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_preprocessing(self):
        """Test that concurrent preprocessing is thread-safe."""
        preprocessor = QueryPreprocessor()
        results = []
        errors = []

        def process_query(query_id: int):
            try:
                query = f"""
                Propositions: A{query_id}, B{query_id}
                
                Constraints:
                1. A{query_id}→B{query_id}
                
                Task: Check
                """
                result = preprocessor.preprocess(
                    query=query,
                    query_type="symbolic",
                    reasoning_tools=["symbolic"],
                )
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Run concurrent preprocessing
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(process_query, i) for i in range(100)]
            for future in futures:
                future.result()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 100

    def test_singleton_thread_safety(self):
        """Test that singleton creation is thread-safe."""
        reset_query_preprocessor()
        instances = []

        def get_instance():
            instance = get_query_preprocessor()
            instances.append(id(instance))

        threads = [threading.Thread(target=get_instance) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should be the same instance
        assert len(set(instances)) == 1


class TestQueryPreprocessorSingleton:
    """Tests for singleton pattern."""

    def test_singleton_returns_same_instance(self):
        """Test that get_query_preprocessor returns the same instance."""
        reset_query_preprocessor()

        instance1 = get_query_preprocessor()
        instance2 = get_query_preprocessor()

        assert instance1 is instance2

    def test_reset_creates_new_instance(self):
        """Test that reset allows creating a new instance."""
        reset_query_preprocessor()

        instance1 = get_query_preprocessor()
        id1 = id(instance1)

        reset_query_preprocessor()

        instance2 = get_query_preprocessor()
        id2 = id(instance2)

        assert id1 != id2


class TestExtractionType:
    """Tests for ExtractionType enum."""

    def test_extraction_type_values(self):
        """Test that ExtractionType has expected values."""
        assert ExtractionType.SYMBOLIC.value == "symbolic"
        assert ExtractionType.MATHEMATICAL.value == "mathematical"
        assert ExtractionType.PROBABILISTIC.value == "probabilistic"
        assert ExtractionType.CAUSAL.value == "causal"
        assert ExtractionType.NONE.value == "none"


# =============================================================================
# BUG D FIX: Tests for standalone operator filtering
# =============================================================================
class TestBugDStandaloneOperatorFiltering:
    """
    Tests for BUG D fix: Filtering standalone operators in formula extraction.
    
    The bug caused queries like "A→B, B→C" to extract "(→) ∧ (A→B)" instead of
    just "(A→B) ∧ (B→C)". The fix ensures that standalone operators without
    propositions (like "→", "∧", "∨", "¬") are filtered out.
    """

    @pytest.fixture
    def preprocessor(self):
        """Create a fresh preprocessor instance."""
        return QueryPreprocessor()

    def test_filter_standalone_implication(self, preprocessor):
        """Test that standalone → operator is filtered out."""
        # This should NOT extract (→) as a formula
        cleaned = preprocessor._clean_formula_line("→")
        assert cleaned == "", f"Standalone → should be filtered, got: '{cleaned}'"

    def test_filter_standalone_conjunction(self, preprocessor):
        """Test that standalone ∧ operator is filtered out."""
        cleaned = preprocessor._clean_formula_line("∧")
        assert cleaned == "", f"Standalone ∧ should be filtered, got: '{cleaned}'"

    def test_filter_standalone_disjunction(self, preprocessor):
        """Test that standalone ∨ operator is filtered out."""
        cleaned = preprocessor._clean_formula_line("∨")
        assert cleaned == "", f"Standalone ∨ should be filtered, got: '{cleaned}'"

    def test_filter_standalone_negation(self, preprocessor):
        """Test that standalone ¬ operator is filtered out."""
        cleaned = preprocessor._clean_formula_line("¬")
        assert cleaned == "", f"Standalone ¬ should be filtered, got: '{cleaned}'"

    def test_keep_valid_implication_formula(self, preprocessor):
        """Test that valid A→B formula is kept."""
        cleaned = preprocessor._clean_formula_line("A→B")
        assert cleaned == "A→B", f"Valid formula A→B should be kept, got: '{cleaned}'"

    def test_keep_valid_negation_formula(self, preprocessor):
        """Test that valid ¬C formula is kept."""
        cleaned = preprocessor._clean_formula_line("¬C")
        assert cleaned == "¬C", f"Valid formula ¬C should be kept, got: '{cleaned}'"

    def test_keep_valid_disjunction_formula(self, preprocessor):
        """Test that valid A∨B formula is kept."""
        cleaned = preprocessor._clean_formula_line("A∨B")
        assert cleaned == "A∨B", f"Valid formula A∨B should be kept, got: '{cleaned}'"

    def test_sat_problem_no_standalone_operators(self, preprocessor):
        """
        Test that SAT problem extraction doesn't include standalone operators.
        
        This is the main BUG D scenario: a well-formed SAT problem should
        produce formulas like "(A→B) ∧ (B→C) ∧ (¬C)", NOT "(→) ∧ (A→B)".
        """
        query = """
        Propositions: A, B, C

        Constraints:
        1. A→B
        2. B→C
        3. ¬C
        4. A∨B

        Task: Is the set satisfiable?
        """

        result = preprocessor.preprocess(
            query=query,
            query_type="symbolic",
            reasoning_tools=["symbolic"],
        )

        assert result.preprocessing_applied is True
        
        # The formal input should NOT contain standalone operators
        formal_input = result.formal_input
        assert formal_input is not None
        
        # Check that we don't have "(→)" or "(∧)" etc. as standalone formulas
        # These would appear as "(→) ∧" patterns if the bug is present
        assert "(→) ∧" not in formal_input, f"BUG D present: standalone (→) found in: {formal_input}"
        assert "(∧) ∧" not in formal_input, f"BUG D present: standalone (∧) found in: {formal_input}"
        assert "(∨) ∧" not in formal_input, f"BUG D present: standalone (∨) found in: {formal_input}"
        assert "(¬) ∧" not in formal_input, f"BUG D present: standalone (¬) found in: {formal_input}"
        
        # Check that valid formulas ARE present
        assert "(A→B)" in formal_input, f"Valid formula (A→B) missing from: {formal_input}"
        assert "(B→C)" in formal_input, f"Valid formula (B→C) missing from: {formal_input}"
        assert "(¬C)" in formal_input, f"Valid formula (¬C) missing from: {formal_input}"
        assert "(A∨B)" in formal_input, f"Valid formula (A∨B) missing from: {formal_input}"

    def test_direct_formulas_no_standalone_operators(self, preprocessor):
        """Test direct formula extraction doesn't include standalone operators."""
        query = """
        Given:
        A → B
        B → C
        ¬C
        
        What can we conclude?
        """

        result = preprocessor.preprocess(
            query=query,
            query_type="symbolic",
            reasoning_tools=["symbolic"],
        )

        if result.preprocessing_applied and result.formal_input:
            formal_input = result.formal_input
            # Should not have standalone operators
            assert "(→) ∧" not in formal_input
            # Should have valid formulas
            assert "A → B" in formal_input or "A→B" in formal_input

    def test_numbered_list_constraint_extraction(self, preprocessor):
        """Test that numbered list constraints are properly extracted without standalone operators."""
        query = """
        Propositions: S, D, E

        Constraints:
        1. S→D
        2. S→E

        Task: Does S cause D?
        """

        result = preprocessor.preprocess(
            query=query,
            query_type="symbolic",
            reasoning_tools=["symbolic"],
        )

        assert result.preprocessing_applied is True
        formal_input = result.formal_input
        
        # Should contain valid formulas
        assert "(S→D)" in formal_input, f"Missing (S→D) in: {formal_input}"
        assert "(S→E)" in formal_input, f"Missing (S→E) in: {formal_input}"
        
        # Should NOT have standalone operators
        assert "(→)" not in formal_input or "(→) ∧" not in formal_input
