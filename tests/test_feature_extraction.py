"""
Comprehensive test suite for feature_extraction.py
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from feature_extraction import (
    ExtractionResult,
    FeatureExtractor,
    FeatureTier,
    MultimodalFeatureExtractor,
    MultiTierFeatureExtractor,
    ProblemStructure,
    SemanticFeatureExtractor,
    StructuralFeatureExtractor,
    SyntacticFeatureExtractor,
)


@pytest.fixture
def simple_text_problem():
    """Create simple text problem."""
    return "If x = 5 and y = 10, what is x + y?"


@pytest.fixture
def complex_text_problem():
    """Create complex text problem."""
    return """
    Consider a probabilistic system where events A and B are independent.
    If P(A) = 0.3 and P(B) = 0.5, what is the probability that either A or B occurs?
    Use the formula for the union of independent events.
    """


@pytest.fixture
def dict_problem():
    """Create dictionary problem."""
    return {
        "text": "Find the shortest path in a graph",
        "graph": {
            "nodes": ["A", "B", "C", "D"],
            "edges": [("A", "B"), ("B", "C"), ("C", "D"), ("A", "D")],
        },
    }


@pytest.fixture
def syntactic_extractor():
    """Create SyntacticFeatureExtractor."""
    return SyntacticFeatureExtractor()


@pytest.fixture
def structural_extractor():
    """Create StructuralFeatureExtractor."""
    return StructuralFeatureExtractor()


@pytest.fixture
def semantic_extractor():
    """Create SemanticFeatureExtractor."""
    return SemanticFeatureExtractor()


@pytest.fixture
def multimodal_extractor():
    """Create MultimodalFeatureExtractor."""
    return MultimodalFeatureExtractor()


@pytest.fixture
def multitier_extractor():
    """Create MultiTierFeatureExtractor."""
    return MultiTierFeatureExtractor()


class TestEnums:
    """Test enum classes."""

    def test_feature_tier_enum(self):
        """Test FeatureTier enum."""
        assert FeatureTier.TIER1_SYNTACTIC.value == 1
        assert FeatureTier.TIER4_MULTIMODAL.value == 4


class TestDataClasses:
    """Test dataclass structures."""

    def test_extraction_result_creation(self):
        """Test creating ExtractionResult."""
        features = np.array([1.0, 2.0, 3.0])

        result = ExtractionResult(
            features=features,
            tier=FeatureTier.TIER1_SYNTACTIC,
            extraction_time_ms=10.5,
            feature_names=["f1", "f2", "f3"],
        )

        assert result.tier == FeatureTier.TIER1_SYNTACTIC
        assert len(result.features) == 3

    def test_problem_structure_creation(self):
        """Test creating ProblemStructure."""
        structure = ProblemStructure(text="test problem", tokens=["test", "problem"])

        assert structure.text == "test problem"
        assert len(structure.tokens) == 2


class TestSyntacticFeatureExtractor:
    """Test SyntacticFeatureExtractor."""

    def test_initialization(self, syntactic_extractor):
        """Test extractor initialization."""
        assert syntactic_extractor is not None
        assert len(syntactic_extractor.feature_names) > 0

    def test_extract_simple_text(self, syntactic_extractor, simple_text_problem):
        """Test extraction from simple text."""
        features = syntactic_extractor.extract(simple_text_problem)

        assert isinstance(features, np.ndarray)
        assert len(features) == len(syntactic_extractor.feature_names)

    def test_extract_complex_text(self, syntactic_extractor, complex_text_problem):
        """Test extraction from complex text."""
        features = syntactic_extractor.extract(complex_text_problem)

        assert isinstance(features, np.ndarray)
        assert len(features) > 0

    def test_extract_dict(self, syntactic_extractor, dict_problem):
        """Test extraction from dictionary."""
        features = syntactic_extractor.extract(dict_problem)

        assert isinstance(features, np.ndarray)

    def test_feature_values_valid(self, syntactic_extractor, simple_text_problem):
        """Test that feature values are valid."""
        features = syntactic_extractor.extract(simple_text_problem)

        # All features should be finite
        assert np.all(np.isfinite(features))

        # Most features should be non-negative
        assert np.sum(features >= 0) > len(features) * 0.8

    def test_get_feature_names(self, syntactic_extractor):
        """Test getting feature names."""
        names = syntactic_extractor.get_feature_names()

        assert isinstance(names, list)
        assert len(names) > 0

    def test_equation_detection(self, syntactic_extractor):
        """Test equation detection feature."""
        problem_with_equation = "Solve x + 5 = 10"
        features = syntactic_extractor.extract(problem_with_equation)

        # has_equation feature should be positive
        equation_idx = syntactic_extractor.feature_names.index("has_equation")
        assert features[equation_idx] > 0

    def test_probability_detection(self, syntactic_extractor):
        """Test probability detection feature."""
        problem_with_prob = "What is the probability of rolling a 6?"
        features = syntactic_extractor.extract(problem_with_prob)

        # has_probability feature should be positive
        prob_idx = syntactic_extractor.feature_names.index("has_probability")
        assert features[prob_idx] > 0


class TestStructuralFeatureExtractor:
    """Test StructuralFeatureExtractor."""

    def test_initialization(self, structural_extractor):
        """Test extractor initialization."""
        assert structural_extractor is not None

    def test_extract_text(self, structural_extractor, complex_text_problem):
        """Test extraction from text."""
        features = structural_extractor.extract(complex_text_problem)

        assert isinstance(features, np.ndarray)
        assert len(features) > 0

    def test_extract_dict_with_graph(self, structural_extractor, dict_problem):
        """Test extraction from dictionary with graph."""
        features = structural_extractor.extract(dict_problem)

        assert isinstance(features, np.ndarray)

    def test_parse_structure(self, structural_extractor, dict_problem):
        """Test parsing problem structure."""
        structure = structural_extractor._parse_structure(dict_problem)

        assert structure.graph is not None
        assert structure.text is not None

    def test_graph_feature_extraction(self, structural_extractor):
        """Test graph feature extraction."""
        import networkx as nx

        G = nx.Graph()
        G.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])

        features = structural_extractor._extract_graph_features(G)

        assert isinstance(features, list)
        assert len(features) > 0

    def test_structural_complexity(self, structural_extractor):
        """Test structural complexity features."""
        structure = ProblemStructure(
            text="This is a test. This is another sentence.",
            tokens=["This", "is", "a", "test"],
        )

        features = structural_extractor._extract_structural_complexity(structure)

        assert isinstance(features, list)
        assert all(isinstance(f, float) for f in features)


class TestSemanticFeatureExtractor:
    """Test SemanticFeatureExtractor."""

    def test_initialization(self, semantic_extractor):
        """Test extractor initialization."""
        assert semantic_extractor is not None

    def test_extract_text(self, semantic_extractor, complex_text_problem):
        """Test extraction from text."""
        features = semantic_extractor.extract(complex_text_problem)

        assert isinstance(features, np.ndarray)
        assert len(features) > 0

    def test_semantic_similarity(self, semantic_extractor):
        """Test semantic similarity extraction."""
        structure = ProblemStructure(
            text="If premise is true, then conclusion follows",
            tokens=["premise", "true", "conclusion", "follows"],
        )

        features = semantic_extractor._extract_semantic_similarity(structure)

        assert isinstance(features, list)
        assert all(0 <= f <= 1 for f in features)

    def test_reasoning_patterns(self, semantic_extractor):
        """Test reasoning pattern extraction."""
        structure = ProblemStructure(text="If A then B, therefore C")

        features = semantic_extractor._extract_reasoning_patterns(structure)

        assert isinstance(features, list)

    def test_conceptual_features(self, semantic_extractor):
        """Test conceptual feature extraction."""
        structure = ProblemStructure(
            text="All men are mortal. Some men are wise.",
            tokens=["all", "men", "are", "mortal", "some", "wise"],
        )

        features = semantic_extractor._extract_conceptual_features(structure)

        assert isinstance(features, list)

    def test_inference_complexity(self, semantic_extractor):
        """Test inference complexity extraction."""
        structure = ProblemStructure(text="Therefore, if not A and not B, then C")

        features = semantic_extractor._extract_inference_complexity(structure)

        assert isinstance(features, list)
        assert len(features) > 0


class TestMultimodalFeatureExtractor:
    """Test MultimodalFeatureExtractor."""

    def test_initialization(self, multimodal_extractor):
        """Test extractor initialization."""
        assert multimodal_extractor is not None

    def test_detect_modalities_text(self, multimodal_extractor):
        """Test modality detection for text."""
        problem = "This is a text problem"

        modalities = multimodal_extractor._detect_modalities(problem)

        assert "text" in modalities

    def test_detect_modalities_graph(self, multimodal_extractor, dict_problem):
        """Test modality detection for graph."""
        modalities = multimodal_extractor._detect_modalities(dict_problem)

        assert "text" in modalities or "graph" in modalities

    def test_extract_text(self, multimodal_extractor, simple_text_problem):
        """Test extraction from text."""
        features = multimodal_extractor.extract(simple_text_problem)

        assert isinstance(features, np.ndarray)
        assert len(features) > 0

    def test_extract_dict(self, multimodal_extractor, dict_problem):
        """Test extraction from dictionary."""
        features = multimodal_extractor.extract(dict_problem)

        assert isinstance(features, np.ndarray)

    def test_graph_features(self, multimodal_extractor, dict_problem):
        """Test graph feature extraction."""
        features = multimodal_extractor._extract_graph_features(dict_problem)

        assert isinstance(features, list)
        assert len(features) == 50  # Fixed size

    def test_table_features(self, multimodal_extractor):
        """Test table feature extraction."""
        problem = {"table": [[1, 2, 3], [4, 5, 6]]}

        features = multimodal_extractor._extract_table_features(problem)

        assert isinstance(features, list)
        assert len(features) == 50

    def test_formula_features(self, multimodal_extractor):
        """Test formula feature extraction."""
        problem = {"formula": "x^2 + 2x + 1 = 0"}

        features = multimodal_extractor._extract_formula_features(problem)

        assert isinstance(features, list)
        assert len(features) == 50

    def test_cross_modal_features(self, multimodal_extractor):
        """Test cross-modal feature extraction."""
        modalities = ["text", "graph"]

        features = multimodal_extractor._extract_cross_modal_features({}, modalities)

        assert isinstance(features, list)
        assert len(features) > 0


class TestMultiTierFeatureExtractor:
    """Test MultiTierFeatureExtractor."""

    def test_initialization(self, multitier_extractor):
        """Test extractor initialization."""
        assert multitier_extractor is not None
        assert multitier_extractor.tier1_extractor is not None
        assert multitier_extractor.tier2_extractor is not None

    def test_extract_tier1(self, multitier_extractor, simple_text_problem):
        """Test Tier 1 extraction."""
        features = multitier_extractor.extract_tier1(simple_text_problem)

        assert isinstance(features, np.ndarray)
        assert len(features) > 0

    def test_extract_tier2(self, multitier_extractor, simple_text_problem):
        """Test Tier 2 extraction."""
        features = multitier_extractor.extract_tier2(simple_text_problem)

        assert isinstance(features, np.ndarray)
        # Should include Tier 1 features
        assert len(features) > len(multitier_extractor.tier1_extractor.feature_names)

    def test_extract_tier3(self, multitier_extractor, simple_text_problem):
        """Test Tier 3 extraction."""
        features = multitier_extractor.extract_tier3(simple_text_problem)

        assert isinstance(features, np.ndarray)
        # Should be larger than Tier 2
        tier2_features = multitier_extractor.extract_tier2(simple_text_problem)
        assert len(features) > len(tier2_features)

    def test_extract_tier4(self, multitier_extractor, simple_text_problem):
        """Test Tier 4 extraction."""
        features = multitier_extractor.extract_tier4(simple_text_problem)

        assert isinstance(features, np.ndarray)
        # Should be largest
        tier3_features = multitier_extractor.extract_tier3(simple_text_problem)
        assert len(features) > len(tier3_features)

    def test_extract_adaptive_fast(self, multitier_extractor, simple_text_problem):
        """Test adaptive extraction with fast budget."""
        features = multitier_extractor.extract_adaptive(
            simple_text_problem, time_budget_ms=5
        )

        assert isinstance(features, np.ndarray)
        # Should use lowest tier

    def test_extract_adaptive_slow(self, multitier_extractor, simple_text_problem):
        """Test adaptive extraction with slow budget."""
        features = multitier_extractor.extract_adaptive(
            simple_text_problem, time_budget_ms=1000
        )

        assert isinstance(features, np.ndarray)
        # Should use higher tier

    def test_feature_caching(self, multitier_extractor, simple_text_problem):
        """Test feature caching."""
        # First extraction
        features1 = multitier_extractor.extract_adaptive(
            simple_text_problem, time_budget_ms=100
        )

        # Second extraction (should use cache)
        features2 = multitier_extractor.extract_adaptive(
            simple_text_problem, time_budget_ms=100
        )

        np.testing.assert_array_equal(features1, features2)

    def test_get_statistics(self, multitier_extractor, simple_text_problem):
        """Test getting statistics."""
        # Perform some extractions
        multitier_extractor.extract_tier1(simple_text_problem)
        multitier_extractor.extract_tier2(simple_text_problem)

        stats = multitier_extractor.get_statistics()

        assert isinstance(stats, dict)
        assert FeatureTier.TIER1_SYNTACTIC.value in stats


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_string(self, syntactic_extractor):
        """Test with empty string."""
        features = syntactic_extractor.extract("")

        assert isinstance(features, np.ndarray)

    def test_very_long_text(self, syntactic_extractor):
        """Test with very long text."""
        long_text = "word " * 10000

        features = syntactic_extractor.extract(long_text)

        assert isinstance(features, np.ndarray)

    def test_special_characters(self, syntactic_extractor):
        """Test with special characters."""
        text = "∀x∃y: x → y ∧ ¬(x ∨ y)"

        features = syntactic_extractor.extract(text)

        assert isinstance(features, np.ndarray)

    def test_unicode(self, syntactic_extractor):
        """Test with unicode characters."""
        text = "问题：如果 x = 5，那么 x + 2 = ?"

        features = syntactic_extractor.extract(text)

        assert isinstance(features, np.ndarray)

    def test_none_input(self, syntactic_extractor):
        """Test with None input."""
        features = syntactic_extractor.extract(None)

        assert isinstance(features, np.ndarray)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
