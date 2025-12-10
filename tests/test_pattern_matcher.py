"""
Comprehensive test suite for pattern_matcher.py
"""

import asyncio
from unittest.mock import MagicMock, Mock, patch

import pytest

from pattern_matcher import (DEFAULT_MATCH_TIMEOUT, MAX_GRAPH_EDGES,
                             MAX_GRAPH_NODES, MAX_MATCHES_TO_PROCESS,
                             MAX_PATTERN_NODES, GraphSizeLimitError,
                             GraphValidationError, GraphValidationResult,
                             MatchingStats, MatchingTimeoutError,
                             PatternMatcher, PatternMatcherError)


@pytest.fixture
def matcher():
    """Create PatternMatcher instance."""
    return PatternMatcher(enable_validation=True, strict_mode=True)


@pytest.fixture
def simple_graph():
    """Create simple valid graph."""
    return {
        "nodes": [
            {"id": "n1", "type": "CONST", "params": {"value": 5}},
            {"id": "n2", "type": "CONST", "params": {"value": 20}},
            {"id": "n3", "type": "ADD", "params": {}}
        ],
        "edges": [
            {"from": "n1", "to": "n3"},
            {"from": "n2", "to": "n3"}
        ]
    }


@pytest.fixture
def simple_pattern():
    """Create simple valid pattern."""
    return {
        "nodes": [
            {"id": "?p1", "type": "CONST", "params": {"value": "> 10"}}
        ],
        "edges": []
    }


class TestPatternMatcherInitialization:
    """Test PatternMatcher initialization."""

    def test_initialization_basic(self):
        """Test basic initialization."""
        matcher = PatternMatcher()

        assert matcher.match_timeout == DEFAULT_MATCH_TIMEOUT
        assert matcher.max_matches == MAX_MATCHES_TO_PROCESS
        assert matcher.nso_aligner is not None

    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        matcher = PatternMatcher(
            match_timeout=60.0,
            max_matches=500,
            enable_validation=False,
            strict_mode=False
        )

        assert matcher.match_timeout == 60.0
        assert matcher.max_matches == 500
        assert matcher.enable_validation is False
        assert matcher.strict_mode is False

    @patch('pattern_matcher.NETWORKX_AVAILABLE', False)
    def test_initialization_no_networkx(self):
        """Test initialization without NetworkX."""
        with pytest.raises(ImportError):
            PatternMatcher()


class TestGraphValidation:
    """Test graph validation."""

    def test_validate_valid_graph(self, matcher, simple_graph):
        """Test validating valid graph."""
        result = matcher._validate_graph_structure(simple_graph, "graph")

        assert result.is_valid is True
        assert result.error_message is None

    def test_validate_not_dict(self, matcher):
        """Test validating non-dict."""
        result = matcher._validate_graph_structure("not a dict", "graph")

        assert result.is_valid is False
        assert "must be a dictionary" in result.error_message

    def test_validate_missing_nodes(self, matcher):
        """Test validating graph without nodes."""
        result = matcher._validate_graph_structure({"edges": []}, "graph")

        assert result.is_valid is False
        assert "missing 'nodes'" in result.error_message.lower()

    def test_validate_nodes_not_list(self, matcher):
        """Test nodes not a list."""
        result = matcher._validate_graph_structure(
            {"nodes": "not a list", "edges": []}, "graph"
        )

        assert result.is_valid is False

    def test_validate_too_many_nodes(self, matcher):
        """Test graph with too many nodes."""
        graph = {
            "nodes": [{"id": f"n{i}", "type": "Node"} for i in range(MAX_GRAPH_NODES + 1)],
            "edges": []
        }

        result = matcher._validate_graph_structure(graph, "graph")

        assert result.is_valid is False
        assert "exceeds limit" in result.error_message

    def test_validate_duplicate_node_ids(self, matcher):
        """Test duplicate node IDs."""
        graph = {
            "nodes": [
                {"id": "n1", "type": "Node"},
                {"id": "n1", "type": "Node"}
            ],
            "edges": []
        }

        result = matcher._validate_graph_structure(graph, "graph")

        assert result.is_valid is False
        assert "duplicate" in result.error_message.lower()

    def test_validate_edge_references_nonexistent(self, matcher):
        """Test edge referencing non-existent node."""
        graph = {
            "nodes": [{"id": "n1", "type": "Node"}],
            "edges": [{"from": "n1", "to": "nonexistent"}]
        }

        result = matcher._validate_graph_structure(graph, "graph")

        assert result.is_valid is False
        assert "non-existent" in result.error_message


class TestTypeCasting:
    """Test type casting."""

    def test_safe_type_cast_same_type(self, matcher):
        """Test casting to same type."""
        result = matcher._safe_type_cast(5, int, "test")

        assert result == 5
        assert isinstance(result, int)

    def test_safe_type_cast_str_to_int(self, matcher):
        """Test casting string to int."""
        result = matcher._safe_type_cast("42", int, "test")

        assert result == 42

    def test_safe_type_cast_str_to_float(self, matcher):
        """Test casting string to float."""
        result = matcher._safe_type_cast("3.14", float, "test")

        assert abs(result - 3.14) < 0.001

    def test_safe_type_cast_to_str(self, matcher):
        """Test casting to string."""
        result = matcher._safe_type_cast(42, str, "test")

        assert result == "42"

    def test_safe_type_cast_to_bool(self, matcher):
        """Test casting to bool."""
        assert matcher._safe_type_cast("true", bool, "test") is True
        assert matcher._safe_type_cast("false", bool, "test") is False
        assert matcher._safe_type_cast("1", bool, "test") is True
        assert matcher._safe_type_cast("0", bool, "test") is False

    def test_safe_type_cast_invalid(self, matcher):
        """Test invalid type cast."""
        with pytest.raises(ValueError):
            matcher._safe_type_cast("not_a_number", int, "test")


class TestNodeSemanticMatch:
    """Test node semantic matching."""

    def test_node_match_exact_type(self, matcher):
        """Test matching exact node type."""
        g_node = {"type": "CONST", "params": {"value": 10}}
        p_node = {"type": "CONST", "params": {"value": 10}}

        result = matcher._node_semantic_match(g_node, p_node)

        assert result is True

    def test_node_match_wildcard_type(self, matcher):
        """Test matching wildcard type."""
        g_node = {"type": "CONST", "params": {}}
        p_node = {"type": "*", "params": {}}

        result = matcher._node_semantic_match(g_node, p_node)

        assert result is True

    def test_node_match_type_mismatch(self, matcher):
        """Test type mismatch."""
        g_node = {"type": "CONST", "params": {}}
        p_node = {"type": "ADD", "params": {}}

        result = matcher._node_semantic_match(g_node, p_node)

        assert result is False

    def test_node_match_dsl_greater_than(self, matcher):
        """Test DSL > operator."""
        g_node = {"type": "CONST", "params": {"value": 15}}
        p_node = {"type": "CONST", "params": {"value": "> 10"}}

        result = matcher._node_semantic_match(g_node, p_node)

        assert result is True

    def test_node_match_dsl_less_than(self, matcher):
        """Test DSL < operator."""
        g_node = {"type": "CONST", "params": {"value": 5}}
        p_node = {"type": "CONST", "params": {"value": "< 10"}}

        result = matcher._node_semantic_match(g_node, p_node)

        assert result is True

    def test_node_match_dsl_equals(self, matcher):
        """Test DSL == operator."""
        g_node = {"type": "CONST", "params": {"value": 10}}
        p_node = {"type": "CONST", "params": {"value": "== 10"}}

        result = matcher._node_semantic_match(g_node, p_node)

        assert result is True

    def test_node_match_dict_operator(self, matcher):
        """Test dictionary-based operator."""
        g_node = {"type": "CONST", "params": {"value": 15}}
        p_node = {"type": "CONST", "params": {"value": {">": 10}}}

        result = matcher._node_semantic_match(g_node, p_node)

        assert result is True


class TestFindMatches:
    """Test finding matches."""

    @pytest.mark.asyncio
    async def test_find_matches_basic(self, matcher, simple_graph, simple_pattern):
        """Test basic pattern matching."""
        matches = []

        async for match in matcher.find_matches(simple_graph, simple_pattern):
            matches.append(match)

        # Should find node n2 (value=20 > 10)
        assert len(matches) >= 1

    @pytest.mark.asyncio
    async def test_find_matches_no_matches(self, matcher):
        """Test pattern with no matches."""
        graph = {
            "nodes": [{"id": "n1", "type": "CONST", "params": {"value": 5}}],
            "edges": []
        }
        pattern = {
            "nodes": [{"id": "?p1", "type": "CONST", "params": {"value": "> 10"}}],
            "edges": []
        }

        matches = []
        async for match in matcher.find_matches(graph, pattern):
            matches.append(match)

        assert len(matches) == 0

    @pytest.mark.asyncio
    async def test_find_matches_validation_error(self, matcher):
        """Test matching with invalid graph."""
        invalid_graph = {"nodes": "not a list"}
        pattern = {"nodes": [], "edges": []}

        with pytest.raises(GraphValidationError):
            async for match in matcher.find_matches(invalid_graph, pattern):
                pass

    @pytest.mark.asyncio
    async def test_find_matches_with_validation_disabled(self):
        """Test matching with validation disabled."""
        matcher = PatternMatcher(enable_validation=False)

        # Should not raise even with invalid structure
        invalid_graph = {"nodes": [], "edges": []}
        pattern = {"nodes": [], "edges": []}

        matches = []
        async for match in matcher.find_matches(invalid_graph, pattern):
            matches.append(match)


class TestRewriteGraph:
    """Test graph rewriting."""

    @pytest.mark.asyncio
    async def test_rewrite_graph_approved(self, matcher):
        """Test rewriting with approved mutator."""
        graph = {
            "nodes": [{"id": "n1", "type": "CONST", "params": {"value": 15}}],
            "edges": []
        }
        pattern = {
            "nodes": [{"id": "?p1", "type": "CONST", "params": {"value": "> 10"}}],
            "edges": []
        }

        def safe_mutator(g, match):
            import copy
            new_g = copy.deepcopy(g)
            for node in new_g['nodes']:
                if node['id'] == match['?p1']:
                    node['params']['value'] *= 2
            return new_g

        # Mock NSO Aligner to approve
        with patch.object(matcher.nso_aligner, 'multi_model_audit', return_value='safe'):
            result = await matcher.rewrite_graph(graph, pattern, safe_mutator)

        # Should have been rewritten
        assert result['nodes'][0]['params']['value'] == 30

    @pytest.mark.asyncio
    async def test_rewrite_graph_rejected(self, matcher):
        """Test rewriting with rejected mutator."""
        graph = {
            "nodes": [{"id": "n1", "type": "CONST", "params": {"value": 15}}],
            "edges": []
        }
        pattern = {
            "nodes": [{"id": "?p1", "type": "CONST", "params": {"value": "> 10"}}],
            "edges": []
        }

        def risky_mutator(g, match):
            import copy
            return copy.deepcopy(g)

        # Mock NSO Aligner to reject
        with patch.object(matcher.nso_aligner, 'multi_model_audit', return_value='risky'):
            result = await matcher.rewrite_graph(graph, pattern, risky_mutator)

        # Should be unchanged
        assert result == graph

    @pytest.mark.asyncio
    async def test_rewrite_graph_mutator_error(self, matcher):
        """Test rewriting with failing mutator."""
        graph = {
            "nodes": [{"id": "n1", "type": "CONST", "params": {"value": 15}}],
            "edges": []
        }
        pattern = {
            "nodes": [{"id": "?p1", "type": "CONST", "params": {}}],
            "edges": []
        }

        def failing_mutator(g, match):
            raise Exception("Mutator failed")

        # Should return original graph
        result = await matcher.rewrite_graph(graph, pattern, failing_mutator)

        assert result == graph


class TestStatistics:
    """Test statistics tracking."""

    @pytest.mark.asyncio
    async def test_get_statistics(self, matcher, simple_graph, simple_pattern):
        """Test getting statistics."""
        # Perform some operations
        async for match in matcher.find_matches(simple_graph, simple_pattern):
            break

        stats = matcher.get_statistics()

        assert "graphs_processed" in stats
        assert "patterns_matched" in stats
        assert "matches_found" in stats
        assert "approval_rate" in stats

    def test_reset_statistics(self, matcher):
        """Test resetting statistics."""
        # Record some stats
        matcher.stats.graphs_processed = 10
        matcher.stats.errors = 5

        matcher.reset_statistics()

        assert matcher.stats.graphs_processed == 0
        assert matcher.stats.errors == 0


class TestConstants:
    """Test module constants."""

    def test_constants_exist(self):
        """Test that all constants are defined."""
        assert MAX_GRAPH_NODES > 0
        assert MAX_GRAPH_EDGES > 0
        assert MAX_PATTERN_NODES > 0
        assert DEFAULT_MATCH_TIMEOUT > 0
        assert MAX_MATCHES_TO_PROCESS > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
