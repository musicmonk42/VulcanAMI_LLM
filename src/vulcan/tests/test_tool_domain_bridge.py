"""
Comprehensive tests for the ToolDomainBridge module.

Tests the bridge between tool selection and semantic domains, ensuring:
- Correct tool-to-domain mapping
- Cross-domain query detection
- Transfer candidate identification
- Domain execution ordering
- Thread safety
- Statistics tracking

These tests follow industry best practices:
- Comprehensive coverage of all public methods
- Edge case testing
- Thread safety verification
- Performance considerations
- Clear documentation
"""

import logging
import threading
import time
import unittest
from typing import Any, Dict, List, Set
from unittest.mock import MagicMock, patch

import pytest

logger = logging.getLogger(__name__)


class TestToolDomainBridge(unittest.TestCase):
    """Test cases for the ToolDomainBridge class."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            from vulcan.reasoning.tool_domain_bridge import ToolDomainBridge
            self.bridge = ToolDomainBridge()
        except ImportError as e:
            self.skipTest(f"ToolDomainBridge not available: {e}")

    def test_bridge_initialization(self):
        """Test that bridge initializes correctly with expected attributes."""
        self.assertIsNotNone(self.bridge)
        self.assertIsNotNone(self.bridge.TOOL_TO_DOMAIN)
        self.assertIsNotNone(self.bridge.DOMAIN_RELATIONSHIPS)
        self.assertIsNotNone(self.bridge.DOMAIN_PRIORITY)
        self.assertEqual(len(self.bridge._transfer_history), 0)

    def test_get_domains_for_tools_single_tool(self):
        """Test domain mapping for single tool."""
        # Test known tool mappings
        domains = self.bridge.get_domains_for_tools(['probabilistic'])
        self.assertEqual(domains, {'statistical'})

        domains = self.bridge.get_domains_for_tools(['causal'])
        self.assertEqual(domains, {'causal_reasoning'})

        domains = self.bridge.get_domains_for_tools(['symbolic'])
        self.assertEqual(domains, {'logical'})

    def test_get_domains_for_tools_multiple_tools(self):
        """Test domain mapping for multiple tools."""
        domains = self.bridge.get_domains_for_tools(['probabilistic', 'causal'])
        self.assertEqual(domains, {'statistical', 'causal_reasoning'})

        domains = self.bridge.get_domains_for_tools(['symbolic', 'multimodal', 'analogical'])
        self.assertEqual(domains, {'logical', 'perceptual', 'analogical'})

    def test_get_domains_for_tools_unknown_tool(self):
        """Test domain mapping for unknown tools defaults to 'general'."""
        domains = self.bridge.get_domains_for_tools(['unknown_tool'])
        self.assertEqual(domains, {'general'})

        domains = self.bridge.get_domains_for_tools(['probabilistic', 'unknown_tool'])
        self.assertEqual(domains, {'statistical', 'general'})

    def test_get_domains_for_tools_empty_list(self):
        """Test domain mapping for empty tool list."""
        domains = self.bridge.get_domains_for_tools([])
        self.assertEqual(domains, set())

    def test_get_domains_for_tools_case_insensitive(self):
        """Test that tool names are case-insensitive."""
        domains1 = self.bridge.get_domains_for_tools(['PROBABILISTIC'])
        domains2 = self.bridge.get_domains_for_tools(['probabilistic'])
        domains3 = self.bridge.get_domains_for_tools(['Probabilistic'])
        
        self.assertEqual(domains1, domains2)
        self.assertEqual(domains2, domains3)

    def test_is_cross_domain_query_true(self):
        """Test cross-domain detection when multiple domains are involved."""
        # Two different specific domains
        self.assertTrue(
            self.bridge.is_cross_domain_query(['probabilistic', 'causal'])
        )
        
        # Three different specific domains
        self.assertTrue(
            self.bridge.is_cross_domain_query(['symbolic', 'multimodal', 'analogical'])
        )

    def test_is_cross_domain_query_false_single_domain(self):
        """Test cross-domain detection with single domain."""
        # Single tool
        self.assertFalse(
            self.bridge.is_cross_domain_query(['probabilistic'])
        )
        
        # Same domain tools
        self.assertFalse(
            self.bridge.is_cross_domain_query(['probabilistic', 'bayesian'])
        )

    def test_is_cross_domain_query_false_with_general(self):
        """Test that 'general' domain doesn't count for cross-domain."""
        # Specific + general is not cross-domain
        self.assertFalse(
            self.bridge.is_cross_domain_query(['probabilistic', 'general'])
        )
        
        # Only general tools
        self.assertFalse(
            self.bridge.is_cross_domain_query(['general', 'default'])
        )

    def test_can_transfer_between_same_domain(self):
        """Test that same-domain transfer is always allowed."""
        self.assertTrue(
            self.bridge.can_transfer_between('statistical', 'statistical')
        )
        self.assertTrue(
            self.bridge.can_transfer_between('logical', 'logical')
        )

    def test_can_transfer_between_related_domains(self):
        """Test transfer between related domains."""
        # statistical -> causal_reasoning should be allowed
        self.assertTrue(
            self.bridge.can_transfer_between('statistical', 'causal_reasoning')
        )
        
        # causal_reasoning -> logical should be allowed
        self.assertTrue(
            self.bridge.can_transfer_between('causal_reasoning', 'logical')
        )

    def test_can_transfer_between_unrelated_domains(self):
        """Test transfer between unrelated domains."""
        # Check for domains that shouldn't have direct transfer
        # This depends on DOMAIN_RELATIONSHIPS configuration
        # Verify the relationship exists or doesn't
        related = self.bridge.DOMAIN_RELATIONSHIPS.get('perceptual', [])
        if 'statistical' not in related:
            self.assertFalse(
                self.bridge.can_transfer_between('perceptual', 'statistical')
            )

    def test_get_transfer_candidates(self):
        """Test getting transfer candidates for a domain."""
        candidates = self.bridge.get_transfer_candidates('statistical')
        
        self.assertIsInstance(candidates, list)
        self.assertIn('causal_reasoning', candidates)
        self.assertIn('general', candidates)

    def test_get_transfer_candidates_unknown_domain(self):
        """Test getting transfer candidates for unknown domain."""
        candidates = self.bridge.get_transfer_candidates('unknown_domain')
        self.assertEqual(candidates, [])

    def test_identify_primary_domain_from_query_type(self):
        """Test primary domain identification from query type."""
        # Reasoning query type should prefer logical
        primary = self.bridge.identify_primary_domain(
            ['general'], 'reasoning'
        )
        # Should prefer logical for reasoning queries
        self.assertIsInstance(primary, str)

    def test_identify_primary_domain_from_first_tool(self):
        """Test primary domain identification from first tool."""
        primary = self.bridge.identify_primary_domain(
            ['causal', 'probabilistic'], 'general'
        )
        self.assertEqual(primary, 'causal_reasoning')

    def test_identify_primary_domain_single_specific(self):
        """Test primary domain with single specific domain."""
        primary = self.bridge.identify_primary_domain(
            ['symbolic'], 'general'
        )
        self.assertEqual(primary, 'logical')

    def test_identify_primary_domain_fallback_to_general(self):
        """Test primary domain falls back to general."""
        primary = self.bridge.identify_primary_domain(
            ['general', 'default'], 'execution'
        )
        self.assertEqual(primary, 'general')

    def test_get_domain_execution_order_primary_first(self):
        """Test that primary domain comes first in execution order."""
        domains = {'statistical', 'causal_reasoning', 'general'}
        primary = 'causal_reasoning'
        
        order = self.bridge.get_domain_execution_order(domains, primary)
        
        self.assertEqual(order[0], 'causal_reasoning')
        self.assertEqual(len(order), 3)
        self.assertEqual(set(order), domains)

    def test_get_domain_execution_order_respects_priority(self):
        """Test that execution order respects domain priority."""
        domains = {'statistical', 'logical', 'general'}
        primary = 'logical'
        
        order = self.bridge.get_domain_execution_order(domains, primary)
        
        # Primary should be first
        self.assertEqual(order[0], 'logical')
        
        # Remaining should be sorted by priority
        remaining_order = order[1:]
        for i in range(len(remaining_order) - 1):
            priority_i = self.bridge.DOMAIN_PRIORITY.get(remaining_order[i], 100)
            priority_j = self.bridge.DOMAIN_PRIORITY.get(remaining_order[i + 1], 100)
            self.assertLessEqual(priority_i, priority_j)

    def test_get_domain_execution_order_missing_primary(self):
        """Test execution order when primary is not in domains."""
        domains = {'statistical', 'general'}
        primary = 'logical'  # Not in domains
        
        order = self.bridge.get_domain_execution_order(domains, primary)
        
        # Should still contain all domains
        self.assertEqual(set(order), domains)

    def test_get_bidirectional_transfers(self):
        """Test getting all valid transfer pairs."""
        domains = {'statistical', 'causal_reasoning'}
        
        transfers = self.bridge.get_bidirectional_transfers(domains)
        
        self.assertIsInstance(transfers, list)
        # Check that transfers are tuples
        for transfer in transfers:
            self.assertEqual(len(transfer), 2)
            self.assertIn(transfer[0], domains)
            self.assertIn(transfer[1], domains)

    def test_record_transfer(self):
        """Test recording transfer history."""
        initial_len = len(self.bridge._transfer_history)
        
        self.bridge.record_transfer(
            source_domain='statistical',
            target_domain='causal_reasoning',
            success=True,
            concepts_transferred=3,
        )
        
        self.assertEqual(len(self.bridge._transfer_history), initial_len + 1)
        
        # Check last record
        last_record = self.bridge._transfer_history[-1]
        self.assertEqual(last_record['source'], 'statistical')
        self.assertEqual(last_record['target'], 'causal_reasoning')
        self.assertTrue(last_record['success'])
        self.assertEqual(last_record['concepts'], 3)
        self.assertIn('timestamp', last_record)

    def test_record_transfer_history_limit(self):
        """Test that transfer history respects max limit."""
        # Fill history beyond limit
        for i in range(self.bridge._max_history + 10):
            self.bridge.record_transfer(
                source_domain='domain_a',
                target_domain='domain_b',
                success=True,
                concepts_transferred=1,
            )
        
        # Should not exceed max
        self.assertLessEqual(
            len(self.bridge._transfer_history),
            self.bridge._max_history
        )

    def test_get_transfer_statistics_empty(self):
        """Test statistics with no transfers."""
        # Create fresh bridge
        from vulcan.reasoning.tool_domain_bridge import ToolDomainBridge
        fresh_bridge = ToolDomainBridge()
        
        stats = fresh_bridge.get_transfer_statistics()
        
        self.assertEqual(stats['total_transfers'], 0)
        self.assertEqual(stats['successful'], 0)
        self.assertEqual(stats['failed'], 0)
        self.assertEqual(stats['success_rate'], 0.0)

    def test_get_transfer_statistics_with_data(self):
        """Test statistics with transfer data."""
        # Record some transfers
        self.bridge.record_transfer('a', 'b', True, 2)
        self.bridge.record_transfer('b', 'c', True, 3)
        self.bridge.record_transfer('c', 'd', False, 0)
        
        stats = self.bridge.get_transfer_statistics()
        
        self.assertEqual(stats['total_transfers'], 3)
        self.assertEqual(stats['successful'], 2)
        self.assertEqual(stats['failed'], 1)
        self.assertAlmostEqual(stats['success_rate'], 2/3, places=5)
        self.assertEqual(stats['total_concepts_transferred'], 5)


class TestToolDomainBridgeSingleton(unittest.TestCase):
    """Test the singleton accessor for ToolDomainBridge."""

    def test_singleton_returns_same_instance(self):
        """Test that get_tool_domain_bridge returns singleton."""
        try:
            from vulcan.reasoning.tool_domain_bridge import get_tool_domain_bridge

            bridge1 = get_tool_domain_bridge()
            bridge2 = get_tool_domain_bridge()

            self.assertIs(bridge1, bridge2)
        except ImportError:
            self.skipTest("Module not available")


class TestToolDomainBridgeThreadSafety(unittest.TestCase):
    """Thread safety tests for ToolDomainBridge."""

    def test_concurrent_domain_mapping(self):
        """Test that concurrent domain mappings are safe."""
        try:
            from vulcan.reasoning.tool_domain_bridge import ToolDomainBridge
            bridge = ToolDomainBridge()
        except ImportError:
            self.skipTest("Module not available")
            return

        results = []
        results_lock = threading.Lock()
        errors = []

        def map_domains(thread_num):
            try:
                tools = ['probabilistic', 'causal', 'symbolic']
                for _ in range(100):
                    domains = bridge.get_domains_for_tools(tools)
                    with results_lock:
                        results.append(domains)
            except Exception as e:
                with results_lock:
                    errors.append((thread_num, str(e)))

        threads = [threading.Thread(target=map_domains, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Errors: {errors}")
        self.assertEqual(len(results), 1000)  # 10 threads * 100 iterations

    def test_concurrent_transfer_recording(self):
        """Test that concurrent transfer recordings are safe."""
        try:
            from vulcan.reasoning.tool_domain_bridge import ToolDomainBridge
            bridge = ToolDomainBridge()
        except ImportError:
            self.skipTest("Module not available")
            return

        errors = []
        errors_lock = threading.Lock()

        def record_transfers(thread_num):
            try:
                for i in range(50):
                    bridge.record_transfer(
                        source_domain=f'domain_{thread_num}',
                        target_domain='general',
                        success=True,
                        concepts_transferred=i,
                    )
            except Exception as e:
                with errors_lock:
                    errors.append((thread_num, str(e)))

        threads = [threading.Thread(target=record_transfers, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Errors: {errors}")
        # Should have recorded many transfers (up to max_history)
        self.assertGreater(len(bridge._transfer_history), 0)


class TestToolDomainBridgeEdgeCases(unittest.TestCase):
    """Edge case tests for ToolDomainBridge."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            from vulcan.reasoning.tool_domain_bridge import ToolDomainBridge
            self.bridge = ToolDomainBridge()
        except ImportError as e:
            self.skipTest(f"ToolDomainBridge not available: {e}")

    def test_tools_with_whitespace(self):
        """Test handling of tools with whitespace."""
        domains = self.bridge.get_domains_for_tools(['  probabilistic  ', 'causal'])
        self.assertIn('statistical', domains)

    def test_empty_string_tool(self):
        """Test handling of empty string tool."""
        domains = self.bridge.get_domains_for_tools([''])
        self.assertEqual(domains, {'general'})

    def test_all_general_tools(self):
        """Test with all general tools."""
        is_cross = self.bridge.is_cross_domain_query(['general', 'default', 'general'])
        self.assertFalse(is_cross)

    def test_identify_primary_empty_tools(self):
        """Test primary domain identification with empty tools."""
        primary = self.bridge.identify_primary_domain([], 'general')
        self.assertEqual(primary, 'general')

    def test_execution_order_empty_domains(self):
        """Test execution order with empty domains."""
        order = self.bridge.get_domain_execution_order(set(), 'general')
        self.assertEqual(order, [])

    def test_bidirectional_transfers_single_domain(self):
        """Test bidirectional transfers with single domain."""
        transfers = self.bridge.get_bidirectional_transfers({'statistical'})
        self.assertEqual(transfers, [])


if __name__ == "__main__":
    unittest.main()
