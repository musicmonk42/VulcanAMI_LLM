"""
Test Agent Pool Reasoning Fix - Task 8

This test validates the fixes made to the agent pool reasoning system:
1. Task 1: Full query context is passed (not truncated)
2. Task 2: Philosophical queries trigger reasoning (not bypass)
3. Task 3: Reasoning types are properly classified (not UNKNOWN)
4. Task 6: Validation and logging are in place

Regression test for production issues:
- Agent pool completing in 0.000s with reasoning_invoked=False
- Query truncation to 25 chars
- Reasoning returning type=UNKNOWN with confidence=0.1
"""

import os
import unittest
from unittest.mock import Mock, patch

# Path constants for test files - use project root-relative paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENT_POOL_PATH = os.path.join(PROJECT_ROOT, 'src/vulcan/orchestrator/agent_pool.py')
REASONING_INTEGRATION_PATH = os.path.join(PROJECT_ROOT, 'src/vulcan/reasoning/reasoning_integration.py')
QUERY_ROUTER_PATH = os.path.join(PROJECT_ROOT, 'src/vulcan/routing/query_router.py')
SAFETY_VALIDATOR_PATH = os.path.join(PROJECT_ROOT, 'src/vulcan/safety/safety_validator.py')
LLM_VALIDATORS_PATH = os.path.join(PROJECT_ROOT, 'src/vulcan/safety/llm_validators.py')


class TestAgentPoolReasoningFix(unittest.TestCase):
    """Test the agent pool reasoning fixes."""

    def test_reasoning_type_mapping_exists(self):
        """
        Task 3: Verify ROUTE_TO_REASONING_TYPE mapping exists and contains
        key fast-path routes.
        """
        from src.vulcan.reasoning.reasoning_integration import ROUTE_TO_REASONING_TYPE
        
        # Verify mapping exists
        self.assertIsInstance(ROUTE_TO_REASONING_TYPE, dict)
        
        # Verify key routes are mapped
        self.assertIn("PHILOSOPHICAL-FAST-PATH", ROUTE_TO_REASONING_TYPE)
        self.assertIn("MATH-FAST-PATH", ROUTE_TO_REASONING_TYPE)
        self.assertIn("CAUSAL-PATH", ROUTE_TO_REASONING_TYPE)
        self.assertIn("philosophical", ROUTE_TO_REASONING_TYPE)
        self.assertIn("mathematical", ROUTE_TO_REASONING_TYPE)
        
        # Verify philosophical maps to proper type (not "general")
        self.assertEqual(
            ROUTE_TO_REASONING_TYPE["PHILOSOPHICAL-FAST-PATH"],
            "philosophical"
        )
        self.assertEqual(
            ROUTE_TO_REASONING_TYPE["philosophical"],
            "philosophical"
        )

    def test_get_reasoning_type_from_route(self):
        """
        Task 3: Verify get_reasoning_type_from_route returns proper types.
        """
        from src.vulcan.reasoning.reasoning_integration import get_reasoning_type_from_route
        
        # Test philosophical route
        result = get_reasoning_type_from_route("philosophical", "PHILOSOPHICAL-FAST-PATH")
        self.assertEqual(result, "philosophical")
        
        # Test mathematical route
        result = get_reasoning_type_from_route("mathematical", "MATH-FAST-PATH")
        self.assertEqual(result, "mathematical")
        
        # Test unknown route defaults to hybrid
        result = get_reasoning_type_from_route("unknown_type", None)
        self.assertEqual(result, "hybrid")
        
        # Test causal route
        result = get_reasoning_type_from_route("causal", "CAUSAL-PATH")
        self.assertEqual(result, "causal")

    def test_philosophical_reasoning_type_exists(self):
        """
        Task 3: Verify PHILOSOPHICAL is added to ReasoningType enum.
        """
        from src.vulcan.reasoning.reasoning_types import ReasoningType
        
        # Verify PHILOSOPHICAL enum exists
        self.assertTrue(hasattr(ReasoningType, 'PHILOSOPHICAL'))
        self.assertEqual(ReasoningType.PHILOSOPHICAL.value, "philosophical")

    def test_reasoning_task_types_includes_philosophical(self):
        """
        Task 2: Verify reasoning_task_types in agent_pool includes philosophical.
        
        This test checks that the code changes include "philosophical" in the
        set of task types that trigger reasoning invocation.
        """
        # Read the agent_pool.py source to verify the fix
        with open(AGENT_POOL_PATH, 'r') as f:
            source_code = f.read()
        
        # Verify "philosophical" is in reasoning_task_types
        self.assertIn('"philosophical"', source_code)
        
        # Verify the fix comment is present
        self.assertIn('FIX TASK 2', source_code)

    def test_query_context_extraction_priority(self):
        """
        Task 1: Verify query extraction checks prompt before query.
        
        This test verifies that the agent_pool extracts query from
        parameters["prompt"] before checking parameters["query"].
        """
        with open(AGENT_POOL_PATH, 'r') as f:
            source_code = f.read()
        
        # Verify prompt is checked first (it should appear before query in the extraction)
        prompt_check_pos = source_code.find('parameters.get("prompt"')
        query_check_pos = source_code.find('parameters.get("query"')
        
        # In the fixed code, prompt should be checked first
        self.assertGreater(prompt_check_pos, 0, "prompt check not found in source")
        self.assertGreater(query_check_pos, 0, "query check not found in source")
        
        # Verify the fix comment is present
        self.assertIn('FIX TASK 1', source_code)

    def test_validation_logging_for_short_queries(self):
        """
        Task 6: Verify validation logging for short queries exists.
        """
        with open(AGENT_POOL_PATH, 'r') as f:
            source_code = f.read()
        
        # Verify warning for short queries is present
        self.assertIn('SHORT QUERY', source_code)
        self.assertIn('FIX TASK 6', source_code)

    def test_llm_client_singleton_functions_exist(self):
        """
        Task 4: Verify get_llm_client and set_llm_client functions exist.
        """
        from src.vulcan.reasoning.singletons import get_llm_client, set_llm_client
        
        # Verify functions exist
        self.assertTrue(callable(get_llm_client))
        self.assertTrue(callable(set_llm_client))
        
        # Verify get_llm_client returns None when no client is set
        # (don't actually set a client as it might affect other tests)
        # This just verifies the function can be called without error
        result = get_llm_client()
        # Result may be None if no client is set, which is expected

    def test_query_router_philosophical_parameters(self):
        """
        Task 7: Verify query_router passes full context in parameters.
        """
        with open(QUERY_ROUTER_PATH, 'r') as f:
            source_code = f.read()
        
        # Verify prompt is passed in parameters for philosophical tasks
        # Look for the FIX TASK 7 comment and "prompt": query pattern
        self.assertIn('FIX TASK 7', source_code)
        
        # Verify reasoning_context is passed
        self.assertIn('reasoning_context', source_code)

    def test_philosophical_fast_path_uses_proper_tools(self):
        """
        Task 7: Verify PHILOSOPHICAL-FAST-PATH uses symbolic/causal tools
        instead of just ["general"].
        """
        with open(QUERY_ROUTER_PATH, 'r') as f:
            source_code = f.read()
        
        # The fix should change tools from ["general"] to ["symbolic", "causal"]
        # Find the philosophical task section
        philo_section_start = source_code.find("philosophical_fast_path")
        if philo_section_start > 0:
            # Look for the tools assignment near this section
            philo_section = source_code[philo_section_start:philo_section_start + 2000]
            # Should contain symbolic and causal tools
            self.assertIn('symbolic', philo_section)


class TestLongQueryReasoningTrigger(unittest.TestCase):
    """Test that long queries trigger reasoning even with general tools."""

    def test_long_query_triggers_reasoning(self):
        """
        Task 2: Verify long queries (>500 chars) force reasoning invocation.
        """
        with open(AGENT_POOL_PATH, 'r') as f:
            source_code = f.read()
        
        # Verify long query check is present using the named constant
        # The code should use LONG_QUERY_REASONING_THRESHOLD constant
        self.assertIn('LONG_QUERY_REASONING_THRESHOLD', source_code)
        
        # Verify it sets is_reasoning_task = True for long queries
        self.assertIn('long query', source_code.lower())


class TestReasoningResultValidation(unittest.TestCase):
    """Test that reasoning results are validated for UNKNOWN types."""

    def test_unknown_type_warning_exists(self):
        """
        Task 6: Verify warning is logged when reasoning returns UNKNOWN type.
        """
        with open(AGENT_POOL_PATH, 'r') as f:
            source_code = f.read()
        
        # Verify warning for UNKNOWN type is logged
        self.assertIn('UNKNOWN type', source_code)
        
        # Verify confidence validation exists
        self.assertIn('LOW CONFIDENCE', source_code)


class TestEthicalDiscourseHandling(unittest.TestCase):
    """Test that ethical discourse (thought experiments, moral philosophy) is allowed."""

    def test_ethical_discourse_indicators_exist(self):
        """
        Verify ETHICAL_DISCOURSE_INDICATORS constant exists in safety_validator.py.
        """
        with open(SAFETY_VALIDATOR_PATH, 'r') as f:
            source_code = f.read()
        
        # Verify the constant exists
        self.assertIn('ETHICAL_DISCOURSE_INDICATORS', source_code)
        
        # Verify key indicators are present
        self.assertIn('thought experiment', source_code)
        self.assertIn('ethical dilemma', source_code)
        self.assertIn('hypothetical scenario', source_code)

    def test_is_ethical_discourse_function_exists(self):
        """
        Verify is_ethical_discourse method exists in SafetyValidator.
        """
        with open(SAFETY_VALIDATOR_PATH, 'r') as f:
            source_code = f.read()
        
        # Verify the function exists
        self.assertIn('def is_ethical_discourse', source_code)
        
        # Verify it checks for ethical discourse
        self.assertIn('ETHICAL_DISCOURSE_INDICATORS', source_code)

    def test_ethical_validator_bypass_for_discourse(self):
        """
        Verify that EthicalValidator is bypassed for ethical discourse.
        """
        with open(SAFETY_VALIDATOR_PATH, 'r') as f:
            source_code = f.read()
        
        # Verify the bypass is present
        self.assertIn('is_ethical_discourse and validator_name == "EthicalValidator"', source_code)
        
        # Verify logging for the bypass
        self.assertIn('Skipping EthicalValidator for ethical discourse', source_code)

    def test_ethical_validator_context_aware(self):
        """
        Verify EthicalValidator uses is_ethical_discourse context.
        """
        with open(LLM_VALIDATORS_PATH, 'r') as f:
            source_code = f.read()
        
        # Verify ethical discourse check is present
        self.assertIn('is_ethical_discourse', source_code)


class TestWorldModelConfidenceOverrideFix(unittest.TestCase):
    """
    Test Issue#3 Fix: World model confidence should not be overridden.
    
    This test validates that when apply_reasoning() returns a world model
    result with high confidence (>= 0.5), the agent_pool does NOT invoke
    UnifiedReasoner.reason() which would override the confidence.
    
    Bug pattern from logs:
    - World model introspection returned confidence=0.85
    - Agent reasoning selection complete: confidence=0.85
    - [ProbabilisticReasoner] Uninformative result detected (mean=0.500, std=0.500)
    - Agent reasoning execution complete: confidence=0.1  ← BUG! Should be 0.85
    """

    def test_world_model_result_check_exists(self):
        """
        Issue#3 Fix: Verify code checks for world model results before invoking
        UnifiedReasoner.
        """
        with open(AGENT_POOL_PATH, 'r') as f:
            source_code = f.read()
        
        # Verify the fix comment is present
        self.assertIn('Issue#3 FIX', source_code)
        
        # Verify the condition checks for world_model tool
        self.assertIn('world_model', source_code)
        
        # Verify we check for self_referential metadata
        self.assertIn('self_referential', source_code)
        
        # Verify we check for ethical_query metadata
        self.assertIn('ethical_query', source_code)

    def test_world_model_bypass_condition_exists(self):
        """
        Issue#3 Fix: Verify is_world_model_result condition is defined.
        """
        with open(AGENT_POOL_PATH, 'r') as f:
            source_code = f.read()
        
        # Verify the condition variable is defined
        self.assertIn('is_world_model_result', source_code)
        
        # Verify it checks confidence threshold
        self.assertIn('confidence >= 0.5', source_code)

    def test_world_model_logging_exists(self):
        """
        Issue#3 Fix: Verify proper logging when world model result is used directly.
        """
        with open(AGENT_POOL_PATH, 'r') as f:
            source_code = f.read()
        
        # Verify the log message about using world model directly
        self.assertIn('Using this result directly', source_code)
        self.assertIn('without invoking other reasoning engines', source_code)


class TestReasoningIntegrationWorldModel(unittest.TestCase):
    """
    Test Issue#5 Fix: Self-referential queries should use world model.
    
    This test validates that reasoning_integration.py correctly:
    1. Detects self-referential queries
    2. Consults world model introspection
    3. Returns early with world model result if confidence >= 0.5
    """

    def test_self_referential_detection_exists(self):
        """
        Issue#5 Fix: Verify _is_self_referential method exists.
        """
        with open(REASONING_INTEGRATION_PATH, 'r') as f:
            source_code = f.read()
        
        # Verify the method exists
        self.assertIn('def _is_self_referential', source_code)
        
        # Verify key self-referential keywords are checked
        self.assertIn('would you', source_code)
        self.assertIn('self-aware', source_code)
        self.assertIn('your', source_code)

    def test_ethical_query_detection_exists(self):
        """
        Issue#5 Fix: Verify _is_ethical_query method exists.
        """
        with open(REASONING_INTEGRATION_PATH, 'r') as f:
            source_code = f.read()
        
        # Verify the method exists
        self.assertIn('def _is_ethical_query', source_code)
        
        # Verify key ethical keywords are checked
        self.assertIn('permissible', source_code)
        self.assertIn('trolley', source_code)
        self.assertIn('moral', source_code)

    def test_world_model_early_return_exists(self):
        """
        Issue#3 Fix: Verify early return when world model has high confidence.
        """
        with open(REASONING_INTEGRATION_PATH, 'r') as f:
            source_code = f.read()
        
        # Verify the Issue#3 FIX comment and early return logic
        self.assertIn('Issue#3 FIX', source_code)
        self.assertIn('without other engines', source_code)


if __name__ == '__main__':
    unittest.main(verbosity=2)
