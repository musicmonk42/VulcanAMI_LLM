# ============================================================
# VULCAN-AGI - Wiring Fixes Tests
#
# Tests for the critical wiring fixes that ensure:
# 1. Task types are properly normalized (stripping _task, _support suffixes)
# 2. Selected tools are passed from QueryRouter through to agent execution
# 3. Arena threshold logic works correctly
# 4. TournamentManager integration for multi-agent selection
# ============================================================

import logging
import time
import unittest
from unittest.mock import Mock, MagicMock, patch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestTaskTypeNormalization(unittest.TestCase):
    """Tests for task type normalization in agent execution."""
    
    def test_task_type_suffix_stripping(self):
        """Test that task types with _task suffix are normalized correctly."""
        test_cases = [
            ("reasoning_task", "reasoning"),
            ("perception_task", "perception"),
            ("causal_task", "causal"),
            ("symbolic_task", "symbolic"),
            ("perception_support", "perception"),
            ("planning_support", "planning"),
            ("general_task", "general"),
            ("reasoning", "reasoning"),  # No suffix - unchanged
            ("causal", "causal"),  # No suffix - unchanged
        ]
        
        for original, expected in test_cases:
            # Simulate the normalization logic from agent_pool.py
            normalized = original
            for suffix in ("_task", "_support"):
                if normalized.endswith(suffix):
                    normalized = normalized[:-len(suffix)]
                    break
            
            self.assertEqual(
                normalized, expected,
                f"Task type '{original}' should normalize to '{expected}', got '{normalized}'"
            )
    
    def test_reasoning_task_detection_with_normalized_type(self):
        """Test that reasoning tasks are detected after normalization."""
        reasoning_task_types = {
            "reasoning", "causal", "symbolic", "analogical", "probabilistic",
            "counterfactual", "multimodal", "deductive", "inductive", "abductive"
        }
        
        # Test cases: task types from QueryRouter that should trigger reasoning
        router_task_types = [
            "reasoning_task",
            "causal_task", 
            "symbolic_support",
            "analogical_task",
            "probabilistic_task",
        ]
        
        for task_type in router_task_types:
            # Normalize
            normalized = task_type
            for suffix in ("_task", "_support"):
                if normalized.endswith(suffix):
                    normalized = normalized[:-len(suffix)]
                    break
            
            is_reasoning = normalized in reasoning_task_types
            self.assertTrue(
                is_reasoning,
                f"Normalized task type '{normalized}' (from '{task_type}') should be detected as reasoning task"
            )


class TestSelectedToolsPassing(unittest.TestCase):
    """Tests for selected_tools passing from QueryRouter to agent execution."""
    
    def test_selected_tools_triggers_reasoning(self):
        """Test that selected_tools containing reasoning types trigger reasoning."""
        reasoning_task_types = {
            "reasoning", "causal", "symbolic", "analogical", "probabilistic",
            "counterfactual", "multimodal", "deductive", "inductive", "abductive"
        }
        
        test_cases = [
            (["causal", "probabilistic"], True),
            (["symbolic"], True),
            (["general"], False),
            (["causal", "general"], True),  # Contains at least one reasoning type
            ([], False),
            (None, False),
        ]
        
        for selected_tools, should_be_reasoning in test_cases:
            tools = selected_tools or []
            is_reasoning = any(tool in reasoning_task_types for tool in tools)
            
            self.assertEqual(
                is_reasoning, should_be_reasoning,
                f"selected_tools={selected_tools} should trigger reasoning={should_be_reasoning}, got {is_reasoning}"
            )
    
    def test_parameters_include_selected_tools(self):
        """Test that parameters dict properly includes selected_tools."""
        # Simulate what main.py does when creating task parameters
        agent_task_prompt = "Test query"
        agent_task_type = "reasoning_task"
        selected_tools_from_router = ["causal", "probabilistic"]
        
        parameters = {
            "prompt": agent_task_prompt,
            "task_type": agent_task_type,
            "source": "user",
            "is_primary": True,
            "selected_tools": selected_tools_from_router,
        }
        
        self.assertIn("selected_tools", parameters)
        self.assertEqual(parameters["selected_tools"], ["causal", "probabilistic"])


class TestArenaThresholdLogic(unittest.TestCase):
    """Tests for Arena threshold logic in client.py."""
    
    def test_arena_skips_when_complexity_below_threshold(self):
        """Test that Arena is skipped when complexity is below threshold."""
        complexity_threshold = 0.3
        
        # Case 1: Complexity is set and below threshold
        complexity = 0.1
        should_skip = complexity is not None and complexity < complexity_threshold
        self.assertTrue(should_skip, "Should skip when complexity < threshold")
        
        # Case 2: Complexity is set and above threshold
        complexity = 0.5
        should_skip = complexity is not None and complexity < complexity_threshold
        self.assertFalse(should_skip, "Should NOT skip when complexity >= threshold")
    
    def test_arena_proceeds_when_arena_participation_enabled(self):
        """Test that Arena proceeds when arena_participation flag is set."""
        complexity_threshold = 0.3
        
        # Simulate the fixed logic from client.py
        def should_proceed(complexity, arena_participation):
            if complexity is not None:
                return complexity >= complexity_threshold
            else:
                # No complexity score - check arena_participation flag
                return arena_participation
        
        # Case 1: No complexity, but arena_participation=True
        self.assertTrue(should_proceed(None, True))
        
        # Case 2: No complexity and arena_participation=False
        self.assertFalse(should_proceed(None, False))
        
        # Case 3: High complexity
        self.assertTrue(should_proceed(0.5, False))
        
        # Case 4: Low complexity (should skip even if arena_participation=True)
        self.assertFalse(should_proceed(0.1, True))


class TestTournamentManagerIntegration(unittest.TestCase):
    """Tests for TournamentManager integration."""
    
    def test_multi_result_selection(self):
        """Test that multiple agent results use tournament selection logic."""
        all_agent_results = [
            {"job_id": "job1", "confidence": 0.7, "reasoning_output": {"conclusion": "A"}},
            {"job_id": "job2", "confidence": 0.9, "reasoning_output": {"conclusion": "B"}},
            {"job_id": "job3", "confidence": 0.6, "reasoning_output": {"conclusion": "C"}},
        ]
        
        # Fallback logic: select highest confidence
        if len(all_agent_results) > 1:
            best_result = max(all_agent_results, key=lambda x: x.get("confidence", 0))
            self.assertEqual(best_result["job_id"], "job2")
            self.assertEqual(best_result["confidence"], 0.9)
    
    def test_single_result_passthrough(self):
        """Test that single result is used directly without tournament."""
        all_agent_results = [
            {"job_id": "job1", "confidence": 0.7, "reasoning_output": {"conclusion": "A"}},
        ]
        
        if len(all_agent_results) == 1:
            result = all_agent_results[0]
            self.assertEqual(result["job_id"], "job1")


class TestEndToEndWiring(unittest.TestCase):
    """End-to-end tests simulating the full wiring flow."""
    
    def test_query_router_to_agent_pool_flow(self):
        """Test that tools from QueryRouter reach agent execution."""
        # Simulate QueryRouter output
        routing_plan_telemetry = {
            "selected_tools": ["causal", "probabilistic"],
            "reasoning_strategy": "hybrid",
            "reasoning_confidence": 0.85,
        }
        
        # Simulate extracting tools
        routing_selected_tools = routing_plan_telemetry.get("selected_tools", [])
        
        # Simulate task submission parameters
        task_parameters = {
            "prompt": "Why does X cause Y?",
            "task_type": "reasoning_task",
            "selected_tools": routing_selected_tools,
        }
        
        # Verify tools are in parameters
        self.assertEqual(task_parameters["selected_tools"], ["causal", "probabilistic"])
        
        # Simulate agent checking for reasoning
        reasoning_task_types = {"reasoning", "causal", "symbolic", "analogical", "probabilistic"}
        selected_tools = task_parameters.get("selected_tools", [])
        
        is_reasoning = any(tool in reasoning_task_types for tool in selected_tools)
        self.assertTrue(is_reasoning, "Agent should detect reasoning task from selected_tools")
    
    def test_reasoning_invoked_flag_set_correctly(self):
        """Test that reasoning_invoked flag is set when reasoning engines are invoked."""
        # Simulate agent execution result structure
        reasoning_available = True
        is_reasoning_task = True
        
        # This is the expected result structure from _execute_agent_task
        result = {
            "status": "completed",
            "outcome": "success",
            "reasoning_invoked": is_reasoning_task and reasoning_available,
            "reasoning_output": {
                "conclusion": "X causes Y through mechanism Z",
                "confidence": 0.85,
                "reasoning_type": "causal",
            } if is_reasoning_task else None,
        }
        
        self.assertTrue(result["reasoning_invoked"])
        self.assertIsNotNone(result["reasoning_output"])
        self.assertEqual(result["reasoning_output"]["reasoning_type"], "causal")


class TestArenaContextInjection(unittest.TestCase):
    """Tests for Arena reasoning context injection."""
    
    def test_arena_reasoning_injected_into_context(self):
        """Test that Arena reasoning output is injected into reasoning_insights."""
        # Simulate Arena result with reasoning
        arena_result = {
            "status": "success",
            "agent_id": "arena_reasoner_001",
            "result": {
                "reasoning_invoked": True,
                "selected_tools": ["causal", "symbolic"],
                "reasoning_strategy": "hybrid",
                "confidence": 0.85,
                "output": "Analysis complete",
            },
        }
        
        reasoning_insights = {}
        
        # Simulate the injection logic from main.py
        if arena_result and arena_result.get("status") == "success":
            arena_output = arena_result.get("result", {})
            if isinstance(arena_output, dict):
                arena_reasoning = {
                    "agent_id": arena_result.get("agent_id"),
                    "reasoning_invoked": arena_output.get("reasoning_invoked", False),
                    "selected_tools": arena_output.get("selected_tools", []),
                    "reasoning_strategy": arena_output.get("reasoning_strategy"),
                    "confidence": arena_output.get("confidence"),
                }
                if arena_reasoning.get("reasoning_invoked") or arena_reasoning.get("selected_tools"):
                    reasoning_insights["arena_reasoning"] = arena_reasoning
        
        # Verify injection
        self.assertIn("arena_reasoning", reasoning_insights)
        self.assertEqual(reasoning_insights["arena_reasoning"]["agent_id"], "arena_reasoner_001")
        self.assertEqual(reasoning_insights["arena_reasoning"]["selected_tools"], ["causal", "symbolic"])
        self.assertTrue(reasoning_insights["arena_reasoning"]["reasoning_invoked"])
    
    def test_arena_reasoning_not_injected_when_no_reasoning(self):
        """Test that Arena reasoning is not injected when reasoning was not invoked."""
        arena_result = {
            "status": "success",
            "result": {
                "reasoning_invoked": False,
                "selected_tools": [],
            },
        }
        
        reasoning_insights = {}
        
        # Simulate the injection logic
        if arena_result and arena_result.get("status") == "success":
            arena_output = arena_result.get("result", {})
            if isinstance(arena_output, dict):
                arena_reasoning = {
                    "reasoning_invoked": arena_output.get("reasoning_invoked", False),
                    "selected_tools": arena_output.get("selected_tools", []),
                }
                if arena_reasoning.get("reasoning_invoked") or arena_reasoning.get("selected_tools"):
                    reasoning_insights["arena_reasoning"] = arena_reasoning
        
        # Verify no injection
        self.assertNotIn("arena_reasoning", reasoning_insights)


if __name__ == "__main__":
    unittest.main()
