# ============================================================
# VULCAN-AGI Orchestrator - Agent Lifecycle Tests
# Comprehensive test suite for agent_lifecycle.py
# ============================================================

from vulcan.orchestrator.agent_lifecycle import (AgentCapability,
                                                 AgentMetadata, AgentState,
                                                 JobProvenance,
                                                 StateTransitionRules,
                                                 create_agent_metadata,
                                                 create_job_provenance,
                                                 validate_state_machine)
import sys
import time
import unittest
from collections import defaultdict
from pathlib import Path
from unittest.mock import Mock, patch

# Add src directory to path if needed
src_path = Path(__file__).parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import components to test - Updated import path

# ============================================================
# TEST: AGENT STATE ENUM
# ============================================================


class TestAgentState(unittest.TestCase):
    """Test AgentState enum and its methods"""

    def test_state_values(self):
        """Test that all states have correct string values"""
        self.assertEqual(AgentState.UNINITIALIZED.value, "uninitialized")
        self.assertEqual(AgentState.INITIALIZING.value, "initializing")
        self.assertEqual(AgentState.IDLE.value, "idle")
        self.assertEqual(AgentState.WORKING.value, "working")
        self.assertEqual(AgentState.RECOVERING.value, "recovering")
        self.assertEqual(AgentState.RETIRING.value, "retiring")
        self.assertEqual(AgentState.TERMINATED.value, "terminated")
        self.assertEqual(AgentState.ERROR.value, "error")
        self.assertEqual(AgentState.SUSPENDED.value, "suspended")

    def test_state_string_representation(self):
        """Test __str__ method"""
        self.assertEqual(str(AgentState.IDLE), "idle")
        self.assertEqual(str(AgentState.WORKING), "working")

    def test_state_repr(self):
        """Test __repr__ method"""
        self.assertEqual(repr(AgentState.IDLE), "AgentState.IDLE")
        self.assertEqual(repr(AgentState.WORKING), "AgentState.WORKING")

    def test_is_active(self):
        """Test is_active method"""
        # Active states
        self.assertTrue(AgentState.INITIALIZING.is_active())
        self.assertTrue(AgentState.IDLE.is_active())
        self.assertTrue(AgentState.WORKING.is_active())
        self.assertTrue(AgentState.RECOVERING.is_active())

        # Inactive states
        self.assertFalse(AgentState.UNINITIALIZED.is_active())
        self.assertFalse(AgentState.RETIRING.is_active())
        self.assertFalse(AgentState.TERMINATED.is_active())
        self.assertFalse(AgentState.ERROR.is_active())
        self.assertFalse(AgentState.SUSPENDED.is_active())

    def test_is_terminal(self):
        """Test is_terminal method"""
        self.assertTrue(AgentState.TERMINATED.is_terminal())

        # Non-terminal states
        self.assertFalse(AgentState.IDLE.is_terminal())
        self.assertFalse(AgentState.WORKING.is_terminal())
        self.assertFalse(AgentState.ERROR.is_terminal())

    def test_is_error_state(self):
        """Test is_error_state method"""
        self.assertTrue(AgentState.ERROR.is_error_state())
        self.assertTrue(AgentState.RECOVERING.is_error_state())

        # Non-error states
        self.assertFalse(AgentState.IDLE.is_error_state())
        self.assertFalse(AgentState.WORKING.is_error_state())

    def test_can_accept_work(self):
        """Test can_accept_work method"""
        self.assertTrue(AgentState.IDLE.can_accept_work())

        # Cannot accept work
        self.assertFalse(AgentState.WORKING.can_accept_work())
        self.assertFalse(AgentState.RECOVERING.can_accept_work())
        self.assertFalse(AgentState.ERROR.can_accept_work())


# ============================================================
# TEST: AGENT CAPABILITY ENUM
# ============================================================


class TestAgentCapability(unittest.TestCase):
    """Test AgentCapability enum and its methods"""

    def test_capability_values(self):
        """Test that all capabilities have correct string values"""
        self.assertEqual(AgentCapability.PERCEPTION.value, "perception")
        self.assertEqual(AgentCapability.REASONING.value, "reasoning")
        self.assertEqual(AgentCapability.LEARNING.value, "learning")
        self.assertEqual(AgentCapability.PLANNING.value, "planning")
        self.assertEqual(AgentCapability.EXECUTION.value, "execution")
        self.assertEqual(AgentCapability.MEMORY.value, "memory")
        self.assertEqual(AgentCapability.SAFETY.value, "safety")
        self.assertEqual(AgentCapability.GENERAL.value, "general")

    def test_is_specialized(self):
        """Test is_specialized method"""
        self.assertTrue(AgentCapability.PERCEPTION.is_specialized())
        self.assertTrue(AgentCapability.REASONING.is_specialized())
        self.assertFalse(AgentCapability.GENERAL.is_specialized())

    def test_can_handle_capability(self):
        """Test can_handle_capability method"""
        # GENERAL can handle anything
        self.assertTrue(
            AgentCapability.GENERAL.can_handle_capability(AgentCapability.PERCEPTION)
        )
        self.assertTrue(
            AgentCapability.GENERAL.can_handle_capability(AgentCapability.REASONING)
        )
        self.assertTrue(
            AgentCapability.GENERAL.can_handle_capability(AgentCapability.GENERAL)
        )

        # Specialized capabilities match exactly
        self.assertTrue(
            AgentCapability.PERCEPTION.can_handle_capability(AgentCapability.PERCEPTION)
        )
        self.assertFalse(
            AgentCapability.PERCEPTION.can_handle_capability(AgentCapability.REASONING)
        )


# ============================================================
# TEST: STATE TRANSITION RULES
# ============================================================


class TestStateTransitionRules(unittest.TestCase):
    """Test state transition validation"""

    def test_valid_transitions_from_uninitialized(self):
        """Test valid transitions from UNINITIALIZED state"""
        self.assertTrue(
            StateTransitionRules.is_valid_transition(
                AgentState.UNINITIALIZED, AgentState.INITIALIZING
            )
        )
        self.assertTrue(
            StateTransitionRules.is_valid_transition(
                AgentState.UNINITIALIZED, AgentState.TERMINATED
            )
        )

        # Invalid transitions
        self.assertFalse(
            StateTransitionRules.is_valid_transition(
                AgentState.UNINITIALIZED, AgentState.IDLE
            )
        )
        self.assertFalse(
            StateTransitionRules.is_valid_transition(
                AgentState.UNINITIALIZED, AgentState.WORKING
            )
        )

    def test_valid_transitions_from_initializing(self):
        """Test valid transitions from INITIALIZING state"""
        self.assertTrue(
            StateTransitionRules.is_valid_transition(
                AgentState.INITIALIZING, AgentState.IDLE
            )
        )
        self.assertTrue(
            StateTransitionRules.is_valid_transition(
                AgentState.INITIALIZING, AgentState.ERROR
            )
        )
        self.assertTrue(
            StateTransitionRules.is_valid_transition(
                AgentState.INITIALIZING, AgentState.TERMINATED
            )
        )

        # Invalid
        self.assertFalse(
            StateTransitionRules.is_valid_transition(
                AgentState.INITIALIZING, AgentState.WORKING
            )
        )

    def test_valid_transitions_from_idle(self):
        """Test valid transitions from IDLE state"""
        self.assertTrue(
            StateTransitionRules.is_valid_transition(
                AgentState.IDLE, AgentState.WORKING
            )
        )
        self.assertTrue(
            StateTransitionRules.is_valid_transition(
                AgentState.IDLE, AgentState.RETIRING
            )
        )
        self.assertTrue(
            StateTransitionRules.is_valid_transition(
                AgentState.IDLE, AgentState.SUSPENDED
            )
        )
        self.assertTrue(
            StateTransitionRules.is_valid_transition(
                AgentState.IDLE, AgentState.TERMINATED
            )
        )

        # Invalid
        self.assertFalse(
            StateTransitionRules.is_valid_transition(
                AgentState.IDLE, AgentState.INITIALIZING
            )
        )

    def test_valid_transitions_from_working(self):
        """Test valid transitions from WORKING state"""
        self.assertTrue(
            StateTransitionRules.is_valid_transition(
                AgentState.WORKING, AgentState.IDLE
            )
        )
        self.assertTrue(
            StateTransitionRules.is_valid_transition(
                AgentState.WORKING, AgentState.ERROR
            )
        )
        self.assertTrue(
            StateTransitionRules.is_valid_transition(
                AgentState.WORKING, AgentState.RETIRING
            )
        )
        self.assertTrue(
            StateTransitionRules.is_valid_transition(
                AgentState.WORKING, AgentState.TERMINATED
            )
        )

    def test_terminal_state_no_transitions(self):
        """Test that TERMINATED state has no valid transitions"""
        for state in AgentState:
            self.assertFalse(
                StateTransitionRules.is_valid_transition(AgentState.TERMINATED, state)
            )

    def test_get_allowed_transitions(self):
        """Test get_allowed_transitions method"""
        allowed = StateTransitionRules.get_allowed_transitions(AgentState.IDLE)
        self.assertIn(AgentState.WORKING, allowed)
        self.assertIn(AgentState.RETIRING, allowed)
        self.assertIn(AgentState.SUSPENDED, allowed)
        self.assertIn(AgentState.TERMINATED, allowed)
        self.assertEqual(len(allowed), 4)

    def test_validate_transition_logging(self):
        """Test validate_transition with logging"""
        # Valid transition should return True
        self.assertTrue(
            StateTransitionRules.validate_transition(
                AgentState.IDLE, AgentState.WORKING, "test_agent"
            )
        )

        # Invalid transition should return False
        self.assertFalse(
            StateTransitionRules.validate_transition(
                AgentState.IDLE, AgentState.INITIALIZING, "test_agent"
            )
        )


# ============================================================
# TEST: AGENT METADATA
# ============================================================


class TestAgentMetadata(unittest.TestCase):
    """Test AgentMetadata class"""

    def setUp(self):
        """Create test metadata instance"""
        self.metadata = AgentMetadata(
            agent_id="test_agent_001",
            state=AgentState.INITIALIZING,
            capability=AgentCapability.GENERAL,
            created_at=time.time(),
            last_active=time.time(),
        )

    def test_initialization(self):
        """Test metadata initialization"""
        self.assertEqual(self.metadata.agent_id, "test_agent_001")
        self.assertEqual(self.metadata.state, AgentState.INITIALIZING)
        self.assertEqual(self.metadata.capability, AgentCapability.GENERAL)
        self.assertEqual(self.metadata.tasks_completed, 0)
        self.assertEqual(self.metadata.tasks_failed, 0)
        self.assertEqual(len(self.metadata.state_history), 1)  # Initial state recorded

    def test_invalid_state_type(self):
        """Test that invalid state type raises error"""
        with self.assertRaises(ValueError):
            AgentMetadata(
                agent_id="test",
                state="invalid",  # Should be AgentState enum
                capability=AgentCapability.GENERAL,
                created_at=time.time(),
                last_active=time.time(),
            )

    def test_invalid_capability_type(self):
        """Test that invalid capability type raises error"""
        with self.assertRaises(ValueError):
            AgentMetadata(
                agent_id="test",
                state=AgentState.IDLE,
                capability="invalid",  # Should be AgentCapability enum
                created_at=time.time(),
                last_active=time.time(),
            )

    def test_valid_state_transition(self):
        """Test valid state transition"""
        success = self.metadata.transition_state(
            AgentState.IDLE, "Initialization complete"
        )
        self.assertTrue(success)
        self.assertEqual(self.metadata.state, AgentState.IDLE)
        self.assertEqual(len(self.metadata.state_history), 2)
        self.assertEqual(self.metadata.total_state_transitions, 1)

    def test_invalid_state_transition(self):
        """Test invalid state transition"""
        success = self.metadata.transition_state(
            AgentState.WORKING,  # Invalid from INITIALIZING
            "Invalid transition",
        )
        self.assertFalse(success)
        self.assertEqual(self.metadata.state, AgentState.INITIALIZING)  # Unchanged

    def test_state_history_bounded(self):
        """Test that state history is bounded to 100 entries"""
        # Transition through states many times
        for i in range(150):
            if i % 2 == 0:
                self.metadata.transition_state(AgentState.IDLE, f"transition_{i}")
            else:
                self.metadata.transition_state(AgentState.WORKING, f"transition_{i}")

        # Should only keep last 100
        self.assertLessEqual(len(self.metadata.state_history), 100)

    def test_record_task_completion_success(self):
        """Test recording successful task completion"""
        self.metadata.record_task_completion(success=True, duration_s=1.5)

        self.assertEqual(self.metadata.tasks_completed, 1)
        self.assertEqual(self.metadata.tasks_failed, 0)
        self.assertEqual(self.metadata.consecutive_errors, 0)
        self.assertAlmostEqual(self.metadata.average_task_duration_s, 1.5)
        self.assertIsNotNone(self.metadata.last_successful_task_time)

    def test_record_task_completion_failure(self):
        """Test recording failed task completion"""
        self.metadata.record_task_completion(success=False, duration_s=0.5)

        self.assertEqual(self.metadata.tasks_completed, 0)
        self.assertEqual(self.metadata.tasks_failed, 1)
        self.assertEqual(self.metadata.consecutive_errors, 1)
        self.assertIsNotNone(self.metadata.last_error_time)

    def test_average_task_duration_calculation(self):
        """Test exponential moving average for task duration"""
        # First task
        self.metadata.record_task_completion(success=True, duration_s=2.0)
        self.assertAlmostEqual(self.metadata.average_task_duration_s, 2.0)

        # Second task (should use EMA with alpha=0.3)
        self.metadata.record_task_completion(success=True, duration_s=4.0)
        expected = 0.3 * 4.0 + 0.7 * 2.0  # 1.2 + 1.4 = 2.6
        self.assertAlmostEqual(self.metadata.average_task_duration_s, expected)

    def test_record_error(self):
        """Test error recording"""
        test_error = ValueError("Test error")
        context = {"phase": "execution", "task_id": "test_123"}

        self.metadata.record_error(test_error, context)

        self.assertEqual(len(self.metadata.error_history), 1)
        self.assertEqual(self.metadata.consecutive_errors, 1)

        error_record = self.metadata.error_history[0]
        self.assertEqual(error_record["error_type"], "ValueError")
        self.assertEqual(error_record["error_message"], "Test error")
        self.assertEqual(error_record["context"], context)

    def test_error_history_bounded(self):
        """Test that error history is bounded to 50 entries"""
        for i in range(75):
            self.metadata.record_error(Exception(f"Error {i}"), {})

        self.assertEqual(len(self.metadata.error_history), 50)

    def test_should_recover_true(self):
        """Test should_recover returns True when appropriate"""
        self.metadata.state = AgentState.ERROR
        self.metadata.consecutive_errors = 3

        self.assertTrue(self.metadata.should_recover())

    def test_should_recover_false_too_many_errors(self):
        """Test should_recover returns False with too many errors"""
        self.metadata.state = AgentState.ERROR
        self.metadata.consecutive_errors = 5

        self.assertFalse(self.metadata.should_recover())

    def test_should_recover_false_terminal_state(self):
        """Test should_recover returns False in terminal state"""
        self.metadata.state = AgentState.TERMINATED
        self.metadata.consecutive_errors = 2

        self.assertFalse(self.metadata.should_recover())

    def test_should_retire_high_consecutive_errors(self):
        """Test should_retire returns True with high consecutive errors"""
        self.metadata.consecutive_errors = 5

        self.assertTrue(self.metadata.should_retire())

    def test_should_retire_high_failure_rate(self):
        """Test should_retire returns True with high failure rate"""
        # 8 failures out of 10 tasks = 80% failure rate
        for _ in range(8):
            self.metadata.record_task_completion(success=False, duration_s=1.0)
        for _ in range(2):
            self.metadata.record_task_completion(success=True, duration_s=1.0)

        self.assertTrue(self.metadata.should_retire())

    def test_should_retire_idle_too_long(self):
        """Test should_retire returns True when idle too long"""
        self.metadata.state = AgentState.IDLE
        self.metadata.last_active = time.time() - 400  # 400 seconds ago

        self.assertTrue(self.metadata.should_retire())

    def test_get_health_score_perfect(self):
        """Test health score calculation with perfect performance"""
        # Perfect success rate
        for _ in range(10):
            self.metadata.record_task_completion(success=True, duration_s=0.1)

        health_score = self.metadata.get_health_score()

        # Should be high (close to 1.0)
        self.assertGreater(health_score, 0.7)
        self.assertLessEqual(health_score, 1.0)

    def test_get_health_score_poor(self):
        """Test health score calculation with poor performance"""
        # Poor success rate
        for _ in range(10):
            self.metadata.record_task_completion(success=False, duration_s=1.0)

        # Recent error
        self.metadata.record_error(Exception("test"), {})

        health_score = self.metadata.get_health_score()

        # Should be low
        self.assertLess(health_score, 0.5)
        self.assertGreaterEqual(health_score, 0.0)

    def test_get_health_score_new_agent(self):
        """Test health score for new agent with no task history"""
        health_score = self.metadata.get_health_score()

        # Should default to neutral (0.8 for new agents)
        self.assertAlmostEqual(health_score, 0.8)

    def test_is_healthy(self):
        """Test is_healthy method"""
        # Good performance
        for _ in range(10):
            self.metadata.record_task_completion(success=True, duration_s=0.1)

        self.assertTrue(self.metadata.is_healthy(threshold=0.5))

        # Poor performance
        for _ in range(10):
            self.metadata.record_task_completion(success=False, duration_s=1.0)

        self.assertFalse(self.metadata.is_healthy(threshold=0.5))

    def test_get_summary(self):
        """Test get_summary method"""
        self.metadata.transition_state(AgentState.IDLE, "Ready")
        self.metadata.record_task_completion(success=True, duration_s=1.0)

        summary = self.metadata.get_summary()

        self.assertEqual(summary["agent_id"], "test_agent_001")
        self.assertEqual(summary["state"], "idle")
        self.assertEqual(summary["capability"], "general")
        self.assertEqual(summary["tasks_completed"], 1)
        self.assertEqual(summary["tasks_failed"], 0)
        self.assertIn("health_score", summary)
        self.assertIn("performance_metrics", summary)
        self.assertTrue(summary["can_accept_work"])


# ============================================================
# TEST: JOB PROVENANCE
# ============================================================


class TestJobProvenance(unittest.TestCase):
    """Test JobProvenance class"""

    def setUp(self):
        """Create test provenance instance"""
        self.provenance = JobProvenance(
            job_id="job_001",
            agent_id="agent_001",
            graph_id="graph_001",
            parameters={"param1": "value1"},
            hardware_used={"cpu": "Intel i7"},
            start_time=time.time(),
            end_time=None,
            outcome=None,
            result=None,
            error=None,
            resource_consumption={},
        )

    def test_initialization(self):
        """Test provenance initialization"""
        self.assertEqual(self.provenance.job_id, "job_001")
        self.assertEqual(self.provenance.agent_id, "agent_001")
        self.assertEqual(self.provenance.retry_count, 0)
        self.assertEqual(self.provenance.priority, 0)
        self.assertIsNone(self.provenance.outcome)

    def test_complete_success(self):
        """Test completing job successfully"""
        result_data = {"status": "ok", "value": 42}

        self.provenance.complete("success", result=result_data)

        self.assertEqual(self.provenance.outcome, "success")
        self.assertEqual(self.provenance.result, result_data)
        self.assertIsNotNone(self.provenance.end_time)
        self.assertIsNotNone(self.provenance.actual_duration)

    def test_complete_failure(self):
        """Test completing job with failure"""
        error_msg = "Task execution failed"

        self.provenance.complete("failed", error=error_msg)

        self.assertEqual(self.provenance.outcome, "failed")
        self.assertEqual(self.provenance.error, error_msg)
        self.assertIsNotNone(self.provenance.end_time)

    def test_start_execution(self):
        """Test start_execution method"""
        time.sleep(0.1)  # Small delay

        self.provenance.start_execution()

        self.assertIsNotNone(self.provenance.execution_start_time)
        self.assertIsNotNone(self.provenance.queue_time)
        self.assertGreater(self.provenance.queue_time, 0)

    def test_is_complete(self):
        """Test is_complete method"""
        self.assertFalse(self.provenance.is_complete())

        self.provenance.complete("success")

        self.assertTrue(self.provenance.is_complete())

    def test_is_successful(self):
        """Test is_successful method"""
        self.assertFalse(self.provenance.is_successful())

        self.provenance.complete("success")

        self.assertTrue(self.provenance.is_successful())

    def test_is_failed(self):
        """Test is_failed method"""
        self.assertFalse(self.provenance.is_failed())

        self.provenance.complete("failed")
        self.assertTrue(self.provenance.is_failed())

        self.provenance.outcome = "timeout"
        self.assertTrue(self.provenance.is_failed())

        self.provenance.outcome = "cancelled"
        self.assertTrue(self.provenance.is_failed())

    def test_should_retry_true(self):
        """Test should_retry returns True when appropriate"""
        self.provenance.complete("failed")
        self.provenance.retry_count = 2

        self.assertTrue(self.provenance.should_retry(max_retries=3))

    def test_should_retry_false_max_retries(self):
        """Test should_retry returns False at max retries"""
        self.provenance.complete("failed")
        self.provenance.retry_count = 3

        self.assertFalse(self.provenance.should_retry(max_retries=3))

    def test_should_retry_false_cancelled(self):
        """Test should_retry returns False for cancelled jobs"""
        self.provenance.complete("cancelled")
        self.provenance.retry_count = 0

        self.assertFalse(self.provenance.should_retry(max_retries=3))

    def test_get_duration(self):
        """Test get_duration method"""
        self.assertIsNone(self.provenance.get_duration())

        time.sleep(0.1)
        self.provenance.complete("success")

        duration = self.provenance.get_duration()
        self.assertIsNotNone(duration)
        self.assertGreater(duration, 0)

    def test_get_execution_duration(self):
        """Test get_execution_duration method"""
        self.assertIsNone(self.provenance.get_execution_duration())

        time.sleep(0.05)
        self.provenance.start_execution()
        time.sleep(0.05)
        self.provenance.complete("success")

        exec_duration = self.provenance.get_execution_duration()
        self.assertIsNotNone(exec_duration)
        self.assertGreater(exec_duration, 0)

        # Execution duration should be less than total duration
        total_duration = self.provenance.get_duration()
        self.assertLess(exec_duration, total_duration)

    def test_add_checkpoint(self):
        """Test add_checkpoint method"""
        self.provenance.add_checkpoint("/path/to/checkpoint1.pkl")
        self.provenance.add_checkpoint("/path/to/checkpoint2.pkl")

        self.assertEqual(len(self.provenance.checkpoint_paths), 2)
        self.assertIn("/path/to/checkpoint1.pkl", self.provenance.checkpoint_paths)

        # Adding duplicate should not increase count
        self.provenance.add_checkpoint("/path/to/checkpoint1.pkl")
        self.assertEqual(len(self.provenance.checkpoint_paths), 2)

    def test_add_child_job(self):
        """Test add_child_job method"""
        self.provenance.add_child_job("child_job_001")
        self.provenance.add_child_job("child_job_002")

        self.assertEqual(len(self.provenance.child_job_ids), 2)
        self.assertIn("child_job_001", self.provenance.child_job_ids)

        # Adding duplicate should not increase count
        self.provenance.add_child_job("child_job_001")
        self.assertEqual(len(self.provenance.child_job_ids), 2)

    def test_update_resource_consumption(self):
        """Test update_resource_consumption method"""
        self.provenance.update_resource_consumption(
            {"cpu_seconds": 10.5, "memory_mb": 256}
        )

        self.assertEqual(self.provenance.resource_consumption["cpu_seconds"], 10.5)
        self.assertEqual(self.provenance.resource_consumption["memory_mb"], 256)

        # Update with additional resources
        self.provenance.update_resource_consumption({"gpu_seconds": 5.2})

        self.assertEqual(self.provenance.resource_consumption["gpu_seconds"], 5.2)
        self.assertEqual(self.provenance.resource_consumption["cpu_seconds"], 10.5)

    def test_get_summary(self):
        """Test get_summary method"""
        self.provenance.start_execution()
        time.sleep(0.05)
        self.provenance.complete("success", result={"value": 42})

        summary = self.provenance.get_summary()

        self.assertEqual(summary["job_id"], "job_001")
        self.assertEqual(summary["agent_id"], "agent_001")
        self.assertEqual(summary["outcome"], "success")
        self.assertIsNotNone(summary["duration"])
        self.assertIsNotNone(summary["execution_duration"])
        self.assertIsNotNone(summary["queue_time"])
        self.assertFalse(summary["has_error"])

    def test_to_dict(self):
        """Test to_dict method for serialization"""
        self.provenance.complete("success", result={"value": 42})

        data_dict = self.provenance.to_dict()

        self.assertEqual(data_dict["job_id"], "job_001")
        self.assertEqual(data_dict["outcome"], "success")
        self.assertIn("start_time", data_dict)
        self.assertIn("end_time", data_dict)
        self.assertIn("parameters", data_dict)
        self.assertIn("metadata", data_dict)


# ============================================================
# TEST: FACTORY FUNCTIONS
# ============================================================


class TestFactoryFunctions(unittest.TestCase):
    """Test factory functions"""

    def test_create_agent_metadata(self):
        """Test create_agent_metadata factory function"""
        metadata = create_agent_metadata(
            agent_id="factory_agent_001",
            capability=AgentCapability.PERCEPTION,
            location="cloud",
            hardware_spec={"gpu": "NVIDIA A100"},
        )

        self.assertEqual(metadata.agent_id, "factory_agent_001")
        self.assertEqual(metadata.state, AgentState.INITIALIZING)
        self.assertEqual(metadata.capability, AgentCapability.PERCEPTION)
        self.assertEqual(metadata.location, "cloud")
        self.assertEqual(metadata.hardware_spec["gpu"], "NVIDIA A100")

    def test_create_agent_metadata_defaults(self):
        """Test create_agent_metadata with default values"""
        metadata = create_agent_metadata(agent_id="default_agent")

        self.assertEqual(metadata.capability, AgentCapability.GENERAL)
        self.assertEqual(metadata.location, "local")
        self.assertEqual(metadata.hardware_spec, {})

    def test_create_job_provenance(self):
        """Test create_job_provenance factory function"""
        provenance = create_job_provenance(
            job_id="factory_job_001",
            graph_id="graph_001",
            parameters={"key": "value"},
            priority=5,
            timeout_seconds=30.0,
            parent_job_id="parent_job",
        )

        self.assertEqual(provenance.job_id, "factory_job_001")
        self.assertEqual(provenance.graph_id, "graph_001")
        self.assertEqual(provenance.parameters["key"], "value")
        self.assertEqual(provenance.priority, 5)
        self.assertEqual(provenance.timeout_seconds, 30.0)
        self.assertEqual(provenance.parent_job_id, "parent_job")
        self.assertEqual(provenance.agent_id, "")  # Not assigned yet

    def test_create_job_provenance_defaults(self):
        """Test create_job_provenance with default values"""
        provenance = create_job_provenance(
            job_id="default_job", graph_id="default_graph"
        )

        self.assertEqual(provenance.parameters, {})
        self.assertEqual(provenance.priority, 0)
        self.assertIsNone(provenance.timeout_seconds)
        self.assertIsNone(provenance.parent_job_id)


# ============================================================
# TEST: STATE MACHINE VALIDATION
# ============================================================


class TestStateMachineValidation(unittest.TestCase):
    """Test state machine validation"""

    def test_validate_state_machine_success(self):
        """Test that state machine validation passes"""
        # Should not raise exception
        try:
            result = validate_state_machine()
            self.assertTrue(result)
        except Exception as e:
            self.fail(f"validate_state_machine raised exception: {e}")

    def test_all_states_have_transitions(self):
        """Test that all states are defined in transition rules"""
        all_states = set(AgentState)
        defined_states = set(StateTransitionRules.VALID_TRANSITIONS.keys())

        self.assertEqual(all_states, defined_states)

    def test_transition_frozensets(self):
        """Test that transition values are frozensets"""
        for state, transitions in StateTransitionRules.VALID_TRANSITIONS.items():
            self.assertIsInstance(transitions, frozenset)

            # All transition targets should be valid AgentState values
            for target in transitions:
                self.assertIsInstance(target, AgentState)


# ============================================================
# TEST: INTEGRATION SCENARIOS
# ============================================================


class TestIntegrationScenarios(unittest.TestCase):
    """Test realistic integration scenarios"""

    def test_agent_lifecycle_happy_path(self):
        """Test complete agent lifecycle from creation to retirement"""
        # Create agent
        agent = create_agent_metadata(
            agent_id="integration_agent", capability=AgentCapability.EXECUTION
        )

        # Initialize
        self.assertTrue(agent.transition_state(AgentState.IDLE, "Ready"))

        # Assign work
        self.assertTrue(agent.transition_state(AgentState.WORKING, "Task assigned"))

        # Complete successfully
        agent.record_task_completion(success=True, duration_s=1.5)
        self.assertTrue(agent.transition_state(AgentState.IDLE, "Task complete"))

        # Do more work
        for i in range(5):
            self.assertTrue(agent.transition_state(AgentState.WORKING, f"Task {i}"))
            agent.record_task_completion(success=True, duration_s=1.0 + i * 0.1)
            self.assertTrue(
                agent.transition_state(AgentState.IDLE, f"Task {i} complete")
            )

        # Retire gracefully
        self.assertTrue(agent.transition_state(AgentState.RETIRING, "Scaling down"))
        self.assertTrue(agent.transition_state(AgentState.TERMINATED, "Retired"))

        # Verify final state
        self.assertEqual(agent.state, AgentState.TERMINATED)
        self.assertEqual(agent.tasks_completed, 6)
        self.assertEqual(agent.tasks_failed, 0)
        self.assertGreater(agent.get_health_score(), 0.7)

    def test_agent_error_recovery(self):
        """Test agent error handling and recovery"""
        agent = create_agent_metadata(agent_id="error_recovery_agent")
        agent.transition_state(AgentState.IDLE, "Ready")

        # Task fails
        agent.transition_state(AgentState.WORKING, "Task 1")
        agent.record_task_completion(success=False, duration_s=0.5)
        agent.transition_state(AgentState.ERROR, "Task failed")

        # Should be recoverable
        self.assertTrue(agent.should_recover())
        self.assertFalse(agent.should_retire())

        # Recover
        agent.transition_state(AgentState.RECOVERING, "Attempting recovery")
        agent.transition_state(AgentState.IDLE, "Recovered")

        # Successfully complete next task
        agent.transition_state(AgentState.WORKING, "Task 2")
        agent.record_task_completion(success=True, duration_s=1.0)
        agent.transition_state(AgentState.IDLE, "Success")

        # Consecutive errors should be reset
        self.assertEqual(agent.consecutive_errors, 0)

    def test_job_with_retries(self):
        """Test job retry logic"""
        job = create_job_provenance(
            job_id="retry_job", graph_id="test_graph", timeout_seconds=10.0
        )

        # First attempt fails
        job.start_execution()
        time.sleep(0.01)
        job.complete("failed", error="Network timeout")

        # Should retry
        self.assertTrue(job.should_retry(max_retries=3))
        self.assertEqual(job.retry_count, 0)

        # Increment retry counter
        job.retry_count += 1
        job.outcome = None  # Reset for retry

        # Second attempt also fails
        job.start_execution()
        time.sleep(0.01)
        job.complete("timeout", error="Timeout again")

        # Should still retry
        self.assertTrue(job.should_retry(max_retries=3))
        job.retry_count += 1
        job.outcome = None

        # Third attempt succeeds
        job.start_execution()
        time.sleep(0.01)
        job.complete("success", result={"value": 42})

        # Verify final state
        self.assertTrue(job.is_successful())
        self.assertEqual(job.retry_count, 2)
        self.assertEqual(job.result["value"], 42)

    def test_parent_child_job_relationship(self):
        """Test parent-child job relationships"""
        # Create parent job
        parent = create_job_provenance(job_id="parent_job", graph_id="parent_graph")

        # Create child jobs
        child1 = create_job_provenance(
            job_id="child_job_1", graph_id="child_graph_1", parent_job_id=parent.job_id
        )

        child2 = create_job_provenance(
            job_id="child_job_2", graph_id="child_graph_2", parent_job_id=parent.job_id
        )

        # Register children with parent
        parent.add_child_job(child1.job_id)
        parent.add_child_job(child2.job_id)

        # Verify relationships
        self.assertEqual(len(parent.child_job_ids), 2)
        self.assertEqual(child1.parent_job_id, parent.job_id)
        self.assertEqual(child2.parent_job_id, parent.job_id)

        # Complete children
        child1.complete("success")
        child2.complete("success")

        # Complete parent
        parent.complete("success")

        # Verify summary
        parent_summary = parent.get_summary()
        self.assertEqual(parent_summary["num_children"], 2)
        self.assertTrue(parent_summary["has_parent"] is False)


# ============================================================
# TEST SUITE RUNNER
# ============================================================


def suite():
    """Create test suite"""
    test_suite = unittest.TestSuite()

    # Add all test cases
    test_suite.addTest(unittest.makeSuite(TestAgentState))
    test_suite.addTest(unittest.makeSuite(TestAgentCapability))
    test_suite.addTest(unittest.makeSuite(TestStateTransitionRules))
    test_suite.addTest(unittest.makeSuite(TestAgentMetadata))
    test_suite.addTest(unittest.makeSuite(TestJobProvenance))
    test_suite.addTest(unittest.makeSuite(TestFactoryFunctions))
    test_suite.addTest(unittest.makeSuite(TestStateMachineValidation))
    test_suite.addTest(unittest.makeSuite(TestIntegrationScenarios))

    return test_suite


if __name__ == "__main__":
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
