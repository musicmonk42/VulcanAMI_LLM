"""Tests that all extracted classes are importable from their canonical locations."""
import pytest


class TestWorldModelImports:
    def test_observation_from_observation_types(self):
        from src.vulcan.world_model.observation_types import Observation
        assert Observation is not None

    def test_model_context_from_observation_types(self):
        from src.vulcan.world_model.observation_types import ModelContext
        assert ModelContext is not None

    def test_observation_processor(self):
        from src.vulcan.world_model.observation_processor import ObservationProcessor
        assert ObservationProcessor is not None

    def test_consistency_validator(self):
        from src.vulcan.world_model.consistency_validator import ConsistencyValidator
        assert ConsistencyValidator is not None


class TestAgentPoolImports:
    def test_types_from_agent_pool_types(self):
        from src.vulcan.orchestrator.agent_pool_types import (
            TOOL_SELECTION_PRIORITY_ORDER,
            SIMPLE_MODE,
            is_privileged_result,
        )
        assert TOOL_SELECTION_PRIORITY_ORDER is not None

    def test_proxy_from_agent_pool_proxy(self):
        from src.vulcan.orchestrator.agent_pool_proxy import (
            AgentPoolProxy,
            is_main_process,
        )
        assert AgentPoolProxy is not None


class TestToolSelectorImports:
    def test_selection_types(self):
        from src.vulcan.reasoning.selection.selection_types import (
            SelectionMode,
            SelectionRequest,
            SelectionResult,
        )
        assert SelectionMode is not None

    def test_feature_extraction(self):
        from src.vulcan.reasoning.selection.feature_extraction import (
            MultiTierFeatureExtractor,
        )
        assert MultiTierFeatureExtractor is not None

    def test_confidence(self):
        from src.vulcan.reasoning.selection.confidence import (
            ToolConfidenceCalibrator,
        )
        assert ToolConfidenceCalibrator is not None

    def test_bandit(self):
        from src.vulcan.reasoning.selection.bandit import ToolSelectionBandit
        assert ToolSelectionBandit is not None


class TestUnifiedImports:
    def test_estimation(self):
        from src.vulcan.reasoning.unified.estimation import (
            reasoning_task_to_plan_step,
        )
        assert callable(reasoning_task_to_plan_step)

    def test_verification(self):
        from src.vulcan.reasoning.unified.verification import (
            postprocess_result,
        )
        assert callable(postprocess_result)

    def test_orchestrator_types(self):
        from src.vulcan.reasoning.unified.orchestrator_types import (
            is_test_environment,
        )
        assert callable(is_test_environment)


class TestPlatformGlobals:
    def test_globals_importable(self):
        from src.platform.globals import (
            get_settings,
            get_app,
            get_service_manager,
            get_flash_manager,
            init_app,
            is_services_init_complete,
            is_services_init_failed,
        )
        assert callable(get_settings)
        assert callable(get_app)
        assert callable(init_app)
