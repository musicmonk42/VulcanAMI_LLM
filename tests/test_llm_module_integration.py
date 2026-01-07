"""
Integration tests for VULCAN LLM modules.

This module tests the integration of all LLM modules in src/vulcan and related directories:
- exp_probe_1p34m (model checkpoints)
- graphix_vulcan_llm.py (main LLM interface)
- src/data (training data)
- src/execution (LLM execution)
- src/generation (text generation)
- src/integration (component integration)
- src/llm_core (transformer core)
- src/local_llm (local LLM provider)
- src/logs (logging)
- src/training (training utilities)
- src/vulcan/llm (hybrid LLM executor)
- src/vulcan/memory (memory systems)
- src/vulcan/reasoning (reasoning engines)
- src/vulcan/safety (safety validators)

Author: VULCAN-AGI Team
Date: 2026-01-07
"""

import os
import pytest
from pathlib import Path

# Repository root
REPO_ROOT = Path(__file__).parent.parent


class TestLLMModuleImports:
    """Test that all LLM modules can be imported."""

    def test_graphix_vulcan_llm_imports(self):
        """Test main LLM module imports correctly."""
        from graphix_vulcan_llm import GraphixVulcanLLM, build_llm

        assert GraphixVulcanLLM is not None
        assert build_llm is not None

    def test_src_llm_core_imports(self):
        """Test LLM core module imports."""
        from src.llm_core import (
            GraphixTransformer,
            GraphixTransformerConfig,
            TRANSFORMER_AVAILABLE,
        )

        assert GraphixTransformer is not None
        assert GraphixTransformerConfig is not None
        assert TRANSFORMER_AVAILABLE is True

    def test_src_integration_imports(self):
        """Test integration module imports."""
        from src.integration import (
            CognitiveLoop,
            LoopRuntimeConfig,
            LoopSamplingConfig,
            COGNITIVE_LOOP_AVAILABLE,
        )

        assert CognitiveLoop is not None
        assert LoopRuntimeConfig is not None
        assert LoopSamplingConfig is not None
        assert COGNITIVE_LOOP_AVAILABLE is True

    def test_src_generation_imports(self):
        """Test generation module imports."""
        from src.generation import (
            UnifiedGeneration,
            SafeGeneration,
            ExplainableGeneration,
            UNIFIED_GENERATION_AVAILABLE,
            SAFE_GENERATION_AVAILABLE,
        )

        assert UnifiedGeneration is not None
        assert SafeGeneration is not None
        assert ExplainableGeneration is not None
        assert UNIFIED_GENERATION_AVAILABLE is True
        assert SAFE_GENERATION_AVAILABLE is True

    def test_src_execution_imports(self):
        """Test execution module imports."""
        from src.execution import (
            LLMExecutor,
            DynamicArchitecture,
            LLM_EXECUTOR_AVAILABLE,
            DYNAMIC_ARCHITECTURE_AVAILABLE,
        )

        assert LLMExecutor is not None
        assert DynamicArchitecture is not None
        assert LLM_EXECUTOR_AVAILABLE is True
        assert DYNAMIC_ARCHITECTURE_AVAILABLE is True

    def test_src_training_imports(self):
        """Test training module imports."""
        from src.training import (
            GovernedTrainer,
            SelfImprovingTraining,
            GOVERNED_TRAINER_AVAILABLE,
            SELF_IMPROVING_AVAILABLE,
        )

        assert GovernedTrainer is not None
        assert GOVERNED_TRAINER_AVAILABLE is True
        # SelfImprovingTraining may be None if dependencies are missing
        # but the module should still import

    def test_src_local_llm_imports(self):
        """Test local LLM module imports."""
        from src.local_llm import LOCAL_GPT_AVAILABLE

        # LocalGPTProvider requires torch, may not be available
        # but the module import should succeed
        assert LOCAL_GPT_AVAILABLE is not None

    def test_vulcan_llm_imports(self):
        """Test vulcan.llm module imports."""
        from src.vulcan.llm import (
            get_module_info,
            validate_llm_module,
        )

        info = get_module_info()
        assert "version" in info
        assert "components" in info
        assert info["components"]["mock_llm"] is True
        assert info["components"]["hybrid_executor"] is True

    def test_vulcan_reasoning_imports(self):
        """Test vulcan.reasoning module imports."""
        from src.vulcan.reasoning import (
            ReasoningType,
            get_module_status,
        )

        assert ReasoningType is not None
        status = get_module_status()
        assert "language" in status
        # At minimum, language reasoning should be available
        assert status["language"] is True

    def test_vulcan_safety_imports(self):
        """Test vulcan.safety module imports."""
        from src.vulcan.safety import (
            get_safety_validator,
            SAFETY_VALIDATOR_AVAILABLE,
        )

        validator = get_safety_validator()
        assert validator is not None


class TestModelCheckpoints:
    """Test that model checkpoint files exist."""

    def test_exp_probe_model_files_exist(self):
        """Test exp_probe_1p34m model files exist."""
        model_path = REPO_ROOT / "exp_probe_1p34m"
        required_files = [
            "llm_best_model.json",
            "llm_best_model.pt",
            "llm_last_model.pt",
            "llm_meta_state.json",
        ]

        assert model_path.exists(), f"Model path {model_path} does not exist"

        for filename in required_files:
            filepath = model_path / filename
            assert filepath.exists(), f"Required model file {filepath} does not exist"


class TestDataDirectories:
    """Test that data directories exist and contain expected files."""

    def test_src_data_corpus_exists(self):
        """Test src/data/corpus directory exists."""
        corpus_path = REPO_ROOT / "src" / "data" / "corpus"
        assert corpus_path.exists(), f"Corpus path {corpus_path} does not exist"

        # Should contain text files
        txt_files = list(corpus_path.glob("*.txt"))
        assert len(txt_files) > 0, "Corpus should contain text files"

    def test_src_data_validation_exists(self):
        """Test src/data/validation directory exists."""
        validation_path = REPO_ROOT / "src" / "data" / "validation"
        assert validation_path.exists(), f"Validation path {validation_path} does not exist"

    def test_src_logs_exists(self):
        """Test src/logs directory exists."""
        logs_path = REPO_ROOT / "src" / "logs"
        assert logs_path.exists(), f"Logs path {logs_path} does not exist"


class TestLLMFunctionality:
    """Test basic LLM functionality."""

    def test_llm_initialization(self):
        """Test LLM can be initialized."""
        from graphix_vulcan_llm import GraphixVulcanLLM

        llm = GraphixVulcanLLM()
        assert llm is not None
        assert llm.transformer is not None
        assert llm.bridge is not None
        assert llm.cog_loop is not None

    def test_llm_health_check(self):
        """Test LLM health check passes."""
        from graphix_vulcan_llm import GraphixVulcanLLM

        llm = GraphixVulcanLLM()
        health = llm.health_check()
        assert health["status"] == "healthy"

    def test_llm_generation(self):
        """Test basic generation works."""
        from graphix_vulcan_llm import GraphixVulcanLLM

        llm = GraphixVulcanLLM()
        result = llm.generate("Hello", max_tokens=8, explain=False)

        assert result is not None
        assert hasattr(result, "tokens")
        assert hasattr(result, "text")
        assert len(result.tokens) > 0

    def test_llm_internal_components_status(self):
        """Test internal components status method."""
        from graphix_vulcan_llm import GraphixVulcanLLM

        llm = GraphixVulcanLLM()
        status = llm.get_internal_components_status()

        assert "core" in status
        assert "safety" in status
        assert "context" in status
        assert "reasoning" in status
        assert "summary" in status

        # Core should be integrated
        assert status["summary"]["core_integrated"] is True


class TestModuleVersions:
    """Test that modules have proper version information."""

    def test_llm_core_version(self):
        """Test llm_core has version."""
        from src.llm_core import __version__

        assert __version__ is not None
        assert len(__version__) > 0

    def test_integration_version(self):
        """Test integration has version."""
        from src.integration import __version__

        assert __version__ is not None
        assert len(__version__) > 0

    def test_generation_version(self):
        """Test generation has version."""
        from src.generation import __version__

        assert __version__ is not None
        assert len(__version__) > 0

    def test_execution_version(self):
        """Test execution has version."""
        from src.execution import __version__

        assert __version__ is not None
        assert len(__version__) > 0

    def test_training_version(self):
        """Test training has version."""
        from src.training import __version__

        assert __version__ is not None
        assert len(__version__) > 0

    def test_local_llm_version(self):
        """Test local_llm has version."""
        from src.local_llm import __version__

        assert __version__ is not None
        assert len(__version__) > 0

    def test_vulcan_llm_version(self):
        """Test vulcan.llm has version."""
        from src.vulcan.llm import __version__

        assert __version__ is not None
        assert len(__version__) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
