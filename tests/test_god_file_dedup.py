"""Verify God files no longer define extracted classes inline."""
import ast
import pytest
from pathlib import Path


def _get_class_names(filepath: Path) -> list[str]:
    """Return all top-level class names defined in a Python file."""
    try:
        source = filepath.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError):
        return []
    return [node.name for node in ast.iter_child_nodes(tree) if isinstance(node, ast.ClassDef)]


# Classes that were extracted and should no longer be defined inline
WORLD_MODEL_EXTRACTED = {
    "Observation", "ModelContext", "ComponentIntegrationError",
    "NullObjectiveHierarchy", "NullMotivationalIntrospection",
    "NullMetaReasoningComponent", "ObservationProcessor",
    "InterventionManager", "PredictionManager",
    "ConsistencyValidator", "CodeLLMClient",
}

AGENT_POOL_EXTRACTED = {
    "AgentPoolProxy",
}

TOOL_SELECTOR_EXTRACTED = {
    "MultiTierFeatureExtractor", "ToolConfidenceCalibrator",
    "ValueOfInformationGate", "DistributionMonitor",
    "ToolSelectionBandit", "SelectionMode", "SelectionRequest",
    "SelectionResult", "CausalToolWrapper", "AnalogicalToolWrapper",
    "MultimodalToolWrapper", "CryptographicToolWrapper",
    "PhilosophicalToolWrapper",
}

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestWorldModelDedup:
    def test_no_extracted_classes_in_god_file(self):
        god_file = REPO_ROOT / "src/vulcan/world_model/world_model_core.py"
        if not god_file.exists():
            pytest.skip("world_model_core.py not found")
        defined = set(_get_class_names(god_file))
        duplicates = defined & WORLD_MODEL_EXTRACTED
        assert not duplicates, (
            f"world_model_core.py still defines extracted classes: {duplicates}"
        )


class TestAgentPoolDedup:
    def test_no_extracted_classes_in_god_file(self):
        god_file = REPO_ROOT / "src/vulcan/orchestrator/agent_pool.py"
        if not god_file.exists():
            pytest.skip("agent_pool.py not found")
        defined = set(_get_class_names(god_file))
        duplicates = defined & AGENT_POOL_EXTRACTED
        assert not duplicates, (
            f"agent_pool.py still defines extracted classes: {duplicates}"
        )


class TestToolSelectorDedup:
    def test_no_extracted_classes_in_god_file(self):
        god_file = REPO_ROOT / "src/vulcan/reasoning/selection/tool_selector.py"
        if not god_file.exists():
            pytest.skip("tool_selector.py not found")
        defined = set(_get_class_names(god_file))
        duplicates = defined & TOOL_SELECTOR_EXTRACTED
        assert not duplicates, (
            f"tool_selector.py still defines extracted classes: {duplicates}"
        )
