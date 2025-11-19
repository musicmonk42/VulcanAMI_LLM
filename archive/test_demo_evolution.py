# test_demo_evolution.py

"""
Comprehensive test suite for demo_evolution.py
Targets 85%+ code coverage with unit and integration tests.

Run with:
    pytest test_demo_evolution.py -v --cov=demo_evolution --cov-report=html --cov-report=term-missing
"""

import pytest
import asyncio
import json
import tempfile
import os
import sys
import logging 
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock, mock_open, call
from datetime import datetime
from dataclasses import asdict

# Add both parent directory and src directory to path for imports
test_dir = Path(__file__).parent
project_root = test_dir.parent
src_dir = project_root / "src"
demo_dir = project_root / "demo"

# Add paths in order of priority
for path in [str(project_root), str(demo_dir), str(src_dir), str(test_dir)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from demo_evolution import (
    EvolutionConfig,
    CodeSafetyValidator,
    EvolutionDemo,
    setup_logging,
    demo_evolution_cycle,
    main
)


# FIX: Isolate the logging environment completely for stability
@pytest.fixture(autouse=True)
def cleanup_logging_handlers():
    """Fixture to ensure the root logger is clean before and after tests."""
    root_logger = logging.getLogger()
    # Before test: clean all handlers
    handlers_to_restore = []
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handlers_to_restore.append(handler)
    
    yield
    
    # After test: restore original handlers and clean up any added handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        if hasattr(handler, 'close'):
            handler.close()

    for handler in handlers_to_restore:
        root_logger.addHandler(handler)


# Test Fixtures
@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def basic_config(temp_dir):
    """Basic evolution configuration."""
    return EvolutionConfig(
        proposal_dir=temp_dir / "proposals",
        output_dir=temp_dir / "output",
        grammar_version="3.4.0",
        timeout_seconds=30,
        skip_validation=False,
        enable_code_safety=True,
        max_code_size=10000,
        verbose=False
    )


@pytest.fixture
def unsafe_config(temp_dir):
    """Configuration with safety disabled."""
    return EvolutionConfig(
        proposal_dir=temp_dir / "proposals",
        output_dir=temp_dir / "output",
        enable_code_safety=False,
        verbose=False
    )


@pytest.fixture
def mock_runtime():
    """Mock UnifiedRuntime."""
    runtime = MagicMock()
    runtime._validate_graph = MagicMock(return_value=True)
    runtime.execute_graph = AsyncMock(return_value={"output": "def execute(x): return x + 10"})
    runtime.learn_node_type = MagicMock()
    runtime.save_node_executors = MagicMock()
    runtime.close = AsyncMock()
    # Ensure save_node_executors returns something awaitable if it were async
    runtime.save_node_executors.return_value = None 
    return runtime


@pytest.fixture
def sample_proposal():
    """Sample valid graph proposal."""
    return {
        "grammar_version": "3.4.0",
        "id": "test_proposal",
        "type": "Graph",
        "nodes": [
            {"id": "n1", "type": "InputNode", "value": "test"},
            {"id": "n2", "type": "OutputNode"}
        ],
        "edges": [
            {"id": "e1", "from": "n1", "to": "n2", "type": "data"}
        ]
    }


@pytest.fixture
def logger(cleanup_logging_handlers): # Inject the cleanup fixture
    """Test logger."""
    return setup_logging(verbose=False)


# Test EvolutionConfig
class TestEvolutionConfig:
    def test_config_defaults(self, temp_dir):
        """Test default configuration values."""
        config = EvolutionConfig(
            proposal_dir=temp_dir,
            output_dir=temp_dir
        )
        assert config.grammar_version == "3.4.0"
        assert config.timeout_seconds == 30
        assert config.skip_validation is False
        assert config.enable_code_safety is True
        assert config.max_code_size == 10000

    def test_config_custom_values(self, temp_dir):
        """Test custom configuration values."""
        config = EvolutionConfig(
            proposal_dir=temp_dir,
            output_dir=temp_dir,
            grammar_version="3.5.0",
            timeout_seconds=60,
            skip_validation=True,
            enable_code_safety=False,
            max_code_size=5000,
            verbose=True
        )
        assert config.grammar_version == "3.5.0"
        assert config.timeout_seconds == 60
        assert config.skip_validation is True
        assert config.enable_code_safety is False
        assert config.max_code_size == 5000
        assert config.verbose is True

    def test_config_serialization(self, basic_config):
        """Test configuration can be serialized."""
        config_dict = basic_config.to_json_dict() # Use the method
        assert isinstance(config_dict, dict)
        assert config_dict["grammar_version"] == "3.4.0"
        # Check that paths are strings
        assert isinstance(config_dict["proposal_dir"], str)
        assert isinstance(config_dict["output_dir"], str)


# Test CodeSafetyValidator
class TestCodeSafetyValidator:
    def test_validator_initialization(self, logger):
        """Test validator initialization."""
        validator = CodeSafetyValidator(logger)
        assert validator.logger == logger
        assert len(validator.BLOCKED_IMPORTS) > 0
        assert len(validator.BLOCKED_BUILTINS) > 0
        assert len(validator.BLOCKED_ATTRIBUTES) > 0

    def test_safe_code_passes(self, logger):
        """Test that safe code passes validation."""
        validator = CodeSafetyValidator(logger)
        safe_code = """
def execute(x):
    result = x + 10
    return result
"""
        is_safe, error = validator.validate(safe_code)
        assert is_safe is True
        assert error is None

    def test_empty_code_fails(self, logger):
        """Test that empty code fails validation."""
        validator = CodeSafetyValidator(logger)
        is_safe, error = validator.validate("")
        assert is_safe is False
        assert "empty" in error.lower()

    def test_non_string_code_fails(self, logger):
        """Test that non-string code fails validation."""
        validator = CodeSafetyValidator(logger)
        is_safe, error = validator.validate(123)
        assert is_safe is False
        assert "not a string" in error.lower()

    def test_oversized_code_fails(self, logger):
        """Test that oversized code fails validation."""
        validator = CodeSafetyValidator(logger)
        large_code = "x = 1\n" * 10000
        is_safe, error = validator.validate(large_code, max_size=1000)
        assert is_safe is False
        assert "exceeds maximum size" in error.lower()

    def test_blocked_import_os(self, logger):
        """Test that 'os' import is blocked."""
        validator = CodeSafetyValidator(logger)
        code = "import os\nos.system('ls')"
        is_safe, error = validator.validate(code)
        assert is_safe is False
        assert "os" in error.lower()

    def test_blocked_import_subprocess(self, logger):
        """Test that 'subprocess' import is blocked."""
        validator = CodeSafetyValidator(logger)
        code = "import subprocess\nsubprocess.call(['ls'])"
        is_safe, error = validator.validate(code)
        assert is_safe is False
        assert "subprocess" in error.lower()

    def test_blocked_from_import(self, logger):
        """Test that 'from os import' is blocked."""
        validator = CodeSafetyValidator(logger)
        code = "from os import system\nsystem('ls')"
        is_safe, error = validator.validate(code)
        assert is_safe is False

    def test_blocked_builtin_eval(self, logger):
        """Test that eval() is blocked."""
        validator = CodeSafetyValidator(logger)
        code = "result = eval('1 + 1')"
        is_safe, error = validator.validate(code)
        assert is_safe is False
        assert "eval" in error.lower()

    def test_blocked_builtin_exec(self, logger):
        """Test that exec() is blocked."""
        validator = CodeSafetyValidator(logger)
        code = "exec('print(1)')"
        is_safe, error = validator.validate(code)
        assert is_safe is False
        assert "exec" in error.lower()

    def test_blocked_builtin_compile(self, logger):
        """Test that compile() is blocked."""
        validator = CodeSafetyValidator(logger)
        code = "compile('1+1', '<string>', 'eval')"
        is_safe, error = validator.validate(code)
        assert is_safe is False

    def test_blocked_attribute_globals(self, logger):
        """Test that __globals__ access is blocked."""
        validator = CodeSafetyValidator(logger)
        code = "def f(): pass\nx = f.__globals__" # Example usage
        is_safe, error = validator.validate(code)
        assert is_safe is False
        assert "__globals__" in error.lower()

    def test_blocked_attribute_code(self, logger):
        """Test that __code__ access is blocked."""
        validator = CodeSafetyValidator(logger)
        code = "def f(): pass\nx = f.__code__" # Example usage
        is_safe, error = validator.validate(code)
        assert is_safe is False
        assert "__code__" in error.lower()

    def test_syntax_error_detection(self, logger):
        """Test that syntax errors are detected."""
        validator = CodeSafetyValidator(logger)
        code = "def broken(\npass"
        is_safe, error = validator.validate(code)
        assert is_safe is False
        assert "syntax error" in error.lower()

    def test_multiple_imports_blocked(self, logger):
        """Test that multiple dangerous imports are blocked."""
        validator = CodeSafetyValidator(logger)
        codes = [
            "import sys",
            "import shutil",
            "import pathlib",
            "import socket",
            "import pickle",
            "import ctypes"
        ]
        for code in codes:
            is_safe, error = validator.validate(code)
            assert is_safe is False, f"Should block: {code}"


# Test EvolutionDemo Initialization
class TestEvolutionDemoInit:
    @pytest.mark.asyncio
    async def test_demo_initialization(self, basic_config, logger):
        """Test demo initialization."""
        # Patch UnifiedRuntime just during instantiation of EvolutionDemo
        with patch('demo_evolution.UnifiedRuntime', MagicMock()) as mock_runtime_class:
            demo = EvolutionDemo(basic_config, logger)
            assert demo.config == basic_config
            assert demo.logger == logger
            assert demo.runtime is None
            assert isinstance(demo.safety_validator, CodeSafetyValidator)

    @pytest.mark.asyncio
    async def test_output_directory_created(self, basic_config, logger):
        """Test that output directory is created."""
        # Ensure dir does not exist before test
        import shutil
        if basic_config.output_dir.exists():
             # Use shutil for potentially non-empty dirs during cleanup phase
             shutil.rmtree(basic_config.output_dir, ignore_errors=True)
        # Patch UnifiedRuntime just during instantiation of EvolutionDemo
        with patch('demo_evolution.UnifiedRuntime', MagicMock()):
            demo = EvolutionDemo(basic_config, logger)
            assert basic_config.output_dir.exists()
            assert basic_config.output_dir.is_dir()

    @pytest.mark.asyncio
    async def test_context_manager_entry(self, basic_config, logger, mock_runtime):
        """Test async context manager entry."""
        with patch('demo_evolution.UnifiedRuntime', return_value=mock_runtime) as mock_runtime_class:
            demo = EvolutionDemo(basic_config, logger)
            async with demo as d:
                assert d == demo
                assert d.runtime == mock_runtime
                mock_runtime_class.assert_called_once() # Ensure runtime was instantiated

    @pytest.mark.asyncio
    async def test_context_manager_entry_no_runtime(self, basic_config, logger):
        """Test context manager entry fails if runtime unavailable."""
        with patch('demo_evolution.UnifiedRuntime', None): # Simulate unavailable runtime
            demo = EvolutionDemo(basic_config, logger)
            with pytest.raises(ImportError, match="UnifiedRuntime is not available"):
                async with demo:
                    pass # pragma: no cover

    @pytest.mark.asyncio
    async def test_context_manager_exit(self, basic_config, logger, mock_runtime):
        """Test async context manager exit."""
        with patch('demo_evolution.UnifiedRuntime', return_value=mock_runtime):
            demo = EvolutionDemo(basic_config, logger)
            async with demo:
                pass # Body of context manager

            # Verify cleanup was called
            mock_runtime.save_node_executors.assert_called_once()
            mock_runtime.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_exit_save_error(self, basic_config, logger, mock_runtime):
        """Test context manager exit handles save error."""
        mock_runtime.save_node_executors.side_effect = Exception("Save failed")
        with patch('demo_evolution.UnifiedRuntime', return_value=mock_runtime):
            demo = EvolutionDemo(basic_config, logger)
            # Should not raise, just log error
            async with demo:
                pass
            mock_runtime.close.assert_called_once() # Close should still be called

    @pytest.mark.asyncio
    async def test_context_manager_exit_close_error(self, basic_config, logger, mock_runtime):
        """Test context manager exit handles close error."""
        mock_runtime.close.side_effect = Exception("Close failed")
        with patch('demo_evolution.UnifiedRuntime', return_value=mock_runtime):
            demo = EvolutionDemo(basic_config, logger)
            # Should not raise, just log error
            async with demo:
                pass
            mock_runtime.save_node_executors.assert_called_once() # Save should still be called


# Test File Validation
class TestFileValidation:
    def test_validate_existing_file(self, basic_config, logger, temp_dir):
        """Test validation of existing file."""
        # Patch UnifiedRuntime just during instantiation of EvolutionDemo
        with patch('demo_evolution.UnifiedRuntime', MagicMock()):
            demo = EvolutionDemo(basic_config, logger)

        # Create test file
        test_file = temp_dir / "test.json"
        test_file.write_text("{}")

        # Should not raise
        demo._validate_file_path(test_file, "Test file")

    def test_validate_missing_file(self, basic_config, logger, temp_dir):
        """Test validation of missing file."""
        # Patch UnifiedRuntime just during instantiation of EvolutionDemo
        with patch('demo_evolution.UnifiedRuntime', MagicMock()):
            demo = EvolutionDemo(basic_config, logger)

        test_file = temp_dir / "missing.json"

        with pytest.raises(FileNotFoundError):
            demo._validate_file_path(test_file, "Test file")

    def test_validate_directory_not_file(self, basic_config, logger, temp_dir):
        """Test validation rejects directory."""
        # Patch UnifiedRuntime just during instantiation of EvolutionDemo
        with patch('demo_evolution.UnifiedRuntime', MagicMock()):
            demo = EvolutionDemo(basic_config, logger)

        with pytest.raises(ValueError, match="not a file"):
            demo._validate_file_path(temp_dir, "Test file")

    def test_validate_warns_non_json_extension(self, basic_config, logger, temp_dir, caplog):
        """Test warning for non-JSON extension."""
        # Patch UnifiedRuntime just during instantiation of EvolutionDemo
        with patch('demo_evolution.UnifiedRuntime', MagicMock()):
            demo = EvolutionDemo(basic_config, logger)

        test_file = temp_dir / "test.txt"
        test_file.write_text("{}")

        demo._validate_file_path(test_file, "Test file")
        # Should log warning but not fail
        assert "doesn't have .json extension" in caplog.text


# Test JSON Loading
class TestJSONLoading:
    def test_load_valid_json(self, basic_config, logger, temp_dir, sample_proposal):
        """Test loading valid JSON file."""
        # Patch UnifiedRuntime just during instantiation of EvolutionDemo
        with patch('demo_evolution.UnifiedRuntime', MagicMock()):
            demo = EvolutionDemo(basic_config, logger)

        json_file = temp_dir / "test.json"
        with open(json_file, 'w') as f:
            json.dump(sample_proposal, f)

        data = demo._load_json_file(json_file, "Test file")
        assert data == sample_proposal

    def test_load_invalid_json(self, basic_config, logger, temp_dir):
        """Test loading invalid JSON file."""
        # Patch UnifiedRuntime just during instantiation of EvolutionDemo
        with patch('demo_evolution.UnifiedRuntime', MagicMock()):
            demo = EvolutionDemo(basic_config, logger)

        json_file = temp_dir / "invalid.json"
        json_file.write_text("not valid json {")

        with pytest.raises(ValueError, match="Invalid JSON"):
            demo._load_json_file(json_file, "Test file")

    def test_load_missing_file(self, basic_config, logger, temp_dir):
        """Test loading missing file."""
        # Patch UnifiedRuntime just during instantiation of EvolutionDemo
        with patch('demo_evolution.UnifiedRuntime', MagicMock()):
            demo = EvolutionDemo(basic_config, logger)

        json_file = temp_dir / "missing.json"

        with pytest.raises(FileNotFoundError):
            demo._load_json_file(json_file, "Test file")

    def test_load_generic_error(self, basic_config, logger, temp_dir):
        """Test handling generic error during file load."""
        # Patch UnifiedRuntime just during instantiation of EvolutionDemo
        with patch('demo_evolution.UnifiedRuntime', MagicMock()):
            demo = EvolutionDemo(basic_config, logger)
        json_file = temp_dir / "test.json"
        json_file.write_text("{}")

        with patch("builtins.open", side_effect=OSError("Read error")):
            with pytest.raises(OSError):
                 demo._load_json_file(json_file, "Test file")


# Test Graph Structure Validation
class TestGraphValidation:
    def test_validate_complete_graph(self, basic_config, logger, sample_proposal):
        """Test validation of complete graph."""
        # Patch UnifiedRuntime just during instantiation of EvolutionDemo
        with patch('demo_evolution.UnifiedRuntime', MagicMock()):
            demo = EvolutionDemo(basic_config, logger)

        # Should not raise
        demo._validate_graph_structure(sample_proposal, "Test graph")

    def test_validate_missing_grammar_version(self, basic_config, logger):
        """Test validation fails for missing grammar_version."""
        # Patch UnifiedRuntime just during instantiation of EvolutionDemo
        with patch('demo_evolution.UnifiedRuntime', MagicMock()):
            demo = EvolutionDemo(basic_config, logger)

        incomplete = {"id": "test", "type": "Graph", "nodes": [], "edges": []}

        with pytest.raises(ValueError, match="missing required field: grammar_version"):
            demo._validate_graph_structure(incomplete, "Test graph")

    def test_validate_missing_nodes(self, basic_config, logger):
        """Test validation fails for missing nodes."""
        # Patch UnifiedRuntime just during instantiation of EvolutionDemo
        with patch('demo_evolution.UnifiedRuntime', MagicMock()):
            demo = EvolutionDemo(basic_config, logger)

        incomplete = {"grammar_version": "3.4.0", "id": "test", "type": "Graph", "edges": []}

        with pytest.raises(ValueError, match="missing required field: nodes"):
            demo._validate_graph_structure(incomplete, "Test graph")

    def test_validate_nodes_not_list(self, basic_config, logger):
        """Test validation fails when nodes is not a list."""
        # Patch UnifiedRuntime just during instantiation of EvolutionDemo
        with patch('demo_evolution.UnifiedRuntime', MagicMock()):
            demo = EvolutionDemo(basic_config, logger)

        invalid = {
            "grammar_version": "3.4.0",
            "id": "test",
            "type": "Graph",
            "nodes": "not a list",
            "edges": []
        }

        with pytest.raises(ValueError, match="'nodes' must be a list"):
            demo._validate_graph_structure(invalid, "Test graph")

    def test_validate_edges_not_list(self, basic_config, logger):
        """Test validation fails when edges is not a list."""
        # Patch UnifiedRuntime just during instantiation of EvolutionDemo
        with patch('demo_evolution.UnifiedRuntime', MagicMock()):
            demo = EvolutionDemo(basic_config, logger)

        invalid = {
            "grammar_version": "3.4.0",
            "id": "test",
            "type": "Graph",
            "nodes": [],
            "edges": "not a list"
        }

        with pytest.raises(ValueError, match="'edges' must be a list"):
            demo._validate_graph_structure(invalid, "Test graph")

    def test_validate_version_mismatch_warning(self, basic_config, logger, caplog):
        """Test warning for grammar version mismatch."""
        # Patch UnifiedRuntime just during instantiation of EvolutionDemo
        with patch('demo_evolution.UnifiedRuntime', MagicMock()):
            demo = EvolutionDemo(basic_config, logger)

        mismatched = {
            "grammar_version": "3.2.0",
            "id": "test",
            "type": "Graph",
            "nodes": [],
            "edges": []
        }

        demo._validate_graph_structure(mismatched, "Test graph")
        # Should log warning
        assert "grammar version mismatch" in caplog.text


# Test Proposal Loading
class TestProposalLoading:
    @pytest.mark.asyncio
    async def test_load_proposals_success(self, basic_config, logger, sample_proposal, mock_runtime):
        """Test successful proposal loading."""
        demo = EvolutionDemo(basic_config, logger)
        demo.runtime = mock_runtime

        # Create proposal files
        basic_config.proposal_dir.mkdir(parents=True, exist_ok=True)

        causal_file = basic_config.proposal_dir / "gemini_evolution_proposal.json"
        ai_file = basic_config.proposal_dir / "grok_generative_ai_node_proposal.json"

        with open(causal_file, 'w') as f:
            json.dump(sample_proposal, f)
        with open(ai_file, 'w') as f:
            json.dump(sample_proposal, f)

        causal, ai = await demo.load_and_validate_proposals()

        assert causal == sample_proposal
        assert ai == sample_proposal
        assert mock_runtime._validate_graph.call_count == 2

    @pytest.mark.asyncio
    async def test_load_proposals_skip_validation(self, basic_config, logger, sample_proposal, mock_runtime):
        """Test loading proposals with validation skipped."""
        basic_config.skip_validation = True
        demo = EvolutionDemo(basic_config, logger)
        demo.runtime = mock_runtime

        # Create proposal files
        basic_config.proposal_dir.mkdir(parents=True, exist_ok=True)

        causal_file = basic_config.proposal_dir / "gemini_evolution_proposal.json"
        ai_file = basic_config.proposal_dir / "grok_generative_ai_node_proposal.json"

        with open(causal_file, 'w') as f:
            json.dump(sample_proposal, f)
        with open(ai_file, 'w') as f:
            json.dump(sample_proposal, f)

        await demo.load_and_validate_proposals()

        # Validation should not be called
        mock_runtime._validate_graph.assert_not_called()

    @pytest.mark.asyncio
    async def test_load_proposals_validation_failure(self, basic_config, logger, sample_proposal, mock_runtime):
        """Test proposal loading when validation fails."""
        demo = EvolutionDemo(basic_config, logger)
        demo.runtime = mock_runtime
        mock_runtime._validate_graph.return_value = False

        # Create proposal files
        basic_config.proposal_dir.mkdir(parents=True, exist_ok=True)

        causal_file = basic_config.proposal_dir / "gemini_evolution_proposal.json"
        ai_file = basic_config.proposal_dir / "grok_generative_ai_node_proposal.json"

        with open(causal_file, 'w') as f:
            json.dump(sample_proposal, f)
        with open(ai_file, 'w') as f:
            json.dump(sample_proposal, f)

        with pytest.raises(ValueError, match="failed runtime validation"):
            await demo.load_and_validate_proposals()

    @pytest.mark.asyncio
    async def test_load_proposals_runtime_validation_exception(self, basic_config, logger, sample_proposal, mock_runtime):
        """Test proposal loading when runtime validation raises exception."""
        demo = EvolutionDemo(basic_config, logger)
        demo.runtime = mock_runtime
        mock_runtime._validate_graph.side_effect = Exception("Runtime validation error")

        # Create proposal files
        basic_config.proposal_dir.mkdir(parents=True, exist_ok=True)

        causal_file = basic_config.proposal_dir / "gemini_evolution_proposal.json"
        ai_file = basic_config.proposal_dir / "grok_generative_ai_node_proposal.json"

        with open(causal_file, 'w') as f:
            json.dump(sample_proposal, f)
        with open(ai_file, 'w') as f:
            json.dump(sample_proposal, f)

        with pytest.raises(Exception, match="Runtime validation error"):
            await demo.load_and_validate_proposals()


# Test Node Generation
class TestNodeGeneration:
    @pytest.mark.asyncio
    async def test_generate_node_success(self, basic_config, logger, mock_runtime):
        """Test successful node generation."""
        demo = EvolutionDemo(basic_config, logger)
        demo.runtime = mock_runtime

        code = await demo.generate_and_learn_node()

        assert "def execute" in code
        mock_runtime.execute_graph.assert_called_once()
        mock_runtime.learn_node_type.assert_called_once_with("AdderNode", code)
        # Check if output files were created
        assert (basic_config.output_dir / "test_generation_graph.json").exists()
        assert (basic_config.output_dir / "generated_adder_node.py").exists()

    @pytest.mark.asyncio
    async def test_generate_node_timeout(self, basic_config, logger, mock_runtime):
        """Test node generation timeout."""
        basic_config.timeout_seconds = 1
        demo = EvolutionDemo(basic_config, logger)
        demo.runtime = mock_runtime

        # Make execute_graph hang
        async def slow_execute(*args, **kwargs):
            await asyncio.sleep(10)
            return {"output": "code"} # pragma: no cover

        mock_runtime.execute_graph.side_effect = slow_execute

        with pytest.raises(TimeoutError, match="timed out"):
            await demo.generate_and_learn_node()

    @pytest.mark.asyncio
    async def test_generate_node_execution_exception(self, basic_config, logger, mock_runtime):
        """Test node generation fails on execution exception."""
        demo = EvolutionDemo(basic_config, logger)
        demo.runtime = mock_runtime
        mock_runtime.execute_graph.side_effect = Exception("Execution error")

        with pytest.raises(Exception, match="Execution error"):
            await demo.generate_and_learn_node()

    @pytest.mark.asyncio
    async def test_generate_node_no_output(self, basic_config, logger, mock_runtime):
        """Test node generation with no output."""
        demo = EvolutionDemo(basic_config, logger)
        demo.runtime = mock_runtime
        mock_runtime.execute_graph.return_value = {}

        with pytest.raises(ValueError, match="No code generated"):
            await demo.generate_and_learn_node()

    @pytest.mark.asyncio
    async def test_generate_node_output_not_string(self, basic_config, logger, mock_runtime):
        """Test node generation where output is not a string."""
        demo = EvolutionDemo(basic_config, logger)
        demo.runtime = mock_runtime
        mock_runtime.execute_graph.return_value = {"output": 123}

        with pytest.raises(ValueError, match="Generated code is not a string"):
            await demo.generate_and_learn_node()

    @pytest.mark.asyncio
    async def test_generate_node_unsafe_code(self, basic_config, logger, mock_runtime):
        """Test node generation with unsafe code."""
        demo = EvolutionDemo(basic_config, logger)
        demo.runtime = mock_runtime
        mock_runtime.execute_graph.return_value = {
            "output": "import os\nos.system('rm -rf /')"
        }

        with pytest.raises(ValueError, match="failed safety validation"):
            await demo.generate_and_learn_node()

    @pytest.mark.asyncio
    async def test_generate_node_safety_disabled(self, unsafe_config, logger, mock_runtime):
        """Test node generation with safety disabled."""
        demo = EvolutionDemo(unsafe_config, logger)
        demo.runtime = mock_runtime
        mock_runtime.execute_graph.return_value = {
            "output": "import os\nos.system('ls')"
        }

        # Should succeed even with dangerous code
        code = await demo.generate_and_learn_node()
        assert "import os" in code
        mock_runtime.learn_node_type.assert_called_once_with("AdderNode", code)

    @pytest.mark.asyncio
    async def test_generate_node_learn_error(self, basic_config, logger, mock_runtime):
        """Test node generation when learning fails."""
        demo = EvolutionDemo(basic_config, logger)
        demo.runtime = mock_runtime
        mock_runtime.learn_node_type.side_effect = Exception("Learning failed")

        with pytest.raises(Exception, match="Learning failed"):
            await demo.generate_and_learn_node()


# Test Learned Node Testing
class TestLearnedNodeTesting:
    @pytest.mark.asyncio
    async def test_test_node_success(self, basic_config, logger, mock_runtime):
        """Test successful learned node testing."""
        demo = EvolutionDemo(basic_config, logger)
        demo.runtime = mock_runtime
        mock_runtime.execute_graph.return_value = {"output": 20}

        result = await demo.test_learned_node()

        assert result["output"] == 20
        mock_runtime.execute_graph.assert_called_once()
        # Check output files
        assert (basic_config.output_dir / "test_worker_graph.json").exists()
        assert (basic_config.output_dir / "test_result.json").exists()

        # Check result file content
        result_path = basic_config.output_dir / "test_result.json"
        with open(result_path, 'r') as f:
            result_data = json.load(f)
        assert result_data["test_passed"] is True
        assert result_data["actual_output"] == 20

    @pytest.mark.asyncio
    async def test_test_node_wrong_output(self, basic_config, logger, mock_runtime, caplog):
        """Test learned node with unexpected output."""
        demo = EvolutionDemo(basic_config, logger)
        demo.runtime = mock_runtime
        mock_runtime.execute_graph.return_value = {"output": 15}  # Expected 20

        result = await demo.test_learned_node()

        # Should log warning
        assert "Test result unexpected" in caplog.text
        assert result["output"] == 15

        # Check result file content
        result_path = basic_config.output_dir / "test_result.json"
        with open(result_path, 'r') as f:
            result_data = json.load(f)
        assert result_data["test_passed"] is False
        assert result_data["actual_output"] == 15

    @pytest.mark.asyncio
    async def test_test_node_no_output(self, basic_config, logger, mock_runtime):
        """Test learned node with no output."""
        demo = EvolutionDemo(basic_config, logger)
        demo.runtime = mock_runtime
        mock_runtime.execute_graph.return_value = {}

        with pytest.raises(ValueError, match="missing 'output' field"):
            await demo.test_learned_node()

    @pytest.mark.asyncio
    async def test_test_node_timeout(self, basic_config, logger, mock_runtime):
        """Test learned node testing timeout."""
        basic_config.timeout_seconds = 1
        demo = EvolutionDemo(basic_config, logger)
        demo.runtime = mock_runtime

        async def slow_execute(*args, **kwargs):
            await asyncio.sleep(10)
            return {"output": 20} # pragma: no cover

        mock_runtime.execute_graph.side_effect = slow_execute

        with pytest.raises(TimeoutError, match="timed out"):
            await demo.test_learned_node()

    @pytest.mark.asyncio
    async def test_test_node_execution_exception(self, basic_config, logger, mock_runtime):
        """Test learned node testing fails on execution exception."""
        demo = EvolutionDemo(basic_config, logger)
        demo.runtime = mock_runtime
        mock_runtime.execute_graph.side_effect = Exception("Worker execution error")

        with pytest.raises(Exception, match="Worker execution error"):
            await demo.test_learned_node()


# Test Complete Cycle
class TestCompleteCycle:
    @pytest.mark.asyncio
    async def test_complete_cycle_success(self, basic_config, logger, sample_proposal, mock_runtime):
        """Test complete evolution cycle."""
        demo = EvolutionDemo(basic_config, logger)
        demo.runtime = mock_runtime

        # Setup proposal files
        basic_config.proposal_dir.mkdir(parents=True, exist_ok=True)

        causal_file = basic_config.proposal_dir / "gemini_evolution_proposal.json"
        ai_file = basic_config.proposal_dir / "grok_generative_ai_node_proposal.json"

        with open(causal_file, 'w') as f:
            json.dump(sample_proposal, f)
        with open(ai_file, 'w') as f:
            json.dump(sample_proposal, f)

        mock_runtime.execute_graph.side_effect = [
            {"output": "def execute(x): return x + 10"},  # Generation
            {"output": 20}  # Testing
        ]

        results = await demo.run_complete_cycle()

        assert results["success"] is True
        assert results["error"] is None
        assert "load_proposals" in results["steps"]
        assert results["steps"]["load_proposals"]["status"] == "success"
        assert "generate_node" in results["steps"]
        assert results["steps"]["generate_node"]["status"] == "success"
        assert "test_node" in results["steps"]
        assert results["steps"]["test_node"]["status"] == "success"
        assert (basic_config.output_dir / "evolution_cycle_results.json").exists()

    @pytest.mark.asyncio
    async def test_complete_cycle_proposal_load_failure(self, basic_config, logger, mock_runtime):
        """Test cycle when proposal loading fails."""
        demo = EvolutionDemo(basic_config, logger)
        demo.runtime = mock_runtime

        # Don't create proposal files

        with pytest.raises(FileNotFoundError):
            await demo.run_complete_cycle()

        # Check results file
        results_path = basic_config.output_dir / "evolution_cycle_results.json"
        assert results_path.exists()
        with open(results_path, 'r') as f:
            results = json.load(f)
        assert results["success"] is False
        assert "not found" in results["error"]

    @pytest.mark.asyncio
    async def test_complete_cycle_generation_failure(self, basic_config, logger, sample_proposal, mock_runtime):
        """Test cycle when generation fails."""
        demo = EvolutionDemo(basic_config, logger)
        demo.runtime = mock_runtime

        # Setup proposal files
        basic_config.proposal_dir.mkdir(parents=True, exist_ok=True)

        causal_file = basic_config.proposal_dir / "gemini_evolution_proposal.json"
        ai_file = basic_config.proposal_dir / "grok_generative_ai_node_proposal.json"

        with open(causal_file, 'w') as f:
            json.dump(sample_proposal, f)
        with open(ai_file, 'w') as f:
            json.dump(sample_proposal, f)

        mock_runtime.execute_graph.side_effect = Exception("Generation failed")

        with pytest.raises(Exception, match="Generation failed"):
            await demo.run_complete_cycle()

        # Check results file
        results_path = basic_config.output_dir / "evolution_cycle_results.json"
        assert results_path.exists()
        with open(results_path, 'r') as f:
            results = json.load(f)
        assert results["success"] is False
        assert "Generation failed" in results["error"]
        assert "load_proposals" in results["steps"] # Should pass this step
        assert "generate_node" not in results["steps"] # Should fail here


    @pytest.mark.asyncio
    async def test_complete_cycle_test_failure(self, basic_config, logger, sample_proposal, mock_runtime):
        """Test cycle when testing fails."""
        demo = EvolutionDemo(basic_config, logger)
        demo.runtime = mock_runtime

        # Setup proposal files
        basic_config.proposal_dir.mkdir(parents=True, exist_ok=True)

        causal_file = basic_config.proposal_dir / "gemini_evolution_proposal.json"
        ai_file = basic_config.proposal_dir / "grok_generative_ai_node_proposal.json"

        with open(causal_file, 'w') as f:
            json.dump(sample_proposal, f)
        with open(ai_file, 'w') as f:
            json.dump(sample_proposal, f)

        mock_runtime.execute_graph.side_effect = [
            {"output": "def execute(x): return x + 10"},  # Generation
            Exception("Testing failed")  # Testing
        ]

        with pytest.raises(Exception, match="Testing failed"):
            await demo.run_complete_cycle()

        # Check results file
        results_path = basic_config.output_dir / "evolution_cycle_results.json"
        assert results_path.exists()
        with open(results_path, 'r') as f:
            results = json.load(f)
        assert results["success"] is False
        assert "Testing failed" in results["error"]
        assert "load_proposals" in results["steps"]
        assert "generate_node" in results["steps"]
        assert "test_node" not in results["steps"]


# Test Backward Compatibility
class TestBackwardCompatibility:
    @pytest.mark.asyncio
    async def test_demo_evolution_cycle_function(self):
        """Test backward compatible demo_evolution_cycle function."""
        with patch('demo_evolution.EvolutionDemo') as mock_demo_class, \
             patch('demo_evolution.setup_logging') as mock_logging:
            mock_demo = MagicMock() # Use MagicMock for easier async patching
            mock_demo.__aenter__ = AsyncMock(return_value=mock_demo)
            mock_demo.__aexit__ = AsyncMock(return_value=False)
            mock_demo.run_complete_cycle = AsyncMock(return_value={"success": True}) # Need to return dict
            mock_demo_class.return_value = mock_demo

            await demo_evolution_cycle()

            mock_logging.assert_called_once()
            mock_demo_class.assert_called_once()
            mock_demo.run_complete_cycle.assert_called_once()


# Test CLI Argument Parsing
class TestCLIArguments:
    
    # FIX: Mock asyncio.run to simulate success, avoiding coroutine confusion.
    @patch('demo_evolution.asyncio.run', return_value=None)
    @patch('sys.exit')
    def test_main_with_defaults(self, mock_exit, mock_run):
        """Test main with default arguments."""
        test_args = ['demo_evolution.py']

        with patch('sys.argv', test_args), \
             patch('demo_evolution.UNIFIED_RUNTIME_AVAILABLE', True), \
             patch('demo_evolution.EvolutionDemo'):

            main()

            mock_run.assert_called_once()
            # main() reaches the final sys.exit(0) after asyncio.run returns None
            mock_exit.assert_called_with(0)


    # FIX: Use a simple Mock to simulate success.
    @patch('demo_evolution.asyncio.run', return_value=None)
    @patch('sys.exit')
    def test_main_with_custom_args(self, mock_exit, mock_run):
        """Test main with custom arguments."""
        test_args = [
            'demo_evolution.py',
            '--proposal-dir', '/tmp/proposals',
            '--output-dir', '/tmp/output',
            '--grammar-version', '3.5.0',
            '--timeout', '60',
            '--skip-validation',
            '--disable-code-safety',
            '--max-code-size', '5000',
            '--verbose'
        ]

        with patch('sys.argv', test_args), \
             patch('demo_evolution.UNIFIED_RUNTIME_AVAILABLE', True), \
             patch('demo_evolution.setup_logging') as mock_setup_logging, \
             patch('demo_evolution.EvolutionConfig') as mock_config_class, \
             patch('demo_evolution.EvolutionDemo'):

            mock_config_instance = MagicMock()
            mock_config_class.return_value = mock_config_instance

            main()

            # Check config passed to EvolutionConfig constructor
            args_config, kwargs_config = mock_config_class.call_args
            assert kwargs_config['proposal_dir'] == Path('/tmp/proposals')
            assert kwargs_config['output_dir'] == Path('/tmp/output')
            assert kwargs_config['grammar_version'] == '3.5.0'
            assert kwargs_config['timeout_seconds'] == 60
            assert kwargs_config['skip_validation'] is True
            assert kwargs_config['enable_code_safety'] is False
            assert kwargs_config['max_code_size'] == 5000
            assert kwargs_config['verbose'] is True

            # Check verbose passed to setup_logging
            args_log, kwargs_log = mock_setup_logging.call_args
            assert kwargs_log['verbose'] is True

            mock_run.assert_called_once()
            mock_exit.assert_called_with(0) # Expect successful exit code


    # FIX: Mock asyncio.run to raise the exception synchronously
    @patch('sys.exit')
    def test_main_keyboard_interrupt(self, mock_exit):
        """Test main handles keyboard interrupt."""
        test_args = ['demo_evolution.py']

        # We raise KeyboardInterrupt inside the asyncio.run block
        with patch('sys.argv', test_args), \
             patch('demo_evolution.UNIFIED_RUNTIME_AVAILABLE', True), \
             patch('demo_evolution.asyncio.run', side_effect=KeyboardInterrupt) as mock_asyncio_run, \
             patch('demo_evolution.EvolutionDemo'):

            # The code relies on the SystemExit exception raised by main() to stop the process.
            # We must catch the SystemExit that is raised by sys.exit(130).
            with pytest.raises(SystemExit):
                main()

            # Assert asyncio.run was called
            mock_asyncio_run.assert_called_once()
            # Assert sys.exit was called with the specific code for KeyboardInterrupt
            mock_exit.assert_called_with(130)


    # FIX: Mock asyncio.run to raise the exception synchronously
    @patch('sys.exit')
    def test_main_unexpected_error(self, mock_exit):
        """Test main handles unexpected errors."""
        test_args = ['demo_evolution.py']

        # We raise a generic Exception inside the asyncio.run block
        with patch('sys.argv', test_args), \
             patch('demo_evolution.UNIFIED_RUNTIME_AVAILABLE', True), \
             patch('demo_evolution.asyncio.run', side_effect=Exception("Test error")) as mock_asyncio_run, \
             patch('demo_evolution.EvolutionDemo'):

            # We must catch the SystemExit that is raised by sys.exit(1).
            with pytest.raises(SystemExit):
                main()

            # Assert asyncio.run was called
            mock_asyncio_run.assert_called_once()
            # Assert sys.exit was called with the general error code
            mock_exit.assert_called_with(1)

    @patch('demo_evolution.asyncio.run', Mock()) # Simple Mock to ensure it's not called
    @patch('sys.exit')
    def test_main_no_runtime_available(self, mock_exit, mock_run):
        """Test main exits if UnifiedRuntime is not available."""
        test_args = ['demo_evolution.py']

        with patch('sys.argv', test_args), \
             patch('demo_evolution.UNIFIED_RUNTIME_AVAILABLE', False):

            # We must catch the SystemExit that is raised by sys.exit(1) before asyncio.run
            with pytest.raises(SystemExit):
                main()

            mock_run.assert_not_called() # Should exit before calling asyncio.run
            mock_exit.assert_called_with(1) # Expect failure exit code

    @patch('sys.exit')
    def test_main_system_exit_in_run(self, mock_exit):
        """Test main propagates SystemExit from run."""
        test_args = ['demo_evolution.py']

        # Simulate SystemExit being raised inside the run coroutine
        # This will be propagated by asyncio.run()
        with patch('sys.argv', test_args), \
             patch('demo_evolution.UNIFIED_RUNTIME_AVAILABLE', True), \
             patch('demo_evolution.asyncio.run', side_effect=SystemExit(5)):
             
             # The SystemExit exception should propagate out of main()
             with pytest.raises(SystemExit) as exc:
                 main()
             assert exc.value.code == 5
             mock_exit.assert_not_called() # sys.exit(5) is raised directly by the mocked asyncio.run
             


# Test Logging Setup
class TestLoggingSetup:
    def test_logging_setup_default(self, temp_dir):
        """Test logging setup with defaults."""
        # Change CWD to temp_dir to ensure log file is created there
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        root_logger = logging.getLogger()
        try:
            # Need to remove existing handlers properly
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)
                if hasattr(handler, 'close'):
                    handler.close()

            logger = setup_logging(verbose=False)
            assert logger.name == "DemoEvolution"
            # Check handlers (StreamHandler and FileHandler expected on root logger)
            root_handlers = logging.getLogger().handlers
            assert len(root_handlers) >= 1 # Check at least StreamHandler added
            assert any(isinstance(h, logging.FileHandler) for h in root_handlers)
            # Find the root logger's effective level
            assert logging.getLogger().getEffectiveLevel() == logging.INFO # Default level
            # Check log file creation
            assert (temp_dir / 'demo_evolution.log').exists()
        finally:
            os.chdir(original_cwd) # Change back CWD
            # Clean up root logger handlers again
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)
                if hasattr(handler, 'close'):
                    handler.close()


    def test_logging_setup_verbose(self, temp_dir):
        """Test logging setup in verbose mode."""
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        root_logger = logging.getLogger()
        try:
            # Ensure handlers are cleared before test
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)
                if hasattr(handler, 'close'):
                    handler.close()

            logger = setup_logging(verbose=True)
            assert logger.name == "DemoEvolution"
            root_handlers = logging.getLogger().handlers
            assert len(root_handlers) >= 1
            assert any(isinstance(h, logging.FileHandler) for h in root_handlers)
            assert logging.getLogger().getEffectiveLevel() == logging.DEBUG # Verbose level
            assert (temp_dir / 'demo_evolution.log').exists()
        finally:
            os.chdir(original_cwd)
            # Clean up root logger handlers again
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)
                if hasattr(handler, 'close'):
                    handler.close()


# Integration Tests
class TestIntegration:
    @pytest.mark.asyncio
    async def test_full_workflow_with_mocks(self, basic_config, logger, sample_proposal, mock_runtime):
        """Test complete workflow with mocked components."""
        with patch('demo_evolution.UnifiedRuntime', return_value=mock_runtime):
            # Setup files
            basic_config.proposal_dir.mkdir(parents=True, exist_ok=True)

            causal_file = basic_config.proposal_dir / "gemini_evolution_proposal.json"
            ai_file = basic_config.proposal_dir / "grok_generative_ai_node_proposal.json"

            with open(causal_file, 'w') as f:
                json.dump(sample_proposal, f)
            with open(ai_file, 'w') as f:
                json.dump(sample_proposal, f)

            # Setup mock responses
            mock_runtime.execute_graph.side_effect = [
                {"output": "def execute(x): return x + 10"}, # Generation
                {"output": 20}  # Testing
            ]

            async with EvolutionDemo(basic_config, logger) as demo:
                results = await demo.run_complete_cycle()

            assert results["success"] is True

            # Verify all steps completed
            assert all(
                step in results["steps"]
                for step in ["load_proposals", "generate_node", "test_node"]
            )

            # Verify output files created
            assert (basic_config.output_dir / "test_generation_graph.json").exists()
            assert (basic_config.output_dir / "generated_adder_node.py").exists()
            assert (basic_config.output_dir / "test_worker_graph.json").exists()
            assert (basic_config.output_dir / "test_result.json").exists()
            assert (basic_config.output_dir / "evolution_cycle_results.json").exists()

            # Verify calls
            mock_runtime._validate_graph.assert_called()
            assert mock_runtime.execute_graph.call_count == 2
            mock_runtime.learn_node_type.assert_called_once_with("AdderNode", "def execute(x): return x + 10")
            mock_runtime.save_node_executors.assert_called_once()
            mock_runtime.close.assert_called_once()