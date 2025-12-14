"""
Comprehensive test suite for stdio_policy.py
"""

import json
import os
import sys
import tempfile
import threading
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, Mock, mock_open, patch

import pytest

from stdio_policy import (
    StdIOConfig,
    StdIOHandle,
    _is_windows,
    _normalize_text,
    _should_disable_color,
    install,
    json_print,
    safe_print,
    self_test,
)


@pytest.fixture
def temp_jsonl_file():
    """Create temporary JSONL file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = f.name
    yield path
    # Cleanup
    Path(path).unlink(missing_ok=True)


class TestStdIOConfig:
    """Test StdIOConfig dataclass."""

    def test_initialization_defaults(self):
        """Test default configuration."""
        config = StdIOConfig()

        assert config.replace_builtins is True
        assert config.patch_ray is True
        assert config.patch_colorama is True
        assert config.normalize_newlines is True
        assert config.newline == "\n"
        assert config.flush is True

    def test_initialization_custom(self):
        """Test custom configuration."""
        config = StdIOConfig(
            replace_builtins=False,
            normalize_newlines=False,
            newline="\r\n",
            max_len=500,
        )

        assert config.replace_builtins is False
        assert config.normalize_newlines is False
        assert config.newline == "\r\n"
        assert config.max_len == 500

    def test_jsonl_path(self):
        """Test JSONL path configuration."""
        config = StdIOConfig(jsonl_path="test.jsonl")

        assert config.jsonl_path == "test.jsonl"


class TestUtilityFunctions:
    """Test utility functions."""

    def test_is_windows(self):
        """Test Windows detection."""
        result = _is_windows()

        # Result depends on platform
        assert isinstance(result, bool)
        assert result == (os.name == "nt")

    def test_should_disable_color_no_color_env(self):
        """Test color disabling with NO_COLOR env."""
        env = {"NO_COLOR": "1"}
        config = StdIOConfig()

        result = _should_disable_color(env, config)

        assert result is True

    def test_should_disable_color_pytest_env(self):
        """Test color disabling in pytest."""
        env = {"PYTEST_CURRENT_TEST": "test"}
        config = StdIOConfig()

        result = _should_disable_color(env, config)

        assert result is True

    def test_should_disable_color_forced_enable(self):
        """Test forced color enabling."""
        env = {"NO_COLOR": "1"}
        config = StdIOConfig(enable_color=True)

        result = _should_disable_color(env, config)

        assert result is False

    def test_normalize_text_newlines(self):
        """Test text normalization with newlines."""
        config = StdIOConfig(normalize_newlines=True, newline="\n")

        result = _normalize_text("line1\r\nline2\rline3", config)

        assert result == "line1\nline2\nline3"

    def test_normalize_text_custom_newline(self):
        """Test normalization with custom newline."""
        config = StdIOConfig(normalize_newlines=True, newline="\r\n")

        result = _normalize_text("line1\nline2", config)

        assert result == "line1\r\nline2"

    def test_normalize_text_truncation(self):
        """Test text truncation."""
        config = StdIOConfig(max_len=10)
        long_text = "x" * 20

        result = _normalize_text(long_text, config)

        assert len(result) <= 25  # 10 + truncation message
        assert "truncated" in result


class TestSafePrint:
    """Test safe_print function."""

    def test_safe_print_basic(self):
        """Test basic printing."""
        output = StringIO()

        safe_print("Hello", "World", file=output)

        assert "Hello World" in output.getvalue()

    def test_safe_print_custom_separator(self):
        """Test printing with custom separator."""
        output = StringIO()

        safe_print("A", "B", "C", sep="-", file=output)

        assert "A-B-C" in output.getvalue()

    def test_safe_print_custom_end(self):
        """Test printing with custom end."""
        output = StringIO()

        safe_print("Test", end="***", file=output)

        assert output.getvalue().endswith("***")

    def test_safe_print_to_file(self, temp_jsonl_file):
        """Test printing to file."""
        with open(temp_jsonl_file, "w", encoding="utf-8") as f:
            safe_print("Test output", file=f)

        with open(temp_jsonl_file, "r", encoding="utf-8") as f:
            content = f.read()

        assert "Test output" in content

    def test_safe_print_none_handling(self):
        """Test printing None values."""
        output = StringIO()

        safe_print("Before", None, "After", file=output)

        assert "Before  After" in output.getvalue()

    def test_safe_print_with_jsonl(self, temp_jsonl_file):
        """Test printing with JSONL audit."""
        output = StringIO()
        config = StdIOConfig(jsonl_path=temp_jsonl_file, replace_builtins=False)

        safe_print("Test", "Message", file=output, cfg=config, effect="Effect.IO.Test")

        # Check console output
        assert "Test Message" in output.getvalue()

        # Check JSONL file
        with open(temp_jsonl_file, "r", encoding="utf-8") as f:
            line = f.readline()
            entry = json.loads(line)

        assert entry["type"] == "io.print"
        assert entry["text"] == "Test Message"
        assert entry["effect"] == "Effect.IO.Test"


class TestJsonPrint:
    """Test json_print function."""

    def test_json_print_basic(self):
        """Test basic JSON printing."""
        output = StringIO()

        json_print(data={"key": "value"}, effect="Effect.IO.Test", file=output)

        parsed = json.loads(output.getvalue().strip())

        assert parsed["type"] == "io.json"
        assert parsed["data"] == {"key": "value"}
        assert parsed["effect"] == "Effect.IO.Test"

    def test_json_print_with_extra_fields(self):
        """Test JSON printing with extra fields."""
        output = StringIO()

        json_print(data="test", custom_field="custom_value", file=output)

        parsed = json.loads(output.getvalue().strip())

        assert parsed["custom_field"] == "custom_value"

    def test_json_print_includes_pid_tid(self):
        """Test that PID and TID are included."""
        output = StringIO()
        config = StdIOConfig(include_pid_tid=True, replace_builtins=False)

        json_print(data="test", cfg=config, file=output)

        parsed = json.loads(output.getvalue().strip())

        assert "pid" in parsed
        assert "thread" in parsed

    def test_json_print_includes_timestamp(self):
        """Test that timestamp is included."""
        output = StringIO()
        config = StdIOConfig(include_ts=True, replace_builtins=False)

        json_print(data="test", cfg=config, file=output)

        parsed = json.loads(output.getvalue().strip())

        assert "ts" in parsed
        assert isinstance(parsed["ts"], (int, float))


class TestStdIOHandle:
    """Test StdIOHandle class."""

    def test_handle_initialization(self):
        """Test handle initialization."""
        config = StdIOConfig()
        handle = StdIOHandle(cfg=config)

        assert handle.cfg == config
        assert handle.restored is False

    def test_handle_restore(self):
        """Test handle restore."""
        config = StdIOConfig()
        handle = StdIOHandle(cfg=config)

        handle.restore()

        assert handle.restored is True

    def test_handle_restore_idempotent(self):
        """Test that restore can be called multiple times."""
        config = StdIOConfig()
        handle = StdIOHandle(cfg=config)

        handle.restore()
        handle.restore()  # Should not raise

        assert handle.restored is True

    def test_handle_context_manager(self):
        """Test handle as context manager."""
        config = StdIOConfig()

        with StdIOHandle(cfg=config) as handle:
            assert handle.restored is False

        # Should be restored after exiting context
        assert handle.restored is True


class TestInstallation:
    """Test installation and uninstallation."""

    def test_install_basic(self):
        """Test basic installation."""
        handle = install(
            replace_builtins=False,
            patch_ray=False,
            patch_colorama=False,
            patch_tqdm=False,
        )

        try:
            assert isinstance(handle, StdIOHandle)
        finally:
            handle.restore()

    def test_install_as_context_manager(self):
        """Test installation with context manager."""
        output = StringIO()

        with install(replace_builtins=False) as handle:
            assert isinstance(handle, StdIOHandle)
            safe_print("Test message", file=output)

        # Should be restored
        assert handle.restored is True

    def test_install_replace_builtins(self):
        """Test replacing builtins.print."""
        import builtins

        original_print = builtins.print

        with install(replace_builtins=True):
            # builtins.print should be replaced
            assert builtins.print != original_print

        # Should be restored
        assert builtins.print == original_print

    def test_install_idempotent(self):
        """Test that multiple installs are safe."""
        handle1 = install(replace_builtins=False)
        handle2 = install(replace_builtins=False)

        try:
            assert isinstance(handle1, StdIOHandle)
            assert isinstance(handle2, StdIOHandle)
        finally:
            handle1.restore()
            handle2.restore()

    @patch("stdio_policy._patch_colorama")
    def test_install_patches_colorama(self, mock_patch):
        """Test Colorama patching."""
        mock_patch.return_value = True

        with install(patch_colorama=True):
            assert mock_patch.called

    @patch("stdio_policy._patch_ray")
    def test_install_patches_ray(self, mock_patch):
        """Test Ray patching."""
        mock_patch.return_value = True

        with install(patch_ray=True):
            assert mock_patch.called

    @patch("stdio_policy._patch_tqdm")
    def test_install_patches_tqdm(self, mock_patch):
        """Test tqdm patching."""
        mock_patch.return_value = True

        with install(patch_tqdm=True):
            assert mock_patch.called


class TestThreadSafety:
    """Test thread safety."""

    def test_concurrent_printing(self):
        """Test concurrent safe_print calls."""
        output = StringIO()
        config = StdIOConfig(lock_print=True, replace_builtins=False)

        def print_many(thread_id):
            for i in range(10):
                safe_print(f"Thread {thread_id}: {i}", cfg=config, file=output)

        threads = [threading.Thread(target=print_many, args=(i,)) for i in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should not crash and should have output
        result = output.getvalue()
        assert len(result) > 0

    def test_concurrent_jsonl_writing(self, temp_jsonl_file):
        """Test concurrent JSONL writing."""
        config = StdIOConfig(
            jsonl_path=temp_jsonl_file, lock_print=True, replace_builtins=False
        )

        def write_many(thread_id):
            output = StringIO()
            for i in range(5):
                safe_print(f"Thread {thread_id}: {i}", cfg=config, file=output)

        threads = [threading.Thread(target=write_many, args=(i,)) for i in range(3)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Check that all lines are valid JSON
        with open(temp_jsonl_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) == 15  # 3 threads * 5 messages
        for line in lines:
            json.loads(line)  # Should not raise


class TestSelfTest:
    """Test self_test function."""

    def test_self_test_runs(self):
        """Test that self_test runs without errors."""
        with install(replace_builtins=False):
            result = self_test()

        assert isinstance(result, dict)
        assert "os" in result
        assert "print_ok" in result

    def test_self_test_print_ok(self):
        """Test that self_test reports print success."""
        with install(replace_builtins=False):
            result = self_test()

        assert result["print_ok"] is True


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dirname_handling(self):
        """Test handling of JSONL path with empty dirname."""
        output = StringIO()
        # Use filename in current directory (no dirname)
        config = StdIOConfig(jsonl_path="test.jsonl", replace_builtins=False)

        try:
            safe_print("Test", cfg=config, file=output)

            # Should create file in current directory
            assert Path("test.jsonl").exists()
        finally:
            # Cleanup
            Path("test.jsonl").unlink(missing_ok=True)

    def test_write_failure_fallback(self):
        """Test fallback when stream write fails."""
        # Create a mock stream that raises on write
        mock_stream = MagicMock()
        mock_stream.write.side_effect = Exception("Write failed")

        config = StdIOConfig(replace_builtins=False)

        # Should not raise, should fall back to sys.__stdout__
        from stdio_policy import _write

        _write(mock_stream, "test", config)

    def test_jsonl_write_failure(self):
        """Test handling of JSONL write failure."""
        output = StringIO()
        # Use invalid path
        config = StdIOConfig(
            jsonl_path="/invalid/path/test.jsonl", replace_builtins=False
        )

        # Should not crash, just log warning
        safe_print("Test", cfg=config, file=output)

        assert "Test" in output.getvalue()

    def test_large_text_truncation(self):
        """Test that very large text is truncated."""
        output = StringIO()
        config = StdIOConfig(max_len=100, replace_builtins=False)
        large_text = "x" * 1000

        safe_print(large_text, cfg=config, file=output)

        result = output.getvalue()

        # Should be truncated
        assert len(result) < 1000
        assert "truncated" in result


class TestPatching:
    """Test library patching functions."""

    def test_patch_colorama_success(self):
        """Test successful Colorama patching."""
        mock_colorama = MagicMock()
        mock_colorama.init = Mock()
        mock_colorama.deinit = Mock()

        # Patch colorama in sys.modules so the import inside _patch_colorama finds it
        with patch.dict("sys.modules", {"colorama": mock_colorama}):
            from stdio_policy import _patch_colorama

            config = StdIOConfig()

            result = _patch_colorama(config)

            assert result is True
            assert mock_colorama.init.called

    def test_patch_colorama_not_available(self):
        """Test Colorama patching when not available."""
        # Make the import fail
        with patch.dict("sys.modules", {"colorama": None}):
            from stdio_policy import _patch_colorama

            config = StdIOConfig()

            result = _patch_colorama(config)

            # Should return False when not available
            assert result is False

    def test_patch_tqdm_success(self):
        """Test successful tqdm patching."""
        mock_tqdm = MagicMock()

        # Patch tqdm in sys.modules
        with patch.dict("sys.modules", {"tqdm": mock_tqdm}):
            from stdio_policy import _patch_tqdm

            config = StdIOConfig()

            result = _patch_tqdm(config)

            assert result is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
