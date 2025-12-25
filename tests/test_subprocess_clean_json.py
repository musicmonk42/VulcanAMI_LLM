"""
Test that subprocess outputs clean JSON to stdout with logs to stderr.

This tests the fix for the 'Dirty JSON' issue where logging output to stdout
before JSON caused parse failures (e.g., 'Extra data: line 1 column 5').

This is a standalone test that doesn't import graphix_arena directly to avoid
import issues with optional dependencies.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest


class TestSubprocessCleanJson:
    """Test that subprocess outputs clean JSON to stdout with logs to stderr."""

    def test_subprocess_logs_go_to_stderr(self):
        """Verify that the subprocess script routes logs to stderr, not stdout.
        
        This validates the fix in graphix_arena.py where we configure logging
        to write to stderr BEFORE importing modules. Without this fix, logging
        from llm_client.py (and other modules) goes to stdout, contaminating
        the JSON response and causing parse failures like:
        
            "Failed to parse agent output: Extra data: line 1 column 5 (char 4)"
        
        This caused 40-second timeouts as the parent process waited for valid JSON.
        """
        # Get the project paths
        project_root = Path(__file__).resolve().parent.parent
        src_dir = project_root / "src"

        # The script that graphix_arena generates - it configures logging to stderr first
        # This mirrors the script in graphix_arena.py _run_agent method
        script = (
            "import sys, logging; "
            "logging.basicConfig(stream=sys.stderr, level=logging.INFO, "
            "format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'); "
            "import json; from llm_client import GraphixLLMClient; "
            "client=GraphixLLMClient('test_agent'); "
            'messages = [{"role": "user", "content": "test prompt"}]; '
            "print(json.dumps(client.chat(messages)))"
        )

        # Set up environment with correct PYTHONPATH
        import os
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{project_root}:{src_dir}"

        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            env=env,
            cwd=str(project_root),
            timeout=60,
        )

        stdout_content = result.stdout.decode()
        stderr_content = result.stderr.decode()

        # stdout should contain ONLY valid JSON (no log lines)
        assert stdout_content.strip(), f"stdout should not be empty. stderr: {stderr_content[:500]}"
        
        # Parse the JSON - this should succeed without "Extra data" error
        try:
            parsed = json.loads(stdout_content)
            assert isinstance(parsed, dict), "Output should be a JSON object"
            # The response from mock mode should have expected structure
            assert "response" in parsed, "Response should contain 'response' key"
            assert "ir" in parsed, "Response should contain 'ir' key for IR graph"
            assert "proposal_id" in parsed, "Response should contain 'proposal_id' key"
        except json.JSONDecodeError as e:
            pytest.fail(
                f"stdout should be valid JSON but got error: {e}\n"
                f"stdout content: {stdout_content[:500]}\n"
                f"stderr content: {stderr_content[:500]}"
            )

        # Verify that if there are any log messages (INFO, WARNING, etc.),
        # they appear in stderr, not stdout
        # Common log patterns that should NOT be in stdout
        log_patterns = ["- INFO -", "- WARNING -", "- ERROR -", "- DEBUG -"]
        for pattern in log_patterns:
            assert pattern not in stdout_content, (
                f"Log pattern '{pattern}' found in stdout (should be in stderr).\n"
                f"stdout: {stdout_content[:200]}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
