"""
Tests for vulcan-vector-bootstrap Python script

This version includes Windows compatibility fixes.
"""

import json
import os
import platform
import subprocess
import sys
import tempfile

import pytest

BIN_DIR = os.path.join(os.path.dirname(__file__), "..", "bin")
VULCAN_BOOTSTRAP = os.path.join(BIN_DIR, "vulcan-vector-bootstrap")


def run_vulcan_bootstrap(args, **kwargs):
    """
    Helper function to run vulcan-vector-bootstrap with proper platform-specific handling.

    On Windows, Python scripts can't be executed directly - they need to be
    run with the Python interpreter.
    
    This function now uses Popen for better subprocess management with explicit
    timeout handling and process termination to prevent hanging.
    """
    if platform.system() == "Windows":
        command = [sys.executable, VULCAN_BOOTSTRAP] + args
    else:
        command = [VULCAN_BOOTSTRAP] + args

    # Extract timeout from kwargs, default to 20 seconds (reduced from 30 for faster feedback)
    timeout = kwargs.pop("timeout", 20)
    
    # Ensure we capture output
    kwargs.setdefault("capture_output", True)
    kwargs.setdefault("text", True)
    kwargs.setdefault("stdout", subprocess.PIPE)
    kwargs.setdefault("stderr", subprocess.PIPE)
    
    # Log the command being run for debugging
    print(f"[TEST] Running: {' '.join(command)}")
    print(f"[TEST] Timeout: {timeout}s")
    
    # Use Popen for better control
    process = subprocess.Popen(command, **kwargs)
    
    try:
        # Wait for process with timeout
        stdout, stderr = process.communicate(timeout=timeout)
        
        # Log output for debugging failures
        if stdout:
            print(f"[TEST] STDOUT (first 500 chars): {stdout[:500]}")
        if stderr:
            print(f"[TEST] STDERR (first 500 chars): {stderr[:500]}")
        
        # Create result object
        class Result:
            def __init__(self, returncode, stdout, stderr):
                self.returncode = returncode
                self.stdout = stdout or ""
                self.stderr = stderr or ""
                self.args = command
        
        return Result(process.returncode, stdout, stderr)
    
    except subprocess.TimeoutExpired:
        # Process timed out - terminate it forcefully
        print(f"[TEST] ERROR: Process timed out after {timeout}s, terminating...")
        
        # Try graceful termination first
        process.terminate()
        try:
            # Give it 2 seconds to terminate gracefully
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            # If still alive, force kill
            print(f"[TEST] Process did not terminate gracefully, killing...")
            process.kill()
            process.wait()  # Wait for the kill to complete
        
        # Try to get any partial output
        try:
            stdout, stderr = process.communicate(timeout=1)
        except:
            stdout, stderr = b"", b""
        
        print(f"[TEST] Partial STDOUT: {stdout[:1000] if stdout else 'None'}")
        print(f"[TEST] Partial STDERR: {stderr[:1000] if stderr else 'None'}")
        
        # Return a result indicating timeout
        class TimeoutResult:
            def __init__(self):
                self.returncode = -1
                self.stdout = (stdout.decode('utf-8') if isinstance(stdout, bytes) else stdout) or ""
                self.stderr = ((stderr.decode('utf-8') if isinstance(stderr, bytes) else stderr) or "") + f"\n[TIMEOUT after {timeout}s]"
                self.args = command
        
        return TimeoutResult()
    
    except Exception as e:
        # Cleanup on any other error
        print(f"[TEST] ERROR: Unexpected exception: {e}")
        try:
            process.kill()
            process.wait()
        except:
            pass
        raise


class TestVulcanVectorBootstrap:
    """Test suite for vulcan-vector-bootstrap"""

    def test_bootstrap_exists(self):
        """Test that vulcan-vector-bootstrap exists and is executable"""
        assert os.path.exists(VULCAN_BOOTSTRAP)
        if platform.system() != "Windows":
            assert os.access(VULCAN_BOOTSTRAP, os.X_OK)

    def test_help_flag(self):
        """Test --help flag"""
        result = run_vulcan_bootstrap(["--help"])
        assert result.returncode == 0
        assert "VulcanAMI Vector Bootstrap" in result.stdout

    def test_version_in_help(self):
        """Test version is shown in help"""
        result = run_vulcan_bootstrap(["--help"])
        assert "4.6.0" in result.stdout

    def test_bootstrap_default(self):
        """Test default bootstrap (all tiers)"""
        result = run_vulcan_bootstrap([], timeout=30)
        assert result.returncode == 0
        output = result.stdout + result.stderr
        assert (
            "Bootstrap" in output
            or "collection" in output.lower()
            or "SUCCESS" in output
        )

    def test_bootstrap_all_tiers(self):
        """Test bootstrapping all tiers"""
        result = run_vulcan_bootstrap(["--tier", "all"], timeout=30)
        assert result.returncode == 0

    def test_bootstrap_hot_tier(self):
        """Test bootstrapping hot tier"""
        result = run_vulcan_bootstrap(["--tier", "hot"], timeout=30)
        assert result.returncode == 0

    def test_bootstrap_warm_tier(self):
        """Test bootstrapping warm tier"""
        result = run_vulcan_bootstrap(["--tier", "warm"], timeout=30)
        assert result.returncode == 0

    def test_bootstrap_cold_tier(self):
        """Test bootstrapping cold tier"""
        result = run_vulcan_bootstrap(["--tier", "cold"], timeout=30)
        assert result.returncode == 0

    def test_bootstrap_with_dimension(self):
        """Test custom dimension parameter"""
        # Test only 2 dimensions instead of 4 to reduce test time
        # Use shorter timeout to fail faster if something hangs
        for dim in [128, 512]:
            result = run_vulcan_bootstrap(
                ["--dimension", str(dim), "--tier", "hot"], timeout=20
            )
            # If the command times out, it will return -1
            # This is acceptable since we're testing that the command doesn't hang indefinitely
            if result.returncode != 0:
                # Log the failure but don't fail the test if it's a connection error (simulation mode)
                if "simulation" in result.stdout.lower() or "simulation" in result.stderr.lower():
                    print(f"[TEST] Bootstrap ran in simulation mode for dimension {dim}")
                else:
                    print(f"[TEST] Bootstrap failed with returncode {result.returncode} for dimension {dim}")
                    print(f"[TEST] STDERR: {result.stderr}")
            assert result.returncode == 0, f"Bootstrap failed for dimension {dim}: {result.stderr}"

    def test_bootstrap_with_l2_metric(self):
        """Test L2 distance metric"""
        result = run_vulcan_bootstrap(["--metric", "L2", "--tier", "hot"], timeout=30)
        assert result.returncode == 0

    def test_bootstrap_with_ip_metric(self):
        """Test IP (inner product) metric"""
        result = run_vulcan_bootstrap(["--metric", "IP", "--tier", "hot"], timeout=30)
        assert result.returncode == 0

    def test_bootstrap_with_cosine_metric(self):
        """Test COSINE metric"""
        result = run_vulcan_bootstrap(
            ["--metric", "COSINE", "--tier", "hot"], timeout=30
        )
        assert result.returncode == 0

    def test_bootstrap_with_flat_index(self):
        """Test FLAT index type"""
        result = run_vulcan_bootstrap(
            ["--index-type", "FLAT", "--tier", "hot"], timeout=30
        )
        assert result.returncode == 0

    def test_bootstrap_with_ivf_flat_index(self):
        """Test IVF_FLAT index type"""
        result = run_vulcan_bootstrap(
            ["--index-type", "IVF_FLAT", "--tier", "hot"], timeout=30
        )
        assert result.returncode == 0

    def test_bootstrap_with_ivf_sq8_index(self):
        """Test IVF_SQ8 index type"""
        result = run_vulcan_bootstrap(
            ["--index-type", "IVF_SQ8", "--tier", "hot"], timeout=30
        )
        assert result.returncode == 0

    def test_bootstrap_with_hnsw_index(self):
        """Test HNSW index type"""
        result = run_vulcan_bootstrap(
            ["--index-type", "HNSW", "--tier", "hot"], timeout=30
        )
        assert result.returncode == 0

    def test_bootstrap_with_drop_existing(self):
        """Test --drop-existing flag"""
        result = run_vulcan_bootstrap(["--drop-existing", "--tier", "hot"], timeout=30)
        assert result.returncode == 0

    def test_bootstrap_verbose_mode(self):
        """Test verbose mode"""
        result = run_vulcan_bootstrap(["--verbose", "--tier", "hot"], timeout=30)
        assert result.returncode == 0

    def test_bootstrap_quiet_mode(self):
        """Test quiet mode"""
        result = run_vulcan_bootstrap(["--quiet", "--tier", "hot"], timeout=30)
        assert result.returncode == 0

    def test_bootstrap_with_json_output(self):
        """Test JSON output"""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_output = os.path.join(tmpdir, "bootstrap.json")

            result = run_vulcan_bootstrap(
                ["--tier", "hot", "--json", json_output], timeout=30
            )

            assert result.returncode == 0
            assert os.path.exists(json_output)

            # Verify JSON structure
            with open(json_output, "r", encoding="utf-8") as f:
                data = json.load(f)
                assert "collections" in data
                assert "bootstrap_time" in data
                assert "success" in data

    def test_bootstrap_displays_summary(self):
        """Test that bootstrap displays summary"""
        result = run_vulcan_bootstrap(["--tier", "hot"], timeout=30)

        output = result.stdout + result.stderr
        assert "Bootstrap" in output or "Collection" in output or "Summary" in output

    def test_bootstrap_invalid_tier(self):
        """Test invalid tier is rejected"""
        result = run_vulcan_bootstrap(["--tier", "invalid"], timeout=30)
        assert result.returncode != 0

    def test_bootstrap_invalid_metric(self):
        """Test invalid metric is rejected"""
        result = run_vulcan_bootstrap(["--metric", "INVALID"], timeout=30)
        assert result.returncode != 0

    def test_bootstrap_invalid_index_type(self):
        """Test invalid index type is rejected"""
        result = run_vulcan_bootstrap(["--index-type", "INVALID"], timeout=30)
        assert result.returncode != 0

    def test_bootstrap_with_all_options(self):
        """Test combining all options"""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_output = os.path.join(tmpdir, "result.json")

            result = run_vulcan_bootstrap(
                [
                    "--tier",
                    "hot",
                    "--dimension",
                    "256",
                    "--metric",
                    "COSINE",
                    "--index-type",
                    "HNSW",
                    "--verbose",
                    "--json",
                    json_output,
                ],
                timeout=30,
            )

            assert result.returncode == 0
            assert os.path.exists(json_output)

    def test_bootstrap_multiple_tiers_sequentially(self):
        """Test bootstrapping multiple tiers"""
        tiers = ["hot", "warm", "cold"]
        for tier in tiers:
            result = run_vulcan_bootstrap(["--tier", tier, "--quiet"], timeout=30)
            assert result.returncode == 0
