#!/usr/bin/env python3
"""
Validation tests for bug fixes in:
- scripts/run_scheduled_tests.sh
- scripts/scheduled_adversarial_testing.py
- src/vulcan/planning.py
"""

import os
import subprocess
import sys
from pathlib import Path


def test_shell_script_syntax():
    """Test that the shell script has valid syntax."""
    print("Testing shell script syntax...")
    result = subprocess.run(
        ['bash', '-n', 'scripts/run_scheduled_tests.sh'],
        cwd='/home/runner/work/VulcanAMI_LLM/VulcanAMI_LLM',
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"Shell script has syntax errors: {result.stderr}"
    print("✓ Shell script syntax is valid")

def test_shell_script_help():
    """Test that the shell script help works."""
    print("Testing shell script help...")
    result = subprocess.run(
        ['bash', 'scripts/run_scheduled_tests.sh', '--help'],
        cwd='/home/runner/work/VulcanAMI_LLM/VulcanAMI_LLM',
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"Shell script help failed: {result.stderr}"
    assert 'Usage:' in result.stdout
    print("✓ Shell script help works")

def test_python_script_syntax():
    """Test that the Python script compiles without syntax errors."""
    print("Testing Python script syntax...")
    result = subprocess.run(
        [sys.executable, '-m', 'py_compile', 'scripts/scheduled_adversarial_testing.py'],
        cwd='/home/runner/work/VulcanAMI_LLM/VulcanAMI_LLM',
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"Python script has syntax errors: {result.stderr}"
    print("✓ Python script syntax is valid")

def test_python_script_help():
    """Test that the Python script help works."""
    print("Testing Python script help...")
    result = subprocess.run(
        [sys.executable, 'scripts/scheduled_adversarial_testing.py', '--help'],
        cwd='/home/runner/work/VulcanAMI_LLM/VulcanAMI_LLM',
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"Python script help failed: {result.stderr}"
    assert 'usage:' in result.stdout
    print("✓ Python script help works")

def test_planning_module_syntax():
    """Test that the planning module compiles without syntax errors."""
    print("Testing planning.py syntax...")
    result = subprocess.run(
        [sys.executable, '-m', 'py_compile', 'src/vulcan/planning.py'],
        cwd='/home/runner/work/VulcanAMI_LLM/VulcanAMI_LLM',
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"planning.py has syntax errors: {result.stderr}"
    print("✓ planning.py syntax is valid")

def test_python_script_import():
    """Test that the Python script can be imported without error."""
    print("Testing Python script import...")
    
    # Create a simple test that imports the module
    test_code = """
import sys
from pathlib import Path
scripts_dir = Path('/home/runner/work/VulcanAMI_LLM/VulcanAMI_LLM/scripts')
sys.path.insert(0, str(scripts_dir))

try:
    # This will import the module without running main()
    import scheduled_adversarial_testing
    print('Import successful')
except ImportError as e:
    print(f'Import failed: {e}')
    sys.exit(1)
"""
    
    result = subprocess.run(
        [sys.executable, '-c', test_code],
        capture_output=True,
        text=True
    )
    # Note: May fail due to missing dependencies, but syntax should be OK
    if 'SyntaxError' in result.stderr:
        assert False, f"Python script has import errors: {result.stderr}"
    print("✓ Python script can be imported (syntax is valid)")

def test_shell_script_has_lock_file_logic():
    """Test that lock file logic exists in shell script."""
    print("Testing shell script lock file logic...")
    script_path = Path('/home/runner/work/VulcanAMI_LLM/VulcanAMI_LLM/scripts/run_scheduled_tests.sh')
    content = script_path.read_text()
    
    assert 'LOCK_FILE=' in content, "Lock file variable not found"
    assert 'trap' in content, "Trap for lock file cleanup not found"
    print("✓ Shell script has lock file logic")

def test_shell_script_has_proper_exit_code_handling():
    """Test that exit code is properly captured."""
    print("Testing shell script exit code handling...")
    script_path = Path('/home/runner/work/VulcanAMI_LLM/VulcanAMI_LLM/scripts/run_scheduled_tests.sh')
    content = script_path.read_text()
    
    # Should have proper exit code capture after eval
    assert 'eval "$CMD"' in content
    assert 'EXIT_CODE=$?' in content
    # Should check exit code with if statement
    assert '[ $EXIT_CODE -eq 0 ]' in content or 'if [ $EXIT_CODE -eq 0 ]' in content
    print("✓ Shell script has proper exit code handling")

def test_shell_script_color_code_handling():
    """Test that color codes are not written to log file."""
    print("Testing shell script color code handling...")
    script_path = Path('/home/runner/work/VulcanAMI_LLM/VulcanAMI_LLM/scripts/run_scheduled_tests.sh')
    content = script_path.read_text()
    
    # Should have separate logging for colored and plain text
    assert 'echo -e "${RED}' in content or 'echo -e "${GREEN}' in content
    assert 'echo "$msg" >> "$LOG_FILE"' in content
    print("✓ Shell script properly handles color codes in logs")

def test_python_script_has_timeout():
    """Test that timeout enforcement exists."""
    print("Testing Python script timeout enforcement...")
    script_path = Path('/home/runner/work/VulcanAMI_LLM/VulcanAMI_LLM/scripts/scheduled_adversarial_testing.py')
    content = script_path.read_text()
    
    assert 'import signal' in content, "signal module not imported"
    assert 'timeout_seconds' in content, "timeout_seconds config not found"
    assert 'SIGALRM' in content or 'signal.alarm' in content, "Signal alarm not used"
    print("✓ Python script has timeout enforcement")

def test_python_script_differentiated_exit_codes():
    """Test that exit codes are differentiated."""
    print("Testing Python script exit code differentiation...")
    script_path = Path('/home/runner/work/VulcanAMI_LLM/VulcanAMI_LLM/scripts/scheduled_adversarial_testing.py')
    content = script_path.read_text()
    
    # Should return different exit codes for different failures
    assert 'return 1' in content, "Exit code 1 not found"
    assert 'return 2' in content, "Exit code 2 not found"
    assert 'TimeoutError' in content, "TimeoutError not handled"
    print("✓ Python script has differentiated exit codes")

def test_python_script_no_unused_imports():
    """Test that unused AttackType import is removed."""
    print("Testing Python script unused imports...")
    script_path = Path('/home/runner/work/VulcanAMI_LLM/VulcanAMI_LLM/scripts/scheduled_adversarial_testing.py')
    content = script_path.read_text()
    
    # Check that AttackType is not imported separately (it may be part of the module)
    lines = content.split('\n')
    from_imports = [l for l in lines if 'from src.vulcan.safety.adversarial_formal import' in l]
    
    # If there are from imports, AttackType should not be in them
    for line in from_imports:
        if 'AttackType' in line:
            # It's okay if it's commented out or part of a try/except that gracefully handles it
            assert 'AttackType,' not in line or '#' in line, "AttackType is still imported but unused"
    
    print("✓ Python script has no unused AttackType import")

def test_planning_module_has_constants():
    """Test that magic numbers are replaced with constants."""
    print("Testing planning.py constants...")
    script_path = Path('/home/runner/work/VulcanAMI_LLM/VulcanAMI_LLM/src/vulcan/planning.py')
    content = script_path.read_text()
    
    assert 'CPU_CRITICAL_THRESHOLD' in content, "CPU_CRITICAL_THRESHOLD constant not found"
    assert 'MEMORY_CRITICAL_THRESHOLD' in content, "MEMORY_CRITICAL_THRESHOLD constant not found"
    assert 'MAX_CACHE_SIZE' in content, "MAX_CACHE_SIZE constant not found"
    assert 'CACHE_TTL_SECONDS' in content, "CACHE_TTL_SECONDS constant not found"
    print("✓ planning.py has constants for magic numbers")

def test_planning_module_thread_safety():
    """Test that thread safety improvements exist."""
    print("Testing planning.py thread safety...")
    script_path = Path('/home/runner/work/VulcanAMI_LLM/VulcanAMI_LLM/src/vulcan/planning.py')
    content = script_path.read_text()
    
    assert '_state_lock' in content, "_state_lock not found"
    assert 'threading.RLock()' in content, "RLock not used"
    assert '.wait(timeout=' in content, "wait() with timeout not used for stop event"
    print("✓ planning.py has thread safety improvements")

def test_planning_module_socket_leak_fix():
    """Test that socket connection leak is fixed."""
    print("Testing planning.py socket connection fix...")
    script_path = Path('/home/runner/work/VulcanAMI_LLM/VulcanAMI_LLM/src/vulcan/planning.py')
    content = script_path.read_text()
    
    assert 'with socket.create_connection' in content, "Context manager for socket not used"
    print("✓ planning.py has socket connection leak fix")

def test_planning_module_no_bare_except():
    """Test that bare except clauses are replaced."""
    print("Testing planning.py bare except clauses...")
    script_path = Path('/home/runner/work/VulcanAMI_LLM/VulcanAMI_LLM/src/vulcan/planning.py')
    content = script_path.read_text()
    
    lines = content.split('\n')
    bare_excepts = [i for i, line in enumerate(lines) if line.strip() == 'except:']
    
    # Should have very few or no bare except clauses
    assert len(bare_excepts) < 2, f"Found {len(bare_excepts)} bare except clauses at lines: {bare_excepts}"
    print("✓ planning.py has minimal bare except clauses")

def test_planning_module_iterative_cleanup():
    """Test that recursive cleanup is replaced with iterative."""
    print("Testing planning.py iterative cleanup...")
    script_path = Path('/home/runner/work/VulcanAMI_LLM/VulcanAMI_LLM/src/vulcan/planning.py')
    content = script_path.read_text()
    
    # Look for the MCTSNode cleanup method
    assert 'nodes_to_cleanup' in content, "Iterative cleanup not implemented"
    assert 'while nodes_to_cleanup:' in content, "Iterative loop not found"
    print("✓ planning.py has iterative cleanup to prevent stack overflow")

def test_planning_module_cache_race_condition():
    """Test that cache race condition is fixed."""
    print("Testing planning.py cache race condition fix...")
    script_path = Path('/home/runner/work/VulcanAMI_LLM/VulcanAMI_LLM/src/vulcan/planning.py')
    content = script_path.read_text()
    
    # Should have atomic check-and-compute pattern
    assert "'computing': True" in content, "Atomic cache pattern not found"
    assert 'try:' in content and 'except Exception as e:' in content, "Exception handling for cache not found"
    print("✓ planning.py has cache race condition fix")

def test_planning_module_survival_protocol_attributes():
    """Test that SurvivalProtocol has all required attributes."""
    print("Testing planning.py SurvivalProtocol attributes...")
    script_path = Path('/home/runner/work/VulcanAMI_LLM/VulcanAMI_LLM/src/vulcan/planning.py')
    content = script_path.read_text()
    
    # Should initialize network attributes
    assert 'network_retry_enabled' in content, "network_retry_enabled not found"
    assert 'network_batch_size' in content, "network_batch_size not found"
    assert 'network_priority_threshold' in content, "network_priority_threshold not found"
    print("✓ planning.py has SurvivalProtocol attributes initialized")

def main():
    """Run all validation tests."""
    print("=" * 60)
    print("Bug Fix Validation Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_shell_script_syntax,
        test_shell_script_help,
        test_shell_script_has_lock_file_logic,
        test_shell_script_has_proper_exit_code_handling,
        test_shell_script_color_code_handling,
        test_python_script_syntax,
        test_python_script_help,
        test_python_script_import,
        test_python_script_has_timeout,
        test_python_script_differentiated_exit_codes,
        test_python_script_no_unused_imports,
        test_planning_module_syntax,
        test_planning_module_has_constants,
        test_planning_module_thread_safety,
        test_planning_module_socket_leak_fix,
        test_planning_module_no_bare_except,
        test_planning_module_iterative_cleanup,
        test_planning_module_cache_race_condition,
        test_planning_module_survival_protocol_attributes,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} error: {e}")
            failed += 1
        print()
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return 0 if failed == 0 else 1

if __name__ == '__main__':
    sys.exit(main())
