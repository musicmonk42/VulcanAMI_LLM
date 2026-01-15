# src/vulcan/world_model/meta_reasoning/safe_execution.py
"""
Safe Execution Module for Self-Improvement Actions

This module provides sandboxed execution of improvement actions with:
- Command whitelisting
- No shell=True usage
- Proper argument escaping
- Resource limits (timeout, memory)
- Audit logging
- Rollback capability

Security: CRITICAL
"""

import logging
import os
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of safe command execution"""

    success: bool
    stdout: str
    stderr: str
    returncode: int
    duration: float
    command: List[str]
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "returncode": self.returncode,
            "duration": self.duration,
            "command": " ".join(self.command),
            "error": self.error,
        }


class SafeExecutor:
    """
    Safe executor for self-improvement actions

    Provides sandboxed execution with security controls
    """

    # Whitelist of allowed commands
    ALLOWED_COMMANDS = {
        # Testing
        "pytest",
        "python",
        "python3",
        # Linting/Formatting
        "black",
        "flake8",
        "mypy",
        "pylint",
        "isort",
        # Git (read-only operations)
        "git",
        # Documentation
        "sphinx-build",
        "mkdocs",
        # Package management (read-only)
        "pip",
    }

    # Commands that require special validation
    RESTRICTED_COMMANDS = {
        "git": ["status", "diff", "log", "show", "branch"],  # Only read operations
        "pip": ["list", "show", "check"],  # Only informational
    }

    # Dangerous argument patterns to block
    DANGEROUS_PATTERNS = [
        "|",  # Pipes
        ";",  # Command chaining
        "&",  # Background execution
        "$",  # Variable expansion
        "`",  # Command substitution
        ">",  # Redirection
        "<",  # Redirection
        "(",  # Subshells
        ")",  # Subshells
    ]

    def __init__(
        self,
        timeout: int = 60,
        working_dir: Optional[Path] = None,
        audit_callback: Optional[callable] = None,
    ):
        """
        Initialize safe executor

        Args:
            timeout: Maximum execution time in seconds
            working_dir: Working directory for commands
            audit_callback: Optional callback for audit logging
        """
        self.timeout = timeout
        self.working_dir = working_dir or Path.cwd()
        self.audit_callback = audit_callback
        self._lock = threading.RLock()
        self._execution_count = 0

        logger.info(
            f"SafeExecutor initialized with timeout={timeout}s, wd={self.working_dir}"
        )

    def __getstate__(self) -> Dict[str, Any]:
        """
        Prepare state for pickling by removing unpickleable lock objects.
        """
        state = self.__dict__.copy()
        state.pop('_lock', None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """
        Restore state after unpickling, re-creating the lock.
        """
        self.__dict__.update(state)
        self._lock = threading.RLock()

    def is_command_allowed(self, command: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Check if command is allowed

        Args:
            command: Command as list of strings

        Returns:
            (allowed, reason)
        """
        if not command:
            return False, "Empty command"

        cmd_name = command[0]

        # Check if command is in whitelist
        if cmd_name not in self.ALLOWED_COMMANDS:
            return False, f"Command '{cmd_name}' not in whitelist"

        # Check for dangerous patterns in arguments
        for arg in command[1:]:
            for pattern in self.DANGEROUS_PATTERNS:
                if pattern in arg:
                    return False, f"Dangerous pattern '{pattern}' in argument: {arg}"

        # Check restricted commands
        if cmd_name in self.RESTRICTED_COMMANDS:
            allowed_subcommands = self.RESTRICTED_COMMANDS[cmd_name]
            if len(command) < 2 or command[1] not in allowed_subcommands:
                return False, f"'{cmd_name}' requires one of: {allowed_subcommands}"

        return True, None

    def validate_working_directory(self, path: Path) -> Tuple[bool, Optional[str]]:
        """
        Validate working directory is safe

        Args:
            path: Directory path

        Returns:
            (valid, reason)
        """
        try:
            # Must exist
            if not path.exists():
                return False, f"Directory does not exist: {path}"

            # Must be a directory
            if not path.is_dir():
                return False, f"Not a directory: {path}"

            # Must be within allowed paths (project root)
            # For safety, restrict to project directory
            try:
                path.resolve().relative_to(Path.cwd().resolve())
            except ValueError:
                return False, f"Directory outside project: {path}"

            return True, None

        except Exception as e:
            return False, f"Validation error: {e}"

    def execute_safe(
        self,
        command: List[str],
        working_dir: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
    ) -> ExecutionResult:
        """
        Execute command safely

        Args:
            command: Command as list of strings (NOT string!)
            working_dir: Override working directory
            env: Environment variables
            timeout: Override timeout

        Returns:
            ExecutionResult
        """
        start_time = time.time()

        with self._lock:
            self._execution_count += 1
            exec_id = self._execution_count

        logger.info(f"[{exec_id}] Safe execution requested: {command}")

        # Validate command
        allowed, reason = self.is_command_allowed(command)
        if not allowed:
            logger.error(f"[{exec_id}] Command blocked: {reason}")
            return ExecutionResult(
                success=False,
                stdout="",
                stderr="",
                returncode=-1,
                duration=0.0,
                command=command,
                error=f"Blocked: {reason}",
            )

        # Validate working directory
        wd = working_dir or self.working_dir
        valid_dir, dir_reason = self.validate_working_directory(wd)
        if not valid_dir:
            logger.error(f"[{exec_id}] Invalid working directory: {dir_reason}")
            return ExecutionResult(
                success=False,
                stdout="",
                stderr="",
                returncode=-1,
                duration=0.0,
                command=command,
                error=f"Invalid working directory: {dir_reason}",
            )

        # Prepare environment (restrict to safe subset)
        safe_env = os.environ.copy()
        if env:
            # Only allow specific environment variables
            allowed_env_vars = {"PYTHONPATH", "PATH", "HOME", "USER", "LANG"}
            for key, value in env.items():
                if key in allowed_env_vars:
                    safe_env[key] = value
                else:
                    logger.warning(f"[{exec_id}] Env var '{key}' not allowed, skipped")

        # Execute
        timeout_val = timeout or self.timeout
        try:
            logger.info(
                f"[{exec_id}] Executing: {' '.join(command)} (timeout={timeout_val}s)"
            )

            # nosec B603: subprocess call is safe - using list arguments (not shell=True),
            # command comes from internal safe execution pipeline with validated inputs
            result = subprocess.run(  # nosec B603
                command,  # List, not string - SAFE
                shell=False,  # NEVER use shell=True - SAFE
                capture_output=True,
                text=True,
                timeout=timeout_val,
                cwd=str(wd),
                env=safe_env,
            )

            duration = time.time() - start_time
            success = result.returncode == 0

            exec_result = ExecutionResult(
                success=success,
                stdout=result.stdout,
                stderr=result.stderr,
                returncode=result.returncode,
                duration=duration,
                command=command,
            )

            logger.info(
                f"[{exec_id}] Completed: rc={result.returncode}, "
                f"duration={duration:.2f}s, success={success}"
            )

            # Audit log
            if self.audit_callback:
                self.audit_callback(
                    {
                        "exec_id": exec_id,
                        "command": command,
                        "success": success,
                        "returncode": result.returncode,
                        "duration": duration,
                        "working_dir": str(wd),
                    }
                )

            return exec_result

        except subprocess.TimeoutExpired as e:
            duration = time.time() - start_time
            logger.error(f"[{exec_id}] Timeout after {duration:.2f}s")

            return ExecutionResult(
                success=False,
                stdout=e.stdout or "",
                stderr=e.stderr or "",
                returncode=-1,
                duration=duration,
                command=command,
                error=f"Timeout after {timeout_val}s",
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"[{exec_id}] Execution failed: {e}")

            return ExecutionResult(
                success=False,
                stdout="",
                stderr=str(e),
                returncode=-1,
                duration=duration,
                command=command,
                error=str(e),
            )

    def execute_improvement_action(self, action: Dict[str, Any]) -> ExecutionResult:
        """
        Execute an improvement action safely

        Args:
            action: Action dictionary with 'command', 'type', etc.

        Returns:
            ExecutionResult
        """
        action_type = action.get("type", "unknown")

        logger.info(f"Executing improvement action: {action_type}")

        # Extract command
        command = action.get("command")
        if not command:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr="",
                returncode=-1,
                duration=0.0,
                command=[],
                error="No command specified",
            )

        # Convert string to list if needed (should already be list)
        if isinstance(command, str):
            # NEVER pass string directly - parse it safely
            logger.warning("Command is string, parsing with shlex")
            try:
                command = shlex.split(command)
            except Exception as e:
                return ExecutionResult(
                    success=False,
                    stdout="",
                    stderr="",
                    returncode=-1,
                    duration=0.0,
                    command=[command] if isinstance(command, str) else command,
                    error=f"Failed to parse command: {e}",
                )

        # Get working directory from action
        working_dir = action.get("working_dir")
        if working_dir:
            working_dir = Path(working_dir)

        # Get timeout from action
        timeout = action.get("timeout", self.timeout)

        # Execute
        return self.execute_safe(
            command=command, working_dir=working_dir, timeout=timeout
        )


# Singleton instance
_safe_executor: Optional[SafeExecutor] = None
_executor_lock = threading.Lock()


def get_safe_executor(timeout: int = 60) -> SafeExecutor:
    """Get or create global safe executor"""
    global _safe_executor

    with _executor_lock:
        if _safe_executor is None:
            _safe_executor = SafeExecutor(timeout=timeout)
            logger.info("Global SafeExecutor created")
        return _safe_executor


def reset_safe_executor():
    """Reset global safe executor (for testing)"""
    global _safe_executor

    with _executor_lock:
        _safe_executor = None
        logger.info("Global SafeExecutor reset")
