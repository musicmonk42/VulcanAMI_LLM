"""
self_improvement_apply.py - Extracted self-improvement application functions from WorldModel.

Contains LLM response parsing, AST validation, diff/commit application,
and the main improvement execution pipeline.
Phase 1 of WorldModel decomposition.
"""

import ast
import difflib
import logging
import subprocess
import time
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def _parse_llm_response(wm, response_text: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse the LLM's structured response for file path and content."""
    lines = response_text.strip().split("\n")
    file_path = None
    code_lines = []
    in_code_block = False

    for line in lines:
        if line.startswith("FILE:"):
            file_path = line.replace("FILE:", "").strip()
        elif line.strip().startswith("```python"):
            in_code_block = True
        elif line.strip().startswith("```") and in_code_block:
            in_code_block = False
        elif in_code_block:
            code_lines.append(line)

    if file_path:
        return file_path, "\n".join(code_lines)
    return None, None


def _validate_code_ast(wm, content: str) -> None:
    """Validate code content via AST parsing."""
    if not content:
        raise ValueError("Code content is empty.")
    ast.parse(content)


def _apply_diff_and_commit(
    wm, file_path: str, original_code: str, updated_code: str, commit_message: str
) -> Tuple[str, bool]:
    """
    Apply diff and commit changes.
    Returns tuple of (diff_summary, commit_succeeded).
    """
    full_path = wm.repo_root / file_path

    # 1. Generate Diff
    diff_lines = list(
        difflib.unified_diff(
            original_code.splitlines(),
            updated_code.splitlines(),
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            lineterm="",
        )
    )
    diff_summary = "\n".join(diff_lines)

    # 2. Apply Change
    full_path.parent.mkdir(parents=True, exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(updated_code)

    # 3. Git Commit
    try:
        from vulcan.settings import settings
        auto_commit_enabled = settings.self_improvement_auto_commit
    except (ImportError, AttributeError):
        auto_commit_enabled = False

    if not auto_commit_enabled:
        logger.info("Self-improvement auto-commit disabled (set VULCAN_SELF_IMPROVEMENT_AUTO_COMMIT=true to enable)")
        return diff_summary, False

    if not (wm.repo_root / ".git").exists():
        logger.warning(
            f"Cannot commit: {wm.repo_root} is not a Git repository. Skipping commit."
        )
        return diff_summary, False

    try:
        subprocess.run(  # nosec B603 B607
            ["git", "add", file_path],
            cwd=wm.repo_root,
            check=True,
            capture_output=True,
        )

        commit_result = subprocess.run(  # nosec B603 B607
            ["git", "commit", "-m", f"vulcan(auto): {commit_message}"],
            cwd=wm.repo_root,
            check=True,
            capture_output=True,
            text=True,
        )

        hash_result = subprocess.run(  # nosec B603 B607
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=wm.repo_root,
            check=True,
            capture_output=True,
            text=True,
        )

        logger.info(f"Git Commit successful: {hash_result.stdout.strip()}")
        return diff_summary, True

    except subprocess.CalledProcessError as e:
        if "nothing to commit" in e.stderr:
            logger.info(
                "Commit skipped: No functional changes detected by Git after writing."
            )
            return diff_summary, False
        logger.error(f"Git commit failed for {file_path}: {e.stderr}")
        raise RuntimeError(f"Git commit failed: {e.stderr}") from e
    except Exception as e:
        logger.error(f"Critical error during file application or Git: {e}")
        raise


def _execute_improvement(wm, improvement_action: Dict[str, Any]):
    """
    Execute an improvement action using the full LLM -> AST -> Diff -> Git pipeline.

    Note: External LLM code generation is DISABLED by VULCAN Policy.
    Improvements are deferred to human review instead of crashing.
    """
    from .self_improvement_engine import _build_llm_prompt_for_improvement

    objective_type = improvement_action.get("_drive_metadata", {}).get(
        "objective_type", "unknown"
    )
    high_level_goal = improvement_action.get("high_level_goal", objective_type)

    logger.info(
        f"Executing integrated improvement pipeline for: {objective_type}"
    )

    success = False
    result: Dict[str, Any] = {"status": "failed", "error": "Initialization error"}

    try:
        prompt = _build_llm_prompt_for_improvement(wm, improvement_action)

        improvement_type = improvement_action.get("type", "unknown")
        is_code_improvement = improvement_type in [
            "fix_bugs", "enhance_safety", "optimize_performance"
        ]

        if is_code_improvement:
            logger.warning(
                f"[Self-Improvement] External LLM code generation is DISABLED by VULCAN Policy. "
                f"Code improvement '{objective_type}' will be deferred for human review. "
                f"Non-code improvements (config, tests, docs) CAN proceed autonomously. "
                f"To enable code generation, implement internal templates or symbolic reasoning."
            )
        else:
            logger.info(
                f"[Self-Improvement] Non-code improvement '{objective_type}' can proceed autonomously. "
                f"External LLM code generation is DISABLED, but this improvement doesn't require it."
            )

        result = {
            "status": "deferred",
            "objective_type": objective_type,
            "reason": "external_llm_code_generation_disabled",
            "message": (
                "Self-improvement deferred: External LLM code generation is disabled "
                "by VULCAN Policy. The improvement has been queued for human review. "
                "OpenAI is only permitted for language interpretation, not code generation."
            ),
            "prompt_preview": prompt[:500] + "..." if len(prompt) > 500 else prompt,
            "execution_timestamp": time.time(),
            "requires_human_review": True,
        }

        logger.info(
            f"Improvement '{objective_type}' deferred for human review "
            f"(reason: external LLM code generation disabled)"
        )

        wm.self_improvement_drive.record_outcome(
            objective_type, success=False, details=result
        )

        return result

    except Exception as e:
        result["error"] = f"Improvement preparation failed: {str(e)}"
        logger.error(result["error"], exc_info=True)

        wm.self_improvement_drive.record_outcome(
            objective_type, success=False, details=result
        )
        return result
