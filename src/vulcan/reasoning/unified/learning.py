"""
Learning integration for unified reasoning orchestration.

Learns from reasoning results with mathematical accuracy integration,
connecting the learning system to mathematical verification results.

Extracted from orchestrator.py for modularity.

Author: VulcanAMI Team
"""

import logging
import time
from typing import Any

from .cache import get_weight_manager
from .config import (
    MATH_ACCURACY_PENALTY,
    MATH_ACCURACY_REWARD,
    MATH_WEIGHT_ADJUSTMENT_PENALTY,
)
from .types import ReasoningTask
from ..reasoning_types import ReasoningResult

logger = logging.getLogger(__name__)


def learn_from_reasoning(
    reasoner: Any, task: ReasoningTask, result: ReasoningResult
) -> None:
    """
    Learn from reasoning result with mathematical accuracy integration.

    Args:
        reasoner: UnifiedReasoner instance.
        task: The completed reasoning task.
        result: The reasoning result.
    """
    if not reasoner.learner:
        return

    try:
        learning_data = {
            "task": task,
            "result": result,
            "timestamp": time.time(),
        }

        if hasattr(result, 'metadata') and result.metadata:
            math_verification = result.metadata.get('math_verification')
            if math_verification:
                learning_data['math_verification'] = math_verification

                verification_status = math_verification.get(
                    'status', 'unknown'
                )
                if verification_status == 'verified':
                    learning_data['math_accuracy_bonus'] = (
                        MATH_ACCURACY_REWARD
                    )
                    learning_data['learning_signal'] = 'positive'
                    logger.info(
                        "[Learning] Mathematical accuracy reward applied "
                        f"for tool "
                        f"{task.task_type.value if task.task_type else 'unknown'}"
                    )
                elif verification_status == 'error_detected':
                    learning_data['math_accuracy_penalty'] = (
                        MATH_ACCURACY_PENALTY
                    )
                    learning_data['learning_signal'] = 'negative'
                    learning_data['errors'] = math_verification.get(
                        'errors', []
                    )
                    logger.info(
                        "[Learning] Mathematical accuracy penalty applied "
                        f"for tool "
                        f"{task.task_type.value if task.task_type else 'unknown'}, "
                        f"errors: {math_verification.get('errors', [])}"
                    )

                    tool_name = (
                        task.task_type.value
                        if task.task_type
                        else "unknown"
                    )
                    weight_manager = get_weight_manager()
                    weight_manager.adjust_weight(
                        tool_name, MATH_WEIGHT_ADJUSTMENT_PENALTY
                    )

        reasoner.learner.update(learning_data)
    except Exception as e:
        logger.warning(f"Learning update failed: {e}")
