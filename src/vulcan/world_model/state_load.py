"""
state_load.py - Load world model state from disk.

Extracted from world_model_core.py to reduce class size.
Functions take `wm` (WorldModel instance) as first parameter.
"""

import json
import logging
from pathlib import Path as FilePath

logger = logging.getLogger(__name__)


def load_state(wm, path: str):
    """Load world model state from disk"""

    # Note: Use the 'FilePath' alias for pathlib.Path
    load_path = FilePath(path)

    if not load_path.exists():
        logger.warning("No saved state found at %s", load_path)
        return

    with open(load_path / "world_model_state.json", "r", encoding="utf-8") as f:
        state = json.load(f)

    wm.model_version = state["model_version"]
    wm.observation_count = state["observation_count"]

    # Load router state
    if wm.router:
        wm.router.load_state(str(load_path))

    # Load safety validator state if available
    if wm.safety_validator and (load_path / "safety_state.json").exists():
        try:
            with open(load_path / "safety_state.json", "r", encoding="utf-8") as f:
                json.load(f)
            logger.info("Safety validator state loaded")
        except Exception as e:
            logger.error("Error loading safety state: %s", e)

    # Load meta-reasoning state if available
    if (
        wm.meta_reasoning_enabled
        and (load_path / "meta_reasoning_state.json").exists()
    ):
        try:
            with open(
                load_path / "meta_reasoning_state.json", "r", encoding="utf-8"
            ) as f:
                json.load(f)
            logger.info("Meta-reasoning state loaded")
        except Exception as e:
            logger.error("Error loading meta-reasoning state: %s", e)

    # Self-improvement state loaded by drive itself

    logger.info("World model state loaded from %s", load_path)
