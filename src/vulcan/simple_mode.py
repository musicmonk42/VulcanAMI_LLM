# ============================================================
# VULCAN-AGI Simple Mode Configuration
# Disables heavy components for fast chat responses
# Performance optimization: reduces startup time and CPU usage
# ============================================================
"""
Simple mode configuration - disables heavy components for basic chat.

When VULCAN_SIMPLE_MODE=true, the system operates in a lightweight mode
optimized for fast chat responses with minimal resource usage.

Environment Variables:
    VULCAN_SIMPLE_MODE: Enable simple mode (default: false)
    SKIP_BERT_EMBEDDINGS: Skip BERT model loading (default: follows SIMPLE_MODE)
    OPENAI_ONLY_MODE: Use only OpenAI for inference (default: follows SIMPLE_MODE)
    DISABLE_SELF_IMPROVEMENT: Disable self-improvement system (default: follows SIMPLE_MODE)
    MIN_AGENTS: Minimum number of agents (default: 1 in simple mode, 10 otherwise)
    MAX_AGENTS: Maximum number of agents (default: 5 in simple mode, 100 otherwise)
    AGENT_CHECK_INTERVAL: Agent status check interval in seconds (default: 300 in simple mode)
    MAX_PROVENANCE_RECORDS: Maximum provenance records to keep (default: 50 in simple mode)
"""

import logging
import os

logger = logging.getLogger(__name__)


def _str_to_bool(value: str | None, default: bool = False) -> bool:
    """Convert string to boolean value.
    
    Args:
        value: String value to convert, or None.
        default: Default value if value is None.
        
    Returns:
        Boolean value.
    """
    if value is None:
        return default
    return value.lower() in ("true", "1", "yes", "on")


# ============================================================
# SIMPLE MODE FLAG
# ============================================================

SIMPLE_MODE = _str_to_bool(os.getenv("VULCAN_SIMPLE_MODE", "false"))

# ============================================================
# COMPONENT FLAGS (follow SIMPLE_MODE by default)
# ============================================================

# Skip BERT embeddings - use OpenAI directly (saves 3.5s+ per request)
SKIP_BERT_EMBEDDINGS = _str_to_bool(
    os.getenv("SKIP_BERT_EMBEDDINGS"),
    default=SIMPLE_MODE
)

# Use only OpenAI for inference
OPENAI_ONLY_MODE = _str_to_bool(
    os.getenv("OPENAI_ONLY_MODE"),
    default=SIMPLE_MODE
)

# Disable self-improvement system
DISABLE_SELF_IMPROVEMENT = _str_to_bool(
    os.getenv("DISABLE_SELF_IMPROVEMENT"),
    default=SIMPLE_MODE
)

# Disable world model for faster responses
ENABLE_WORLD_MODEL = not _str_to_bool(
    os.getenv("DISABLE_WORLD_MODEL"),
    default=SIMPLE_MODE
)

# Disable meta-reasoning for faster responses
ENABLE_META_REASONING = not _str_to_bool(
    os.getenv("DISABLE_META_REASONING"),
    default=SIMPLE_MODE
)

# ============================================================
# AGENT POOL CONFIGURATION
# ============================================================

# Default agent pool sizes
# CPU OPTIMIZATION: Reduced from 10/100 to 2/10 to prevent CPU thrashing
# CPU CLOUD FIX: Reduced min_agents from 5 to 2 to reduce context-switching
# overhead on CPU-only cloud instances where agents compete for CPU resources.
DEFAULT_MIN_AGENTS = int(os.getenv("MIN_AGENTS", "1" if SIMPLE_MODE else "2"))
DEFAULT_MAX_AGENTS = int(os.getenv("MAX_AGENTS", "5" if SIMPLE_MODE else "10"))

# Agent status check interval (seconds)
# Increase to 300s (5 minutes) in simple mode to reduce CPU usage
AGENT_CHECK_INTERVAL = int(
    os.getenv("AGENT_CHECK_INTERVAL", "300" if SIMPLE_MODE else "30")
)

# ============================================================
# MEMORY MANAGEMENT
# ============================================================

# Maximum provenance records to prevent unbounded memory growth
MAX_PROVENANCE_RECORDS = int(
    os.getenv("MAX_PROVENANCE_RECORDS", "50" if SIMPLE_MODE else "1000")
)

# Provenance TTL in seconds (1 hour default)
PROVENANCE_TTL_SECONDS = int(
    os.getenv("PROVENANCE_TTL_SECONDS", "3600")
)

# ============================================================
# INITIALIZATION LOGGING
# ============================================================

if SIMPLE_MODE:
    logger.info("🚀 VULCAN Simple Mode: Optimized for fast chat responses")
    logger.info(f"  - SKIP_BERT_EMBEDDINGS: {SKIP_BERT_EMBEDDINGS}")
    logger.info(f"  - OPENAI_ONLY_MODE: {OPENAI_ONLY_MODE}")
    logger.info(f"  - DISABLE_SELF_IMPROVEMENT: {DISABLE_SELF_IMPROVEMENT}")
    logger.info(f"  - Agent pool: {DEFAULT_MIN_AGENTS}-{DEFAULT_MAX_AGENTS} agents")
    logger.info(f"  - Max provenance records: {MAX_PROVENANCE_RECORDS}")
else:
    logger.debug("VULCAN running in full mode (all features enabled)")


def get_simple_mode_config() -> dict:
    """Get simple mode configuration as a dictionary."""
    return {
        "simple_mode": SIMPLE_MODE,
        "skip_bert_embeddings": SKIP_BERT_EMBEDDINGS,
        "openai_only_mode": OPENAI_ONLY_MODE,
        "disable_self_improvement": DISABLE_SELF_IMPROVEMENT,
        "enable_world_model": ENABLE_WORLD_MODEL,
        "enable_meta_reasoning": ENABLE_META_REASONING,
        "min_agents": DEFAULT_MIN_AGENTS,
        "max_agents": DEFAULT_MAX_AGENTS,
        "agent_check_interval": AGENT_CHECK_INTERVAL,
        "max_provenance_records": MAX_PROVENANCE_RECORDS,
        "provenance_ttl_seconds": PROVENANCE_TTL_SECONDS,
    }


def is_simple_mode() -> bool:
    """Check if simple mode is enabled."""
    return SIMPLE_MODE


def should_skip_bert() -> bool:
    """Check if BERT embeddings should be skipped."""
    return SKIP_BERT_EMBEDDINGS


def should_skip_self_improvement() -> bool:
    """Check if self-improvement should be skipped."""
    return DISABLE_SELF_IMPROVEMENT
