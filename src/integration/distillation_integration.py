"""
VULCAN Distillation System Integration Module.

Single entry point for knowledge distillation with ensemble mode. This module
connects existing components (HybridLLMExecutor, OpenAIKnowledgeDistiller) into
a unified interface for application integration.

Usage:
    from src.integration.distillation_integration import DistillationSystem
    
    # Initialize once at startup
    system = DistillationSystem(graphix_llm)
    
    # Use in request handler
    response = await system.execute(prompt, user_opted_in=True)
    
    # Check status
    status = system.get_status()

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │               DISTILLATION SYSTEM                               │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  DistillationSystem.execute(prompt)                             │
    │         │                                                       │
    │         ▼                                                       │
    │  HybridLLMExecutor (ensemble mode)                              │
    │     │                   │                                       │
    │     ▼                   ▼                                       │
    │  Local LLM        OpenAI API                                    │
    │     │                   │                                       │
    │     └───────┬───────────┘                                       │
    │             │                                                   │
    │             ▼                                                   │
    │  Response Selection + OpenAIKnowledgeDistiller                  │
    │             │                                                   │
    │             ▼                                                   │
    │  DistillationStorageBackend (JSONL)                             │
    │             │                                                   │
    │             ▼                                                   │
    │  Training Worker reads examples                                 │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
"""

import asyncio
import logging
import os
import sys
from typing import Any, Callable, Dict, List, Optional

# Add src to path for imports
_here = os.path.dirname(os.path.abspath(__file__))
_src = os.path.dirname(_here)
if _src not in sys.path:
    sys.path.insert(0, _src)

logger = logging.getLogger(__name__)

# ============================================================
# IMPORTS - Use ONLY existing components
# ============================================================

try:
    from vulcan.distillation import (
        initialize_knowledge_distiller,
        get_knowledge_distiller,
        validate_distillation_module,
    )
    DISTILLATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Distillation module not available: {e}")
    DISTILLATION_AVAILABLE = False

try:
    from vulcan.llm.hybrid_executor import HybridLLMExecutor
    EXECUTOR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"HybridLLMExecutor not available: {e}")
    EXECUTOR_AVAILABLE = False

try:
    from vulcan.llm.openai_client import get_openai_client
    OPENAI_CLIENT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"OpenAI client not available: {e}")
    OPENAI_CLIENT_AVAILABLE = False
    def get_openai_client():
        return None


# ============================================================
# DISTILLATION SYSTEM CLASS
# ============================================================

class DistillationSystem:
    """
    Unified interface for VULCAN distillation system.
    
    Handles:
    - Distiller initialization (5-stage filtering)
    - Hybrid executor setup (ensemble mode by default)
    - Request execution with automatic distillation capture
    - Status monitoring and statistics
    
    Attributes:
        llm: The local GraphixVulcanLLM instance
        mode: Execution mode (ensemble, local_first, openai_first, parallel)
        distiller: OpenAIKnowledgeDistiller instance
        executor: HybridLLMExecutor instance
        config: Configuration dictionary
        
    Example:
        >>> system = DistillationSystem(my_llm, mode="ensemble")
        >>> result = await system.execute("What is AI?", user_opted_in=True)
        >>> print(result["text"])
        >>> print(result["source"])  # "ensemble", "openai", or "local"
    """

    def __init__(
        self,
        graphix_vulcan_llm: Any,
        mode: str = "ensemble",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize complete distillation system.
        
        Args:
            graphix_vulcan_llm: GraphixVulcanLLM instance (local model)
            mode: Execution mode (ensemble, local_first, openai_first, parallel)
            config: Optional configuration overrides
            
        Raises:
            RuntimeError: If distillation module validation fails
            ImportError: If required components are not available
        """
        self.llm = graphix_vulcan_llm
        self.mode = mode
        self.config = {**self._default_config(), **(config or {})}
        
        # Validate imports
        if not DISTILLATION_AVAILABLE:
            raise ImportError(
                "Distillation module not available. "
                "Ensure vulcan.distillation package is installed."
            )
        
        if not EXECUTOR_AVAILABLE:
            raise ImportError(
                "HybridLLMExecutor not available. "
                "Ensure vulcan.llm.hybrid_executor is installed."
            )
        
        # Validate distillation module
        if not validate_distillation_module():
            logger.warning(
                "Distillation module validation failed - some components may be unavailable"
            )
        
        # Initialize distiller (5-stage filtering: PII, secrets, governance, quality, opt-in)
        self.distiller = initialize_knowledge_distiller(
            local_llm=graphix_vulcan_llm,
            storage_path=self.config["storage_path"],
            max_buffer_size=self.config["max_buffer_size"],
            retention_days=self.config["retention_days"],
            require_opt_in=self.config["require_opt_in"],
            enable_pii_redaction=self.config["enable_pii_redaction"],
            enable_governance_check=self.config["enable_governance_check"],
        )
        
        # Initialize hybrid executor (ensemble mode enables both LLMs)
        self.executor = HybridLLMExecutor(
            local_llm=graphix_vulcan_llm,
            openai_client_getter=get_openai_client if OPENAI_CLIENT_AVAILABLE else None,
            mode=mode,
            timeout=self.config["executor_timeout"],
            openai_max_tokens=self.config["openai_max_tokens"],
        )
        
        logger.info(
            f"✓ Distillation system initialized ("
            f"mode={mode}, "
            f"opt_in_required={self.config['require_opt_in']}, "
            f"pii_redaction={self.config['enable_pii_redaction']})"
        )

    async def execute(
        self,
        prompt: str,
        user_opted_in: bool = False,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        enable_distillation: bool = True,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Execute request with distillation enabled.
        
        In ensemble mode:
        1. Both local LLM and OpenAI are called
        2. Responses are combined/selected based on quality
        3. If user opted in, OpenAI response is captured for training
        
        Args:
            prompt: User's query
            user_opted_in: Whether user consented to training data capture
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system_prompt: Optional system prompt (uses default if None)
            enable_distillation: Whether to capture for training
            conversation_history: Optional conversation history for context
            
        Returns:
            Dictionary with:
            - text: Generated response
            - source: Response source (ensemble, openai, local)
            - systems_used: List of systems that contributed
            - metadata: Additional execution metadata
        """
        # Only enable distillation if user opted in
        should_distill = enable_distillation and user_opted_in
        
        result = await self.executor.execute(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
            enable_distillation=should_distill,
            conversation_history=conversation_history,
        )
        
        # Add distillation metadata
        result["distillation_enabled"] = should_distill
        result["user_opted_in"] = user_opted_in
        
        logger.debug(
            f"Request executed: source={result.get('source')}, "
            f"systems={result.get('systems_used')}, "
            f"distillation={'captured' if should_distill else 'skipped'}"
        )
        
        return result

    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status.
        
        Returns:
            Dictionary with:
            - mode: Current execution mode
            - distiller: Distiller status and statistics
            - executor: Executor statistics (if available)
            - config: Current configuration
        """
        distiller_status = {}
        if self.distiller:
            try:
                distiller_status = self.distiller.get_status()
            except Exception as e:
                distiller_status = {"error": str(e)}
        
        executor_stats = {}
        if hasattr(self.executor, 'get_stats'):
            try:
                executor_stats = self.executor.get_stats()
            except Exception as e:
                executor_stats = {"error": str(e)}
        
        return {
            "mode": self.mode,
            "distiller": distiller_status,
            "executor": executor_stats,
            "config": {
                k: v for k, v in self.config.items()
                if k != "encryption_key"  # Don't expose secrets
            },
            "components_available": {
                "distillation": DISTILLATION_AVAILABLE,
                "executor": EXECUTOR_AVAILABLE,
                "openai_client": OPENAI_CLIENT_AVAILABLE,
            },
        }

    def set_mode(self, mode: str) -> None:
        """
        Change execution mode dynamically.
        
        Args:
            mode: New execution mode (ensemble, local_first, openai_first, parallel)
        """
        valid_modes = ("ensemble", "local_first", "openai_first", "parallel")
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}. Must be one of: {valid_modes}")
        
        self.executor.mode = mode
        self.mode = mode
        logger.info(f"Execution mode changed to: {mode}")

    @staticmethod
    def _default_config() -> Dict[str, Any]:
        """Default configuration for distillation system."""
        return {
            # Storage
            "storage_path": "data/distillation/examples.jsonl",
            "max_buffer_size": 100,
            "retention_days": 30,
            
            # Privacy & Security
            "require_opt_in": True,
            "enable_pii_redaction": True,
            "enable_governance_check": True,
            
            # Executor
            "executor_timeout": 30.0,
            "openai_max_tokens": 2000,
        }


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def initialize_distillation_system(
    graphix_vulcan_llm: Any,
    mode: str = "ensemble",
    **config_overrides,
) -> DistillationSystem:
    """
    Quick initialization of distillation system.
    
    Args:
        graphix_vulcan_llm: Local LLM instance
        mode: Execution mode (default: ensemble)
        **config_overrides: Configuration overrides
        
    Returns:
        Initialized DistillationSystem
        
    Example:
        >>> system = initialize_distillation_system(llm, mode="ensemble")
        >>> result = await system.execute("What is AI?", user_opted_in=True)
    """
    return DistillationSystem(
        graphix_vulcan_llm=graphix_vulcan_llm,
        mode=mode,
        config=config_overrides if config_overrides else None,
    )


def check_system_requirements() -> Dict[str, bool]:
    """
    Check if all required components are available.
    
    Returns:
        Dictionary with component availability status
    """
    return {
        "distillation_module": DISTILLATION_AVAILABLE,
        "hybrid_executor": EXECUTOR_AVAILABLE,
        "openai_client": OPENAI_CLIENT_AVAILABLE,
        "all_required": DISTILLATION_AVAILABLE and EXECUTOR_AVAILABLE,
    }
