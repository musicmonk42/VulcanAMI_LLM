"""
Cache and weight management for unified reasoning module.

This module provides thread-safe caching and tool weight management using
singleton patterns. It ensures weights are shared between the learning system
and ensemble reasoner to prevent the "weight propagation bug".

Following highest industry standards:
- Singleton pattern with double-checked locking
- Thread-safe operations with RLock
- Named constants for weight bounds
- Comprehensive logging
- Weight normalization
- Graceful degradation

Author: VulcanAMI Team
License: Proprietary
"""

import hashlib
import logging
import threading
from typing import Any, Dict, List, Optional

from .config import (
    CACHE_HASH_LENGTH,
    DEFAULT_TOOL_WEIGHT,
    MAX_TOOL_WEIGHT,
    MIN_TOOL_WEIGHT,
)

logger = logging.getLogger(__name__)


def compute_query_hash(query_data: Any) -> str:
    """
    Compute a consistent hash for query data.
    
    This function creates a deterministic hash from query data for use as
    cache keys. Uses SHA-256 with configurable truncation length.
    
    Args:
        query_data: The query data to hash (string, dict, or other)
        
    Returns:
        First CACHE_HASH_LENGTH chars of SHA-256 hex digest (default 32 chars)
        
    Examples:
        >>> hash1 = compute_query_hash("What is 2+2?")
        >>> hash2 = compute_query_hash("What is 2+2?")
        >>> hash1 == hash2
        True
        
        >>> hash3 = compute_query_hash({"question": "What is 2+2?"})
        >>> len(hash3)
        32
        
    Note:
        Uses CACHE_HASH_LENGTH = 32 (128 bits) to ensure extremely low
        collision probability. Birthday paradox: 50% collision after
        ~2^64 operations (sqrt of 2^128 hash space).
        
    Thread Safety:
        This function is thread-safe and reentrant.
    """
    query_str = str(query_data) if not isinstance(query_data, str) else query_data
    return hashlib.sha256(query_str.encode('utf-8')).hexdigest()[:CACHE_HASH_LENGTH]


class ToolWeightManager:
    """
    Singleton manager for tool weights shared between Learning and Ensemble.
    
    This class solves the "weight propagation bug" where the learning system
    was updating tool weights in its own dictionary, but the ensemble was
    reading from a separate dictionary, so weights never propagated.
    
    The singleton pattern ensures both systems use the same weight storage,
    with thread-safe operations and weight bounds to prevent "death spiral"
    where accumulated penalties cause tools to have zero or negative weights.
    
    Attributes:
        _weights: Internal dictionary mapping tool names to weight values
        _update_lock: RLock for thread-safe weight updates
        
    Examples:
        >>> manager = ToolWeightManager()
        >>> manager.set_weight("symbolic", 1.5)
        >>> manager.get_weight("symbolic")
        1.5
        
        >>> # Adjust weight by delta
        >>> manager.adjust_weight("symbolic", -0.1)
        >>> manager.get_weight("symbolic")
        1.4
        
        >>> # Get normalized weights for ensemble
        >>> weights = manager.get_all_weights(["symbolic", "causal", "analogical"])
        >>> sum(weights.values())
        1.0
        
    Note:
        Weight values are automatically floored at MIN_TOOL_WEIGHT (0.01) and
        capped at MAX_TOOL_WEIGHT (10.0) to prevent degenerate behavior.
        
        Default weight for unseen tools is DEFAULT_TOOL_WEIGHT (1.0), which
        represents neutral weight before learning adjustments.
        
    Thread Safety:
        All public methods are thread-safe using RLock for reentrancy support.
    """
    
    _instance: Optional['ToolWeightManager'] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'ToolWeightManager':
        """
        Create or return singleton instance using double-checked locking.
        
        Returns:
            The singleton ToolWeightManager instance
            
        Thread Safety:
            Uses double-checked locking pattern to ensure only one instance
            is created even with concurrent calls from multiple threads.
        """
        if cls._instance is None:
            with cls._lock:
                # Double-check locking
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._weights: Dict[str, float] = {}
                    instance._update_lock = threading.RLock()
                    cls._instance = instance
        return cls._instance
    
    def get_weight(self, tool: str, default: float = DEFAULT_TOOL_WEIGHT) -> float:
        """
        Get weight for a single tool.
        
        Returns the stored weight for the given tool, or the default weight
        if the tool hasn't been seen before. Using default=1.0 (neutral weight)
        ensures ensemble weighting works correctly even before learning has
        adjusted weights.
        
        Args:
            tool: Tool name (e.g., "symbolic", "causal", "probabilistic")
            default: Default weight for unseen tools (default: 1.0)
            
        Returns:
            Weight value (default if tool not in weights)
            
        Examples:
            >>> manager = ToolWeightManager()
            >>> manager.get_weight("new_tool")
            1.0
            
            >>> manager.set_weight("new_tool", 2.0)
            >>> manager.get_weight("new_tool")
            2.0
            
        Thread Safety:
            Thread-safe with RLock protection.
        """
        with self._update_lock:
            return self._weights.get(tool, default)
    
    def set_weight(self, tool: str, value: float) -> None:
        """
        Set absolute weight value for a tool.
        
        Sets the weight to the specified value, with automatic bounds checking
        to prevent degenerate behavior. Weights are floored at MIN_TOOL_WEIGHT
        and capped at MAX_TOOL_WEIGHT.
        
        Args:
            tool: Tool name
            value: New weight value
            
        Examples:
            >>> manager = ToolWeightManager()
            >>> manager.set_weight("symbolic", 1.5)
            >>> manager.get_weight("symbolic")
            1.5
            
            >>> # Negative weights are floored
            >>> manager.set_weight("tool", -1.0)
            >>> manager.get_weight("tool")
            0.01
            
        Note:
            Weights below MIN_TOOL_WEIGHT are floored to prevent the "death
            spiral" where accumulated penalties cause tools to have zero or
            negative weights, breaking ensemble calculations.
            
        Thread Safety:
            Thread-safe with RLock protection.
        """
        with self._update_lock:
            # Floor weight at minimum positive value
            if value < MIN_TOOL_WEIGHT:
                logger.warning(
                    f"[WeightManager] Flooring low weight for '{tool}': "
                    f"{value:.4f} -> {MIN_TOOL_WEIGHT}"
                )
                value = MIN_TOOL_WEIGHT
            
            # Cap weight at maximum to prevent dominance
            if value > MAX_TOOL_WEIGHT:
                logger.warning(
                    f"[WeightManager] Capping high weight for '{tool}': "
                    f"{value:.4f} -> {MAX_TOOL_WEIGHT}"
                )
                value = MAX_TOOL_WEIGHT
            
            self._weights[tool] = value
            logger.debug(f"[WeightManager] {tool} = {value:.4f}")
    
    def adjust_weight(self, tool: str, delta: float) -> None:
        """
        Adjust weight by delta (used by Learning system).
        
        Increments or decrements the tool's weight by the specified delta.
        If the tool hasn't been seen before, initializes from DEFAULT_TOOL_WEIGHT
        (1.0) instead of 0.0, so tools start with sensible weights before
        learning adjusts them.
        
        Args:
            tool: Tool name
            delta: Amount to adjust weight (positive or negative)
            
        Examples:
            >>> manager = ToolWeightManager()
            >>> manager.adjust_weight("symbolic", 0.1)
            >>> manager.get_weight("symbolic")
            1.1
            
            >>> manager.adjust_weight("symbolic", -0.2)
            >>> manager.get_weight("symbolic")
            0.9
            
        Note:
            Results are automatically bounded by MIN_TOOL_WEIGHT and
            MAX_TOOL_WEIGHT to prevent degenerate behavior.
            
        Thread Safety:
            Thread-safe with RLock protection.
        """
        with self._update_lock:
            # Start from DEFAULT_TOOL_WEIGHT (neutral) instead of 0.0
            current = self._weights.get(tool, DEFAULT_TOOL_WEIGHT)
            new_weight = current + delta
            
            # Apply bounds
            if new_weight < MIN_TOOL_WEIGHT:
                logger.warning(
                    f"[WeightManager] Flooring weight after adjustment for '{tool}': "
                    f"{current:.4f} + {delta:.4f} = {new_weight:.4f} -> {MIN_TOOL_WEIGHT}"
                )
                new_weight = MIN_TOOL_WEIGHT
            
            if new_weight > MAX_TOOL_WEIGHT:
                logger.warning(
                    f"[WeightManager] Capping weight after adjustment for '{tool}': "
                    f"{current:.4f} + {delta:.4f} = {new_weight:.4f} -> {MAX_TOOL_WEIGHT}"
                )
                new_weight = MAX_TOOL_WEIGHT
            
            self._weights[tool] = new_weight
            logger.info(f"[WeightManager] {tool}: {current:.4f} → {new_weight:.4f} (Δ{delta:+.4f})")
    
    def get_all_weights(self, tools: List[str]) -> Dict[str, float]:
        """
        Get normalized weights for multiple tools (used by Ensemble).
        
        Returns a dictionary of normalized weights that sum to 1.0. Uses
        DEFAULT_TOOL_WEIGHT (1.0) for unseen tools to ensure ensemble
        weighting works correctly even before learning has adjusted weights.
        
        Args:
            tools: List of tool names
            
        Returns:
            Dictionary mapping tool names to normalized weights (sum = 1.0)
            
        Examples:
            >>> manager = ToolWeightManager()
            >>> manager.set_weight("symbolic", 2.0)
            >>> manager.set_weight("causal", 1.0)
            >>> weights = manager.get_all_weights(["symbolic", "causal"])
            >>> weights
            {'symbolic': 0.6666..., 'causal': 0.3333...}
            >>> abs(sum(weights.values()) - 1.0) < 1e-10
            True
            
        Note:
            If all weights are zero (shouldn't happen with proper bounds),
            falls back to uniform weights (1/n for each tool).
            
        Thread Safety:
            Thread-safe with RLock protection.
        """
        with self._update_lock:
            # Use DEFAULT_TOOL_WEIGHT as default for unseen tools
            weights = {t: self._weights.get(t, DEFAULT_TOOL_WEIGHT) for t in tools}
            
            # Normalize if total > 0
            total = sum(weights.values())
            if total > 0:
                weights = {k: v / total for k, v in weights.items()}
            else:
                # If all zero (shouldn't happen with bounds), use uniform weights
                n = len(tools)
                if n > 0:
                    weights = {t: 1.0 / n for t in tools}
                else:
                    weights = {}
            
            return weights
    
    def get_raw_weights(self) -> Dict[str, float]:
        """
        Get raw (non-normalized) weights for debugging.
        
        Returns a copy of the internal weight dictionary without normalization.
        Useful for debugging and monitoring weight evolution.
        
        Returns:
            Dictionary mapping tool names to raw weight values
            
        Examples:
            >>> manager = ToolWeightManager()
            >>> manager.set_weight("symbolic", 2.0)
            >>> manager.set_weight("causal", 3.0)
            >>> manager.get_raw_weights()
            {'symbolic': 2.0, 'causal': 3.0}
            
        Thread Safety:
            Thread-safe with RLock protection. Returns a copy to prevent
            external modification of internal state.
        """
        with self._update_lock:
            return self._weights.copy()


# ==============================================================================
# GLOBAL SINGLETON ACCESSOR
# ==============================================================================

_weight_manager: Optional[ToolWeightManager] = None
_weight_manager_lock = threading.Lock()


def get_weight_manager() -> ToolWeightManager:
    """
    Get the singleton ToolWeightManager instance.
    
    This function provides thread-safe access to the global ToolWeightManager
    singleton. Uses double-checked locking to ensure only one instance is
    created even with concurrent access.
    
    Returns:
        The singleton ToolWeightManager instance
        
    Examples:
        >>> # In Learning system
        >>> from vulcan.reasoning.unified.cache import get_weight_manager
        >>> get_weight_manager().adjust_weight("causal", 0.01)
        
        >>> # In Ensemble system
        >>> from vulcan.reasoning.unified.cache import get_weight_manager
        >>> weights = get_weight_manager().get_all_weights(["causal", "symbolic"])
        
    Note:
        The ToolWeightManager class also uses __new__ for singleton pattern,
        but this accessor adds an additional layer of protection against race
        conditions when multiple threads call get_weight_manager() simultaneously.
        
    Thread Safety:
        Thread-safe with double-checked locking pattern.
    """
    global _weight_manager
    
    # Double-checked locking pattern
    if _weight_manager is None:
        with _weight_manager_lock:
            if _weight_manager is None:
                _weight_manager = ToolWeightManager()
    
    return _weight_manager
