"""
Bridge between tool selection and semantic domains.

Maps tools to domains for cross-domain knowledge transfer using the SemanticBridge module.
This enables the reasoning system to:
- Identify when queries span multiple domains
- Determine transfer candidates between domains
- Establish execution order for multi-domain queries
- Use SemanticBridge for actual concept transfer operations

Architecture:
    ToolSelector → ToolDomainBridge → SemanticBridge
                        ↓
                   Domain Mapping
                        ↓
                   Cross-Domain Transfer
"""

import logging
import threading
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Try to import SemanticBridge components
SEMANTIC_BRIDGE_AVAILABLE = False
try:
    from vulcan.semantic_bridge import (
        SemanticBridge, 
        DomainRegistry, 
        create_semantic_bridge,
        DomainProfile,
        DomainCriticality,
    )
    SEMANTIC_BRIDGE_AVAILABLE = True
except ImportError as e:
    logger.debug(f"SemanticBridge not available: {e}")
    # Will use fallback static mappings


class ToolDomainBridge:
    """
    Maps tools to semantic domains and identifies cross-domain queries.
    
    This bridge enables the reasoning system to leverage cross-domain
    knowledge transfer when queries use multiple tools from different
    conceptual domains.
    
    Attributes:
        TOOL_TO_DOMAIN: Mapping of tool names to their primary domains
        DOMAIN_RELATIONSHIPS: Which domains can transfer knowledge to which
        
    Example:
        >>> bridge = ToolDomainBridge()
        >>> domains = bridge.get_domains_for_tools(['probabilistic', 'causal'])
        >>> print(domains)  # {'statistical', 'causal_reasoning'}
        >>> print(bridge.is_cross_domain_query(['probabilistic', 'causal']))  # True
    """
    
    # Map tools to their primary semantic domains
    TOOL_TO_DOMAIN: Dict[str, str] = {
        # Probabilistic/Statistical reasoning
        'probabilistic': 'statistical',
        'bayesian': 'statistical',
        
        # Causal reasoning
        'causal': 'causal_reasoning',
        'counterfactual': 'causal_reasoning',
        
        # Logical/Symbolic reasoning
        'symbolic': 'logical',
        'deductive': 'logical',
        'formal': 'logical',
        
        # Perceptual/Multimodal
        'multimodal': 'perceptual',
        'visual': 'perceptual',
        'audio': 'perceptual',
        
        # Analogical reasoning
        'analogical': 'analogical',
        'analogy': 'analogical',
        
        # General/Fallback
        'general': 'general',
        'default': 'general',
    }
    
    # Define domain relationships (which domains can transfer knowledge to which)
    # Each domain maps to a list of domains it can transfer TO
    DOMAIN_RELATIONSHIPS: Dict[str, List[str]] = {
        'statistical': ['causal_reasoning', 'general', 'analogical'],
        'causal_reasoning': ['statistical', 'logical', 'general'],
        'logical': ['causal_reasoning', 'general', 'analogical'],
        'perceptual': ['analogical', 'general'],
        'analogical': ['perceptual', 'logical', 'general', 'statistical'],
        'general': ['statistical', 'causal_reasoning', 'logical', 'perceptual', 'analogical'],
    }
    
    # Domain priority for execution ordering (lower = higher priority)
    DOMAIN_PRIORITY: Dict[str, int] = {
        'logical': 1,
        'statistical': 2,
        'causal_reasoning': 3,
        'perceptual': 4,
        'analogical': 5,
        'general': 6,
    }
    
    def __init__(self, semantic_bridge: Optional[Any] = None):
        """Initialize the tool-to-domain bridge.
        
        Args:
            semantic_bridge: Optional SemanticBridge instance for actual transfers.
                            If not provided, will try to get from singletons.
        """
        self.logger = logging.getLogger(f"{__name__}.ToolDomainBridge")
        self._transfer_history: List[Dict[str, Any]] = []
        self._max_history = 100
        self._history_lock = threading.Lock()  # Thread safety for transfer history
        
        # Initialize SemanticBridge integration
        self._semantic_bridge = semantic_bridge
        self._domain_registry: Optional[Any] = None
        self._init_semantic_bridge()
    
    def _init_semantic_bridge(self):
        """Initialize SemanticBridge integration.
        
        This enables proper cross-domain concept transfer using the
        semantic_bridge module instead of just static mappings.
        """
        if self._semantic_bridge is not None:
            self.logger.info("ToolDomainBridge using provided SemanticBridge instance")
            return
        
        # Try to get from singletons
        if SEMANTIC_BRIDGE_AVAILABLE:
            try:
                from vulcan.reasoning.singletons import get_semantic_bridge
                self._semantic_bridge = get_semantic_bridge()
                if self._semantic_bridge:
                    self.logger.info("ToolDomainBridge connected to singleton SemanticBridge")
                    # Get domain registry from semantic bridge
                    if hasattr(self._semantic_bridge, 'domain_registry'):
                        self._domain_registry = self._semantic_bridge.domain_registry
                    return
            except Exception as e:
                self.logger.debug(f"Could not get SemanticBridge from singletons: {e}")
        
        # Fallback: Try direct creation
        if SEMANTIC_BRIDGE_AVAILABLE:
            try:
                self._semantic_bridge = create_semantic_bridge()
                self.logger.info("ToolDomainBridge created new SemanticBridge instance")
                if hasattr(self._semantic_bridge, 'domain_registry'):
                    self._domain_registry = self._semantic_bridge.domain_registry
                return
            except Exception as e:
                self.logger.warning(f"Could not create SemanticBridge: {e}")
        
        self.logger.info("ToolDomainBridge running without SemanticBridge (using static mappings)")
    
    def get_semantic_bridge(self) -> Optional[Any]:
        """Get the SemanticBridge instance if available.
        
        Returns:
            SemanticBridge instance or None if not available
        """
        return self._semantic_bridge
    
    def transfer_concepts(
        self,
        source_domain: str,
        target_domain: str,
        concepts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Transfer concepts between domains using SemanticBridge.
        
        This method uses the actual SemanticBridge for concept transfer
        if available, otherwise falls back to a simple copy operation.
        
        Args:
            source_domain: Source domain name
            target_domain: Target domain name
            concepts: List of concept dictionaries to transfer
            
        Returns:
            Dict with transfer results including success status
        """
        if not self.can_transfer_between(source_domain, target_domain):
            return {
                'success': False,
                'error': f"Transfer not allowed: {source_domain} → {target_domain}",
                'transferred_count': 0,
            }
        
        # Use SemanticBridge if available
        if self._semantic_bridge is not None:
            try:
                # Use the transfer engine from semantic bridge
                if hasattr(self._semantic_bridge, 'transfer_engine'):
                    transfer_results = []
                    for concept in concepts:
                        result = self._semantic_bridge.transfer_engine.transfer(
                            concept=concept,
                            source_domain=source_domain,
                            target_domain=target_domain,
                        )
                        transfer_results.append(result)
                    
                    # Robust success detection - handle various result types
                    def is_successful_result(r) -> bool:
                        """Check if a transfer result indicates success."""
                        if r is None:
                            return False
                        # Check for success attribute (TransferDecision objects)
                        if hasattr(r, 'success'):
                            return bool(r.success)
                        # Check for dict with success key
                        if isinstance(r, dict):
                            return r.get('success', False) or r.get('transferred', False)
                        # Non-None result without explicit failure assumed successful
                        return True
                    
                    successful = sum(1 for r in transfer_results if is_successful_result(r))
                    
                    # Record the transfer
                    self.record_transfer(source_domain, target_domain, successful > 0, successful)
                    
                    return {
                        'success': successful > 0,
                        'transferred_count': successful,
                        'total_attempted': len(concepts),
                        'results': transfer_results,
                    }
                else:
                    self.logger.debug("SemanticBridge has no transfer_engine, using fallback")
            except Exception as e:
                self.logger.warning(f"SemanticBridge transfer failed: {e}, using fallback")
        
        # Fallback: Simple pass-through (no actual transformation)
        self.record_transfer(source_domain, target_domain, True, len(concepts))
        
        return {
            'success': True,
            'transferred_count': len(concepts),
            'total_attempted': len(concepts),
            'method': 'fallback_passthrough',
        }
    
    def get_domains_for_tools(self, tools: List[str]) -> Set[str]:
        """
        Get the set of domains for a list of tools.
        
        Args:
            tools: List of tool names
            
        Returns:
            Set of domain names corresponding to the tools
            
        Example:
            >>> bridge.get_domains_for_tools(['probabilistic', 'causal'])
            {'statistical', 'causal_reasoning'}
        """
        domains = set()
        for tool in tools:
            # Normalize tool name
            tool_lower = tool.lower().strip()
            domain = self.TOOL_TO_DOMAIN.get(tool_lower, 'general')
            domains.add(domain)
        
        self.logger.debug(f"Tools {tools} mapped to domains {domains}")
        return domains
    
    def is_cross_domain_query(self, tools: List[str]) -> bool:
        """
        Check if a query spans multiple semantic domains.
        
        A query is cross-domain if it uses tools from more than one
        specific domain (excluding 'general' which is universal).
        
        Args:
            tools: List of selected tool names
            
        Returns:
            True if query uses tools from multiple specific domains
            
        Example:
            >>> bridge.is_cross_domain_query(['probabilistic', 'causal'])
            True
            >>> bridge.is_cross_domain_query(['probabilistic', 'general'])
            False
        """
        domains = self.get_domains_for_tools(tools)
        # Filter out 'general' for this check - it's a universal fallback
        specific_domains = domains - {'general'}
        
        is_cross = len(specific_domains) > 1
        
        if is_cross:
            self.logger.debug(
                f"Cross-domain query detected: {specific_domains}"
            )
        
        return is_cross
    
    def can_transfer_between(
        self,
        source_domain: str,
        target_domain: str,
    ) -> bool:
        """
        Check if knowledge transfer is possible between two domains.
        
        Args:
            source_domain: Source domain name
            target_domain: Target domain name
            
        Returns:
            True if transfer is possible, False otherwise
        """
        # Same domain always allows "transfer" (no-op)
        if source_domain == target_domain:
            return True
        
        # Check relationship mapping
        related = self.DOMAIN_RELATIONSHIPS.get(source_domain, [])
        can_transfer = target_domain in related
        
        self.logger.debug(
            f"Transfer {source_domain} → {target_domain}: {can_transfer}"
        )
        
        return can_transfer
    
    def get_transfer_candidates(
        self,
        source_domain: str,
    ) -> List[str]:
        """
        Get all domains that source_domain can transfer to.
        
        Args:
            source_domain: Source domain name
            
        Returns:
            List of target domain names that can receive transfers
        """
        return self.DOMAIN_RELATIONSHIPS.get(source_domain, [])
    
    def identify_primary_domain(
        self,
        tools: List[str],
        query_type: str,
    ) -> str:
        """
        Identify the primary domain for a query.
        
        The primary domain is determined by:
        1. Query type preference (if specified)
        2. First tool's domain (if specific)
        3. Domain with highest priority
        4. 'general' as fallback
        
        Args:
            tools: List of selected tools
            query_type: Query type from analysis ('reasoning', 'perception', etc.)
            
        Returns:
            Primary domain name
            
        Example:
            >>> bridge.identify_primary_domain(['causal', 'probabilistic'], 'reasoning')
            'causal_reasoning'
        """
        # Map query types to preferred domains
        type_to_domain = {
            'reasoning': 'logical',
            'causal': 'causal_reasoning',
            'perception': 'perceptual',
            'execution': 'general',
            'planning': 'logical',
            'learning': 'statistical',
            'general': 'general',
        }
        
        # Get domains for all tools
        domains = self.get_domains_for_tools(tools)
        specific_domains = domains - {'general'}
        
        # If only one specific domain, use it
        if len(specific_domains) == 1:
            return list(specific_domains)[0]
        
        # Check query type preference
        if query_type:
            preferred = type_to_domain.get(query_type.lower(), 'general')
            if preferred in domains:
                return preferred
        
        # If first tool maps to a specific domain, prefer that
        if tools:
            first_tool_domain = self.TOOL_TO_DOMAIN.get(tools[0].lower(), 'general')
            if first_tool_domain != 'general':
                return first_tool_domain
        
        # Use highest priority domain from available
        if specific_domains:
            return min(
                specific_domains,
                key=lambda d: self.DOMAIN_PRIORITY.get(d, 100)
            )
        
        return 'general'
    
    def get_domain_execution_order(
        self,
        domains: Set[str],
        primary_domain: str,
    ) -> List[str]:
        """
        Get optimal execution order for domains.
        
        Execution order considers:
        1. Primary domain first
        2. Foundational domains before dependent ones
        3. Priority ordering for remaining
        
        Args:
            domains: Set of domains to execute
            primary_domain: Primary domain for the query
            
        Returns:
            Ordered list of domains for execution
            
        Example:
            >>> bridge.get_domain_execution_order(
            ...     {'statistical', 'causal_reasoning', 'general'},
            ...     'causal_reasoning'
            ... )
            ['causal_reasoning', 'statistical', 'general']
        """
        # Start with primary domain
        order = []
        if primary_domain in domains:
            order.append(primary_domain)
        
        # Get remaining domains
        remaining = domains - {primary_domain}
        
        # Sort remaining by priority
        sorted_remaining = sorted(
            remaining,
            key=lambda d: self.DOMAIN_PRIORITY.get(d, 100)
        )
        
        order.extend(sorted_remaining)
        
        self.logger.debug(
            f"Execution order: {order} (primary={primary_domain})"
        )
        
        return order
    
    def get_bidirectional_transfers(
        self,
        domains: Set[str],
    ) -> List[Tuple[str, str]]:
        """
        Get all valid transfer pairs between a set of domains.
        
        Args:
            domains: Set of domain names
            
        Returns:
            List of (source, target) tuples for valid transfers
        """
        transfers = []
        domain_list = list(domains)
        
        for i, source in enumerate(domain_list):
            for target in domain_list[i + 1:]:
                # Check both directions
                if self.can_transfer_between(source, target):
                    transfers.append((source, target))
                if self.can_transfer_between(target, source):
                    transfers.append((target, source))
        
        return transfers
    
    def record_transfer(
        self,
        source_domain: str,
        target_domain: str,
        success: bool,
        concepts_transferred: int,
    ) -> None:
        """
        Record a transfer for history tracking.
        
        Args:
            source_domain: Source domain
            target_domain: Target domain
            success: Whether transfer was successful
            concepts_transferred: Number of concepts transferred
        """
        import time
        
        record = {
            'source': source_domain,
            'target': target_domain,
            'success': success,
            'concepts': concepts_transferred,
            'timestamp': time.time(),
        }
        
        with self._history_lock:
            self._transfer_history.append(record)
            
            # Trim history if needed
            if len(self._transfer_history) > self._max_history:
                self._transfer_history = self._transfer_history[-self._max_history:]
    
    def get_transfer_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about domain transfers.
        
        Returns:
            Dictionary with transfer statistics
        """
        with self._history_lock:
            if not self._transfer_history:
                return {
                    'total_transfers': 0,
                    'successful': 0,
                    'failed': 0,
                    'success_rate': 0.0,
                }
            
            # Create a snapshot of history for thread-safe iteration
            history_snapshot = list(self._transfer_history)
        
        successful = sum(1 for t in history_snapshot if t['success'])
        total = len(history_snapshot)
        total_concepts = sum(t['concepts'] for t in history_snapshot)
        
        return {
            'total_transfers': total,
            'successful': successful,
            'failed': total - successful,
            'success_rate': successful / total if total > 0 else 0.0,
            'total_concepts_transferred': total_concepts,
        }


# Module-level singleton for convenience
_bridge_instance: Optional[ToolDomainBridge] = None


def get_tool_domain_bridge() -> ToolDomainBridge:
    """Get the singleton ToolDomainBridge instance."""
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = ToolDomainBridge()
    return _bridge_instance
