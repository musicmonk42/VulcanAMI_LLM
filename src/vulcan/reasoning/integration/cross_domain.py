"""
Cross-domain knowledge transfer for reasoning integration.

Applies cross-domain knowledge transfer using SemanticBridge and DomainBridge.
"""

import logging
import time
from typing import Any, Dict, List

from .types import LOG_PREFIX

logger = logging.getLogger(__name__)


def apply_cross_domain_transfer(
    orchestrator,
    query: str,
    query_analysis: Dict[str, Any],
    selected_tools: List[str],
) -> Dict[str, Any]:
    """
    Apply cross-domain knowledge transfer using SemanticBridge.
    
    This method enables knowledge learned in one domain to be applied
    in related domains, improving reasoning quality for queries that
    span multiple conceptual areas.
    
    Processing Flow:
        1. Identify domains involved from selected tools
        2. Determine primary domain based on query type
        3. Find applicable concepts from SemanticBridge
        4. Validate transfer compatibility between domains
        5. Execute transfers for compatible concepts
        6. Record transfer for learning
    
    Args:
        orchestrator: ReasoningIntegration instance
        query: The query string being processed
        query_analysis: Analysis results with type, complexity, etc.
        selected_tools: List of tools selected for this query
        
    Returns:
        Dictionary containing:
            - success: Whether transfer was successful
            - domains: List of domains involved
            - primary_domain: Identified primary domain
            - transferred_concepts: List of transferred concept info
            - transfer_count: Number of concepts transferred
            - error: Error message if failed
            
    Example:
        >>> result = apply_cross_domain_transfer(
        ...     orchestrator,
        ...     query="What causes X given Y?",
        ...     query_analysis={'type': 'reasoning', 'complexity': 0.6},
        ...     selected_tools=['causal', 'probabilistic']
        ... )
        >>> print(result['transfer_count'])
        2
    """
    # Ensure components are initialized
    orchestrator._init_components()
    
    # Validate prerequisites
    if orchestrator._semantic_bridge is None:
        logger.debug(f"{LOG_PREFIX} SemanticBridge not available for cross-domain transfer")
        return {
            'success': False,
            'error': 'semantic_bridge_unavailable',
            'domains': [],
        }
    
    if orchestrator._domain_bridge is None:
        logger.debug(f"{LOG_PREFIX} DomainBridge not available for cross-domain transfer")
        return {
            'success': False,
            'error': 'domain_bridge_unavailable',
            'domains': [],
        }
    
    transfer_start = time.perf_counter()
    
    try:
        # Step 1: Get domains involved
        domains = orchestrator._domain_bridge.get_domains_for_tools(selected_tools)
        
        # FIX: Early exit if only one domain - no cross-domain transfer possible
        if len(domains) < 2:
            logger.debug(
                f"{LOG_PREFIX} Single domain query - cross-domain transfer not applicable"
            )
            return {
                'success': False,
                'error': 'single_domain_query',
                'domains': list(domains),
                'transfer_count': 0,
            }
        
        # Step 2: Identify primary domain
        query_type = query_analysis.get('type', 'general')
        primary_domain = orchestrator._domain_bridge.identify_primary_domain(
            selected_tools, query_type
        )
        
        logger.info(
            f"{LOG_PREFIX} Cross-domain transfer: domains={domains}, "
            f"primary={primary_domain}"
        )
        
        # Step 3: Get applicable concepts from primary domain
        applicable_concepts = []
        try:
            applicable_concepts = orchestrator._semantic_bridge.get_applicable_concepts(
                domain=primary_domain,
                min_confidence=0.6,
            )
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Failed to get applicable concepts: {e}")
        
        # Step 4: Try to transfer concepts from related domains
        transferred = []
        for source_domain in domains:
            if source_domain == primary_domain:
                continue
            
            # Check if transfer is possible
            if not orchestrator._domain_bridge.can_transfer_between(source_domain, primary_domain):
                continue
            
            # Get source domain concepts
            try:
                source_concepts = orchestrator._semantic_bridge.get_applicable_concepts(
                    domain=source_domain,
                    min_confidence=0.5,
                )
            except Exception as e:
                logger.debug(f"{LOG_PREFIX} Failed to get concepts from {source_domain}: {e}")
                continue
            
            # Validate and transfer each concept (limit to top 3)
            for concept in source_concepts[:3]:
                try:
                    # Validate compatibility
                    compatibility = orchestrator._semantic_bridge.validate_transfer_compatibility(
                        concept=concept,
                        source=source_domain,
                        target=primary_domain,
                    )
                    
                    if not compatibility.is_compatible():
                        # Log why transfer was rejected for debugging
                        concept_id = getattr(concept, 'concept_id', str(concept)[:20])
                        logger.debug(
                            f"{LOG_PREFIX} Transfer rejected for {concept_id}: "
                            f"score={compatibility.compatibility_score:.2f}, "
                            f"risks={compatibility.risks}"
                        )
                        continue
                    
                    # Execute transfer
                    transferred_concept = orchestrator._semantic_bridge.transfer_concept(
                        concept=concept,
                        source_domain=source_domain,
                        target_domain=primary_domain,
                    )
                    
                    if transferred_concept is not None:
                        concept_id = getattr(concept, 'concept_id', str(concept)[:20])
                        transferred.append({
                            'concept_id': concept_id,
                            'source': source_domain,
                            'target': primary_domain,
                            'confidence': compatibility.confidence,
                        })
                        logger.debug(
                            f"{LOG_PREFIX} Transferred concept from "
                            f"{source_domain} → {primary_domain}"
                        )
                        
                except Exception as e:
                    logger.debug(f"{LOG_PREFIX} Concept transfer failed: {e}")
                    continue
        
        # Record transfer in domain bridge
        if transferred:
            # Note: Safe set subtraction - handle edge cases
            # If domains has exactly one element equal to primary_domain,
            # (domains - {primary_domain}) is empty and [0] would raise IndexError
            other_domains = list(domains - {primary_domain})
            source_domain = other_domains[0] if other_domains else 'unknown'
            orchestrator._domain_bridge.record_transfer(
                source_domain=source_domain,
                target_domain=primary_domain,
                success=True,
                concepts_transferred=len(transferred),
            )
        
        transfer_time_ms = (time.perf_counter() - transfer_start) * 1000
        
        logger.info(
            f"{LOG_PREFIX} Cross-domain transfer complete: "
            f"transferred={len(transferred)}, time={transfer_time_ms:.1f}ms"
        )
        
        return {
            'success': True,
            'domains': list(domains),
            'primary_domain': primary_domain,
            'applicable_concepts': len(applicable_concepts),
            'transferred_concepts': transferred,
            'transfer_count': len(transferred),
            'transfer_time_ms': transfer_time_ms,
        }
        
    except Exception as e:
        logger.warning(f"{LOG_PREFIX} Cross-domain transfer failed: {e}")
        # Note: Properly check if domains variable exists
        # Use 'domains' in locals() to check if local variable is defined
        domains_list = list(locals().get('domains', set()) or [])
        return {
            'success': False,
            'error': str(e),
            'domains': domains_list,
        }
