"""
Domain parsing and structure extraction for analogical reasoning.

This module provides functions for extracting domain structures from natural language queries,
including entity recognition, relation extraction, and domain-specific concept identification.

Features:
    - Multi-pattern query parsing (7 different natural language patterns)
    - Domain-specific concept dictionaries (software, biology, economics, physics)
    - NLP-based entity extraction using spaCy (with fallback)
    - Relation extraction using regex patterns
    - Auto-detection of domains from content keywords

References:
    This implements domain parsing for Structure-Mapping Theory (SMT) based analogical
    reasoning as described in Gentner (1983).
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Dict, Tuple

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Import spaCy NLP helper function
try:
    import spacy
    SPACY_AVAILABLE = True
    
    # Lazy loading of spaCy model
    _nlp = None
    _nlp_loaded = False
    import threading
    _nlp_lock = threading.Lock()
    
    def get_nlp():
        """Lazy-load spaCy model on first use."""
        global _nlp, _nlp_loaded
        if _nlp_loaded:
            return _nlp
        with _nlp_lock:
            if _nlp_loaded:
                return _nlp
            for model_name in ["en_core_web_lg", "en_core_web_md", "en_core_web_sm"]:
                try:
                    _nlp = spacy.load(model_name)
                    logger.info(f"Loaded spaCy model '{model_name}' for domain parsing")
                    break
                except (OSError, Exception) as e:
                    logger.debug(f"Could not load spaCy model '{model_name}': {e}")
                    continue
            if _nlp is None:
                logger.warning("spaCy model not loaded. Install with: python -m spacy download en_core_web_lg")
            _nlp_loaded = True
        return _nlp
except ImportError:
    SPACY_AVAILABLE = False
    def get_nlp():
        """Return None when spaCy is not available."""
        return None
    logger.info("spaCy not available, domain parsing will use pattern-based extraction only")


# ============================================================================
# Domain-Specific Concept Dictionaries
# ============================================================================

# Software/Distributed Systems concepts
SOFTWARE_CONCEPTS = {
    # Coordination/Leadership
    'leader election': {'role': 'coordinator', 'type': 'mechanism'},
    'leader': {'role': 'coordinator', 'type': 'entity'},
    'coordinator': {'role': 'coordinator', 'type': 'entity'},
    'master': {'role': 'coordinator', 'type': 'entity'},
    'primary': {'role': 'coordinator', 'type': 'entity'},
    # Consensus
    'quorum': {'role': 'consensus_mechanism', 'type': 'mechanism'},
    'consensus': {'role': 'agreement', 'type': 'property'},
    'paxos': {'role': 'consensus_mechanism', 'type': 'mechanism'},
    'raft': {'role': 'consensus_mechanism', 'type': 'mechanism'},
    # Validation
    'fencing token': {'role': 'validator', 'type': 'mechanism'},
    'token': {'role': 'validator', 'type': 'mechanism'},
    # Problems
    'split brain': {'role': 'conflict_state', 'type': 'problem'},
    'write divergence': {'role': 'inconsistency', 'type': 'problem'},
    'partition': {'role': 'network_failure', 'type': 'problem'},
    'failure': {'role': 'error', 'type': 'problem'},
    # Workers
    'worker': {'role': 'executor', 'type': 'entity'},
    'follower': {'role': 'executor', 'type': 'entity'},
    'replica': {'role': 'executor', 'type': 'entity'},
    'node': {'role': 'component', 'type': 'entity'},
    # Operations
    'load balancer': {'role': 'distributor', 'type': 'mechanism'},
    'scheduler': {'role': 'coordinator', 'type': 'mechanism'},
    'heartbeat': {'role': 'health_check', 'type': 'mechanism'},
}

# Biology/Biological Systems concepts
BIOLOGY_CONCEPTS = {
    # Coordination
    'control center': {'role': 'coordinator', 'type': 'entity'},
    'control centres': {'role': 'coordinator', 'type': 'entity'},
    'brain': {'role': 'coordinator', 'type': 'entity'},
    'nervous system': {'role': 'coordinator', 'type': 'entity'},
    # Signaling
    'hormone cascade': {'role': 'signal_mechanism', 'type': 'mechanism'},
    'hormone': {'role': 'signal', 'type': 'entity'},
    'hormonal': {'role': 'signal', 'type': 'property'},
    'neurotransmitter': {'role': 'signal', 'type': 'entity'},
    'signal': {'role': 'signal', 'type': 'entity'},
    # Stability/Problems
    'metabolic instability': {'role': 'inconsistency', 'type': 'problem'},
    'metabolic stability': {'role': 'consistency', 'type': 'property'},
    'homeostasis': {'role': 'consistency', 'type': 'property'},
    'competing centers': {'role': 'conflict_state', 'type': 'problem'},
    # Mechanisms
    'feedback': {'role': 'control_mechanism', 'type': 'mechanism'},
    'negative feedback': {'role': 'control_mechanism', 'type': 'mechanism'},
    'positive feedback': {'role': 'amplification', 'type': 'mechanism'},
    # Executors
    'muscle': {'role': 'executor', 'type': 'entity'},
    'organ': {'role': 'component', 'type': 'entity'},
    'cell': {'role': 'component', 'type': 'entity'},
}

# Economics concepts
ECONOMICS_CONCEPTS = {
    'market': {'role': 'coordinator', 'type': 'mechanism'},
    'price': {'role': 'signal', 'type': 'entity'},
    'supply': {'role': 'resource', 'type': 'entity'},
    'demand': {'role': 'requirement', 'type': 'entity'},
    'equilibrium': {'role': 'consistency', 'type': 'property'},
    'inflation': {'role': 'inconsistency', 'type': 'problem'},
}

# Physics concepts
PHYSICS_CONCEPTS = {
    'force': {'role': 'cause', 'type': 'entity'},
    'energy': {'role': 'resource', 'type': 'entity'},
    'equilibrium': {'role': 'consistency', 'type': 'property'},
    'entropy': {'role': 'disorder', 'type': 'property'},
}

# Domain keyword lists for auto-detection
DOMAIN_KEYWORDS = {
    'software': ['distributed system', 'software', 'computer', 'node', 'server', 
                'leader election', 'consensus', 'quorum', 'database', 'network'],
    'biology': ['biology', 'biological', 'organism', 'cell', 'body', 'hormone',
               'brain', 'muscle', 'organ', 'metabolic'],
    'economics': ['economics', 'economic', 'market', 'price', 'supply', 'demand',
                 'trade', 'currency', 'inflation'],
    'physics': ['physics', 'physical', 'force', 'energy', 'particle', 'wave',
               'quantum', 'gravity', 'momentum'],
}


# ============================================================================
# Main Domain Parsing Functions
# ============================================================================

def parse_domains_from_query(query_text: str) -> Tuple[Dict, Dict]:
    """
    Parse source and target domains from natural language query.
    
    Supports multiple natural language patterns:
    - "Domain S (software): ..." / "Domain T (biology): ..."
    - "source: ... target: ..."
    - "X is like Y" / "X are like Y"
    - "What's the biological equivalent of X?"
    - "In domain A, ... What's the analog in domain B?"
    - "How is X like Y?"
    - Implicit domain detection from keywords
    
    Args:
        query_text: Natural language describing the analogy
        
    Returns:
        Tuple of (source_domain_dict, target_domain_dict) where each dict contains:
        - name: Domain name
        - entities: Dict of extracted entities with roles/types
        - relations: List of extracted relations
        - concepts: List of identified concepts
        - raw_text: Original text
        
    Examples:
        >>> source, target = parse_domains_from_query(
        ...     "Domain S (software): leader election prevents split brain. "
        ...     "Domain T (biology): brain coordinates muscles."
        ... )
        >>> source['name']
        'software'
        >>> target['name']
        'biology'
        
    Note:
        This function implements multi-pattern parsing to handle various natural
        language formulations of analogical reasoning queries.
    """
    source = {'name': 'source', 'entities': {}, 'relations': [], 'concepts': [], 'raw_text': ''}
    target = {'name': 'target', 'entities': {}, 'relations': [], 'concepts': [], 'raw_text': ''}
    
    # Pattern 1: "Domain S (label): content" format
    domain_s_match = re.search(
        r'Domain\s+S\s*\(([^)]+)\)[:\s]*(.*?)(?=Domain\s+T|Task:|$)',
        query_text, re.DOTALL | re.IGNORECASE
    )
    domain_t_match = re.search(
        r'Domain\s+T\s*\(([^)]+)\)[:\s]*(.*?)(?=Task:|$)',
        query_text, re.DOTALL | re.IGNORECASE
    )
    
    if domain_s_match:
        source['name'] = domain_s_match.group(1).strip()
        source_text = domain_s_match.group(2).strip()
        source = extract_domain_structure(source_text, source['name'])
    
    if domain_t_match:
        target['name'] = domain_t_match.group(1).strip()
        target_text = domain_t_match.group(2).strip()
        target = extract_domain_structure(target_text, target['name'])
    
    # If explicit domain labels found, return early
    if domain_s_match and domain_t_match:
        return source, target
    
    # Pattern 2: "source: ... target: ..." patterns
    if not domain_s_match and not domain_t_match:
        source_match = re.search(r'source[:\s]+(.+?)(?=target|$)', query_text, re.IGNORECASE | re.DOTALL)
        target_match = re.search(r'target[:\s]+(.+?)$', query_text, re.IGNORECASE | re.DOTALL)
        
        if source_match:
            source = extract_domain_structure(source_match.group(1).strip(), 'source')
        if target_match:
            target = extract_domain_structure(target_match.group(1).strip(), 'target')
        
        if source_match or target_match:
            return source, target
    
    # Pattern 3: "X is like Y" / "X are like Y"
    like_match = re.search(
        r'([^,.]+?)\s+(?:is|are)\s+like\s+([^,.?]+)',
        query_text, re.IGNORECASE
    )
    if like_match:
        source_text = like_match.group(1).strip()
        target_text = like_match.group(2).strip()
        source = extract_domain_structure(source_text, 'source')
        target = extract_domain_structure(target_text, 'target')
        return source, target
    
    # Pattern 4: "How is X like Y?" / "How are X similar to Y?"
    how_like_match = re.search(
        r'how\s+(?:is|are)\s+(.+?)\s+(?:like|similar\s+to)\s+(.+?)\??$',
        query_text, re.IGNORECASE
    )
    if how_like_match:
        source_text = how_like_match.group(1).strip()
        target_text = how_like_match.group(2).strip()
        source = extract_domain_structure(source_text, 'source')
        target = extract_domain_structure(target_text, 'target')
        return source, target
    
    # Pattern 5: "What's the X equivalent of Y?"
    equivalent_match = re.search(
        r"what(?:'s|s| is) the (\w+)\s+equivalent(?:\s+of\s+(.+?))?(?:\?|$)",
        query_text, re.IGNORECASE
    )
    if equivalent_match:
        target_domain = equivalent_match.group(1).strip()
        source_text = equivalent_match.group(2).strip() if equivalent_match.group(2) else query_text
        source = extract_domain_structure(source_text, 'source')
        target = extract_domain_structure('', target_domain)
        target['name'] = target_domain
        return source, target
    
    # Pattern 6: "In X, ... What's the analog in Y?"
    in_domain_match = re.search(
        r'in\s+(\w+(?:\s+\w+)?(?:\s+systems?)?)[,:](.+?)(?:what(?:\'s| is) the (?:analog|equivalent)(?:\s+in\s+(\w+))?)',
        query_text, re.IGNORECASE | re.DOTALL
    )
    if in_domain_match:
        source_domain = in_domain_match.group(1).strip()
        source_text = in_domain_match.group(2).strip()
        target_domain = in_domain_match.group(3).strip() if in_domain_match.group(3) else 'target'
        source = extract_domain_structure(source_text, source_domain)
        target = extract_domain_structure('', target_domain)
        return source, target
    
    # Pattern 7: Implicit domain detection from content
    source, target = infer_domains_from_content(query_text)
    
    return source, target


def infer_domains_from_content(query_text: str) -> Tuple[Dict, Dict]:
    """
    Infer source and target domains from content keywords.
    
    When explicit patterns fail, detects domains from:
    - Domain keywords (distributed systems, biology, economics, etc.)
    - Technical concepts belonging to specific domains
    
    Args:
        query_text: The query text to analyze
        
    Returns:
        Tuple of (source_domain_dict, target_domain_dict)
        
    Examples:
        >>> source, target = infer_domains_from_content(
        ...     "leader election in distributed systems is like brain in biology"
        ... )
        >>> source['name']
        'software'
        >>> target['name']
        'biology'
        
    Note:
        This provides graceful fallback when explicit domain markers are absent.
        Scores domains by keyword frequency and selects top 2 as source and target.
    """
    query_lower = query_text.lower()
    
    # Score each domain by keyword matches
    domain_scores = {domain: 0 for domain in DOMAIN_KEYWORDS}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for keyword in keywords:
            if keyword in query_lower:
                domain_scores[domain] += 1
    
    # Get top 2 domains
    sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
    top_domains = [d for d, s in sorted_domains[:2] if s > 0]
    
    source = {'name': 'source', 'entities': {}, 'relations': [], 'concepts': [], 'raw_text': query_text}
    target = {'name': 'target', 'entities': {}, 'relations': [], 'concepts': [], 'raw_text': ''}
    
    if len(top_domains) >= 2:
        # Use top 2 detected domains
        source['name'] = top_domains[0]
        target['name'] = top_domains[1]
        source = extract_domain_structure(query_text, top_domains[0])
        target = extract_domain_structure('', top_domains[1])
    elif len(top_domains) == 1:
        # Only one domain detected, use it as source
        source['name'] = top_domains[0]
        source = extract_domain_structure(query_text, top_domains[0])
    else:
        # No domains detected - try to extract structure from full text
        source = extract_domain_structure(query_text, 'source')
    
    return source, target


def extract_domain_structure(text: str, domain_name: str) -> Dict:
    """
    Extract structured domain representation from text.
    
    Identifies:
    - Key concepts/entities with roles and types
    - Relations between entities (causes, prevents, controls)
    - Structural roles (coordinator, mechanism, problem, etc.)
    
    Uses domain-specific concept dictionaries and NLP-based extraction
    when spaCy is available.
    
    Args:
        text: Description of the domain
        domain_name: Name of the domain (e.g., 'software', 'biology')
        
    Returns:
        Dict with keys:
        - name: Domain name
        - entities: Dict mapping entity_name -> {name, role, type}
        - relations: List of dicts with {type, source, target}
        - concepts: List of identified concept strings
        - raw_text: Original text
        
    Examples:
        >>> domain = extract_domain_structure(
        ...     "leader election prevents split brain", 
        ...     "software"
        ... )
        >>> 'leader' in domain['concepts']
        True
        >>> any(r['type'] == 'prevents' for r in domain['relations'])
        True
        
    Note:
        This implements domain structure extraction for analogical mapping,
        supporting the Structure-Mapping Theory framework.
    """
    domain = {
        'name': domain_name,
        'entities': {},
        'relations': [],
        'concepts': [],
        'raw_text': text,
    }
    
    if not text:
        return domain
    
    text_lower = text.lower()
    
    # Select concepts based on domain name
    domain_name_lower = domain_name.lower()
    if 'software' in domain_name_lower or 'distributed' in domain_name_lower:
        concepts = SOFTWARE_CONCEPTS
    elif 'biology' in domain_name_lower or 'bio' in domain_name_lower:
        concepts = BIOLOGY_CONCEPTS
    elif 'econom' in domain_name_lower:
        concepts = ECONOMICS_CONCEPTS
    elif 'physic' in domain_name_lower:
        concepts = PHYSICS_CONCEPTS
    else:
        # Auto-detect based on content - check all concept sets
        scores = {
            'software': sum(1 for c in SOFTWARE_CONCEPTS if c in text_lower),
            'biology': sum(1 for c in BIOLOGY_CONCEPTS if c in text_lower),
            'economics': sum(1 for c in ECONOMICS_CONCEPTS if c in text_lower),
            'physics': sum(1 for c in PHYSICS_CONCEPTS if c in text_lower),
        }
        best_domain = max(scores.items(), key=lambda x: x[1])
        if best_domain[1] > 0:
            concepts = {
                'software': SOFTWARE_CONCEPTS,
                'biology': BIOLOGY_CONCEPTS,
                'economics': ECONOMICS_CONCEPTS,
                'physics': PHYSICS_CONCEPTS,
            }[best_domain[0]]
        else:
            # No matches - combine all concepts
            concepts = {**SOFTWARE_CONCEPTS, **BIOLOGY_CONCEPTS, **ECONOMICS_CONCEPTS, **PHYSICS_CONCEPTS}
    
    # Extract entities that appear in the text
    for concept, props in concepts.items():
        if concept in text_lower:
            entity_name = concept.replace(' ', '_')
            domain['entities'][entity_name] = {
                'name': concept,
                'role': props['role'],
                'type': props['type'],
            }
            domain['concepts'].append(concept)
    
    # Use spaCy NER for additional entity extraction
    nlp_instance = get_nlp()
    if nlp_instance and text:
        try:
            doc = nlp_instance(text)
            for ent in doc.ents:
                entity_name = ent.text.lower().replace(' ', '_')
                if entity_name not in domain['entities']:
                    # Assign role based on entity label
                    role_map = {
                        'ORG': 'organization',
                        'PERSON': 'actor',
                        'GPE': 'location',
                        'PRODUCT': 'component',
                        'EVENT': 'event',
                    }
                    domain['entities'][entity_name] = {
                        'name': ent.text,
                        'role': role_map.get(ent.label_, 'entity'),
                        'type': ent.label_.lower(),
                    }
        except Exception as e:
            logger.debug(f"spaCy extraction failed: {e}")
    
    # Extract relations using regex patterns
    
    # Pattern: "X causes Y" or "X leads to Y"
    cause_patterns = re.findall(
        r'(\w+[\w\s]*?)\s+(?:causes?|leads?\s+to)\s+(\w+[\w\s]*?)(?:\.|,|$)', 
        text_lower
    )
    for cause, effect in cause_patterns:
        domain['relations'].append({
            'type': 'causes',
            'source': cause.strip(),
            'target': effect.strip(),
        })
    
    # Pattern: "X prevents Y" or "X blocks Y"
    prevent_patterns = re.findall(
        r'(\w+[\w\s]*?)\s+(?:prevents?|blocks?)\s+(\w+[\w\s]*?)(?:\.|,|$)', 
        text_lower
    )
    for preventer, prevented in prevent_patterns:
        domain['relations'].append({
            'type': 'prevents',
            'source': preventer.strip(),
            'target': prevented.strip(),
        })
    
    # Pattern: "X coordinates Y" or "X controls Y"
    control_patterns = re.findall(
        r'(\w+[\w\s]*?)\s+(?:coordinates?|controls?|manages?)\s+(\w+[\w\s]*?)(?:\.|,|$)', 
        text_lower
    )
    for controller, controlled in control_patterns:
        domain['relations'].append({
            'type': 'controls',
            'source': controller.strip(),
            'target': controlled.strip(),
        })
    
    return domain
