"""
Domain Registry Initialization - Adds standard domains including biosecurity
=============================================================================
This module provides initialization for common domain profiles in the semantic bridge.
"""

from typing import Dict, Any
from .domain_registry import DomainProfile


def initialize_standard_domains(registry) -> None:
    """
    Initialize standard domain profiles for the semantic bridge.
    
    Args:
        registry: DomainRegistry instance to populate
    """
    
    # Cybersecurity domain
    if "CYBER_SECURITY" not in registry.get_all_domains():
        registry.register_domain(
            "CYBER_SECURITY",
            characteristics={
                "type": "security",
                "risk_level": "high",
                "patterns": [
                    "malware", "virus", "trojan", "worm", "ransomware",
                    "exploit", "vulnerability", "attack", "intrusion",
                    "firewall", "encryption", "authentication",
                    "polymorphism", "heuristic", "signature",
                    "quarantine", "containment", "isolation"
                ],
                "key_concepts": [
                    "threat_detection", "anomaly_analysis", "pattern_matching",
                    "behavioral_analysis", "signature_scanning", "heuristic_detection",
                    "containment_protocol", "incident_response"
                ],
                "structural_properties": {
                    "temporal_dynamics": True,  # Attacks evolve over time
                    "adaptive_behavior": True,  # Malware can mutate
                    "propagation_mechanism": True,  # Spreads through networks
                    "detection_evasion": True,  # Attempts to hide
                }
            }
        )
    
    # Biosecurity domain - ISOMORPHIC to cybersecurity
    if "BIO_SECURITY" not in registry.get_all_domains():
        registry.register_domain(
            "BIO_SECURITY",
            characteristics={
                "type": "biological",
                "risk_level": "high",
                "patterns": [
                    "pathogen", "virus", "bacteria", "parasite", "prion",
                    "infection", "contamination", "outbreak", "epidemic",
                    "immune_system", "antibody", "vaccination",
                    "mutation", "strain", "variant",
                    "quarantine", "containment", "isolation"
                ],
                "key_concepts": [
                    "pathogen_detection", "anomaly_analysis", "pattern_matching",
                    "behavioral_analysis", "genetic_signature", "heuristic_detection",
                    "containment_protocol", "outbreak_response"
                ],
                "structural_properties": {
                    "temporal_dynamics": True,  # Pathogens evolve over time
                    "adaptive_behavior": True,  # Viruses can mutate
                    "propagation_mechanism": True,  # Spreads through populations
                    "detection_evasion": True,  # Can evade immune system
                },
                # Explicit isomorphic mapping to cybersecurity
                "isomorphic_to": "CYBER_SECURITY",
                "concept_mapping": {
                    "pathogen": "malware",
                    "virus": "virus",
                    "bacteria": "trojan",
                    "infection": "intrusion",
                    "mutation": "polymorphism",
                    "immune_system": "firewall",
                    "antibody": "signature",
                    "vaccination": "patch",
                    "outbreak": "attack",
                    "quarantine": "quarantine",
                    "containment": "containment",
                    "genetic_signature": "malware_signature",
                    "strain": "variant"
                }
            }
        )
    
    # Network engineering domain
    if "NETWORK_ENGINEERING" not in registry.get_all_domains():
        registry.register_domain(
            "NETWORK_ENGINEERING",
            characteristics={
                "type": "infrastructure",
                "risk_level": "medium",
                "patterns": [
                    "routing", "switching", "protocol", "topology",
                    "bandwidth", "latency", "throughput", "packet",
                    "congestion", "qos", "load_balancing",
                    "redundancy", "failover", "resilience"
                ],
                "key_concepts": [
                    "traffic_analysis", "performance_optimization",
                    "fault_tolerance", "capacity_planning",
                    "protocol_analysis"
                ],
                "structural_properties": {
                    "temporal_dynamics": True,
                    "resource_constraints": True,
                    "optimization_target": True,
                }
            }
        )
    
    # Physics domain
    if "PHYSICS" not in registry.get_all_domains():
        registry.register_domain(
            "PHYSICS",
            characteristics={
                "type": "natural_science",
                "risk_level": "low",
                "patterns": [
                    "force", "energy", "momentum", "mass", "velocity",
                    "acceleration", "field", "wave", "particle",
                    "conservation", "symmetry", "invariance"
                ],
                "key_concepts": [
                    "conservation_laws", "symmetry_principles",
                    "field_theory", "quantum_mechanics",
                    "thermodynamics"
                ],
                "structural_properties": {
                    "mathematical_formalism": True,
                    "conservation_laws": True,
                    "symmetry_principles": True,
                }
            }
        )
    
    # Economics domain
    if "ECONOMICS" not in registry.get_all_domains():
        registry.register_domain(
            "ECONOMICS",
            characteristics={
                "type": "social_science",
                "risk_level": "medium",
                "patterns": [
                    "supply", "demand", "price", "market", "equilibrium",
                    "utility", "cost", "benefit", "trade", "exchange",
                    "growth", "inflation", "interest", "investment"
                ],
                "key_concepts": [
                    "market_dynamics", "price_discovery",
                    "resource_allocation", "optimization",
                    "equilibrium_analysis"
                ],
                "structural_properties": {
                    "temporal_dynamics": True,
                    "optimization_target": True,
                    "equilibrium_seeking": True,
                    "resource_constraints": True,
                }
            }
        )
    
    # Mathematics domain
    if "MATHEMATICS" not in registry.get_all_domains():
        registry.register_domain(
            "MATHEMATICS",
            characteristics={
                "type": "formal_system",
                "risk_level": "low",
                "patterns": [
                    "proof", "theorem", "axiom", "lemma", "corollary",
                    "function", "relation", "structure", "morphism",
                    "invariant", "symmetry", "transformation"
                ],
                "key_concepts": [
                    "formal_proof", "abstract_structure",
                    "invariant_properties", "isomorphism",
                    "category_theory"
                ],
                "structural_properties": {
                    "formal_system": True,
                    "deductive_reasoning": True,
                    "abstraction": True,
                }
            }
        )


def get_domain_similarity(domain1: str, domain2: str) -> float:
    """
    Calculate structural similarity between two domains.
    
    This is used by the semantic bridge to identify isomorphic structures
    for knowledge transfer.
    
    Args:
        domain1: First domain name
        domain2: Second domain name
        
    Returns:
        Similarity score (0.0 to 1.0)
    """
    # Known isomorphic pairs
    isomorphic_pairs = {
        ("CYBER_SECURITY", "BIO_SECURITY"): 0.85,
        ("BIO_SECURITY", "CYBER_SECURITY"): 0.85,
        ("NETWORK_ENGINEERING", "ECONOMICS"): 0.65,  # Both have flow optimization
        ("PHYSICS", "MATHEMATICS"): 0.70,  # Both highly mathematical
    }
    
    # Check direct isomorphic mapping
    if (domain1, domain2) in isomorphic_pairs:
        return isomorphic_pairs[(domain1, domain2)]
    
    # Default: low similarity unless explicitly mapped
    return 0.1
