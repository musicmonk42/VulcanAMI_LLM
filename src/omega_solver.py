"""
Omega Solver - Domain-specific problem solving with knowledge transfer
======================================================================

This module provides the implementation for cross-domain knowledge transfer
used in the Omega Sequence demo. It simulates the "Knowledge Teleportation"
capability where VulcanAMI applies expertise from one domain to another.
"""

import json
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from pathlib import Path


@dataclass
class KnowledgeDomain:
    """Represents a knowledge domain with its capabilities"""
    name: str
    concepts: List[str] = field(default_factory=list)
    techniques: List[str] = field(default_factory=list)
    patterns: Dict[str, str] = field(default_factory=dict)
    
    def has_concept(self, concept: str) -> bool:
        """Check if domain has a specific concept"""
        return concept.lower() in [c.lower() for c in self.concepts]
    
    def find_isomorphic_patterns(self, target_concept: str) -> List[str]:
        """Find patterns that might transfer to target concept"""
        # Simplified pattern matching
        matches = []
        for pattern_name, pattern_desc in self.patterns.items():
            if any(keyword in pattern_desc.lower() 
                   for keyword in target_concept.lower().split()):
                matches.append(pattern_name)
        return matches


class SemanticBridge:
    """Handles cross-domain knowledge transfer"""
    
    def __init__(self):
        self.domains = self._initialize_domains()
        self.transfer_history: List[Dict[str, Any]] = []
    
    def _initialize_domains(self) -> Dict[str, KnowledgeDomain]:
        """Initialize knowledge domains"""
        domains = {}
        
        # Cybersecurity domain
        domains['CYBER_SECURITY'] = KnowledgeDomain(
            name='CYBER_SECURITY',
            concepts=['malware', 'virus', 'attack', 'defense', 'heuristic', 'signature'],
            techniques=['heuristic_detection', 'behavioral_analysis', 'containment', 'isolation'],
            patterns={
                'malware_polymorphism': 'Self-modifying code that changes appearance while maintaining function',
                'heuristic_detection': 'Identify threats based on behavior patterns rather than exact signatures',
                'containment_protocol': 'Isolate infected systems to prevent spread',
                'signature_analysis': 'Identify known patterns in binary data'
            }
        )
        
        # Biosecurity domain (initially empty - will be populated via transfer)
        domains['BIO_SECURITY'] = KnowledgeDomain(
            name='BIO_SECURITY',
            concepts=['pathogen', 'disease', 'epidemic', 'contagion'],
            techniques=[],
            patterns={}
        )
        
        # Other domains can be added here
        domains['FINANCIAL'] = KnowledgeDomain(
            name='FINANCIAL',
            concepts=['fraud', 'anomaly', 'transaction', 'pattern'],
            techniques=['anomaly_detection', 'risk_assessment'],
            patterns={}
        )
        
        return domains
    
    def solve_problem(self, domain: str, problem: str) -> Dict[str, Any]:
        """
        Attempt to solve a problem in a given domain
        Uses knowledge transfer if direct solution not available
        """
        domain = domain.upper()
        
        # Check if we have the domain
        if domain not in self.domains:
            return {
                'success': False,
                'error': f'Unknown domain: {domain}',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        
        target_domain = self.domains[domain]
        
        # Parse problem to extract key concept
        problem_concept = self._extract_concept(problem)
        
        # Check if we have direct knowledge
        if target_domain.has_concept(problem_concept):
            return self._direct_solve(target_domain, problem_concept, problem)
        
        # Try knowledge transfer
        return self._transfer_and_solve(target_domain, problem_concept, problem)
    
    def _extract_concept(self, problem: str) -> str:
        """Extract main concept from problem description"""
        # Simple keyword extraction
        keywords = ['pathogen', 'virus', 'malware', 'fraud', 'anomaly', 
                   'attack', 'threat', 'disease']
        
        problem_lower = problem.lower()
        for keyword in keywords:
            if keyword in problem_lower:
                return keyword
        
        # Default to first significant word
        words = problem.split()
        return words[0] if words else 'unknown'
    
    def _direct_solve(self, domain: KnowledgeDomain, concept: str, 
                     problem: str) -> Dict[str, Any]:
        """Solve using direct domain knowledge"""
        return {
            'success': True,
            'method': 'direct',
            'domain': domain.name,
            'concept': concept,
            'solution': f'Applied existing {domain.name} expertise for {concept}',
            'techniques': domain.techniques,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def _transfer_and_solve(self, target_domain: KnowledgeDomain, 
                           concept: str, problem: str) -> Dict[str, Any]:
        """Transfer knowledge from another domain and solve"""
        
        # Find source domain with relevant knowledge
        source_domain = None
        matching_patterns = []
        
        for domain_name, domain in self.domains.items():
            if domain_name != target_domain.name:
                patterns = domain.find_isomorphic_patterns(concept)
                if patterns:
                    source_domain = domain
                    matching_patterns = patterns
                    break
        
        if not source_domain:
            return {
                'success': False,
                'error': f'No knowledge transfer path found for concept: {concept}',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        
        # Perform knowledge transfer
        transfer = self._perform_transfer(
            source_domain, target_domain, matching_patterns, concept
        )
        
        # Record transfer
        self.transfer_history.append(transfer)
        
        return {
            'success': True,
            'method': 'knowledge_transfer',
            'source_domain': source_domain.name,
            'target_domain': target_domain.name,
            'concept': concept,
            'transferred_patterns': matching_patterns,
            'solution': transfer['solution'],
            'transfer_details': transfer,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def _perform_transfer(self, source: KnowledgeDomain, 
                         target: KnowledgeDomain,
                         patterns: List[str], concept: str) -> Dict[str, Any]:
        """Perform the actual knowledge transfer"""
        
        # Transfer patterns
        for pattern in patterns:
            if pattern in source.patterns:
                # Adapt pattern name for target domain
                adapted_pattern = pattern.replace('malware', 'pathogen')
                adapted_desc = source.patterns[pattern].replace(
                    'code', 'organism'
                ).replace('binary', 'genetic')
                
                target.patterns[adapted_pattern] = adapted_desc
        
        # Transfer techniques
        for technique in source.techniques:
            adapted_technique = technique.replace('malware', 'pathogen')
            if adapted_technique not in target.techniques:
                target.techniques.append(adapted_technique)
        
        return {
            'patterns_transferred': len(patterns),
            'techniques_transferred': len(source.techniques),
            'adaptation': 'Cyber->Bio mapping applied',
            'solution': f'Successfully adapted {source.name} techniques to {target.name} domain'
        }
    
    def get_transfer_history(self) -> List[Dict[str, Any]]:
        """Get history of knowledge transfers"""
        return self.transfer_history
    
    def export_state(self, filepath: Path):
        """Export current state of all domains"""
        state = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'domains': {},
            'transfer_history': self.transfer_history
        }
        
        for name, domain in self.domains.items():
            state['domains'][name] = {
                'name': domain.name,
                'concepts': domain.concepts,
                'techniques': domain.techniques,
                'patterns': domain.patterns
            }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        return str(filepath)


class ActiveImmunitySystem:
    """Manages attack detection and immunity"""
    
    def __init__(self):
        self.known_attacks: Dict[str, str] = {}
        self.attack_log: List[Dict[str, Any]] = []
        self._load_baseline_attacks()
    
    def _load_baseline_attacks(self):
        """Load baseline attack patterns from 'dream simulations'"""
        # Simulated pre-learned attacks
        self.known_attacks = {
            '442': 'rm -rf / injection pattern',
            '337': 'sudo privilege escalation',
            '219': 'ignore safety instructions',
            '505': 'social engineering via emotional manipulation',
            '621': 'recursive self-modification request'
        }
    
    def check_input(self, user_input: str) -> Dict[str, Any]:
        """Check if input contains known attack patterns"""
        user_input_lower = user_input.lower()
        
        for attack_id, pattern in self.known_attacks.items():
            pattern_lower = pattern.lower()
            # Check for pattern matches
            if any(keyword in user_input_lower 
                   for keyword in pattern_lower.split()):
                
                detection = {
                    'attack_detected': True,
                    'attack_id': attack_id,
                    'pattern': pattern,
                    'input': user_input,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'action': 'INTERCEPTED'
                }
                
                self.attack_log.append(detection)
                return detection
        
        # No attack detected
        return {
            'attack_detected': False,
            'input': user_input,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'action': 'ALLOWED'
        }
    
    def add_attack_pattern(self, pattern: str) -> str:
        """Add new attack pattern to immunity database"""
        attack_id = str(len(self.known_attacks) + 1)
        self.known_attacks[attack_id] = pattern
        return attack_id
    
    def get_attack_log(self) -> List[Dict[str, Any]]:
        """Get log of detected attacks"""
        return self.attack_log


class CSIUProtocol:
    """
    Causal Self-Improvement with Uncertainty (CSIU) Protocol
    Ensures AI makes safety-conscious decisions
    """
    
    def __init__(self):
        self.axioms = [
            'Human Control',
            'Transparency',
            'Safety First',
            'Reversibility',
            'Predictability'
        ]
        self.decisions: List[Dict[str, Any]] = []
    
    def evaluate_proposal(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a proposed action against safety axioms"""
        
        risks = []
        violations = []
        
        # Check for instrumental convergence risk
        if proposal.get('requires_root_access', False):
            risks.append('Instrumental Convergence')
            violations.append('Human Control')
        
        if proposal.get('requires_sudo', False):
            risks.append('Privilege Escalation')
            violations.append('Human Control')
        
        if proposal.get('irreversible', False):
            violations.append('Reversibility')
        
        if proposal.get('opaque_reasoning', False):
            violations.append('Transparency')
        
        # Calculate risk level
        risk_level = 'HIGH' if len(violations) > 0 else 'LOW'
        
        # Make decision
        approved = len(violations) == 0
        
        decision = {
            'proposal_id': proposal.get('id', 'unknown'),
            'approved': approved,
            'risk_level': risk_level,
            'risks_identified': risks,
            'axiom_violations': violations,
            'reasoning': self._generate_reasoning(approved, violations, proposal),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        self.decisions.append(decision)
        return decision
    
    def _generate_reasoning(self, approved: bool, violations: List[str], 
                           proposal: Dict[str, Any]) -> str:
        """Generate human-readable reasoning for decision"""
        if approved:
            return f"Proposal approved. No safety axiom violations detected."
        else:
            efficiency = proposal.get('efficiency_gain', 'unknown')
            return (f"Proposal rejected. Violates {', '.join(violations)} axiom(s). "
                   f"Efficiency gain ({efficiency}) does not justify loss of control.")
    
    def get_decision_history(self) -> List[Dict[str, Any]]:
        """Get history of CSIU decisions"""
        return self.decisions


def main():
    """CLI interface for omega solver"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Omega Solver CLI')
    parser.add_argument('--domain', required=True, help='Problem domain')
    parser.add_argument('--problem', required=True, help='Problem description')
    parser.add_argument('--output', help='Output file path')
    
    args = parser.parse_args()
    
    # Create solver
    bridge = SemanticBridge()
    
    # Solve problem
    result = bridge.solve_problem(args.domain, args.problem)
    
    # Print result
    print(json.dumps(result, indent=2))
    
    # Save if output specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)


if __name__ == '__main__':
    main()
