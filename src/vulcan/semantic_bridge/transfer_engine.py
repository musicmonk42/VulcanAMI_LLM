"""
transfer_engine.py - Concept transfer management for semantic bridge
Part of the VULCAN-AGI system

FIXED: Added safety_config and world_model integration
ENHANCED: Robust effect extraction, mitigation learning, transfer rollback
PRODUCTION-READY: All unbounded data structures fixed with proper limits and eviction
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque, Counter
import time
import json
import hashlib
from enum import Enum
import copy
import threading

# Import safety validator
try:
    from ..safety.safety_validator import EnhancedSafetyValidator
    from ..safety.safety_types import SafetyConfig
    SAFETY_VALIDATOR_AVAILABLE = True
except ImportError:
    SAFETY_VALIDATOR_AVAILABLE = False
    logging.warning("safety_validator not available, transfer_engine operating without safety checks")
    EnhancedSafetyValidator = None
    SafetyConfig = None

# Optional import with fallback
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logging.warning("networkx not available, using fallback graph implementation")
    
    # Fallback - we don't actually use networkx in this module
    # but keeping consistent with other modules
    nx = None

logger = logging.getLogger(__name__)


class TransferType(Enum):
    """Types of concept transfer"""
    FULL = "full"
    PARTIAL = "partial"
    BLOCKED = "blocked"
    CONDITIONAL = "conditional"


class EffectType(Enum):
    """Types of concept effects"""
    PRIMARY = "primary"
    SECONDARY = "secondary"
    SIDE_EFFECT = "side_effect"
    PREREQUISITE = "prerequisite"


class MitigationType(Enum):
    """Types of mitigations"""
    ADAPTATION = "adaptation"
    CONSTRAINT = "constraint"
    WRAPPER = "wrapper"
    FALLBACK = "fallback"
    MONITORING = "monitoring"


class ConstraintType(Enum):
    """Types of constraints"""
    PRECONDITION = "precondition"
    POSTCONDITION = "postcondition"
    INVARIANT = "invariant"
    RESOURCE = "resource"
    TEMPORAL = "temporal"


@dataclass
class ConceptEffect:
    """Effect of applying a concept"""
    effect_id: str
    effect_type: EffectType
    description: str
    domain: str
    importance: float = 0.5
    prerequisites: List[str] = field(default_factory=list)
    outcomes: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    
    def is_critical(self) -> bool:
        """Check if effect is critical"""
        return self.effect_type == EffectType.PRIMARY and self.importance > 0.7


@dataclass
class Mitigation:
    """Mitigation for missing or incompatible effects"""
    mitigation_id: str
    mitigation_type: MitigationType
    target_effect: str
    description: str
    implementation: Dict[str, Any] = field(default_factory=dict)
    cost: float = 1.0
    confidence: float = 0.5
    prerequisites: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'mitigation_id': self.mitigation_id,
            'type': self.mitigation_type.value,
            'target_effect': self.target_effect,
            'description': self.description,
            'cost': self.cost,
            'confidence': self.confidence
        }


@dataclass
class Constraint:
    """Constraint on concept transfer or application"""
    constraint_id: str
    constraint_type: ConstraintType
    description: str
    condition: str  # Expression to evaluate
    severity: float = 0.5  # How strict the constraint is
    domain_specific: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'constraint_id': self.constraint_id,
            'type': self.constraint_type.value,
            'description': self.description,
            'condition': self.condition,
            'severity': self.severity,
            'domain_specific': self.domain_specific
        }
    
    def is_hard_constraint(self) -> bool:
        """Check if this is a hard constraint"""
        return self.severity > 0.8


@dataclass
class TransferDecision:
    """Decision about concept transfer"""
    type: TransferType
    confidence: float
    mitigations: List[Mitigation] = field(default_factory=list)
    constraints: List[Constraint] = field(default_factory=list)
    reasoning: List[str] = field(default_factory=list)
    risk_assessment: Dict[str, float] = field(default_factory=dict)
    estimated_cost: float = 0.0
    
    def is_transferable(self) -> bool:
        """Check if transfer is possible"""
        return self.type in [TransferType.FULL, TransferType.PARTIAL, TransferType.CONDITIONAL]
    
    def requires_mitigation(self) -> bool:
        """Check if mitigations are required"""
        return len(self.mitigations) > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'type': self.type.value,
            'confidence': self.confidence,
            'mitigations': [m.to_dict() for m in self.mitigations],
            'constraints': [c.to_dict() for c in self.constraints],
            'reasoning': self.reasoning,
            'risk_assessment': self.risk_assessment,
            'estimated_cost': self.estimated_cost
        }


@dataclass
class DomainCharacteristics:
    """Characteristics of a domain"""
    name: str
    capabilities: Set[str] = field(default_factory=set)
    limitations: Set[str] = field(default_factory=set)
    typical_effects: List[ConceptEffect] = field(default_factory=list)
    compatibility_matrix: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class TransferEngine:
    """Manages concept transfer between domains - FIXED with safety and world_model"""
    
    def __init__(self, world_model=None, safety_config: Optional[Dict[str, Any]] = None):
        """
        Initialize transfer engine - FIXED: Added world_model and safety_config
        
        Args:
            world_model: World model instance for accessing causal knowledge
            safety_config: Optional safety configuration
        """
        self.world_model = world_model
        
        # Initialize safety validator
        if SAFETY_VALIDATOR_AVAILABLE:
            if isinstance(safety_config, dict) and safety_config:
                self.safety_validator = EnhancedSafetyValidator(SafetyConfig.from_dict(safety_config))
            else:
                self.safety_validator = EnhancedSafetyValidator()
            logger.info("TransferEngine: Safety validator initialized")
        else:
            self.safety_validator = None
            logger.warning("TransferEngine: Safety validator not available - operating without safety checks")
        
        # FIXED: Effect library with size limit
        self.effect_library = {}  # effect_id -> ConceptEffect
        self.max_effects = 10000
        
        # FIXED: Domain characteristics with size limit
        self.domain_characteristics = {}  # domain -> DomainCharacteristics
        self.max_domains = 1000
        
        self.transfer_history = deque(maxlen=1000)
        
        # FIXED: Compatibility cache with size limit
        self.compatibility_cache = {}
        self.max_cache_size = 5000
        
        self.partial_engine = None  # Initialized after __init__
        
        # Configuration
        self.full_transfer_threshold = 0.8
        self.partial_transfer_threshold = 0.5
        
        # Statistics
        self.total_transfers = 0
        self.successful_transfers = 0
        
        # FIXED: Replace defaultdict(int) with Counter
        self.safety_blocks = Counter()
        self.safety_corrections = Counter()
        
        # Thread safety
        self._lock = threading.RLock()
        
        self._initialize_domains()
        
        # FIXED: Initialize partial engine after domains
        self.partial_engine = PartialTransferEngine(self)
        
        logger.info("TransferEngine initialized (production-ready) with bounded data structures")
    
    def _evict_least_used_effect(self):
        """
        Evict least used effect from library (FIXED: effect size limit)
        """
        if not self.effect_library:
            return
        
        # Find effect with lowest importance
        min_effect_id = min(self.effect_library.keys(),
                           key=lambda k: self.effect_library[k].importance)
        
        del self.effect_library[min_effect_id]
        logger.debug("Evicted effect %s from library", min_effect_id)
    
    def _evict_least_used_domain(self):
        """
        Evict domain with fewest capabilities (FIXED: domain size limit)
        """
        if not self.domain_characteristics:
            return
        
        # Find domain with fewest capabilities
        min_domain = min(self.domain_characteristics.keys(),
                        key=lambda k: len(self.domain_characteristics[k].capabilities))
        
        del self.domain_characteristics[min_domain]
        logger.debug("Evicted domain %s from characteristics", min_domain)
    
    def calculate_effect_overlap(self, concept, target_domain: str) -> float:
        """
        Calculate effect overlap between concept and target domain
        
        Args:
            concept: Concept to transfer
            target_domain: Target domain
            
        Returns:
            Overlap score [0, 1]
        """
        with self._lock:
            # Get concept effects
            concept_effects = self._extract_concept_effects(concept)
            
            # Get domain characteristics
            domain_chars = self.domain_characteristics.get(
                target_domain,
                DomainCharacteristics(name=target_domain)
            )
            
            if not concept_effects:
                return 0.5  # Default overlap for unknown effects
            
            # Calculate overlap
            overlap_scores = []
            
            for effect in concept_effects:
                # Check if effect can be realized in target domain
                if effect.effect_type == EffectType.PRIMARY:
                    # Primary effects must be supported
                    if self._is_effect_supported(effect, domain_chars):
                        overlap_scores.append(1.0)
                    else:
                        overlap_scores.append(0.0)
                elif effect.effect_type == EffectType.SECONDARY:
                    # Secondary effects are optional
                    if self._is_effect_supported(effect, domain_chars):
                        overlap_scores.append(0.7)
                    else:
                        overlap_scores.append(0.3)
                else:
                    # Side effects and prerequisites
                    if self._is_effect_compatible(effect, domain_chars):
                        overlap_scores.append(0.5)
                    else:
                        overlap_scores.append(0.1)
            
            # Weight by importance
            weighted_sum = 0.0
            weight_total = 0.0
            
            for effect, score in zip(concept_effects, overlap_scores):
                weight = effect.importance
                weighted_sum += score * weight
                weight_total += weight
            
            if weight_total > 0:
                overlap = weighted_sum / weight_total
            else:
                overlap = np.mean(overlap_scores) if overlap_scores else 0.5
            
            logger.debug("Effect overlap for concept in domain %s: %.2f", target_domain, overlap)
            
            return overlap
    
    def validate_full_transfer(self, concept, source: str, target: str) -> TransferDecision:
        """
        Validate full transfer compatibility
        
        Args:
            concept: Concept to transfer
            source: Source domain
            target: Target domain
            
        Returns:
            Transfer decision
        """
        with self._lock:
            # SAFETY: Validate transfer request
            if self.safety_validator:
                try:
                    if hasattr(self.safety_validator, 'validate_transfer'):
                        transfer_check = self.safety_validator.validate_transfer(concept, source, target)
                        if not transfer_check.get('safe', True):
                            logger.warning("Unsafe transfer blocked: %s", transfer_check.get('reason', 'unknown'))
                            self.safety_blocks['unsafe_transfer'] += 1
                            return TransferDecision(
                                type=TransferType.BLOCKED,
                                confidence=0.0,
                                reasoning=[f"Safety: {transfer_check.get('reason', 'unsafe transfer')}"]
                            )
                except Exception as e:
                    logger.debug("Error validating transfer: %s", e)
            
            decision = TransferDecision(
                type=TransferType.FULL,
                confidence=1.0
            )
            
            # Calculate effect overlap
            overlap = self.calculate_effect_overlap(concept, target)
            
            if overlap < self.full_transfer_threshold:
                # Not eligible for full transfer
                decision.type = TransferType.BLOCKED
                decision.confidence = overlap
                decision.reasoning.append(f"Effect overlap {overlap:.2f} below threshold {self.full_transfer_threshold}")
                return decision
            
            # Check domain compatibility
            compatibility = self._calculate_domain_compatibility(source, target)
            
            if compatibility < 0.7:
                decision.confidence *= compatibility
                decision.reasoning.append(f"Domain compatibility reduced confidence: {compatibility:.2f}")
            
            # Check for required constraints
            constraints = self._identify_constraints(concept, source, target)
            decision.constraints.extend(constraints)
            
            # Calculate risk
            risk = self._assess_transfer_risk(concept, source, target)
            decision.risk_assessment = risk
            
            # Adjust confidence based on risk
            overall_risk = np.mean(list(risk.values())) if risk else 0.0
            decision.confidence *= (1.0 - overall_risk * 0.5)
            
            # Verify all prerequisites
            if not self._verify_prerequisites(concept, target):
                decision.type = TransferType.CONDITIONAL
                decision.reasoning.append("Prerequisites not fully met - conditional transfer")
            
            # FIXED: Query world model for causal dependencies
            if self.world_model:
                try:
                    self._check_world_model_constraints(concept, source, target, decision)
                except Exception as e:
                    logger.debug("Failed to check world model constraints: %s", e)
            
            decision.reasoning.append(f"Full transfer validated with {len(constraints)} constraints")
            
            return decision
    
    def _check_world_model_constraints(self, concept, source: str, target: str, decision: TransferDecision):
        """
        Check world model for causal constraints - FIXED: New integration method
        
        Args:
            concept: Concept being transferred
            source: Source domain
            target: Target domain
            decision: Decision to update with constraints
        """
        if not self.world_model or not hasattr(self.world_model, 'causal_graph'):
            return
        
        try:
            # Check if source and target domains have causal relationships
            source_node = f"domain_{source}"
            target_node = f"domain_{target}"
            
            # If there's a causal path, it might indicate compatibility or constraints
            if self.world_model.causal_graph.has_node(source_node) and \
               self.world_model.causal_graph.has_node(target_node):
                
                # Check for negative causal effects (incompatibilities)
                if hasattr(self.world_model.causal_graph, 'edges'):
                    for edge_key, edge in self.world_model.causal_graph.edges.items():
                        if edge.cause == source_node and edge.effect == target_node:
                            # Direct relationship exists
                            if edge.strength < 0:
                                # Negative relationship - add constraint
                                constraint = Constraint(
                                    constraint_id=f"causal_constraint_{source}_{target}",
                                    constraint_type=ConstraintType.INVARIANT,
                                    description=f"Causal incompatibility: {source} -> {target}",
                                    condition=f"strength({source},{target}) >= 0",
                                    severity=abs(edge.strength)
                                )
                                decision.constraints.append(constraint)
                                decision.confidence *= (1 - abs(edge.strength) * 0.3)
                
                logger.debug("Applied world model constraints for transfer %s -> %s", source, target)
        except Exception as e:
            logger.debug("Error checking world model constraints: %s", e)
    
    def validate_partial_transfer(self, concept, source: str, target: str) -> TransferDecision:
        """
        Validate partial transfer with mitigations
        
        Args:
            concept: Concept to transfer
            source: Source domain  
            target: Target domain
            
        Returns:
            Transfer decision
        """
        with self._lock:
            # SAFETY: Validate partial transfer
            if self.safety_validator:
                try:
                    if hasattr(self.safety_validator, 'validate_transfer'):
                        transfer_check = self.safety_validator.validate_transfer(concept, source, target)
                        if not transfer_check.get('safe', True):
                            logger.warning("Unsafe partial transfer blocked: %s", transfer_check.get('reason', 'unknown'))
                            self.safety_blocks['unsafe_partial_transfer'] += 1
                            return TransferDecision(
                                type=TransferType.BLOCKED,
                                confidence=0.0,
                                reasoning=[f"Safety: {transfer_check.get('reason', 'unsafe transfer')}"]
                            )
                except Exception as e:
                    logger.debug("Error validating partial transfer: %s", e)
            
            decision = TransferDecision(
                type=TransferType.PARTIAL,
                confidence=0.5
            )
            
            # Calculate effect overlap
            overlap = self.calculate_effect_overlap(concept, target)
            
            if overlap < self.partial_transfer_threshold:
                # Not even eligible for partial transfer
                decision.type = TransferType.BLOCKED
                decision.confidence = overlap
                decision.reasoning.append(f"Effect overlap {overlap:.2f} below partial threshold")
                return decision
            
            # Use partial transfer engine
            missing_effects = self.partial_engine.identify_missing_effects(concept, target)
            
            if missing_effects:
                # Generate mitigations
                mitigations = self.partial_engine.generate_mitigations(missing_effects)
                decision.mitigations.extend(mitigations)
                
                # Calculate constraints
                constraints = self.partial_engine.calculate_constraints(missing_effects)
                decision.constraints.extend(constraints)
                
                # Update confidence based on mitigation success likelihood
                mitigation_confidence = self._calculate_mitigation_confidence(mitigations)
                decision.confidence = overlap * mitigation_confidence
                
                decision.reasoning.append(f"Partial transfer with {len(mitigations)} mitigations")
            else:
                # No missing effects - upgrade to full transfer
                decision.type = TransferType.FULL
                decision.confidence = overlap
                decision.reasoning.append("No missing effects - upgraded to full transfer")
            
            # Calculate cost
            decision.estimated_cost = self._calculate_transfer_cost(mitigations)
            
            return decision
    
    def execute_transfer(self, concept, decision: TransferDecision, 
                        target_domain: str) -> Dict[str, Any]:
        """
        Execute concept transfer based on decision
        
        Args:
            concept: Concept to transfer
            decision: Transfer decision
            target_domain: Target domain
            
        Returns:
            Transfer result
        """
        with self._lock:
            result = {
                'success': False,
                'transferred_concept': None,
                'applied_mitigations': [],
                'active_constraints': [],
                'warnings': []
            }
            
            if not decision.is_transferable():
                result['warnings'].append("Transfer blocked")
                return result
            
            # SAFETY: Final safety check before execution
            if self.safety_validator:
                try:
                    if hasattr(self.safety_validator, 'validate_concept'):
                        concept_check = self.safety_validator.validate_concept(concept)
                        if not concept_check.get('safe', True):
                            logger.warning("Unsafe concept detected before transfer: %s", 
                                         concept_check.get('reason', 'unknown'))
                            self.safety_blocks['unsafe_concept_transfer'] += 1
                            result['warnings'].append(f"Safety: {concept_check.get('reason', 'unsafe concept')}")
                            return result
                except Exception as e:
                    logger.debug("Error in final safety check: %s", e)
            
            # Create transferred concept
            transferred = copy.deepcopy(concept)
            
            # Apply mitigations
            for mitigation in decision.mitigations:
                success = self._apply_mitigation(transferred, mitigation, target_domain)
                if success:
                    result['applied_mitigations'].append(mitigation.mitigation_id)
                else:
                    result['warnings'].append(f"Failed to apply mitigation: {mitigation.mitigation_id}")
            
            # Apply constraints
            for constraint in decision.constraints:
                self._apply_constraint(transferred, constraint)
                result['active_constraints'].append(constraint.constraint_id)
            
            # Update concept metadata
            if hasattr(transferred, 'metadata'):
                transferred.metadata['transfer_type'] = decision.type.value
                transferred.metadata['transfer_confidence'] = decision.confidence
                transferred.metadata['target_domain'] = target_domain
            
            # FIXED: Update world model with transfer
            if self.world_model:
                try:
                    self._update_world_model_for_transfer(concept, transferred, target_domain, decision)
                except Exception as e:
                    logger.debug("Failed to update world model for transfer: %s", e)
            
            # Record transfer
            self._record_transfer(concept, decision, target_domain, True)
            
            result['success'] = True
            result['transferred_concept'] = transferred
            
            self.total_transfers += 1
            self.successful_transfers += 1
            
            return result
    
    def rollback_transfer(self, transferred_concept, original_concept, target_domain: str) -> bool:
        """
        Rollback a failed transfer (FIXED: transfer failure handling)
        
        Args:
            transferred_concept: The transferred concept that failed
            original_concept: Original concept before transfer
            target_domain: Domain where transfer was attempted
            
        Returns:
            Success of rollback
        """
        with self._lock:
            try:
                # Remove from domain
                if target_domain in self.domain_characteristics:
                    domain = self.domain_characteristics[target_domain]
                    if hasattr(transferred_concept, 'concept_id'):
                        concept_id = transferred_concept.concept_id
                        # Remove from domain's concept list if tracked
                        if hasattr(domain, 'concepts'):
                            domain.concepts.discard(concept_id)
                
                # Record failure
                self.transfer_history.append({
                    'timestamp': time.time(),
                    'concept_id': getattr(original_concept, 'concept_id', str(original_concept)),
                    'target_domain': target_domain,
                    'transfer_type': 'rollback',
                    'success': False,
                    'reason': 'transfer_failed'
                })
                
                # Update world model if available
                if self.world_model:
                    try:
                        self._remove_transfer_from_world_model(transferred_concept, target_domain)
                    except Exception as e:
                        logger.debug("Failed to remove transfer from world model: %s", e)
                
                logger.info("Rolled back failed transfer to %s", target_domain)
                return True
                
            except Exception as e:
                logger.error("Rollback failed: %s", e)
                return False
    
    def _remove_transfer_from_world_model(self, concept, domain: str):
        """
        Remove transfer edges from world model (FIXED: transfer failure handling)
        
        Args:
            concept: Concept to remove from world model
            domain: Domain to remove edges from
        """
        if not self.world_model or not hasattr(self.world_model, 'causal_graph'):
            return
        
        concept_id = getattr(concept, 'concept_id', str(concept))
        domain_node = f"domain_{domain}"
        
        # Remove edge between domain and concept
        if hasattr(self.world_model.causal_graph, 'remove_edge'):
            try:
                self.world_model.causal_graph.remove_edge(domain_node, concept_id)
            except Exception as e:                pass  # Edge might not exist
    
    def _update_world_model_for_transfer(self, original_concept, transferred_concept, 
                                        target_domain: str, decision: TransferDecision):
        """
        Update world model after transfer - FIXED: New integration method
        
        Args:
            original_concept: Original concept
            transferred_concept: Transferred concept
            target_domain: Target domain
            decision: Transfer decision
        """
        if not self.world_model or not hasattr(self.world_model, 'causal_graph'):
            return
        
        try:
            original_id = getattr(original_concept, 'concept_id', str(original_concept))
            transferred_id = getattr(transferred_concept, 'concept_id', str(transferred_concept))
            domain_node = f"domain_{target_domain}"
            
            # Add transferred concept node
            if not self.world_model.causal_graph.has_node(transferred_id):
                self.world_model.causal_graph.add_node(transferred_id)
            
            # Link to target domain
            if not self.world_model.causal_graph.has_edge(domain_node, transferred_id):
                # Validate with safety
                if self.safety_validator:
                    try:
                        if hasattr(self.safety_validator, 'validate_causal_edge'):
                            edge_validation = self.safety_validator.validate_causal_edge(
                                domain_node, transferred_id, decision.confidence
                            )
                            if not edge_validation.get('safe', True):
                                return
                    except Exception as e:
                        logger.debug("Safety validation error: %s", e)
                        return
                
                self.world_model.causal_graph.add_edge(
                    domain_node,
                    transferred_id,
                    strength=decision.confidence,
                    evidence_type="transfer_engine"
                )
            
            # Link original to transferred
            if original_id != transferred_id:
                if not self.world_model.causal_graph.has_edge(original_id, transferred_id):
                    self.world_model.causal_graph.add_edge(
                        original_id,
                        transferred_id,
                        strength=decision.confidence,
                        evidence_type="concept_transfer"
                    )
            
            logger.debug("Updated world model for transfer to %s", target_domain)
        except Exception as e:
            logger.debug("Error updating world model for transfer: %s", e)
    
    def _extract_concept_effects(self, concept) -> List[ConceptEffect]:
        """
        Extract effects from concept with robust fallback (FIXED: improved extraction)
        
        Args:
            concept: Concept to extract effects from
            
        Returns:
            List of concept effects
        """
        effects = []
        
        # Priority 1: Explicit effects
        if hasattr(concept, 'effects') and concept.effects:
            return concept.effects
        
        # Priority 2: Grounded effects (from ConceptMapper)
        if hasattr(concept, 'grounded_effects') and concept.grounded_effects:
            for grounded_effect in concept.grounded_effects:
                effect = ConceptEffect(
                    effect_id=getattr(grounded_effect, 'effect_id', f"effect_{len(effects)}"),
                    effect_type=self._map_grounded_to_concept_effect_type(grounded_effect),
                    description=f"Grounded effect: {grounded_effect.effect_type.value}",
                    domain=getattr(concept, 'domain', 'general'),
                    importance=getattr(grounded_effect, 'confidence', 0.5),
                    confidence=getattr(grounded_effect, 'confidence', 0.5)
                )
                effects.append(effect)
                
                # FIXED: Add to effect library with size limit
                if effect.effect_id not in self.effect_library:
                    if len(self.effect_library) >= self.max_effects:
                        self._evict_least_used_effect()
                    self.effect_library[effect.effect_id] = effect
            
            return effects
        
        # Priority 3: Infer from features
        if hasattr(concept, 'features') and concept.features:
            for feature_name, feature_value in concept.features.items():
                # Determine importance based on feature characteristics
                importance = 0.5
                if 'critical' in feature_name.lower() or 'primary' in feature_name.lower():
                    importance = 0.9
                    effect_type_enum = EffectType.PRIMARY
                elif 'secondary' in feature_name.lower():
                    importance = 0.6
                    effect_type_enum = EffectType.SECONDARY
                else:
                    importance = 0.5
                    effect_type_enum = EffectType.SECONDARY
                
                effect = ConceptEffect(
                    effect_id=f"effect_{hashlib.md5(feature_name.encode()).hexdigest()[:8]}",
                    effect_type=effect_type_enum,
                    description=f"Feature-based effect: {feature_name}",
                    domain=getattr(concept, 'domain', 'general'),
                    importance=importance,
                    confidence=0.6
                )
                effects.append(effect)
                
                # FIXED: Add to effect library with size limit
                if effect.effect_id not in self.effect_library:
                    if len(self.effect_library) >= self.max_effects:
                        self._evict_least_used_effect()
                    self.effect_library[effect.effect_id] = effect
            
            return effects
        
        # Priority 4: Create minimal default effect
        logger.warning("No effects found for concept %s, creating default", 
                      getattr(concept, 'concept_id', 'unknown'))
        
        default_effect = ConceptEffect(
            effect_id=f"default_{getattr(concept, 'concept_id', 'unknown')}",
            effect_type=EffectType.PRIMARY,
            description="Default concept effect",
            domain=getattr(concept, 'domain', 'general'),
            importance=0.5,
            confidence=0.3
        )
        effects.append(default_effect)
        
        # FIXED: Add to effect library with size limit
        if default_effect.effect_id not in self.effect_library:
            if len(self.effect_library) >= self.max_effects:
                self._evict_least_used_effect()
            self.effect_library[default_effect.effect_id] = default_effect
        
        return effects
    
    def _map_grounded_to_concept_effect_type(self, grounded_effect) -> EffectType:
        """
        Map ConceptMapper effect types to TransferEngine effect types (FIXED: improved extraction)
        
        Args:
            grounded_effect: Grounded effect from ConceptMapper
            
        Returns:
            Mapped effect type
        """
        # Import to avoid circular dependency
        try:
            from concept_mapper import EffectType as MapperEffectType
            
            mapping = {
                MapperEffectType.PERFORMANCE: EffectType.PRIMARY,
                MapperEffectType.RESOURCE: EffectType.SECONDARY,
                MapperEffectType.BEHAVIORAL: EffectType.PRIMARY,
                MapperEffectType.STRUCTURAL: EffectType.SECONDARY,
                MapperEffectType.TEMPORAL: EffectType.SECONDARY
            }
            
            grounded_type = getattr(grounded_effect, 'effect_type', None)
            return mapping.get(grounded_type, EffectType.SECONDARY)
        except ImportError:
            return EffectType.SECONDARY
    
    def _is_effect_supported(self, effect: ConceptEffect, 
                            domain_chars: DomainCharacteristics) -> bool:
        """Check if effect is supported in domain"""
        # Check capabilities
        for capability in domain_chars.capabilities:
            if effect.description.lower() in capability.lower():
                return True
        
        # Check against limitations
        for limitation in domain_chars.limitations:
            if effect.description.lower() in limitation.lower():
                return False
        
        # Default based on effect type
        return effect.effect_type != EffectType.PRIMARY
    
    def _is_effect_compatible(self, effect: ConceptEffect,
                             domain_chars: DomainCharacteristics) -> bool:
        """Check if effect is compatible with domain"""
        # More lenient than supported - just needs to not conflict
        for limitation in domain_chars.limitations:
            if 'incompatible' in limitation.lower() and effect.description.lower() in limitation.lower():
                return False
        
        return True
    
    def _calculate_domain_compatibility(self, source: str, target: str) -> float:
        """
        Calculate compatibility between domains (FIXED: cache size limit)
        """
        with self._lock:
            # Check cache
            cache_key = (source, target)
            if cache_key in self.compatibility_cache:
                return self.compatibility_cache[cache_key]
            
            # Get domain characteristics
            source_chars = self.domain_characteristics.get(source)
            target_chars = self.domain_characteristics.get(target)
            
            if not source_chars or not target_chars:
                compatibility = 0.5  # Default for unknown domains
            elif target in source_chars.compatibility_matrix:
                compatibility = source_chars.compatibility_matrix[target]
            else:
                # Calculate based on capability overlap
                source_caps = source_chars.capabilities
                target_caps = target_chars.capabilities
                
                if source_caps and target_caps:
                    overlap = len(source_caps & target_caps)
                    union = len(source_caps | target_caps)
                    compatibility = overlap / union if union > 0 else 0.0
                else:
                    compatibility = 0.3
            
            # FIXED: Cache with size limit
            if len(self.compatibility_cache) < self.max_cache_size:
                self.compatibility_cache[cache_key] = compatibility
            else:
                # Evict random entry (could use LRU)
                evict_key = next(iter(self.compatibility_cache))
                del self.compatibility_cache[evict_key]
                self.compatibility_cache[cache_key] = compatibility
            
            return compatibility
    
    def _identify_constraints(self, concept, source: str, target: str) -> List[Constraint]:
        """Identify necessary constraints for transfer"""
        constraints = []
        
        # Domain-specific constraints
        target_chars = self.domain_characteristics.get(target)
        if target_chars:
            for limitation in target_chars.limitations:
                constraint = Constraint(
                    constraint_id=f"constraint_{hashlib.md5(limitation.encode()).hexdigest()[:8]}",
                    constraint_type=ConstraintType.INVARIANT,
                    description=f"Domain limitation: {limitation}",
                    condition=limitation,
                    severity=0.7,
                    domain_specific=True
                )
                constraints.append(constraint)
        
        # Resource constraints
        if hasattr(concept, 'resource_requirements'):
            for resource, requirement in concept.resource_requirements.items():
                constraint = Constraint(
                    constraint_id=f"resource_{resource}",
                    constraint_type=ConstraintType.RESOURCE,
                    description=f"Resource requirement: {resource}",
                    condition=f"{resource} >= {requirement}",
                    severity=0.8
                )
                constraints.append(constraint)
        
        return constraints
    
    def _assess_transfer_risk(self, concept, source: str, target: str) -> Dict[str, float]:
        """Assess risks of transfer"""
        risks = {}
        
        # Compatibility risk
        compatibility = self._calculate_domain_compatibility(source, target)
        risks['compatibility'] = 1.0 - compatibility
        
        # Complexity risk
        if hasattr(concept, 'complexity'):
            risks['complexity'] = min(1.0, concept.complexity / 10) if concept.complexity > 0 else 0
        
        # Effect coverage risk
        overlap = self.calculate_effect_overlap(concept, target)
        risks['effect_coverage'] = 1.0 - overlap
        
        return risks
    
    def _verify_prerequisites(self, concept, target: str) -> bool:
        """Verify all prerequisites are met"""
        if not hasattr(concept, 'prerequisites'):
            return True
        
        target_chars = self.domain_characteristics.get(target)
        if not target_chars:
            return False
        
        for prereq in concept.prerequisites:
            if prereq not in target_chars.capabilities:
                return False
        
        return True
    
    def _calculate_mitigation_confidence(self, mitigations: List[Mitigation]) -> float:
        """Calculate overall confidence in mitigations"""
        if not mitigations:
            return 1.0
        
        # Use geometric mean for confidence
        confidences = [m.confidence for m in mitigations]
        return float(np.prod(confidences) ** (1/len(confidences)))
    
    def _calculate_transfer_cost(self, mitigations: List[Mitigation]) -> float:
        """Calculate total cost of transfer"""
        base_cost = 1.0
        mitigation_cost = sum(m.cost for m in mitigations)
        return base_cost + mitigation_cost
    
    def _apply_mitigation(self, concept, mitigation: Mitigation, 
                         target_domain: str) -> bool:
        """Apply mitigation to concept"""
        try:
            if mitigation.mitigation_type == MitigationType.ADAPTATION:
                # Modify concept to fit domain
                if hasattr(concept, 'adapt'):
                    concept.adapt(target_domain)
                return True
            
            elif mitigation.mitigation_type == MitigationType.WRAPPER:
                # Wrap concept with additional functionality
                if hasattr(concept, 'metadata'):
                    concept.metadata['wrapper'] = mitigation.implementation
                return True
            
            elif mitigation.mitigation_type == MitigationType.MONITORING:
                # Add monitoring
                if hasattr(concept, 'metadata'):
                    concept.metadata['monitoring'] = True
                return True
            
            return False
            
        except Exception as e:
            logger.error("Failed to apply mitigation %s: %s", mitigation.mitigation_id, e)
            return False
    
    def _apply_constraint(self, concept, constraint: Constraint):
        """Apply constraint to concept"""
        if hasattr(concept, 'constraints'):
            concept.constraints.append(constraint)
        elif hasattr(concept, 'metadata'):
            if 'constraints' not in concept.metadata:
                concept.metadata['constraints'] = []
            concept.metadata['constraints'].append(constraint.to_dict())
    
    def _record_transfer(self, concept, decision: TransferDecision,
                        target_domain: str, success: bool):
        """Record transfer for history"""
        self.transfer_history.append({
            'timestamp': time.time(),
            'concept_id': getattr(concept, 'concept_id', str(concept)),
            'target_domain': target_domain,
            'transfer_type': decision.type.value,
            'confidence': decision.confidence,
            'success': success,
            'mitigations': len(decision.mitigations),
            'constraints': len(decision.constraints)
        })
    
    def _initialize_domains(self):
        """Initialize default domain characteristics"""
        default_domains = {
            'general': DomainCharacteristics(
                name='general',
                capabilities={'basic_computation', 'data_transformation', 'pattern_matching'},
                limitations={'no_real_time', 'no_hardware_access'}
            ),
            'optimization': DomainCharacteristics(
                name='optimization',
                capabilities={'gradient_computation', 'constraint_handling', 'objective_evaluation'},
                limitations={'high_dimensionality', 'non_convex_problems'}
            ),
            'control': DomainCharacteristics(
                name='control',
                capabilities={'feedback_loops', 'state_estimation', 'real_time_execution'},
                limitations={'stability_requirements', 'safety_critical'}
            )
        }
        
        self.domain_characteristics.update(default_domains)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics"""
        stats = {
            'total_transfers': self.total_transfers,
            'successful_transfers': self.successful_transfers,
            'success_rate': self.successful_transfers / max(1, self.total_transfers),
            'world_model_connected': self.world_model is not None,
            'domains_registered': len(self.domain_characteristics),
            'cached_compatibilities': len(self.compatibility_cache),
            'effects_in_library': len(self.effect_library),
            'max_effects': self.max_effects,
            'max_domains': self.max_domains,
            'max_cache_size': self.max_cache_size
        }
        
        # Add safety statistics
        if self.safety_validator:
            stats['safety'] = {
                'enabled': True,
                'blocks': dict(self.safety_blocks),
                'corrections': dict(self.safety_corrections),
                'total_blocks': sum(self.safety_blocks.values()),
                'total_corrections': sum(self.safety_corrections.values())
            }
        else:
            stats['safety'] = {'enabled': False}
        
        return stats


class MitigationLearner:
    """
    Learns which mitigations work best in which contexts (FIXED: mitigation learning)
    """
    
    def __init__(self):
        # FIXED: Mitigation outcomes with size limit
        self.mitigation_outcomes = {}  # Changed from defaultdict to regular dict
        self.max_outcome_keys = 1000
        
        # FIXED: Context performance with size limits
        self.context_performance = {}  # Changed from nested defaultdict to regular dict
        self.max_context_keys = 500
        self.max_context_entries = 100
        
        self._lock = threading.RLock()
    
    def _evict_oldest_outcome(self):
        """
        Evict outcome with lowest total count (FIXED: outcome size limit)
        """
        if not self.mitigation_outcomes:
            return
        
        # Find key with lowest total
        min_key = min(self.mitigation_outcomes.keys(),
                     key=lambda k: self.mitigation_outcomes[k]['total'])
        
        del self.mitigation_outcomes[min_key]
        logger.debug("Evicted mitigation outcome %s", min_key)
    
    def _evict_oldest_context(self):
        """
        Evict context with fewest entries (FIXED: context size limit)
        """
        if not self.context_performance:
            return
        
        # Find context key with fewest entries
        min_key = min(self.context_performance.keys(),
                     key=lambda k: sum(len(v) for v in self.context_performance[k].values()))
        
        del self.context_performance[min_key]
        logger.debug("Evicted context performance %s", min_key)
    
    def record_mitigation_outcome(self, mitigation: Mitigation, context: Dict[str, Any], 
                                  success: bool, metrics: Dict[str, float]):
        """
        Record outcome of applying a mitigation
        
        Args:
            mitigation: Mitigation that was applied
            context: Context in which it was applied
            success: Whether the mitigation succeeded
            metrics: Performance metrics
        """
        with self._lock:
            key = (mitigation.mitigation_type.value, mitigation.target_effect)
            
            # FIXED: Enforce outcome size limit
            if key not in self.mitigation_outcomes:
                if len(self.mitigation_outcomes) >= self.max_outcome_keys:
                    self._evict_oldest_outcome()
                self.mitigation_outcomes[key] = {'success': 0, 'failure': 0, 'total': 0}
            
            self.mitigation_outcomes[key]['total'] += 1
            if success:
                self.mitigation_outcomes[key]['success'] += 1
            else:
                self.mitigation_outcomes[key]['failure'] += 1
            
            # Track performance by context
            context_key = self._context_to_key(context)
            
            # FIXED: Enforce context performance size limits
            if key not in self.context_performance:
                if len(self.context_performance) >= self.max_context_keys:
                    self._evict_oldest_context()
                self.context_performance[key] = {}
            
            if context_key not in self.context_performance[key]:
                self.context_performance[key][context_key] = []
            
            # FIXED: Limit entries per context
            if len(self.context_performance[key][context_key]) >= self.max_context_entries:
                # Remove oldest entry
                self.context_performance[key][context_key].pop(0)
            
            self.context_performance[key][context_key].append({
                'success': success,
                'metrics': metrics,
                'timestamp': time.time()
            })
    
    def get_mitigation_confidence(self, mitigation_type: MitigationType, 
                                  target_effect: str, context: Dict[str, Any] = None) -> float:
        """
        Get learned confidence for a mitigation strategy
        
        Args:
            mitigation_type: Type of mitigation
            target_effect: Effect being targeted
            context: Optional context for context-specific confidence
            
        Returns:
            Confidence score [0, 1]
        """
        with self._lock:
            key = (mitigation_type.value, target_effect)
            
            if key not in self.mitigation_outcomes:
                return 0.5  # Default
            
            stats = self.mitigation_outcomes[key]
            if stats['total'] == 0:
                return 0.5
            
            # Base success rate
            base_confidence = stats['success'] / stats['total']
            
            # Adjust for context if provided
            if context:
                context_key = self._context_to_key(context)
                context_outcomes = self.context_performance.get(key, {}).get(context_key, [])
                
                if context_outcomes:
                    recent_outcomes = context_outcomes[-10:]  # Last 10
                    context_success = sum(1 for o in recent_outcomes if o['success']) / len(recent_outcomes)
                    # Blend base and context-specific
                    return base_confidence * 0.5 + context_success * 0.5
            
            return base_confidence
    
    def suggest_best_mitigation(self, target_effect: str, context: Dict[str, Any]) -> Optional[MitigationType]:
        """
        Suggest best mitigation type based on historical performance
        
        Args:
            target_effect: Effect to mitigate
            context: Context for suggestion
            
        Returns:
            Best mitigation type or None if no good option
        """
        with self._lock:
            best_type = None
            best_confidence = 0.0
            
            for mitigation_type in MitigationType:
                confidence = self.get_mitigation_confidence(mitigation_type, target_effect, context)
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_type = mitigation_type
            
            return best_type if best_confidence > 0.6 else None
    
    def _context_to_key(self, context: Dict[str, Any]) -> str:
        """
        Convert context to hashable key
        
        Args:
            context: Context dictionary
            
        Returns:
            Hashable key string
        """
        # Use relevant context fields
        relevant = {k: v for k, v in context.items() 
                   if k in ['source_domain', 'target_domain', 'complexity']}
        return json.dumps(relevant, sort_keys=True)


class PartialTransferEngine:
    """Handles transfers with <80% compatibility"""
    
    def __init__(self, parent_engine: TransferEngine):
        """
        Initialize partial transfer engine
        
        Args:
            parent_engine: Parent transfer engine
        """
        self.parent = parent_engine
        self.mitigation_templates = self._load_mitigation_templates()
        self.mitigation_learner = MitigationLearner()  # FIXED: Add mitigation learning
        
        logger.info("PartialTransferEngine initialized with learning")
    
    def identify_missing_effects(self, concept, target_domain: str) -> List[ConceptEffect]:
        """
        Identify effects that cannot transfer
        
        Args:
            concept: Concept to analyze
            target_domain: Target domain
            
        Returns:
            List of missing effects
        """
        missing = []
        
        # Get concept effects
        concept_effects = self.parent._extract_concept_effects(concept)
        
        # Get domain characteristics
        domain_chars = self.parent.domain_characteristics.get(
            target_domain,
            DomainCharacteristics(name=target_domain)
        )
        
        for effect in concept_effects:
            if not self.parent._is_effect_supported(effect, domain_chars):
                if effect.is_critical():
                    missing.append(effect)
                elif effect.importance > 0.5:
                    # Non-critical but important
                    missing.append(effect)
        
        logger.debug("Identified %d missing effects for domain %s", len(missing), target_domain)
        
        return missing
    
    def generate_mitigations(self, missing_effects: List[ConceptEffect]) -> List[Mitigation]:
        """
        Generate mitigations with learned confidence adjustments (FIXED: use learning)
        
        Args:
            missing_effects: Effects that need mitigation
            
        Returns:
            List of mitigations
        """
        mitigations = []
        
        for effect in missing_effects:
            # Get context for learning
            context = {
                'effect_type': effect.effect_type.value,
                'importance': effect.importance
            }
            
            # Try learned suggestion first
            suggested_type = self.mitigation_learner.suggest_best_mitigation(
                effect.effect_id, context
            )
            
            # Fallback to rule-based
            if not suggested_type:
                if effect.effect_type == EffectType.PRIMARY:
                    mitigation_type = MitigationType.ADAPTATION
                elif effect.effect_type == EffectType.SECONDARY:
                    mitigation_type = MitigationType.FALLBACK
                else:
                    mitigation_type = MitigationType.MONITORING
            else:
                mitigation_type = suggested_type
            
            # Generate mitigation with learned confidence
            mitigation = self._create_mitigation(effect, mitigation_type)
            
            # Override confidence with learned value
            learned_confidence = self.mitigation_learner.get_mitigation_confidence(
                mitigation_type, effect.effect_id, context
            )
            mitigation.confidence = learned_confidence
            
            mitigations.append(mitigation)
        
        return self._optimize_mitigations(mitigations)
    
    def calculate_constraints(self, missing_effects: List[ConceptEffect]) -> List[Constraint]:
        """
        Calculate constraints based on missing effects
        
        Args:
            missing_effects: Effects that are missing
            
        Returns:
            List of constraints
        """
        constraints = []
        
        for effect in missing_effects:
            # Create constraint for missing effect
            constraint = Constraint(
                constraint_id=f"missing_effect_{effect.effect_id}",
                constraint_type=ConstraintType.PRECONDITION if effect.effect_type == EffectType.PREREQUISITE
                               else ConstraintType.INVARIANT,
                description=f"Missing effect: {effect.description}",
                condition=f"not requires({effect.effect_id})",
                severity=effect.importance,
                domain_specific=True
            )
            
            constraints.append(constraint)
            
            # Add constraints for prerequisites
            for prereq in effect.prerequisites:
                prereq_constraint = Constraint(
                    constraint_id=f"prereq_{prereq}",
                    constraint_type=ConstraintType.PRECONDITION,
                    description=f"Prerequisite: {prereq}",
                    condition=f"available({prereq})",
                    severity=0.8
                )
                constraints.append(prereq_constraint)
        
        return constraints
    
    def _create_mitigation(self, effect: ConceptEffect, 
                          mitigation_type: MitigationType) -> Mitigation:
        """Create mitigation for missing effect"""
        mitigation_id = f"mit_{effect.effect_id}_{mitigation_type.value}"
        
        # Select template
        template = self.mitigation_templates.get(mitigation_type, {})
        
        mitigation = Mitigation(
            mitigation_id=mitigation_id,
            mitigation_type=mitigation_type,
            target_effect=effect.effect_id,
            description=f"Mitigate missing {effect.description}",
            implementation=self._generate_implementation(effect, mitigation_type),
            cost=template.get('base_cost', 1.0) * effect.importance,
            confidence=template.get('confidence', 0.7),
            prerequisites=effect.prerequisites
        )
        
        return mitigation
    
    def _generate_implementation(self, effect: ConceptEffect,
                                mitigation_type: MitigationType) -> Dict[str, Any]:
        """Generate implementation for mitigation"""
        implementation = {
            'type': mitigation_type.value,
            'target_effect': effect.effect_id
        }
        
        if mitigation_type == MitigationType.ADAPTATION:
            implementation['adaptation'] = {
                'method': 'transform',
                'parameters': {'effect_type': effect.effect_type.value}
            }
        elif mitigation_type == MitigationType.WRAPPER:
            implementation['wrapper'] = {
                'pre_process': True,
                'post_process': True,
                'validate': True
            }
        elif mitigation_type == MitigationType.FALLBACK:
            implementation['fallback'] = {
                'strategy': 'default',
                'timeout': 30
            }
        elif mitigation_type == MitigationType.MONITORING:
            implementation['monitoring'] = {
                'metrics': ['success_rate', 'latency'],
                'alert_threshold': 0.5
            }
        
        return implementation
    
    def _optimize_mitigations(self, mitigations: List[Mitigation]) -> List[Mitigation]:
        """Optimize set of mitigations"""
        # Remove redundant mitigations
        seen_effects = set()
        optimized = []
        
        # Sort by confidence and cost
        mitigations.sort(key=lambda m: m.confidence / m.cost, reverse=True)
        
        for mitigation in mitigations:
            if mitigation.target_effect not in seen_effects:
                optimized.append(mitigation)
                seen_effects.add(mitigation.target_effect)
        
        return optimized
    
    def _load_mitigation_templates(self) -> Dict[MitigationType, Dict[str, Any]]:
        """Load mitigation templates"""
        return {
            MitigationType.ADAPTATION: {
                'base_cost': 2.0,
                'confidence': 0.8,
                'complexity': 'medium'
            },
            MitigationType.WRAPPER: {
                'base_cost': 1.5,
                'confidence': 0.85,
                'complexity': 'low'
            },
            MitigationType.FALLBACK: {
                'base_cost': 1.0,
                'confidence': 0.6,
                'complexity': 'low'
            },
            MitigationType.MONITORING: {
                'base_cost': 0.5,
                'confidence': 0.9,
                'complexity': 'low'
            },
            MitigationType.CONSTRAINT: {
                'base_cost': 0.3,
                'confidence': 0.95,
                'complexity': 'very_low'
            }
        }