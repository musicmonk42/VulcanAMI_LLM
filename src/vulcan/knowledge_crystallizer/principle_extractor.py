"""
principle_extractor.py - Principle extraction from execution traces for Knowledge Crystallizer
Part of the VULCAN-AGI system
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque, Counter
import time
import json
import hashlib
from enum import Enum
import re
import copy
import threading
from abc import ABC, abstractmethod
from datetime import datetime

# Optional imports with fallbacks
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("scipy not available, some statistical features will be limited")

logger = logging.getLogger(__name__)


class PatternType(Enum):
    """Types of patterns"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    ITERATIVE = "iterative"
    HIERARCHICAL = "hierarchical"
    RECURSIVE = "recursive"
    BRANCHING = "branching"
    COMPOSITE = "composite"


class MetricType(Enum):
    """Types of metrics"""
    PERFORMANCE = "performance"
    ACCURACY = "accuracy"
    EFFICIENCY = "efficiency"
    RELIABILITY = "reliability"
    SCALABILITY = "scalability"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    QUALITY = "quality"


class ExtractionStrategy(Enum):
    """Extraction strategies"""
    CONSERVATIVE = "conservative"  # High confidence required
    BALANCED = "balanced"  # Default balanced approach
    AGGRESSIVE = "aggressive"  # Lower threshold, more principles
    EXPLORATORY = "exploratory"  # Experimental patterns


@dataclass
class Pattern:
    """Pattern representation"""
    pattern_type: PatternType
    components: List[Any]
    structure: Dict[str, Any] = field(default_factory=dict)
    frequency: float = 0.0
    confidence: float = 0.5
    complexity: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'pattern_type': self.pattern_type.value,
            'components': [str(c) for c in self.components],  # Convert to strings
            'structure': self.structure,
            'frequency': self.frequency,
            'confidence': self.confidence,
            'complexity': self.complexity,
            'metadata': self.metadata
        }
    
    def signature(self) -> str:
        """Get unique signature for pattern"""
        try:
            # Convert non-serializable to serializable
            serializable_components = []
            for c in self.components:
                if isinstance(c, np.ndarray):
                    serializable_components.append(('array', c.tolist()))
                elif hasattr(c, 'to_dict'):
                    try:
                        serializable_components.append(('dict', c.to_dict()))
                    except Exception as e:                        serializable_components.append(('str', str(c)))
                elif isinstance(c, (list, tuple)):
                    serializable_components.append(('list', [str(item) for item in c]))
                elif isinstance(c, dict):
                    serializable_components.append(('dict', {str(k): str(v) for k, v in c.items()}))
                else:
                    serializable_components.append(('str', str(c)))
            
            # Make structure serializable
            serializable_structure = {}
            for k, v in self.structure.items():
                if isinstance(v, (str, int, float, bool, type(None))):
                    serializable_structure[k] = v
                else:
                    serializable_structure[k] = str(v)
            
            content = json.dumps({
                'type': self.pattern_type.value,
                'components': serializable_components,
                'structure': serializable_structure
            }, sort_keys=True)
            return hashlib.md5(content.encode()).hexdigest()
        except Exception as e:
            logger.warning("Failed to create signature: %s", e)
            # Fallback to id-based hash
            fallback = f"{self.pattern_type.value}_{len(self.components)}_{self.complexity}"
            return hashlib.md5(fallback.encode()).hexdigest()
    
    def is_similar_to(self, other: 'Pattern', threshold: float = 0.7) -> bool:
        """Check if patterns are similar"""
        try:
            if not isinstance(other, Pattern):
                return False
            
            if self.pattern_type != other.pattern_type:
                return False
            
            # Component similarity
            if not self.components or not other.components:
                return False
            
            self_set = set(str(c) for c in self.components)
            other_set = set(str(c) for c in other.components)
            
            if not self_set or not other_set:
                return False
            
            common = self_set & other_set
            union = self_set | other_set
            
            if not union:
                return False
            
            similarity = len(common) / len(union)
            return similarity >= threshold
        except Exception as e:
            logger.warning("Error comparing patterns: %s", e)
            return False
    
    def __eq__(self, other):
        """Equality comparison"""
        if not isinstance(other, Pattern):
            return False
        return self.signature() == other.signature()
    
    def __hash__(self):
        """Make pattern hashable"""
        return hash(self.signature())


@dataclass
class Metric:
    """Performance metric"""
    name: str
    metric_type: MetricType
    value: float
    unit: Optional[str] = None
    threshold: Optional[float] = None
    is_success: bool = True
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'metric_type': self.metric_type.value,
            'value': self.value,
            'unit': self.unit,
            'threshold': self.threshold,
            'is_success': self.is_success,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }
    
    def meets_threshold(self) -> bool:
        """Check if metric meets threshold"""
        try:
            if self.threshold is None:
                return True
            
            if not isinstance(self.value, (int, float)):
                return False
            
            if not isinstance(self.threshold, (int, float)):
                return True
            
            if self.metric_type in [MetricType.ACCURACY, MetricType.RELIABILITY, MetricType.QUALITY]:
                return self.value >= self.threshold
            elif self.metric_type in [MetricType.LATENCY]:
                return self.value <= self.threshold  # Lower is better
            elif self.metric_type == MetricType.EFFICIENCY:
                return self.value >= self.threshold
            else:
                return True
        except Exception as e:
            logger.warning("Error checking threshold: %s", e)
            return True
    
    def normalize_value(self, min_val: float = 0, max_val: float = 1) -> float:
        """Normalize metric value to [0, 1]"""
        try:
            if not isinstance(self.value, (int, float)):
                return 0.5
            
            if not isinstance(min_val, (int, float)) or not isinstance(max_val, (int, float)):
                return 0.5
            
            if max_val == min_val:
                return 0.5
            
            if max_val < min_val:
                min_val, max_val = max_val, min_val
            
            normalized = (self.value - min_val) / (max_val - min_val)
            return max(0.0, min(1.0, normalized))
        except Exception as e:
            logger.warning("Error normalizing value: %s", e)
            return 0.5


@dataclass
class ExecutionTrace:
    """Execution trace for analysis"""
    trace_id: str
    actions: List[Dict[str, Any]]
    outcomes: Dict[str, Any]
    context: Dict[str, Any]
    metrics: List[Metric] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    success: bool = True
    domain: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)
    patterns: List[Pattern] = field(default_factory=list)  # Detected patterns
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'trace_id': self.trace_id,
            'actions': self.actions,
            'outcomes': self.outcomes,
            'context': self.context,
            'metrics': [m.to_dict() for m in self.metrics],
            'timestamp': self.timestamp,
            'success': self.success,
            'domain': self.domain,
            'metadata': self.metadata,
            'patterns': [p.to_dict() for p in self.patterns]
        }
    
    def get_duration(self) -> float:
        """Get execution duration if available"""
        try:
            if 'execution_time' in self.outcomes:
                return float(self.outcomes['execution_time'])
            elif 'start_time' in self.metadata and 'end_time' in self.metadata:
                return float(self.metadata['end_time']) - float(self.metadata['start_time'])
            return 0.0
        except (ValueError, TypeError):
            return 0.0
    
    def get_action_sequence(self) -> List[str]:
        """Get sequence of action types"""
        try:
            return [a.get('type', 'unknown') for a in self.actions if isinstance(a, dict)]
        except Exception as e:            return []
    
    def __eq__(self, other):
        """Equality comparison based on trace_id"""
        if not isinstance(other, ExecutionTrace):
            return False
        return self.trace_id == other.trace_id
    
    def __hash__(self):
        """Make trace hashable"""
        return hash(self.trace_id)


@dataclass
class SuccessFactor:
    """Factor contributing to success"""
    factor_type: str
    importance: float
    evidence_count: int
    conditions: List[str] = field(default_factory=list)
    metrics: List[Metric] = field(default_factory=list)
    correlation: float = 0.0  # Correlation with success
    causality_score: float = 0.0  # Estimated causal influence
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'factor_type': self.factor_type,
            'importance': self.importance,
            'evidence_count': self.evidence_count,
            'conditions': self.conditions,
            'metrics': [m.to_dict() for m in self.metrics],
            'correlation': self.correlation,
            'causality_score': self.causality_score,
            'metadata': self.metadata
        }
    
    def update_importance(self, new_evidence: float):
        """Update importance with new evidence"""
        try:
            if not isinstance(new_evidence, (int, float)):
                return
            
            # Exponential moving average
            alpha = 0.3
            self.importance = alpha * new_evidence + (1 - alpha) * self.importance
            self.importance = max(0.0, min(1.0, self.importance))  # Clamp to [0, 1]
            self.evidence_count += 1
        except Exception as e:
            logger.warning("Error updating importance: %s", e)


@dataclass
class PrincipleCandidate:
    """Candidate principle before validation"""
    pattern: Pattern
    evidence: List[ExecutionTrace] = field(default_factory=list)
    origin_domain: str = "general"
    success_indicators: List[Metric] = field(default_factory=list)
    required_metrics: List[str] = field(default_factory=list)
    success_factors: List[SuccessFactor] = field(default_factory=list)
    confidence: float = 0.5
    stability: float = 0.5  # How stable the pattern is
    generalizability: float = 0.5  # How well it generalizes
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_evidence(self, trace: ExecutionTrace):
        """Add supporting evidence"""
        if trace is None or not isinstance(trace, ExecutionTrace):
            return
        
        if trace not in self.evidence:
            self.evidence.append(trace)
            self._update_confidence()
            self._update_stability()
    
    def _update_confidence(self):
        """Update confidence based on evidence"""
        try:
            if not self.evidence:
                self.confidence = 0.0
                return
            
            # Base confidence on evidence count
            evidence_factor = min(1.0, len(self.evidence) / 10)
            
            # Success rate
            success_count = sum(1 for e in self.evidence if e and e.success)
            success_rate = success_count / len(self.evidence)
            
            # Consistency of metrics
            metric_consistency = self._calculate_metric_consistency()
            
            # Pattern complexity penalty
            complexity_penalty = 1.0 / (1.0 + self.pattern.complexity * 0.1)
            
            # Combined confidence
            self.confidence = (
                evidence_factor * 0.25 +
                success_rate * 0.4 +
                metric_consistency * 0.25 +
                complexity_penalty * 0.1
            )
            
            self.confidence = max(0.0, min(1.0, self.confidence))
        except Exception as e:
            logger.warning("Error updating confidence: %s", e)
            self.confidence = 0.5
    
    def _update_stability(self):
        """Update stability score"""
        try:
            if len(self.evidence) < 2:
                self.stability = 0.5
                return
            
            # Check consistency across time
            timestamps = [e.timestamp for e in self.evidence if e]
            if not timestamps or len(timestamps) < 2:
                self.stability = 0.5
                return
            
            time_spread = max(timestamps) - min(timestamps)
            
            if time_spread > 0:
                # Check for regular occurrence
                intervals = [timestamps[i+1] - timestamps[i] 
                            for i in range(len(timestamps)-1)]
                if intervals:
                    mean_interval = np.mean(intervals)
                    if mean_interval > 0:
                        cv = np.std(intervals) / mean_interval
                        self.stability = 1.0 / (1.0 + cv)
                    else:
                        self.stability = 0.5
                else:
                    self.stability = 0.5
            else:
                self.stability = 0.5
            
            self.stability = max(0.0, min(1.0, self.stability))
        except Exception as e:
            logger.warning("Error updating stability: %s", e)
            self.stability = 0.5
    
    def _calculate_metric_consistency(self) -> float:
        """Calculate consistency of metrics across evidence"""
        try:
            if not self.success_indicators:
                return 0.5
            
            # Check how consistently metrics appear
            metric_counts = Counter()
            for trace in self.evidence:
                if trace and hasattr(trace, 'metrics'):
                    for metric in trace.metrics:
                        if metric:
                            metric_counts[metric.name] += 1
            
            if not metric_counts:
                return 0.5
            
            # Consistency is high if metrics appear in most traces
            avg_appearance = np.mean(list(metric_counts.values()))
            consistency = avg_appearance / max(1, len(self.evidence))
            
            return min(1.0, consistency)
        except Exception as e:
            logger.warning("Error calculating metric consistency: %s", e)
            return 0.5
    
    def calculate_generalizability(self) -> float:
        """Calculate how well the pattern generalizes"""
        try:
            if not self.evidence:
                return 0.0
            
            # Domain diversity
            domains = set(e.domain for e in self.evidence if e)
            domain_diversity = len(domains) / max(1, len(self.evidence))
            
            # Context diversity
            context_keys = set()
            for e in self.evidence:
                if e and hasattr(e, 'context') and isinstance(e.context, dict):
                    context_keys.update(e.context.keys())
            context_diversity = min(1.0, len(context_keys) / 10)
            
            # Success consistency
            success_count = sum(1 for e in self.evidence if e and e.success)
            success_rate = success_count / len(self.evidence)
            
            self.generalizability = (
                domain_diversity * 0.4 +
                context_diversity * 0.3 +
                success_rate * 0.3
            )
            
            self.generalizability = max(0.0, min(1.0, self.generalizability))
            return self.generalizability
        except Exception as e:
            logger.warning("Error calculating generalizability: %s", e)
            self.generalizability = 0.5
            return 0.5


@dataclass
class CrystallizedPrinciple:
    """Validated, reusable principle"""
    id: str
    name: str
    description: str
    core_pattern: Pattern
    applicable_domains: List[str] = field(default_factory=list)
    contraindicated_domains: List[str] = field(default_factory=list)
    confidence: float = 0.5
    version: float = 1.0
    measurement_requirements: List[str] = field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0
    last_updated: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    domain: str = "general"
    problem_types: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)  # Other principle IDs
    tags: List[str] = field(default_factory=list)
    
    def apply(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Apply principle to problem"""
        try:
            solution = {
                'principle_id': self.id,
                'pattern': self.core_pattern.to_dict(),
                'confidence': self.confidence,
                'approach': self.description,
                'requirements': self.measurement_requirements
            }
            
            # Check domain compatibility
            if problem and isinstance(problem, dict) and 'domain' in problem:
                problem_domain = problem['domain']
                
                if problem_domain in self.contraindicated_domains:
                    solution['warning'] = f"Domain {problem_domain} is contraindicated"
                    solution['confidence'] *= 0.5
                elif (problem_domain not in self.applicable_domains and 
                      'general' not in self.applicable_domains):
                    solution['warning'] = f"Domain {problem_domain} not validated"
                    solution['confidence'] *= 0.8
            
            return solution
        except Exception as e:
            logger.error("Error applying principle: %s", e)
            return {
                'principle_id': self.id,
                'error': str(e),
                'confidence': 0.0
            }
    
    def update_stats(self, success: bool):
        """Update success/failure statistics"""
        try:
            if success:
                self.success_count += 1
            else:
                self.failure_count += 1
            
            # Update confidence based on performance
            total = self.success_count + self.failure_count
            if total > 0:
                success_rate = self.success_count / total
                # Bayesian update with prior
                self.confidence = (self.confidence + success_rate) / 2
                self.confidence = max(0.0, min(1.0, self.confidence))
            
            self.last_updated = time.time()
        except Exception as e:
            logger.warning("Error updating stats: %s", e)
    
    def get_success_rate(self) -> float:
        """Get success rate"""
        try:
            total = self.success_count + self.failure_count
            return self.success_count / total if total > 0 else 0.5
        except Exception as e:            return 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'core_pattern': self.core_pattern.to_dict(),
            'applicable_domains': self.applicable_domains,
            'contraindicated_domains': self.contraindicated_domains,
            'confidence': self.confidence,
            'version': self.version,
            'measurement_requirements': self.measurement_requirements,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'success_rate': self.get_success_rate(),
            'last_updated': self.last_updated,
            'metadata': self.metadata,
            'domain': self.domain,
            'problem_types': self.problem_types,
            'dependencies': self.dependencies,
            'tags': self.tags
        }


# Alias for compatibility
Principle = CrystallizedPrinciple


class PrincipleExtractor:
    """Extracts principles from execution traces"""
    
    def __init__(self,
                 min_evidence_count: int = 3,
                 min_confidence: float = 0.6,
                 strategy: ExtractionStrategy = ExtractionStrategy.BALANCED):
        """
        Initialize principle extractor
        
        Args:
            min_evidence_count: Minimum evidence required
            min_confidence: Minimum confidence threshold
            strategy: Extraction strategy
        """
        self.min_evidence_count = max(1, min_evidence_count)
        self.min_confidence = max(0.0, min(1.0, min_confidence))
        self.strategy = strategy
        
        # Adjust thresholds based on strategy
        self._adjust_thresholds()
        
        # Pattern detection
        self.pattern_detector = PatternDetector()
        
        # Success analysis
        self.success_analyzer = SuccessAnalyzer()
        
        # Abstraction engine
        self.abstraction_engine = AbstractionEngine()
        
        # Tracking
        self.extraction_history = deque(maxlen=1000)
        self.candidate_pool = {}  # pattern_signature -> PrincipleCandidate
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Statistics
        self.total_extracted = 0
        self.total_crystallized = 0
        
        # Cache cleanup timer
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # 5 minutes
        
        logger.info("PrincipleExtractor initialized (scipy: %s, strategy: %s)", 
                   SCIPY_AVAILABLE, strategy.value)
    
    def _adjust_thresholds(self):
        """Adjust thresholds based on strategy"""
        if self.strategy == ExtractionStrategy.CONSERVATIVE:
            self.min_evidence_count = max(5, self.min_evidence_count)
            self.min_confidence = max(0.8, self.min_confidence)
        elif self.strategy == ExtractionStrategy.AGGRESSIVE:
            self.min_evidence_count = max(1, self.min_evidence_count - 1)
            self.min_confidence = max(0.4, self.min_confidence - 0.2)
        elif self.strategy == ExtractionStrategy.EXPLORATORY:
            self.min_evidence_count = 1
            self.min_confidence = 0.3
    
    def _cleanup_cache(self):
        """Periodically cleanup cache"""
        try:
            current_time = time.time()
            if current_time - self._last_cleanup > self._cleanup_interval:
                # Remove low-confidence candidates
                to_remove = []
                for sig, candidate in self.candidate_pool.items():
                    if candidate.confidence < 0.3 or len(candidate.evidence) == 0:
                        to_remove.append(sig)
                
                for sig in to_remove:
                    del self.candidate_pool[sig]
                
                self._last_cleanup = current_time
                logger.debug("Cleaned up %d candidates from pool", len(to_remove))
        except Exception as e:
            logger.warning("Error during cache cleanup: %s", e)
    
    def extract_from_trace(self, execution_trace: ExecutionTrace) -> List[CrystallizedPrinciple]:
        """
        Extract principles from single execution trace
        
        Args:
            execution_trace: Execution trace
            
        Returns:
            List of extracted principles
        """
        with self.lock:
            try:
                # Validate input
                if execution_trace is None:
                    return []
                
                # Ensure execution trace has required attributes
                if not isinstance(execution_trace, ExecutionTrace):
                    # Convert dict to ExecutionTrace if needed
                    if isinstance(execution_trace, dict):
                        execution_trace = ExecutionTrace(
                            trace_id=execution_trace.get('trace_id', f'trace_{time.time()}'),
                            actions=execution_trace.get('actions', []),
                            outcomes=execution_trace.get('outcomes', {}),
                            context=execution_trace.get('context', {}),
                            success=execution_trace.get('success', True),
                            domain=execution_trace.get('domain', 'general'),
                            metadata=execution_trace.get('metadata', {})
                        )
                    else:
                        logger.warning("Invalid trace type: %s", type(execution_trace))
                        return []
                
                # Periodic cache cleanup
                self._cleanup_cache()
                
                # Extract candidates
                candidates = self.extract_candidates(execution_trace)
                
                # Filter by confidence
                valid_candidates = [
                    c for c in candidates 
                    if c and c.confidence >= self.min_confidence
                ]
                
                # Convert to crystallized principles
                principles = []
                for candidate in valid_candidates:
                    try:
                        principle = self._crystallize_candidate(candidate)
                        if principle:
                            principles.append(principle)
                            self.total_crystallized += 1
                    except Exception as e:
                        logger.error("Error crystallizing candidate: %s", e)
                
                # Track extraction
                self.extraction_history.append({
                    'trace_id': execution_trace.trace_id,
                    'candidates_found': len(candidates),
                    'principles_extracted': len(principles),
                    'timestamp': time.time()
                })
                
                logger.info("Extracted %d principles from trace %s",
                           len(principles), execution_trace.trace_id)
                
                return principles
            except Exception as e:
                logger.error("Error extracting from trace: %s", e)
                return []
    
    def extract_from_batch(self, traces: List[ExecutionTrace]) -> List[CrystallizedPrinciple]:
        """
        Extract principles from multiple traces
        
        Args:
            traces: List of execution traces
            
        Returns:
            List of extracted principles
        """
        with self.lock:
            try:
                if not traces:
                    return []
                
                all_candidates = []
                
                # Process each trace
                for trace in traces:
                    if trace:
                        try:
                            candidates = self.extract_candidates(trace)
                            all_candidates.extend(candidates)
                        except Exception as e:
                            logger.error("Error processing trace %s: %s", 
                                       getattr(trace, 'trace_id', 'unknown'), e)
                
                # Merge similar candidates
                merged = self._merge_candidates(all_candidates)
                
                # Filter and crystallize
                principles = []
                for candidate in merged:
                    try:
                        if (candidate and 
                            candidate.confidence >= self.min_confidence and
                            len(candidate.evidence) >= self.min_evidence_count):
                            principle = self._crystallize_candidate(candidate)
                            if principle:
                                principles.append(principle)
                                self.total_crystallized += 1
                    except Exception as e:
                        logger.error("Error crystallizing candidate: %s", e)
                
                logger.info("Extracted %d principles from %d traces", 
                           len(principles), len(traces))
                
                return principles
            except Exception as e:
                logger.error("Error extracting from batch: %s", e)
                return []
    
    def extract_candidates(self, execution_trace: ExecutionTrace) -> List[PrincipleCandidate]:
        """
        Extract candidate principles from execution trace
        
        Args:
            execution_trace: Execution trace to analyze
            
        Returns:
            List of principle candidates
        """
        try:
            if not execution_trace or not isinstance(execution_trace, ExecutionTrace):
                return []
            
            candidates = []
            
            # Detect patterns in trace
            patterns = self.pattern_detector.detect_patterns(execution_trace)
            
            # Store detected patterns in trace
            execution_trace.patterns = patterns
            
            # Analyze success factors
            success_factors = self.analyze_success_factors(execution_trace)
            
            # Create candidates for each pattern
            for pattern in patterns:
                if not pattern:
                    continue
                
                try:
                    # Check if pattern already exists in pool
                    signature = pattern.signature()
                    
                    if signature in self.candidate_pool:
                        # Update existing candidate
                        candidate = self.candidate_pool[signature]
                        candidate.add_evidence(execution_trace)
                    else:
                        # Create new candidate
                        candidate = PrincipleCandidate(
                            pattern=pattern,
                            evidence=[execution_trace],
                            origin_domain=execution_trace.domain,
                            success_indicators=self._extract_success_indicators(execution_trace),
                            required_metrics=self._identify_required_metrics(execution_trace),
                            success_factors=success_factors
                        )
                        
                        # Calculate initial confidence
                        candidate._update_confidence()
                        
                        # Add to pool
                        self.candidate_pool[signature] = candidate
                    
                    candidates.append(candidate)
                except Exception as e:
                    logger.error("Error processing pattern: %s", e)
            
            self.total_extracted += len(candidates)
            
            return candidates
        except Exception as e:
            logger.error("Error extracting candidates: %s", e)
            return []
    
    def analyze_success_factors(self, trace: ExecutionTrace) -> List[SuccessFactor]:
        """
        Analyze factors contributing to success
        
        Args:
            trace: Execution trace
            
        Returns:
            List of success factors
        """
        try:
            if not trace:
                return []
            return self.success_analyzer.analyze(trace)
        except Exception as e:
            logger.error("Error analyzing success factors: %s", e)
            return []
    
    def abstract_to_principle(self, success_factors: List[SuccessFactor]) -> Optional[CrystallizedPrinciple]:
        """
        Abstract success factors to reusable principle
        
        Args:
            success_factors: List of success factors
            
        Returns:
            Crystallized principle or None
        """
        try:
            if not success_factors:
                return None
            
            # Use abstraction engine
            abstracted = self.abstraction_engine.abstract(success_factors)
            
            if abstracted and isinstance(abstracted, dict):
                principle = CrystallizedPrinciple(
                    id=f"principle_{int(time.time())}_{np.random.randint(1000)}",
                    name=abstracted.get('name', 'Unknown Principle'),
                    description=abstracted.get('description', 'No description'),
                    core_pattern=abstracted.get('pattern', Pattern(
                        pattern_type=PatternType.SEQUENTIAL,
                        components=['default']
                    )),
                    applicable_domains=abstracted.get('domains', ['general']),
                    confidence=abstracted.get('confidence', 0.5),
                    measurement_requirements=abstracted.get('requirements', []),
                    domain=abstracted.get('domains', ['general'])[0] if abstracted.get('domains') else 'general',
                    tags=abstracted.get('tags', [])
                )
                
                return principle
            
            return None
        except Exception as e:
            logger.error("Error abstracting to principle: %s", e)
            return None
    
    def calculate_principle_confidence(self, evidence: List[ExecutionTrace]) -> float:
        """
        Calculate confidence score for principle
        
        Args:
            evidence: Supporting evidence
            
        Returns:
            Confidence score [0, 1]
        """
        try:
            if not evidence:
                return 0.0
            
            # Factors for confidence calculation
            factors = []
            
            # 1. Evidence count factor
            evidence_factor = min(1.0, len(evidence) / 10)
            factors.append(('evidence_count', evidence_factor, 0.15))
            
            # 2. Success rate
            success_count = sum(1 for e in evidence if e and e.success)
            success_rate = success_count / len(evidence)
            factors.append(('success_rate', success_rate, 0.35))
            
            # 3. Domain consistency
            domains = [e.domain for e in evidence if e]
            if domains:
                domain_consistency = 1.0 - (len(set(domains)) - 1) / max(1, len(domains))
            else:
                domain_consistency = 0.5
            factors.append(('domain_consistency', domain_consistency, 0.15))
            
            # 4. Metric stability
            metric_stability = self._calculate_metric_stability(evidence)
            factors.append(('metric_stability', metric_stability, 0.15))
            
            # 5. Temporal consistency
            temporal_consistency = self._calculate_temporal_consistency(evidence)
            factors.append(('temporal_consistency', temporal_consistency, 0.1))
            
            # 6. Statistical significance (if scipy available)
            if SCIPY_AVAILABLE:
                significance = self._calculate_statistical_significance(evidence)
                factors.append(('statistical_significance', significance, 0.1))
            else:
                factors.append(('statistical_significance', 0.5, 0.1))
            
            # Calculate weighted confidence
            confidence = sum(value * weight for _, value, weight in factors)
            
            logger.debug("Calculated confidence %.2f from %d evidence items", 
                        confidence, len(evidence))
            
            return min(1.0, max(0.0, confidence))
        except Exception as e:
            logger.error("Error calculating confidence: %s", e)
            return 0.5
    
    def _merge_candidates(self, candidates: List[PrincipleCandidate]) -> List[PrincipleCandidate]:
        """Merge similar candidates"""
        try:
            if not candidates:
                return []
            
            merged = {}
            
            for candidate in candidates:
                if not candidate:
                    continue
                
                try:
                    signature = candidate.pattern.signature()
                    
                    if signature in merged:
                        # Merge evidence
                        existing = merged[signature]
                        for trace in candidate.evidence:
                            if trace and trace not in existing.evidence:
                                existing.add_evidence(trace)
                        
                        # Merge success factors
                        if candidate.success_factors:
                            existing.success_factors.extend(candidate.success_factors)
                    else:
                        merged[signature] = candidate
                except Exception as e:
                    logger.warning("Error merging candidate: %s", e)
            
            return list(merged.values())
        except Exception as e:
            logger.error("Error in merge candidates: %s", e)
            return candidates
    
    def _extract_success_indicators(self, trace: ExecutionTrace) -> List[Metric]:
        """Extract success indicator metrics"""
        try:
            if not trace:
                return []
            
            indicators = []
            
            for metric in trace.metrics:
                if metric and metric.is_success and metric.meets_threshold():
                    indicators.append(metric)
            
            # Add derived indicators
            if trace.success:
                # Overall success indicator
                indicators.append(Metric(
                    name="overall_success",
                    metric_type=MetricType.RELIABILITY,
                    value=1.0,
                    is_success=True
                ))
            
            # Performance indicators
            duration = trace.get_duration()
            if duration > 0 and duration < 1.0:  # Fast execution
                indicators.append(Metric(
                    name="fast_execution",
                    metric_type=MetricType.PERFORMANCE,
                    value=duration,
                    unit="seconds",
                    threshold=1.0,
                    is_success=True
                ))
            
            return indicators
        except Exception as e:
            logger.error("Error extracting success indicators: %s", e)
            return []
    
    def _identify_required_metrics(self, trace: ExecutionTrace) -> List[str]:
        """Identify required metrics for principle"""
        try:
            if not trace:
                return []
            
            required = set()
            
            # Metrics that appeared in successful execution
            if trace.success:
                for metric in trace.metrics:
                    if metric and metric.meets_threshold():
                        required.add(metric.name)
            
            # Add standard requirements based on domain
            if trace.domain == "performance":
                required.update(['execution_time', 'throughput', 'latency'])
            elif trace.domain == "accuracy":
                required.update(['accuracy', 'precision', 'recall'])
            else:
                required.update(['success_rate', 'confidence'])
            
            return list(required)
        except Exception as e:
            logger.error("Error identifying required metrics: %s", e)
            return []
    
    def _crystallize_candidate(self, candidate: PrincipleCandidate) -> Optional[CrystallizedPrinciple]:
        """Convert candidate to crystallized principle"""
        try:
            if not candidate:
                return None
            
            # Check evidence threshold
            if len(candidate.evidence) < self.min_evidence_count:
                return None
            
            # Calculate generalizability
            candidate.calculate_generalizability()
            
            # Check if generalizable enough
            if self.strategy != ExtractionStrategy.EXPLORATORY and candidate.generalizability < 0.4:
                return None
            
            # Abstract to principle
            principle = self.abstract_to_principle(candidate.success_factors)
            
            if not principle:
                # Create from candidate directly
                principle = CrystallizedPrinciple(
                    id=f"principle_{candidate.pattern.signature()[:8]}",
                    name=f"Pattern-based principle from {candidate.origin_domain}",
                    description=self._generate_description(candidate),
                    core_pattern=candidate.pattern,
                    applicable_domains=[candidate.origin_domain],
                    confidence=candidate.confidence,
                    measurement_requirements=candidate.required_metrics,
                    domain=candidate.origin_domain,
                    metadata={
                        'stability': candidate.stability,
                        'generalizability': candidate.generalizability
                    }
                )
            
            # Set confidence
            principle.confidence = self.calculate_principle_confidence(candidate.evidence)
            
            # Set domain if not set
            if not hasattr(principle, 'domain') or not principle.domain:
                principle.domain = candidate.origin_domain
            
            # Add problem types
            principle.problem_types = self._infer_problem_types(candidate)
            
            return principle if principle.confidence >= self.min_confidence else None
        except Exception as e:
            logger.error("Error crystallizing candidate: %s", e)
            return None
    
    def _generate_description(self, candidate: PrincipleCandidate) -> str:
        """Generate description for principle"""
        try:
            components = []
            
            # Pattern type
            components.append(f"A {candidate.pattern.pattern_type.value} pattern")
            
            # Success factors
            if candidate.success_factors:
                top_factor = max(candidate.success_factors, key=lambda f: f.importance)
                components.append(f"driven by {top_factor.factor_type}")
            
            # Domain
            components.append(f"applicable to {candidate.origin_domain}")
            
            # Success rate
            if candidate.evidence:
                success_count = sum(1 for e in candidate.evidence if e and e.success)
                success_rate = success_count / len(candidate.evidence)
                components.append(f"with {success_rate:.0%} success rate")
            
            # Stability
            if candidate.stability > 0.7:
                components.append("showing high stability")
            
            return " ".join(components)
        except Exception as e:
            logger.error("Error generating description: %s", e)
            return "Pattern-based principle"
    
    def _infer_problem_types(self, candidate: PrincipleCandidate) -> List[str]:
        """Infer applicable problem types"""
        try:
            problem_types = set()
            
            # Based on pattern type
            if candidate.pattern.pattern_type == PatternType.ITERATIVE:
                problem_types.add("optimization")
                problem_types.add("refinement")
            elif candidate.pattern.pattern_type == PatternType.CONDITIONAL:
                problem_types.add("decision")
                problem_types.add("classification")
            elif candidate.pattern.pattern_type == PatternType.SEQUENTIAL:
                problem_types.add("workflow")
                problem_types.add("process")
            
            # Based on metrics
            for metric in candidate.success_indicators:
                if metric:
                    if metric.metric_type == MetricType.ACCURACY:
                        problem_types.add("prediction")
                    elif metric.metric_type == MetricType.PERFORMANCE:
                        problem_types.add("optimization")
                    elif metric.metric_type == MetricType.EFFICIENCY:
                        problem_types.add("resource_management")
            
            return list(problem_types) if problem_types else ["general"]
        except Exception as e:
            logger.error("Error inferring problem types: %s", e)
            return ["general"]
    
    def _calculate_metric_stability(self, evidence: List[ExecutionTrace]) -> float:
        """Calculate stability of metrics across evidence"""
        try:
            if not evidence:
                return 0.0
            
            # Collect all metric values by name
            metric_values = defaultdict(list)
            
            for trace in evidence:
                if trace and hasattr(trace, 'metrics'):
                    for metric in trace.metrics:
                        if metric:
                            metric_values[metric.name].append(metric.value)
            
            if not metric_values:
                return 0.5
            
            # Calculate coefficient of variation for each metric
            stabilities = []
            
            for name, values in metric_values.items():
                if len(values) > 1:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    
                    if mean_val != 0:
                        cv = std_val / abs(mean_val)
                        stability = 1.0 / (1.0 + cv)  # Convert CV to stability
                    else:
                        stability = 1.0 if std_val == 0 else 0.5
                    
                    stabilities.append(stability)
            
            return np.mean(stabilities) if stabilities else 0.5
        except Exception as e:
            logger.warning("Error calculating metric stability: %s", e)
            return 0.5
    
    def _calculate_temporal_consistency(self, evidence: List[ExecutionTrace]) -> float:
        """Calculate temporal consistency of evidence"""
        try:
            if len(evidence) < 2:
                return 1.0
            
            # Filter valid traces
            valid_evidence = [e for e in evidence if e]
            if len(valid_evidence) < 2:
                return 1.0
            
            # Sort by timestamp
            sorted_evidence = sorted(valid_evidence, key=lambda e: e.timestamp)
            
            # Check success consistency over time
            success_sequence = [e.success for e in sorted_evidence]
            
            if len(success_sequence) < 2:
                return 1.0
            
            # Count transitions
            transitions = sum(
                1 for i in range(1, len(success_sequence))
                if success_sequence[i] != success_sequence[i-1]
            )
            
            # Fewer transitions = more consistent
            consistency = 1.0 - (transitions / (len(success_sequence) - 1))
            
            return consistency
        except Exception as e:
            logger.warning("Error calculating temporal consistency: %s", e)
            return 0.5
    
    def _calculate_statistical_significance(self, evidence: List[ExecutionTrace]) -> float:
        """Calculate statistical significance using scipy if available"""
        try:
            # Use simpler heuristic for < 5 samples
            if len(evidence) < 5:
                if len(evidence) >= 3:
                    success_count = sum(1 for e in evidence if e and e.success)
                    success_rate = success_count / len(evidence)
                    return 0.3 + success_rate * 0.4  # 0.3-0.7 range
                elif len(evidence) >= 1:
                    success_count = sum(1 for e in evidence if e and e.success)
                    return 0.3 if success_count > 0 else 0.1
                return 0.3
            
            if not SCIPY_AVAILABLE:
                return 0.5
            
            # Compare success vs failure metrics
            success_metrics = []
            failure_metrics = []
            
            for trace in evidence:
                if not trace or not hasattr(trace, 'metrics'):
                    continue
                
                # Use a composite score
                score = sum(m.normalize_value() for m in trace.metrics if m and m.is_success)
                if trace.success:
                    success_metrics.append(score)
                else:
                    failure_metrics.append(score)
            
            if len(success_metrics) > 1 and len(failure_metrics) > 1:
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(success_metrics, failure_metrics)
                
                # Convert p-value to significance score
                significance = 1.0 - p_value if p_value < 0.5 else 0.5
                return max(0.0, min(1.0, significance))
            elif len(success_metrics) > 0:
                # All successes - high significance
                return 0.9
            else:
                # All failures - low significance  
                return 0.1
        except Exception as e:
            logger.warning("Error calculating statistical significance: %s", e)
            return 0.5


class PatternDetector:
    """Detects patterns in execution traces"""
    
    def __init__(self):
        """Initialize pattern detector"""
        self.pattern_cache = {}
        self.min_pattern_length = 2
        self.max_pattern_length = 10
        self.lock = threading.RLock()
        
    def detect_patterns(self, trace: ExecutionTrace) -> List[Pattern]:
        """
        Detect patterns in execution trace
        
        Args:
            trace: Execution trace
            
        Returns:
            List of detected patterns
        """
        try:
            if not trace or not isinstance(trace, ExecutionTrace):
                return []
            
            patterns = []
            
            # Sequential patterns
            seq_patterns = self._detect_sequential_patterns(trace)
            patterns.extend(seq_patterns)
            
            # Conditional patterns
            cond_patterns = self._detect_conditional_patterns(trace)
            patterns.extend(cond_patterns)
            
            # Iterative patterns
            iter_patterns = self._detect_iterative_patterns(trace)
            patterns.extend(iter_patterns)
            
            # Hierarchical patterns
            hier_patterns = self._detect_hierarchical_patterns(trace)
            patterns.extend(hier_patterns)
            
            # Composite patterns
            if len(patterns) > 1:
                composite = self._detect_composite_patterns(patterns)
                patterns.extend(composite)
            
            return patterns
        except Exception as e:
            logger.error("Error detecting patterns: %s", e)
            return []
    
    def _detect_sequential_patterns(self, trace: ExecutionTrace) -> List[Pattern]:
        """Detect sequential execution patterns"""
        try:
            patterns = []
            
            if not trace.actions or len(trace.actions) < self.min_pattern_length:
                return patterns
            
            # Look for action sequences
            for window_size in range(self.min_pattern_length, 
                                    min(self.max_pattern_length, len(trace.actions) + 1)):
                for i in range(len(trace.actions) - window_size + 1):
                    sequence = trace.actions[i:i+window_size]
                    
                    # Check if sequence has meaningful structure
                    action_types = [a.get('type', 'action') if isinstance(a, dict) else str(a) 
                                  for a in sequence]
                    
                    if len(set(action_types)) > 1:  # More than one action type
                        # Create pattern
                        pattern = Pattern(
                            pattern_type=PatternType.SEQUENTIAL,
                            components=action_types,
                            structure={
                                'sequence_length': window_size,
                                'position': i
                            },
                            frequency=1.0,
                            confidence=min(1.0, 0.5 + 0.1 * (window_size - 1)),
                            complexity=window_size
                        )
                        
                        patterns.append(pattern)
            
            return patterns
        except Exception as e:
            logger.error("Error detecting sequential patterns: %s", e)
            return []
    
    def _detect_conditional_patterns(self, trace: ExecutionTrace) -> List[Pattern]:
        """Detect conditional execution patterns"""
        try:
            patterns = []
            
            if not trace.actions:
                return patterns
            
            # Look for condition-action pairs
            for i, action in enumerate(trace.actions):
                if self._is_conditional_action(action):
                    # Found conditional
                    # Look for then/else branches
                    branches = self._extract_branches(trace.actions, i)
                    
                    action_type = action.get('type', 'condition') if isinstance(action, dict) else str(action)
                    
                    pattern = Pattern(
                        pattern_type=PatternType.CONDITIONAL,
                        components=[action_type] + branches,
                        structure={
                            'condition': action.get('condition', 'unknown') if isinstance(action, dict) else 'unknown',
                            'position': i,
                            'branches': len(branches)
                        },
                        frequency=1.0,
                        confidence=0.6,
                        complexity=len(branches) + 1
                    )
                    patterns.append(pattern)
            
            return patterns
        except Exception as e:
            logger.error("Error detecting conditional patterns: %s", e)
            return []
    
    def _detect_iterative_patterns(self, trace: ExecutionTrace) -> List[Pattern]:
        """Detect iterative/loop patterns"""
        try:
            patterns = []
            
            if not trace.actions:
                return patterns
            
            # Look for repeated action types
            action_types = [a.get('type', str(a)) if isinstance(a, dict) else str(a) 
                          for a in trace.actions]
            
            # Find repetitions
            for length in range(self.min_pattern_length, 
                              min(self.max_pattern_length, len(action_types) // 2 + 1)):
                for start in range(len(action_types) - length):
                    segment = action_types[start:start+length]
                    
                    # Count repetitions
                    count = 0
                    pos = start + length
                    while pos + length <= len(action_types):
                        if action_types[pos:pos+length] == segment:
                            count += 1
                            pos += length
                        else:
                            break
                    
                    if count > 0:
                        pattern = Pattern(
                            pattern_type=PatternType.ITERATIVE,
                            components=segment,
                            structure={
                                'iteration_count': count + 1,
                                'loop_length': length,
                                'start_position': start
                            },
                            frequency=count + 1,
                            confidence=min(1.0, 0.7 + 0.05 * count),
                            complexity=length * (count + 1)
                        )
                        patterns.append(pattern)
            
            return patterns
        except Exception as e:
            logger.error("Error detecting iterative patterns: %s", e)
            return []
    
    def _detect_hierarchical_patterns(self, trace: ExecutionTrace) -> List[Pattern]:
        """Detect hierarchical/nested patterns"""
        try:
            patterns = []
            
            if not trace.actions:
                return patterns
            
            # Look for nested structures in actions
            depth_stack = []
            
            for i, action in enumerate(trace.actions):
                if self._is_start_marker(action):
                    depth_stack.append((i, action))
                elif self._is_end_marker(action) and depth_stack:
                    start_idx, start_action = depth_stack.pop()
                    
                    # Extract nested actions
                    nested = trace.actions[start_idx+1:i]
                    if nested:
                        start_type = (start_action.get('type', 'start') if isinstance(start_action, dict) 
                                    else str(start_action))
                        nested_types = [a.get('type', 'action') if isinstance(a, dict) else str(a) 
                                      for a in nested]
                        
                        pattern = Pattern(
                            pattern_type=PatternType.HIERARCHICAL,
                            components=[start_type] + nested_types,
                            structure={
                                'depth': len(depth_stack) + 1,
                                'nested_count': len(nested),
                                'start_position': start_idx
                            },
                            frequency=1.0,
                            confidence=0.65,
                            complexity=len(nested) + 2
                        )
                        patterns.append(pattern)
            
            return patterns
        except Exception as e:
            logger.error("Error detecting hierarchical patterns: %s", e)
            return []
    
    def _detect_composite_patterns(self, patterns: List[Pattern]) -> List[Pattern]:
        """Detect composite patterns from simpler patterns"""
        try:
            composite = []
            
            if not patterns or len(patterns) < 2:
                return composite
            
            # Look for patterns that occur together
            for i, p1 in enumerate(patterns):
                if not p1:
                    continue
                
                for p2 in patterns[i+1:]:
                    if not p2:
                        continue
                    
                    if self._patterns_compatible(p1, p2):
                        combined = Pattern(
                            pattern_type=PatternType.COMPOSITE,
                            components=[p1.signature()[:8], p2.signature()[:8]],
                            structure={
                                'sub_patterns': [p1.pattern_type.value, p2.pattern_type.value],
                                'combined_complexity': p1.complexity + p2.complexity
                            },
                            frequency=min(p1.frequency, p2.frequency),
                            confidence=np.mean([p1.confidence, p2.confidence]) * 0.9,
                            complexity=p1.complexity + p2.complexity
                        )
                        composite.append(combined)
            
            return composite
        except Exception as e:
            logger.error("Error detecting composite patterns: %s", e)
            return []
    
    def _is_conditional_action(self, action: Any) -> bool:
        """Check if action is conditional"""
        try:
            indicators = ['condition', 'if', 'when', 'check', 'test', 'evaluate']
            action_str = str(action).lower()
            return any(ind in action_str for ind in indicators)
        except Exception as e:            return False
    
    def _is_start_marker(self, action: Any) -> bool:
        """Check if action marks start of a block"""
        try:
            indicators = ['start', 'begin', 'open', 'enter', '{']
            action_str = str(action).lower()
            return any(ind in action_str for ind in indicators)
        except Exception as e:            return False
    
    def _is_end_marker(self, action: Any) -> bool:
        """Check if action marks end of a block"""
        try:
            indicators = ['end', 'finish', 'close', 'exit', '}']
            action_str = str(action).lower()
            return any(ind in action_str for ind in indicators)
        except Exception as e:            return False
    
    def _extract_branches(self, actions: List[Any], condition_idx: int) -> List[str]:
        """Extract branches from conditional"""
        try:
            branches = []
            
            # Simple heuristic: next few actions after condition
            for i in range(condition_idx + 1, min(condition_idx + 4, len(actions))):
                if i < len(actions):
                    action = actions[i]
                    branch = action.get('type', 'action') if isinstance(action, dict) else str(action)
                    branches.append(branch)
            
            return branches
        except Exception as e:
            logger.warning("Error extracting branches: %s", e)
            return []
    
    def _patterns_compatible(self, p1: Pattern, p2: Pattern) -> bool:
        """Check if patterns can be combined"""
        try:
            if not p1 or not p2:
                return False
            
            # Don't combine same types
            if p1.pattern_type == p2.pattern_type:
                return False
            
            # Don't combine if too complex
            if p1.complexity + p2.complexity > 15:
                return False
            
            return True
        except Exception as e:            return False


class SuccessAnalyzer:
    """Analyzes success factors in traces"""
    
    def __init__(self):
        """Initialize success analyzer"""
        self.factor_weights = {
            'metric': 0.8,
            'action': 0.7,
            'timing': 0.6,
            'context': 0.5,
            'resource': 0.4
        }
    
    def analyze(self, trace: ExecutionTrace) -> List[SuccessFactor]:
        """
        Analyze success factors in trace
        
        Args:
            trace: Execution trace
            
        Returns:
            List of success factors
        """
        try:
            if not trace:
                return []
            
            factors = []
            
            # Analyze different factor types
            
            # 1. Action-based factors
            action_factors = self._analyze_action_factors(trace)
            factors.extend(action_factors)
            
            # 2. Metric-based factors
            metric_factors = self._analyze_metric_factors(trace)
            factors.extend(metric_factors)
            
            # 3. Context-based factors
            context_factors = self._analyze_context_factors(trace)
            factors.extend(context_factors)
            
            # 4. Timing factors
            timing_factors = self._analyze_timing_factors(trace)
            factors.extend(timing_factors)
            
            # 5. Pattern-based factors
            pattern_factors = self._analyze_pattern_factors(trace)
            factors.extend(pattern_factors)
            
            # Calculate correlations if possible
            if SCIPY_AVAILABLE:
                self._calculate_correlations(factors, trace)
            
            # Sort by importance
            factors.sort(key=lambda f: f.importance if f else 0, reverse=True)
            
            return factors
        except Exception as e:
            logger.error("Error analyzing trace: %s", e)
            return []
    
    def _analyze_action_factors(self, trace: ExecutionTrace) -> List[SuccessFactor]:
        """Analyze action-related success factors"""
        try:
            factors = []
            
            if not trace.actions:
                return factors
            
            # Critical actions (first and last)
            first_action = trace.actions[0]
            first_type = first_action.get('type', 'unknown') if isinstance(first_action, dict) else str(first_action)
            
            factors.append(SuccessFactor(
                factor_type='initial_action',
                importance=self.factor_weights['action'] * 0.9,
                evidence_count=1,
                conditions=[f"starts_with_{first_type}"]
            ))
            
            if len(trace.actions) > 1:
                last_action = trace.actions[-1]
                last_type = last_action.get('type', 'unknown') if isinstance(last_action, dict) else str(last_action)
                
                factors.append(SuccessFactor(
                    factor_type='final_action',
                    importance=self.factor_weights['action'] * 0.8,
                    evidence_count=1,
                    conditions=[f"ends_with_{last_type}"]
                ))
            
            # Action diversity
            action_types = set()
            for a in trace.actions:
                atype = a.get('type', 'unknown') if isinstance(a, dict) else str(a)
                action_types.add(atype)
            
            if len(action_types) > 3:
                factors.append(SuccessFactor(
                    factor_type='action_diversity',
                    importance=self.factor_weights['action'] * 0.6,
                    evidence_count=1,
                    conditions=[f"diverse_actions_{len(action_types)}"]
                ))
            
            # Action sequence length
            if trace.success and len(trace.actions) < 10:
                factors.append(SuccessFactor(
                    factor_type='efficient_sequence',
                    importance=self.factor_weights['action'] * 0.7,
                    evidence_count=1,
                    conditions=[f"action_count_{len(trace.actions)}"]
                ))
            
            return factors
        except Exception as e:
            logger.error("Error analyzing action factors: %s", e)
            return []
    
    def _analyze_metric_factors(self, trace: ExecutionTrace) -> List[SuccessFactor]:
        """Analyze metric-related success factors"""
        try:
            factors = []
            
            for metric in trace.metrics:
                if not metric:
                    continue
                
                if metric.meets_threshold() and metric.is_success:
                    # Calculate importance based on metric type and value
                    base_importance = self.factor_weights['metric']
                    
                    if metric.metric_type == MetricType.ACCURACY:
                        importance = base_importance * 1.0
                    elif metric.metric_type == MetricType.RELIABILITY:
                        importance = base_importance * 0.9
                    else:
                        importance = base_importance * 0.7
                    
                    # Adjust by how much threshold was exceeded
                    if metric.threshold:
                        try:
                            excess = (metric.value - metric.threshold) / metric.threshold
                            importance *= (1 + min(0.2, excess))
                        except (ZeroDivisionError, TypeError):
                            pass
                    
                    factor = SuccessFactor(
                        factor_type=f'metric_{metric.name}',
                        importance=importance,
                        evidence_count=1,
                        conditions=[f"{metric.name}>={metric.threshold}" if metric.threshold 
                                  else f"{metric.name}_success"],
                        metrics=[metric]
                    )
                    factors.append(factor)
            
            return factors
        except Exception as e:
            logger.error("Error analyzing metric factors: %s", e)
            return []
    
    def _analyze_context_factors(self, trace: ExecutionTrace) -> List[SuccessFactor]:
        """Analyze context-related success factors"""
        try:
            factors = []
            
            if not trace.context:
                return factors
            
            # Environment factor
            if 'environment' in trace.context:
                factors.append(SuccessFactor(
                    factor_type='environment',
                    importance=self.factor_weights['context'],
                    evidence_count=1,
                    conditions=[f"env_{trace.context['environment']}"]
                ))
            
            # Resource factors
            if 'resources' in trace.context:
                resources = trace.context['resources']
                if isinstance(resources, dict):
                    for resource, value in resources.items():
                        if value and value > 0:
                            importance = self.factor_weights['resource']
                            # Critical resources get higher importance
                            if resource in ['memory', 'cpu', 'gpu']:
                                importance *= 1.2
                            
                            factors.append(SuccessFactor(
                                factor_type=f'resource_{resource}',
                                importance=importance,
                                evidence_count=1,
                                conditions=[f"{resource}_available_{value}"]
                            ))
            
            # Configuration factors
            if 'config' in trace.context:
                config = trace.context['config']
                if isinstance(config, dict):
                    for key, value in config.items():
                        if value:  # Non-empty config
                            factors.append(SuccessFactor(
                                factor_type=f'config_{key}',
                                importance=self.factor_weights['context'] * 0.8,
                                evidence_count=1,
                                conditions=[f"{key}={value}"]
                            ))
            
            return factors
        except Exception as e:
            logger.error("Error analyzing context factors: %s", e)
            return []
    
    def _analyze_timing_factors(self, trace: ExecutionTrace) -> List[SuccessFactor]:
        """Analyze timing-related success factors"""
        try:
            factors = []
            
            # Execution speed
            duration = trace.get_duration()
            if duration > 0:
                if duration < 0.1:  # Very fast
                    importance = self.factor_weights['timing'] * 1.2
                    condition = 'very_fast_execution'
                elif duration < 1.0:  # Fast
                    importance = self.factor_weights['timing']
                    condition = 'fast_execution'
                elif duration < 10.0:  # Normal
                    importance = self.factor_weights['timing'] * 0.8
                    condition = 'normal_execution'
                else:  # Slow but successful
                    importance = self.factor_weights['timing'] * 0.5
                    condition = 'slow_execution'
                
                factors.append(SuccessFactor(
                    factor_type='execution_speed',
                    importance=importance,
                    evidence_count=1,
                    conditions=[condition, f"duration_{duration:.2f}s"]
                ))
            
            # Time of day factor (if available)
            try:
                hour = datetime.fromtimestamp(trace.timestamp).hour
                if 9 <= hour <= 17:  # Business hours
                    factors.append(SuccessFactor(
                        factor_type='business_hours',
                        importance=self.factor_weights['timing'] * 0.3,
                        evidence_count=1,
                        conditions=['business_hours_execution']
                    ))
            except Exception as e:                logger.debug(f"{self.__class__.__name__ if hasattr(self, '__class__') else 'Operation'} error: {e}")
            
            return factors
        except Exception as e:
            logger.error("Error analyzing timing factors: %s", e)
            return []
    
    def _analyze_pattern_factors(self, trace: ExecutionTrace) -> List[SuccessFactor]:
        """Analyze pattern-related success factors"""
        try:
            factors = []
            
            if not hasattr(trace, 'patterns') or not trace.patterns:
                return factors
            
            for pattern in trace.patterns:
                if not pattern:
                    continue
                
                importance = self.factor_weights['action'] * pattern.confidence
                
                factor = SuccessFactor(
                    factor_type=f'pattern_{pattern.pattern_type.value}',
                    importance=importance,
                    evidence_count=1,
                    conditions=[f"has_{pattern.pattern_type.value}_pattern"],
                    metadata={'pattern': pattern.to_dict()}
                )
                factors.append(factor)
            
            return factors
        except Exception as e:
            logger.error("Error analyzing pattern factors: %s", e)
            return []
    
    def _calculate_correlations(self, factors: List[SuccessFactor], trace: ExecutionTrace):
        """Calculate correlation between factors and success"""
        try:
            if not SCIPY_AVAILABLE or not factors:
                return
            
            # Simple correlation based on presence in successful trace
            base_correlation = 0.7 if trace.success else -0.3
            
            for factor in factors:
                if not factor:
                    continue
                
                # Adjust correlation based on factor type and success
                if trace.success:
                    if 'metric' in factor.factor_type:
                        factor.correlation = base_correlation * 1.2
                    else:
                        factor.correlation = base_correlation
                else:
                    factor.correlation = base_correlation
                
                # Estimate causality (simplified)
                factor.causality_score = abs(factor.correlation) * factor.importance
        except Exception as e:
            logger.warning("Error calculating correlations: %s", e)


class AbstractionEngine:
    """Engine for abstracting patterns to principles"""
    
    def __init__(self):
        """Initialize abstraction engine"""
        self.abstraction_rules = self._load_abstraction_rules()
        self.naming_templates = self._load_naming_templates()
        
    def abstract(self, success_factors: List[SuccessFactor]) -> Optional[Dict[str, Any]]:
        """
        Abstract success factors to principle
        
        Args:
            success_factors: List of success factors
            
        Returns:
            Abstracted principle specification
        """
        try:
            if not success_factors:
                return None
            
            # Group factors by type
            grouped = self._group_factors(success_factors)
            
            # Identify dominant patterns
            dominant_type, dominant_factors = self._identify_dominant(grouped)
            
            # Generate abstraction
            abstracted = {
                'name': self._generate_name(dominant_type, success_factors),
                'description': self._generate_description(success_factors, grouped),
                'pattern': self._abstract_pattern(success_factors),
                'domains': self._identify_domains(success_factors),
                'requirements': self._extract_requirements(success_factors),
                'confidence': self._calculate_abstraction_confidence(success_factors),
                'tags': self._generate_tags(success_factors)
            }
            
            return abstracted
        except Exception as e:
            logger.error("Error abstracting factors: %s", e)
            return None
    
    def _load_abstraction_rules(self) -> Dict[str, Any]:
        """Load abstraction rules"""
        return {
            'metric_patterns': {
                'high_accuracy': 'Optimization for accuracy',
                'fast_execution': 'Performance optimization',
                'resource_efficient': 'Resource optimization',
                'reliable': 'Reliability enhancement',
                'scalable': 'Scalability improvement'
            },
            'action_patterns': {
                'sequential': 'Step-by-step execution',
                'parallel': 'Concurrent processing',
                'iterative': 'Iterative refinement',
                'conditional': 'Conditional logic',
                'hierarchical': 'Nested structure'
            },
            'context_patterns': {
                'environment': 'Environment-specific',
                'resource': 'Resource-dependent',
                'config': 'Configuration-based'
            }
        }
    
    def _load_naming_templates(self) -> Dict[str, List[str]]:
        """Load naming templates"""
        return {
            'high_importance': [
                "Critical {type} Principle",
                "Essential {type} Pattern",
                "Core {type} Strategy"
            ],
            'medium_importance': [
                "Important {type} Principle",
                "{type} Optimization Pattern",
                "Standard {type} Approach"
            ],
            'low_importance': [
                "Supporting {type} Principle",
                "Auxiliary {type} Pattern",
                "Optional {type} Enhancement"
            ]
        }
    
    def _group_factors(self, factors: List[SuccessFactor]) -> Dict[str, List[SuccessFactor]]:
        """Group factors by base type"""
        try:
            grouped = defaultdict(list)
            
            for factor in factors:
                if not factor:
                    continue
                
                base_type = factor.factor_type.split('_')[0]
                grouped[base_type].append(factor)
            
            return dict(grouped)
        except Exception as e:
            logger.error("Error grouping factors: %s", e)
            return {}
    
    def _identify_dominant(self, grouped: Dict[str, List[SuccessFactor]]) -> Tuple[str, List[SuccessFactor]]:
        """Identify dominant factor type"""
        try:
            if not grouped:
                return 'unknown', []
            
            # Score each group by total importance
            scores = {}
            for group_type, factors in grouped.items():
                scores[group_type] = sum(f.importance for f in factors if f)
            
            if not scores:
                return 'unknown', []
            
            # Get dominant type
            dominant_type = max(scores, key=scores.get)
            
            return dominant_type, grouped[dominant_type]
        except Exception as e:
            logger.error("Error identifying dominant: %s", e)
            return 'unknown', []
    
    def _generate_name(self, dominant_type: str, factors: List[SuccessFactor]) -> str:
        """Generate principle name"""
        try:
            # Calculate average importance
            importances = [f.importance for f in factors if f]
            avg_importance = np.mean(importances) if importances else 0.5
            
            # Select template category
            if avg_importance > 0.8:
                templates = self.naming_templates['high_importance']
            elif avg_importance > 0.6:
                templates = self.naming_templates['medium_importance']
            else:
                templates = self.naming_templates['low_importance']
            
            # Select random template
            template = np.random.choice(templates)
            
            # Clean up type name
            clean_type = dominant_type.replace('_', ' ').title()
            
            return template.format(type=clean_type)
        except Exception as e:
            logger.error("Error generating name: %s", e)
            return "General Principle"
    
    def _generate_description(self, factors: List[SuccessFactor], 
                            grouped: Dict[str, List[SuccessFactor]]) -> str:
        """Generate principle description"""
        try:
            descriptions = []
            
            # Describe each group
            for base_type, group_factors in grouped.items():
                if not group_factors:
                    continue
                
                if base_type == 'metric':
                    metric_names = [f.factor_type.replace('metric_', '') 
                                   for f in group_factors[:3] if f]
                    if metric_names:
                        descriptions.append(f"Optimizes {', '.join(metric_names)} metrics")
                elif base_type == 'action':
                    descriptions.append(f"Employs {len(group_factors)} action patterns")
                elif base_type == 'resource':
                    resources = [f.factor_type.replace('resource_', '') 
                               for f in group_factors if f]
                    if resources:
                        descriptions.append(f"Requires {', '.join(resources)} resources")
                elif base_type == 'timing':
                    descriptions.append("Time-sensitive execution")
                elif base_type == 'pattern':
                    pattern_types = set(f.factor_type.replace('pattern_', '') 
                                      for f in group_factors if f)
                    if pattern_types:
                        descriptions.append(f"Uses {', '.join(pattern_types)} patterns")
            
            # Add success correlation if significant
            high_correlation = [f for f in factors if f and f.correlation > 0.7]
            if high_correlation:
                descriptions.append(f"Strongly correlated with success ({len(high_correlation)} factors)")
            
            return ". ".join(descriptions) if descriptions else "General optimization principle"
        except Exception as e:
            logger.error("Error generating description: %s", e)
            return "General principle"
    
    def _abstract_pattern(self, factors: List[SuccessFactor]) -> Pattern:
        """Create abstracted pattern from factors"""
        try:
            # Determine pattern type based on factors
            pattern_factors = [f for f in factors if f and 'pattern' in f.factor_type]
            
            if pattern_factors:
                # Use most important pattern
                main_pattern = max(pattern_factors, key=lambda f: f.importance)
                pattern_type_str = main_pattern.factor_type.replace('pattern_', '').upper()
                try:
                    pattern_type = PatternType[pattern_type_str]
                except KeyError:
                    pattern_type = PatternType.SEQUENTIAL
            else:
                # Infer from other factors
                if any(f and 'iterative' in f.factor_type for f in factors):
                    pattern_type = PatternType.ITERATIVE
                elif any(f and 'parallel' in f.factor_type for f in factors):
                    pattern_type = PatternType.PARALLEL
                elif any(f and 'condition' in str(f.conditions) for f in factors if f):
                    pattern_type = PatternType.CONDITIONAL
                else:
                    pattern_type = PatternType.SEQUENTIAL
            
            # Extract components from conditions
            components = []
            for factor in factors[:10]:  # Limit components
                if factor:
                    components.extend(factor.conditions[:2])  # Limit per factor
            
            # Calculate pattern confidence
            importances = [f.importance for f in factors if f]
            confidence = np.mean(importances) if importances else 0.5
            
            unique_components = list(set(components))
            
            return Pattern(
                pattern_type=pattern_type,
                components=unique_components[:15],  # Final limit
                structure={
                    'abstracted': True,
                    'factor_count': len(factors)
                },
                confidence=confidence,
                complexity=len(unique_components)
            )
        except Exception as e:
            logger.error("Error abstracting pattern: %s", e)
            return Pattern(
                pattern_type=PatternType.SEQUENTIAL,
                components=['default'],
                confidence=0.5,
                complexity=1
            )
    
    def _identify_domains(self, factors: List[SuccessFactor]) -> List[str]:
        """Identify applicable domains from factors"""
        try:
            domains = set()
            
            # Check factor conditions for domain hints
            for factor in factors:
                if not factor:
                    continue
                
                for condition in factor.conditions:
                    # Domain patterns
                    if 'domain_' in condition:
                        parts = condition.split('domain_')
                        if len(parts) > 1:
                            domain = parts[1].split('_')[0]
                            domains.add(domain)
                    elif 'env_' in condition:
                        parts = condition.split('env_')
                        if len(parts) > 1:
                            env = parts[1].split('_')[0]
                            domains.add(env)
                    
                    # Infer from metric types
                    for metric in factor.metrics:
                        if not metric:
                            continue
                        
                        if metric.metric_type == MetricType.PERFORMANCE:
                            domains.add('performance')
                        elif metric.metric_type == MetricType.ACCURACY:
                            domains.add('accuracy')
                        elif metric.metric_type == MetricType.SCALABILITY:
                            domains.add('scalability')
            
            return list(domains) if domains else ['general']
        except Exception as e:
            logger.error("Error identifying domains: %s", e)
            return ['general']
    
    def _extract_requirements(self, factors: List[SuccessFactor]) -> List[str]:
        """Extract measurement requirements from factors"""
        try:
            requirements = set()
            
            for factor in factors:
                if not factor:
                    continue
                
                # Add metric requirements
                for metric in factor.metrics:
                    if metric:
                        requirements.add(metric.name)
                        if metric.unit:
                            requirements.add(f"{metric.name}_{metric.unit}")
                
                # Add condition requirements
                for condition in factor.conditions:
                    # Extract measurable conditions
                    if '>=' in condition or '<=' in condition or '=' in condition:
                        # Extract the metric name
                        parts = re.split(r'[><=]', condition)
                        if parts:
                            requirements.add(parts[0].strip())
            
            return list(requirements)
        except Exception as e:
            logger.error("Error extracting requirements: %s", e)
            return []
    
    def _calculate_abstraction_confidence(self, factors: List[SuccessFactor]) -> float:
        """Calculate confidence in abstraction"""
        try:
            if not factors:
                return 0.0
            
            valid_factors = [f for f in factors if f]
            if not valid_factors:
                return 0.0
            
            # Base confidence on factor importance
            importance_score = np.mean([f.importance for f in valid_factors])
            
            # Evidence strength
            evidence_score = min(1.0, sum(f.evidence_count for f in valid_factors) / 20)
            
            # Correlation strength
            correlation_scores = [abs(f.correlation) for f in valid_factors if f.correlation != 0]
            correlation_score = np.mean(correlation_scores) if correlation_scores else 0.5
            
            # Combine scores
            confidence = (
                importance_score * 0.4 +
                evidence_score * 0.3 +
                correlation_score * 0.3
            )
            
            return min(1.0, confidence)
        except Exception as e:
            logger.error("Error calculating abstraction confidence: %s", e)
            return 0.5
    
    def _generate_tags(self, factors: List[SuccessFactor]) -> List[str]:
        """Generate tags for the principle"""
        try:
            tags = set()
            
            # Add factor type tags
            for factor in factors:
                if not factor:
                    continue
                
                base_type = factor.factor_type.split('_')[0]
                tags.add(base_type)
                
                # Add metric type tags
                for metric in factor.metrics:
                    if metric:
                        tags.add(metric.metric_type.value)
            
            # Add importance level
            valid_factors = [f for f in factors if f]
            if valid_factors:
                avg_importance = np.mean([f.importance for f in valid_factors])
                if avg_importance > 0.8:
                    tags.add('critical')
                elif avg_importance > 0.6:
                    tags.add('important')
                else:
                    tags.add('supporting')
            
            # Add pattern tags
            pattern_factors = [f for f in factors if f and 'pattern' in f.factor_type]
            for pf in pattern_factors:
                pattern_type = pf.factor_type.replace('pattern_', '')
                tags.add(f"{pattern_type}_pattern")
            
            return list(tags)
        except Exception as e:
            logger.error("Error generating tags: %s", e)
            return []
