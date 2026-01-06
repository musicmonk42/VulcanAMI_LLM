"""
System Observer - Converts system events into world model observations.

This is the "nervous system" that connects the reasoning engines to the
world model's "brain". Without this, the world model operates blind.

The SystemObserver:
1. Receives events from query processing pipeline
2. Converts them to Observation objects
3. Feeds them to WorldModel for causal learning
4. Tracks patterns for recommendation engine

Usage:
    from vulcan.world_model.system_observer import SystemObserver
    
    observer = SystemObserver(world_model)
    
    # At query start
    observer.observe_query_start(query_id, query, classification)
    
    # After engine execution
    observer.observe_engine_result(query_id, engine_name, result)
    
    # At query completion
    observer.observe_outcome(query_id, response, user_feedback)
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .world_model_core import WorldModel

logger = logging.getLogger(__name__)


# =============================================================================
# Shared Constants for Query Pattern Detection
# =============================================================================
# These constants are used by both SystemObserver and WorldModel for consistent
# query classification. They detect formal logic, probability, and self-referential
# queries to support proper routing and observation categorization.

# Formal logic symbols and keywords
FORMAL_LOGIC_SYMBOLS = frozenset([
    '→', '∧', '∨', '¬', '∀', '∃', '⊢', '⊨', '↔', '⇒', '⇔'
])
FORMAL_LOGIC_KEYWORDS = frozenset([
    'forall', 'exists', 'implies', 'entails', 'satisfiable', 'valid'
])

# Probability-related keywords
PROBABILITY_KEYWORDS = frozenset([
    'probability', 'likelihood', 'bayes', 'bayesian', 'posterior',
    'prior', 'p(', 'conditional', 'expectation', 'marginal'
])

# Self-referential query keywords
SELF_REFERENTIAL_KEYWORDS = frozenset([
    'you want', 'your goal', 'self-aware', 'you have',
    'do you', 'are you', 'your capabilities', 'yourself',
    'your purpose', 'your objectives'
])

# =============================================================================
# Configuration Constants
# =============================================================================
# Limits for history storage and truncation

QUERY_TRUNCATION_LIMIT = 200  # Max chars to store from query text
ERROR_MESSAGE_TRUNCATION_LIMIT = 500  # Max chars for error messages
QUERY_HISTORY_MAX_SIZE = 1000  # Max query events to retain
ENGINE_HISTORY_MAX_SIZE = 5000  # Max engine events to retain  
OUTCOME_HISTORY_MAX_SIZE = 1000  # Max outcome events to retain


class EventType(Enum):
    """Types of system events to observe"""
    QUERY_START = "query_start"
    ENGINE_EXECUTION = "engine_execution"
    ENGINE_RESULT = "engine_result"
    OUTCOME = "outcome"
    ERROR = "error"
    VALIDATION_FAILURE = "validation_failure"


@dataclass
class SystemEvent:
    """Represents a system event to be observed"""
    event_type: EventType
    timestamp: float
    query_id: str
    data: Dict[str, Any]


class SystemObserver:
    """
    Observes system events and feeds them to the world model.
    
    This creates the observation pipeline that allows the world model
    to learn from system behavior and provide recommendations.
    
    The observer converts raw system events into structured observations
    that the WorldModel can use to:
    - Build causal understanding of system behavior
    - Track engine performance patterns
    - Learn which engines work best for which query types
    - Identify validation failure patterns
    
    Attributes:
        world_model: Reference to the WorldModel instance
        enabled: Whether observation is active
        query_history: Recent query events
        engine_history: Recent engine events
        outcome_history: Recent outcome events
        stats: Observation statistics
    """
    
    def __init__(self, world_model: "WorldModel"):
        """
        Initialize SystemObserver.
        
        Args:
            world_model: WorldModel instance to feed observations to
        """
        self.world_model = world_model
        self.enabled = True
        
        # History tracking with bounded queues using configured limits
        self.query_history: deque = deque(maxlen=QUERY_HISTORY_MAX_SIZE)
        self.engine_history: deque = deque(maxlen=ENGINE_HISTORY_MAX_SIZE)
        self.outcome_history: deque = deque(maxlen=OUTCOME_HISTORY_MAX_SIZE)
        
        # Statistics
        self.stats = {
            'total_queries': 0,
            'total_engine_executions': 0,
            'total_outcomes': 0,
            'errors_observed': 0,
            'validation_failures': 0
        }
        
        logger.info("SystemObserver initialized - World model observation pipeline active")
    
    def observe_query_start(
        self,
        query_id: str,
        query: str,
        classification: Dict[str, Any]
    ) -> None:
        """
        Observe query start event.
        
        Called when a new query enters the processing pipeline.
        Captures query characteristics for pattern learning.
        
        Args:
            query_id: Unique query identifier
            query: Query text
            classification: Query classification (category, complexity, tools)
        """
        if not self.enabled:
            return
        
        event = SystemEvent(
            event_type=EventType.QUERY_START,
            timestamp=time.time(),
            query_id=query_id,
            data={
                'query': query[:QUERY_TRUNCATION_LIMIT] if query else "",
                'query_type': classification.get('category'),
                'complexity': classification.get('complexity'),
                'tools_selected': classification.get('tools', []),
                'skip_reasoning': classification.get('skip_reasoning', False)
            }
        )
        
        self.query_history.append(event)
        self.stats['total_queries'] += 1
        
        # Convert to world model observation
        try:
            from .world_model_core import Observation
            
            obs = Observation(
                timestamp=event.timestamp,
                variables={
                    'query_type': event.data['query_type'],
                    'complexity': event.data['complexity'],
                    'tools_count': len(event.data['tools_selected']),
                    'has_formal_logic': self._check_formal_logic(query),
                    'has_probability': self._check_probability(query),
                    'is_self_referential': self._check_self_referential(query)
                },
                domain='query_routing',
                metadata={'query_id': query_id}
            )
            
            self.world_model.update_from_observation(obs)
            logger.debug(f"Query start observed: {query_id}")
            
        except Exception as e:
            logger.error(f"Failed to update world model on query start: {e}")
    
    def observe_engine_execution(
        self,
        query_id: str,
        engine_name: str,
        execution_time_ms: float,
        started: bool = True
    ) -> None:
        """
        Observe engine execution start/end.
        
        Args:
            query_id: Query identifier
            engine_name: Name of reasoning engine
            execution_time_ms: Execution time in milliseconds
            started: True if starting, False if completed
        """
        if not self.enabled:
            return
        
        event = SystemEvent(
            event_type=EventType.ENGINE_EXECUTION,
            timestamp=time.time(),
            query_id=query_id,
            data={
                'engine': engine_name,
                'execution_time_ms': execution_time_ms,
                'started': started
            }
        )
        
        self.engine_history.append(event)
        if not started:
            self.stats['total_engine_executions'] += 1
    
    def observe_engine_result(
        self,
        query_id: str,
        engine_name: str,
        result: Dict[str, Any],
        success: bool,
        execution_time_ms: float
    ) -> None:
        """
        Observe engine result.
        
        Called after a reasoning engine produces a result.
        This is critical for learning which engines succeed on which queries.
        
        Args:
            query_id: Query identifier
            engine_name: Name of reasoning engine
            result: Reasoning result dictionary
            success: Whether execution succeeded
            execution_time_ms: Execution time in milliseconds
        """
        if not self.enabled:
            return
        
        confidence = result.get('confidence', 0.0) if isinstance(result, dict) else 0.0
        
        event = SystemEvent(
            event_type=EventType.ENGINE_RESULT,
            timestamp=time.time(),
            query_id=query_id,
            data={
                'engine': engine_name,
                'confidence': confidence,
                'success': success,
                'execution_time_ms': execution_time_ms,
                'reasoning_type': result.get('reasoning_type') if isinstance(result, dict) else None,
                'tokens_used': result.get('tokens_used', 0) if isinstance(result, dict) else 0
            }
        )
        
        self.engine_history.append(event)
        
        # Convert to world model observation
        try:
            from .world_model_core import Observation
            
            obs = Observation(
                timestamp=event.timestamp,
                variables={
                    'engine_name': engine_name,
                    'confidence': confidence,
                    'success': success,
                    'execution_time_ms': execution_time_ms,
                    'high_confidence': confidence >= 0.7,
                    'low_confidence': confidence < 0.3
                },
                domain='reasoning_execution',
                metadata={'query_id': query_id}
            )
            
            self.world_model.update_from_observation(obs)
            
            # Update causal graph if available: engine → outcome
            if hasattr(self.world_model, 'causal_graph') and self.world_model.causal_graph:
                try:
                    # Record that this engine produces success/failure outcomes
                    outcome_node = 'success' if success else 'failure'
                    self.world_model.causal_graph.add_edge(
                        f'engine_{engine_name}',
                        outcome_node,
                        strength=0.7 if success else 0.8,
                        evidence_type='observation'
                    )
                except Exception as e:
                    logger.debug(f"Could not update causal graph: {e}")
            
            logger.debug(
                f"Engine result observed: {engine_name} "
                f"(confidence={confidence:.2f}, success={success})"
            )
            
        except Exception as e:
            logger.error(f"Failed to update world model on engine result: {e}")
    
    def observe_validation_failure(
        self,
        query_id: str,
        engine_name: str,
        reason: str,
        query: str,
        result: Dict[str, Any]
    ) -> None:
        """
        Observe answer validation failure.
        
        This helps world model learn which engines produce nonsensical
        outputs for which query types - critical for routing improvements.
        
        Args:
            query_id: Query identifier
            engine_name: Engine that produced invalid result
            reason: Why validation failed
            query: Original query
            result: Invalid result
        """
        if not self.enabled:
            return
        
        query_type = self._infer_query_type(query)
        
        event = SystemEvent(
            event_type=EventType.VALIDATION_FAILURE,
            timestamp=time.time(),
            query_id=query_id,
            data={
                'engine': engine_name,
                'reason': reason,
                'query_type': query_type,
                'result_type': self._infer_result_type(result)
            }
        )
        
        self.engine_history.append(event)
        self.stats['validation_failures'] += 1
        
        # Feed to world model - this is important learning signal
        try:
            from .world_model_core import Observation
            
            obs = Observation(
                timestamp=event.timestamp,
                variables={
                    'engine': engine_name,
                    'validation_failed': True,
                    'reason': reason,
                    'query_type': query_type,
                    'result_mismatch': True
                },
                domain='quality_control',
                metadata={'query_id': query_id}
            )
            
            self.world_model.update_from_observation(obs)
            
            # Strong causal link: this engine + this query type → failure
            if hasattr(self.world_model, 'causal_graph') and self.world_model.causal_graph:
                try:
                    self.world_model.causal_graph.add_edge(
                        f'{engine_name}_on_{query_type}',
                        'validation_failure',
                        strength=0.9,
                        evidence_type='observation'
                    )
                except Exception as e:
                    logger.debug(f"Could not update causal graph for validation failure: {e}")
            
            logger.warning(
                f"Validation failure observed: {engine_name} on {query_type} - {reason}"
            )
            
        except Exception as e:
            logger.error(f"Failed to update world model on validation failure: {e}")
    
    def observe_outcome(
        self,
        query_id: str,
        response: Dict[str, Any],
        user_feedback: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Observe final query outcome.
        
        Called when query processing completes. Captures whether
        reasoning was used vs fallback, confidence, and user feedback.
        
        Args:
            query_id: Query identifier
            response: Final response dictionary
            user_feedback: Optional user feedback (rating, etc)
        """
        if not self.enabled:
            return
        
        source = response.get('source', 'unknown') if isinstance(response, dict) else 'unknown'
        confidence = response.get('confidence', 0.0) if isinstance(response, dict) else 0.0
        response_time = response.get('response_time_ms', 0) if isinstance(response, dict) else 0
        
        event = SystemEvent(
            event_type=EventType.OUTCOME,
            timestamp=time.time(),
            query_id=query_id,
            data={
                'source': source,
                'confidence': confidence,
                'response_time_ms': response_time,
                'user_rating': user_feedback.get('rating') if user_feedback else None,
                'user_satisfied': user_feedback.get('satisfied') if user_feedback else None
            }
        )
        
        self.outcome_history.append(event)
        self.stats['total_outcomes'] += 1
        
        # Convert to world model observation
        try:
            from .world_model_core import Observation
            
            obs = Observation(
                timestamp=event.timestamp,
                variables={
                    'source': source,
                    'confidence': confidence,
                    'response_time_ms': response_time,
                    'used_reasoning': source != 'openai' and source != 'fallback',
                    'used_fallback': source == 'openai' or source == 'fallback',
                    'user_satisfied': event.data['user_satisfied']
                },
                domain='outcome',
                metadata={'query_id': query_id}
            )
            
            self.world_model.update_from_observation(obs)
            logger.debug(f"Outcome observed: {query_id} (source={source})")
            
        except Exception as e:
            logger.error(f"Failed to update world model on outcome: {e}")
    
    def observe_error(
        self,
        query_id: str,
        error_type: str,
        error_message: str,
        component: str
    ) -> None:
        """
        Observe system error.
        
        Tracks errors for pattern detection and self-improvement triggers.
        
        Args:
            query_id: Query identifier
            error_type: Type of error
            error_message: Error message
            component: Component where error occurred
        """
        if not self.enabled:
            return
        
        event = SystemEvent(
            event_type=EventType.ERROR,
            timestamp=time.time(),
            query_id=query_id,
            data={
                'error_type': error_type,
                'error_message': error_message[:ERROR_MESSAGE_TRUNCATION_LIMIT],
                'component': component
            }
        )
        
        self.engine_history.append(event)
        self.stats['errors_observed'] += 1
        
        # Feed to world model
        try:
            from .world_model_core import Observation
            
            obs = Observation(
                timestamp=event.timestamp,
                variables={
                    'error_occurred': True,
                    'error_type': error_type,
                    'component': component
                },
                domain='errors',
                metadata={'query_id': query_id}
            )
            
            self.world_model.update_from_observation(obs)
            logger.error(f"Error observed: {component} - {error_type}")
            
        except Exception as e:
            logger.error(f"Failed to update world model on error: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get observation statistics.
        
        Returns:
            Dictionary with observation counts and history sizes
        """
        return {
            **self.stats,
            'query_history_size': len(self.query_history),
            'engine_history_size': len(self.engine_history),
            'outcome_history_size': len(self.outcome_history),
            'enabled': self.enabled
        }
    
    def get_engine_performance(self, engine_name: str) -> Dict[str, Any]:
        """
        Get performance statistics for a specific engine.
        
        Args:
            engine_name: Name of the engine
            
        Returns:
            Dictionary with success rate, avg confidence, avg execution time
        """
        engine_events = [
            e for e in self.engine_history
            if e.event_type == EventType.ENGINE_RESULT
            and e.data.get('engine') == engine_name
        ]
        
        if not engine_events:
            return {'engine': engine_name, 'executions': 0, 'message': 'No data'}
        
        successes = sum(1 for e in engine_events if e.data.get('success', False))
        confidences = [e.data.get('confidence', 0) for e in engine_events]
        exec_times = [e.data.get('execution_time_ms', 0) for e in engine_events]
        
        return {
            'engine': engine_name,
            'executions': len(engine_events),
            'success_rate': successes / len(engine_events) if engine_events else 0,
            'avg_confidence': sum(confidences) / len(confidences) if confidences else 0,
            'avg_execution_time_ms': sum(exec_times) / len(exec_times) if exec_times else 0
        }
    
    def get_validation_failure_patterns(self) -> List[Dict[str, Any]]:
        """
        Get patterns of validation failures.
        
        Returns:
            List of failure patterns grouped by engine and query type
        """
        failures = [
            e for e in self.engine_history
            if e.event_type == EventType.VALIDATION_FAILURE
        ]
        
        # Group by engine + query_type
        patterns = {}
        for failure in failures:
            key = f"{failure.data.get('engine', 'unknown')}_{failure.data.get('query_type', 'unknown')}"
            if key not in patterns:
                patterns[key] = {
                    'engine': failure.data.get('engine'),
                    'query_type': failure.data.get('query_type'),
                    'count': 0,
                    'reasons': []
                }
            patterns[key]['count'] += 1
            reason = failure.data.get('reason', '')
            if reason and reason not in patterns[key]['reasons'][:5]:
                patterns[key]['reasons'].append(reason)
        
        return list(patterns.values())
    
    # Helper methods for feature detection using shared constants
    
    def _check_formal_logic(self, query: str) -> bool:
        """Check if query contains formal logic notation using shared constants."""
        if not query:
            return False
        # Check for symbols
        if any(sym in query for sym in FORMAL_LOGIC_SYMBOLS):
            return True
        # Check for keywords
        query_lower = query.lower()
        return any(kw in query_lower for kw in FORMAL_LOGIC_KEYWORDS)
    
    def _check_probability(self, query: str) -> bool:
        """Check if query involves probability using shared constants."""
        if not query:
            return False
        query_lower = query.lower()
        return any(kw in query_lower for kw in PROBABILITY_KEYWORDS)
    
    def _check_self_referential(self, query: str) -> bool:
        """Check if query is self-referential using shared constants."""
        if not query:
            return False
        query_lower = query.lower()
        return any(kw in query_lower for kw in SELF_REFERENTIAL_KEYWORDS)
    
    def _infer_query_type(self, query: str) -> str:
        """Infer query type from content"""
        if not query:
            return 'unknown'
        
        query_lower = query.lower()
        
        if self._check_formal_logic(query):
            return 'formal_logic'
        elif self._check_probability(query):
            return 'probabilistic'
        elif any(kw in query_lower for kw in ['compute', 'calculate', 'solve', 'integral', 'derivative']):
            return 'mathematical'
        elif self._check_self_referential(query):
            return 'self_referential'
        elif any(kw in query_lower for kw in ['cause', 'effect', 'causal', 'intervention']):
            return 'causal'
        elif any(kw in query_lower for kw in ['should', 'ethical', 'moral', 'right', 'wrong']):
            return 'ethical'
        else:
            return 'general'
    
    def _infer_result_type(self, result: Dict[str, Any]) -> str:
        """Infer result type from content"""
        if not result or not isinstance(result, dict):
            return 'unknown'
        
        conclusion = str(result.get('conclusion', ''))
        
        if 'x**' in conclusion or 'derivative' in conclusion.lower():
            return 'mathematical'
        elif any(sym in conclusion for sym in ['∀', '∃', '→', '∧', '⊢']):
            return 'formal_logic'
        elif 'exp(' in conclusion or 'sin(' in conclusion or 'cos(' in conclusion:
            return 'calculus'
        elif any(word in conclusion.lower() for word in ['probability', 'p(', 'bayes']):
            return 'probabilistic'
        else:
            return 'general'


# Singleton instance for global access
_system_observer: Optional[SystemObserver] = None


def get_system_observer() -> Optional[SystemObserver]:
    """
    Get the global SystemObserver instance.
    
    Returns:
        SystemObserver instance or None if not initialized
    """
    return _system_observer


def initialize_system_observer(world_model: "WorldModel") -> SystemObserver:
    """
    Initialize the global SystemObserver.
    
    Args:
        world_model: WorldModel instance to observe
        
    Returns:
        Initialized SystemObserver instance
    """
    global _system_observer
    _system_observer = SystemObserver(world_model)
    logger.info("Global SystemObserver initialized")
    return _system_observer
