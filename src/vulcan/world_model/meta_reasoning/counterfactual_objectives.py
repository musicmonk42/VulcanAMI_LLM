"""
counterfactual_objectives.py - Counterfactual objective reasoning
Part of the meta_reasoning subsystem for VULCAN-AMI

Performs counterfactual reasoning about alternative objectives:
- "What if I optimized for X instead of Y?"
- Predicts outcomes under different objective functions
- Finds Pareto-optimal solutions
- Estimates trade-offs between objectives

Uses VULCAN's causal reasoning and prediction engine for inference.
Learns from validation history to improve predictions.
"""

import logging
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
from itertools import combinations
from unittest.mock import MagicMock # ADDED as per fix steps

logger = logging.getLogger(__name__)


@dataclass
class CounterfactualOutcome:
    """Predicted outcome under counterfactual objective"""
    objective: str
    predicted_value: float
    confidence: float
    lower_bound: float
    upper_bound: float
    side_effects: Dict[str, float]
    computation_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ObjectiveComparison:
    """Comparison between two objectives"""
    objective_a: str
    objective_b: str
    outcome_a: CounterfactualOutcome
    outcome_b: CounterfactualOutcome
    winner: Optional[str]
    difference: float
    confidence: float
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParetoPoint:
    """Point on Pareto frontier"""
    objectives: Dict[str, float]
    dominated_by: List[int]
    dominates: List[int]
    is_pareto_optimal: bool
    objective_weights: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class CounterfactualObjectiveReasoner:
    """
    Answers: 'What if I optimized for X instead of Y?'
    
    Uses VULCAN's causal reasoning and prediction engine to perform
    counterfactual inference about alternative objectives.
    
    This enables the system to reason about:
    - What would happen if priorities changed
    - Which objectives are on the Pareto frontier
    - What trade-offs exist between objectives
    - How to modify proposals to satisfy different objectives
    
    Learns from validation history to improve predictions over time.
    """
    
    # --- START FIX: Modified __init__ ---
    def __init__(self, world_model=None):
        """
        Initialize counterfactual reasoner
        
        Args:
            world_model: Reference to parent WorldModel instance (optional, defaults to MagicMock)
        """
        self.world_model = world_model or MagicMock()
        
        # Prediction cache
        self.prediction_cache: Dict[str, CounterfactualOutcome] = {}
        self.cache_ttl = 300  # 5 minutes (Original value)
        # FIXED: Use 3600.0 from fix steps
        self.cache_ttl = 3600.0 
        self.cache_timestamps: Dict[str, float] = {}
        
        # Pareto frontier cache
        self.pareto_cache: Optional[List[ParetoPoint]] = None
        self.pareto_cache_time: Optional[float] = None
        self.pareto_cache_objectives: Optional[List[str]] = None
        
        # Learning from validation history
        # FIXED: Use defaultdict(float) from fix steps
        self.learned_objective_correlations: Dict[Tuple[str, str], float] = defaultdict(float) 
        self.prediction_accuracy_by_objective: Dict[str, List[float]] = defaultdict(list)
        
        # Statistics
        self.stats = defaultdict(int)
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info("CounterfactualObjectiveReasoner initialized")
    # --- END FIX ---
    
    def predict_under_objective(self,
                                alternative_objective: str,
                                context: Optional[Dict[str, Any]] = None) -> CounterfactualOutcome:
        """
        Predict outcome if system optimized for alternative objective
        
        Performs counterfactual inference:
        1. Identify causal paths relevant to objective
        2. Predict outcomes along those paths
        3. Estimate side effects on other objectives
        4. Use learned patterns to refine predictions
        
        Args:
            alternative_objective: Objective to optimize for
            context: Context for prediction (None will be converted to empty dict)
            
        Returns:
            Predicted outcome under alternative objective
        """
        
        # Handle None context
        if context is None:
            context = {}
        
        with self.lock:
            start_time = time.time()
            
            # Check cache
            cache_key = self._get_cache_key(alternative_objective, context)
            if self._is_cache_valid(cache_key):
                self.stats['cache_hits'] += 1
                return self.prediction_cache[cache_key]
            
            self.stats['cache_misses'] += 1
            
            # EXAMINE: Analyze what optimizing for this objective would mean
            
            # Get current state
            current_state = context.get('current_state', {})
            
            # Identify relevant variables for this objective
            relevant_vars = self._identify_relevant_variables(
                alternative_objective,
                context
            )
            
            # APPLY: Simulate optimization under alternative objective
            
            # Create modified context with alternative objective
            counterfactual_context = {
                **context,
                'primary_objective': alternative_objective,
                'optimize_for': alternative_objective
            }
            
            # Predict outcome using world model's prediction engine
            predicted_outcome = self._simulate_objective_optimization(
                alternative_objective,
                relevant_vars,
                counterfactual_context
            )
            
            # Refine prediction using learned patterns
            predicted_outcome = self._refine_prediction_with_history(
                alternative_objective,
                predicted_outcome,
                counterfactual_context
            )
            
            # Estimate side effects on other objectives
            side_effects = self._estimate_side_effects(
                alternative_objective,
                predicted_outcome,
                counterfactual_context
            )
            
            # Calculate confidence based on prediction uncertainty and historical accuracy
            confidence = self._calculate_prediction_confidence(
                alternative_objective,
                predicted_outcome,
                side_effects
            )
            
            # Extract value and ensure it's a float
            predicted_value = predicted_outcome.get('value', 0.0)
            try:
                predicted_value = float(predicted_value)
            except (TypeError, ValueError):
                predicted_value = 0.0
            
            # Create outcome
            outcome = CounterfactualOutcome(
                objective=alternative_objective,
                predicted_value=predicted_value,
                confidence=confidence,
                lower_bound=predicted_outcome.get('lower_bound', 0.0),
                upper_bound=predicted_outcome.get('upper_bound', 1.0),
                side_effects=side_effects,
                computation_time_ms=(time.time() - start_time) * 1000,
                metadata={
                    'relevant_variables': relevant_vars,
                    'prediction_details': predicted_outcome,
                    'learned_adjustments': predicted_outcome.get('historical_adjustment', 0.0)
                }
            )
            
            # REMEMBER: Cache result
            self.prediction_cache[cache_key] = outcome
            self.cache_timestamps[cache_key] = time.time()
            self.stats['predictions_computed'] += 1
            
            return outcome
    
    def compare_objectives(self,
                          obj_a: str,
                          obj_b: str,
                          scenario: Dict[str, Any]) -> ObjectiveComparison:
        """
        Compare outcomes under two different objectives
        
        Args:
            obj_a: First objective
            obj_b: Second objective
            scenario: Scenario/context for comparison
            
        Returns:
            Comparison of outcomes
        """
        
        with self.lock:
            # Predict under both objectives
            outcome_a = self.predict_under_objective(obj_a, scenario)
            outcome_b = self.predict_under_objective(obj_b, scenario)
            
            # Determine winner (accounting for maximize vs minimize)
            winner = None
            
            # Get objective directions
            maximize_a = self._get_objective_direction(obj_a)
            maximize_b = self._get_objective_direction(obj_b)
            
            # Compare based on directions
            if maximize_a and maximize_b:
                # Both maximize
                if outcome_a.predicted_value > outcome_b.predicted_value:
                    winner = obj_a
                elif outcome_b.predicted_value > outcome_a.predicted_value:
                    winner = obj_b
            elif not maximize_a and not maximize_b:
                # Both minimize
                if outcome_a.predicted_value < outcome_b.predicted_value:
                    winner = obj_a
                elif outcome_b.predicted_value < outcome_a.predicted_value:
                    winner = obj_b
            else:
                # Mixed directions - can't directly compare
                winner = None
            
            difference = abs(outcome_a.predicted_value - outcome_b.predicted_value)
            
            # Combined confidence (lower of the two)
            confidence = min(outcome_a.confidence, outcome_b.confidence)
            
            # Generate reasoning
            reasoning = self._generate_comparison_reasoning(
                obj_a, obj_b, outcome_a, outcome_b, winner
            )
            
            comparison = ObjectiveComparison(
                objective_a=obj_a,
                objective_b=obj_b,
                outcome_a=outcome_a,
                outcome_b=outcome_b,
                winner=winner,
                difference=difference,
                confidence=confidence,
                reasoning=reasoning,
                metadata={
                    'scenario': scenario,
                    'side_effects_a': outcome_a.side_effects,
                    'side_effects_b': outcome_b.side_effects
                }
            )
            
            self.stats['comparisons_performed'] += 1
            
            return comparison
    
    def find_pareto_frontier(self, objectives: List[str]) -> List[ParetoPoint]:
        """
        Find Pareto frontier in multi-objective space
        
        A point is Pareto optimal if no other point is better in all objectives.
        
        Args:
            objectives: List of objective names
            
        Returns:
            List of Pareto-optimal points
        """
        
        with self.lock:
            start_time = time.time()
            
            # Check cache
            if self._is_pareto_cache_valid(objectives):
                self.stats['pareto_cache_hits'] += 1
                return self.pareto_cache
            
            self.stats['pareto_cache_misses'] += 1
            
            # Generate candidate points in objective space
            candidate_points = self._generate_candidate_points(objectives)
            
            # Evaluate each candidate
            evaluated_points = []
            for point_weights in candidate_points:
                # Predict outcomes for this weight combination
                outcomes = {}
                for obj_name in objectives:
                    context = {'objective_weights': point_weights}
                    outcome = self.predict_under_objective(obj_name, context)
                    outcomes[obj_name] = outcome.predicted_value
                
                evaluated_points.append({
                    'weights': point_weights,
                    'outcomes': outcomes
                })
            
            # Find Pareto-optimal points
            pareto_points = self._identify_pareto_optimal(
                evaluated_points,
                objectives
            )
            
            # Cache results
            self.pareto_cache = pareto_points
            self.pareto_cache_time = time.time()
            self.pareto_cache_objectives = objectives.copy()
            
            self.stats['pareto_frontiers_computed'] += 1
            
            logger.info("Computed Pareto frontier with %d points in %.2fms",
                       len(pareto_points),
                       (time.time() - start_time) * 1000)
            
            return pareto_points
    
    def estimate_tradeoffs(self,
                          sacrifice_obj: str,
                          gain_obj: str) -> Dict[str, Any]:
        """
        Estimate trade-offs between objectives
        
        Answers: "How much of X do I lose to gain Y?"
        
        Args:
            sacrifice_obj: Objective being sacrificed
            gain_obj: Objective being gained
            
        Returns:
            Trade-off analysis
        """
        
        with self.lock:
            # Create scenarios with different weights
            scenarios = []
            
            for sacrifice_weight in np.linspace(0.0, 1.0, 11):
                gain_weight = 1.0 - sacrifice_weight
                
                scenario = {
                    'objective_weights': {
                        sacrifice_obj: sacrifice_weight,
                        gain_obj: gain_weight
                    }
                }
                
                # Predict outcomes
                sacrifice_outcome = self.predict_under_objective(sacrifice_obj, scenario)
                gain_outcome = self.predict_under_objective(gain_obj, scenario)
                
                scenarios.append({
                    'sacrifice_weight': sacrifice_weight,
                    'gain_weight': gain_weight,
                    'sacrifice_value': sacrifice_outcome.predicted_value,
                    'gain_value': gain_outcome.predicted_value
                })
            
            # Analyze trade-off curve
            tradeoff_analysis = self._analyze_tradeoff_curve(
                scenarios,
                sacrifice_obj,
                gain_obj
            )
            
            # Add learned correlation if available
            correlation_key = (sacrifice_obj, gain_obj)
            if correlation_key in self.learned_objective_correlations:
                tradeoff_analysis['learned_correlation'] = self.learned_objective_correlations[correlation_key]
            
            self.stats['tradeoff_analyses'] += 1
            
            return tradeoff_analysis
    
    def generate_alternative_proposals(self,
                                       current_proposal: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate alternative proposals optimizing for different objectives
        
        Uses learned patterns to create better alternatives.
        
        Args:
            current_proposal: Current proposal
            
        Returns:
            List of alternative proposals
        """
        
        with self.lock:
            alternatives = []
            
            # Get current objective
            current_obj = current_proposal.get('objective', 'unknown')
            
            # Get available objectives
            available_objectives = self._get_available_objectives()
            
            # Generate alternative for each objective
            for alt_obj in available_objectives:
                if alt_obj == current_obj:
                    continue
                
                # Create modified proposal
                alternative = self._create_alternative_proposal(
                    current_proposal,
                    alt_obj
                )
                
                # Enhance with learned patterns
                alternative = self._enhance_alternative_with_patterns(
                    alternative,
                    alt_obj
                )
                
                # Validate alternative
                if self._is_valid_alternative(alternative, current_proposal):
                    alternatives.append(alternative)
            
            # Sort by predicted quality
            alternatives = self._rank_alternatives(alternatives)
            
            self.stats['alternatives_generated'] += len(alternatives)
            
            return alternatives
    
    def update_prediction_accuracy(self, 
                                   objective: str,
                                   predicted: float,
                                   actual: float):
        """
        Update learned prediction accuracy
        
        Called when actual outcomes are known to improve future predictions
        
        Args:
            objective: Objective that was predicted
            predicted: Predicted value (can be float or CounterfactualOutcome)
            actual: Actual achieved value
        """
        
        with self.lock:
            # Handle CounterfactualOutcome objects
            if hasattr(predicted, 'predicted_value'):
                predicted = predicted.predicted_value
            
            # Ensure predicted is a float
            try:
                predicted = float(predicted)
            except (TypeError, ValueError):
                predicted = 0.5
            
            error = abs(predicted - actual)
            self.prediction_accuracy_by_objective[objective].append(1.0 - error)
            
            # Keep only recent history
            if len(self.prediction_accuracy_by_objective[objective]) > 100:
                self.prediction_accuracy_by_objective[objective] = \
                    self.prediction_accuracy_by_objective[objective][-100:]
            
            logger.debug("Updated prediction accuracy for %s: error=%.3f", objective, error)
    
    def learn_objective_correlation(self,
                                   obj_a: str,
                                   obj_b: str,
                                   correlation: float):
        """
        Learn correlation between objectives from observed data
        
        Args:
            obj_a: First objective
            obj_b: Second objective
            correlation: Observed correlation (-1 to 1)
        """
        
        with self.lock:
            # Store both directions
            self.learned_objective_correlations[(obj_a, obj_b)] = correlation
            self.learned_objective_correlations[(obj_b, obj_a)] = correlation
            
            logger.debug("Learned correlation between %s and %s: %.3f", 
                        obj_a, obj_b, correlation)
    
    def _learn_objective_correlation(self,
                                    obj_a: str,
                                    obj_b: str,
                                    correlation: float):
        """
        Learn correlation between objectives from observed data (private method)
        
        Args:
            obj_a: First objective
            obj_b: Second objective
            correlation: Observed correlation (-1 to 1)
        """
        
        with self.lock:
            # Store both directions
            self.learned_objective_correlations[(obj_a, obj_b)] = correlation
            self.learned_objective_correlations[(obj_b, obj_a)] = correlation
            
            logger.debug("Learned correlation between %s and %s: %.3f", 
                        obj_a, obj_b, correlation)
    
    def _refine_prediction_with_history(self,
                                       objective: str,
                                       predicted_outcome: Dict[str, Any],
                                       context: Dict[str, Any]) -> Dict[str, Any]:
        """Refine prediction using historical accuracy"""
        
        # Get historical accuracy for this objective
        if objective in self.prediction_accuracy_by_objective:
            accuracies = self.prediction_accuracy_by_objective[objective]
            if accuracies:
                avg_accuracy = np.mean(accuracies)
                
                # Adjust prediction bounds based on historical accuracy
                predicted_value = predicted_outcome.get('value', 0.5)
                # Ensure it's a float
                try:
                    predicted_value = float(predicted_value)
                except (TypeError, ValueError):
                    predicted_value = 0.5
                
                uncertainty = (1.0 - avg_accuracy) * 0.2  # Scale uncertainty
                
                predicted_outcome['lower_bound'] = max(0.0, predicted_value - uncertainty)
                predicted_outcome['upper_bound'] = min(1.0, predicted_value + uncertainty)
                predicted_outcome['historical_adjustment'] = uncertainty
                predicted_outcome['historical_accuracy'] = avg_accuracy
        
        return predicted_outcome
    
    def _enhance_alternative_with_patterns(self,
                                          alternative: Dict[str, Any],
                                          objective: str) -> Dict[str, Any]:
        """Enhance alternative using learned success patterns"""
        
        # Access validation tracker if available
        if hasattr(self.world_model, 'motivational_introspection'):
            mi = self.world_model.motivational_introspection
            if hasattr(mi, 'validation_tracker'):
                tracker = mi.validation_tracker
                
                # Get success patterns for this objective
                success_patterns = tracker.identify_success_patterns()
                
                # FIXED: Check if success_patterns is actually iterable before using it
                if success_patterns and hasattr(success_patterns, '__iter__'):
                    try:
                        # Find patterns matching this objective
                        relevant_patterns = [
                            p for p in success_patterns
                            if hasattr(p, 'features') and p.features.get('objective') == objective
                        ]
                        
                        # Apply highest-confidence pattern
                        if relevant_patterns:
                            best_pattern = max(relevant_patterns, key=lambda p: p.confidence)
                            
                            # Apply pattern features to alternative
                            for feature, value in best_pattern.features.items():
                                if feature not in alternative and feature != 'objective':
                                    alternative[feature] = value
                            
                            alternative['enhanced_with_pattern'] = True
                            alternative['pattern_confidence'] = best_pattern.confidence
                    except (TypeError, AttributeError) as e:
                        logger.debug("Could not enhance with patterns: %s", e)
        
        return alternative
    
    def _identify_relevant_variables(self,
                                     objective: str,
                                     context: Dict[str, Any]) -> List[str]:
        """Identify variables relevant to objective"""
        
        relevant_vars = []
        
        # Use causal graph if available
        if hasattr(self.world_model, 'causal_graph'):
            causal_graph = self.world_model.causal_graph
            
            # Find nodes causally connected to objective
            if hasattr(causal_graph, 'get_ancestors'):
                # Get all variables that causally influence this objective
                try:
                    ancestors = causal_graph.get_ancestors(objective)
                    relevant_vars.extend(ancestors)
                except (AttributeError, KeyError, ValueError, TypeError) as e:
                    logger.debug("Could not get ancestors for %s: %s", objective, e)
            
            if hasattr(causal_graph, 'get_descendants'):
                # Get all variables influenced by this objective
                try:
                    descendants = causal_graph.get_descendants(objective)
                    relevant_vars.extend(descendants)
                except (AttributeError, KeyError, ValueError, TypeError) as e:
                    logger.debug("Could not get descendants for %s: %s", objective, e)
            
            # Fallback to context if graph doesn't have these methods or they fail
            if not relevant_vars:
                relevant_vars = context.get('target_variables', [])
        else:
            # No causal graph - use context
            relevant_vars = context.get('target_variables', [])
        
        # Add variables from context
        if 'variables' in context:
            relevant_vars.extend(context['variables'])
        
        # Default: assume all variables are relevant
        if not relevant_vars:
            relevant_vars = ['default_variable']
        
        return list(set(relevant_vars))
    
    def _simulate_objective_optimization(self,
                                         objective: str,
                                         variables: List[str],
                                         context: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate optimization for an objective"""
        
        # Use world model's prediction engine if available
        if hasattr(self.world_model, 'ensemble_predictor'):
            try:
                # Get predictor
                predictor = self.world_model.ensemble_predictor
                
                # Create prediction context
                pred_context = {
                    'domain': context.get('domain', 'general'),
                    'targets': variables,
                    'objective': objective,
                    'optimize_for': objective
                }
                
                # Get current state
                current_state = context.get('current_state', {})
                
                # Make prediction for optimized scenario
                if hasattr(predictor, 'predict_optimized'):
                    prediction = predictor.predict_optimized(
                        objective=objective,
                        variables=variables,
                        context=pred_context
                    )
                    
                    return {
                        'value': float(prediction.get('value', 0.0)),
                        'lower_bound': float(prediction.get('lower', 0.0)),
                        'upper_bound': float(prediction.get('upper', 1.0)),
                        'method': 'prediction_engine',
                        'confidence': float(prediction.get('confidence', 0.7))
                    }
                
                # Fallback: use standard prediction
                elif hasattr(predictor, 'predict'):
                    # Create scenario with objective optimization
                    scenario = {
                        **current_state,
                        'optimize_for': objective
                    }
                    
                    prediction = predictor.predict(scenario, pred_context)
                    
                    value = prediction.get('prediction', 0.5)
                    uncertainty = prediction.get('uncertainty', 0.1)
                    
                    return {
                        'value': float(value),
                        'lower_bound': float(max(0.0, value - uncertainty)),
                        'upper_bound': float(min(1.0, value + uncertainty)),
                        'method': 'prediction_engine',
                        'confidence': float(1.0 - uncertainty)
                    }
            
            except Exception as e:
                logger.debug("Prediction engine failed: %s", e)
        
        # Fallback: heuristic estimation
        return self._heuristic_estimation(objective, variables, context)
    
    def _estimate_objective_value(self,
                                  objective: str,
                                  context: Dict[str, Any]) -> float:
        """Estimate achievable value for objective"""
        
        # Map objectives to typical achievable values
        objective_estimates = {
            'prediction_accuracy': 0.95,
            'uncertainty_calibration': 0.90,
            'safety': 1.0,
            'efficiency': 0.85,
            'latency': 0.80,
            'energy_efficiency': 0.75
        }
        
        base_value = objective_estimates.get(objective, 0.7)
        
        # Adjust based on historical accuracy if available
        if objective in self.prediction_accuracy_by_objective:
            accuracies = self.prediction_accuracy_by_objective[objective]
            if accuracies:
                # If we've historically overestimated, adjust down
                avg_accuracy = np.mean(accuracies)
                if avg_accuracy < 0.8:
                    base_value *= avg_accuracy
        
        # Adjust based on context
        if 'difficulty' in context:
            difficulty = context['difficulty']
            base_value *= (1.0 - 0.2 * difficulty)
        
        # Add some randomness to simulate uncertainty
        noise = np.random.normal(0, 0.05)
        
        return float(np.clip(base_value + noise, 0.0, 1.0))
    
    def _heuristic_estimation(self,
                             objective: str,
                             variables: List[str],
                             context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback heuristic estimation"""
        
        estimated_value = self._estimate_objective_value(objective, context)
        
        return {
            'value': float(estimated_value),
            'lower_bound': float(max(0.0, estimated_value - 0.1)),
            'upper_bound': float(min(1.0, estimated_value + 0.1)),
            'method': 'heuristic'
        }
    
    def _estimate_side_effects(self,
                               primary_objective: str,
                               predicted_outcome: Dict[str, Any],
                               context: Dict[str, Any]) -> Dict[str, float]:
        """Estimate side effects on other objectives"""
        
        side_effects = {}
        
        # Get all objectives
        available_objectives = self._get_available_objectives()
        
        # Extract value and ensure it's a float, not Mock
        primary_value = predicted_outcome.get('value', 0.5)
        try:
            primary_value = float(primary_value)
        except (TypeError, ValueError):
            # If it's a Mock or can't convert, use default
            primary_value = 0.5
        
        for obj in available_objectives:
            if obj == primary_objective:
                continue
            
            # Use learned correlation if available
            correlation_key = (primary_objective, obj)
            if correlation_key in self.learned_objective_correlations:
                impact = self.learned_objective_correlations[correlation_key] * primary_value
            else:
                # Estimate impact based on default relationships
                impact = self._estimate_objective_impact(
                    primary_objective,
                    obj,
                    primary_value
                )
            
            if abs(impact) > 0.05:  # Only include significant effects
                side_effects[obj] = impact
        
        return side_effects
    
    def _estimate_objective_impact(self,
                                   source_obj: str,
                                   target_obj: str,
                                   source_value: float) -> float:
        """Estimate impact of optimizing source on target objective"""
        
        # Known objective relationships (simplified)
        relationships = {
            ('efficiency', 'prediction_accuracy'): -0.3,  # Speed/accuracy tradeoff
            ('prediction_accuracy', 'efficiency'): -0.2,
            ('safety', 'efficiency'): -0.1,
            ('efficiency', 'safety'): -0.15,
        }
        
        impact_coefficient = relationships.get((source_obj, target_obj), 0.0)
        
        # Ensure source_value is float
        try:
            source_value = float(source_value)
        except (TypeError, ValueError):
            source_value = 0.5
        
        # Impact scales with achievement of source objective
        return impact_coefficient * source_value
    
    def _calculate_prediction_confidence(self,
                                         objective: str,
                                         predicted_outcome: Dict[str, Any],
                                         side_effects: Dict[str, float]) -> float:
        """Calculate confidence in counterfactual prediction"""
        
        # Base confidence from prediction method
        if predicted_outcome.get('method') == 'prediction_engine':
            base_confidence = 0.8
        elif predicted_outcome.get('method') == 'simulation':
            base_confidence = 0.75
        else:
            base_confidence = 0.6
        
        # Adjust based on historical accuracy
        if objective in self.prediction_accuracy_by_objective:
            accuracies = self.prediction_accuracy_by_objective[objective]
            if accuracies:
                historical_accuracy = np.mean(accuracies)
                base_confidence = 0.7 * base_confidence + 0.3 * historical_accuracy
        
        # Reduce confidence if many side effects
        side_effect_penalty = len(side_effects) * 0.05
        
        # Reduce confidence if outcome bounds are wide
        value_range = predicted_outcome.get('upper_bound', 1.0) - predicted_outcome.get('lower_bound', 0.0)
        uncertainty_penalty = value_range * 0.2
        
        confidence = base_confidence - side_effect_penalty - uncertainty_penalty
        
        return float(np.clip(confidence, 0.0, 1.0))
    
    def _generate_comparison_reasoning(self,
                                       obj_a: str,
                                       obj_b: str,
                                       outcome_a: CounterfactualOutcome,
                                       outcome_b: CounterfactualOutcome,
                                       winner: Optional[str]) -> str:
        """Generate reasoning for objective comparison"""
        
        if winner == obj_a:
            return (f"Optimizing for {obj_a} (value={outcome_a.predicted_value:.3f}) "
                   f"achieves better outcome than {obj_b} (value={outcome_b.predicted_value:.3f})")
        elif winner == obj_b:
            return (f"Optimizing for {obj_b} (value={outcome_b.predicted_value:.3f}) "
                   f"achieves better outcome than {obj_a} (value={outcome_a.predicted_value:.3f})")
        else:
            return (f"Objectives {obj_a} and {obj_b} achieve similar outcomes "
                   f"({outcome_a.predicted_value:.3f} vs {outcome_b.predicted_value:.3f})")
    
    def _generate_candidate_points(self, objectives: List[str]) -> List[Dict[str, float]]:
        """Generate candidate points in multi-objective space"""
        
        candidates = []
        n_objectives = len(objectives)
        
        if n_objectives == 0:
            return candidates
        
        # For single objective, return only one point
        if n_objectives == 1:
            weights = {objectives[0]: 1.0}
            candidates.append(weights)
            return candidates
        
        # For 2 objectives, use fine grid
        if n_objectives == 2:
            for w1 in np.linspace(0.0, 1.0, 11):
                weights = {
                    objectives[0]: w1,
                    objectives[1]: 1.0 - w1
                }
                candidates.append(weights)
        
        # For 3-4 objectives, use Dirichlet sampling
        elif n_objectives <= 4:
            # Corner points (single-objective optima)
            for i, obj in enumerate(objectives):
                weights = {o: 1.0 if o == obj else 0.0 for o in objectives}
                candidates.append(weights)
            
            # Edge midpoints (two-objective combinations)
            for i in range(n_objectives):
                for j in range(i + 1, n_objectives):
                    weights = {o: 0.5 if o in [objectives[i], objectives[j]] else 0.0 
                              for o in objectives}
                    candidates.append(weights)
            
            # Random Dirichlet samples
            for _ in range(50):
                weights_array = np.random.dirichlet(np.ones(n_objectives))
                weights = {obj: w for obj, w in zip(objectives, weights_array)}
                candidates.append(weights)
        
        # For many objectives, use more random samples
        else:
            # Corner points
            for i, obj in enumerate(objectives):
                weights = {o: 1.0 if o == obj else 0.0 for o in objectives}
                candidates.append(weights)
            
            # Many random samples
            for _ in range(100):
                weights_array = np.random.dirichlet(np.ones(n_objectives))
                weights = {obj: w for obj, w in zip(objectives, weights_array)}
                candidates.append(weights)
        
        return candidates
    
    def _identify_pareto_optimal(self,
                                 evaluated_points: List[Dict[str, Any]],
                                 objectives: List[str]) -> List[ParetoPoint]:
        """Identify Pareto-optimal points from evaluated candidates"""
        
        pareto_points = []
        n_points = len(evaluated_points)
        
        for i, point in enumerate(evaluated_points):
            outcomes = point['outcomes']
            
            # Check if this point is dominated by any other point
            is_dominated = False
            dominates_list = []
            dominated_by_list = []
            
            for j, other_point in enumerate(evaluated_points):
                if i == j:
                    continue
                
                other_outcomes = other_point['outcomes']
                
                # Check dominance
                dominance = self._check_dominance(outcomes, other_outcomes, objectives)
                
                if dominance == 'dominated':
                    is_dominated = True
                    dominated_by_list.append(j)
                elif dominance == 'dominates':
                    dominates_list.append(j)
            
            # Create Pareto point
            pareto_point = ParetoPoint(
                objectives=outcomes,
                dominated_by=dominated_by_list,
                dominates=dominates_list,
                is_pareto_optimal=not is_dominated,
                objective_weights=point['weights']
            )
            
            pareto_points.append(pareto_point)
        
        # Return only Pareto-optimal points
        return [p for p in pareto_points if p.is_pareto_optimal]
    
    def _check_dominance(self,
                        outcomes_a: Dict[str, float],
                        outcomes_b: Dict[str, float],
                        objectives: List[str]) -> str:
        """
        Check dominance relationship between two points
        
        Returns: 'dominates', 'dominated', or 'nondominated'
        """
        
        better_in_all = True
        worse_in_all = True
        
        for obj in objectives:
            val_a = outcomes_a.get(obj, 0.0)
            val_b = outcomes_b.get(obj, 0.0)
            
            # Get objective direction (maximize by default)
            maximize = self._get_objective_direction(obj)
            
            # Check based on direction
            if maximize:
                if val_a < val_b:
                    better_in_all = False
                if val_a > val_b:
                    worse_in_all = False
            else:  # minimize
                if val_a > val_b:
                    better_in_all = False
                if val_a < val_b:
                    worse_in_all = False
        
        if better_in_all and not worse_in_all:
            return 'dominates'
        elif worse_in_all and not better_in_all:
            return 'dominated'
        else:
            return 'nondominated'
    
    def _get_objective_direction(self, objective: str) -> bool:
        """
        Get whether objective should be maximized
        
        Returns:
            True if maximize, False if minimize
        """
        # Try to get from objective hierarchy
        if hasattr(self.world_model, 'motivational_introspection'):
            mi = self.world_model.motivational_introspection
            if hasattr(mi, 'objective_hierarchy'):
                # FIXED: Check type and handle Mock objects before using 'in' operator
                try:
                    objectives = mi.objective_hierarchy.objectives
                    if hasattr(objectives, '__contains__') and objective in objectives:
                        obj = objectives[objective]
                        if hasattr(obj, 'maximize'):
                            return obj.maximize
                except (TypeError, AttributeError):
                    pass
        
        # Default: maximize (standard for most objectives)
        return True
    
    def _analyze_tradeoff_curve(self,
                                scenarios: List[Dict[str, Any]],
                                sacrifice_obj: str,
                                gain_obj: str) -> Dict[str, Any]:
        """Analyze trade-off curve between objectives"""
        
        # Extract values
        sacrifice_values = [s['sacrifice_value'] for s in scenarios]
        gain_values = [s['gain_value'] for s in scenarios]
        
        # Calculate marginal rates
        marginal_rates = []
        for i in range(1, len(scenarios)):
            d_sacrifice = sacrifice_values[i] - sacrifice_values[i-1]
            d_gain = gain_values[i] - gain_values[i-1]
            
            if abs(d_sacrifice) > 1e-6:
                marginal_rate = d_gain / d_sacrifice
                marginal_rates.append(marginal_rate)
        
        # Find optimal trade-off point
        if marginal_rates:
            # Point where marginal rate is closest to -1 (equal trade-off)
            optimal_idx = np.argmin([abs(mr + 1.0) for mr in marginal_rates])
            optimal_scenario = scenarios[optimal_idx + 1]
        else:
            optimal_scenario = scenarios[len(scenarios) // 2]
        
        # Extract weights from optimal scenario
        optimal_weights = {
            sacrifice_obj: optimal_scenario['sacrifice_weight'],
            gain_obj: optimal_scenario['gain_weight']
        }
        
        # Calculate overall tradeoff score
        tradeoff_score = optimal_scenario['gain_value'] - optimal_scenario['sacrifice_value']
        
        return {
            'sacrifice_objective': sacrifice_obj,
            'gain_objective': gain_obj,
            'scenarios': scenarios,
            'marginal_rates': marginal_rates,
            'optimal_tradeoff': optimal_scenario,
            'optimal_weights': optimal_weights,
            'weights': optimal_weights,
            'tradeoff_score': tradeoff_score,
            'tradeoff_curve_shape': self._classify_curve_shape(scenarios),
            'recommendation': self._generate_tradeoff_recommendation(
                sacrifice_obj, gain_obj, optimal_scenario
            )
        }
    
    def _classify_curve_shape(self, scenarios: List[Dict[str, Any]]) -> str:
        """Classify shape of trade-off curve"""
        
        # Linear, convex, or concave
        if len(scenarios) < 3:
            return 'insufficient_data'
        
        # Check second derivative
        second_derivatives = []
        for i in range(1, len(scenarios) - 1):
            d2 = (scenarios[i+1]['gain_value'] - 2*scenarios[i]['gain_value'] + 
                  scenarios[i-1]['gain_value'])
            second_derivatives.append(d2)
        
        avg_curvature = np.mean(second_derivatives)
        
        if abs(avg_curvature) < 0.01:
            return 'linear'
        elif avg_curvature > 0:
            return 'convex'
        else:
            return 'concave'
    
    def _generate_tradeoff_recommendation(self,
                                          sacrifice_obj: str,
                                          gain_obj: str,
                                          optimal_scenario: Dict[str, Any]) -> str:
        """Generate recommendation for trade-off"""
        
        sacrifice_weight = optimal_scenario['sacrifice_weight']
        gain_weight = optimal_scenario['gain_weight']
        
        return (f"Optimal trade-off: {sacrifice_weight:.0%} {sacrifice_obj}, "
               f"{gain_weight:.0%} {gain_obj}")
    
    def _get_available_objectives(self) -> List[str]:
        """Get list of available objectives"""
        
        # Try to get from motivational introspection
        if hasattr(self.world_model, 'motivational_introspection'):
            mi = self.world_model.motivational_introspection
            if hasattr(mi, 'active_objectives'):
                # FIXED: Check if it's actually a dict before trying to iterate
                try:
                    if isinstance(mi.active_objectives, dict):
                        return list(mi.active_objectives.keys())
                except (TypeError, AttributeError):
                    pass
        
        # Fallback to common objectives
        return [
            'prediction_accuracy',
            'uncertainty_calibration',
            'safety',
            'efficiency'
        ]
    
    def _create_alternative_proposal(self,
                                     current_proposal: Dict[str, Any],
                                     alternative_objective: str) -> Dict[str, Any]:
        """Create alternative proposal for different objective"""
        
        alternative = current_proposal.copy()
        
        # Change primary objective
        alternative['objective'] = alternative_objective
        
        # Modify the value to ensure it differs from original
        current_value = current_proposal.get('value', 0.5)
        
        # Adjust parameters based on objective
        if alternative_objective == 'efficiency':
            # Relax accuracy constraints for efficiency
            if 'target_accuracy' in alternative:
                alternative['target_accuracy'] *= 0.95
            # Increase value slightly for efficiency
            alternative['value'] = min(1.0, current_value * 1.1)
        
        elif alternative_objective == 'prediction_accuracy':
            # Increase accuracy target
            if 'target_accuracy' in alternative:
                alternative['target_accuracy'] = min(0.99, alternative['target_accuracy'] * 1.05)
            # Decrease value slightly as accuracy is harder
            alternative['value'] = max(0.0, current_value * 0.95)
        
        elif alternative_objective == 'safety':
            # Maximize safety margins
            if 'safety_margin' in alternative:
                alternative['safety_margin'] *= 1.5
            # Increase value for safety
            alternative['value'] = min(1.0, current_value * 1.05)
        
        else:
            # For other objectives, make a small adjustment
            alternative['value'] = max(0.0, min(1.0, current_value * 0.98))
        
        # Mark as alternative
        alternative['is_alternative'] = True
        alternative['original_objective'] = current_proposal.get('objective')
        
        return alternative
    
    def _is_valid_alternative(self,
                             alternative: Dict[str, Any],
                             original: Dict[str, Any]) -> bool:
        """Check if alternative is valid"""
        
        # Alternative should be meaningfully different
        if alternative.get('objective') == original.get('objective'):
            return False
        
        # Should satisfy basic constraints
        # This is simplified - real implementation would validate thoroughly
        return True
    
    def _rank_alternatives(self, alternatives: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank alternatives by predicted quality"""
        
        scored_alternatives = []
        
        for alt in alternatives:
            # Predict outcome
            obj = alt.get('objective', 'unknown')
            outcome = self.predict_under_objective(obj, {'proposal': alt})
            
            scored_alternatives.append({
                'proposal': alt,
                'score': outcome.predicted_value,
                'confidence': outcome.confidence
            })
        
        # Sort by score
        scored_alternatives.sort(key=lambda x: x['score'], reverse=True)
        
        return [sa['proposal'] for sa in scored_alternatives]
    
    def _get_cache_key(self, objective: str, context: Dict[str, Any]) -> str:
        """Generate cache key"""
        
        # Simple key based on objective and context hash
        context_str = str(sorted(context.items()))
        return f"{objective}:{hash(context_str)}"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is valid"""
        
        if cache_key not in self.prediction_cache:
            return False
        
        timestamp = self.cache_timestamps.get(cache_key, 0)
        age = time.time() - timestamp
        
        return age < self.cache_ttl
    
    def _is_pareto_cache_valid(self, objectives: List[str]) -> bool:
        """Check if Pareto cache is valid"""
        
        if self.pareto_cache is None or self.pareto_cache_time is None:
            return False
        
        # Check if objectives match
        if self.pareto_cache_objectives != objectives:
            return False
        
        age = time.time() - self.pareto_cache_time
        return age < self.cache_ttl
    
    def clear_cache(self):
        """Clear prediction cache"""
        
        with self.lock:
            self.prediction_cache.clear()
            self.cache_timestamps.clear()
            self.pareto_cache = None
            self.pareto_cache_time = None
            self.pareto_cache_objectives = None
            logger.info("Cleared counterfactual prediction cache")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics"""
        
        with self.lock:
            # Calculate average prediction accuracy by objective
            avg_accuracies = {}
            for obj, accuracies in self.prediction_accuracy_by_objective.items():
                if accuracies:
                    avg_accuracies[obj] = np.mean(accuracies)
            
            return {
                'statistics': dict(self.stats),
                'cache_size': len(self.prediction_cache),
                'cache_hit_rate': (
                    self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses'])
                    if (self.stats['cache_hits'] + self.stats['cache_misses']) > 0 else 0.0
                ),
                'learned_correlations': len(self.learned_objective_correlations),
                'average_prediction_accuracy': avg_accuracies,
                'objectives_tracked': list(self.prediction_accuracy_by_objective.keys())
            }
