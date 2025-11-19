"""
Utility Model for Tool Selection System

Defines and computes utility functions that balance quality, time, energy, and risk
to optimize tool selection decisions based on context and constraints.

Fixed version with variable name collision resolved and proper error handling.
"""

import numpy as np
import time as time_module
import threading
from typing import Dict, Any, Optional, Tuple, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class ContextMode(Enum):
    """Operational context modes"""
    RUSH = "rush"               # Prioritize speed
    ACCURATE = "accurate"       # Prioritize quality
    EFFICIENT = "efficient"     # Prioritize energy efficiency
    BALANCED = "balanced"       # Balance all factors
    EXPLORATORY = "exploratory" # Exploration/learning mode
    CONSERVATIVE = "conservative" # Risk-averse mode


@dataclass
class UtilityWeights:
    """Weights for utility components"""
    quality: float = 1.0
    time_penalty: float = 1.0
    energy_penalty: float = 1.0
    risk_penalty: float = 1.0
    
    def normalize(self):
        """Normalize weights to sum to 1"""
        total = self.quality + self.time_penalty + self.energy_penalty + self.risk_penalty
        if total > 0:
            self.quality /= total
            self.time_penalty /= total
            self.energy_penalty /= total
            self.risk_penalty /= total
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            'quality': self.quality,
            'time_penalty': self.time_penalty,
            'energy_penalty': self.energy_penalty,
            'risk_penalty': self.risk_penalty
        }


@dataclass
class UtilityContext:
    """Context for utility computation"""
    mode: ContextMode
    time_budget: float      # milliseconds
    energy_budget: float    # millijoules
    min_quality: float      # [0, 1]
    max_risk: float         # [0, 1]
    user_preferences: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UtilityComponents:
    """Individual utility components"""
    quality_score: float
    time_score: float
    energy_score: float
    risk_score: float
    raw_quality: float
    raw_time: float
    raw_energy: float
    raw_risk: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            'quality_score': self.quality_score,
            'time_score': self.time_score,
            'energy_score': self.energy_score,
            'risk_score': self.risk_score,
            'raw_quality': self.raw_quality,
            'raw_time': self.raw_time,
            'raw_energy': self.raw_energy,
            'raw_risk': self.raw_risk
        }


class UtilityFunction:
    """Base class for utility functions"""
    
    def compute(self, value: float, context: Any) -> float:
        """Compute utility for a value"""
        raise NotImplementedError
    
    def gradient(self, value: float, context: Any) -> float:
        """Compute gradient of utility function"""
        try:
            # Numerical gradient by default
            epsilon = 1e-6
            return (self.compute(value + epsilon, context) - 
                    self.compute(value - epsilon, context)) / (2 * epsilon)
        except Exception as e:
            logger.error(f"Gradient computation failed: {e}")
            return 0.0


class LinearUtility(UtilityFunction):
    """Linear utility function"""
    
    def __init__(self, scale: float = 1.0, offset: float = 0.0):
        self.scale = scale
        self.offset = offset
    
    def compute(self, value: float, context: Any) -> float:
        """Linear utility: u(x) = scale * x + offset"""
        try:
            return self.scale * value + self.offset
        except Exception as e:
            logger.error(f"Linear utility computation failed: {e}")
            return 0.0
    
    def gradient(self, value: float, context: Any) -> float:
        """Gradient of linear utility"""
        return self.scale


class ExponentialUtility(UtilityFunction):
    """Exponential utility function (for risk-aversion)"""
    
    def __init__(self, risk_aversion: float = 1.0):
        self.risk_aversion = risk_aversion
    
    def compute(self, value: float, context: Any) -> float:
        """Exponential utility: u(x) = 1 - exp(-risk_aversion * x)"""
        try:
            return 1.0 - np.exp(-self.risk_aversion * value)
        except Exception as e:
            logger.error(f"Exponential utility computation failed: {e}")
            return 0.0
    
    def gradient(self, value: float, context: Any) -> float:
        """Gradient of exponential utility"""
        try:
            return self.risk_aversion * np.exp(-self.risk_aversion * value)
        except Exception as e:
            logger.error(f"Exponential gradient computation failed: {e}")
            return 0.0


class LogarithmicUtility(UtilityFunction):
    """Logarithmic utility function (diminishing returns)"""
    
    def __init__(self, scale: float = 1.0):
        self.scale = scale
    
    def compute(self, value: float, context: Any) -> float:
        """Logarithmic utility: u(x) = scale * log(1 + x)"""
        try:
            return self.scale * np.log(1 + max(0, value))
        except Exception as e:
            logger.error(f"Logarithmic utility computation failed: {e}")
            return 0.0
    
    def gradient(self, value: float, context: Any) -> float:
        """Gradient of logarithmic utility"""
        try:
            return self.scale / (1 + max(0, value))
        except Exception as e:
            logger.error(f"Logarithmic gradient computation failed: {e}")
            return 0.0


class ThresholdUtility(UtilityFunction):
    """Threshold-based utility function"""
    
    def __init__(self, threshold: float, above_value: float = 1.0, below_value: float = 0.0):
        self.threshold = threshold
        self.above_value = above_value
        self.below_value = below_value
    
    def compute(self, value: float, context: Any) -> float:
        """Threshold utility: high if above threshold, low otherwise"""
        try:
            return self.above_value if value >= self.threshold else self.below_value
        except Exception as e:
            logger.error(f"Threshold utility computation failed: {e}")
            return 0.0


class SigmoidUtility(UtilityFunction):
    """Sigmoid utility function (smooth threshold)"""
    
    def __init__(self, center: float = 0.5, steepness: float = 10.0):
        self.center = center
        self.steepness = steepness
    
    def compute(self, value: float, context: Any) -> float:
        """Sigmoid utility: u(x) = 1 / (1 + exp(-steepness * (x - center)))"""
        try:
            return 1.0 / (1.0 + np.exp(-self.steepness * (value - self.center)))
        except Exception as e:
            logger.error(f"Sigmoid utility computation failed: {e}")
            return 0.5
    
    def gradient(self, value: float, context: Any) -> float:
        """Gradient of sigmoid utility"""
        try:
            s = self.compute(value, context)
            return self.steepness * s * (1 - s)
        except Exception as e:
            logger.error(f"Sigmoid gradient computation failed: {e}")
            return 0.0


class UtilityModel:
    """
    Main utility model for tool selection
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        # Default weights for different modes
        self.mode_weights = {
            ContextMode.RUSH: UtilityWeights(
                quality=0.7, time_penalty=2.0, energy_penalty=0.5, risk_penalty=1.0
            ),
            ContextMode.ACCURATE: UtilityWeights(
                quality=2.0, time_penalty=0.3, energy_penalty=0.5, risk_penalty=1.5
            ),
            ContextMode.EFFICIENT: UtilityWeights(
                quality=0.8, time_penalty=1.0, energy_penalty=2.0, risk_penalty=1.0
            ),
            ContextMode.BALANCED: UtilityWeights(
                quality=1.0, time_penalty=1.0, energy_penalty=1.0, risk_penalty=1.0
            ),
            ContextMode.EXPLORATORY: UtilityWeights(
                quality=0.5, time_penalty=0.5, energy_penalty=0.5, risk_penalty=0.5
            ),
            ContextMode.CONSERVATIVE: UtilityWeights(
                quality=1.5, time_penalty=1.0, energy_penalty=1.0, risk_penalty=2.0
            )
        }
        
        # Component utility functions
        self.quality_function = SigmoidUtility(center=0.7, steepness=10.0)
        self.time_function = ExponentialUtility(risk_aversion=2.0)
        self.energy_function = LinearUtility(scale=-1.0)
        self.risk_function = ExponentialUtility(risk_aversion=3.0)
        
        # Normalization parameters (learned from data)
        self.normalization = {
            'time_ms': config.get('typical_time_ms', 1000),
            'energy_mj': config.get('typical_energy_mj', 100),
            'quality_scale': config.get('quality_scale', 1.0),
            'risk_scale': config.get('risk_scale', 1.0)
        }
        
        # User preference learning
        self.preference_history = deque(maxlen=1000)
        self.learned_weights = {}
        
        # Utility computation cache
        self.cache = {}
        self.cache_ttl = config.get('cache_ttl', 60)  # seconds
        
        # Statistics
        self.computation_stats = defaultdict(lambda: {
            'count': 0,
            'avg_utility': 0.0,
            'avg_time_ms': 0.0
        })
        
        # CRITICAL FIX: Add locks for thread safety
        self.cache_lock = threading.RLock()
        self.stats_lock = threading.RLock()
        self.weights_lock = threading.RLock()
    
    # CRITICAL FIX: Resolve variable name collision
    def compute_utility(self, 
                       quality: float,
                       time: float,
                       energy: float,
                       risk: float,
                       context: Optional[Union[Dict[str, Any], UtilityContext]] = None) -> float:
        """
        Compute overall utility
        
        Args:
            quality: Quality score [0, 1]
            time: Execution time in milliseconds
            energy: Energy consumption in millijoules
            risk: Risk level [0, 1]
            context: Context for utility computation
            
        Returns:
            Overall utility value
        """
        
        try:
            # CRITICAL FIX: Get current time to avoid variable collision
            current_time = time_module.time()
            
            # Convert dict context to UtilityContext
            if isinstance(context, dict):
                context = self._dict_to_context(context)
            elif context is None:
                context = UtilityContext(
                    mode=ContextMode.BALANCED,
                    time_budget=5000,
                    energy_budget=1000,
                    min_quality=0.5,
                    max_risk=0.5
                )
            
            # Check cache
            cache_key = self._compute_cache_key(quality, time, energy, risk, context)
            
            with self.cache_lock:
                if cache_key in self.cache:
                    cached_result, timestamp = self.cache[cache_key]
                    # CRITICAL FIX: Use current_time instead of time parameter
                    if current_time - timestamp < self.cache_ttl:
                        return cached_result
            
            # Get weights for context
            weights = self.get_weights(context)
            
            # Compute components
            components = self.compute_components(quality, time, energy, risk, context)
            
            # Weighted combination
            utility = (
                weights.quality * components.quality_score -
                weights.time_penalty * components.time_score -
                weights.energy_penalty * components.energy_score -
                weights.risk_penalty * components.risk_score
            )
            
            # Cache result
            with self.cache_lock:
                self.cache[cache_key] = (utility, current_time)
                
                # CRITICAL FIX: Prevent unbounded cache growth
                if len(self.cache) > 10000:
                    # Remove oldest 20% of entries
                    sorted_items = sorted(self.cache.items(), key=lambda x: x[1][1])
                    self.cache = dict(sorted_items[-8000:])
            
            # Update statistics
            self._update_statistics(context.mode, utility, time)
            
            return utility
        except Exception as e:
            logger.error(f"Utility computation failed: {e}")
            return 0.0
    
    def compute_components(self,
                          quality: float,
                          time_ms: float,
                          energy_mj: float,
                          risk: float,
                          context: UtilityContext) -> UtilityComponents:
        """
        Compute individual utility components
        """
        
        try:
            # Normalize inputs
            norm_time = min(1.0, time_ms / context.time_budget) if context.time_budget > 0 else 1.0
            norm_energy = min(1.0, energy_mj / context.energy_budget) if context.energy_budget > 0 else 1.0
            
            # Compute component scores
            quality_score = self.quality_function.compute(quality, context)
            time_score = self.time_function.compute(norm_time, context)
            energy_score = self.energy_function.compute(norm_energy, context)
            risk_score = self.risk_function.compute(risk, context)
            
            return UtilityComponents(
                quality_score=quality_score,
                time_score=time_score,
                energy_score=energy_score,
                risk_score=risk_score,
                raw_quality=quality,
                raw_time=time_ms,
                raw_energy=energy_mj,
                raw_risk=risk
            )
        except Exception as e:
            logger.error(f"Component computation failed: {e}")
            return UtilityComponents(
                quality_score=0.0, time_score=0.0, energy_score=0.0, risk_score=0.0,
                raw_quality=quality, raw_time=time_ms, raw_energy=energy_mj, raw_risk=risk
            )
    
    def get_weights(self, context: UtilityContext) -> UtilityWeights:
        """
        Get weights for given context
        """
        
        try:
            with self.weights_lock:
                # Start with mode-specific weights
                if context.mode in self.mode_weights:
                    weights = UtilityWeights(**self.mode_weights[context.mode].to_dict())
                else:
                    weights = UtilityWeights()
                
                # Apply user preferences
                if context.user_preferences:
                    self._apply_user_preferences(weights, context.user_preferences)
                
                # Apply learned adjustments
                if context.mode in self.learned_weights:
                    self._apply_learned_weights(weights, self.learned_weights[context.mode])
                
                # Normalize
                weights.normalize()
                
                return weights
        except Exception as e:
            logger.error(f"Weight computation failed: {e}")
            return UtilityWeights()
    
    def _apply_user_preferences(self, weights: UtilityWeights, preferences: Dict[str, float]):
        """Apply user preference adjustments"""
        
        try:
            if 'quality_importance' in preferences:
                weights.quality *= preferences['quality_importance']
            
            if 'time_sensitivity' in preferences:
                weights.time_penalty *= preferences['time_sensitivity']
            
            if 'energy_consciousness' in preferences:
                weights.energy_penalty *= preferences['energy_consciousness']
            
            if 'risk_tolerance' in preferences:
                weights.risk_penalty *= (2.0 - preferences['risk_tolerance'])
        except Exception as e:
            logger.error(f"User preference application failed: {e}")
    
    def _apply_learned_weights(self, weights: UtilityWeights, learned: Dict[str, float]):
        """Apply learned weight adjustments"""
        
        try:
            alpha = 0.3  # Learning rate
            
            for key in ['quality', 'time_penalty', 'energy_penalty', 'risk_penalty']:
                if key in learned:
                    current = getattr(weights, key)
                    adjusted = (1 - alpha) * current + alpha * learned[key]
                    setattr(weights, key, adjusted)
        except Exception as e:
            logger.error(f"Learned weight application failed: {e}")
    
    def compute_expected_utility(self, 
                                 quality_dist: Tuple[float, float],
                                 time_dist: Tuple[float, float],
                                 energy_dist: Tuple[float, float],
                                 risk_dist: Tuple[float, float],
                                 context: UtilityContext) -> float:
        """
        Compute expected utility from distributions
        
        Args:
            quality_dist: (mean, std) of quality
            time_dist: (mean, std) of time
            energy_dist: (mean, std) of energy
            risk_dist: (mean, std) of risk
            context: Utility context
            
        Returns:
            Expected utility
        """
        
        try:
            # Simple approximation using means
            # Could use Monte Carlo for more accuracy
            mean_utility = self.compute_utility(
                quality_dist[0],
                time_dist[0],
                energy_dist[0],
                risk_dist[0],
                context
            )
            
            # Penalty for uncertainty (risk aversion)
            time_budget = context.time_budget if context.time_budget > 0 else 1000
            energy_budget = context.energy_budget if context.energy_budget > 0 else 100
            
            uncertainty_penalty = (
                quality_dist[1] * 0.1 +
                time_dist[1] / time_budget * 0.2 +
                energy_dist[1] / energy_budget * 0.1 +
                risk_dist[1] * 0.3
            )
            
            return mean_utility - uncertainty_penalty
        except Exception as e:
            logger.error(f"Expected utility computation failed: {e}")
            return 0.0
    
    def compute_marginal_utility(self,
                                component: str,
                                current_value: float,
                                delta: float,
                                other_values: Dict[str, float],
                                context: UtilityContext) -> float:
        """
        Compute marginal utility of changing one component
        """
        
        try:
            # Current utility
            current_utility = self.compute_utility(
                other_values.get('quality', 0.5),
                other_values.get('time', 1000),
                other_values.get('energy', 100),
                other_values.get('risk', 0.5),
                context
            )
            
            # Updated values
            updated_values = other_values.copy()
            updated_values[component] = current_value + delta
            
            # New utility
            new_utility = self.compute_utility(
                updated_values.get('quality', 0.5),
                updated_values.get('time', 1000),
                updated_values.get('energy', 100),
                updated_values.get('risk', 0.5),
                context
            )
            
            return (new_utility - current_utility) / delta if abs(delta) > 1e-10 else 0.0
        except Exception as e:
            logger.error(f"Marginal utility computation failed: {e}")
            return 0.0
    
    def optimize_weights(self, history: List[Dict[str, Any]], target_mode: ContextMode):
        """
        Optimize weights based on historical preferences
        """
        
        try:
            if not history:
                return
            
            # Extract features and outcomes
            features = []
            outcomes = []
            
            for entry in history:
                if 'components' in entry and 'user_satisfaction' in entry:
                    features.append([
                        entry['components']['quality_score'],
                        entry['components']['time_score'],
                        entry['components']['energy_score'],
                        entry['components']['risk_score']
                    ])
                    outcomes.append(entry['user_satisfaction'])
            
            if not features:
                return
            
            # Simple linear regression to learn weights
            X = np.array(features)
            y = np.array(outcomes)
            
            # Add small regularization
            XtX = X.T @ X + 0.01 * np.eye(X.shape[1])
            Xty = X.T @ y
            
            # Solve for weights
            learned = np.linalg.solve(XtX, Xty)
            
            # Store learned weights
            with self.weights_lock:
                self.learned_weights[target_mode] = {
                    'quality': max(0, float(learned[0])),
                    'time_penalty': max(0, float(learned[1])),
                    'energy_penalty': max(0, float(learned[2])),
                    'risk_penalty': max(0, float(learned[3]))
                }
        except Exception as e:
            logger.error(f"Weight optimization failed: {e}")
    
    def explain_utility(self, 
                       quality: float,
                       time: float,
                       energy: float,
                       risk: float,
                       context: UtilityContext) -> Dict[str, Any]:
        """
        Explain utility computation
        """
        
        try:
            weights = self.get_weights(context)
            components = self.compute_components(quality, time, energy, risk, context)
            utility = self.compute_utility(quality, time, energy, risk, context)
            
            # Calculate contributions
            contributions = {
                'quality': weights.quality * components.quality_score,
                'time': -weights.time_penalty * components.time_score,
                'energy': -weights.energy_penalty * components.energy_score,
                'risk': -weights.risk_penalty * components.risk_score
            }
            
            # Identify bottleneck
            bottleneck = min(contributions.items(), key=lambda x: x[1])[0]
            
            return {
                'total_utility': utility,
                'weights': weights.to_dict(),
                'components': components.to_dict(),
                'contributions': contributions,
                'bottleneck': bottleneck,
                'explanation': self._generate_explanation(
                    utility, weights, components, contributions, bottleneck
                )
            }
        except Exception as e:
            logger.error(f"Utility explanation failed: {e}")
            return {
                'total_utility': 0.0,
                'error': str(e)
            }
    
    def _generate_explanation(self, utility: float, weights: UtilityWeights,
                             components: UtilityComponents,
                             contributions: Dict[str, float],
                             bottleneck: str) -> str:
        """Generate human-readable explanation"""
        
        try:
            if utility > 0.7:
                overall = "High utility"
            elif utility > 0.4:
                overall = "Moderate utility"
            else:
                overall = "Low utility"
            
            explanation = f"{overall} (score: {utility:.2f}). "
            
            if bottleneck == 'quality':
                explanation += f"Limited by quality ({components.raw_quality:.2f}). Consider more accurate tools."
            elif bottleneck == 'time':
                explanation += f"Limited by time ({components.raw_time:.0f}ms). Consider faster tools."
            elif bottleneck == 'energy':
                explanation += f"Limited by energy ({components.raw_energy:.0f}mJ). Consider more efficient tools."
            else:
                explanation += f"Limited by risk ({components.raw_risk:.2f}). Consider safer tools."
            
            return explanation
        except Exception as e:
            logger.error(f"Explanation generation failed: {e}")
            return "Utility computation explanation unavailable"
    
    def _dict_to_context(self, d: Dict[str, Any]) -> UtilityContext:
        """Convert dictionary to UtilityContext"""
        
        try:
            mode_str = d.get('mode', 'balanced')
            if isinstance(mode_str, str):
                mode = ContextMode[mode_str.upper()] if mode_str.upper() in ContextMode.__members__ else ContextMode.BALANCED
            else:
                mode = mode_str
            
            return UtilityContext(
                mode=mode,
                time_budget=d.get('time_budget', 5000),
                energy_budget=d.get('energy_budget', 1000),
                min_quality=d.get('min_quality', 0.5),
                max_risk=d.get('max_risk', 0.5),
                user_preferences=d.get('user_preferences', {}),
                metadata=d.get('metadata', {})
            )
        except Exception as e:
            logger.error(f"Context conversion failed: {e}")
            return UtilityContext(
                mode=ContextMode.BALANCED,
                time_budget=5000,
                energy_budget=1000,
                min_quality=0.5,
                max_risk=0.5
            )
    
    def _compute_cache_key(self, quality: float, time: float, energy: float, 
                          risk: float, context: UtilityContext) -> str:
        """Compute cache key"""
        
        try:
            # Round values for caching
            key_parts = [
                round(quality, 2),
                round(time, 0),
                round(energy, 0),
                round(risk, 2),
                context.mode.value
            ]
            
            return "_".join(map(str, key_parts))
        except Exception as e:
            logger.error(f"Cache key computation failed: {e}")
            return f"key_{time_module.time()}"
    
    def _update_statistics(self, mode: ContextMode, utility: float, time_ms: float):
        """Update computation statistics"""
        
        try:
            with self.stats_lock:
                stats = self.computation_stats[mode]
                stats['count'] += 1
                
                # Exponential moving average
                alpha = 0.1
                stats['avg_utility'] = (1 - alpha) * stats['avg_utility'] + alpha * utility
                stats['avg_time_ms'] = (1 - alpha) * stats['avg_time_ms'] + alpha * time_ms
        except Exception as e:
            logger.error(f"Statistics update failed: {e}")
    
    def save_config(self, path: str):
        """Save utility model configuration"""
        
        try:
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with self.weights_lock:
                config = {
                    'mode_weights': {
                        mode.value: weights.to_dict()
                        for mode, weights in self.mode_weights.items()
                    },
                    'learned_weights': {
                        k.value if isinstance(k, ContextMode) else k: v
                        for k, v in self.learned_weights.items()
                    },
                    'normalization': self.normalization
                }
            
            with self.stats_lock:
                config['statistics'] = dict(self.computation_stats)
            
            with open(save_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Utility model configuration saved to {save_path}")
        except Exception as e:
            logger.error(f"Configuration save failed: {e}")
    
    def load_config(self, path: str):
        """Load utility model configuration"""
        
        try:
            load_path = Path(path)
            
            if not load_path.exists():
                logger.warning(f"Configuration file {load_path} not found")
                return
            
            with open(load_path, 'r') as f:
                config = json.load(f)
            
            # Load mode weights
            with self.weights_lock:
                for mode_str, weights_dict in config.get('mode_weights', {}).items():
                    try:
                        mode = ContextMode[mode_str.upper()]
                        self.mode_weights[mode] = UtilityWeights(**weights_dict)
                    except (KeyError, TypeError) as e:
                        logger.warning(f"Failed to load weights for mode {mode_str}: {e}")
                
                # Load learned weights
                self.learned_weights = {}
                for mode_str, weights_val in config.get('learned_weights', {}).items():
                    try:
                        mode = ContextMode[mode_str.upper()]
                        self.learned_weights[mode] = weights_val
                    except (KeyError, TypeError) as e:
                        logger.warning(f"Failed to load learned weights for mode {mode_str}: {e}")
            
            # Load normalization
            self.normalization.update(config.get('normalization', {}))
            
            logger.info(f"Utility model configuration loaded from {load_path}")
        except Exception as e:
            logger.error(f"Configuration load failed: {e}")
    
    def clear_cache(self):
        """Clear utility cache"""
        with self.cache_lock:
            self.cache.clear()
            logger.info("Utility cache cleared")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get computation statistics"""
        try:
            with self.stats_lock:
                with self.cache_lock:
                    return {
                        'computation_stats': dict(self.computation_stats),
                        'cache_size': len(self.cache),
                        'cache_ttl': self.cache_ttl
                    }
        except Exception as e:
            logger.error(f"Get statistics failed: {e}")
            return {}