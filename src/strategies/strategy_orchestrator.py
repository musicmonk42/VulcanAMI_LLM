"""
Strategy Orchestrator - Wires all strategy components together

This is the main entry point for the strategies module, providing intelligent
tool selection through:
- Multi-tier feature extraction with VOI-gated progression
- Distribution drift detection and alerting
- Stochastic cost prediction with uncertainty quantification
- Tool health monitoring and degradation handling
INTEGRATED: Schema validation for cost configurations and queries
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.utils.performance_metrics import PerformanceTimer

logger = logging.getLogger(__name__)

# Schema Registry for validation
SchemaRegistry = None
_SCHEMA_REGISTRY_AVAILABLE = False
try:
    # Try relative import first (when installed as package)
    from ..vulcan.schema_registry import SchemaRegistry
    _SCHEMA_REGISTRY_AVAILABLE = True
except (ImportError, ValueError):
    # Fall back to absolute import for development/testing
    try:
        from vulcan.schema_registry import SchemaRegistry
        _SCHEMA_REGISTRY_AVAILABLE = True
    except ImportError as e:
        logger.debug(f"SchemaRegistry not available: {e}")
        SchemaRegistry = None

# Try importing strategy components with fallbacks
try:
    from .cost_model import StochasticCostModel, CostObservation, CostComponent
    COST_MODEL_AVAILABLE = True
except ImportError as e:
    logger.warning(f"StochasticCostModel not available: {e}")
    COST_MODEL_AVAILABLE = False
    StochasticCostModel = None
    CostObservation = None
    CostComponent = None

try:
    from .distribution_monitor import DistributionMonitor, DriftSeverity
    DISTRIBUTION_MONITOR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"DistributionMonitor not available: {e}")
    DISTRIBUTION_MONITOR_AVAILABLE = False
    DistributionMonitor = None
    DriftSeverity = None

try:
    from .feature_extraction import MultiTierFeatureExtractor, FeatureTier, ExtractionResult
    FEATURE_EXTRACTOR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"MultiTierFeatureExtractor not available: {e}")
    FEATURE_EXTRACTOR_AVAILABLE = False
    MultiTierFeatureExtractor = None
    FeatureTier = None
    ExtractionResult = None

try:
    from .tool_monitor import ToolMonitor, HealthStatus
    TOOL_MONITOR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ToolMonitor not available: {e}")
    TOOL_MONITOR_AVAILABLE = False
    ToolMonitor = None
    HealthStatus = None

try:
    from .value_of_information import ValueOfInformationGate, DecisionState, VOIAction
    VOI_GATE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ValueOfInformationGate not available: {e}")
    VOI_GATE_AVAILABLE = False
    ValueOfInformationGate = None
    DecisionState = None
    VOIAction = None


@dataclass
class StrategyDecision:
    """Result of strategy analysis"""
    recommended_tool: str
    confidence: float
    estimated_cost_ms: float
    drift_detected: bool
    drift_severity: Optional[str] = None
    feature_tier_used: int = 1
    voi_decision: str = "proceed"
    health_status: str = "healthy"
    metadata: Dict[str, Any] = field(default_factory=dict)


class StrategyOrchestrator:
    """
    Orchestrates all strategy components for intelligent tool selection.
    
    This is the main entry point for the strategies module.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        # Initialize schema registry if available
        self.schema_registry = None
        self.validate_configs = config.get('validate_configs', True)
        if _SCHEMA_REGISTRY_AVAILABLE and SchemaRegistry and self.validate_configs:
            try:
                self.schema_registry = SchemaRegistry.get_instance()
                logger.info("[StrategyOrchestrator] SchemaRegistry integrated")
            except Exception as e:
                logger.warning(f"[StrategyOrchestrator] Failed to init SchemaRegistry: {e}")
                self.schema_registry = None
                self.validate_configs = False
        
        # Add CPU awareness for monitoring
        self._cpu_caps = None
        try:
            from src.utils.cpu_capabilities import get_cpu_capabilities
            self._cpu_caps = get_cpu_capabilities()
            logger.info(
                f"[StrategyOrchestrator] CPU: {self._cpu_caps.get_best_vector_instruction_set()} "
                f"({self._cpu_caps.get_performance_tier()})"
            )
        except Exception as e:
            logger.debug(f"[StrategyOrchestrator] CPU detection unavailable: {e}")
        
        # Initialize all strategy components with fallbacks
        self.cost_model = None
        self.distribution_monitor = None
        self.feature_extractor = None
        self.tool_monitor = None
        self.voi_gate = None
        
        if COST_MODEL_AVAILABLE and StochasticCostModel:
            try:
                self.cost_model = StochasticCostModel(config.get('cost_model', {}))
                logger.info("[StrategyOrchestrator] StochasticCostModel initialized")
            except Exception as e:
                logger.warning(f"[StrategyOrchestrator] Failed to init StochasticCostModel: {e}")
        
        if DISTRIBUTION_MONITOR_AVAILABLE and DistributionMonitor:
            try:
                self.distribution_monitor = DistributionMonitor(config.get('drift', {}))
                logger.info("[StrategyOrchestrator] DistributionMonitor initialized")
            except Exception as e:
                logger.warning(f"[StrategyOrchestrator] Failed to init DistributionMonitor: {e}")
        
        if FEATURE_EXTRACTOR_AVAILABLE and MultiTierFeatureExtractor:
            try:
                # MultiTierFeatureExtractor takes no arguments
                self.feature_extractor = MultiTierFeatureExtractor()
                logger.info("[StrategyOrchestrator] MultiTierFeatureExtractor initialized")
            except Exception as e:
                logger.warning(f"[StrategyOrchestrator] Failed to init MultiTierFeatureExtractor: {e}")
        
        if TOOL_MONITOR_AVAILABLE and ToolMonitor:
            try:
                self.tool_monitor = ToolMonitor(config.get('monitoring', {}))
                logger.info("[StrategyOrchestrator] ToolMonitor initialized")
            except Exception as e:
                logger.warning(f"[StrategyOrchestrator] Failed to init ToolMonitor: {e}")
        
        if VOI_GATE_AVAILABLE and ValueOfInformationGate:
            try:
                self.voi_gate = ValueOfInformationGate(config.get('voi', {}))
                logger.info("[StrategyOrchestrator] ValueOfInformationGate initialized")
            except Exception as e:
                logger.warning(f"[StrategyOrchestrator] Failed to init ValueOfInformationGate: {e}")
        
        # Available tools
        self.available_tools = config.get('tools', [
            'symbolic', 'probabilistic', 'causal', 'neural', 'hybrid'
        ])
        
        # Statistics
        self.total_decisions = 0
        self.total_drift_detections = 0
        self.total_voi_gathers = 0
        
        # Component availability summary
        components = []
        if self.cost_model:
            components.append("cost_model")
        if self.distribution_monitor:
            components.append("drift_monitor")
        if self.feature_extractor:
            components.append("feature_extractor")
        if self.tool_monitor:
            components.append("tool_monitor")
        if self.voi_gate:
            components.append("voi_gate")
        
        logger.info(f"[StrategyOrchestrator] Initialized with {len(components)}/5 components: {components}")
    
    # Default values for decision making (can be configured)
    DEFAULT_INITIAL_UNCERTAINTY = 0.5
    DEFAULT_INITIAL_CONFIDENCE = 0.5
    
    def analyze(
        self,
        query: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> StrategyDecision:
        """
        Analyze a query and recommend the best tool.
        
        This is the main entry point for tool selection with comprehensive 
        performance tracking throughout the pipeline.
        Includes schema validation for queries and cost configurations.
        Note: This is a synchronous method for simpler integration.
        """
        with PerformanceTimer("strategy_analysis", "full_pipeline"):
            start_time = time.time()
            context = context or {}
            features = None
            tier_used = 1
            drift_detected = False
            drift_summary = {}
            voi_decision = 'proceed'
            extraction_time_ms = 0.0
            validation_status = {"query_valid": True, "configs_valid": True}
            
            # Schema validation for query (if it's a dict with expected structure)
            if self.validate_configs and self.schema_registry:
                try:
                    if isinstance(query, dict) and 'query' in query:
                        # Validate as reasoning_query
                        query_validation = self.schema_registry.validate(query, "reasoning_query")
                        validation_status["query_valid"] = query_validation.valid
                        if not query_validation.valid:
                            logger.warning(
                                f"Query schema validation failed: {len(query_validation.errors)} error(s)"
                            )
                except Exception as e:
                    logger.debug(f"Query validation error: {e}")
            
            # Step 1: Extract features (start with Tier 1)
            with PerformanceTimer("strategy_analysis", "feature_extraction"):
                if self.feature_extractor:
                    try:
                        extract_start = time.time()
                        features = self.feature_extractor.extract_tier1(query)
                        extraction_time_ms = (time.time() - extract_start) * 1000
                        tier_used = 1
                        logger.info(
                            f"[Strategy] Extracted {len(features) if features is not None else 0} "
                            f"Tier-1 features in {extraction_time_ms:.1f}ms"
                        )
                    except Exception as e:
                        logger.warning(f"[Strategy] Feature extraction failed: {e}")
                        features = np.zeros(15)  # Fallback
                else:
                    # Create dummy features if extractor not available
                    features = np.zeros(15)
            
            # Step 2: Check for distribution drift
            with PerformanceTimer("strategy_analysis", "drift_detection"):
                if self.distribution_monitor and features is not None:
                    try:
                        drift_detected = self.distribution_monitor.detect_shift(features.reshape(1, -1))
                        drift_summary = self.distribution_monitor.get_drift_summary()
                        
                        if drift_detected:
                            self.total_drift_detections += 1
                            logger.warning(f"[Strategy] Distribution drift detected! Summary: {drift_summary}")
                    except Exception as e:
                        logger.warning(f"[Strategy] Drift detection failed: {e}")
            
            # Step 3: VOI decision - should we gather more features?
            if self.voi_gate and DecisionState and VOIAction and features is not None:
                try:
                    decision_state = DecisionState(
                        features=features,
                        uncertainty=self.DEFAULT_INITIAL_UNCERTAINTY,
                        current_confidence=self.DEFAULT_INITIAL_CONFIDENCE,
                        gathered_information=[],
                        remaining_budget={'time_ms': context.get('budget_ms', 1000)}
                    )
                    
                    # FIX: Pass extracted uncertainty and confidence values instead of DecisionState
                    # The should_gather_more method expects numeric values, not a DecisionState object
                    # It returns a boolean: True if more info should be gathered
                    # Use getattr with defaults to handle missing attributes gracefully
                    uncertainty_val = getattr(decision_state, 'uncertainty', self.DEFAULT_INITIAL_UNCERTAINTY)
                    confidence_val = getattr(decision_state, 'current_confidence', self.DEFAULT_INITIAL_CONFIDENCE)
                    
                    should_gather = self.voi_gate.should_gather_more(
                        uncertainty=uncertainty_val if uncertainty_val is not None else self.DEFAULT_INITIAL_UNCERTAINTY,
                        confidence=confidence_val if confidence_val is not None else self.DEFAULT_INITIAL_CONFIDENCE,
                        query_id=context.get('query_id')
                    )
                    
                    if should_gather:
                        voi_decision = 'gather_more'
                        self.total_voi_gathers += 1
                        
                        # Extract higher-tier features since VOI recommends gathering more
                        if self.feature_extractor:
                            try:
                                extract_start = time.time()
                                # Default to tier 2 for "gather_more" decision
                                features = self.feature_extractor.extract_tier2(query)
                                tier_used = 2
                                
                                extraction_time_ms = (time.time() - extract_start) * 1000
                                logger.info(
                                    f"[Strategy] VOI: Upgraded to Tier-{tier_used} features "
                                    f"({extraction_time_ms:.1f}ms)"
                                )
                            except Exception as e:
                                logger.warning(f"[Strategy] Higher-tier extraction failed: {e}")
                except Exception as e:
                    logger.warning(f"[Strategy] VOI analysis failed: {e}")
            
            # Step 4: Predict costs for each tool
            with PerformanceTimer("strategy_analysis", "cost_prediction"):
                tool_costs = {}
                tool_predictions = {}
                
                if self.cost_model and CostComponent and features is not None:
                    try:
                        for tool in self.available_tools:
                            cost_pred = self.cost_model.predict_cost(tool, features)
                            tool_costs[tool] = cost_pred
                            tool_predictions[tool] = cost_pred.get(
                                CostComponent.TIME_MS.value, {}
                            ).get('mean', float('inf'))
            except Exception as e:
                logger.warning(f"[Strategy] Cost prediction failed: {e}")
                # Fallback to default predictions
                for tool in self.available_tools:
                    tool_predictions[tool] = 1000.0  # 1 second default
        else:
            # Fallback predictions
            for tool in self.available_tools:
                tool_predictions[tool] = 1000.0
        
        # Step 5: Get tool health status
        health_status = "healthy"
        if self.tool_monitor and HealthStatus:
            try:
                health_status = self.tool_monitor.get_health_status().value
            except Exception as e:
                logger.warning(f"[Strategy] Health check failed: {e}")
        
        # Step 6: Select best tool (lowest cost + health consideration)
        best_tool = self.available_tools[0] if self.available_tools else "symbolic"
        best_score = float('inf')
        
        for tool, cost in tool_predictions.items():
            # Get tool health score
            tool_health_score = 1.0
            if self.tool_monitor:
                try:
                    tool_metrics = self.tool_monitor.tool_metrics.get(tool)
                    if tool_metrics:
                        tool_health_score = tool_metrics.health_score
                except Exception:
                    pass
            
            # Penalize unhealthy tools
            adjusted_cost = cost / max(0.1, tool_health_score)
            
            if adjusted_cost < best_score:
                best_score = adjusted_cost
                best_tool = tool
        
        # Calculate confidence based on cost variance
        confidence = 0.5
        if best_tool in tool_costs and CostComponent:
            try:
                cost_info = tool_costs[best_tool].get(CostComponent.TIME_MS.value, {})
                mean_cost = cost_info.get('mean', 100)
                std_cost = cost_info.get('std', 10)
                # Confidence inversely related to coefficient of variation
                cv = std_cost / max(1, mean_cost)
                confidence = max(0.1, min(0.99, 1 - cv))
            except Exception:
                pass
        
        # Update statistics
        self.total_decisions += 1
        
        # Record for monitoring
        if self.distribution_monitor and features is not None:
            try:
                self.distribution_monitor.update(features.reshape(1, -1))
            except Exception as e:
                logger.debug(f"[Strategy] Distribution update failed: {e}")
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        logger.info(
            f"[Strategy] Decision: tool={best_tool}, confidence={confidence:.2f}, "
            f"drift={drift_detected}, tier={tier_used}, elapsed={elapsed_ms:.1f}ms"
        )
        
        return StrategyDecision(
            recommended_tool=best_tool,
            confidence=confidence,
            estimated_cost_ms=tool_predictions.get(best_tool, 0),
            drift_detected=drift_detected,
            drift_severity=drift_summary.get('max_severity') if drift_detected else None,
            feature_tier_used=tier_used,
            voi_decision=voi_decision,
            health_status=health_status,
            metadata={
                'all_costs': {t: tool_predictions.get(t, 0) for t in self.available_tools},
                'analysis_time_ms': elapsed_ms,
                'drift_summary': drift_summary if drift_detected else None,
                'validation_status': validation_status,
            }
        )
    
    def record_execution(
        self,
        tool_name: str,
        success: bool,
        latency_ms: float,
        features: Optional[np.ndarray] = None,
        confidence: float = 1.0,
        energy_mj: float = 0.0
    ):
        """
        Record tool execution for model updates.
        Call this after a tool completes execution.
        """
        # Update tool monitor
        if self.tool_monitor:
            try:
                self.tool_monitor.record_execution(
                    tool_name=tool_name,
                    success=success,
                    latency_ms=latency_ms,
                    confidence=confidence,
                    energy_mj=energy_mj
                )
            except Exception as e:
                logger.debug(f"[Strategy] Tool monitor update failed: {e}")
        
        # Update cost model with observation
        if self.cost_model and CostObservation and CostComponent and features is not None:
            try:
                observation = CostObservation(
                    tool_name=tool_name,
                    component=CostComponent.TIME_MS,
                    value=latency_ms,
                    features=features,
                    complexity=0.5,
                    cold_start=False,
                    health_score=1.0 if success else 0.5
                )
                self.cost_model.update(observation)
            except Exception as e:
                logger.debug(f"[Strategy] Cost model update failed: {e}")
        
        logger.debug(
            f"[Strategy] Recorded execution: {tool_name} success={success} latency={latency_ms:.1f}ms"
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive strategy statistics"""
        stats = {
            'orchestrator': {
                'total_decisions': self.total_decisions,
                'total_drift_detections': self.total_drift_detections,
                'total_voi_gathers': self.total_voi_gathers,
                'drift_rate': self.total_drift_detections / max(1, self.total_decisions),
                'voi_gather_rate': self.total_voi_gathers / max(1, self.total_decisions),
            },
            'components_available': {
                'cost_model': self.cost_model is not None,
                'distribution_monitor': self.distribution_monitor is not None,
                'feature_extractor': self.feature_extractor is not None,
                'tool_monitor': self.tool_monitor is not None,
                'voi_gate': self.voi_gate is not None,
            }
        }
        
        if self.cost_model:
            try:
                stats['cost_model'] = self.cost_model.get_statistics()
            except Exception:
                pass
        
        if self.distribution_monitor:
            try:
                stats['distribution_monitor'] = self.distribution_monitor.get_statistics()
            except Exception:
                pass
        
        if self.tool_monitor:
            try:
                stats['tool_monitor'] = self.tool_monitor.get_statistics()
            except Exception:
                pass
        
        if self.voi_gate:
            try:
                stats['voi_gate'] = self.voi_gate.get_statistics()
            except Exception:
                pass
        
        return stats
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get tool health status"""
        result = {
            'overall_status': 'healthy',
            'tools': {}
        }
        
        if self.tool_monitor:
            try:
                result['overall_status'] = self.tool_monitor.get_health_status().value
                for name, metrics in self.tool_monitor.tool_metrics.items():
                    result['tools'][name] = metrics.to_dict() if hasattr(metrics, 'to_dict') else str(metrics)
            except Exception as e:
                logger.debug(f"[Strategy] Get health status failed: {e}")
        
        return result
    
    def get_drift_status(self) -> Dict[str, Any]:
        """Get distribution drift status"""
        if self.distribution_monitor:
            try:
                return self.distribution_monitor.get_drift_summary()
            except Exception as e:
                logger.debug(f"[Strategy] Get drift status failed: {e}")
        return {'status': 'monitoring_unavailable'}
    
    def reset_cost_predictions(self, tool_name: Optional[str] = None):
        """
        Reset the learned cost predictions to allow re-learning.
        
        This is the "Cost Hallucination" fix - when the tool_monitor has learned
        incorrect latency estimates (e.g., 259ms when reality is 54,000ms), calling
        this method will clear the learned predictions and allow the router to
        re-evaluate tools as "Unknown" and re-learn the "new normal".
        
        Args:
            tool_name: If provided, reset only predictions for this tool.
                      If None, reset all tool predictions to defaults.
        """
        if self.tool_monitor:
            try:
                self.tool_monitor.reset_cost_predictions(tool_name)
                logger.info(
                    f"[StrategyOrchestrator] Cost predictions reset "
                    f"({'all tools' if tool_name is None else tool_name})"
                )
            except Exception as e:
                logger.warning(f"[Strategy] Failed to reset cost predictions: {e}")
        else:
            logger.warning("[Strategy] ToolMonitor not available for cost prediction reset")
    
    def save_state(self, path: str):
        """Save all strategy component states"""
        import os
        os.makedirs(path, exist_ok=True)
        
        if self.cost_model:
            try:
                self.cost_model.save_model(f"{path}/cost_model")
            except Exception as e:
                logger.warning(f"[Strategy] Failed to save cost model: {e}")
        
        if self.distribution_monitor:
            try:
                self.distribution_monitor.save_state(f"{path}/distribution")
            except Exception as e:
                logger.warning(f"[Strategy] Failed to save distribution state: {e}")
        
        if self.voi_gate:
            try:
                self.voi_gate.save_state(f"{path}/voi")
            except Exception as e:
                logger.warning(f"[Strategy] Failed to save VOI state: {e}")
        
        if self.tool_monitor:
            try:
                self.tool_monitor.export_metrics(f"{path}/tool_metrics.json")
            except Exception as e:
                logger.warning(f"[Strategy] Failed to save tool metrics: {e}")
        
        logger.info(f"[StrategyOrchestrator] State saved to {path}")
    
    def load_state(self, path: str):
        """Load all strategy component states"""
        if self.cost_model:
            try:
                self.cost_model.load_model(f"{path}/cost_model")
            except Exception as e:
                logger.warning(f"[Strategy] Failed to load cost model: {e}")
        
        if self.distribution_monitor:
            try:
                self.distribution_monitor.load_state(f"{path}/distribution")
            except Exception as e:
                logger.warning(f"[Strategy] Failed to load distribution state: {e}")
        
        if self.voi_gate:
            try:
                self.voi_gate.load_state(f"{path}/voi")
            except Exception as e:
                logger.warning(f"[Strategy] Failed to load VOI state: {e}")
        
        logger.info(f"[StrategyOrchestrator] State loaded from {path}")
