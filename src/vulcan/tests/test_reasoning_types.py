"""
Comprehensive Test Suite for Reasoning Types

Tests all dataclasses with comprehensive validation, edge cases,
and error handling.
"""

import pytest
import time
from typing import Dict, Any

from vulcan.reasoning.reasoning_types import (
    ReasoningType,
    SelectionMode,
    PortfolioStrategy,
    UtilityContext,
    ModalityType,
    ReasoningStep,
    ReasoningChain,
    ReasoningResult,
    SelectionResult,
    PortfolioResult,
    CostEstimate,
    SafetyAssessment,
    CalibrationData,
    MonitoringData,
    ValueOfInformation,
    DistributionShift
)


# Enum Tests
class TestEnums:
    """Test all enum types"""
    
    def test_reasoning_type_enum(self):
        assert ReasoningType.DEDUCTIVE.value == "deductive"
        assert ReasoningType.INDUCTIVE.value == "inductive"
        assert ReasoningType.ABDUCTIVE.value == "abductive"
        assert ReasoningType.HIERARCHICAL.value == "hierarchical"
        assert len(ReasoningType) >= 14
    
    def test_selection_mode_enum(self):
        assert SelectionMode.FAST.value == "fast"
        assert SelectionMode.ACCURATE.value == "accurate"
        assert SelectionMode.SAFE.value == "safe"
    
    def test_portfolio_strategy_enum(self):
        assert PortfolioStrategy.SEQUENTIAL.value == "sequential"
        assert PortfolioStrategy.PARALLEL.value == "parallel"
        assert PortfolioStrategy.CASCADE.value == "cascade"
    
    def test_utility_context_enum(self):
        assert UtilityContext.RUSH.value == "rush"
        assert UtilityContext.BALANCED.value == "balanced"
    
    def test_modality_type_enum(self):
        assert ModalityType.TEXT.value == "text"
        assert ModalityType.VISION.value == "vision"
        assert ModalityType.AUDIO.value == "audio"


# ReasoningStep Tests
class TestReasoningStep:
    """Test ReasoningStep dataclass"""
    
    def test_valid_reasoning_step(self):
        step = ReasoningStep(
            step_id="step_001",
            step_type=ReasoningType.DEDUCTIVE,
            input_data={"premise": "A"},
            output_data={"conclusion": "B"},
            confidence=0.85,
            explanation="Deduced B from A"
        )
        
        assert step.step_id == "step_001"
        assert step.confidence == 0.85
        assert step.step_type == ReasoningType.DEDUCTIVE
    
    def test_reasoning_step_with_modality(self):
        step = ReasoningStep(
            step_id="step_001",
            step_type=ReasoningType.MULTIMODAL,
            input_data={},
            output_data={},
            confidence=0.9,
            explanation="Test",
            modality=ModalityType.TEXT
        )
        
        assert step.modality == ModalityType.TEXT
    
    def test_reasoning_step_invalid_confidence_type(self):
        with pytest.raises(TypeError, match="Confidence must be numeric"):
            ReasoningStep(
                step_id="step_001",
                step_type=ReasoningType.DEDUCTIVE,
                input_data={},
                output_data={},
                confidence="high",  # Invalid type
                explanation="Test"
            )
    
    def test_reasoning_step_confidence_out_of_range_low(self):
        with pytest.raises(ValueError, match="Confidence must be in"):
            ReasoningStep(
                step_id="step_001",
                step_type=ReasoningType.DEDUCTIVE,
                input_data={},
                output_data={},
                confidence=-0.1,  # Too low
                explanation="Test"
            )
    
    def test_reasoning_step_confidence_out_of_range_high(self):
        with pytest.raises(ValueError, match="Confidence must be in"):
            ReasoningStep(
                step_id="step_001",
                step_type=ReasoningType.DEDUCTIVE,
                input_data={},
                output_data={},
                confidence=1.5,  # Too high
                explanation="Test"
            )
    
    def test_reasoning_step_empty_step_id(self):
        with pytest.raises(ValueError, match="step_id must be a non-empty string"):
            ReasoningStep(
                step_id="",  # Empty
                step_type=ReasoningType.DEDUCTIVE,
                input_data={},
                output_data={},
                confidence=0.8,
                explanation="Test"
            )
    
    def test_reasoning_step_invalid_step_type(self):
        with pytest.raises(TypeError, match="step_type must be ReasoningType enum"):
            ReasoningStep(
                step_id="step_001",
                step_type="deductive",  # String instead of enum
                input_data={},
                output_data={},
                confidence=0.8,
                explanation="Test"
            )
    
    def test_reasoning_step_future_timestamp(self):
        with pytest.raises(ValueError, match="Invalid timestamp"):
            ReasoningStep(
                step_id="step_001",
                step_type=ReasoningType.DEDUCTIVE,
                input_data={},
                output_data={},
                confidence=0.8,
                explanation="Test",
                timestamp=time.time() + 100000  # Far future
            )


# ReasoningChain Tests
class TestReasoningChain:
    """Test ReasoningChain dataclass"""
    
    def test_valid_reasoning_chain(self):
        steps = [
            ReasoningStep(
                step_id="step_001",
                step_type=ReasoningType.DEDUCTIVE,
                input_data={},
                output_data={},
                confidence=0.9,
                explanation="Test"
            )
        ]
        
        chain = ReasoningChain(
            chain_id="chain_001",
            steps=steps,
            initial_query={"question": "test"},
            final_conclusion="result",
            total_confidence=0.9,
            reasoning_types_used={ReasoningType.DEDUCTIVE},
            modalities_involved=set(),
            safety_checks=[],
            audit_trail=[]
        )
        
        assert chain.chain_id == "chain_001"
        assert len(chain.steps) == 1
        assert chain.total_confidence == 0.9
    
    def test_reasoning_chain_empty_steps(self):
        with pytest.raises(ValueError, match="must have at least one step"):
            ReasoningChain(
                chain_id="chain_001",
                steps=[],  # Empty
                initial_query={},
                final_conclusion="result",
                total_confidence=0.9,
                reasoning_types_used=set(),
                modalities_involved=set(),
                safety_checks=[],
                audit_trail=[]
            )
    
    def test_reasoning_chain_invalid_step_type(self):
        with pytest.raises(TypeError, match="must be a ReasoningStep instance"):
            ReasoningChain(
                chain_id="chain_001",
                steps=[{"not": "a_step"}],  # Invalid type
                initial_query={},
                final_conclusion="result",
                total_confidence=0.9,
                reasoning_types_used=set(),
                modalities_involved=set(),
                safety_checks=[],
                audit_trail=[]
            )
    
    def test_reasoning_chain_invalid_confidence(self):
        steps = [
            ReasoningStep(
                step_id="step_001",
                step_type=ReasoningType.DEDUCTIVE,
                input_data={},
                output_data={},
                confidence=0.9,
                explanation="Test"
            )
        ]
        
        with pytest.raises(ValueError, match="total_confidence must be in"):
            ReasoningChain(
                chain_id="chain_001",
                steps=steps,
                initial_query={},
                final_conclusion="result",
                total_confidence=2.0,  # Too high
                reasoning_types_used=set(),
                modalities_involved=set(),
                safety_checks=[],
                audit_trail=[]
            )


# ReasoningResult Tests
class TestReasoningResult:
    """Test ReasoningResult dataclass"""
    
    def test_valid_reasoning_result(self):
        result = ReasoningResult(
            conclusion="Test conclusion",
            confidence=0.85,
            reasoning_type=ReasoningType.DEDUCTIVE,
            explanation="Test explanation"
        )
        
        assert result.conclusion == "Test conclusion"
        assert result.confidence == 0.85
        assert result.reasoning_type == ReasoningType.DEDUCTIVE
    
    def test_reasoning_result_with_chain(self):
        steps = [
            ReasoningStep(
                step_id="step_001",
                step_type=ReasoningType.DEDUCTIVE,
                input_data={},
                output_data={},
                confidence=0.9,
                explanation="Test"
            )
        ]
        
        chain = ReasoningChain(
            chain_id="chain_001",
            steps=steps,
            initial_query={},
            final_conclusion="result",
            total_confidence=0.9,
            reasoning_types_used={ReasoningType.DEDUCTIVE},
            modalities_involved=set(),
            safety_checks=[],
            audit_trail=[]
        )
        
        result = ReasoningResult(
            conclusion="Test",
            confidence=0.9,
            reasoning_type=ReasoningType.DEDUCTIVE,
            reasoning_chain=chain
        )
        
        assert result.reasoning_chain is not None
        assert result.reasoning_chain.chain_id == "chain_001"
    
    def test_reasoning_result_invalid_confidence(self):
        with pytest.raises(ValueError, match="confidence must be in"):
            ReasoningResult(
                conclusion="Test",
                confidence=1.5,  # Too high
                reasoning_type=ReasoningType.DEDUCTIVE
            )
    
    def test_reasoning_result_invalid_uncertainty(self):
        with pytest.raises(ValueError, match="uncertainty must be in"):
            ReasoningResult(
                conclusion="Test",
                confidence=0.8,
                reasoning_type=ReasoningType.DEDUCTIVE,
                uncertainty=-0.1  # Negative
            )


# SelectionResult Tests
class TestSelectionResult:
    """Test SelectionResult dataclass"""
    
    def test_valid_selection_result(self):
        result = SelectionResult(
            selected_tool="symbolic",
            execution_result={"answer": 42},
            confidence=0.9,
            calibrated_confidence=0.85,
            execution_time_ms=100.0,
            energy_used_mj=50.0,
            strategy_used=PortfolioStrategy.SEQUENTIAL,
            all_results={"symbolic": {"answer": 42}}
        )
        
        assert result.selected_tool == "symbolic"
        assert result.confidence == 0.9
        assert result.execution_time_ms == 100.0
    
    def test_selection_result_negative_time(self):
        with pytest.raises(ValueError, match="execution_time_ms must be non-negative"):
            SelectionResult(
                selected_tool="symbolic",
                execution_result={},
                confidence=0.9,
                calibrated_confidence=0.85,
                execution_time_ms=-10.0,  # Negative
                energy_used_mj=50.0,
                strategy_used=PortfolioStrategy.SEQUENTIAL,
                all_results={}
            )
    
    def test_selection_result_negative_energy(self):
        with pytest.raises(ValueError, match="energy_used_mj must be non-negative"):
            SelectionResult(
                selected_tool="symbolic",
                execution_result={},
                confidence=0.9,
                calibrated_confidence=0.85,
                execution_time_ms=100.0,
                energy_used_mj=-50.0,  # Negative
                strategy_used=PortfolioStrategy.SEQUENTIAL,
                all_results={}
            )


# PortfolioResult Tests
class TestPortfolioResult:
    """Test PortfolioResult dataclass"""
    
    def test_valid_portfolio_result(self):
        result = PortfolioResult(
            primary_result={"answer": 42},
            all_results={"tool1": {"answer": 42}, "tool2": {"answer": 43}},
            strategy=PortfolioStrategy.PARALLEL,
            tools_used=["tool1", "tool2"],
            execution_time_ms=150.0,
            energy_used=75.0,
            confidence_scores={"tool1": 0.9, "tool2": 0.85},
            consensus_achieved=True
        )
        
        assert result.strategy == PortfolioStrategy.PARALLEL
        assert len(result.tools_used) == 2
        assert result.consensus_achieved is True
    
    def test_portfolio_result_invalid_confidence_score(self):
        with pytest.raises(ValueError, match="Confidence score .* must be in"):
            PortfolioResult(
                primary_result={},
                all_results={},
                strategy=PortfolioStrategy.PARALLEL,
                tools_used=["tool1"],
                execution_time_ms=100.0,
                energy_used=50.0,
                confidence_scores={"tool1": 1.5},  # Too high
                consensus_achieved=False
            )


# CostEstimate Tests
class TestCostEstimate:
    """Test CostEstimate dataclass"""
    
    def test_valid_cost_estimate(self):
        estimate = CostEstimate(
            time_ms=100.0,
            energy_mj=50.0,
            memory_mb=256.0,
            confidence_interval=(90.0, 110.0)
        )
        
        assert estimate.time_ms == 100.0
        assert estimate.confidence_interval == (90.0, 110.0)
    
    def test_cost_estimate_negative_time(self):
        with pytest.raises(ValueError, match="time_ms must be non-negative"):
            CostEstimate(
                time_ms=-10.0,  # Negative
                energy_mj=50.0,
                memory_mb=256.0,
                confidence_interval=(0.0, 100.0)
            )
    
    def test_cost_estimate_invalid_interval_order(self):
        with pytest.raises(ValueError, match="lower bound .* must be <= upper bound"):
            CostEstimate(
                time_ms=100.0,
                energy_mj=50.0,
                memory_mb=256.0,
                confidence_interval=(110.0, 90.0)  # Reversed
            )
    
    def test_cost_estimate_negative_percentile(self):
        with pytest.raises(ValueError, match="Percentile value .* must be non-negative"):
            CostEstimate(
                time_ms=100.0,
                energy_mj=50.0,
                memory_mb=256.0,
                confidence_interval=(90.0, 110.0),
                percentiles={"p50": -10.0}  # Negative
            )


# SafetyAssessment Tests
class TestSafetyAssessment:
    """Test SafetyAssessment dataclass"""
    
    def test_valid_safety_assessment(self):
        assessment = SafetyAssessment(
            is_safe=True,
            safety_level="LOW",
            violations=[],
            mitigations=[],
            confidence=0.95
        )
        
        assert assessment.is_safe is True
        assert assessment.safety_level == "LOW"
        assert assessment.confidence == 0.95
    
    def test_safety_assessment_invalid_level(self):
        with pytest.raises(ValueError, match="safety_level must be one of"):
            SafetyAssessment(
                is_safe=False,
                safety_level="INVALID",  # Invalid level
                violations=["violation1"],
                mitigations=[],
                confidence=0.8
            )
    
    def test_safety_assessment_valid_levels(self):
        valid_levels = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'MINIMAL', 'UNKNOWN']
        
        for level in valid_levels:
            assessment = SafetyAssessment(
                is_safe=True,
                safety_level=level,
                violations=[],
                mitigations=[],
                confidence=0.9
            )
            assert assessment.safety_level == level


# CalibrationData Tests
class TestCalibrationData:
    """Test CalibrationData dataclass"""
    
    def test_valid_calibration_data(self):
        data = CalibrationData(
            raw_confidence=0.8,
            calibrated_confidence=0.75,
            actual_outcome=True,
            tool_name="symbolic"
        )
        
        assert data.raw_confidence == 0.8
        assert data.calibrated_confidence == 0.75
        assert data.actual_outcome is True
    
    def test_calibration_data_with_features(self):
        data = CalibrationData(
            raw_confidence=0.8,
            calibrated_confidence=0.75,
            actual_outcome=True,
            tool_name="symbolic",
            features=[1.0, 2.0, 3.0]
        )
        
        assert len(data.features) == 3
    
    def test_calibration_data_invalid_feature_type(self):
        with pytest.raises(TypeError, match="Feature .* must be numeric"):
            CalibrationData(
                raw_confidence=0.8,
                calibrated_confidence=0.75,
                actual_outcome=True,
                tool_name="symbolic",
                features=[1.0, "invalid", 3.0]  # Invalid type
            )


# MonitoringData Tests
class TestMonitoringData:
    """Test MonitoringData dataclass"""
    
    def test_valid_monitoring_data(self):
        data = MonitoringData(
            tool_name="symbolic",
            latency_ms=50.0,
            throughput=100.0,
            error_rate=0.01,
            resource_usage={"cpu": 0.5, "memory": 0.3},
            health_score=0.95
        )
        
        assert data.tool_name == "symbolic"
        assert data.latency_ms == 50.0
        assert data.health_score == 0.95
    
    def test_monitoring_data_negative_latency(self):
        with pytest.raises(ValueError, match="latency_ms must be non-negative"):
            MonitoringData(
                tool_name="symbolic",
                latency_ms=-10.0,  # Negative
                throughput=100.0,
                error_rate=0.01,
                resource_usage={},
                health_score=0.9
            )
    
    def test_monitoring_data_invalid_error_rate(self):
        with pytest.raises(ValueError, match="error_rate must be in"):
            MonitoringData(
                tool_name="symbolic",
                latency_ms=50.0,
                throughput=100.0,
                error_rate=1.5,  # > 1.0
                resource_usage={},
                health_score=0.9
            )
    
    def test_monitoring_data_negative_resource_usage(self):
        with pytest.raises(ValueError, match="Resource usage .* must be non-negative"):
            MonitoringData(
                tool_name="symbolic",
                latency_ms=50.0,
                throughput=100.0,
                error_rate=0.01,
                resource_usage={"cpu": -0.5},  # Negative
                health_score=0.9
            )


# ValueOfInformation Tests
class TestValueOfInformation:
    """Test ValueOfInformation dataclass"""
    
    def test_valid_voi(self):
        voi = ValueOfInformation(
            expected_value=10.0,
            information_gain=5.0,
            cost=2.0,
            net_value=8.0,
            recommendation="acquire",
            source="tool1",
            confidence=0.9
        )
        
        assert voi.expected_value == 10.0
        assert voi.net_value == 8.0
        assert voi.recommendation == "acquire"
    
    def test_voi_negative_cost(self):
        with pytest.raises(ValueError, match="cost must be non-negative"):
            ValueOfInformation(
                expected_value=10.0,
                information_gain=5.0,
                cost=-2.0,  # Negative
                net_value=8.0,
                recommendation="acquire",
                source="tool1",
                confidence=0.9
            )
    
    def test_voi_empty_recommendation(self):
        with pytest.raises(ValueError, match="recommendation must be a non-empty string"):
            ValueOfInformation(
                expected_value=10.0,
                information_gain=5.0,
                cost=2.0,
                net_value=8.0,
                recommendation="",  # Empty
                source="tool1",
                confidence=0.9
            )


# DistributionShift Tests
class TestDistributionShift:
    """Test DistributionShift dataclass"""
    
    def test_valid_distribution_shift(self):
        shift = DistributionShift(
            drift_detected=True,
            drift_type="SUDDEN",
            severity="HIGH",
            affected_features=[0, 1, 5],
            confidence=0.85
        )
        
        assert shift.drift_detected is True
        assert shift.drift_type == "SUDDEN"
        assert len(shift.affected_features) == 3
    
    def test_distribution_shift_invalid_drift_type(self):
        with pytest.raises(ValueError, match="drift_type must be one of"):
            DistributionShift(
                drift_detected=True,
                drift_type="INVALID",  # Invalid type
                severity="HIGH",
                affected_features=[],
                confidence=0.8
            )
    
    def test_distribution_shift_invalid_severity(self):
        with pytest.raises(ValueError, match="severity must be one of"):
            DistributionShift(
                drift_detected=True,
                drift_type="SUDDEN",
                severity="INVALID",  # Invalid severity
                affected_features=[],
                confidence=0.8
            )
    
    def test_distribution_shift_negative_feature_index(self):
        with pytest.raises(ValueError, match="affected_features.* must be non-negative"):
            DistributionShift(
                drift_detected=True,
                drift_type="SUDDEN",
                severity="HIGH",
                affected_features=[0, -1, 2],  # Negative index
                confidence=0.8
            )
    
    def test_distribution_shift_valid_types_and_severities(self):
        valid_types = ['SUDDEN', 'GRADUAL', 'INCREMENTAL', 'RECURRING', 'NONE']
        valid_severities = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'MINIMAL']
        
        for drift_type in valid_types:
            for severity in valid_severities:
                shift = DistributionShift(
                    drift_detected=True,
                    drift_type=drift_type,
                    severity=severity,
                    affected_features=[],
                    confidence=0.8
                )
                assert shift.drift_type == drift_type
                assert shift.severity == severity


# Integration Tests
class TestIntegration:
    """Integration tests across types"""
    
    def test_complete_reasoning_workflow(self):
        # Create steps
        step1 = ReasoningStep(
            step_id="step_001",
            step_type=ReasoningType.DEDUCTIVE,
            input_data={"premise": "A"},
            output_data={"conclusion": "B"},
            confidence=0.9,
            explanation="Deduced B from A"
        )
        
        step2 = ReasoningStep(
            step_id="step_002",
            step_type=ReasoningType.PROBABILISTIC,
            input_data={"evidence": "observations"},
            output_data={"probability": 0.85},
            confidence=0.85,
            explanation="Probabilistic inference"
        )
        
        # Create chain
        chain = ReasoningChain(
            chain_id="chain_001",
            steps=[step1, step2],
            initial_query={"question": "test"},
            final_conclusion="result",
            total_confidence=0.875,
            reasoning_types_used={ReasoningType.DEDUCTIVE, ReasoningType.PROBABILISTIC},
            modalities_involved=set(),
            safety_checks=[],
            audit_trail=[]
        )
        
        # Create result
        result = ReasoningResult(
            conclusion="Final conclusion",
            confidence=0.875,
            reasoning_type=ReasoningType.HYBRID,
            reasoning_chain=chain,
            explanation="Combined reasoning"
        )
        
        assert result.reasoning_chain.chain_id == "chain_001"
        assert len(result.reasoning_chain.steps) == 2
    
    def test_selection_with_monitoring(self):
        # Selection result
        selection = SelectionResult(
            selected_tool="symbolic",
            execution_result={"answer": 42},
            confidence=0.9,
            calibrated_confidence=0.85,
            execution_time_ms=100.0,
            energy_used_mj=50.0,
            strategy_used=PortfolioStrategy.SEQUENTIAL,
            all_results={"symbolic": {"answer": 42}}
        )
        
        # Monitoring data
        monitoring = MonitoringData(
            tool_name="symbolic",
            latency_ms=100.0,
            throughput=10.0,
            error_rate=0.0,
            resource_usage={"cpu": 0.3},
            health_score=0.95
        )
        
        assert selection.selected_tool == monitoring.tool_name
        assert selection.execution_time_ms == monitoring.latency_ms


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])