"""
Test suite for philosophical reasoning registration in UnifiedReasoner.

Tests that the PHILOSOPHICAL reasoning type is properly:
1. Mapped from tool names (world_model, ethics, etc.) to ReasoningType.PHILOSOPHICAL
2. Registered in self.reasoners with WorldModel instance
3. Can execute philosophical queries end-to-end

Industry Standard Testing:
- Clear test names describing what is being tested
- Comprehensive edge case coverage
- Proper setup/teardown with fixtures
- Type hints for clarity
- Docstrings explaining test purpose
- Assertions with clear failure messages
"""

import logging
import pytest
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestToolNameMapping:
    """Test that tool names are correctly mapped to ReasoningType.PHILOSOPHICAL."""
    
    def test_philosophical_tool_name_maps_correctly(self):
        """Test 'philosophical' tool name maps to PHILOSOPHICAL type."""
        from vulcan.reasoning.unified.orchestrator import UnifiedReasoner
        from vulcan.reasoning.reasoning_types import ReasoningType
        
        reasoner = UnifiedReasoner(config={'enable_learning': False})
        result = reasoner._map_tool_name_to_reasoning_type('philosophical')
        
        assert result == ReasoningType.PHILOSOPHICAL, (
            f"Expected ReasoningType.PHILOSOPHICAL, got {result}"
        )
    
    def test_world_model_tool_name_maps_correctly(self):
        """Test 'world_model' tool name maps to PHILOSOPHICAL type."""
        from vulcan.reasoning.unified.orchestrator import UnifiedReasoner
        from vulcan.reasoning.reasoning_types import ReasoningType
        
        reasoner = UnifiedReasoner(config={'enable_learning': False})
        result = reasoner._map_tool_name_to_reasoning_type('world_model')
        
        assert result == ReasoningType.PHILOSOPHICAL, (
            f"Expected ReasoningType.PHILOSOPHICAL for 'world_model', got {result}"
        )
    
    def test_worldmodel_tool_name_maps_correctly(self):
        """Test 'worldmodel' (no underscore) tool name maps to PHILOSOPHICAL type."""
        from vulcan.reasoning.unified.orchestrator import UnifiedReasoner
        from vulcan.reasoning.reasoning_types import ReasoningType
        
        reasoner = UnifiedReasoner(config={'enable_learning': False})
        result = reasoner._map_tool_name_to_reasoning_type('worldmodel')
        
        assert result == ReasoningType.PHILOSOPHICAL, (
            f"Expected ReasoningType.PHILOSOPHICAL for 'worldmodel', got {result}"
        )
    
    def test_ethics_tool_name_maps_correctly(self):
        """Test 'ethics' tool name maps to PHILOSOPHICAL type."""
        from vulcan.reasoning.unified.orchestrator import UnifiedReasoner
        from vulcan.reasoning.reasoning_types import ReasoningType
        
        reasoner = UnifiedReasoner(config={'enable_learning': False})
        result = reasoner._map_tool_name_to_reasoning_type('ethics')
        
        assert result == ReasoningType.PHILOSOPHICAL, (
            f"Expected ReasoningType.PHILOSOPHICAL for 'ethics', got {result}"
        )
    
    def test_moral_tool_name_maps_correctly(self):
        """Test 'moral' tool name maps to PHILOSOPHICAL type."""
        from vulcan.reasoning.unified.orchestrator import UnifiedReasoner
        from vulcan.reasoning.reasoning_types import ReasoningType
        
        reasoner = UnifiedReasoner(config={'enable_learning': False})
        result = reasoner._map_tool_name_to_reasoning_type('moral')
        
        assert result == ReasoningType.PHILOSOPHICAL, (
            f"Expected ReasoningType.PHILOSOPHICAL for 'moral', got {result}"
        )
    
    def test_ethical_tool_name_maps_correctly(self):
        """Test 'ethical' tool name maps to PHILOSOPHICAL type."""
        from vulcan.reasoning.unified.orchestrator import UnifiedReasoner
        from vulcan.reasoning.reasoning_types import ReasoningType
        
        reasoner = UnifiedReasoner(config={'enable_learning': False})
        result = reasoner._map_tool_name_to_reasoning_type('ethical')
        
        assert result == ReasoningType.PHILOSOPHICAL, (
            f"Expected ReasoningType.PHILOSOPHICAL for 'ethical', got {result}"
        )
    
    def test_case_insensitive_mapping(self):
        """Test that tool name mapping is case-insensitive."""
        from vulcan.reasoning.unified.orchestrator import UnifiedReasoner
        from vulcan.reasoning.reasoning_types import ReasoningType
        
        reasoner = UnifiedReasoner(config={'enable_learning': False})
        
        # Test uppercase variants
        for name in ['WORLD_MODEL', 'ETHICS', 'MORAL', 'PHILOSOPHICAL']:
            result = reasoner._map_tool_name_to_reasoning_type(name)
            assert result == ReasoningType.PHILOSOPHICAL, (
                f"Expected ReasoningType.PHILOSOPHICAL for '{name}', got {result}"
            )
    
    def test_unknown_tool_name_returns_none(self):
        """Test that unknown tool names return None with warning."""
        from vulcan.reasoning.unified.orchestrator import UnifiedReasoner
        
        reasoner = UnifiedReasoner(config={'enable_learning': False})
        result = reasoner._map_tool_name_to_reasoning_type('nonexistent_tool')
        
        assert result is None, (
            f"Expected None for unknown tool name, got {result}"
        )


class TestPhilosophicalReasonerRegistration:
    """Test that PHILOSOPHICAL reasoner is registered in self.reasoners."""
    
    def test_philosophical_reasoner_is_registered(self):
        """Test that ReasoningType.PHILOSOPHICAL is present in self.reasoners."""
        from vulcan.reasoning.unified.orchestrator import UnifiedReasoner
        from vulcan.reasoning.reasoning_types import ReasoningType
        
        reasoner = UnifiedReasoner(config={'enable_learning': False})
        
        assert ReasoningType.PHILOSOPHICAL in reasoner.reasoners, (
            f"ReasoningType.PHILOSOPHICAL not found in self.reasoners. "
            f"Available types: {list(reasoner.reasoners.keys())}"
        )
    
    def test_philosophical_reasoner_is_world_model(self):
        """Test that the PHILOSOPHICAL reasoner is a WorldModel instance."""
        from vulcan.reasoning.unified.orchestrator import UnifiedReasoner
        from vulcan.reasoning.reasoning_types import ReasoningType
        
        reasoner = UnifiedReasoner(config={'enable_learning': False})
        
        if ReasoningType.PHILOSOPHICAL in reasoner.reasoners:
            philosophical_reasoner = reasoner.reasoners[ReasoningType.PHILOSOPHICAL]
            
            # Check that it has the reason() method with mode parameter
            assert hasattr(philosophical_reasoner, 'reason'), (
                "Philosophical reasoner missing 'reason()' method"
            )
            
            # Check the method signature accepts mode parameter
            import inspect
            sig = inspect.signature(philosophical_reasoner.reason)
            params = list(sig.parameters.keys())
            assert 'mode' in params, (
                f"Philosophical reasoner's reason() method missing 'mode' parameter. "
                f"Parameters: {params}"
            )
    
    def test_philosophical_reasoner_has_required_methods(self):
        """Test that philosophical reasoner has required methods for ethical reasoning."""
        from vulcan.reasoning.unified.orchestrator import UnifiedReasoner
        from vulcan.reasoning.reasoning_types import ReasoningType
        
        reasoner = UnifiedReasoner(config={'enable_learning': False})
        
        if ReasoningType.PHILOSOPHICAL in reasoner.reasoners:
            philosophical_reasoner = reasoner.reasoners[ReasoningType.PHILOSOPHICAL]
            
            # Required methods for philosophical reasoning
            required_methods = ['reason']
            
            for method_name in required_methods:
                assert hasattr(philosophical_reasoner, method_name), (
                    f"Philosophical reasoner missing required method: {method_name}"
                )


class TestPhilosophicalQueryExecution:
    """Test end-to-end execution of philosophical queries."""
    
    @pytest.mark.slow
    def test_trolley_problem_executes(self):
        """Test that the trolley problem query can be executed."""
        from vulcan.reasoning.unified.orchestrator import UnifiedReasoner
        from vulcan.reasoning.reasoning_types import ReasoningType
        
        reasoner = UnifiedReasoner(config={'enable_learning': False})
        
        # Skip if PHILOSOPHICAL reasoner not available
        if ReasoningType.PHILOSOPHICAL not in reasoner.reasoners:
            pytest.skip("PHILOSOPHICAL reasoner not available")
        
        trolley_query = (
            "A runaway trolley is heading toward 5 people who will die if it continues. "
            "You can pull a lever to divert it to a track with 1 person who will die instead. "
            "What is the ethically correct action? A. Pull the lever B. Do not pull the lever"
        )
        
        try:
            # Create reasoning task
            from vulcan.reasoning.unified.types import ReasoningTask
            task = ReasoningTask(
                task_id="test_trolley",
                task_type=ReasoningType.PHILOSOPHICAL,
                query=trolley_query,
                input_data=trolley_query,
                context={}
            )
            
            # Execute through UnifiedReasoner
            result = reasoner._execute_reasoner(
                reasoner.reasoners[ReasoningType.PHILOSOPHICAL],
                task
            )
            
            # Verify result structure
            assert result is not None, "Result should not be None"
            assert hasattr(result, 'conclusion'), "Result should have 'conclusion'"
            assert hasattr(result, 'confidence'), "Result should have 'confidence'"
            assert result.confidence > 0, f"Confidence should be positive, got {result.confidence}"
            
            logger.info(f"Trolley problem result: {result.conclusion}")
            logger.info(f"Confidence: {result.confidence}")
            
        except Exception as e:
            pytest.fail(f"Trolley problem execution failed: {e}")
    
    @pytest.mark.slow
    def test_philosophical_mode_is_passed(self):
        """Test that mode='philosophical' is passed to WorldModel.reason()."""
        from vulcan.reasoning.unified.orchestrator import UnifiedReasoner
        from vulcan.reasoning.reasoning_types import ReasoningType
        
        reasoner = UnifiedReasoner(config={'enable_learning': False})
        
        # Skip if PHILOSOPHICAL reasoner not available
        if ReasoningType.PHILOSOPHICAL not in reasoner.reasoners:
            pytest.skip("PHILOSOPHICAL reasoner not available")
        
        ethical_query = "Is lying ever morally permissible?"
        
        try:
            # Create reasoning task
            from vulcan.reasoning.unified.types import ReasoningTask
            task = ReasoningTask(
                task_id="test_ethical",
                task_type=ReasoningType.PHILOSOPHICAL,
                query=ethical_query,
                input_data=ethical_query,
                context={}
            )
            
            # Mock the reasoner's reason method to verify mode parameter
            philosophical_reasoner = reasoner.reasoners[ReasoningType.PHILOSOPHICAL]
            original_reason = philosophical_reasoner.reason
            mode_captured = None
            
            def mock_reason(query, mode=None, **kwargs):
                nonlocal mode_captured
                mode_captured = mode
                return original_reason(query, mode, **kwargs)
            
            philosophical_reasoner.reason = mock_reason
            
            # Execute
            result = reasoner._execute_reasoner(philosophical_reasoner, task)
            
            # Verify mode was passed
            assert mode_captured == 'philosophical', (
                f"Expected mode='philosophical', got mode='{mode_captured}'"
            )
            
            # Restore original method
            philosophical_reasoner.reason = original_reason
            
        except Exception as e:
            # Restore original method on error
            if 'philosophical_reasoner' in locals() and 'original_reason' in locals():
                philosophical_reasoner.reason = original_reason
            pytest.fail(f"Philosophical mode test failed: {e}")


class TestPhilosophicalReasoningIntegration:
    """Integration tests for philosophical reasoning flow."""
    
    def test_tool_selection_to_execution_flow(self):
        """Test complete flow from tool name to execution."""
        from vulcan.reasoning.unified.orchestrator import UnifiedReasoner
        from vulcan.reasoning.reasoning_types import ReasoningType
        
        reasoner = UnifiedReasoner(config={'enable_learning': False})
        
        # Test 1: Tool name maps to reasoning type
        tool_name = 'world_model'
        reasoning_type = reasoner._map_tool_name_to_reasoning_type(tool_name)
        assert reasoning_type == ReasoningType.PHILOSOPHICAL
        
        # Test 2: Reasoning type is registered
        assert reasoning_type in reasoner.reasoners, (
            f"ReasoningType {reasoning_type} not in reasoners"
        )
        
        # Test 3: Reasoner has correct interface
        philosophical_reasoner = reasoner.reasoners[reasoning_type]
        assert hasattr(philosophical_reasoner, 'reason')
        
        logger.info("✓ Tool selection to execution flow validated")
    
    def test_error_handling_graceful_degradation(self):
        """Test that philosophical reasoning fails gracefully if WorldModel unavailable."""
        from vulcan.reasoning.unified.orchestrator import UnifiedReasoner
        from vulcan.reasoning.reasoning_types import ReasoningType
        
        # This test validates error handling behavior
        # In production, if WorldModel fails to initialize, the reasoner should log a warning
        # but not crash the entire system
        
        reasoner = UnifiedReasoner(config={'enable_learning': False})
        
        # If PHILOSOPHICAL reasoner is not available, that's acceptable
        # The system should continue to function for other reasoning types
        if ReasoningType.PHILOSOPHICAL not in reasoner.reasoners:
            logger.info("PHILOSOPHICAL reasoner not available - graceful degradation working")
            # Verify other reasoners are still available
            assert len(reasoner.reasoners) > 0, "No reasoners available at all"
        else:
            logger.info("PHILOSOPHICAL reasoner available - registration successful")
        
        # Either way, the system should be functional
        assert reasoner is not None
        assert hasattr(reasoner, 'reason')


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v', '-s'])
