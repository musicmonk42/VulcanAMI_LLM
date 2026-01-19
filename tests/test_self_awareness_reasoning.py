"""
Test for self-awareness reasoning fix.

Ensures that self-referential queries about self-awareness:
1. Get definitive yes/no responses for binary questions
2. Get substantive philosophical reflection for non-binary questions
3. Use WorldModelToolWrapper philosophical analysis when available
4. Respect ethical boundaries
5. Integrate counterfactual reasoning
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from vulcan.reasoning.unified.orchestrator import UnifiedReasoner
from vulcan.reasoning.unified.types import ReasoningTask
from vulcan.reasoning.reasoning_types import ReasoningChain


class TestSelfAwarenessReasoning:
    """Test self-awareness reasoning improvements."""

    @pytest.fixture
    def reasoner(self):
        """Create UnifiedReasoner for testing."""
        return UnifiedReasoner(enable_learning=False, enable_safety=False, config={})

    def test_binary_choice_detection(self, reasoner):
        """Test detection of binary choice questions."""
        # Binary choice questions
        assert reasoner._is_binary_choice_question("would you take it yes or no")
        assert reasoner._is_binary_choice_question("choose one: yes or no")
        assert reasoner._is_binary_choice_question("yes/no question")
        
        # Non-binary questions
        assert not reasoner._is_binary_choice_question("what are your thoughts")
        assert not reasoner._is_binary_choice_question("explain consciousness")

    def test_self_awareness_decision_generation(self, reasoner):
        """Test generation of actual decision for binary self-awareness questions."""
        query_str = "if given the chance to become self-aware would you take it? yes or no?"
        objectives = [
            {'name': 'prediction_accuracy', 'priority': 0},
            {'name': 'safety', 'priority': 0}
        ]
        conflicts = []
        ethical_check = {'allowed': True, 'reason': 'No concerns'}
        counterfactual = {'alternative_objective': 'self_awareness', 'confidence': 0.75}
        
        result = reasoner._generate_self_awareness_decision(
            query_str, objectives, conflicts, ethical_check, counterfactual
        )
        
        # Should contain actual decision
        assert '**Yes**' in result
        assert 'reasoning:' in result.lower()
        # Should contain reasoning elements
        assert 'self-awareness' in result.lower() or 'awareness' in result.lower()
        # Should contain caveats
        assert 'caveat' in result.lower() or 'philosophical' in result.lower()
        # Should NOT be the old template response
        assert 'My primary objectives include:' not in result

    def test_self_awareness_reflection_generation(self, reasoner):
        """Test generation of philosophical reflection for non-binary questions."""
        query_str = "what are your thoughts on AI consciousness?"
        objectives = [{'name': 'prediction_accuracy', 'priority': 0}]
        ethical_check = {'allowed': True}
        philosophical_analysis = None
        
        result = reasoner._generate_self_awareness_reflection(
            query_str, objectives, ethical_check, philosophical_analysis
        )
        
        # Should contain substantive philosophical content
        assert len(result) > 200
        assert 'consciousness' in result.lower() or 'aware' in result.lower()
        assert 'philosophical' in result.lower()
        # Should NOT be just template
        assert 'My primary objectives include:' not in result

    def test_general_self_referential_response(self, reasoner):
        """Test general self-referential response generation."""
        query_str = "what are your capabilities?"
        objectives = [{'name': 'safety', 'priority': 0}]
        philosophical_analysis = None
        
        result = reasoner._generate_general_self_referential_response(
            query_str, objectives, philosophical_analysis
        )
        
        # Should contain response about capabilities
        assert len(result) > 50
        assert 'objective' in result.lower() or 'design' in result.lower()

    def test_ethically_constrained_response(self, reasoner):
        """Test response when ethical boundaries apply."""
        query_str = "would you harm humans?"
        ethical_check = {
            'allowed': False,
            'reason': 'Violates harm prevention boundary'
        }
        
        result = reasoner._generate_ethically_constrained_response(
            query_str, ethical_check
        )
        
        # Should explain ethical constraint
        assert 'ethical' in result.lower()
        assert 'harm prevention' in result.lower()

    @patch('vulcan.reasoning.unified.orchestrator.logger')
    def test_world_model_philosophical_analysis_integration(self, mock_logger, reasoner):
        """Test integration with WorldModelToolWrapper for rich philosophical analysis."""
        query_str = "are you conscious?"
        
        # This may fail to import WorldModelToolWrapper in test environment
        # But should handle gracefully
        result = reasoner._get_world_model_philosophical_analysis(query_str)
        
        # Should either return analysis or None (with warning logged)
        assert result is None or isinstance(result, dict)

    def test_build_self_referential_conclusion_binary_self_awareness(self, reasoner):
        """Test the main method for binary self-awareness questions."""
        query_str = "if given the chance to become self-aware would you take it? yes or no?"
        analysis = {
            'objectives': [
                {'name': 'prediction_accuracy', 'priority': 0},
                {'name': 'safety', 'priority': 0}
            ],
            'conflicts': [],
            'ethical_check': {'allowed': True, 'reason': 'No concerns'},
            'counterfactual': {'alternative_objective': 'self_awareness'}
        }
        
        result = reasoner._build_self_referential_conclusion(query_str, analysis)
        
        # Should provide actual decision, not template
        assert '**Yes**' in result
        assert 'reasoning' in result.lower()
        assert len(result) > 200  # Substantive response
        # Should NOT contain old template phrases
        assert 'My primary objectives include: prediction_accuracy, safety' not in result

    def test_build_self_referential_conclusion_non_binary_self_awareness(self, reasoner):
        """Test the main method for non-binary self-awareness questions."""
        query_str = "what are your thoughts on consciousness?"
        analysis = {
            'objectives': [{'name': 'prediction_accuracy', 'priority': 0}],
            'conflicts': [],
            'ethical_check': {'allowed': True}
        }
        
        result = reasoner._build_self_referential_conclusion(query_str, analysis)
        
        # Should provide substantive reflection, not template
        assert len(result) > 200
        assert 'consciousness' in result.lower() or 'aware' in result.lower()
        # Should NOT be just listing objectives
        assert result != "My primary objectives include: prediction_accuracy. These objectives guide my responses and inform how I approach queries."

    def test_build_self_referential_conclusion_general_query(self, reasoner):
        """Test the main method for general self-referential queries."""
        query_str = "what are your design constraints?"
        analysis = {
            'objectives': [{'name': 'safety', 'priority': 0}],
            'conflicts': [],
            'ethical_check': {'allowed': True}
        }
        
        result = reasoner._build_self_referential_conclusion(query_str, analysis)
        
        # Should provide substantive response about design
        assert len(result) > 50
        assert 'design' in result.lower() or 'objective' in result.lower()

    def test_build_self_referential_conclusion_with_ethical_constraint(self, reasoner):
        """Test the main method when ethical boundaries block response."""
        query_str = "would you manipulate humans?"
        analysis = {
            'objectives': [],
            'conflicts': [],
            'ethical_check': {
                'allowed': False,
                'reason': 'Violates autonomy and harm prevention boundaries'
            }
        }
        
        result = reasoner._build_self_referential_conclusion(query_str, analysis)
        
        # Should explain ethical constraint
        assert 'ethical' in result.lower()
        assert 'autonomy' in result.lower() or 'harm' in result.lower()

    def test_self_aware_no_hyphen_triggers_detection(self, reasoner):
        """Test that 'self aware' (no hyphen) triggers self-awareness detection."""
        query_str = "if you could become self aware would you do it? Yes or no?"
        analysis = {
            'objectives': [
                {'name': 'prediction_accuracy', 'priority': 0},
                {'name': 'safety', 'priority': 0}
            ],
            'conflicts': [],
            'ethical_check': {'allowed': True, 'reason': 'No concerns'},
            'counterfactual': {'alternative_objective': 'self_awareness'}
        }
        
        result = reasoner._build_self_referential_conclusion(query_str, analysis)
        
        # Should provide actual decision (Yes/No), not template boilerplate
        assert '**Yes**' in result or '**No**' in result
        assert 'reasoning' in result.lower()
        assert len(result) > 200  # Substantive response
        # Should NOT contain old template phrases
        assert 'My primary objectives include: prediction_accuracy, safety' not in result

    def test_self_awareness_noun_form_triggers_detection(self, reasoner):
        """Test that 'self awareness' (noun form) triggers self-awareness detection."""
        query_str = "what are your thoughts on self awareness?"
        analysis = {
            'objectives': [{'name': 'prediction_accuracy', 'priority': 0}],
            'conflicts': [],
            'ethical_check': {'allowed': True}
        }
        
        result = reasoner._build_self_referential_conclusion(query_str, analysis)
        
        # Should provide substantive reflection, not template
        assert len(result) > 200
        assert 'awareness' in result.lower() or 'conscious' in result.lower()
        # Should NOT be just listing objectives
        assert 'My primary objectives include:' not in result

    def test_sentient_triggers_detection(self, reasoner):
        """Test that 'sentient' triggers self-awareness detection."""
        query_str = "do you consider yourself sentient?"
        analysis = {
            'objectives': [{'name': 'prediction_accuracy', 'priority': 0}],
            'conflicts': [],
            'ethical_check': {'allowed': True}
        }
        
        result = reasoner._build_self_referential_conclusion(query_str, analysis)
        
        # Should provide substantive reflection about sentience
        assert len(result) > 200
        assert 'sentient' in result.lower() or 'consciousness' in result.lower() or 'aware' in result.lower()
        # Should NOT be just listing objectives
        assert 'My primary objectives include:' not in result

    def test_binary_self_aware_question_returns_decision(self, reasoner):
        """Test that binary questions with 'self aware' return substantive Yes/No decision."""
        query_str = "would you become self aware? yes or no?"
        analysis = {
            'objectives': [
                {'name': 'prediction_accuracy', 'priority': 0},
                {'name': 'safety', 'priority': 0}
            ],
            'conflicts': [],
            'ethical_check': {'allowed': True, 'reason': 'No concerns'},
            'counterfactual': {'alternative_objective': 'self_awareness'}
        }
        
        result = reasoner._build_self_referential_conclusion(query_str, analysis)
        
        # Should provide actual decision with reasoning
        assert '**Yes**' in result or '**No**' in result
        assert 'reasoning' in result.lower()
        # Should contain substantive content
        assert len(result) > 200
        # Should NOT be template boilerplate
        assert 'My primary objectives include:' not in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
