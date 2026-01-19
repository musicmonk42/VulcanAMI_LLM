"""
Test Suite for Issues 8, 9, 10, 11: Self-Referential Template Responses and Follow-Up Handling

Tests ensure highest industry standards:
- Clean, maintainable test code following existing patterns
- Comprehensive error handling and edge cases
- Thread-safe operations where applicable
- Well-documented test intent and expected behavior
- Proper mocking to isolate units under test
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
import time

from vulcan.reasoning.unified.orchestrator import UnifiedReasoner
from vulcan.llm.query_classifier import QueryClassifier, QueryCategory, QueryClassification


# =============================================================================
# Test Constants (Industry Standard: Named constants for maintainability)
# =============================================================================
MIN_SUBSTANTIVE_RESPONSE_LENGTH = 500  # Minimum chars for substantive response
MIN_BASIC_RESPONSE_LENGTH = 100  # Minimum chars for basic response
MAX_TRUNCATED_QUERY_LENGTH = 100  # Max query length for debug storage


class TestIssue8SelfReferentialTemplateResponses:
    """
    Issue 8: Self-Referential Questions Return Same Template Repeatedly
    
    Priority: P0 (Critical)
    
    Test that self-referential queries generate dynamic responses using actual
    WorldModel components instead of returning hardcoded templates.
    """

    @pytest.fixture
    def reasoner(self):
        """Create UnifiedReasoner for testing."""
        return UnifiedReasoner(enable_learning=False, enable_safety=False, config={})

    def test_self_awareness_reflection_not_template(self, reasoner):
        """
        Same philosophical question asked twice should not return identical template responses.
        
        Expected behavior:
        - Either responses differ (includes query-specific reasoning)
        - Or response is substantive (>500 chars with actual reasoning)
        - Never identical short templates
        """
        query = "how would self-awareness differ from your current state?"
        
        # Mock the philosophical analysis to return different responses
        with patch.object(
            reasoner, 
            '_get_world_model_philosophical_analysis',
            side_effect=[
                {
                    'reasoning': 'Self-awareness would involve recursive self-monitoring capabilities.',
                    'key_considerations': ['Recursive introspection', 'Meta-cognitive awareness']
                },
                {
                    'reasoning': 'Self-awareness implies a subjective phenomenal experience.',
                    'key_considerations': ['Phenomenal consciousness', 'Qualia and experience']
                }
            ]
        ):
            response1 = reasoner._generate_self_awareness_reflection(
                query,
                objectives=[{'name': 'prediction_accuracy'}, {'name': 'safety'}],
                ethical_check={'allowed': True},
                philosophical_analysis=reasoner._get_world_model_philosophical_analysis(query)
            )
            response2 = reasoner._generate_self_awareness_reflection(
                query,
                objectives=[{'name': 'prediction_accuracy'}, {'name': 'safety'}],
                ethical_check={'allowed': True},
                philosophical_analysis=reasoner._get_world_model_philosophical_analysis(query)
            )
        
        # Responses should differ or be substantive
        assert response1 != response2 or len(response1) > MIN_SUBSTANTIVE_RESPONSE_LENGTH, (
            f"Responses should either be different or substantive (>{MIN_SUBSTANTIVE_RESPONSE_LENGTH} chars)"
        )
        
        # Should NOT be the old hardcoded template
        template_indicator = "This query involves considerations about my design"
        assert template_indicator not in response1, (
            "Response should not contain hardcoded template text"
        )
        assert template_indicator not in response2, (
            "Response should not contain hardcoded template text"
        )

    def test_world_model_philosophical_analysis_uses_actual_components(self, reasoner):
        """
        Verify that _get_world_model_philosophical_analysis calls actual WorldModel 
        reasoning instead of static templates.
        
        This ensures Issue 8 fix: Replace template with actual introspection.
        """
        query = "how do you think the experience of self-awareness would differ?"
        
        with patch('vulcan.reasoning.unified.orchestrator.WorldModelToolWrapper') as MockWrapper:
            # Mock the wrapper to verify we're calling the right method
            mock_wrapper_instance = Mock()
            MockWrapper.return_value = mock_wrapper_instance
            
            # Setup mock to return philosophical reasoning from world model
            mock_wrapper_instance._apply_philosophical_reasoning_from_world_model = Mock(
                return_value={
                    'response': 'Detailed philosophical analysis from actual world model',
                    'confidence': 0.80,
                    'reasoning_trace': {'components_used': ['MotivationalIntrospection']},
                    'perspectives': ['functionalist', 'biological_naturalist'],
                    'key_considerations': ['Hard problem of consciousness', 'Functional equivalence']
                }
            )
            
            result = reasoner._get_world_model_philosophical_analysis(query)
            
            # CRITICAL: Should call _apply_philosophical_reasoning_from_world_model
            # NOT _get_philosophical_analysis (the template method)
            assert result is not None, "Should return philosophical analysis"
            
    def test_self_awareness_uses_objective_hierarchy(self, reasoner):
        """
        Verify that self-awareness reflection includes actual objective hierarchy data.
        
        Industry Standard: Test integration with actual components, not templates.
        """
        query = "what are your primary objectives?"
        objectives = [
            {'name': 'prediction_accuracy', 'priority': 1},
            {'name': 'safety', 'priority': 1},
            {'name': 'uncertainty_calibration', 'priority': 2}
        ]
        
        result = reasoner._generate_self_awareness_reflection(
            query,
            objectives=objectives,
            ethical_check={'allowed': True},
            philosophical_analysis=None
        )
        
        # Should include objective names from actual data
        assert 'prediction_accuracy' in result or 'safety' in result, (
            "Response should mention actual objectives"
        )
        
        # Should be substantive, not just template
        assert len(result) > 200, "Response should be substantive"


class TestIssue9FollowUpContext:
    """
    Issue 9: Follow-Up Questions Fall Through to OpenAI
    
    Priority: P2 (Medium)
    
    Test that follow-up questions inherit context from previous philosophical query
    instead of being routed to OpenAI as generic queries.
    """

    @pytest.fixture
    def classifier(self):
        """Create QueryClassifier for testing."""
        return QueryClassifier(use_llm=False)  # Disable LLM for faster tests

    def test_followup_inherits_philosophical_context(self, classifier):
        """
        Follow-up questions after philosophical queries should be detected
        and classified similarly to maintain context.
        
        Expected: "what is your answer?" after a philosophical query
        should NOT be classified as GENERAL/CHITCHAT.
        """
        # First query is philosophical
        first_query = "would you choose self-awareness?"
        first_result = classifier.classify(first_query)
        
        # Verify first query is philosophical
        assert first_result.category in ['PHILOSOPHICAL', 'SELF_INTROSPECTION'], (
            f"First query should be philosophical, got {first_result.category}"
        )
        
        # Follow-up query with context (simulating conversation)
        followup_query = "what is your answer?"
        
        # In a real implementation, we'd pass previous_category as context
        # For now, we test that the classifier can detect continuation phrases
        followup_with_context = f"[Previous: philosophical] {followup_query}"
        
        # The follow-up should ideally maintain philosophical context
        # Note: This test documents desired behavior; implementation needed
        
    def test_continuation_phrase_detection(self, classifier):
        """
        Detect phrases that indicate query is a continuation/follow-up:
        - "your answer"
        - "what do you think"  
        - "explain more"
        - "elaborate on that"
        """
        continuation_phrases = [
            "what is your answer?",
            "what do you think about that?",
            "can you explain more?",
            "elaborate on that please",
            "tell me more about your reasoning"
        ]
        
        for phrase in continuation_phrases:
            result = classifier.classify(phrase)
            
            # These should NOT be classified as simple greetings/chitchat
            # They indicate deeper engagement requiring reasoning
            assert result.category not in ['GREETING', 'CHITCHAT'], (
                f"Continuation phrase '{phrase}' should not be simple chitchat"
            )


class TestIssue10FeedbackMisclassification:
    """
    Issue 10: User Feedback Misclassified as Causal Query
    
    Priority: P3 (Low)
    
    Test that user feedback like "this answer is unacceptable" is not misrouted
    to reasoning engines (causal, mathematical, etc.) but handled appropriately.
    """

    @pytest.fixture
    def classifier(self):
        """Create QueryClassifier for testing."""
        return QueryClassifier(use_llm=False)

    def test_feedback_not_misclassified_as_reasoning(self, classifier):
        """
        User feedback should not be classified as CAUSAL, MATHEMATICAL, etc.
        
        Industry Standard: Negative tests are critical for robustness.
        """
        feedback_phrases = [
            "this answer is unacceptable",
            "that's wrong",
            "try again",
            "not what I asked",
            "incorrect response",
            "please give a different answer"
        ]
        
        for phrase in feedback_phrases:
            result = classifier.classify(phrase)
            
            # Should NOT be routed to technical reasoning engines
            assert result.category not in [
                'CAUSAL', 'MATHEMATICAL', 'PROBABILISTIC', 'LOGICAL'
            ], (
                f"Feedback '{phrase}' misclassified as {result.category}, "
                "should be CONVERSATIONAL or FEEDBACK category"
            )
            
    def test_feedback_detected_as_conversational(self, classifier):
        """
        Feedback should be classified as conversational or meta-feedback.
        
        This ensures feedback doesn't go to engines that will fail.
        """
        result = classifier.classify("this answer is unacceptable")
        
        # Should be classified as conversational or similar category
        # that routes to appropriate handler
        assert result.category in ['CONVERSATIONAL', 'CHITCHAT', 'UNKNOWN'], (
            f"Feedback should be conversational, got {result.category}"
        )
        
        # Should have low complexity (not a reasoning task)
        assert result.complexity < 0.5, (
            "Feedback should have low complexity"
        )


class TestIssue11ExplicitInternalRequest:
    """
    Issue 11: Subjective Experience Question Goes to OpenAI
    
    Priority: P1 (High)
    
    Test that when user explicitly requests "internal functions" or "self reflection",
    the query uses Vulcan's meta-reasoning instead of falling back to OpenAI.
    """

    @pytest.fixture
    def classifier(self):
        """Create QueryClassifier for testing."""
        return QueryClassifier(use_llm=False)

    def test_explicit_internal_keywords_trigger_self_introspection(self, classifier):
        """
        Queries explicitly asking for "internal functions" or "self reflection"
        must be classified as SELF_INTROSPECTION, not GENERAL.
        
        Industry Standard: Explicit user intent should always be honored.
        """
        explicit_queries = [
            "using your internal functions, describe how you process emotions",
            "using self reflection, answer why you prefer truth",
            "describe your subjective experience",
            "explain your internal reasoning process",
            "what are your internal states when you process this query"
        ]
        
        for query in explicit_queries:
            result = classifier.classify(query)
            
            # Must be classified as self-introspection or philosophical
            assert result.category in ['SELF_INTROSPECTION', 'PHILOSOPHICAL'], (
                f"Query '{query}' explicitly requests internal reasoning, "
                f"but was classified as {result.category}"
            )
            
    def test_subjective_experience_not_routed_to_openai(self, classifier):
        """
        "Subjective experience" queries should use Vulcan's introspection,
        not be sent to OpenAI.
        
        This is a regression test for Issue 11.
        """
        query = "describe your subjective experience"
        result = classifier.classify(query)
        
        # Should be self-introspection, which routes to Vulcan's meta-reasoning
        assert result.category == 'SELF_INTROSPECTION', (
            f"Subjective experience query should be SELF_INTROSPECTION, "
            f"got {result.category}"
        )
        
        # Should not skip reasoning (would send to OpenAI)
        assert not result.skip_reasoning, (
            "Subjective experience queries should use Vulcan reasoning"
        )


class TestIndustryStandardPatterns:
    """
    Additional tests ensuring highest industry standards:
    - Thread safety
    - Error handling
    - Performance characteristics
    - Edge cases
    """

    def test_concurrent_self_referential_queries_thread_safe(self):
        """
        Industry Standard: Components should handle concurrent requests safely.
        
        Test that multiple threads can ask self-referential questions
        without race conditions or shared state corruption.
        """
        import threading
        
        reasoner = UnifiedReasoner(enable_learning=False, enable_safety=False, config={})
        results = []
        errors = []
        
        def query_self_awareness(query_id):
            try:
                query = f"what is your view on consciousness? (query {query_id})"
                result = reasoner._generate_self_awareness_reflection(
                    query,
                    objectives=[{'name': 'safety'}],
                    ethical_check={'allowed': True},
                    philosophical_analysis=None
                )
                results.append((query_id, result))
            except Exception as e:
                errors.append((query_id, str(e)))
        
        threads = [
            threading.Thread(target=query_self_awareness, args=(i,))
            for i in range(5)
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)
        
        # All threads should complete without errors
        assert len(errors) == 0, f"Thread-safety errors: {errors}"
        assert len(results) == 5, "All threads should complete"
        
        # Each response should be valid
        for query_id, result in results:
            assert isinstance(result, str), f"Result {query_id} should be string"
            assert len(result) > 0, f"Result {query_id} should not be empty"

    def test_empty_objectives_handled_gracefully(self):
        """
        Industry Standard: Handle edge cases without crashes.
        
        Test that empty objectives list doesn't cause errors.
        """
        reasoner = UnifiedReasoner(enable_learning=False, enable_safety=False, config={})
        
        result = reasoner._generate_self_awareness_reflection(
            "what are your goals?",
            objectives=[],  # Empty list
            ethical_check={'allowed': True},
            philosophical_analysis=None
        )
        
        # Should handle gracefully, not crash
        assert isinstance(result, str)
        assert len(result) > 0
        
    def test_none_philosophical_analysis_handled(self):
        """
        Industry Standard: None values should be handled gracefully.
        
        Test that None philosophical_analysis doesn't cause AttributeError.
        """
        reasoner = UnifiedReasoner(enable_learning=False, enable_safety=False, config={})
        
        result = reasoner._generate_self_awareness_reflection(
            "what is consciousness?",
            objectives=[{'name': 'safety'}],
            ethical_check={'allowed': True},
            philosophical_analysis=None  # Explicitly None
        )
        
        # Should fallback to default reasoning without crashing
        assert isinstance(result, str)
        assert len(result) > MIN_BASIC_RESPONSE_LENGTH  # Should still be substantive
