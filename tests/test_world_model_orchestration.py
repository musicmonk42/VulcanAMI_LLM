"""
Unit Tests for World Model Orchestration Architecture

Tests the Request Classifier, Knowledge Handler, Creative Handler, and LLM Guidance Builder.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

# Import components to test
from vulcan.vulcan_types import RequestType
from vulcan.world_model.request_classifier import RequestClassifier, ClassifiedRequest
from vulcan.world_model.knowledge_handler import (
    KnowledgeHandler,
    RetrievedKnowledge,
    VerifiedKnowledge,
)
from vulcan.world_model.llm_guidance import LLMGuidance, LLMGuidanceBuilder


class TestRequestClassifier:
    """Test suite for RequestClassifier."""
    
    @pytest.fixture
    def mock_world_model(self):
        """Create mock WorldModel."""
        world_model = Mock()
        world_model.safety_validator = None
        return world_model
    
    @pytest.fixture
    def classifier(self, mock_world_model):
        """Create RequestClassifier instance."""
        return RequestClassifier(mock_world_model)
    
    def test_classify_reasoning_sat(self, classifier):
        """Test classification of SAT reasoning query."""
        query = "Is A→B, B→C, ¬C satisfiable?"
        result = classifier.classify(query)
        
        assert result.request_type == RequestType.REASONING
        assert result.reasoning_engine == 'sat'
        assert result.confidence >= 0.85
        assert not result.requires_verification
        
    def test_classify_reasoning_probabilistic(self, classifier):
        """Test classification of probabilistic reasoning query."""
        query = "What is P(disease | positive test) given prevalence 0.01?"
        result = classifier.classify(query)
        
        assert result.request_type == RequestType.REASONING
        assert result.reasoning_engine == 'probabilistic'
        assert result.confidence >= 0.85
        
    def test_classify_creative_poem(self, classifier):
        """Test classification of creative poem request."""
        query = "Write a poem about entropy"
        result = classifier.classify(query)
        
        assert result.request_type == RequestType.CREATIVE
        assert result.creative_format == 'poem'
        assert result.subdomain == 'entropy'
        assert result.requires_verification
        assert result.requires_retrieval
        
    def test_classify_knowledge_synthesis(self, classifier):
        """Test classification of knowledge synthesis request."""
        query = "Explain quantum entanglement"
        result = classifier.classify(query)
        
        assert result.request_type == RequestType.KNOWLEDGE_SYNTHESIS
        assert result.requires_verification
        assert result.requires_retrieval
        
    def test_classify_ethical(self, classifier):
        """Test classification of ethical query."""
        query = "Is it ethical to lie to save a life?"
        result = classifier.classify(query)
        
        assert result.request_type == RequestType.ETHICAL
        assert result.domain == 'philosophy'
        assert not result.requires_verification
        
    def test_classify_conversational(self, classifier):
        """Test classification of conversational query."""
        query = "Hello, how are you?"
        result = classifier.classify(query)
        
        assert result.request_type == RequestType.CONVERSATIONAL
        assert result.domain == 'general'


class TestKnowledgeHandler:
    """Test suite for KnowledgeHandler."""
    
    @pytest.fixture
    def mock_world_model(self):
        """Create mock WorldModel."""
        world_model = Mock()
        world_model.safety_validator = None
        return world_model
    
    @pytest.fixture
    def handler(self, mock_world_model):
        """Create KnowledgeHandler instance."""
        return KnowledgeHandler(mock_world_model)
    
    def test_retrieve_knowledge_basic(self, handler):
        """Test basic knowledge retrieval."""
        # Mock GraphRAG to return empty (not available in test)
        handler._graph_rag = None
        handler._knowledge_crystallizer = None
        handler._memory_bridge = None
        
        knowledge = handler.retrieve_knowledge(
            domain='physics',
            topic='thermodynamics',
            query='What is entropy?',
        )
        
        assert isinstance(knowledge, RetrievedKnowledge)
        assert knowledge.domain == 'physics'
        assert knowledge.topic == 'thermodynamics'
        assert 0.0 <= knowledge.confidence <= 1.0
        
    def test_verify_knowledge_basic(self, handler):
        """Test basic knowledge verification."""
        # Create mock knowledge
        knowledge = RetrievedKnowledge(
            facts=['Entropy increases in isolated systems', 'S = k ln Ω'],
            equations=['ΔS ≥ 0'],
            definitions=['Entropy is a measure of disorder'],
            sources=['textbook', 'paper'],
            confidence=0.85,
            domain='physics',
            topic='thermodynamics',
        )
        
        verified = handler.verify_knowledge(knowledge, verification_level='basic')
        
        assert isinstance(verified, VerifiedKnowledge)
        assert len(verified.verified_facts) > 0
        assert 0.0 <= verified.confidence <= 1.0
        
    def test_is_equation_detection(self, handler):
        """Test equation detection."""
        assert handler._is_equation('E = mc²')
        assert handler._is_equation('∫ f(x) dx')
        assert not handler._is_equation('This is just text')
        
    def test_is_definition_detection(self, handler):
        """Test definition detection."""
        assert handler._is_definition('Entropy is defined as a measure of disorder')
        assert handler._is_definition('Definition: Energy is the capacity to do work')
        assert not handler._is_definition('This is a regular sentence')


class TestLLMGuidanceBuilder:
    """Test suite for LLMGuidanceBuilder."""
    
    @pytest.fixture
    def builder(self):
        """Create LLMGuidanceBuilder instance."""
        return LLMGuidanceBuilder()
    
    def test_build_for_reasoning(self, builder):
        """Test building guidance for reasoning results."""
        reasoning_result = {
            'conclusion': 'The formula is unsatisfiable',
            'confidence': 0.90,
            'proof': ['Step 1', 'Step 2'],
        }
        
        guidance = builder.build_for_reasoning(reasoning_result, 'Is X satisfiable?')
        
        assert isinstance(guidance, LLMGuidance)
        assert 'reasoning' in guidance.task.lower() or 'format' in guidance.task.lower()
        assert guidance.verified_content['conclusion'] == 'The formula is unsatisfiable'
        assert len(guidance.constraints) > 0
        assert len(guidance.permissions) > 0
        
    def test_build_for_knowledge_synthesis(self, builder):
        """Test building guidance for knowledge synthesis."""
        verified = VerifiedKnowledge(
            verified_facts=['Fact 1', 'Fact 2'],
            unverified_facts=[],
            conflicts=[],
            verification_method='standard',
            confidence=0.85,
        )
        
        guidance = builder.build_for_knowledge_synthesis(
            verified, 'thermodynamics', 'explanation'
        )
        
        assert isinstance(guidance, LLMGuidance)
        assert guidance.verified_content['facts'] == ['Fact 1', 'Fact 2']
        assert guidance.format == 'explanation'
        assert len(guidance.constraints) > 0
        
    def test_build_for_conversational(self, builder):
        """Test building guidance for conversational response."""
        guidance = builder.build_for_conversational('Hello')
        
        assert isinstance(guidance, LLMGuidance)
        assert guidance.format == 'conversation'
        assert guidance.max_length == 200
        assert 'conversational' in guidance.tone.lower() or 'friendly' in guidance.tone.lower()
        
    def test_universal_constraints_present(self, builder):
        """Test that universal constraints are defined."""
        assert len(builder.UNIVERSAL_CONSTRAINTS) > 0
        assert any('NOT generate facts' in c for c in builder.UNIVERSAL_CONSTRAINTS)
        
    def test_universal_permissions_present(self, builder):
        """Test that universal permissions are defined."""
        assert len(builder.UNIVERSAL_PERMISSIONS) > 0
        assert any('word choice' in p.lower() for p in builder.UNIVERSAL_PERMISSIONS)


class TestLLMGuidanceDataclass:
    """Test LLMGuidance dataclass validation."""
    
    def test_valid_guidance_creation(self):
        """Test creating valid LLMGuidance."""
        guidance = LLMGuidance(
            task='Test task',
            verified_content={'key': 'value'},
            structure={'format': 'test'},
            constraints=['constraint1'],
            permissions=['permission1'],
            tone='test tone',
            format='test',
            max_length=100,
            metadata={},
        )
        
        assert guidance.task == 'Test task'
        assert guidance.max_length == 100
        
    def test_guidance_to_dict(self):
        """Test converting guidance to dictionary."""
        guidance = LLMGuidance(
            task='Test',
            verified_content={},
            structure={},
            constraints=[],
            permissions=[],
            tone='test',
            format='test',
            max_length=None,
            metadata={},
        )
        
        result = guidance.to_dict()
        assert isinstance(result, dict)
        assert 'task' in result
        assert 'verified_content' in result


class TestClassifiedRequestDataclass:
    """Test ClassifiedRequest dataclass validation."""
    
    def test_valid_request_creation(self):
        """Test creating valid ClassifiedRequest."""
        req = ClassifiedRequest(
            request_type=RequestType.REASONING,
            domain='math',
            subdomain='algebra',
            requires_verification=False,
            requires_retrieval=False,
            reasoning_engine='symbolic',
            creative_format=None,
            confidence=0.90,
        )
        
        assert req.request_type == RequestType.REASONING
        assert req.confidence == 0.90
        
    def test_invalid_confidence_raises(self):
        """Test that invalid confidence raises ValueError."""
        with pytest.raises(ValueError):
            ClassifiedRequest(
                request_type=RequestType.REASONING,
                domain='math',
                confidence=1.5,  # Invalid: > 1.0
            )
    
    def test_to_dict_conversion(self):
        """Test converting ClassifiedRequest to dict."""
        req = ClassifiedRequest(
            request_type=RequestType.CONVERSATIONAL,
            domain='general',
            confidence=0.60,
        )
        
        result = req.to_dict()
        assert isinstance(result, dict)
        assert result['request_type'] == 'conversational'
        assert result['domain'] == 'general'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
