"""
Tests for API model deprecation and conversion.

This test validates that:
1. ChatRequest shows deprecation warning
2. ChatRequest.to_unified() converts correctly
3. UnifiedChatRequest remains the standard model
"""

import warnings
import pytest
from vulcan.api.models import ChatRequest, UnifiedChatRequest, ChatHistoryMessage


def test_chat_request_deprecation_warning():
    """Test that ChatRequest shows deprecation warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        request = ChatRequest(prompt="Hello", max_tokens=1000)
        
        # Check that deprecation warning was raised
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "ChatRequest is deprecated" in str(w[0].message)
        assert "Use UnifiedChatRequest instead" in str(w[0].message)
        
        # Check that the request was created correctly
        assert request.prompt == "Hello"
        assert request.max_tokens == 1000


def test_chat_request_to_unified_conversion():
    """Test that ChatRequest.to_unified() converts correctly."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Suppress warning for this test
        
        chat_request = ChatRequest(prompt="Test prompt", max_tokens=2500)
        unified_request = chat_request.to_unified()
        
        # Check conversion
        assert isinstance(unified_request, UnifiedChatRequest)
        assert unified_request.message == "Test prompt"
        assert unified_request.max_tokens == 2500
        assert unified_request.history == []
        assert unified_request.conversation_id is None
        
        # Check feature toggles default to True
        assert unified_request.enable_reasoning is True
        assert unified_request.enable_memory is True
        assert unified_request.enable_safety is True
        assert unified_request.enable_planning is True
        assert unified_request.enable_causal is True


def test_unified_chat_request_standard():
    """Test that UnifiedChatRequest works as the standard model."""
    request = UnifiedChatRequest(
        message="Explain quantum entanglement",
        max_tokens=1500,
        history=[
            ChatHistoryMessage(role="user", content="Hi"),
            ChatHistoryMessage(role="assistant", content="Hello!")
        ],
        conversation_id="conv_123",
        enable_reasoning=False,
        enable_memory=True
    )
    
    assert request.message == "Explain quantum entanglement"
    assert request.max_tokens == 1500
    assert len(request.history) == 2
    assert request.history[0].role == "user"
    assert request.history[0].content == "Hi"
    assert request.conversation_id == "conv_123"
    assert request.enable_reasoning is False
    assert request.enable_memory is True


def test_chat_history_message_validation():
    """Test ChatHistoryMessage field validation."""
    # Valid message
    msg = ChatHistoryMessage(role="user", content="Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"
    
    # Invalid role should raise ValidationError
    with pytest.raises(Exception):  # Pydantic ValidationError
        ChatHistoryMessage(role="invalid", content="Hello")
    
    # Empty content should raise ValidationError
    with pytest.raises(Exception):  # Pydantic ValidationError
        ChatHistoryMessage(role="user", content="")


def test_unified_chat_request_defaults():
    """Test UnifiedChatRequest default values."""
    request = UnifiedChatRequest(message="Test")
    
    assert request.message == "Test"
    assert request.max_tokens == 2000
    assert request.history == []
    assert request.conversation_id is None
    assert request.enable_reasoning is True
    assert request.enable_memory is True
    assert request.enable_safety is True
    assert request.enable_planning is True
    assert request.enable_causal is True


def test_unified_chat_request_max_tokens_validation():
    """Test UnifiedChatRequest max_tokens validation."""
    # Valid values
    UnifiedChatRequest(message="Test", max_tokens=1)
    UnifiedChatRequest(message="Test", max_tokens=32000)
    
    # Invalid values should raise ValidationError
    with pytest.raises(Exception):  # Pydantic ValidationError
        UnifiedChatRequest(message="Test", max_tokens=0)
    
    with pytest.raises(Exception):  # Pydantic ValidationError
        UnifiedChatRequest(message="Test", max_tokens=32001)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
