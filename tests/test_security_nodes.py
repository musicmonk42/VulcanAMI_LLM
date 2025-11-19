"""
Comprehensive test suite for security_nodes.py
"""

import pytest
import json
import base64
from unittest.mock import Mock, MagicMock, patch
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

from security_nodes import (
    EncryptNode,
    PolicyNode,
    dispatch_security_node,
    SecurityNodeError,
    MAX_DATA_SIZE,
    MAX_TENSOR_ELEMENTS,
    MAX_STRING_LENGTH,
)


@pytest.fixture
def context():
    """Create test context."""
    return {
        'audit_log': [],
        'ethical_label': 'EU2025:Safe'
    }


@pytest.fixture
def encrypt_node():
    """Create EncryptNode instance."""
    with patch('security_nodes.KeyManager'), \
         patch('security_nodes.NSOAligner'), \
         patch('security_nodes.LLMCompressor', None), \
         patch('security_nodes.HardwareDispatcher', None), \
         patch('security_nodes.GrokKernelAudit', None):
        return EncryptNode(agent_id="test_agent")


@pytest.fixture
def policy_node():
    """Create PolicyNode instance."""
    with patch('security_nodes.NSOAligner'), \
         patch('security_nodes.LLMCompressor', None):
        return PolicyNode()


class TestEncryptNodeInitialization:
    """Test EncryptNode initialization."""
    
    @patch('security_nodes.KeyManager')
    @patch('security_nodes.NSOAligner')
    def test_initialization(self, mock_nso, mock_key_manager):
        """Test basic initialization."""
        node = EncryptNode(agent_id="test_agent")
        
        assert node.agent_id == "test_agent"
        mock_key_manager.assert_called_once_with("test_agent")
    
    @patch('security_nodes.KeyManager')
    @patch('security_nodes.NSOAligner')
    def test_initialization_default_agent_id(self, mock_nso, mock_key_manager):
        """Test initialization with default agent_id."""
        node = EncryptNode()
        
        assert node.agent_id == "security_node_default"


class TestDataValidation:
    """Test data validation."""
    
    def test_validate_data_none(self, encrypt_node):
        """Test validation with None data."""
        with pytest.raises(ValueError, match="cannot be None"):
            encrypt_node._validate_data(None)
    
    def test_validate_data_dict_too_large(self, encrypt_node):
        """Test validation with oversized dict."""
        large_data = {"key": "x" * (MAX_DATA_SIZE + 1)}
        
        with pytest.raises(ValueError, match="exceeds maximum size"):
            encrypt_node._validate_data(large_data)
    
    def test_validate_data_string_too_long(self, encrypt_node):
        """Test validation with oversized string."""
        large_string = "x" * (MAX_STRING_LENGTH + 1)
        
        with pytest.raises(ValueError, match="exceeds maximum length"):
            encrypt_node._validate_data(large_string)
    
    def test_validate_data_bytes_too_large(self, encrypt_node):
        """Test validation with oversized bytes."""
        large_bytes = b"x" * (MAX_DATA_SIZE + 1)
        
        with pytest.raises(ValueError, match="exceeds maximum size"):
            encrypt_node._validate_data(large_bytes)
    
    def test_validate_data_valid(self, encrypt_node):
        """Test validation with valid data."""
        # Should not raise
        encrypt_node._validate_data("valid string")
        encrypt_node._validate_data({"key": "value"})
        encrypt_node._validate_data(b"bytes")


class TestTensorValidation:
    """Test tensor validation."""
    
    def test_validate_tensor_none(self, encrypt_node):
        """Test validation with None tensor."""
        result = encrypt_node._validate_tensor(None)
        
        assert result is False
    
    def test_validate_tensor_too_many_elements(self, encrypt_node):
        """Test validation with too many elements."""
        # Create nested list with too many elements
        large_tensor = [[1] * 1000 for _ in range(1001)]
        
        result = encrypt_node._validate_tensor(large_tensor)
        
        assert result is False
    
    def test_validate_tensor_valid(self, encrypt_node):
        """Test validation with valid tensor."""
        tensor = [[1, 2], [3, 4]]
        
        result = encrypt_node._validate_tensor(tensor)
        
        assert result is True


class TestKeyRetrieval:
    """Test encryption key retrieval."""
    
    def test_get_encryption_key_not_found(self, encrypt_node):
        """Test key retrieval when key not found."""
        encrypt_node.key_manager.get_key = Mock(return_value=None)
        encrypt_node.key_manager.keys = {}
        
        with pytest.raises(ValueError, match="Key not found"):
            encrypt_node._get_encryption_key("nonexistent", "AES")
    
    def test_get_encryption_key_aes_valid(self, encrypt_node):
        """Test AES key retrieval with valid key."""
        fernet_key = Fernet.generate_key()
        encrypt_node.key_manager.get_key = Mock(return_value=fernet_key)
        
        result = encrypt_node._get_encryption_key("test_key", "AES")
        
        assert result == fernet_key
    
    def test_get_encryption_key_aes_invalid(self, encrypt_node):
        """Test AES key retrieval with invalid key."""
        invalid_key = b"invalid_key"
        encrypt_node.key_manager.get_key = Mock(return_value=invalid_key)
        
        with pytest.raises(ValueError, match="Invalid key format"):
            encrypt_node._get_encryption_key("test_key", "AES")
    
    def test_get_encryption_key_rsa(self, encrypt_node):
        """Test RSA key retrieval."""
        # Generate RSA key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        encrypt_node.key_manager.get_key = Mock(return_value=pem)
        
        result = encrypt_node._get_encryption_key("test_key", "RSA")
        
        assert result == pem
    
    def test_get_encryption_key_unsupported_algorithm(self, encrypt_node):
        """Test key retrieval with unsupported algorithm."""
        encrypt_node.key_manager.get_key = Mock(return_value=b"key")
        
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            encrypt_node._get_encryption_key("test_key", "INVALID")


class TestEncryptNodeExecution:
    """Test EncryptNode execution."""
    
    @patch('security_nodes.NSOAligner')
    @patch('security_nodes.KeyManager')
    def test_execute_aes_encryption(self, mock_key_manager, mock_nso, context):
        """Test AES encryption."""
        # Setup
        fernet_key = Fernet.generate_key()
        mock_km_instance = MagicMock()
        mock_km_instance.get_key = Mock(return_value=fernet_key)
        mock_key_manager.return_value = mock_km_instance
        
        mock_nso_instance = MagicMock()
        mock_nso_instance.multi_model_audit = Mock(return_value="safe")
        mock_nso.return_value = mock_nso_instance
        
        node = EncryptNode(agent_id="test")
        
        data = {"sensitive": "data"}
        params = {"algorithm": "AES", "key_id": "test_key"}
        
        result = node.execute(data, params, context)
        
        assert result["encrypted_data"] is not None
        assert result["audit"]["status"] == "success"
    
    @patch('security_nodes.NSOAligner')
    @patch('security_nodes.KeyManager')
    def test_execute_rsa_encryption(self, mock_key_manager, mock_nso, context):
        """Test RSA encryption."""
        # Generate RSA key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        mock_km_instance = MagicMock()
        mock_km_instance.get_key = Mock(return_value=pem)
        mock_key_manager.return_value = mock_km_instance
        
        mock_nso_instance = MagicMock()
        mock_nso_instance.multi_model_audit = Mock(return_value="safe")
        mock_nso.return_value = mock_nso_instance
        
        node = EncryptNode(agent_id="test")
        
        data = "short string"
        params = {"algorithm": "RSA", "key_id": "test_key"}
        
        result = node.execute(data, params, context)
        
        assert result["encrypted_data"] is not None
        assert result["audit"]["status"] == "success"
    
    def test_execute_rsa_data_too_large(self, encrypt_node, context):
        """Test RSA encryption with data too large."""
        # Generate valid RSA key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        encrypt_node.key_manager.get_key = Mock(return_value=pem)
        
        large_data = "x" * 200  # Larger than RSA limit
        params = {"algorithm": "RSA", "key_id": "test_key"}
        
        with pytest.raises(ValueError, match="Data too large for RSA"):
            encrypt_node.execute(large_data, params, context)
    
    def test_execute_missing_key_id(self, encrypt_node, context):
        """Test execution without key_id."""
        params = {"algorithm": "AES"}
        
        with pytest.raises(ValueError, match="Missing key_id"):
            encrypt_node.execute("data", params, context)
    
    def test_execute_invalid_data(self, encrypt_node, context):
        """Test execution with invalid data."""
        params = {"algorithm": "AES", "key_id": "test"}
        
        with pytest.raises(ValueError):
            encrypt_node.execute(None, params, context)


class TestPolicyNodeExecution:
    """Test PolicyNode execution."""
    
    @patch('security_nodes.NSOAligner')
    def test_execute_gdpr_compliant(self, mock_nso, policy_node, context):
        """Test GDPR policy with compliant data."""
        data = {"name": "John", "age": 30}
        params = {"policy": "GDPR", "enforcement": "log"}
        
        result = policy_node.execute(data, params, context)
        
        assert result["compliance"] == "compliant"
        assert result["audit"]["status"] == "success"
    
    @patch('security_nodes.NSOAligner')
    def test_execute_gdpr_pii_detected(self, mock_nso, policy_node, context):
        """Test GDPR policy with PII."""
        data = {"email": "test@example.com"}
        params = {"policy": "GDPR", "enforcement": "restrict"}
        
        with pytest.raises(ValueError, match="Non-compliant"):
            policy_node.execute(data, params, context)
    
    @patch('security_nodes.NSOAligner')
    def test_execute_ccpa(self, mock_nso, policy_node, context):
        """Test CCPA policy."""
        data = {"data": "clean"}
        params = {"policy": "CCPA", "enforcement": "log"}
        
        result = policy_node.execute(data, params, context)
        
        assert result["compliance"] == "compliant"
    
    @patch('security_nodes.NSOAligner')
    def test_execute_itu_f_748_47(self, mock_nso, policy_node, context):
        """Test ITU F.748.47 policy."""
        mock_nso_instance = MagicMock()
        mock_nso_instance.multi_model_audit = Mock(return_value="safe")
        mock_nso.return_value = mock_nso_instance
        
        policy_node.nso_aligner = mock_nso_instance
        
        data = {"ethical": "content"}
        params = {"policy": "ITU-F.748.47", "enforcement": "log"}
        
        result = policy_node.execute(data, params, context)
        
        assert result["compliance"] == "compliant"
    
    def test_execute_unsupported_policy(self, policy_node, context):
        """Test unsupported policy."""
        params = {"policy": "INVALID", "enforcement": "log"}
        
        with pytest.raises(ValueError, match="Unsupported policy"):
            policy_node.execute("data", params, context)
    
    def test_execute_none_data(self, policy_node, context):
        """Test execution with None data."""
        params = {"policy": "GDPR"}
        
        with pytest.raises(ValueError, match="cannot be None"):
            policy_node.execute(None, params, context)


class TestDispatchSecurityNode:
    """Test dispatch_security_node function."""
    
    @patch('security_nodes.EncryptNode')
    def test_dispatch_encrypt_node(self, mock_encrypt_class, context):
        """Test dispatching EncryptNode."""
        mock_instance = MagicMock()
        mock_instance.execute = Mock(return_value={"result": "encrypted"})
        mock_encrypt_class.return_value = mock_instance
        
        node = {
            "type": "EncryptNode",
            "agent_id": "test_agent",
            "params": {"algorithm": "AES", "key_id": "key1"}
        }
        
        result = dispatch_security_node(node, "data", context)
        
        assert result == {"result": "encrypted"}
        mock_encrypt_class.assert_called_once_with("test_agent")
    
    @patch('security_nodes.PolicyNode')
    def test_dispatch_policy_node(self, mock_policy_class, context):
        """Test dispatching PolicyNode."""
        mock_instance = MagicMock()
        mock_instance.execute = Mock(return_value={"compliance": "passed"})
        mock_policy_class.return_value = mock_instance
        
        node = {
            "type": "PolicyNode",
            "params": {"policy": "GDPR"}
        }
        
        result = dispatch_security_node(node, "data", context)
        
        assert result == {"compliance": "passed"}
    
    def test_dispatch_unknown_type(self, context):
        """Test dispatching unknown node type."""
        node = {"type": "UnknownNode", "params": {}}
        
        with pytest.raises(ValueError, match="Unknown security node type"):
            dispatch_security_node(node, "data", context)
    
    @patch('security_nodes.EncryptNode')
    def test_dispatch_context_isolation(self, mock_encrypt_class, context):
        """Test that context is properly isolated."""
        mock_instance = MagicMock()
        mock_instance.execute = Mock(return_value={"result": "ok"})
        mock_encrypt_class.return_value = mock_instance
        
        node = {
            "type": "EncryptNode",
            "params": {},
            "tensor": [[1, 2]],
            "kernel": "code"
        }
        
        dispatch_security_node(node, "data", context)
        
        # Original context should not be polluted with node-specific data
        assert "tensor" not in context
        assert "kernel" not in context


class TestConstants:
    """Test module constants."""
    
    def test_constants_exist(self):
        """Test that all constants are defined."""
        assert MAX_DATA_SIZE > 0
        assert MAX_TENSOR_ELEMENTS > 0
        assert MAX_STRING_LENGTH > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])