"""
Comprehensive test suite for agent_registry.py
"""

import json
import shutil
import tempfile
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from agent_registry import (AGENT_ID_MAX_LENGTH, LOCKOUT_DURATION,
                            MAX_FAILED_ATTEMPTS, AgentCertificate, AgentKey,
                            AgentProfile, AgentRegistry, AgentRole,
                            AuditLogger, CalibrationData, CertificateAuthority,
                            DatabaseConnectionPool, KeyAlgorithm, KeyManager,
                            RateLimiter, RegistryEvent)


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def key_manager(temp_dir):
    """Create key manager."""
    return KeyManager(str(temp_dir / "keys"))


@pytest.fixture
def cert_authority(temp_dir):
    """Create certificate authority."""
    return CertificateAuthority(
        ca_key_path=str(temp_dir / "ca_key.pem"),
        ca_cert_path=str(temp_dir / "ca_cert.pem")
    )


@pytest.fixture
def audit_logger(temp_dir):
    """Create audit logger."""
    return AuditLogger(str(temp_dir / "audit_logs"))


@pytest.fixture
def rate_limiter():
    """Create rate limiter."""
    return RateLimiter(window_seconds=1, max_requests=10)


@pytest.fixture
def agent_registry(temp_dir):
    """Create agent registry."""
    registry = AgentRegistry(
        registry_file=str(temp_dir / "registry.db"),
        key_store_dir=str(temp_dir / "keys"),
        audit_log_dir=str(temp_dir / "audit"),
        ca_key_path=str(temp_dir / "ca_key.pem"),
        ca_cert_path=str(temp_dir / "ca_cert.pem")
    )
    yield registry
    registry.shutdown_registry()


class TestDatabaseConnectionPool:
    """Test database connection pool."""

    def test_pool_creation(self, temp_dir):
        """Test pool creates connections."""
        db_path = temp_dir / "test.db"
        pool = DatabaseConnectionPool(db_path, pool_size=3)

        assert len(pool.connections) == 3
        assert not pool.closed

        pool.close_all()

    def test_get_connection(self, temp_dir):
        """Test getting connection from pool."""
        db_path = temp_dir / "test.db"
        pool = DatabaseConnectionPool(db_path, pool_size=2)

        with pool.get_connection() as conn:
            assert conn is not None
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            assert result[0] == 1

        pool.close_all()

    def test_pool_closed_raises(self, temp_dir):
        """Test accessing closed pool raises error."""
        db_path = temp_dir / "test.db"
        pool = DatabaseConnectionPool(db_path, pool_size=2)
        pool.close_all()

        with pytest.raises(RuntimeError, match="Connection pool is closed"):
            with pool.get_connection():
                pass


class TestKeyManager:
    """Test key manager."""

    def test_generate_rsa_2048(self, key_manager):
        """Test RSA 2048 key generation."""
        public, private = key_manager.generate_key_pair(KeyAlgorithm.RSA_2048)

        assert b"BEGIN PUBLIC KEY" in public
        assert b"BEGIN PRIVATE KEY" in private

    def test_generate_ed25519(self, key_manager):
        """Test ED25519 key generation."""
        public, private = key_manager.generate_key_pair(KeyAlgorithm.ED25519)

        assert b"BEGIN PUBLIC KEY" in public
        assert b"BEGIN PRIVATE KEY" in private

    def test_generate_ecdsa(self, key_manager):
        """Test ECDSA key generation."""
        public, private = key_manager.generate_key_pair(KeyAlgorithm.ECDSA_P256)

        assert b"BEGIN PUBLIC KEY" in public
        assert b"BEGIN PRIVATE KEY" in private

    def test_sign_and_verify_rsa(self, key_manager):
        """Test RSA signing and verification."""
        public, private = key_manager.generate_key_pair(KeyAlgorithm.RSA_2048)
        message = b"Test message for signing"

        signature = key_manager.sign_message(message, private, KeyAlgorithm.RSA_2048)

        assert key_manager.verify_signature(message, signature, public, KeyAlgorithm.RSA_2048)
        assert not key_manager.verify_signature(b"Different message", signature, public, KeyAlgorithm.RSA_2048)

    def test_sign_and_verify_ed25519(self, key_manager):
        """Test ED25519 signing and verification."""
        public, private = key_manager.generate_key_pair(KeyAlgorithm.ED25519)
        message = b"Test message for signing"

        signature = key_manager.sign_message(message, private, KeyAlgorithm.ED25519)

        assert key_manager.verify_signature(message, signature, public, KeyAlgorithm.ED25519)
        assert not key_manager.verify_signature(b"Different message", signature, public, KeyAlgorithm.ED25519)

    def test_encrypt_decrypt_key(self, key_manager):
        """Test key encryption and decryption."""
        key_data = b"Secret key data"
        password = "strong_password_123"

        encrypted, salt_iv = key_manager.encrypt_key(key_data, password)
        decrypted = key_manager.decrypt_key(encrypted, password, salt_iv)

        assert decrypted == key_data

    def test_encrypt_wrong_password(self, key_manager):
        """Test decryption with wrong password fails."""
        key_data = b"Secret key data"
        password = "correct_password"
        wrong_password = "wrong_password"

        encrypted, salt_iv = key_manager.encrypt_key(key_data, password)

        with pytest.raises(Exception):
            key_manager.decrypt_key(encrypted, wrong_password, salt_iv)

    def test_pkcs7_padding(self, key_manager):
        """Test PKCS#7 padding and unpadding."""
        data = b"Test data"
        padded = key_manager._pkcs7_pad(data)

        assert len(padded) % 16 == 0
        assert len(padded) > len(data)

        unpadded = key_manager._pkcs7_unpad(padded)
        assert unpadded == data


class TestCertificateAuthority:
    """Test certificate authority."""

    def test_ca_generation(self, cert_authority):
        """Test CA generation."""
        assert cert_authority.ca_key is not None
        assert cert_authority.ca_cert is not None

    def test_ca_save_load(self, temp_dir):
        """Test CA save and load."""
        ca1 = CertificateAuthority()

        key_path = temp_dir / "test_ca_key.pem"
        cert_path = temp_dir / "test_ca_cert.pem"

        ca1.save_ca(str(key_path), str(cert_path))

        assert key_path.exists()
        assert cert_path.exists()

        ca2 = CertificateAuthority(str(key_path), str(cert_path))
        assert ca2.ca_key is not None
        assert ca2.ca_cert is not None

    def test_issue_certificate(self, cert_authority, key_manager):
        """Test certificate issuance."""
        public_key, _ = key_manager.generate_key_pair(KeyAlgorithm.RSA_2048)

        cert = cert_authority.issue_certificate("test_agent", public_key, valid_days=30)

        assert cert.cert_id is not None
        assert cert.subject is not None
        assert cert.issuer is not None
        assert not cert.is_revoked

    def test_verify_certificate(self, cert_authority, key_manager):
        """Test certificate verification."""
        public_key, _ = key_manager.generate_key_pair(KeyAlgorithm.RSA_2048)

        cert = cert_authority.issue_certificate("test_agent", public_key)

        assert cert_authority.verify_certificate(cert.certificate)

    def test_verify_invalid_certificate(self, cert_authority):
        """Test verification of invalid certificate."""
        fake_cert = b"-----BEGIN CERTIFICATE-----\nFAKE\n-----END CERTIFICATE-----"

        assert not cert_authority.verify_certificate(fake_cert)


class TestAuditLogger:
    """Test audit logger."""

    def test_log_event(self, audit_logger):
        """Test logging an event."""
        audit_logger.log_event(
            RegistryEvent.AGENT_REGISTERED,
            "test_agent",
            {"name": "Test Agent"},
            success=True,
            ip_address="127.0.0.1"
        )

        events = audit_logger.get_recent_events(count=1)
        assert len(events) == 1
        assert events[0]["agent_id"] == "test_agent"
        assert events[0]["event"] == RegistryEvent.AGENT_REGISTERED.value

    def test_search_by_agent(self, audit_logger):
        """Test searching events by agent ID."""
        audit_logger.log_event(RegistryEvent.AGENT_REGISTERED, "agent1", {})
        audit_logger.log_event(RegistryEvent.AGENT_REGISTERED, "agent2", {})
        audit_logger.log_event(RegistryEvent.AUTH_SUCCESS, "agent1", {})

        results = audit_logger.search_events(agent_id="agent1")
        assert len(results) == 2
        assert all(e["agent_id"] == "agent1" for e in results)

    def test_search_by_event_type(self, audit_logger):
        """Test searching events by type."""
        audit_logger.log_event(RegistryEvent.AGENT_REGISTERED, "agent1", {})
        audit_logger.log_event(RegistryEvent.AUTH_SUCCESS, "agent1", {})
        audit_logger.log_event(RegistryEvent.AUTH_FAILURE, "agent2", {})

        results = audit_logger.search_events(event_type=RegistryEvent.AUTH_SUCCESS)
        assert len(results) == 1
        assert results[0]["event"] == RegistryEvent.AUTH_SUCCESS.value

    def test_search_by_time_range(self, audit_logger):
        """Test searching events by time range."""
        start = datetime.utcnow()

        audit_logger.log_event(RegistryEvent.AGENT_REGISTERED, "agent1", {})
        time.sleep(0.1)
        middle = datetime.utcnow()
        time.sleep(0.1)
        audit_logger.log_event(RegistryEvent.AGENT_REGISTERED, "agent2", {})

        results = audit_logger.search_events(start_time=start, end_time=middle)
        assert len(results) == 1


class TestRateLimiter:
    """Test rate limiter."""

    def test_rate_limit_allows(self, rate_limiter):
        """Test rate limiter allows requests."""
        assert rate_limiter.is_allowed("test_id")

    def test_rate_limit_blocks(self, rate_limiter):
        """Test rate limiter blocks excessive requests."""
        identifier = "test_id"

        # Use up all tokens
        for _ in range(10):
            assert rate_limiter.is_allowed(identifier)

        # Should be blocked
        assert not rate_limiter.is_allowed(identifier)

    def test_rate_limit_reset(self, rate_limiter):
        """Test rate limiter reset."""
        identifier = "test_id"

        for _ in range(10):
            rate_limiter.is_allowed(identifier)

        rate_limiter.reset(identifier)

        assert rate_limiter.is_allowed(identifier)

    def test_rate_limit_cleanup(self, rate_limiter):
        """Test rate limiter cleanup."""
        # Add some entries
        for i in range(5):
            rate_limiter.is_allowed(f"id_{i}")

        # Mock old timestamps
        for identifier in rate_limiter.requests:
            rate_limiter.requests[identifier].clear()

        rate_limiter.cleanup()

        # Should have removed empty entries
        assert len(rate_limiter.requests) == 0


class TestAgentRegistry:
    """Test agent registry."""

    def test_register_agent(self, agent_registry):
        """Test agent registration."""
        result = agent_registry.register_agent(
            agent_id="test_agent_001",
            name="Test Agent",
            roles=[AgentRole.EXECUTOR],
            algorithm=KeyAlgorithm.ED25519,
            issue_certificate=True
        )

        assert result["agent_id"] == "test_agent_001"
        assert "public_key" in result
        assert "private_key" in result
        assert "certificate" in result

    def test_register_duplicate_agent(self, agent_registry):
        """Test registering duplicate agent fails."""
        agent_registry.register_agent(
            agent_id="duplicate_agent",
            name="Agent",
            roles=[AgentRole.EXECUTOR]
        )

        with pytest.raises(ValueError, match="already registered"):
            agent_registry.register_agent(
                agent_id="duplicate_agent",
                name="Agent",
                roles=[AgentRole.EXECUTOR]
            )

    def test_register_invalid_agent_id(self, agent_registry):
        """Test registering with invalid agent ID."""
        with pytest.raises(ValueError, match="Invalid agent ID"):
            agent_registry.register_agent(
                agent_id="invalid@agent!id",
                name="Agent",
                roles=[AgentRole.EXECUTOR]
            )

    def test_register_too_long_agent_id(self, agent_registry):
        """Test registering with too long agent ID."""
        long_id = "a" * (AGENT_ID_MAX_LENGTH + 1)

        with pytest.raises(ValueError, match="Invalid agent ID"):
            agent_registry.register_agent(
                agent_id=long_id,
                name="Agent",
                roles=[AgentRole.EXECUTOR]
            )

    def test_verify_signature_valid(self, agent_registry):
        """Test signature verification with valid signature."""
        result = agent_registry.register_agent(
            agent_id="signer_agent",
            name="Signer",
            roles=[AgentRole.EXECUTOR],
            algorithm=KeyAlgorithm.ED25519
        )

        import base64
        private_key = base64.b64decode(result["private_key"])

        message = "Test message"
        signature = agent_registry.key_manager.sign_message(
            message.encode(),
            private_key,
            KeyAlgorithm.ED25519
        )
        signature_b64 = base64.b64encode(signature).decode()

        is_valid = agent_registry.verify_signature(
            agent_id="signer_agent",
            message=message,
            signature=signature_b64
        )

        assert is_valid

    def test_verify_signature_invalid(self, agent_registry):
        """Test signature verification with invalid signature."""
        agent_registry.register_agent(
            agent_id="signer_agent",
            name="Signer",
            roles=[AgentRole.EXECUTOR],
            algorithm=KeyAlgorithm.ED25519
        )

        is_valid = agent_registry.verify_signature(
            agent_id="signer_agent",
            message="Test message",
            signature="invalid_signature_base64=="
        )

        assert not is_valid

    def test_verify_signature_wrong_message(self, agent_registry):
        """Test signature verification with wrong message."""
        result = agent_registry.register_agent(
            agent_id="signer_agent",
            name="Signer",
            roles=[AgentRole.EXECUTOR],
            algorithm=KeyAlgorithm.ED25519
        )

        import base64
        private_key = base64.b64decode(result["private_key"])

        message = "Original message"
        signature = agent_registry.key_manager.sign_message(
            message.encode(),
            private_key,
            KeyAlgorithm.ED25519
        )
        signature_b64 = base64.b64encode(signature).decode()

        is_valid = agent_registry.verify_signature(
            agent_id="signer_agent",
            message="Different message",
            signature=signature_b64
        )

        assert not is_valid

    def test_lockout_after_failed_attempts(self, agent_registry):
        """Test agent lockout after failed attempts."""
        agent_registry.register_agent(
            agent_id="lockout_agent",
            name="Lockout Test",
            roles=[AgentRole.EXECUTOR]
        )

        # Make MAX_FAILED_ATTEMPTS failed attempts
        for _ in range(MAX_FAILED_ATTEMPTS):
            agent_registry.verify_signature(
                agent_id="lockout_agent",
                message="test",
                signature="invalid"
            )

        # Agent should be locked
        agent = agent_registry.agents.get("lockout_agent")
        assert agent.is_locked()

    def test_rotate_key(self, agent_registry):
        """Test key rotation."""
        agent_registry.register_agent(
            agent_id="rotate_agent",
            name="Rotate Test",
            roles=[AgentRole.EXECUTOR]
        )

        result = agent_registry.rotate_key("rotate_agent", KeyAlgorithm.RSA_2048)

        assert "key_id" in result
        assert "public_key" in result
        assert "private_key" in result

    def test_revoke_key(self, agent_registry):
        """Test key revocation."""
        result = agent_registry.register_agent(
            agent_id="revoke_agent",
            name="Revoke Test",
            roles=[AgentRole.EXECUTOR]
        )

        key_id = result["key_id"]

        agent_registry.revoke_key("revoke_agent", key_id, reason="testing")

        assert key_id in agent_registry.revoked_keys

    def test_grant_permission(self, agent_registry):
        """Test granting permission."""
        agent_registry.register_agent(
            agent_id="perm_agent",
            name="Permission Test",
            roles=[AgentRole.VIEWER]
        )

        agent_registry.grant_permission("perm_agent", "execute_graph", "admin")

        assert agent_registry.check_permission("perm_agent", "execute_graph")

    def test_revoke_permission(self, agent_registry):
        """Test revoking permission."""
        agent_registry.register_agent(
            agent_id="perm_agent",
            name="Permission Test",
            roles=[AgentRole.EXECUTOR]
        )

        # Should have execute permission by default
        assert agent_registry.check_permission("perm_agent", "execute_graph")

        agent_registry.revoke_permission("perm_agent", "execute_graph", "admin")

        assert not agent_registry.check_permission("perm_agent", "execute_graph")

    def test_admin_has_all_permissions(self, agent_registry):
        """Test admin role has all permissions."""
        agent_registry.register_agent(
            agent_id="admin_agent",
            name="Admin",
            roles=[AgentRole.ADMIN]
        )

        assert agent_registry.check_permission("admin_agent", "execute_graph")
        assert agent_registry.check_permission("admin_agent", "manage_agents")
        assert agent_registry.check_permission("admin_agent", "view_audit")

    def test_get_agent_info(self, agent_registry):
        """Test getting agent info."""
        agent_registry.register_agent(
            agent_id="info_agent",
            name="Info Test",
            roles=[AgentRole.EXECUTOR]
        )

        info = agent_registry.get_agent_info("info_agent")

        assert info is not None
        assert info["agent_id"] == "info_agent"
        assert info["name"] == "Info Test"
        assert AgentRole.EXECUTOR.value in info["roles"]

    def test_list_agents(self, agent_registry):
        """Test listing agents."""
        agent_registry.register_agent(
            agent_id="agent1",
            name="Agent 1",
            roles=[AgentRole.EXECUTOR]
        )
        agent_registry.register_agent(
            agent_id="agent2",
            name="Agent 2",
            roles=[AgentRole.ADMIN]
        )

        all_agents = agent_registry.list_agents()
        assert len(all_agents) == 2

        admins = agent_registry.list_agents(role=AgentRole.ADMIN)
        assert len(admins) == 1
        assert admins[0]["agent_id"] == "agent2"

    def test_export_ca_certificate(self, agent_registry):
        """Test exporting CA certificate."""
        ca_cert = agent_registry.export_ca_certificate()

        assert "BEGIN CERTIFICATE" in ca_cert
        assert isinstance(ca_cert, str)

    def test_get_audit_logs(self, agent_registry):
        """Test getting audit logs."""
        agent_registry.register_agent(
            agent_id="audit_agent",
            name="Audit Test",
            roles=[AgentRole.EXECUTOR]
        )

        logs = agent_registry.get_audit_logs(limit=10)

        assert len(logs) > 0
        assert any(log["event"] == RegistryEvent.AGENT_REGISTERED.value for log in logs)


class TestThreadSafety:
    """Test thread safety of registry."""

    def test_concurrent_registration(self, agent_registry):
        """Test concurrent agent registration."""
        results = []
        errors = []

        def register(agent_id):
            try:
                result = agent_registry.register_agent(
                    agent_id=agent_id,
                    name=f"Agent {agent_id}",
                    roles=[AgentRole.EXECUTOR]
                )
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=register, args=(f"agent_{i}",))
            for i in range(10)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10
        assert len(errors) == 0

    def test_concurrent_signature_verification(self, agent_registry):
        """Test concurrent signature verification."""
        result = agent_registry.register_agent(
            agent_id="verify_agent",
            name="Verify Test",
            roles=[AgentRole.EXECUTOR],
            algorithm=KeyAlgorithm.ED25519
        )

        import base64
        private_key = base64.b64decode(result["private_key"])

        message = "Test message"
        signature = agent_registry.key_manager.sign_message(
            message.encode(),
            private_key,
            KeyAlgorithm.ED25519
        )
        signature_b64 = base64.b64encode(signature).decode()

        results = []

        def verify():
            is_valid = agent_registry.verify_signature(
                agent_id="verify_agent",
                message=message,
                signature=signature_b64
            )
            results.append(is_valid)

        threads = [threading.Thread(target=verify) for _ in range(20)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(results)  # All should be valid


class TestAgentKey:
    """Test AgentKey dataclass."""

    def test_key_expiration(self):
        """Test key expiration check."""
        expired_key = AgentKey(
            key_id="test_key",
            algorithm=KeyAlgorithm.ED25519,
            public_key=b"public",
            expires_at=datetime.utcnow() - timedelta(hours=1)
        )

        assert expired_key.is_expired()

        valid_key = AgentKey(
            key_id="test_key",
            algorithm=KeyAlgorithm.ED25519,
            public_key=b"public",
            expires_at=datetime.utcnow() + timedelta(hours=1)
        )

        assert not valid_key.is_expired()

    def test_key_serialization(self):
        """Test key to/from dict."""
        key = AgentKey(
            key_id="test_key",
            algorithm=KeyAlgorithm.ED25519,
            public_key=b"public_key_data",
            private_key=b"private_key_data"
        )

        data = key.to_dict()
        restored = AgentKey.from_dict(data)

        assert restored.key_id == key.key_id
        assert restored.algorithm == key.algorithm
        assert restored.public_key == key.public_key


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
