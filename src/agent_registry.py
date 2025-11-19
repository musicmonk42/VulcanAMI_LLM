# src/agent_registry.py
"""
Graphix IR Agent Registry (Production-Ready with Full Cryptographic Security)
Version: 2.0.3 - Base64 decoding error handling fixed
=============================================================================
A comprehensive registry for managing agent identities with proper cryptographic
signatures, key management, certificate support, and enterprise security features.
"""

import json
import logging
import hashlib
import hmac
import time
import os
import secrets
import threading
import sqlite3
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any, Union, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from contextlib import contextmanager
import tempfile

# Cryptographic imports
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding, ed25519, ec
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.x509 import load_pem_x509_certificate, Certificate
from cryptography.x509.oid import NameOID
from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import constant_time

try:
    from vulcan_integration import VulcanGraphixBridge
    VULCAN_INTEGRATION_AVAILABLE = True
except ImportError:
    VULCAN_INTEGRATION_AVAILABLE = False
    VulcanGraphixBridge = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AgentRegistry")

# Security constants
MIN_KEY_SIZE = 2048
MAX_KEY_AGE_DAYS = 365
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_DURATION = 300  # 5 minutes
RATE_LIMIT_WINDOW = 60  # 1 minute
RATE_LIMIT_MAX_REQUESTS = 100
AUDIT_LOG_MAX_SIZE = 10000
KEY_DERIVATION_ITERATIONS = 100000
AGENT_ID_MAX_LENGTH = 128
PERMISSION_NAME_MAX_LENGTH = 64


class KeyAlgorithm(Enum):
    """Supported cryptographic algorithms."""
    RSA_2048 = "rsa_2048"
    RSA_4096 = "rsa_4096"
    ED25519 = "ed25519"
    ECDSA_P256 = "ecdsa_p256"
    ECDSA_P384 = "ecdsa_p384"
    ECDSA_P521 = "ecdsa_p521"


class AgentRole(Enum):
    """Agent roles for access control."""
    VIEWER = "viewer"
    EXECUTOR = "executor"
    ADMIN = "admin"
    AUDITOR = "auditor"
    DEVELOPER = "developer"
    SERVICE = "service"


class RegistryEvent(Enum):
    """Registry audit events."""
    AGENT_REGISTERED = "agent_registered"
    AGENT_UPDATED = "agent_updated"
    AGENT_REVOKED = "agent_revoked"
    AUTH_SUCCESS = "auth_success"
    AUTH_FAILURE = "auth_failure"
    KEY_ROTATED = "key_rotated"
    CERT_ISSUED = "cert_issued"
    CERT_REVOKED = "cert_revoked"
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_REVOKED = "permission_revoked"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"


@dataclass
class CalibrationData:
    """Placeholder for calibration data (for test compatibility)."""
    pass


@dataclass
class AgentKey:
    """Represents a cryptographic key for an agent."""
    key_id: str
    algorithm: KeyAlgorithm
    public_key: bytes
    private_key: Optional[bytes] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if key is expired."""
        if self.expires_at:
            return datetime.utcnow() > self.expires_at
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        import base64
        return {
            "key_id": self.key_id,
            "algorithm": self.algorithm.value,
            "public_key": base64.b64encode(self.public_key).decode('utf-8'),
            "private_key": base64.b64encode(self.private_key).decode('utf-8') if self.private_key else None,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "is_active": self.is_active,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentKey':
        """Create from dictionary."""
        import base64
        return cls(
            key_id=data["key_id"],
            algorithm=KeyAlgorithm(data["algorithm"]),
            public_key=base64.b64decode(data["public_key"]),
            private_key=base64.b64decode(data["private_key"]) if data.get("private_key") else None,
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            is_active=data.get("is_active", True),
            metadata=data.get("metadata", {})
        )


@dataclass
class AgentCertificate:
    """X.509 certificate for an agent."""
    cert_id: str
    certificate: bytes
    issuer: str
    subject: str
    serial_number: str
    not_valid_before: datetime
    not_valid_after: datetime
    is_revoked: bool = False
    revocation_reason: Optional[str] = None
    revocation_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        import base64
        return {
            "cert_id": self.cert_id,
            "certificate": base64.b64encode(self.certificate).decode('utf-8'),
            "issuer": self.issuer,
            "subject": self.subject,
            "serial_number": self.serial_number,
            "not_valid_before": self.not_valid_before.isoformat(),
            "not_valid_after": self.not_valid_after.isoformat(),
            "is_revoked": self.is_revoked,
            "revocation_reason": self.revocation_reason,
            "revocation_time": self.revocation_time.isoformat() if self.revocation_time else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentCertificate':
        """Create from dictionary."""
        import base64
        return cls(
            cert_id=data["cert_id"],
            certificate=base64.b64decode(data["certificate"]),
            issuer=data["issuer"],
            subject=data["subject"],
            serial_number=data["serial_number"],
            not_valid_before=datetime.fromisoformat(data["not_valid_before"]),
            not_valid_after=datetime.fromisoformat(data["not_valid_after"]),
            is_revoked=data.get("is_revoked", False),
            revocation_reason=data.get("revocation_reason"),
            revocation_time=datetime.fromisoformat(data["revocation_time"]) if data.get("revocation_time") else None
        )


@dataclass
class AgentProfile:
    """Complete profile for a registered agent."""
    agent_id: str
    name: str
    roles: List[AgentRole]
    keys: List[AgentKey]
    certificates: List[AgentCertificate] = field(default_factory=list)
    permissions: Dict[str, bool] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_seen: Optional[datetime] = None
    is_active: bool = True
    failed_attempts: int = 0
    locked_until: Optional[datetime] = None
    
    def get_active_key(self) -> Optional[AgentKey]:
        """Get the current active key that is not expired."""
        for key in self.keys:
            if key.is_active and not key.is_expired():
                return key
        return None
    
    def is_locked(self) -> bool:
        """Check if agent is locked out."""
        if self.locked_until is None:
            return False
        return datetime.utcnow() < self.locked_until
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "roles": [r.value for r in self.roles],
            "keys": [k.to_dict() for k in self.keys],
            "certificates": [c.to_dict() for c in self.certificates],
            "permissions": self.permissions,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "is_active": self.is_active,
            "failed_attempts": self.failed_attempts,
            "locked_until": self.locked_until.isoformat() if self.locked_until else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentProfile':
        """Create from dictionary."""
        return cls(
            agent_id=data["agent_id"],
            name=data["name"],
            roles=[AgentRole(r) for r in data["roles"]],
            keys=[AgentKey.from_dict(k) for k in data["keys"]],
            certificates=[AgentCertificate.from_dict(c) for c in data.get("certificates", [])],
            permissions=data.get("permissions", {}),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_seen=datetime.fromisoformat(data["last_seen"]) if data.get("last_seen") else None,
            is_active=data.get("is_active", True),
            failed_attempts=data.get("failed_attempts", 0),
            locked_until=datetime.fromisoformat(data["locked_until"]) if data.get("locked_until") else None
        )


class DatabaseConnectionPool:
    """Thread-safe database connection pool."""
    
    def __init__(self, db_path: Path, pool_size: int = 5):
        self.db_path = db_path
        self.pool_size = pool_size
        self.connections = []
        self.available = threading.Semaphore(pool_size)
        self.lock = threading.RLock()
        self.closed = False
        
        for _ in range(pool_size):
            conn = sqlite3.connect(str(db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self.connections.append(conn)
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool."""
        self.available.acquire()
        try:
            with self.lock:
                if self.closed:
                    raise RuntimeError("Connection pool is closed")
                conn = self.connections.pop()
            yield conn
        finally:
            with self.lock:
                if not self.closed:
                    self.connections.append(conn)
            self.available.release()
    
    def close_all(self):
        """Close all connections."""
        with self.lock:
            self.closed = True
            for conn in self.connections:
                try:
                    conn.close()
                except sqlite3.Error as e:
                    logger.error(f"Error closing connection: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error closing connection: {e}")
            self.connections.clear()


class AuditLogger:
    """Handles audit logging for the registry."""
    
    def __init__(self, log_dir: str = "audit_logs", max_size: int = AUDIT_LOG_MAX_SIZE):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self.current_log = deque(maxlen=max_size)
        self.lock = threading.RLock()
        self.current_date = datetime.utcnow().strftime('%Y%m%d')
        self.log_file = self.log_dir / f"audit_{self.current_date}.jsonl"
        self._log_write_lock = threading.Lock()
        
    def log_event(self, event: RegistryEvent, agent_id: str, details: Dict[str, Any], 
                  success: bool = True, ip_address: Optional[str] = None):
        """Log an audit event."""
        with self.lock:
            entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "event": event.value,
                "agent_id": agent_id,
                "success": success,
                "details": details,
                "ip_address": ip_address
            }
            
            self.current_log.append(entry)
            
            # Check if we need to rotate log file
            current_date = datetime.utcnow().strftime('%Y%m%d')
            if current_date != self.current_date:
                self.current_date = current_date
                self.log_file = self.log_dir / f"audit_{self.current_date}.jsonl"
            
            try:
                with self._log_write_lock:
                    with open(self.log_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    
            except Exception as e:
                logger.error(f"Failed to write audit log: {e}")
    
    def get_recent_events(self, count: int = 100) -> List[Dict[str, Any]]:
        """Get recent audit events."""
        with self.lock:
            return list(self.current_log)[-count:]
    
    def search_events(self, agent_id: Optional[str] = None, 
                     event_type: Optional[RegistryEvent] = None,
                     start_time: Optional[datetime] = None,
                     end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Search audit events."""
        results = []
        
        with self.lock:
            for entry in self.current_log:
                if agent_id and entry["agent_id"] != agent_id:
                    continue
                if event_type and entry["event"] != event_type.value:
                    continue
                
                entry_time = datetime.fromisoformat(entry["timestamp"])
                if start_time and entry_time < start_time:
                    continue
                if end_time and entry_time > end_time:
                    continue
                    
                results.append(entry)
        
        return results
    
    def cleanup_old_logs(self, days: int = 90):
        """Clean up old log files."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        for log_file in self.log_dir.glob("audit_*.jsonl"):
            try:
                # Extract date from filename
                date_str = log_file.stem.replace("audit_", "")
                file_date = datetime.strptime(date_str, "%Y%m%d")
                
                if file_date < cutoff_date:
                    log_file.unlink()
                    logger.info(f"Deleted old audit log: {log_file}")
            except Exception as e:
                logger.error(f"Error cleaning up {log_file}: {e}")


class RateLimiter:
    """Implements rate limiting for security."""
    
    def __init__(self, window_seconds: int = RATE_LIMIT_WINDOW, 
                 max_requests: int = RATE_LIMIT_MAX_REQUESTS):
        self.window = window_seconds
        self.max_requests = max_requests
        self.requests = defaultdict(deque)
        self.lock = threading.RLock()
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed under rate limit."""
        with self.lock:
            now = time.time()
            
            # Clean old requests
            while self.requests[identifier] and self.requests[identifier][0] < now - self.window:
                self.requests[identifier].popleft()
            
            # Check limit
            if len(self.requests[identifier]) >= self.max_requests:
                return False
            
            # Record request
            self.requests[identifier].append(now)
            return True
    
    def reset(self, identifier: str):
        """Reset rate limit for an identifier."""
        with self.lock:
            self.requests.pop(identifier, None)
    
    def cleanup(self):
        """Clean up old entries."""
        with self.lock:
            now = time.time()
            to_remove = []
            
            for identifier, timestamps in self.requests.items():
                # Remove old timestamps
                while timestamps and timestamps[0] < now - self.window:
                    timestamps.popleft()
                
                # Remove empty entries
                if not timestamps:
                    to_remove.append(identifier)
            
            for identifier in to_remove:
                del self.requests[identifier]


class KeyManager:
    """Manages cryptographic key operations."""
    
    def __init__(self, key_store_dir: str = "keys"):
        self.key_store = Path(key_store_dir)
        self.key_store.mkdir(parents=True, exist_ok=True)
        self.backend = default_backend()
    
    def generate_key_pair(self, algorithm: KeyAlgorithm) -> Tuple[bytes, bytes]:
        """Generate a new key pair."""
        if algorithm == KeyAlgorithm.RSA_2048:
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=self.backend
            )
        elif algorithm == KeyAlgorithm.RSA_4096:
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=4096,
                backend=self.backend
            )
        elif algorithm == KeyAlgorithm.ED25519:
            private_key = ed25519.Ed25519PrivateKey.generate()
        elif algorithm == KeyAlgorithm.ECDSA_P256:
            private_key = ec.generate_private_key(ec.SECP256R1(), self.backend)
        elif algorithm == KeyAlgorithm.ECDSA_P384:
            private_key = ec.generate_private_key(ec.SECP384R1(), self.backend)
        elif algorithm == KeyAlgorithm.ECDSA_P521:
            private_key = ec.generate_private_key(ec.SECP521R1(), self.backend)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        # Serialize keys
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        public_key = private_key.public_key()
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        return public_pem, private_pem
    
    def sign_message(self, message: bytes, private_key_pem: bytes, algorithm: KeyAlgorithm) -> bytes:
        """Sign a message with a private key."""
        private_key = serialization.load_pem_private_key(private_key_pem, password=None, backend=self.backend)
        
        if algorithm in [KeyAlgorithm.RSA_2048, KeyAlgorithm.RSA_4096]:
            signature = private_key.sign(
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
        elif algorithm == KeyAlgorithm.ED25519:
            signature = private_key.sign(message)
        elif algorithm in [KeyAlgorithm.ECDSA_P256, KeyAlgorithm.ECDSA_P384, KeyAlgorithm.ECDSA_P521]:
            signature = private_key.sign(message, ec.ECDSA(hashes.SHA256()))
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        return signature
    
    def verify_signature(self, message: bytes, signature: bytes, public_key_pem: bytes, algorithm: KeyAlgorithm) -> bool:
        """Verify a signature with a public key (constant-time comparison)."""
        try:
            public_key = serialization.load_pem_public_key(public_key_pem, backend=self.backend)
            
            if algorithm in [KeyAlgorithm.RSA_2048, KeyAlgorithm.RSA_4096]:
                public_key.verify(
                    signature,
                    message,
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH
                    ),
                    hashes.SHA256()
                )
            elif algorithm == KeyAlgorithm.ED25519:
                public_key.verify(signature, message)
            elif algorithm in [KeyAlgorithm.ECDSA_P256, KeyAlgorithm.ECDSA_P384, KeyAlgorithm.ECDSA_P521]:
                public_key.verify(signature, message, ec.ECDSA(hashes.SHA256()))
            else:
                return False
            
            return True
            
        except InvalidSignature:
            return False
        except Exception as e:
            logger.error(f"Signature verification error: {e}")
            return False
    
    def encrypt_key(self, key_data: bytes, password: str) -> Tuple[bytes, bytes]:
        """Encrypt a key with a password using PKCS#7 padding."""
        salt = os.urandom(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=KEY_DERIVATION_ITERATIONS,
            backend=self.backend
        )
        key = kdf.derive(password.encode())
        
        iv = os.urandom(16)
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=self.backend)
        encryptor = cipher.encryptor()
        
        # PKCS#7 padding
        padded_data = self._pkcs7_pad(key_data)
        encrypted = encryptor.update(padded_data) + encryptor.finalize()
        
        return encrypted, salt + iv
    
    def decrypt_key(self, encrypted_data: bytes, password: str, salt_iv: bytes) -> bytes:
        """Decrypt a key with a password."""
        salt = salt_iv[:16]
        iv = salt_iv[16:]
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=KEY_DERIVATION_ITERATIONS,
            backend=self.backend
        )
        key = kdf.derive(password.encode())
        
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=self.backend)
        decryptor = cipher.decryptor()
        decrypted = decryptor.update(encrypted_data) + decryptor.finalize()
        
        return self._pkcs7_unpad(decrypted)
    
    def _pkcs7_pad(self, data: bytes) -> bytes:
        """PKCS#7 padding."""
        block_size = 16
        padding_length = block_size - (len(data) % block_size)
        padding = bytes([padding_length]) * padding_length
        return data + padding
    
    def _pkcs7_unpad(self, data: bytes) -> bytes:
        """Remove PKCS#7 padding with validation."""
        if not data:
            raise ValueError("Cannot unpad empty data")
        
        padding_length = data[-1]
        
        if padding_length < 1 or padding_length > 16:
            raise ValueError("Invalid padding")
        
        # Verify padding
        for i in range(padding_length):
            if data[-(i + 1)] != padding_length:
                raise ValueError("Invalid padding")
        
        return data[:-padding_length]


class CertificateAuthority:
    """Manages X.509 certificates for agents."""
    
    def __init__(self, ca_key_path: Optional[str] = None, ca_cert_path: Optional[str] = None):
        self.backend = default_backend()
        self.ca_key = None
        self.ca_cert = None
        self.ca_key_path = ca_key_path
        self.ca_cert_path = ca_cert_path
        
        if ca_key_path and ca_cert_path and Path(ca_key_path).exists() and Path(ca_cert_path).exists():
            self.load_ca(ca_key_path, ca_cert_path)
        else:
            self.generate_ca()
            if ca_key_path and ca_cert_path:
                self.save_ca(ca_key_path, ca_cert_path)
    
    def generate_ca(self):
        """Generate a new Certificate Authority."""
        self.ca_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096,
            backend=self.backend
        )
        
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Graphix"),
            x509.NameAttribute(NameOID.COMMON_NAME, "Graphix Agent CA"),
        ])
        
        self.ca_cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            self.ca_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.utcnow()
        ).not_valid_after(
            datetime.utcnow() + timedelta(days=3650)
        ).add_extension(
            x509.BasicConstraints(ca=True, path_length=None),
            critical=True,
        ).sign(self.ca_key, hashes.SHA256(), backend=self.backend)
    
    def save_ca(self, key_path: str, cert_path: str):
        """Save CA to files."""
        # Save key
        key_pem = self.ca_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        with open(key_path, 'wb') as f:
            f.write(key_pem)
        
        # Save cert
        cert_pem = self.ca_cert.public_bytes(serialization.Encoding.PEM)
        with open(cert_path, 'wb') as f:
            f.write(cert_pem)
        
        logger.info(f"CA saved to {key_path} and {cert_path}")
    
    def load_ca(self, key_path: str, cert_path: str):
        """Load CA from files."""
        with open(key_path, 'rb') as f:
            self.ca_key = serialization.load_pem_private_key(f.read(), password=None, backend=self.backend)
        
        with open(cert_path, 'rb') as f:
            self.ca_cert = x509.load_pem_x509_certificate(f.read(), backend=self.backend)
        
        logger.info(f"CA loaded from {key_path} and {cert_path}")
    
    def issue_certificate(self, agent_id: str, public_key_pem: bytes, 
                         valid_days: int = 365) -> AgentCertificate:
        """Issue a certificate for an agent."""
        public_key = serialization.load_pem_public_key(public_key_pem, backend=self.backend)
        
        subject = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Graphix"),
            x509.NameAttribute(NameOID.COMMON_NAME, f"Agent:{agent_id}"),
        ])
        
        now = datetime.utcnow()
        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            self.ca_cert.subject
        ).public_key(
            public_key
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            now
        ).not_valid_after(
            now + timedelta(days=valid_days)
        ).add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName(f"{agent_id}.agents.graphix.local"),
            ]),
            critical=False,
        ).sign(self.ca_key, hashes.SHA256(), backend=self.backend)
        
        cert_pem = cert.public_bytes(serialization.Encoding.PEM)
        
        return AgentCertificate(
            cert_id=str(cert.serial_number),
            certificate=cert_pem,
            issuer=cert.issuer.rfc4514_string(),
            subject=cert.subject.rfc4514_string(),
            serial_number=str(cert.serial_number),
            not_valid_before=cert.not_valid_before,
            not_valid_after=cert.not_valid_after
        )
    
    def verify_certificate(self, cert_pem: bytes) -> bool:
        """Verify a certificate against the CA."""
        try:
            cert = x509.load_pem_x509_certificate(cert_pem, backend=self.backend)
            
            # Verify signature
            self.ca_cert.public_key().verify(
                cert.signature,
                cert.tbs_certificate_bytes,
                padding.PKCS1v15(),
                cert.signature_hash_algorithm,
            )
            
            # Check validity period
            now = datetime.utcnow()
            if now < cert.not_valid_before or now > cert.not_valid_after:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Certificate verification failed: {e}")
            return False


class AgentRegistry:
    """Comprehensive agent registry with full cryptographic security."""
    
    def __init__(self, 
                 registry_file: str = "registry.db",
                 key_store_dir: str = "keys",
                 audit_log_dir: str = "audit_logs",
                 ca_key_path: Optional[str] = None,
                 ca_cert_path: Optional[str] = None):
        """Initialize the agent registry."""
        self.registry_path = Path(registry_file)
        self.agents: Dict[str, AgentProfile] = {}
        self.agents_lock = threading.RLock()
        
        self.logger = logging.getLogger("AgentRegistry")
        self.key_manager = KeyManager(key_store_dir)
        self.cert_authority = CertificateAuthority(ca_key_path, ca_cert_path)
        self.audit_logger = AuditLogger(audit_log_dir)
        self.rate_limiter = RateLimiter()
        
        # Security tracking (thread-safe)
        self.revoked_keys: Set[str] = set()
        self.revoked_keys_lock = threading.RLock()
        self.revoked_certs: Set[str] = set()
        self.revoked_certs_lock = threading.RLock()
        
        # Shutdown flag
        self.shutdown = False
        self.maintenance_thread = None
        
        # Optional VULCAN integration
        self.vulcan_bridge = None
        if VULCAN_INTEGRATION_AVAILABLE:
            try:
                # Get runtime if available for VULCAN bridge
                from unified_runtime_core import get_runtime
                runtime = get_runtime()
                if hasattr(runtime, 'vulcan_bridge') and runtime.vulcan_bridge:
                    self.vulcan_bridge = runtime.vulcan_bridge
                    logger.info("VULCAN integration enabled for agent registry")
            except:
                pass
        
        # Initialize database with connection pool
        self._init_database()
        self.db_pool = DatabaseConnectionPool(self.registry_path, pool_size=5)
        self._load_registry()
        
        # Start maintenance thread
        self._start_maintenance()
        
        self.logger.info("Agent Registry initialized")
    
    def _init_database(self):
        """Initialize SQLite database for persistent storage."""
        conn = sqlite3.connect(str(self.registry_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agents (
                agent_id TEXT PRIMARY KEY,
                profile_data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_agents_created 
            ON agents(created_at)
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS revoked_keys (
                key_id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                revoked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                reason TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_revoked_keys_agent 
            ON revoked_keys(agent_id)
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS revoked_certs (
                cert_id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                revoked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                reason TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_revoked_certs_agent 
            ON revoked_certs(agent_id)
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_registry(self):
        """Load agents from database using JSON instead of pickle."""
        with self.db_pool.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT agent_id, profile_data FROM agents")
            for row in cursor.fetchall():
                try:
                    agent_id = row[0]
                    profile_json = row[1]
                    
                    # Parse JSON
                    profile_dict = json.loads(profile_json)
                    profile = AgentProfile.from_dict(profile_dict)
                    
                    with self.agents_lock:
                        self.agents[agent_id] = profile
                    
                except Exception as e:
                    self.logger.error(f"Failed to load agent {row[0]}: {e}")
            
            # Load revoked keys
            cursor.execute("SELECT key_id FROM revoked_keys")
            with self.revoked_keys_lock:
                self.revoked_keys = {row[0] for row in cursor.fetchall()}
            
            # Load revoked certificates
            cursor.execute("SELECT cert_id FROM revoked_certs")
            with self.revoked_certs_lock:
                self.revoked_certs = {row[0] for row in cursor.fetchall()}
        
        self.logger.info(f"Loaded {len(self.agents)} agents from registry")
    
    def _save_agent(self, profile: AgentProfile):
        """Save agent to database using JSON."""
        try:
            profile_json = json.dumps(profile.to_dict(), ensure_ascii=False)
            
            with self.db_pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO agents (agent_id, profile_data, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                ''', (
                    profile.agent_id,
                    profile_json,
                    profile.created_at,
                    datetime.utcnow()
                ))
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to save agent {profile.agent_id}: {e}")
            raise
    
    def _validate_agent_id(self, agent_id: str) -> bool:
        """Validate agent ID format."""
        if not agent_id or len(agent_id) > AGENT_ID_MAX_LENGTH:
            return False
        
        # Only allow alphanumeric, underscore, hyphen
        import re
        if not re.match(r'^[a-zA-Z0-9_-]+$', agent_id):
            return False
        
        return True
    
    def _validate_permission(self, permission: str) -> bool:
        """Validate permission name."""
        if not permission or len(permission) > PERMISSION_NAME_MAX_LENGTH:
            return False
        
        import re
        if not re.match(r'^[a-z_]+$', permission):
            return False
        
        return True
    
    def register_agent(self, 
                       agent_id: str,
                       name: str,
                       roles: List[Union[str, AgentRole]],
                       algorithm: KeyAlgorithm = KeyAlgorithm.ED25519,
                       issue_certificate: bool = True,
                       metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Register a new agent with cryptographic credentials."""
        # Validate inputs
        if not self._validate_agent_id(agent_id):
            raise ValueError(f"Invalid agent ID: {agent_id}")
        
        if not name or len(name) > 256:
            raise ValueError("Invalid agent name")
        
        if not roles:
            raise ValueError("At least one role is required")
        
        with self.agents_lock:
            # Check if agent exists
            if agent_id in self.agents:
                if self.agents[agent_id].is_active:
                    raise ValueError(f"Agent '{agent_id}' is already registered")
                else:
                    # Reactivate agent
                    self.agents[agent_id].is_active = True
                    self._save_agent(self.agents[agent_id])
                    return {"agent_id": agent_id, "status": "reactivated"}
            
            # Convert roles
            agent_roles = []
            for role in roles:
                if isinstance(role, str):
                    try:
                        agent_roles.append(AgentRole(role))
                    except ValueError:
                        raise ValueError(f"Invalid role: {role}")
                else:
                    agent_roles.append(role)

            # Optional: Validate with VULCAN if available
            if self.vulcan_bridge:
                proposal = {
                    "operation": "register_agent",
                    "agent_id": agent_id,
                    "roles": [r.value for r in agent_roles]
                }
                
                try:
                    validation = self.vulcan_bridge.world_model.evaluate_agent_proposal(proposal)
                    if not validation.get('valid', True):
                        logger.warning(f"VULCAN flagged agent registration: {validation.get('reasoning')}")
                except Exception as e:
                    logger.debug(f"VULCAN validation skipped: {e}")
            
            # Generate key pair
            public_key, private_key = self.key_manager.generate_key_pair(algorithm)
            
            key_id = f"{agent_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            agent_key = AgentKey(
                key_id=key_id,
                algorithm=algorithm,
                public_key=public_key,
                private_key=None,
                expires_at=datetime.utcnow() + timedelta(days=MAX_KEY_AGE_DAYS)
            )
            
            # Issue certificate
            certificates = []
            if issue_certificate:
                cert = self.cert_authority.issue_certificate(agent_id, public_key)
                certificates.append(cert)
            
            # Create profile
            profile = AgentProfile(
                agent_id=agent_id,
                name=name,
                roles=agent_roles,
                keys=[agent_key],
                certificates=certificates,
                metadata=metadata or {}
            )
            
            # Set permissions
            profile.permissions = self._get_default_permissions(agent_roles)
            
            # Save
            self.agents[agent_id] = profile
            self._save_agent(profile)
            
            # Audit log
            self.audit_logger.log_event(
                RegistryEvent.AGENT_REGISTERED,
                agent_id,
                {"name": name, "roles": [r.value for r in agent_roles], "algorithm": algorithm.value}
            )
            
            self.logger.info(f"Agent '{agent_id}' registered")
            
            import base64
            return {
                "agent_id": agent_id,
                "key_id": key_id,
                "public_key": base64.b64encode(public_key).decode('utf-8'),
                "private_key": base64.b64encode(private_key).decode('utf-8'),
                "algorithm": algorithm.value,
                "certificate": base64.b64encode(certificates[0].certificate).decode('utf-8') if certificates else None,
                "expires_at": agent_key.expires_at.isoformat()
            }
    
    def verify_signature(self, 
                        agent_id: str,
                        message: Union[str, bytes],
                        signature: Union[str, bytes],
                        key_id: Optional[str] = None,
                        ip_address: Optional[str] = None) -> bool:
        """Verify a cryptographic signature from an agent (constant-time)."""
        # Rate limiting
        if not self.rate_limiter.is_allowed(f"verify_{agent_id}"):
            self.audit_logger.log_event(
                RegistryEvent.AUTH_FAILURE,
                agent_id,
                {"reason": "rate_limit_exceeded"},
                success=False,
                ip_address=ip_address
            )
            return False
        
        with self.agents_lock:
            agent = self.agents.get(agent_id)
            if not agent:
                self.audit_logger.log_event(
                    RegistryEvent.AUTH_FAILURE,
                    agent_id,
                    {"reason": "agent_not_found"},
                    success=False,
                    ip_address=ip_address
                )
                self.logger.warning(f"Verification failed: Agent '{agent_id}' not found")
                # Sleep to prevent timing attacks
                time.sleep(0.1)
                return False
            
            # Check if locked
            if agent.is_locked():
                self.audit_logger.log_event(
                    RegistryEvent.AUTH_FAILURE,
                    agent_id,
                    {"reason": "agent_locked"},
                    success=False,
                    ip_address=ip_address
                )
                time.sleep(0.1)
                return False
            
            # Convert message
            if isinstance(message, str):
                message = message.encode('utf-8')
            
            # Decode signature
            if isinstance(signature, str):
                import base64
                try:
                    signature = base64.b64decode(signature)
                except Exception as e:
                    self.audit_logger.log_event(
                        RegistryEvent.AUTH_FAILURE,
                        agent_id,
                        {"reason": "invalid_signature_format", "error": str(e)},
                        success=False,
                        ip_address=ip_address
                    )
                    agent.failed_attempts += 1
                    
                    if agent.failed_attempts >= MAX_FAILED_ATTEMPTS:
                        agent.locked_until = datetime.utcnow() + timedelta(seconds=LOCKOUT_DURATION)
                        self.audit_logger.log_event(
                            RegistryEvent.SUSPICIOUS_ACTIVITY,
                            agent_id,
                            {"reason": "max_failed_attempts", "locked_until": agent.locked_until.isoformat()},
                            success=False,
                            ip_address=ip_address
                        )
                    
                    self._save_agent(agent)
                    time.sleep(0.1)
                    return False
            
            # Find key
            if key_id:
                key = next((k for k in agent.keys if k.key_id == key_id), None)
            else:
                key = agent.get_active_key()
            
            if not key:
                self.audit_logger.log_event(
                    RegistryEvent.AUTH_FAILURE,
                    agent_id,
                    {"reason": "no_valid_key"},
                    success=False,
                    ip_address=ip_address
                )
                time.sleep(0.1)
                return False
            
            # Check revocation
            with self.revoked_keys_lock:
                if key.key_id in self.revoked_keys:
                    self.audit_logger.log_event(
                        RegistryEvent.AUTH_FAILURE,
                        agent_id,
                        {"reason": "key_revoked", "key_id": key.key_id},
                        success=False,
                        ip_address=ip_address
                    )
                    time.sleep(0.1)
                    return False
            
            # Verify signature
            is_valid = self.key_manager.verify_signature(message, signature, key.public_key, key.algorithm)
            
            if is_valid:
                agent.failed_attempts = 0
                agent.last_seen = datetime.utcnow()
                self._save_agent(agent)
                
                self.audit_logger.log_event(
                    RegistryEvent.AUTH_SUCCESS,
                    agent_id,
                    {"key_id": key.key_id},
                    success=True,
                    ip_address=ip_address
                )
            else:
                agent.failed_attempts += 1
                
                if agent.failed_attempts >= MAX_FAILED_ATTEMPTS:
                    agent.locked_until = datetime.utcnow() + timedelta(seconds=LOCKOUT_DURATION)
                    self.audit_logger.log_event(
                        RegistryEvent.SUSPICIOUS_ACTIVITY,
                        agent_id,
                        {"reason": "max_failed_attempts", "locked_until": agent.locked_until.isoformat()},
                        success=False,
                        ip_address=ip_address
                    )
                
                self._save_agent(agent)
                
                self.audit_logger.log_event(
                    RegistryEvent.AUTH_FAILURE,
                    agent_id,
                    {"reason": "invalid_signature", "attempts": agent.failed_attempts},
                    success=False,
                    ip_address=ip_address
                )
                
                # Constant-time sleep to prevent timing attacks
                time.sleep(0.1)
            
            return is_valid
    
    def rotate_key(self, 
                  agent_id: str,
                  algorithm: Optional[KeyAlgorithm] = None,
                  issue_certificate: bool = True) -> Dict[str, Any]:
        """Rotate an agent's key and optionally issue new certificate."""
        with self.agents_lock:
            agent = self.agents.get(agent_id)
            if not agent:
                raise ValueError(f"Agent '{agent_id}' not found")
            
            # Get algorithm
            current_key = agent.get_active_key()
            if not algorithm and current_key:
                algorithm = current_key.algorithm
            elif not algorithm:
                algorithm = KeyAlgorithm.ED25519
            
            # Mark current keys as inactive
            for key in agent.keys:
                key.is_active = False
            
            # Generate new key
            public_key, private_key = self.key_manager.generate_key_pair(algorithm)
            
            key_id = f"{agent_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            new_key = AgentKey(
                key_id=key_id,
                algorithm=algorithm,
                public_key=public_key,
                expires_at=datetime.utcnow() + timedelta(days=MAX_KEY_AGE_DAYS)
            )
            
            agent.keys.append(new_key)
            
            # Issue new certificate
            cert_b64 = None
            if issue_certificate:
                cert = self.cert_authority.issue_certificate(agent_id, public_key)
                agent.certificates.append(cert)
                
                import base64
                cert_b64 = base64.b64encode(cert.certificate).decode('utf-8')
            
            self._save_agent(agent)
            
            # Audit log
            self.audit_logger.log_event(
                RegistryEvent.KEY_ROTATED,
                agent_id,
                {"new_key_id": key_id, "algorithm": algorithm.value}
            )
            
            import base64
            return {
                "key_id": key_id,
                "public_key": base64.b64encode(public_key).decode('utf-8'),
                "private_key": base64.b64encode(private_key).decode('utf-8'),
                "algorithm": algorithm.value,
                "certificate": cert_b64,
                "expires_at": new_key.expires_at.isoformat()
            }
    
    def revoke_key(self, agent_id: str, key_id: str, reason: str = "unspecified"):
        """Revoke an agent's key."""
        with self.agents_lock:
            agent = self.agents.get(agent_id)
            if not agent:
                raise ValueError(f"Agent '{agent_id}' not found")
            
            # Find and deactivate key
            key_found = False
            for key in agent.keys:
                if key.key_id == key_id:
                    key.is_active = False
                    key_found = True
                    break
            
            if not key_found:
                raise ValueError(f"Key '{key_id}' not found for agent '{agent_id}'")
            
            # Add to revoked list
            with self.revoked_keys_lock:
                self.revoked_keys.add(key_id)
            
            # Save to database
            with self.db_pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO revoked_keys (key_id, agent_id, revoked_at, reason)
                    VALUES (?, ?, ?, ?)
                ''', (key_id, agent_id, datetime.utcnow(), reason))
                conn.commit()
            
            self._save_agent(agent)
            
            # Audit log
            self.audit_logger.log_event(
                RegistryEvent.CERT_REVOKED,
                agent_id,
                {"key_id": key_id, "reason": reason}
            )
            
            self.logger.info(f"Key '{key_id}' revoked for agent '{agent_id}'")
    
    def revoke_certificate(self, agent_id: str, cert_id: str, reason: str = "unspecified"):
        """Revoke an agent's certificate."""
        with self.agents_lock:
            agent = self.agents.get(agent_id)
            if not agent:
                raise ValueError(f"Agent '{agent_id}' not found")
            
            # Find and revoke certificate
            cert_found = False
            for cert in agent.certificates:
                if cert.cert_id == cert_id:
                    cert.is_revoked = True
                    cert.revocation_reason = reason
                    cert.revocation_time = datetime.utcnow()
                    cert_found = True
                    break
            
            if not cert_found:
                raise ValueError(f"Certificate '{cert_id}' not found for agent '{agent_id}'")
            
            # Add to revoked list
            with self.revoked_certs_lock:
                self.revoked_certs.add(cert_id)
            
            # Save to database
            with self.db_pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO revoked_certs (cert_id, agent_id, revoked_at, reason)
                    VALUES (?, ?, ?, ?)
                ''', (cert_id, agent_id, datetime.utcnow(), reason))
                conn.commit()
            
            self._save_agent(agent)
            
            # Audit log
            self.audit_logger.log_event(
                RegistryEvent.CERT_REVOKED,
                agent_id,
                {"cert_id": cert_id, "reason": reason}
            )
    
    def grant_permission(self, agent_id: str, permission: str, granter_id: str):
        """Grant a permission to an agent."""
        if not self._validate_permission(permission):
            raise ValueError(f"Invalid permission name: {permission}")
        
        with self.agents_lock:
            agent = self.agents.get(agent_id)
            if not agent:
                raise ValueError(f"Agent '{agent_id}' not found")
            
            agent.permissions[permission] = True
            self._save_agent(agent)
            
            self.audit_logger.log_event(
                RegistryEvent.PERMISSION_GRANTED,
                agent_id,
                {"permission": permission, "granted_by": granter_id}
            )
            
            self.logger.info(f"Permission '{permission}' granted to agent '{agent_id}'")
    
    def revoke_permission(self, agent_id: str, permission: str, revoker_id: str):
        """Revoke a permission from an agent."""
        with self.agents_lock:
            agent = self.agents.get(agent_id)
            if not agent:
                raise ValueError(f"Agent '{agent_id}' not found")
            
            agent.permissions[permission] = False
            self._save_agent(agent)
            
            self.audit_logger.log_event(
                RegistryEvent.PERMISSION_REVOKED,
                agent_id,
                {"permission": permission, "revoked_by": revoker_id}
            )
    
    def check_permission(self, agent_id: str, permission: str) -> bool:
        """Check if an agent has a permission."""
        with self.agents_lock:
            agent = self.agents.get(agent_id)
            if not agent or not agent.is_active:
                return False
            
            # Check explicit permission
            if permission in agent.permissions:
                return agent.permissions[permission]
            
            # Admin has all permissions
            if AgentRole.ADMIN in agent.roles:
                return True
            
            # Role-based permissions
            role_permissions = {
                AgentRole.EXECUTOR: {"execute_graph", "submit_graph", "view_status"},
                AgentRole.VIEWER: {"view_status", "view_graph"},
                AgentRole.DEVELOPER: {"execute_graph", "submit_graph", "view_status", "view_graph", "debug"},
                AgentRole.AUDITOR: {"view_audit", "view_status", "view_graph"},
                AgentRole.SERVICE: {"execute_graph", "submit_graph"}
            }
            
            for role in agent.roles:
                if permission in role_permissions.get(role, set()):
                    return True
            
            return False
    
    def _get_default_permissions(self, roles: List[AgentRole]) -> Dict[str, bool]:
        """Get default permissions for roles."""
        permissions = {}
        
        for role in roles:
            if role == AgentRole.ADMIN:
                permissions.update({
                    "execute_graph": True,
                    "submit_graph": True,
                    "view_status": True,
                    "view_graph": True,
                    "modify_graph": True,
                    "delete_graph": True,
                    "manage_agents": True,
                    "view_audit": True
                })
            elif role == AgentRole.EXECUTOR:
                permissions.update({
                    "execute_graph": True,
                    "submit_graph": True,
                    "view_status": True
                })
            elif role == AgentRole.VIEWER:
                permissions.update({
                    "view_status": True,
                    "view_graph": True
                })
            elif role == AgentRole.DEVELOPER:
                permissions.update({
                    "execute_graph": True,
                    "submit_graph": True,
                    "view_status": True,
                    "view_graph": True,
                    "debug": True
                })
            elif role == AgentRole.AUDITOR:
                permissions.update({
                    "view_audit": True,
                    "view_status": True,
                    "view_graph": True
                })
            elif role == AgentRole.SERVICE:
                permissions.update({
                    "execute_graph": True,
                    "submit_graph": True
                })
        
        return permissions
    
    def _start_maintenance(self):
        """Start background maintenance thread."""
        def maintenance_loop():
            while not self.shutdown:
                try:
                    time.sleep(3600)  # Run every hour
                    
                    if self.shutdown:
                        break
                    
                    with self.agents_lock:
                        # Clean expired keys
                        for agent in self.agents.values():
                            expired_keys = [k for k in agent.keys if k.is_expired()]
                            for key in expired_keys:
                                key.is_active = False
                                with self.revoked_keys_lock:
                                    self.revoked_keys.add(key.key_id)
                                
                                self.logger.info(f"Expired key {key.key_id} for agent {agent.agent_id}")
                    
                    # Clean rate limiter
                    self.rate_limiter.cleanup()
                    
                    # Clean old audit logs
                    self.audit_logger.cleanup_old_logs(days=90)
                    
                except Exception as e:
                    self.logger.error(f"Maintenance error: {e}")
        
        self.maintenance_thread = threading.Thread(target=maintenance_loop, daemon=True)
        self.maintenance_thread.start()
    
    def get_agent_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get information about an agent."""
        with self.agents_lock:
            agent = self.agents.get(agent_id)
            if not agent:
                return None
            
            active_key = agent.get_active_key()
            
            return {
                "agent_id": agent.agent_id,
                "name": agent.name,
                "roles": [r.value for r in agent.roles],
                "is_active": agent.is_active,
                "is_locked": agent.is_locked(),
                "created_at": agent.created_at.isoformat(),
                "last_seen": agent.last_seen.isoformat() if agent.last_seen else None,
                "active_key": {
                    "key_id": active_key.key_id,
                    "algorithm": active_key.algorithm.value,
                    "expires_at": active_key.expires_at.isoformat() if active_key.expires_at else None
                } if active_key else None,
                "certificates": len(agent.certificates),
                "permissions": [k for k, v in agent.permissions.items() if v]
            }
    
    def list_agents(self, role: Optional[AgentRole] = None, active_only: bool = True) -> List[Dict[str, Any]]:
        """List registered agents."""
        with self.agents_lock:
            agents = []
            
            for agent in self.agents.values():
                if active_only and not agent.is_active:
                    continue
                
                if role and role not in agent.roles:
                    continue
                
                info = self.get_agent_info(agent.agent_id)
                if info:
                    agents.append(info)
            
            return agents
    
    def export_ca_certificate(self) -> str:
        """Export CA certificate for distribution."""
        return self.cert_authority.ca_cert.public_bytes(serialization.Encoding.PEM).decode('utf-8')
    
    def import_agent_certificate(self, agent_id: str, cert_pem: bytes) -> bool:
        """Import an external certificate for an agent."""
        with self.agents_lock:
            agent = self.agents.get(agent_id)
            if not agent:
                raise ValueError(f"Agent '{agent_id}' not found")
            
            # Verify certificate
            if not self.cert_authority.verify_certificate(cert_pem):
                self.logger.warning(f"Certificate verification failed for agent '{agent_id}'")
                return False
            
            # Parse certificate
            cert = x509.load_pem_x509_certificate(cert_pem, backend=default_backend())
            
            agent_cert = AgentCertificate(
                cert_id=str(cert.serial_number),
                certificate=cert_pem,
                issuer=cert.issuer.rfc4514_string(),
                subject=cert.subject.rfc4514_string(),
                serial_number=str(cert.serial_number),
                not_valid_before=cert.not_valid_before,
                not_valid_after=cert.not_valid_after
            )
            
            agent.certificates.append(agent_cert)
            self._save_agent(agent)
            
            self.logger.info(f"Certificate imported for agent '{agent_id}'")
            return True
    
    def get_audit_logs(self, 
                      agent_id: Optional[str] = None,
                      event_type: Optional[RegistryEvent] = None,
                      start_time: Optional[datetime] = None,
                      end_time: Optional[datetime] = None,
                      limit: int = 100) -> List[Dict[str, Any]]:
        """Get audit logs with pagination."""
        results = self.audit_logger.search_events(
            agent_id=agent_id,
            event_type=event_type,
            start_time=start_time,
            end_time=end_time
        )
        
        return results[:limit]
    
    def shutdown_registry(self):
        """Shutdown the registry cleanly."""
        self.logger.info("Shutting down registry...")
        
        self.shutdown = True
        
        # Wait for maintenance thread
        if self.maintenance_thread and self.maintenance_thread.is_alive():
            self.maintenance_thread.join(timeout=5)
        
        # Close database connections
        self.db_pool.close_all()
        
        self.logger.info("Registry shutdown complete")


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("Graphix Agent Registry - Cryptographic Security Demo")
    print("=" * 60)
    
    # Create registry
    registry = AgentRegistry(
        registry_file="test_registry.db",
        ca_key_path="ca_key.pem",
        ca_cert_path="ca_cert.pem"
    )
    
    try:
        # Register an agent
        print("\n1. Registering new agent...")
        registration = registry.register_agent(
            agent_id="agent_001",
            name="Test Agent",
            roles=[AgentRole.EXECUTOR, AgentRole.DEVELOPER],
            algorithm=KeyAlgorithm.ED25519,
            issue_certificate=True,
            metadata={"version": "1.0", "platform": "linux"}
        )
        
        print(f"   Agent ID: {registration['agent_id']}")
        print(f"   Key ID: {registration['key_id']}")
        print(f"   Algorithm: {registration['algorithm']}")
        print(f"   Public Key: {registration['public_key'][:50]}...")
        
        # Store private key
        import base64
        private_key = base64.b64decode(registration['private_key'])
        
        # Sign a message
        print("\n2. Signing a message...")
        message = "This is a test message for cryptographic signing"
        signature = registry.key_manager.sign_message(
            message.encode(),
            private_key,
            KeyAlgorithm.ED25519
        )
        signature_b64 = base64.b64encode(signature).decode('utf-8')
        print(f"   Message: {message}")
        print(f"   Signature: {signature_b64[:50]}...")
        
        # Verify signature
        print("\n3. Verifying signature...")
        is_valid = registry.verify_signature(
            agent_id="agent_001",
            message=message,
            signature=signature_b64,
            ip_address="127.0.0.1"
        )
        print(f"   Valid: {is_valid}")
        
        # Test invalid signature
        print("\n4. Testing invalid signature...")
        is_valid = registry.verify_signature(
            agent_id="agent_001",
            message="Different message",
            signature=signature_b64,
            ip_address="127.0.0.1"
        )
        print(f"   Valid: {is_valid}")
        
        # Check permissions
        print("\n5. Checking permissions...")
        perms = ["execute_graph", "submit_graph", "view_audit", "manage_agents"]
        for perm in perms:
            has_perm = registry.check_permission("agent_001", perm)
            print(f"   {perm}: {has_perm}")
        
        # Rotate key
        print("\n6. Rotating key...")
        new_key = registry.rotate_key("agent_001", KeyAlgorithm.RSA_2048)
        print(f"   New Key ID: {new_key['key_id']}")
        print(f"   Algorithm: {new_key['algorithm']}")
        
        # List agents
        print("\n7. Listing agents...")
        agents = registry.list_agents()
        for agent in agents:
            print(f"   - {agent['name']} ({agent['agent_id']})")
            print(f"     Roles: {agent['roles']}")
            print(f"     Active: {agent['is_active']}, Locked: {agent['is_locked']}")
        
        # Get audit logs
        print("\n8. Recent audit events...")
        logs = registry.get_audit_logs(limit=5)
        for log in logs:
            print(f"   [{log['timestamp']}] {log['event']}: {log['agent_id']}")
        
        # Export CA certificate
        print("\n9. CA Certificate:")
        ca_cert = registry.export_ca_certificate()
        print(f"   {ca_cert[:200]}...")
        
        print("\n" + "=" * 60)
        print("Cryptographic security demo completed!")
        
    finally:
        # Clean shutdown
        registry.shutdown_registry()
        
        # Clean up test files
        import os
        for f in ["test_registry.db", "ca_key.pem", "ca_cert.pem"]:
            if os.path.exists(f):
                os.remove(f)