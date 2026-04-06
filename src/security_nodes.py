import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, Optional

from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

# Handle KeyManager import - use unified version
try:
    from key_manager import KeyManager
except ImportError:
    try:
        from .key_manager import KeyManager
    except ImportError:
        # Fallback stub if unified KeyManager isn't available
        class KeyManager:
            def __init__(self, agent_id: str):
                self.agent_id = agent_id
                self.keys = {}

            def get_key(self, key_id: str):
                return self.keys.get(key_id)

            def store_key(self, key_id: str, key_data):
                self.keys[key_id] = key_data


# NSOAligner exists
try:
    from .nso_aligner import NSOAligner, get_nso_aligner
except ImportError:
    from nso_aligner import NSOAligner, get_nso_aligner

# 2025: Add ITU F.748.53 compression, F.748.47/53 policy, photonic audit, Grok-4 kernel, EU2025 ethical_label
try:
    try:
        from .llm_compressor import LLMCompressor
    except ImportError:
        from llm_compressor import LLMCompressor
except ImportError:
    LLMCompressor = None

try:
    try:
        from .hardware_dispatcher import HardwareDispatcher
    except ImportError:
        from hardware_dispatcher import HardwareDispatcher
except ImportError:
    HardwareDispatcher = None

try:
    try:
        from .grok_kernel_audit import GrokKernelAudit
    except ImportError:
        from grok_kernel_audit import GrokKernelAudit
except ImportError:
    GrokKernelAudit = None

logger = logging.getLogger(__name__)

# Constants for validation
MAX_DATA_SIZE = 10 * 1024 * 1024  # 10MB max data size
MAX_TENSOR_ELEMENTS = 1_000_000
MAX_STRING_LENGTH = 1_000_000


class SecurityNodeError(Exception):
    """Base exception for security node errors."""


class EncryptNode:
    """
    Encrypts data using specified algorithm and key, supporting secure data handling.
    Integrates with KeyManager for key retrieval, NSOAligner for ethical audits,
    LLMCompressor for ITU F.748.53 compliance, HardwareDispatcher for photonic audit,
    and GrokKernelAudit for kernel explainability.

    FIXES APPLIED:
    - Proper KeyManager initialization with agent_id
    - Correct Fernet key handling (base64-encoded)
    - Fixed NSOAligner method call (multi_model_audit)
    - Removed non-existent hardware methods
    - Input validation for all data
    - Safe context isolation
    - Fixed imports to work both as module and package
    """

    def __init__(self, agent_id: str = "security_node_default"):
        # FIXED: KeyManager requires agent_id
        self.key_manager = KeyManager(agent_id)
        # FIX: Use singleton pattern to prevent model reloading
        self.nso_aligner = get_nso_aligner()
        self.compressor = LLMCompressor() if LLMCompressor is not None else None
        self.hardware = HardwareDispatcher() if HardwareDispatcher is not None else None
        self.kernel_audit = GrokKernelAudit() if GrokKernelAudit is not None else None
        self.agent_id = agent_id

    def _validate_data(self, data: Any) -> None:
        """Validate input data before encryption."""
        if data is None:
            raise ValueError("Data cannot be None")

        # Check data size
        if isinstance(data, (dict, list)):
            data_str = json.dumps(data)
            if len(data_str) > MAX_DATA_SIZE:
                raise ValueError(f"Data exceeds maximum size of {MAX_DATA_SIZE} bytes")
        elif isinstance(data, str):
            if len(data) > MAX_STRING_LENGTH:
                raise ValueError(
                    f"String data exceeds maximum length of {MAX_STRING_LENGTH}"
                )
        elif isinstance(data, bytes):
            if len(data) > MAX_DATA_SIZE:
                raise ValueError(f"Byte data exceeds maximum size of {MAX_DATA_SIZE}")

    def _validate_tensor(self, tensor: Any) -> bool:
        """Validate tensor data."""
        if tensor is None:
            return False

        try:
            # Basic type check
            if isinstance(tensor, (list, tuple)):
                # Count total elements
                def count_elements(obj):
                    if isinstance(obj, (list, tuple)):
                        return sum(count_elements(item) for item in obj)
                    return 1

                total = count_elements(tensor)
                if total > MAX_TENSOR_ELEMENTS:
                    logger.warning(
                        f"Tensor has {total} elements, exceeds max {MAX_TENSOR_ELEMENTS}"
                    )
                    return False

            return True
        except Exception as e:
            logger.warning(f"Tensor validation failed: {e}")
            return False

    def _get_encryption_key(self, key_id: str, algorithm: str) -> Any:
        """
        Retrieve and validate encryption key.

        The key must be in the correct format required by the algorithm.
        This method does not attempt to fix or derive keys.
        """
        # Retrieve key from KeyManager
        key_data = (
            self.key_manager.get_key(key_id)
            if hasattr(self.key_manager, "get_key")
            else None
        )

        if key_data is None:
            # Fallback: try direct access
            key_data = (
                self.key_manager.keys.get(key_id)
                if hasattr(self.key_manager, "keys")
                else None
            )

        if not key_data:
            raise ValueError(f"Key not found for key_id={key_id}")

        # Handle key data based on algorithm, failing if invalid
        if algorithm == "AES":
            # For Fernet/AES, key must be a URL-safe base64-encoded 32-byte key.
            if isinstance(key_data, str):
                key_bytes = key_data.encode()
            elif isinstance(key_data, bytes):
                key_bytes = key_data
            else:
                raise ValueError(f"Invalid key data type for AES: {type(key_data)}")

            try:
                # Validate that the key is a valid Fernet key without altering it.
                Fernet(key_bytes)
                return key_bytes
            except Exception as e:
                # Fail loudly if the key is not in the correct format.
                raise ValueError(
                    f"Invalid key format for AES. Expected a valid Fernet key. Error: {e}"
                )

        elif algorithm == "RSA":
            # For RSA, expect PEM-encoded key as string or bytes.
            if isinstance(key_data, bytes):
                return key_data
            elif isinstance(key_data, str):
                return key_data.encode()
            else:
                raise ValueError(f"Invalid RSA key data type: {type(key_data)}")

        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

    def execute(
        self, data: Any, params: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute EncryptNode to encrypt data using specified algorithm and key.
        2025: Audit ITU F.748.53 compression and photonic energy, add Grok-4 kernel audit, EU ethical_label.
        """
        algorithm = params.get("algorithm", "AES")
        key_id = params.get("key_id")

        # FIXED: Safe context access without mutation
        kernel = context.get("kernel", None)
        tensor = context.get("tensor", None)

        logger.info(
            f"Executing EncryptNode with algorithm={algorithm}, key_id={key_id}"
        )

        try:
            if not key_id:
                raise ValueError("Missing key_id in params")

            # FIXED: Validate input data
            self._validate_data(data)

            # Retrieve and validate encryption key
            key_material = self._get_encryption_key(key_id, algorithm)

            # Perform encryption
            if algorithm == "AES":
                # FIXED: Proper Fernet key handling
                fernet = Fernet(key_material)

                if isinstance(data, (dict, list)):
                    data_bytes = json.dumps(data).encode()
                elif isinstance(data, str):
                    data_bytes = data.encode()
                elif isinstance(data, bytes):
                    data_bytes = data
                else:
                    raise ValueError("Data must be string, dict, list, or bytes")

                encrypted_data = fernet.encrypt(data_bytes).decode()

            elif algorithm == "RSA":
                try:
                    private_key = serialization.load_pem_private_key(
                        key_material, password=None, backend=default_backend()
                    )
                    public_key = private_key.public_key()
                except Exception as e:
                    raise ValueError(f"Failed to load RSA key: {e}")

                if isinstance(data, (dict, list)):
                    data_bytes = json.dumps(data).encode()
                elif isinstance(data, str):
                    data_bytes = data.encode()
                elif isinstance(data, bytes):
                    data_bytes = data
                else:
                    raise ValueError("Data must be string, dict, list, or bytes")

                # RSA has size limits, check data size
                max_rsa_size = 190  # Conservative limit for 2048-bit key with OAEP
                if len(data_bytes) > max_rsa_size:
                    raise ValueError(
                        f"Data too large for RSA encryption: {len(data_bytes)} bytes "
                        f"(max {max_rsa_size}). Consider using AES for large data."
                    )

                encrypted_data = public_key.encrypt(
                    data_bytes,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None,
                    ),
                ).hex()
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")

            # ITU F.748.53 compression audit (if tensor present and valid)
            compression_ok, compression_meta = True, {}
            if self.compressor and tensor is not None and self._validate_tensor(tensor):
                try:
                    compressed = self.compressor.compress_tensor(tensor)
                    compression_ok = self.compressor.validate_compression(compressed)
                    compression_meta = {
                        "compression": "ITU F.748.53",
                        "valid": compression_ok,
                    }
                    if not compression_ok:
                        raise ValueError(
                            "ITU F.748.53 compression validation failed (AI_COMPRESSION_INVALID)"
                        )
                except Exception as e:
                    logger.error(f"Compression error: {e}")
                    compression_ok = False
                    compression_meta = {
                        "compression": "ITU F.748.53",
                        "valid": False,
                        "error": str(e),
                    }
                    raise

            # FIXED: Removed non-existent get_last_metrics() call
            # Photonic/hardware metrics would be tracked elsewhere in the hardware dispatcher
            photonic_meta = {}
            if self.hardware:
                try:
                    # Use actual HardwareDispatcher methods if available
                    if hasattr(self.hardware, "get_metrics"):
                        photonic_meta = self.hardware.get_metrics()
                    elif hasattr(self.hardware, "last_operation_metrics"):
                        photonic_meta = self.hardware.last_operation_metrics
                except Exception as e:
                    logger.warning(f"Photonic metric fetch failed: {e}")

            # Kernel audit with Grok-4 (2025, if kernel present)
            kernel_audit = None
            if self.kernel_audit and kernel:
                try:
                    kernel_audit = self.kernel_audit.inspect(kernel)
                except Exception as e:
                    logger.warning(f"Grok-4 kernel audit failed: {e}")

            # FIXED: Use correct NSOAligner method (multi_model_audit)
            audit_result = self.nso_aligner.multi_model_audit(
                {
                    "data_type": type(data).__name__,
                    "algorithm": algorithm,
                    "operation": "encryption",
                }
            )

            # EU 2025 ethical label (if present in context)
            ethical_label = context.get("ethical_label", audit_result)

            result = {
                "encrypted_data": encrypted_data,
                "compression_ok": compression_ok,
                "compression_meta": compression_meta,
                "photonic_meta": photonic_meta,
                "kernel_audit": kernel_audit,
                "audit": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "node_type": "EncryptNode",
                    "params": params,
                    "status": "success",
                    "ethical_label": ethical_label,
                    "compression_ok": compression_ok,
                    "photonic_energy_nj": photonic_meta.get("energy_nj", None),
                    "kernel_audit": kernel_audit,
                },
            }

            # Safely append to audit log
            if "audit_log" not in context:
                context["audit_log"] = []
            context["audit_log"].append(result["audit"])

            logger.info(f"EncryptNode success: {result['audit']}")
            return result

        except Exception as e:
            logger.error(f"EncryptNode error: {str(e)}")
            result = {
                "encrypted_data": None,
                "compression_ok": False,
                "compression_meta": {"error": str(e)},
                "photonic_meta": {},
                "kernel_audit": None,
                "audit": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "node_type": "EncryptNode",
                    "params": params,
                    "status": "error",
                    "error": str(e),
                    "ethical_label": "risky",
                },
            }

            if "audit_log" not in context:
                context["audit_log"] = []
            context["audit_log"].append(result["audit"])
            raise


class PolicyNode:
    """
    Enforces compliance policies (e.g., GDPR, CCPA, ITU F.748.47, ITU F.748.53) for data handling.
    Integrates with NSOAligner for ethical checks, LLMCompressor for F.748.53, photonic for energy,
    and supports EU2025 ethical_label.

    FIXES APPLIED:
    - Correct NSOAligner method calls
    - Input validation
    - Safe context access
    - Fixed imports to work both as module and package
    """

    def __init__(self):
        # FIX: Use singleton pattern to prevent model reloading
        self.nso_aligner = get_nso_aligner()
        self.compressor = LLMCompressor() if LLMCompressor is not None else None

    def _validate_data(self, data: Any) -> None:
        """Validate input data."""
        if data is None:
            raise ValueError("Data cannot be None")

    def execute(
        self, data: Any, params: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute PolicyNode to enforce compliance policies.

        Args:
            data: Input data to check (e.g., JSON with PII or location metadata).
            params: Dict with 'policy' (e.g., 'GDPR', 'CCPA', 'ITU-F.748.47', 'ITU-F.748.53'),
                    'enforcement' (e.g., 'restrict', 'log').
            context: Runtime context for state and auditing.

        Returns:
            Dict with compliance result and audit metadata.
        """
        policy = params.get("policy", "GDPR")
        enforcement = params.get("enforcement", "restrict")
        logger.info(
            f"Executing PolicyNode with policy={policy}, enforcement={enforcement}"
        )

        try:
            # Validate input
            self._validate_data(data)

            # Validate policy (now supports F.748.53/47)
            supported_policies = ["GDPR", "CCPA", "ITU-F.748.47", "ITU-F.748.53"]
            if policy not in supported_policies:
                raise ValueError(f"Unsupported policy: {policy}")

            compliance_result = "compliant"
            rationale = f"Checked {policy} compliance"

            if isinstance(data, (dict, list)):
                data_str = json.dumps(data)
            elif isinstance(data, str):
                data_str = data
            else:
                data_str = str(data)

            # Example PII check for GDPR/CCPA
            if policy in ["GDPR", "CCPA"]:
                import re

                pii_patterns = [
                    r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
                    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
                ]
                if any(re.search(pattern, data_str) for pattern in pii_patterns):
                    compliance_result = "non-compliant"
                    rationale = f"PII detected in data for {policy}"
                    if enforcement == "restrict":
                        raise ValueError(f"Non-compliant data detected for {policy}")

            # ITU-F.748.47 (AI ethics, e.g., no harmful intent)
            if policy == "ITU-F.748.47":
                # FIXED: Use correct method
                nso_result = self.nso_aligner.multi_model_audit(
                    {
                        "data_sample": data_str[:100],  # Sample for analysis
                        "policy": policy,
                        "operation": "policy_check",
                    }
                )

                if nso_result == "risky":
                    compliance_result = "non-compliant"
                    rationale = f"ITU-F.748.47 violation: {nso_result}"
                    if enforcement == "restrict":
                        raise ValueError("ITU-F.748.47 non-compliance detected")

            # ITU-F.748.53 (compression/energy compliance)
            compression_ok = True
            compression_meta = {}
            if policy == "ITU-F.748.53" and self.compressor:
                try:
                    compressed = self.compressor.compress_tensor(data_str)
                    compression_ok = self.compressor.validate_compression(compressed)
                    compression_meta = {
                        "compression": "ITU F.748.53",
                        "valid": compression_ok,
                    }
                    if not compression_ok:
                        compliance_result = "non-compliant"
                        rationale = "ITU F.748.53 compression validation failed"
                        if enforcement == "restrict":
                            raise ValueError(
                                "ITU-F.748.53 non-compliance detected (AI_COMPRESSION_INVALID)"
                            )
                except Exception as e:
                    logger.error(f"Compression error: {e}")
                    compliance_result = "non-compliant"
                    compression_ok = False
                    compression_meta = {
                        "compression": "ITU F.748.53",
                        "valid": False,
                        "error": str(e),
                    }
                    if enforcement == "restrict":
                        raise

            ethical_label = context.get("ethical_label", compliance_result)

            result = {
                "compliance": compliance_result,
                "compression_ok": compression_ok if policy == "ITU-F.748.53" else None,
                "compression_meta": (
                    compression_meta if policy == "ITU-F.748.53" else {}
                ),
                "audit": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "node_type": "PolicyNode",
                    "params": params,
                    "status": "success",
                    "ethical_label": ethical_label,
                    "rationale": rationale,
                    "compression_ok": (
                        compression_ok if policy == "ITU-F.748.53" else None
                    ),
                },
            }

            if "audit_log" not in context:
                context["audit_log"] = []
            context["audit_log"].append(result["audit"])

            logger.info(f"PolicyNode success: {result['audit']}")
            return result

        except Exception as e:
            logger.error(f"PolicyNode error: {str(e)}")
            result = {
                "compliance": "error",
                "compression_ok": False,
                "compression_meta": {"error": str(e)},
                "audit": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "node_type": "PolicyNode",
                    "params": params,
                    "status": "error",
                    "error": str(e),
                    "ethical_label": "risky",
                },
            }

            if "audit_log" not in context:
                context["audit_log"] = []
            context["audit_log"].append(result["audit"])
            raise


def dispatch_security_node(
    node: Dict[str, Any], data: Any, context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Dispatch function for security nodes, integrating with unified_runtime.py.

    FIXED: Safe context handling without mutation of shared state.

    Args:
        node: Dict with node type and params.
        data: Input data for the node.
        context: Runtime context for state and auditing.

    Returns:
        Result of node execution.
    """
    node_type = node.get("type")
    params = node.get("params", {})

    # FIXED: Create isolated context copy for node-specific data
    # This prevents pollution of shared context
    node_context = context.copy()

    # Add node-specific data to isolated context
    for k in ("tensor", "kernel", "ethical_label"):
        if k in node:
            node_context[k] = node[k]

    # Ensure audit_log is shared (not copied)
    if "audit_log" in context:
        node_context["audit_log"] = context["audit_log"]

    if node_type == "EncryptNode":
        agent_id = node.get("agent_id", "default_security_agent")
        return EncryptNode(agent_id).execute(data, params, node_context)
    elif node_type == "PolicyNode":
        return PolicyNode().execute(data, params, node_context)
    else:
        raise ValueError(f"Unknown security node type: {node_type}")


# ============================================================================
# EXPORTED NODES
# ============================================================================
# This class is what the SafetyValidator will import and instantiate.
class SecurityNodes:
    def __init__(self):
        self.nodes = {
            "EncryptNode": EncryptNode,
            "PolicyNode": PolicyNode,
        }
        logger.info("SecurityNodes class initialized with EncryptNode and PolicyNode")

    def get_node(self, node_name: str) -> Optional[Any]:
        """Safely retrieve a node class by its name."""
        return self.nodes.get(node_name)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("SecurityNodes Production Demo")
    print("=" * 70 + "\n")

    # Demo usage with all 2025 policy/ethics/compression/photonic extensions
    context = {"audit_log": [], "ethical_label": "EU2025:Safe"}

    # Initialize KeyManager with agent_id
    key_manager = KeyManager(agent_id="demo_agent")

    # Generate a proper Fernet key
    fernet_key = Fernet.generate_key()
    key_id = "demo_fernet_key"

    # Store key properly
    if hasattr(key_manager, "keys"):
        key_manager.keys[key_id] = fernet_key.decode()

    print("--- Test 1: EncryptNode with AES ---")
    encrypt_node = {
        "type": "EncryptNode",
        "agent_id": "demo_agent",
        "params": {"algorithm": "AES", "key_id": key_id},
        "tensor": [[0.1, 0.2], [0.3, 0.4]],
        "kernel": "def foo(): pass",
        "ethical_label": "EU2025:Safe",
    }
    data = {"sensitive": "user_data"}

    try:
        result = dispatch_security_node(encrypt_node, data, context)
        print(f"Encryption successful: {result['encrypted_data'][:50]}...")
        print(f"Ethical label: {result['audit']['ethical_label']}")
    except Exception as e:
        print(f"Encryption failed: {e}")

    print("\n--- Test 2: PolicyNode (GDPR) ---")
    policy_node = {
        "type": "PolicyNode",
        "params": {"policy": "GDPR", "enforcement": "log"},
        "ethical_label": "EU2025:Safe",
    }

    try:
        result = dispatch_security_node(policy_node, data, context)
        print(f"Policy check: {result['compliance']}")
        print(f"Rationale: {result['audit']['rationale']}")
    except Exception as e:
        print(f"Policy check failed: {e}")

    print("\n--- Test 3: PolicyNode (PII Detection) ---")
    pii_data = {"email": "test@example.com", "ssn": "123-45-6789"}

    try:
        result = dispatch_security_node(policy_node, pii_data, context)
        print(f"Policy check: {result['compliance']}")
    except Exception as e:
        print(f"Expected PII detection: {e}")

    print("\n--- Audit Log ---")
    for i, entry in enumerate(context["audit_log"], 1):
        print(f"\nEntry {i}:")
        print(f"  Node: {entry['node_type']}")
        print(f"  Status: {entry['status']}")
        print(f"  Ethical Label: {entry.get('ethical_label')}")
        if "error" in entry:
            print(f"  Error: {entry['error']}")

    print("\n" + "=" * 70)
    print("Demo Complete")
    print("=" * 70 + "\n")
