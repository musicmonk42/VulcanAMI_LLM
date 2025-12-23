# ====================================================================
# SCRIPT PROLOGUE:
# Add the project's 'src' directory to the Python path.
# This allows the script to find and import the 'unified_runtime' and
# other core modules when run directly.
# ====================================================================
from starlette.responses import JSONResponse, Response
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi import Limiter, _rate_limit_exceeded_handler
from pydantic import BaseModel, Field, field_validator
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from typing import Any, Dict, List, Literal, Optional
from datetime import datetime
from collections import deque
import threading
import re
import logging
import json
import asyncio
import os
import sys
from pathlib import Path

# Get the directory of the current script (src/)
src_root = Path(__file__).resolve().parent
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))
# ====================================================================

# ====================================================================
# LOAD ENVIRONMENT VARIABLES
# ====================================================================
try:
    import os

    from dotenv import load_dotenv

    # Load .env file from project root
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        print(f"✅ Loaded environment variables from: {env_path}")

        # Verify critical keys
        if os.getenv("OPENAI_API_KEY"):
            print("✅ OPENAI_API_KEY loaded successfully")
        else:
            print("⚠️ OPENAI_API_KEY not found in .env")

        if os.getenv("ANTHROPIC_API_KEY"):
            print("✅ ANTHROPIC_API_KEY loaded successfully")

        if os.getenv("GRAPHIX_API_KEY"):
            print("✅ GRAPHIX_API_KEY loaded successfully")
    # FIX: Don't warn about missing .env file - it's optional in containerized environments
    # Environment variables are typically injected via Docker/K8s, not .env files
except ImportError:
    # Silently fall back to system environment variables (expected in containers)
    import os
except Exception as e:
    print(f"❌ Error loading .env: {e}")
    import os
# ====================================================================

# === Auto-apply bootstrap (must run before UnifiedRuntime/VULCAN init) ===

# Ensure repo root (project root) is on sys.path so we can import vulcan.config
_repo_root = Path(__file__).resolve().parent.parent  # e.g., D:/Graphix
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Enable self-improvement globally via env (belt-and-suspenders)
os.environ.setdefault("VULCAN_ENABLE_SELF_IMPROVEMENT", "1")

# Point to your auto-apply policy file (adjust only if it's elsewhere)
os.environ.setdefault(
    "VULCAN_AUTO_APPLY_POLICY", str(_repo_root / "configs" / "auto_apply_policy.yaml")
)

# Flip the intrinsic drive to auto-apply (no human approval) with safe cadence/budgets
try:
    # Use the namespaced module that matches your file: src/vulcan/config.py
    from vulcan.config import set_config

    set_config("intrinsic_drives_config.enabled", True)
    set_config("intrinsic_drives_config.approval_required", False)
    set_config("intrinsic_drives_config.check_interval_seconds", 120)
    set_config("intrinsic_drives_config.max_cost_usd_per_session", 2.0)
    set_config("intrinsic_drives_config.max_cost_usd_per_day", 10.0)
    # Optional: make the policy path visible in config too
    set_config(
        "intrinsic_drives_config.auto_apply_policy",
        os.environ["VULCAN_AUTO_APPLY_POLICY"],
    )
    print(
        "✅ Applied runtime config overrides for auto-apply (approval_required=False)."
    )
except ModuleNotFoundError as e:
    # More specific handling for module not found
    print(
        f"⚠️  Could not apply runtime config overrides for auto-apply: {e}. Relying on environment variables."
    )
except Exception as e:
    # Fail closed to env vars only; if config isn't available, the drive may remain approval-gated.
    print(
        f"⚠️  Could not apply runtime config overrides for auto-apply: {e}. Relying on environment variables."
    )
# === End auto-apply bootstrap ===


# graphix_arena.py
"""
Graphix Arena (Production-Ready)
=================================
Version: 2.0.0 - All issues fixed, thread-safe, validated
Advanced distributed environment for AI agent collaboration and graph evolution.
"""


# Production Readiness Imports

# YAML for tool selection config
try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    yaml = None
    YAML_AVAILABLE = False

# vLLM for distributed LLM sharding
try:
    from vllm import LLM

    VLLM_AVAILABLE = True
except ImportError:
    LLM = None
    VLLM_AVAILABLE = False

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

# Core dependencies
try:
    from unified_runtime import UnifiedRuntime

    UNIFIED_RUNTIME_AVAILABLE = True
except ImportError:
    UnifiedRuntime = None
    UNIFIED_RUNTIME_AVAILABLE = False

try:
    from security_audit_engine import SecurityAuditEngine

    SECURITY_AUDIT_AVAILABLE = True
except ImportError:
    SecurityAuditEngine = None
    SECURITY_AUDIT_AVAILABLE = False

try:
    from llm_client import GraphixLLMClient

    LLM_CLIENT_AVAILABLE = True
except ImportError:
    GraphixLLMClient = None
    LLM_CLIENT_AVAILABLE = False

# Interpretability, auditing, and observability
try:
    from interpretability_engine import InterpretabilityEngine

    INTERPRETABILITY_AVAILABLE = True
except ImportError:
    InterpretabilityEngine = None
    INTERPRETABILITY_AVAILABLE = False

try:
    from nso_aligner import NSOAligner

    NSO_ALIGNER_AVAILABLE = True
except ImportError:
    NSOAligner = None
    NSO_ALIGNER_AVAILABLE = False

try:
    from observability_manager import ObservabilityManager

    OBSERVABILITY_AVAILABLE = True
except ImportError:
    ObservabilityManager = None
    OBSERVABILITY_AVAILABLE = False

# Slack alerts
try:
    from slack_sdk import WebClient

    SLACK_AVAILABLE = True
except ImportError:
    WebClient = None
    SLACK_AVAILABLE = False

# DistributedSharder
try:
    from distributed_sharder import DistributedSharder

    SHARDER_AVAILABLE = True
except ImportError:
    DistributedSharder = None
    SHARDER_AVAILABLE = False

# Registry
try:
    from language_evolution_registry import LanguageEvolutionRegistry

    REGISTRY_AVAILABLE = True
except ImportError:
    # LanguageEvolutionRegistry is in specs/formal_grammar/ - add to path and retry
    try:
        _specs_path = (
            Path(__file__).resolve().parent.parent / "specs" / "formal_grammar"
        )
        if str(_specs_path) not in sys.path:
            sys.path.insert(0, str(_specs_path))
        from language_evolution_registry import LanguageEvolutionRegistry

        REGISTRY_AVAILABLE = True
    except ImportError:
        LanguageEvolutionRegistry = None
        REGISTRY_AVAILABLE = False

# Import registry backends for initialization (needed for v4.0.0+ API)
try:
    from language_evolution_registry import DevelopmentKMS, InMemoryBackend

    REGISTRY_BACKENDS_AVAILABLE = True
except ImportError:
    InMemoryBackend = None
    DevelopmentKMS = None
    REGISTRY_BACKENDS_AVAILABLE = False

# DataAugmentor
try:
    from data_augmentor import DataAugmentor

    AUGMENTOR_AVAILABLE = True
except ImportError:
    DataAugmentor = None
    AUGMENTOR_AVAILABLE = False

# DriftDetector
try:
    from drift_detector import DriftDetector

    DRIFT_DETECTOR_AVAILABLE = True
except ImportError:
    DriftDetector = None
    DRIFT_DETECTOR_AVAILABLE = False

# TournamentManager
try:
    from tournament_manager import TournamentManager

    TOURNAMENT_AVAILABLE = True
except ImportError:
    TournamentManager = None
    TOURNAMENT_AVAILABLE = False

# Prometheus metrics
try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, generate_latest

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    CONTENT_TYPE_LATEST = "text/plain"

    class Counter:
        def __init__(self, *args, **kwargs):
            pass

        def inc(self, *args, **kwargs):
            pass

    def generate_latest():
        return b"# Prometheus not available\n"


# Feedback Protocol
try:
    from feedback_protocol import dispatch_feedback_protocol

    FEEDBACK_PROTOCOL_AVAILABLE = True
except ImportError:
    FEEDBACK_PROTOCOL_AVAILABLE = False

    def dispatch_feedback_protocol(request, context):
        return {"status": "error", "message": "Feedback protocol dispatcher not found."}


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("GraphixArena")

# Constants
MAX_PAYLOAD_SIZE = 10_000_000  # 10MB
MAX_FEEDBACK_LOG_SIZE = 10000
MAX_AGENT_ID_LENGTH = 256
MAX_GRAPH_ID_LENGTH = 256
MAX_NODES = 10000  # FIX: Added max node count for validation
MAX_REBERT_THRESHOLD = 0.5
MIN_REBERT_THRESHOLD = 0.0


# App Initialization
app = FastAPI(
    title="Graphix Arena",
    description="Advanced distributed environment for AI agent collaboration and graph evolution.",
    version="2.0.0",
)

# Setup Rate Limiting
limiter = Limiter(key_func=get_remote_address, default_limits=["100 per minute"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Setup CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security: API Key Authentication
API_KEY = os.getenv("GRAPHIX_API_KEY", "default-secret-key-for-dev")
API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def get_api_key(api_key: str = Security(api_key_header)):
    """Dependency to validate the API key from the request header."""
    if api_key == API_KEY:
        return api_key
    else:
        raise HTTPException(
            status_code=403,
            detail="Could not validate credentials. Invalid or missing API Key.",
        )


# Custom Exceptions
class AgentNotFoundException(Exception):
    """Exception raised when agent is not found."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        super().__init__(f"Agent '{agent_id}' not found")


class BiasDetectedException(Exception):
    """Exception raised when bias is detected in proposal."""

    def __init__(self, agent_id: str, graph_id: str, label: str, message: str):
        self.agent_id = agent_id
        # preserve graph_id even if blank
        self.graph_id = graph_id
        self.label = label
        self.message = message
        super().__init__(message)


@app.exception_handler(AgentNotFoundException)
async def agent_not_found_handler(request: Request, exc: AgentNotFoundException):
    """Handle AgentNotFoundException."""
    return JSONResponse(
        status_code=404,
        content={
            "error": "AgentNotFound",
            "message": f"Agent '{exc.agent_id}' not found.",
        },
    )


@app.exception_handler(BiasDetectedException)
async def bias_detected_handler(request: Request, exc: BiasDetectedException):
    """Handle BiasDetectedException."""
    return JSONResponse(
        status_code=400,
        content={
            "error": "BiasDetected",
            "message": exc.message,
            "details": {
                "agent_id": exc.agent_id,
                "graph_id": exc.graph_id,
                "risk_label": exc.label,
            },
        },
    )


# Prometheus metrics
agent_dropout = Counter(
    "agent_dropout", "Number of agents who failed consensus or dropped out"
)
bias_detections = Counter(
    "bias_detections", "Number of detected bias or risky proposals"
)

# Agent configuration
AGENT_CONFIG = {
    "generator": {
        "task_prompt": "Generate Graphix IR graph from a specification",
        "description": "Creates a new graph from a spec.",
        "input_schema": "GraphSpec",
    },
    "evolver": {
        "task_prompt": "Evolve Graphix IR graph",
        "description": "Mutates an existing graph based on evolution parameters.",
        "input_schema": "GraphixIRGraph",
    },
    "visualizer": {
        "task_prompt": "Visualize Graphix IR graph",
        "description": "Renders a graph into a 3D matrix representation.",
        "input_schema": "GraphixIRGraph",
    },
}


# Pydantic models
class GraphSpec(BaseModel):
    spec_id: str = Field(..., description="Unique specification identifier")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Parameters for the generator agent"
    )

    @field_validator("spec_id")
    @classmethod
    def validate_spec_id(cls, v):
        if not v or len(v) > MAX_GRAPH_ID_LENGTH:
            raise ValueError(f"spec_id must be 1-{MAX_GRAPH_ID_LENGTH} characters")
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            # The test expected a validation failure here, ensuring the model's logic is correct
            raise ValueError(
                "spec_id must contain only alphanumeric, underscore, or hyphen characters"
            )
        return v


class Node(BaseModel):
    id: str
    label: str
    properties: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def validate_id(cls, v):
        if not v or len(v) > 256:
            raise ValueError("Node id must be 1-256 characters")
        return v


class Edge(BaseModel):
    source_id: str
    target_id: str
    weight: float = 1.0
    properties: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("weight")
    @classmethod
    def validate_weight(cls, v):
        if v < 0:
            raise ValueError("Edge weight must be non-negative")
        return v


class GraphixIRGraph(BaseModel):
    graph_id: str
    nodes: List[Node]
    edges: List[Edge]
    properties: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("graph_id")
    @classmethod
    def validate_graph_id(cls, v):
        if not v or len(v) > MAX_GRAPH_ID_LENGTH:
            raise ValueError(f"graph_id must be 1-{MAX_GRAPH_ID_LENGTH} characters")
        # FIX: Implement character validation for graph_id
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError(
                "graph_id must contain only alphanumeric, underscore, or hyphen characters"
            )
        return v

    @field_validator("nodes")
    @classmethod
    def validate_nodes(cls, v):
        if not v:
            raise ValueError("Graph must have at least one node")
        # FIX: Implement maximum node count validation
        if len(v) > MAX_NODES:
            raise ValueError(f"Graph cannot have more than {MAX_NODES} nodes")
        return v

    @field_validator("edges")
    @classmethod
    def validate_edges(cls, v):
        # Add custom validation if needed (e.g., check source/target exist)
        return v


SCHEMA_MAP = {
    "GraphSpec": GraphSpec,
    "GraphixIRGraph": GraphixIRGraph,
}


def rebert_prune(input_tensor, threshold=0.1):
    """
    Prune tensor using ReBERT-style thresholding.

    Args:
        input_tensor: Input tensor to prune
        threshold: Pruning threshold

    Returns:
        Pruned tensor as list
    """
    if not NUMPY_AVAILABLE:
        logger.warning("NumPy not available, skipping pruning")
        return input_tensor

    # Validate threshold
    if not isinstance(threshold, (int, float)):
        threshold = 0.1
    threshold = max(MIN_REBERT_THRESHOLD, min(MAX_REBERT_THRESHOLD, threshold))

    try:
        if not isinstance(input_tensor, np.ndarray):
            input_tensor = np.array(input_tensor)

        mask = np.abs(input_tensor) > threshold
        pruned = input_tensor * mask
        return pruned.tolist()
    except Exception as e:
        logger.error(f"ReBERT pruning failed: {e}")
        return input_tensor


class GraphixArena:
    """
    Production-ready Graphix Arena with comprehensive error handling and validation.
    """

    def __init__(self, port: int = 8181, host: str = "127.0.0.1"):
        """Initialize Graphix Arena.

        Args:
            port: Port to bind to (1024-65535)
            host: Host address to bind to (default: 127.0.0.1 for localhost)
                  Set to "0.0.0.0" to bind to all interfaces (less secure)
        """
        # Validate port
        if not isinstance(port, int) or port < 1024 or port > 65535:
            raise ValueError(f"Port must be between 1024 and 65535, got {port}")

        self.port = port
        self.host = host  # Store host address

        # Thread safety
        self.lock = threading.RLock()

        # Initialize runtime with fallback
        if UNIFIED_RUNTIME_AVAILABLE and UnifiedRuntime is not None:
            self.runtime = UnifiedRuntime()
        else:
            logger.warning("UnifiedRuntime not available, using mock runtime")
            self.runtime = self._create_mock_runtime()

        # Initialize audit with fallback
        if SECURITY_AUDIT_AVAILABLE and SecurityAuditEngine is not None:
            self.audit = SecurityAuditEngine()
        else:
            logger.warning("SecurityAuditEngine not available, using mock audit")
            self.audit = self._create_mock_audit()

        # Initialize LLM client with fallback and error handling
        if LLM_CLIENT_AVAILABLE and GraphixLLMClient is not None:
            try:
                self.llm_client = GraphixLLMClient(agent_id="arena-agent")
                if self.llm_client.is_available:
                    logger.info(
                        "✅ GraphixLLMClient initialized successfully (real mode)"
                    )
                else:
                    logger.info(
                        "✅ GraphixLLMClient initialized (mock mode - OPENAI_API_KEY not configured)"
                    )
            except Exception as e:
                # Log with exc_info=True to capture full traceback even if exception message is empty
                logger.error(
                    f"❌ Unexpected error initializing GraphixLLMClient: {type(e).__name__}: {e}",
                    exc_info=True,
                )
                self.llm_client = None
        else:
            # Only log at debug level since this is expected when openai package is not installed
            logger.debug(
                "GraphixLLMClient not available - openai package may not be installed"
            )
            logger.debug("💡 To enable: pip install openai tenacity")
            self.llm_client = None

        # Agent configuration
        self.agents = AGENT_CONFIG
        logger.info(
            f"Graphix Arena initialized with {len(self.agents)} agents: {list(self.agents.keys())}"
        )

        # Optional components
        self.llm_models = {}

        # Initialize registry with backends (v4.0.0+ API requires backend and kms)
        self.registry = None
        if REGISTRY_AVAILABLE and LanguageEvolutionRegistry:
            if REGISTRY_BACKENDS_AVAILABLE and InMemoryBackend and DevelopmentKMS:
                try:
                    # Check environment to determine which backend/KMS to use
                    env = os.getenv("ENVIRONMENT", "development")

                    if env == "production":
                        logger.error(
                            "Production environment detected but only development backends available. "
                            "Please configure production-grade storage (Redis/Postgres) and KMS (AWS/Azure)."
                        )
                        self.registry = None
                    else:
                        # Development environment - OK to use in-memory backends
                        backend = InMemoryBackend()
                        kms = DevelopmentKMS()
                        self.registry = LanguageEvolutionRegistry(
                            backend=backend, kms=kms
                        )
                        logger.info(
                            "✅ LanguageEvolutionRegistry initialized with InMemoryBackend (development mode)"
                        )
                        logger.warning(
                            "⚠️  Using InMemoryBackend and DevelopmentKMS - NOT FOR PRODUCTION. "
                            "Set ENVIRONMENT=production and configure Redis/AWS KMS for production use."
                        )
                except Exception as e:
                    logger.warning(
                        f"Failed to initialize LanguageEvolutionRegistry with backends: {e}"
                    )
                    self.registry = None
            else:
                # Try legacy initialization (for older versions that don't require backend/kms)
                try:
                    self.registry = LanguageEvolutionRegistry()
                    logger.info(
                        "✅ LanguageEvolutionRegistry initialized (legacy mode)"
                    )
                except TypeError:
                    # v4.0.0+ requires backend/kms but we couldn't import them
                    logger.warning(
                        "LanguageEvolutionRegistry v4.0.0+ requires backend and kms arguments - backends not available for import"
                    )
                    self.registry = None
                except Exception as e:
                    logger.warning(
                        f"Failed to initialize LanguageEvolutionRegistry: {e}"
                    )
                    self.registry = None

        self.data_augmentor = (
            DataAugmentor() if AUGMENTOR_AVAILABLE and DataAugmentor else None
        )
        self.drift_detector = (
            DriftDetector(
                dim=128, drift_threshold=0.1, history=5, realignment_method="center"
            )
            if DRIFT_DETECTOR_AVAILABLE and DriftDetector
            else None
        )
        if self.drift_detector:
            logger.info(
                f"✓ DriftDetector initialized in Arena (dim=128, drift_threshold=0.1, history=5)"
            )
        else:
            logger.warning(f"⚠ DriftDetector unavailable")

        self.tournament_manager = (
            TournamentManager() if TOURNAMENT_AVAILABLE and TournamentManager else None
        )
        if self.tournament_manager:
            logger.info(f"✓ TournamentManager initialized in Arena")
        else:
            logger.warning(f"⚠ TournamentManager unavailable")

        # Bounded feedback log
        self.feedback_log: deque = deque(maxlen=MAX_FEEDBACK_LOG_SIZE)

        # Initialize interpretability components
        self.interpret_engine = (
            InterpretabilityEngine()
            if INTERPRETABILITY_AVAILABLE and InterpretabilityEngine
            else None
        )
        if self.interpret_engine:
            logger.info(
                f"✓ InterpretabilityEngine initialized in Arena (lazy-load ready)"
            )
        else:
            logger.warning(f"⚠ InterpretabilityEngine unavailable")

        # FIX: use the correct class symbol 'NSOAligner' guarded by availability
        self.nso_aligner = (
            NSOAligner() if NSO_ALIGNER_AVAILABLE and (NSOAligner is not None) else None
        )
        self.obs_manager = (
            ObservabilityManager()
            if OBSERVABILITY_AVAILABLE and ObservabilityManager
            else None
        )

        # Slack setup
        self.slack_client = None
        self.slack_channel = None

        if SLACK_AVAILABLE and WebClient:
            slack_token = os.getenv("SLACK_BOT_TOKEN")
            if slack_token:
                try:
                    self.slack_client = WebClient(token=slack_token)
                    self.slack_channel = os.getenv(
                        "SLACK_ALERT_CHANNEL", "#graphix-arena-alerts"
                    )
                    logger.info(
                        f"Slack alerts enabled for channel {self.slack_channel}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to initialize Slack client: {e}")
            else:
                logger.info("SLACK_BOT_TOKEN not set, Slack alerts disabled")

        # Load tool selection configuration
        # FIXED: Look in configs directory for tool_selection.yaml
        self.tool_selection_config = {}
        if YAML_AVAILABLE:
            try:
                config_path = (
                    Path(__file__).parent.parent / "configs" / "tool_selection.yaml"
                )
                if config_path.exists():
                    with open(config_path, "r", encoding="utf-8") as f:
                        self.tool_selection_config = yaml.safe_load(f)
                    logger.info("Loaded tool_selection.yaml configuration.")
                else:
                    logger.warning(
                        f"tool_selection.yaml not found at {config_path}, using default task routing."
                    )
            except Exception as e:
                logger.error(f"Failed to load or parse tool_selection.yaml: {e}")
        else:
            logger.warning(
                "PyYAML not installed, tool selection rules will not be applied."
            )

    def _create_mock_runtime(self):
        """Create mock runtime with checkpoint support."""

        class MockRuntime:
            def __init__(self):
                self.checkpoints = {}

            def get_checkpoint(self, payload: Dict) -> Optional[Dict]:
                """Get checkpoint for payload."""
                key = payload.get("graph_id") or payload.get("spec_id", "default")
                return self.checkpoints.get(key)

            def save_checkpoint(self, payload: Dict, checkpoint: Dict):
                """Save checkpoint."""
                key = payload.get("graph_id") or payload.get("spec_id", "default")
                self.checkpoints[key] = checkpoint

            def restore_checkpoint(self, checkpoint: Dict):
                """Restore from checkpoint."""
                logger.info("Mock checkpoint restoration")

        return MockRuntime()

    def _create_mock_audit(self):
        """Create mock audit engine."""

        class MockAudit:
            def log_event(self, event_type: str, details: Dict):
                """Log audit event."""
                logger.info(f"Audit: {event_type} - {details}")

        return MockAudit()

    def send_slack_alert(self, message: str):
        """Send Slack alert."""
        if not self.slack_client or not self.slack_channel:
            logger.debug("Slack not configured, skipping alert")
            return

        try:
            self.slack_client.chat_postMessage(channel=self.slack_channel, text=message)
            logger.info(f"Sent Slack alert to {self.slack_channel}")
        except Exception as e:
            logger.warning(f"Slack alert failed: {e}")

    async def _run_agent(self, agent_id: str, task: str, data: Dict) -> Dict:
        """
        Run agent task with validation and error handling.

        Args:
            agent_id: Agent identifier
            task: Task description
            data: Input data

        Returns:
            Agent execution result
        """
        # Validate agent_id
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9_-]*$", agent_id):
            raise ValueError(f"Invalid agent_id format: {agent_id}")

        # vLLM distributed inference path
        if (
            agent_id in ("generator", "evolver")
            and VLLM_AVAILABLE
            and SHARDER_AVAILABLE
        ):
            if LLM is not None and DistributedSharder is not None:
                input_data = data.copy()

                # Apply ReBERT pruning if tensor present
                if "input_tensor" in data and NUMPY_AVAILABLE:
                    try:
                        pruned_tensor = rebert_prune(
                            data["input_tensor"], threshold=0.1
                        )
                        input_data["input_tensor"] = pruned_tensor
                        logger.info(
                            f"ReBERT-pruned {agent_id} input tensor (threshold=0.1)"
                        )
                    except Exception as e:
                        logger.warning(f"ReBERT pruning failed: {e}")

                prompt = f"{task}: {json.dumps(input_data)}"
                prompts = [prompt]

                try:
                    logger.info("Executing task on distributed vLLM backend...")
                    llm_outputs = DistributedSharder.llm_distributed_infer(prompts)

                    return {
                        "model_used": "gpt-4o-mini",
                        "tensor_parallel_size": 4,
                        "pruned_input": input_data.get("input_tensor"),
                        "llm_outputs": llm_outputs,
                        "output": f"[Simulated] {agent_id} completed on sharded vLLM backend",
                    }
                except Exception as e:
                    logger.error(f"Distributed sharded LLM inference failed: {e}")
                    raise ValueError(f"LLM distributed inference failed: {e}")

        # Standard subprocess execution
        logger.info(f"Executing task for agent '{agent_id}' via standard subprocess...")

        content_payload = f"{task}: {json.dumps(data)}"
        script_to_execute = (
            f"import json; from llm_client import GraphixLLMClient; "
            f"client=GraphixLLMClient('{agent_id}'); "
            f'messages = [{{"role": "user", "content": {repr(content_payload)}}}]; '
            f"print(json.dumps(client.chat(messages)))"
        )

        cmd = [sys.executable, "-c", script_to_execute]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            # Add timeout
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=300,  # 5 minute timeout
            )

            if proc.returncode != 0:
                error_output = stderr.decode()
                self.audit.log_event(
                    "agent_error", {"agent_id": agent_id, "error": error_output}
                )
                logger.error(
                    f"Agent '{agent_id}' failed. Return Code: {proc.returncode}. Error: {error_output}"
                )
                raise ValueError(f"Agent {agent_id} failed: {error_output}")

            return json.loads(stdout.decode())

        except asyncio.TimeoutError:
            logger.error(f"Agent {agent_id} execution timeout")
            raise ValueError(f"Agent {agent_id} execution timeout")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse agent output: {e}")
            raise ValueError(f"Invalid agent output: {e}")

    def run_transparent_task(
        self, agent_id: str, task_prompt: str, payload: Dict
    ) -> Dict:
        """
        Run transparent task with interpretability and observability.

        Args:
            agent_id: Agent identifier
            task_prompt: Task prompt
            payload: Input payload

        Returns:
            Transparency results
        """
        interpret_result = None
        audit_result = None
        obs_result = None

        # Interpretability
        if self.interpret_engine and "input_tensor" in payload and NUMPY_AVAILABLE:
            try:
                interpret_result = self.interpret_engine.explain_and_trace(
                    np.array(payload["input_tensor"])
                )
            except Exception as e:
                logger.warning(f"Interpretability failed: {e}")

        # NSO audit
        if self.nso_aligner:
            try:
                audit_result = self.nso_aligner.multi_model_audit(payload)
            except Exception as e:
                logger.warning(f"NSO multi-model audit failed: {e}")

        # Observability
        if self.obs_manager and "input_tensor" in payload and NUMPY_AVAILABLE:
            try:
                obs_result = self.obs_manager.log_tensor_semantics(
                    np.array(payload["input_tensor"])
                )
            except Exception as e:
                logger.warning(f"Observability logging failed: {e}")

        return {
            "interpretability": interpret_result,
            "audit": audit_result,
            "observability": obs_result,
        }

    async def run_shadow_task(
        self, agent_id: str, task_prompt: str, payload: Dict
    ) -> Dict:
        """
        Run shadow task with checkpoint and rollback support.

        Args:
            agent_id: Agent identifier
            task_prompt: Task prompt
            payload: Input payload

        Returns:
            Shadow task result
        """
        logger.info(
            f"Running shadow task for agent {agent_id} (graph_id={payload.get('graph_id', 'N/A')})"
        )

        try:
            # Get checkpoint
            checkpoint = None
            if hasattr(self.runtime, "get_checkpoint"):
                checkpoint = self.runtime.get_checkpoint(payload)

            # Run agent
            result = await self._run_agent(agent_id, task_prompt, payload)

            # Check hallucination rate
            hallucination_rate = 0.0
            if isinstance(result, dict):
                hallucination_rate = result.get(
                    "hallucination_rate",
                    result.get("metrics", {}).get("hallucination_rate", 0.0),
                )

            # Rollback if hallucination rate too high
            if hallucination_rate > 0.05:
                logger.warning(
                    f"Hallucination rate {hallucination_rate:.2%} > 5%, rolling back mutation"
                )

                if checkpoint is not None and hasattr(
                    self.runtime, "restore_checkpoint"
                ):
                    self.runtime.restore_checkpoint(checkpoint)

                self.audit.log_event(
                    "shadow_rollback",
                    {
                        "graph_id": payload.get("graph_id"),
                        "agent_id": agent_id,
                        "hallucination_rate": hallucination_rate,
                        "action": "rollback",
                    },
                )

                return {
                    "status": "rollback",
                    "reason": f"Hallucination rate {hallucination_rate:.2%} exceeded threshold",
                    "hallucination_rate": hallucination_rate,
                }

            # Query LTM for similar topologies
            ltm_results = []
            if self.registry and hasattr(self.registry, "find_similar_topologies"):
                try:
                    ltm_results = self.registry.find_similar_topologies(payload)
                    logger.info(
                        f"Queried LTM for similar topologies, found {len(ltm_results)} matches"
                    )
                except Exception as e:
                    logger.error(f"LTM query failed: {e}")

            # Generate augmented data
            augmented = None
            if self.data_augmentor is not None:
                try:
                    augmented = self.data_augmentor.generate_synthetic_proposal(payload)
                except Exception as e:
                    logger.error(f"Synthetic data augmentation failed: {e}")

            return {
                "status": "success",
                "result": result,
                "hallucination_rate": hallucination_rate,
                "similar_topologies": ltm_results,
                "augmented": augmented,
            }

        except Exception as e:
            logger.error(f"Shadow task failed: {e}")
            return {"status": "error", "reason": str(e)}

    async def rollback_failed_task(self, payload: Dict, reason: str = "") -> Dict:
        """
        Rollback failed task.

        Args:
            payload: Task payload
            reason: Rollback reason

        Returns:
            Rollback result
        """
        graph_id = payload.get("graph_id", payload.get("spec_id", "[unknown]"))
        logger.warning(f"Rolling back task for payload {graph_id}. Reason: {reason}")

        # Attempt checkpoint restoration
        if hasattr(self.runtime, "restore_checkpoint") and hasattr(
            self.runtime, "get_checkpoint"
        ):
            try:
                checkpoint = self.runtime.get_checkpoint(payload)
                if checkpoint:
                    self.runtime.restore_checkpoint(checkpoint)
                    logger.info("Successfully restored checkpoint")
            except Exception as e:
                logger.error(f"Checkpoint restoration failed: {e}")

        # Log rollback
        self.audit.log_event("rollback", {"payload": payload, "reason": reason})

        # Update metrics
        agent_dropout.inc()

        return {"status": "rollback", "reason": reason}

    async def run_agent_task(self, agent_id: str, request: Request):
        """
        Run agent task with comprehensive validation.

        Args:
            agent_id: Agent identifier
            request: HTTP request

        Returns:
            Agent task result
        """
        # Validate agent_id format
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9_-]*$", agent_id):
            raise HTTPException(
                status_code=400, detail=f"Invalid agent_id format: {agent_id}"
            )

        if len(agent_id) > MAX_AGENT_ID_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"agent_id too long: {len(agent_id)} > {MAX_AGENT_ID_LENGTH}",
            )

        # Check agent exists
        if agent_id not in self.agents:
            raise AgentNotFoundException(agent_id=agent_id)

        agent_config = self.agents[agent_id]
        task_prompt = agent_config["task_prompt"]
        schema_name = agent_config.get("input_schema")

        # Parse and validate request
        raw_data = await request.json()

        # Validate payload size
        payload_size = len(json.dumps(raw_data))
        if payload_size > MAX_PAYLOAD_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"Payload too large: {payload_size} > {MAX_PAYLOAD_SIZE}",
            )

        # Validate against schema
        if schema_name and schema_name in SCHEMA_MAP:
            try:
                schema = SCHEMA_MAP[schema_name]
                validated_data = schema.model_validate(raw_data)
                payload = validated_data.model_dump()
                graph_id = payload.get("graph_id") or payload.get(
                    "spec_id", "[unknown_id]"
                )
            except Exception as e:
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid input data for agent '{agent_id}': {e}",
                )
        else:
            payload = raw_data
            graph_id = payload.get("id") or payload.get("graph_id", "[unknown_id]")

        # Drift detection and realignment
        drift_info = None
        if (
            self.registry
            and hasattr(self.registry, "get_embeddings")
            and self.drift_detector is not None
        ):
            try:
                embeddings, agent_ids = self.registry.get_embeddings()
                drift_info = self.drift_detector.realign_if_drift(embeddings, agent_ids)

                if drift_info and drift_info.get("realignment_needed"):
                    logger.warning(
                        f"Drift detected (avg_drift={drift_info['avg_drift']:.4f}); "
                        f"realignment triggered for agents: {drift_info['agents_to_realign']}"
                    )
                    self.audit.log_event(
                        "drift_realignment",
                        {
                            "avg_drift": drift_info["avg_drift"],
                            "agents_to_realign": drift_info["agents_to_realign"],
                        },
                    )
            except Exception as e:
                logger.error(f"Drift detection failed: {e}")

        # Security audit
        audit_label = None
        if self.nso_aligner:
            try:
                audit_label = self.nso_aligner.multi_model_audit(payload)

                if audit_label in ("risky", "bias", "unsafe"):
                    bias_detections.inc()

                    alert_msg = (
                        f"[Bias Detected] Agent: {agent_id}, Graph: {graph_id}, "
                        f"Label: {audit_label}. Proposal rejected."
                    )
                    logger.warning(alert_msg)
                    self.send_slack_alert(alert_msg)

                    raise BiasDetectedException(
                        agent_id=agent_id,
                        graph_id=graph_id,
                        label=audit_label,
                        message="Proposal rejected by security audit engine due to potential bias or risk.",
                    )
            except BiasDetectedException:
                raise
            except Exception as e:
                logger.warning(f"Multi-model audit failed: {e}")

        # Transparent task orchestration
        transparent_result = None
        try:
            transparent_result = self.run_transparent_task(
                agent_id, task_prompt, payload
            )
        except Exception as e:
            logger.warning(f"Transparent task orchestration failed: {e}")

        # --- Tool Selection Logic ---
        execution_handler = None
        if self.tool_selection_config:
            payload_str = json.dumps(payload)
            for rule in self.tool_selection_config.get("rules", []):
                conditions = rule.get("conditions", {})
                agent_match = (
                    conditions.get("agent_id") is None
                    or conditions.get("agent_id") == agent_id
                )

                content_match = True
                if "payload_contains" in conditions:
                    content_match = all(
                        cond in payload_str for cond in conditions["payload_contains"]
                    )

                if agent_match and content_match:
                    action = rule.get("action", {})
                    handler_name = action.get("handler")
                    if handler_name:
                        execution_handler = handler_name
                        logger.info(
                            f"Matched tool selection rule: '{rule.get('name', 'Unnamed')}'. Using handler: '{handler_name}'."
                        )
                        break

        # Execute task
        try:
            # Apply tool selection rule if matched, otherwise use default logic
            if execution_handler == "run_shadow_task":
                result = await self.run_shadow_task(agent_id, task_prompt, payload)
            elif execution_handler == "run_agent_subprocess":
                agent_result = await self._run_agent(agent_id, task_prompt, payload)
                result = {"status": "success", "result": agent_result}
            else:
                # Default logic if no rule matched
                if agent_id in ("generator", "evolver"):
                    result = await self.run_shadow_task(agent_id, task_prompt, payload)
                else:
                    agent_result = await self._run_agent(agent_id, task_prompt, payload)
                    result = {"status": "success", "result": agent_result}

            # Log completion
            self.audit.log_event(
                "agent_task_completed",
                {
                    "graph_id": graph_id,
                    "agent": agent_id,
                    "status": result.get("status", "success"),
                    "audit_label": audit_label,
                },
            )

            # Record agent task as AI interaction for meta-learning
            try:
                from vulcan.routing import record_ai_interaction, TELEMETRY_AVAILABLE

                if TELEMETRY_AVAILABLE:
                    record_ai_interaction(
                        interaction_type="agent_communication",
                        sender="arena",
                        receiver=agent_id,
                        message_type="task_execution",
                        query=task_prompt[:200] if task_prompt else "",
                        result={"status": result.get("status", "success")},
                        metadata={
                            "graph_id": graph_id,
                            "audit_label": audit_label,
                            "has_drift_info": drift_info is not None,
                        },
                    )
            except ImportError:
                pass  # Routing not available
            except Exception as e:
                logger.debug(f"Agent task telemetry recording failed: {e}")

            # Add supplementary data
            if drift_info is not None:
                result["drift_info"] = drift_info

            if transparent_result is not None:
                result["transparency"] = transparent_result

            return result

        except Exception as e:
            logger.error(f"Failed agent task for {agent_id} on {graph_id}: {e}")
            await self.rollback_failed_task(payload, reason=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    async def run_tournament_task(self, request: Request):
        """
        Run tournament task.

        Args:
            request: HTTP request

        Returns:
            Tournament result
        """
        if not self.tournament_manager:
            raise HTTPException(
                status_code=503, detail="TournamentManager not available"
            )

        try:
            data = await request.json()

            # Validate payload size
            payload_size = len(json.dumps(data))
            if payload_size > MAX_PAYLOAD_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"Payload too large: {payload_size} > {MAX_PAYLOAD_SIZE}",
                )

            proposals = data.get("proposals", [])
            fitness = data.get("fitness", [])

            # Validate inputs
            if not proposals:
                raise ValueError("proposals list cannot be empty")

            if not fitness:
                raise ValueError("fitness list cannot be empty")

            if len(proposals) != len(fitness):
                raise ValueError(
                    f"proposals and fitness length mismatch: {len(proposals)} vs {len(fitness)}"
                )

            # Get embedding function
            embedding_func = None
            if "embedding_func" in data:
                func_name = data["embedding_func"]
                if self.registry and hasattr(self.registry, func_name):
                    embedding_func = getattr(self.registry, func_name)

            # Default embedding function
            if embedding_func is None and NUMPY_AVAILABLE:

                def embedding_func(p):
                    return np.random.rand(128).astype("float32")

            elif embedding_func is None:
                raise ValueError(
                    "NumPy not available and no embedding function provided"
                )

            # Run tournament
            meta = {}
            winner_indices = self.tournament_manager.run_adaptive_tournament(
                proposals, fitness, embedding_func, meta=meta
            )

            # Record tournament as AI-to-AI interaction for meta-learning
            try:
                from vulcan.routing import record_ai_interaction, TELEMETRY_AVAILABLE

                if TELEMETRY_AVAILABLE:
                    record_ai_interaction(
                        interaction_type="tournament",
                        sender="arena",
                        receiver="tournament_manager",
                        message_type="tournament_run",
                        query=f"Tournament with {len(proposals)} proposals",
                        result={"winner_count": len(winner_indices), "meta": meta},
                        metadata={
                            "proposals_count": len(proposals),
                            "diversity_penalty": self.tournament_manager.diversity_penalty,
                            "winner_percentage": self.tournament_manager.winner_percentage,
                        },
                    )
            except ImportError:
                pass  # Routing not available
            except Exception as e:
                logger.debug(f"Tournament telemetry recording failed: {e}")

            return {"winners": winner_indices, "meta": meta}

        except ValueError as e:
            logger.error(f"Tournament validation failed: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Tournament task failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def feedback_ingestion(self, request: Request):
        """
        Ingest feedback.

        Args:
            request: HTTP request

        Returns:
            Feedback ingestion result
        """
        try:
            feedback = await request.json()

            # Validate required fields
            required_keys = {"graph_id", "agent_id", "score"}
            missing_keys = required_keys - set(feedback.keys())

            if missing_keys:
                raise HTTPException(
                    status_code=422, detail=f"Missing required fields: {missing_keys}"
                )

            # Validate score
            if not isinstance(feedback.get("score"), (int, float)):
                raise HTTPException(status_code=422, detail="score must be numeric")

            # Store feedback (bounded deque)
            with self.lock:
                self.feedback_log.append(feedback)

            logger.info(f"Feedback received: {feedback}")
            self.audit.log_event("feedback_ingested", feedback)

            # Handle negative feedback
            if feedback.get("score", 1.0) < 0.0 or feedback.get("failed_consensus"):
                await self.rollback_failed_task(
                    feedback, reason="Negative feedback/consensus failure"
                )

            return {"status": "ok", "message": "Feedback recorded"}

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Feedback ingestion failed: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    async def metrics(self):
        """Get Prometheus metrics."""
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    def start(self):
        """
        Start Graphix Arena server.

        NOTE: This is NOT async - uvicorn.run is synchronous.
        """
        import uvicorn

        logger.info(f"Starting Graphix Arena on http://{self.host}:{self.port}")
        if (
            self.host == "0.0.0.0"
        ):  # nosec B104 - This is a security check, not a binding
            logger.warning(
                "⚠️ Binding to 0.0.0.0 (all interfaces) - ensure firewall is configured!"
            )
        logger.info(
            "Security is ENABLED. Use the 'X-API-KEY' header for authentication."
        )
        logger.info(
            "Rate limiting is ENABLED. Default: 100 requests per minute per IP."
        )

        # uvicorn.run is NOT async - do not await
        uvicorn.run(app, host=self.host, port=self.port, log_level="info")


class FeedbackQueryParams(BaseModel):
    proposal_id: Optional[str] = None
    limit: int = 10


class FeedbackDispatchNode(BaseModel):
    type: Literal["FeedbackProtocol", "FeedbackQueryNode"] = "FeedbackProtocol"
    proposal_id: Optional[str] = None
    score: Optional[float] = None
    rationale: Optional[str] = ""
    params: Optional[FeedbackQueryParams] = None


# Endpoint: Feedback
@limiter.limit("60/minute")
@app.post(
    "/api/feedback_dispatch",
    dependencies=[Depends(get_api_key)],
    summary="Dispatch feedback protocol",
)
async def feedback_endpoint(payload: FeedbackDispatchNode, request: Request):
    """
    Handle feedback submission via feedback protocol dispatcher.

    Send either:
    {
      "type": "FeedbackProtocol",
      "proposal_id": "abc123",
      "score": 0.9,
      "rationale": "Looks good"
    }

    or

    {
      "type": "FeedbackQueryNode",
      "params": {
        "proposal_id": "abc123",
        "limit": 5
      }
    }
    """
    try:
        data = payload.model_dump(exclude_none=True)

        # Validate payload size
        payload_size = len(json.dumps(data))
        if payload_size > MAX_PAYLOAD_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"Payload too large: {payload_size} > {MAX_PAYLOAD_SIZE}",
            )

        context = {"audit_log": []}
        result = dispatch_feedback_protocol(data, context)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Feedback endpoint error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/feedback_dispatch", include_in_schema=False)
async def feedback_dispatch_info():
    return {
        "message": "Use POST with Content-Type: application/json and a JSON body.",
        "examples": {
            "submit": {
                "type": "FeedbackProtocol",
                "proposal_id": "example_proposal",
                "score": 0.95,
                "rationale": "High quality",
            },
            "query": {
                "type": "FeedbackQueryNode",
                "params": {"proposal_id": "example_proposal", "limit": 10},
            },
        },
    }


@app.get("/", summary="Service status")
async def root_status():
    return {
        "service": "Graphix Arena",
        "version": "2.0.0",
        "endpoints": [
            "/api/run/{agent_id}",
            "/api/feedback",
            "/api/feedback_dispatch",
            "/api/tournament",
            "/api/metrics",
        ],
    }


@app.get("/health", summary="Arena health", tags=["health"])
async def arena_health():
    """
    Lightweight health endpoint for unified platform aggregation.
    Returns basic operational status without invoking heavy subsystems.
    """
    try:
        # Collect minimal diagnostics
        components = {
            "runtime": bool(getattr(_ARENA_INSTANCE, "runtime", None)),
            "audit": bool(getattr(_ARENA_INSTANCE, "audit", None)),
            "llm_client": bool(getattr(_ARENA_INSTANCE, "llm_client", None)),
            "registry": bool(getattr(_ARENA_INSTANCE, "registry", None)),
            "data_augmentor": bool(getattr(_ARENA_INSTANCE, "data_augmentor", None)),
            "drift_detector": bool(getattr(_ARENA_INSTANCE, "drift_detector", None)),
            "tournament_manager": bool(
                getattr(_ARENA_INSTANCE, "tournament_manager", None)
            ),
            "interpretability": bool(
                getattr(_ARENA_INSTANCE, "interpret_engine", None)
            ),
            "nso_aligner": bool(getattr(_ARENA_INSTANCE, "nso_aligner", None)),
            "observability": bool(getattr(_ARENA_INSTANCE, "obs_manager", None)),
            "slack": bool(getattr(_ARENA_INSTANCE, "slack_client", None)),
        }

        # Simple readiness heuristic: core runtime + audit present
        ready = components["runtime"] and components["audit"]

        # Feedback log size (if initialized)
        feedback_len = 0
        if _ARENA_INSTANCE and hasattr(_ARENA_INSTANCE, "feedback_log"):
            try:
                feedback_len = len(_ARENA_INSTANCE.feedback_log)
            except Exception:
                feedback_len = -1

        return {
            "service": "Graphix Arena",
            "version": "2.0.0",
            "status": "healthy" if ready else "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "components": components,
            "feedback_log_size": feedback_len,
        }
    except Exception as e:
        # Fail closed but return JSON
        return {
            "service": "Graphix Arena",
            "version": "2.0.0",
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }


# ========== NEW: central route registration ==========


def register_routes(arena: "GraphixArena"):
    """
    Register API-key protected & rate-limited routes so they exist
    both when running with uvicorn and when running as a script.
    This is idempotent and will not double-register paths.
    """
    existing_paths = {route.path for route in app.router.routes}

    def add_once(path: str, endpoint, limit: str, methods: List[str]):
        if path not in existing_paths:
            app.add_api_route(
                path,
                limiter.limit(limit)(endpoint),
                methods=methods,
                dependencies=[Depends(get_api_key)],
            )

    # Protected POST routes
    add_once("/api/run/{agent_id}", arena.run_agent_task, "30/minute", ["POST"])
    add_once("/api/feedback", arena.feedback_ingestion, "60/minute", ["POST"])
    add_once("/api/tournament", arena.run_tournament_task, "20/minute", ["POST"])

    # Metrics (no API key, allow scraping)
    if "/api/metrics" not in existing_paths:
        app.add_api_route("/api/metrics", arena.metrics, methods=["GET"])


# Instantiate and register at import time so `uvicorn src.graphix_arena:app` gets all endpoints
try:
    _ARENA_INSTANCE: Optional["GraphixArena"] = GraphixArena()
    register_routes(_ARENA_INSTANCE)
    logger.info("Registered protected routes at import time for uvicorn.")
except Exception as e:
    logger.error(f"Failed to initialize GraphixArena at import: {e}")
    _ARENA_INSTANCE = None

# ========== END NEW SECTION ==========


# Main execution
if __name__ == "__main__":
    # Create arena instance (use already-created instance if available)
    arena = _ARENA_INSTANCE or GraphixArena()

    # Ensure routes are registered (idempotent)
    register_routes(arena)

    # Start server (synchronous)
    arena.start()
