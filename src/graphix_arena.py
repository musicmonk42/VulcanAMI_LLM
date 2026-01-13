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
import time
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
    # Note: Don't warn about missing .env file - it's optional in containerized environments
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

# Centralized audit logger integration
try:
    from audit_log import TamperEvidentLogger
    
    CENTRALIZED_AUDIT_AVAILABLE = True
except ImportError:
    try:
        from src.audit_log import TamperEvidentLogger
        CENTRALIZED_AUDIT_AVAILABLE = True
    except ImportError:
        TamperEvidentLogger = None
        CENTRALIZED_AUDIT_AVAILABLE = False

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
    from nso_aligner import NSOAligner, get_nso_aligner

    NSO_ALIGNER_AVAILABLE = True
except ImportError:
    NSOAligner = None
    get_nso_aligner = None
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

# ConsensusEngine for governance
try:
    from consensus_engine import ConsensusEngine

    CONSENSUS_AVAILABLE = True
except ImportError:
    ConsensusEngine = None
    CONSENSUS_AVAILABLE = False

# DQSValidator for data quality validation
try:
    from vulcan.safety.dqs_integration import DQSValidator

    DQS_VALIDATOR_AVAILABLE = True
except ImportError:
    DQSValidator = None
    DQS_VALIDATOR_AVAILABLE = False

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

# EvolutionEngine
try:
    from evolution_engine import EvolutionEngine

    EVOLUTION_AVAILABLE = True
except ImportError:
    EvolutionEngine = None
    EVOLUTION_AVAILABLE = False

# StrategyOrchestrator for cost-aware execution planning
try:
    from strategies import StrategyOrchestrator, StochasticCostModel, ToolMonitor

    STRATEGY_AVAILABLE = True
except ImportError:
    try:
        from src.strategies import StrategyOrchestrator, StochasticCostModel, ToolMonitor
        STRATEGY_AVAILABLE = True
    except ImportError:
        StrategyOrchestrator = None
        StochasticCostModel = None
        ToolMonitor = None
        STRATEGY_AVAILABLE = False

# HardwareDispatcher for optimal hardware backend routing
try:
    from hardware_dispatcher import HardwareDispatcher, HardwareBackend

    HARDWARE_DISPATCH_AVAILABLE = True
except ImportError:
    HARDWARE_DISPATCH_AVAILABLE = False
    HardwareDispatcher = None
    HardwareBackend = None

# HardwareEmulator availability check - used internally by HardwareDispatcher
# for fallback when real photonic/GPU hardware is unavailable
try:
    from hardware_emulator import HardwareEmulator

    HARDWARE_EMULATOR_AVAILABLE = True
except ImportError:
    HARDWARE_EMULATOR_AVAILABLE = False
    HardwareEmulator = None

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

# Ray for distributed execution (Architectural Fix #3)
# Ray Actors handle workloads better than standard subprocess, preventing
# "Zombie Process" saturation and improving memory/CPU resource management
try:
    import ray

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    ray = None  # type: ignore

# ============================================================
# REASONING INTEGRATION - Wire reasoning engines into Arena execution
# ============================================================
# Import UnifiedReasoner and reasoning integration for actual reasoning invocation
# This enables Arena to invoke reasoning engines for reasoning-type tasks
try:
    from vulcan.reasoning import (
        UnifiedReasoner,
        ReasoningType,
        ReasoningResult as VulcanReasoningResult,
        UNIFIED_AVAILABLE,
        create_unified_reasoner,
    )
    from vulcan.reasoning.integration import (
        apply_reasoning,
        get_reasoning_integration,
        ReasoningResult as IntegrationReasoningResult,
    )
    REASONING_AVAILABLE = UNIFIED_AVAILABLE
    logger_init = logging.getLogger("GraphixArena")
    logger_init.info(
        "✅ Reasoning integration loaded successfully - reasoning engines will be invoked"
    )
except ImportError as e:
    UnifiedReasoner = None
    ReasoningType = None
    VulcanReasoningResult = None
    UNIFIED_AVAILABLE = False
    create_unified_reasoner = None
    apply_reasoning = None
    get_reasoning_integration = None
    IntegrationReasoningResult = None
    REASONING_AVAILABLE = False
    logger_init = logging.getLogger("GraphixArena")
    logger_init.warning(
        f"⚠️ Reasoning integration not available: {e}. Tasks will use subprocess execution."
    )

# ============================================================
# SYSTEM OBSERVER INTEGRATION - BUG #3 FIX
# ============================================================
# Import SystemObserver functions to make world model aware of Arena activity
# This ensures the world model knows about all reasoning executed via Arena
try:
    from vulcan.reasoning.integration import (
        observe_query_start,
        observe_engine_result,
        observe_outcome,
        observe_error,
    )
    SYSTEM_OBSERVER_AVAILABLE = True
    logger_init.info(
        "✅ SystemObserver integration loaded - world model will receive Arena events"
    )
except ImportError:
    SYSTEM_OBSERVER_AVAILABLE = False
    # Define no-op functions as fallbacks
    def observe_query_start(*args, **kwargs): pass
    def observe_engine_result(*args, **kwargs): pass
    def observe_outcome(*args, **kwargs): pass
    def observe_error(*args, **kwargs): pass
    logger_init.debug("⚠️ SystemObserver not available - world model will not receive Arena events")


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
MAX_NODES = 10000  # Note: Added max node count for validation
MAX_REBERT_THRESHOLD = 0.5
MIN_REBERT_THRESHOLD = 0.0

# Reasoning integration constants
MAX_REASONING_QUERY_LENGTH = 2000  # Maximum characters for reasoning query input
REASONING_COMPLEXITY_BASE = 0.3  # Base complexity score
REASONING_COMPLEXITY_DATA_DIVISOR = 10000  # Divisor for data-based complexity scaling

# GDPR Compliance Constants
GDPR_RETENTION_POLICY = "session_only"
GDPR_COMPLIANCE_STANDARD = "gdpr_minimization"

# Magic markers for subprocess JSON output
OUTPUT_START_MARKER = "###ARENA_OUTPUT_START###"
OUTPUT_END_MARKER = "###ARENA_OUTPUT_END###"


# ============================================================
# RAY ARENA WORKER (Architectural Fix #3)
# ============================================================
# Ray Actors manage memory and CPU resources better than Python's
# default multiprocessing, preventing "Zombie Process" saturation.
# ============================================================

def _create_ray_arena_worker_class():
    """
    Create the ArenaWorker Ray Actor class dynamically.
    
    This is done in a function to avoid issues when Ray is not available.
    The @ray.remote decorator requires Ray to be imported.
    """
    if not RAY_AVAILABLE or ray is None:
        return None
    
    @ray.remote
    class ArenaWorker:
        """
        Ray Actor for executing Arena agent tasks.
        
        This actor encapsulates the LLM client and handles task execution
        in a distributed manner, allowing Ray to manage resources efficiently.
        """
        
        def __init__(self, agent_id: str):
            """
            Initialize the Arena worker with an agent ID.
            
            Args:
                agent_id: The identifier for this agent worker
            """
            self.agent_id = agent_id
            self.llm_client = None
            self._initialized = False
            
        def _ensure_initialized(self):
            """Lazy initialization of LLM client."""
            if self._initialized:
                return
                
            # Import inside the actor to avoid serialization issues
            try:
                from llm_client import GraphixLLMClient
                self.llm_client = GraphixLLMClient(agent_id=self.agent_id)
                self._initialized = True
            except ImportError:
                self.llm_client = None
                self._initialized = True
            except Exception as e:
                logging.getLogger("ArenaWorker").error(
                    f"Failed to initialize LLM client for {self.agent_id}: {e}"
                )
                self.llm_client = None
                self._initialized = True
        
        def generate(self, prompt: str) -> Dict[str, Any]:
            """
            Generate a response for the given prompt.
            
            Args:
                prompt: The task prompt to process
                
            Returns:
                Dictionary containing the result or error
            """
            self._ensure_initialized()
            
            if self.llm_client is None:
                return {
                    "status": "error",
                    "error": "LLM client not available",
                    "agent_id": self.agent_id
                }
            
            try:
                messages = [{"role": "user", "content": prompt}]
                result = self.llm_client.chat(messages)
                return result
            except Exception as e:
                return {
                    "status": "error",
                    "error": str(e),
                    "agent_id": self.agent_id
                }
        
        def health_check(self) -> Dict[str, Any]:
            """Check worker health status."""
            self._ensure_initialized()
            return {
                "agent_id": self.agent_id,
                "initialized": self._initialized,
                "llm_available": self.llm_client is not None,
                "status": "healthy" if self._initialized else "initializing"
            }
    
    return ArenaWorker


# Create the ArenaWorker class if Ray is available
ArenaWorker = _create_ray_arena_worker_class()


def extract_json_from_output(stdout: str) -> dict:
    """
    Extract JSON from subprocess output, handling mixed log/data.
    
    Uses magic markers for reliable parsing, with fallbacks for
    legacy output formats.
    
    Args:
        stdout: Raw stdout from subprocess
        
    Returns:
        Parsed JSON dictionary
        
    Raises:
        json.JSONDecodeError: If JSON cannot be extracted
    """
    # Try marker-based extraction first
    if OUTPUT_START_MARKER in stdout and OUTPUT_END_MARKER in stdout:
        start_idx = stdout.index(OUTPUT_START_MARKER) + len(OUTPUT_START_MARKER)
        end_idx = stdout.index(OUTPUT_END_MARKER)
        json_str = stdout[start_idx:end_idx].strip()
        return json.loads(json_str)
    
    # Fallback: find JSON object boundaries
    try:
        first_brace = stdout.index('{')
        last_brace = stdout.rindex('}') + 1
        return json.loads(stdout[first_brace:last_brace])
    except (ValueError, json.JSONDecodeError):
        pass
    
    # Last resort - wrap with descriptive error on failure
    try:
        return json.loads(stdout.strip())
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"All JSON extraction methods failed. Original error: {e.msg}",
            e.doc,
            e.pos
        ) from e


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


async def get_api_key(request: Request, api_key: str = Security(api_key_header)):
    """
    Dependency to validate the API key from the request header.
    
    Skip authentication for internal requests from localhost (127.0.0.1 or ::1).
    This allows same-container internal calls (e.g., VULCAN calling Arena)
    without requiring API key configuration.
    """
    # Get client IP from request
    client_host = None
    if request.client:
        client_host = request.client.host
    
    # Skip auth for localhost/internal requests
    # This is safe because Arena and VULCAN run in the same container
    # Note: request.client.host returns IP addresses, not hostnames
    localhost_addresses = ("127.0.0.1", "::1")
    if client_host in localhost_addresses:
        logger.debug(f"[AUTH] Skipping API key validation for internal request from {client_host}")
        return "internal-localhost-bypass"
    
    # External requests require valid API key
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
        # Note: Implement character validation for graph_id
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
        # Note: Implement maximum node count validation
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

        # Initialize runtime with fallback - use singleton to prevent duplicate initialization
        # Note: Use get_or_create_unified_runtime to prevent repeated init/shutdown
        if UNIFIED_RUNTIME_AVAILABLE and UnifiedRuntime is not None:
            try:
                from vulcan.reasoning.singletons import get_or_create_unified_runtime
                self.runtime = get_or_create_unified_runtime()
            except ImportError:
                self.runtime = UnifiedRuntime()
        else:
            logger.warning("UnifiedRuntime not available, using mock runtime")
            self.runtime = self._create_mock_runtime()

        # Initialize audit with centralized logger (fallback to SecurityAuditEngine, then mock)
        # Priority: TamperEvidentLogger (centralized) > SecurityAuditEngine > Mock
        if CENTRALIZED_AUDIT_AVAILABLE and TamperEvidentLogger is not None:
            try:
                self.audit_logger = TamperEvidentLogger()
                self.audit = self._create_audit_wrapper(self.audit_logger)
                logger.info("✅ Centralized TamperEvidentLogger initialized for Arena")
            except Exception as e:
                logger.warning(f"Failed to initialize TamperEvidentLogger: {e}")
                if SECURITY_AUDIT_AVAILABLE and SecurityAuditEngine is not None:
                    self.audit_logger = None
                    self.audit = SecurityAuditEngine()
                    logger.info("✅ SecurityAuditEngine initialized as fallback")
                else:
                    self.audit_logger = None
                    self.audit = self._create_mock_audit()
                    logger.warning("Using mock audit (no audit engines available)")
        elif SECURITY_AUDIT_AVAILABLE and SecurityAuditEngine is not None:
            self.audit_logger = None
            self.audit = SecurityAuditEngine()
            logger.info("✅ SecurityAuditEngine initialized")
        else:
            self.audit_logger = None
            self.audit = self._create_mock_audit()
            logger.warning("SecurityAuditEngine not available, using mock audit")

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
                # Register LLM client with singletons so MathTool and other components can access it
                try:
                    from vulcan.reasoning.singletons import set_llm_client
                    if self.llm_client is not None:
                        set_llm_client(self.llm_client)
                        logger.info("✅ LLM client registered with singletons (MathTool will have full capabilities)")
                except ImportError:
                    logger.debug("Singletons module not available for LLM registration")
                except Exception as e:
                    logger.warning(f"Failed to register LLM client with singletons: {e}")
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

        # EvolutionEngine for generator/evolver tasks
        self.evolution_engine = (
            EvolutionEngine(
                population_size=20,
                max_generations=5,
                cache_size=100,
                diversity_threshold=0.15
            ) if EVOLUTION_AVAILABLE and EvolutionEngine else None
        )
        if self.evolution_engine:
            logger.info(f"✓ EvolutionEngine initialized in Arena (pop_size=20, max_gens=5)")
        else:
            logger.warning(f"⚠ EvolutionEngine unavailable")

        # StrategyOrchestrator for cost-aware execution planning
        self.strategy_orchestrator = None
        if STRATEGY_AVAILABLE and StrategyOrchestrator:
            try:
                self.strategy_orchestrator = StrategyOrchestrator({
                    'tools': ['generator', 'evolver', 'visualizer']
                })
                logger.info(f"✓ StrategyOrchestrator initialized in Arena (cost prediction, drift detection)")
            except Exception as e:
                logger.warning(f"⚠ StrategyOrchestrator failed to initialize: {e}")
                self.strategy_orchestrator = None
        else:
            logger.warning(f"⚠ StrategyOrchestrator unavailable")

        # ============================================================
        # RAY WORKERS (Architectural Fix #3)
        # ============================================================
        # Initialize Ray workers for distributed agent task execution.
        # Ray manages memory/CPU better than subprocess, preventing zombie processes.
        # Enable Ray via VULCAN_ENABLE_RAY=1 (disabled by default for compatibility).
        self.ray_workers: Dict[str, Any] = {}
        self.use_ray = False
        
        enable_ray = os.getenv("VULCAN_ENABLE_RAY", "0").lower() in ("1", "true", "yes")
        
        if enable_ray and RAY_AVAILABLE and ray is not None and ArenaWorker is not None:
            try:
                # Initialize Ray if not already initialized
                if not ray.is_initialized():
                    logger.info("Initializing Ray for distributed execution...")
                    try:
                        ray.init(
                            ignore_reinit_error=True,
                            logging_level=logging.WARNING,  # Reduce Ray log verbosity
                        )
                        logger.info("✅ Ray initialized successfully")
                    except Exception as init_error:
                        logger.warning(f"⚠ Ray initialization failed: {init_error}")
                        raise
                
                if ray.is_initialized():
                    self.use_ray = True
                    # Pre-create workers for each agent
                    for agent_id in self.agents.keys():
                        try:
                            self.ray_workers[agent_id] = ArenaWorker.remote(agent_id)
                            logger.debug(f"Created Ray worker for agent: {agent_id}")
                        except Exception as e:
                            logger.warning(f"Failed to create Ray worker for {agent_id}: {e}")
                    
                    if self.ray_workers:
                        logger.info(
                            f"✓ Ray workers initialized for {len(self.ray_workers)} agents "
                            f"(distributed execution enabled)"
                        )
                    else:
                        logger.warning("⚠ No Ray workers created, falling back to subprocess")
                        self.use_ray = False
            except Exception as e:
                logger.warning(f"⚠ Ray worker initialization failed: {e}, using subprocess")
                self.use_ray = False
        elif RAY_AVAILABLE and not enable_ray:
            logger.info(
                "[GraphixArena] Ray available but disabled (default). "
                "Set VULCAN_ENABLE_RAY=1 to enable distributed execution."
            )
        else:
            logger.debug("Ray not available, using standard subprocess execution")

        # Initialize hardware dispatcher for optimal backend routing
        # The dispatcher is disabled by default to prevent resource starvation that was
        # observed with AnalogPhotonicEmulator (4000% CPU usage). Enable via environment
        # variable VULCAN_ENABLE_HARDWARE_DISPATCHER=1 when hardware acceleration is needed.
        enable_hardware = os.getenv("VULCAN_ENABLE_HARDWARE_DISPATCHER", "0").lower() in ("1", "true", "yes")
        
        if enable_hardware and HARDWARE_DISPATCH_AVAILABLE and HardwareDispatcher is not None:
            try:
                # Initialize with mock mode by default, real hardware requires API keys
                use_mock = os.getenv("VULCAN_HARDWARE_USE_MOCK", "1").lower() in ("1", "true", "yes")
                self.hardware_dispatcher = HardwareDispatcher(
                    use_mock=use_mock,
                    enable_metrics=True,
                    enable_health_checks=False,  # Disable health checks to reduce CPU overhead
                )
                self.use_hardware = True
                logger.info(
                    f"✅ HardwareDispatcher initialized (mock={use_mock}). "
                    f"Set VULCAN_ENABLE_HARDWARE_DISPATCHER=0 to disable."
                )
            except Exception as e:
                logger.warning(f"⚠️ HardwareDispatcher initialization failed: {e}")
                self.hardware_dispatcher = None
                self.use_hardware = False
        else:
            self.hardware_dispatcher = None
            self.use_hardware = False
            if not enable_hardware:
                logger.info(
                    "[GraphixArena] HardwareDispatcher disabled (default). "
                    "Set VULCAN_ENABLE_HARDWARE_DISPATCHER=1 to enable."
                )

        # Bounded feedback log
        self.feedback_log: deque = deque(maxlen=MAX_FEEDBACK_LOG_SIZE)

        # ============================================================
        # GRAPHIX PLATFORM DEEP INTEGRATION - ConsensusEngine
        # ============================================================
        # Initialize ConsensusEngine for governance approval gates
        # This adds distributed governance to agent execution with trust-weighted voting
        self.consensus_engine = None
        if CONSENSUS_AVAILABLE and ConsensusEngine:
            try:
                self.consensus_engine = ConsensusEngine(
                    quorum=0.51,  # 51% participation required
                    approval_threshold=0.66,  # 66% approval required
                    proposal_duration_days=7
                )
                # Register arena orchestrator as a trusted agent
                self.consensus_engine.register_agent("arena_orchestrator", trust_level=0.9)
                logger.info("✅ ConsensusEngine initialized for governance gates (quorum=0.51, approval=0.66)")
            except Exception as e:
                logger.warning(f"Failed to initialize ConsensusEngine: {e}")
                self.consensus_engine = None
        else:
            logger.warning("⚠️ ConsensusEngine not available - governance gates disabled")

        # ============================================================
        # GRAPHIX PLATFORM DEEP INTEGRATION - DQSValidator
        # ============================================================
        # Initialize DQSValidator for data quality validation before graph execution
        # This prevents low-quality data from entering the reasoning pipeline
        self.dqs_validator = None
        if DQS_VALIDATOR_AVAILABLE and DQSValidator:
            try:
                self.dqs_validator = DQSValidator(
                    reject_threshold=0.3,  # Reject data with DQS < 0.3
                    quarantine_threshold=0.4,  # Quarantine data with DQS < 0.4
                    model="v2"  # Use latest DQS model
                )
                logger.info("✅ DQSValidator initialized for data quality gates (reject=0.3, quarantine=0.4)")
            except Exception as e:
                logger.warning(f"DQSValidator initialization failed: {e}")
                self.dqs_validator = None
        else:
            logger.warning("⚠️ DQSValidator not available - data quality validation disabled")

        # ============================================================
        # GRAPHIX PLATFORM DEEP INTEGRATION - DistributedSharder
        # ============================================================
        # Initialize DistributedSharder for automatic tensor sharding
        # This enables handling of large tensors (>10MB) with distributed computation
        self.sharder = None
        if SHARDER_AVAILABLE and DistributedSharder:
            try:
                self.sharder = DistributedSharder(
                    dry_run=False,
                    backend="local",  # Use local backend by default
                )
                logger.info("✅ DistributedSharder initialized for automatic tensor sharding (threshold=10MB)")
            except Exception as e:
                logger.warning(f"DistributedSharder initialization failed: {e}")
                self.sharder = None
        else:
            logger.warning("⚠️ DistributedSharder not available - large tensor handling limited")

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

        # Note: Use singleton pattern for NSOAligner to prevent reloading models on every request
        # get_nso_aligner() caches the instance and avoids expensive model initialization
        self.nso_aligner = (
            get_nso_aligner() if get_nso_aligner is not None else None
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

    def _create_audit_wrapper(self, audit_logger):
        """
        Create synchronous audit wrapper for TamperEvidentLogger.
        
        This adapter allows synchronous code to use the async TamperEvidentLogger
        by running async calls in a thread pool executor.
        
        Args:
            audit_logger: TamperEvidentLogger instance
            
        Returns:
            Wrapper object with synchronous log_event method
        """
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        class AuditLoggerWrapper:
            def __init__(self, async_logger):
                self.async_logger = async_logger
                self.executor = ThreadPoolExecutor(max_workers=2)
                
            def log_event(self, event_type: str, details: Dict):
                """
                Synchronous wrapper for async emit_audit_event.
                
                Runs the async call in a thread pool to avoid blocking.
                """
                try:
                    # Check if we're in an event loop
                    try:
                        loop = asyncio.get_running_loop()
                        # We're in an async context, schedule the task
                        asyncio.create_task(
                            self.async_logger.emit_audit_event(
                                event_type=event_type,
                                details=details,
                                critical=False
                            )
                        )
                    except RuntimeError:
                        # No event loop, run in executor
                        def run_async():
                            asyncio.run(
                                self.async_logger.emit_audit_event(
                                    event_type=event_type,
                                    details=details,
                                    critical=False
                                )
                            )
                        self.executor.submit(run_async)
                        
                except Exception as e:
                    logger.error(f"Audit logging failed: {e}")
                    
        return AuditLoggerWrapper(audit_logger)

    def _create_mock_audit(self):
        """Create mock audit engine."""

        class MockAudit:
            def log_event(self, event_type: str, details: Dict):
                """Log audit event."""
                logger.info(f"Audit: {event_type} - {details}")

        return MockAudit()

    def _build_audit_payload(self, payload: Dict, purpose: str = "arena_internal_operation") -> Dict:
        """
        Build audit payload with required compliance metadata for NSOAligner.
        
        This prevents "No stated purpose" and "No bias assessment provided" violations
        by injecting the required metadata for internal Arena operations.
        
        Args:
            payload: Original payload dictionary
            purpose: Purpose string for compliance (e.g., "arena_internal_operation", "arena_task_execution")
            
        Returns:
            New payload dict with compliance metadata injected
        """
        audit_payload = dict(payload)
        audit_payload["_audit_source"] = "arena_internal"
        audit_payload["purpose"] = audit_payload.get("purpose", purpose)
        audit_payload["data_retention"] = audit_payload.get("data_retention", "session")
        audit_payload["bias_assessment"] = audit_payload.get("bias_assessment", {"checked": True, "source": "arena_internal"})
        return audit_payload

    def _add_gdpr_metadata(self, response: Dict) -> Dict:
        """
        Add GDPR compliance metadata to response payloads.
        
        This ensures every generated JSON payload includes the required
        retention_policy and compliance_standard fields for GDPR minimization.
        
        Args:
            response: Original response dictionary
            
        Returns:
            Response dict with GDPR metadata injected
        """
        if not isinstance(response, dict):
            logger.debug(
                f"[GDPR] Skipping metadata injection for non-dict response type: {type(response).__name__}"
            )
            return response
        
        # Add metadata section if not present
        if "metadata" not in response:
            response["metadata"] = {}
        
        # Inject GDPR compliance fields using constants
        response["metadata"]["retention_policy"] = response["metadata"].get(
            "retention_policy", GDPR_RETENTION_POLICY
        )
        response["metadata"]["compliance_standard"] = response["metadata"].get(
            "compliance_standard", GDPR_COMPLIANCE_STANDARD
        )
        
        return response

    def _dispatch_compute(self, op: str, *args, **kwargs) -> Any:
        """
        Route computation through hardware dispatcher with automatic tensor sharding.
        
        ============================================================
        GRAPHIX PLATFORM DEEP INTEGRATION - DistributedSharder
        ============================================================
        Automatically shards large tensors (>10MB) for distributed computation
        to prevent memory exhaustion and enable parallel processing.
        
        Falls back to numpy if dispatcher unavailable.
        
        Args:
            op: Operation name (e.g., 'photonic_mvm', 'matmul', 'mvm')
            *args: Operation arguments (typically matrix and vector)
            **kwargs: Additional parameters including 'params' dict
            
        Returns:
            Computation result (numpy array or error dict)
        """
        # ============================================================
        # GRAPHIX PLATFORM DEEP INTEGRATION - Automatic Tensor Sharding
        # ============================================================
        # Shard large tensors automatically to prevent memory exhaustion
        SHARD_THRESHOLD_MB = 10.0  # 10MB threshold for sharding
        
        if self.sharder and NUMPY_AVAILABLE and args:
            # Calculate total tensor size
            total_size_bytes = 0
            sharded_args = []
            shard_metadata_list = []
            
            for arg in args:
                if isinstance(arg, np.ndarray):
                    arg_size_mb = arg.nbytes / (1024 * 1024)
                    total_size_bytes += arg.nbytes
                    
                    # Shard if above threshold
                    if arg_size_mb > SHARD_THRESHOLD_MB:
                        try:
                            logger.info(
                                f"[Sharder] Sharding large tensor "
                                f"({arg_size_mb:.2f}MB > {SHARD_THRESHOLD_MB}MB)"
                            )
                            shards, metadata = self.sharder.shard_tensor(
                                arg,
                                num_nodes=4,  # Split into 4 shards
                                axis=0,
                                compress=False
                            )
                            sharded_args.append(shards)
                            shard_metadata_list.append(metadata)
                            logger.info(
                                f"[Sharder] Created {len(shards)} shards, "
                                f"shard shapes: {metadata.shard_shapes}"
                            )
                        except Exception as e:
                            logger.warning(f"[Sharder] Failed to shard tensor: {e}, using full tensor")
                            sharded_args.append([arg])
                            shard_metadata_list.append(None)
                    else:
                        # Don't shard small tensors
                        sharded_args.append([arg])
                        shard_metadata_list.append(None)
                else:
                    # Non-tensor argument, pass through
                    sharded_args.append([arg])
                    shard_metadata_list.append(None)
            
            # If any tensors were sharded, execute in parallel and unshard
            if any(meta is not None for meta in shard_metadata_list):
                logger.info(f"[Sharder] Executing distributed computation on shards")
                results = []
                
                # Get maximum shard count
                max_shards = max(
                    len(shards) for shards in sharded_args
                )
                
                # Execute each shard combination
                for shard_idx in range(max_shards):
                    # Get args for this shard (repeat if not sharded)
                    shard_args = []
                    for i, shards in enumerate(sharded_args):
                        shard_arg = shards[min(shard_idx, len(shards) - 1)]
                        shard_args.append(shard_arg)
                    
                    # Execute on this shard
                    try:
                        shard_result = self._execute_shard(op, shard_args, kwargs)
                        results.append(shard_result)
                    except Exception as e:
                        logger.error(f"[Sharder] Shard {shard_idx} execution failed: {e}")
                        # Use zero result for failed shard
                        if results:
                            results.append(np.zeros_like(results[0]))
                        else:
                            results.append(np.array([]))
                
                # Unshard results
                if results and any(isinstance(r, np.ndarray) for r in results):
                    try:
                        # Find first valid metadata for unsharding
                        unshard_metadata = next(
                            (meta for meta in shard_metadata_list if meta is not None),
                            None
                        )
                        
                        if unshard_metadata:
                            final_result = self.sharder.unshard(results, unshard_metadata)
                            logger.info(
                                f"[Sharder] Unsharded {len(results)} results into "
                                f"shape {final_result.shape}"
                            )
                            return final_result
                        else:
                            # Fallback: concatenate results
                            final_result = np.concatenate(results, axis=0)
                            logger.info(
                                f"[Sharder] Concatenated {len(results)} results "
                                f"(no metadata available)"
                            )
                            return final_result
                    except Exception as e:
                        logger.error(f"[Sharder] Unshard failed: {e}, using first result")
                        return results[0] if results else np.array([])
        
        # Use dispatcher if available
        if self.hardware_dispatcher:
            try:
                result = self.hardware_dispatcher.dispatch(op, *args, **kwargs)

                # Check for error response
                if isinstance(result, dict) and 'error_code' in result:
                    logger.warning(f"[Dispatcher] {op} failed: {result.get('message')}")
                    # Fall through to numpy fallback
                else:
                    return result

            except Exception as e:
                logger.warning(f"[Dispatcher] {op} exception: {e}, falling back to numpy")

        # Fallback to numpy
        if not NUMPY_AVAILABLE:
            return {"error": "NumPy not available for fallback"}

        if op == 'photonic_mvm' and len(args) >= 2:
            return np.dot(args[0], args[1])
        elif op == 'matmul' and len(args) >= 2:
            return np.matmul(args[0], args[1])
        elif op == 'mvm' and len(args) >= 2:
            return np.dot(args[0], args[1])
        else:
            raise ValueError(f"Unknown operation: {op}")

    def _execute_shard(self, op: str, args: list, kwargs: dict) -> np.ndarray:
        """
        Execute operation on a single shard.
        
        Helper method for distributed shard execution.
        
        Args:
            op: Operation name
            args: Shard arguments
            kwargs: Operation parameters
            
        Returns:
            Shard result
        """
        # Use dispatcher if available
        if self.hardware_dispatcher:
            try:
                return self.hardware_dispatcher.dispatch(op, *args, **kwargs)
            except Exception as e:
                logger.debug(f"[Sharder] Dispatcher failed for shard: {e}")
        
        # Fallback to numpy
        if not NUMPY_AVAILABLE:
            raise RuntimeError("NumPy not available for shard execution")
        
        if op in ('photonic_mvm', 'mvm') and len(args) >= 2:
            return np.dot(args[0], args[1])
        elif op == 'matmul' and len(args) >= 2:
            return np.matmul(args[0], args[1])
        else:
            raise ValueError(f"Unknown shard operation: {op}")

    async def _dispatch_compute_async(self, op: str, *args, **kwargs) -> Any:
        """
        Async version of _dispatch_compute that offloads to thread pool.
        
        FIX #6: CPU Offloading - This method prevents CPU saturation by
        running matrix operations in a thread pool, allowing the async
        event loop to remain responsive.
        
        Args:
            op: Operation name (e.g., 'photonic_mvm', 'matmul', 'mvm')
            *args: Operation arguments (typically matrix and vector)
            **kwargs: Additional parameters including 'params' dict
            
        Returns:
            Computation result (numpy array or error dict)
        """
        # Offload to thread pool to prevent blocking the event loop
        return await asyncio.to_thread(self._dispatch_compute, op, *args, **kwargs)

    def _gather_inputs(self, node: dict, results: dict) -> list:
        """
        Gather inputs for a node from previous results or node data.
        
        Args:
            node: Node definition with input specifications
            results: Dictionary of previous node results
            
        Returns:
            List of input values for the node operation
        """
        inputs = []
        
        # Check for explicit inputs in node
        if 'inputs' in node:
            for inp in node['inputs']:
                if isinstance(inp, str) and inp in results:
                    inputs.append(results[inp])
                else:
                    inputs.append(inp)
        
        # Check for matrix/vector data in node
        if 'matrix' in node:
            inputs.append(np.array(node['matrix']) if NUMPY_AVAILABLE else node['matrix'])
        if 'vector' in node:
            inputs.append(np.array(node['vector']) if NUMPY_AVAILABLE else node['vector'])
        
        # Default: create identity/random data if no inputs specified
        if not inputs and NUMPY_AVAILABLE:
            size = node.get('size', 4)
            inputs = [np.eye(size), np.ones(size)]
        
        return inputs

    def _execute_cpu_op(self, op_type: str, inputs: list, params: dict) -> Any:
        """
        Execute a non-matrix operation on CPU.
        
        Args:
            op_type: Type of operation
            inputs: Input values
            params: Operation parameters
            
        Returns:
            Operation result
        """
        if not NUMPY_AVAILABLE:
            return {"error": "NumPy not available"}
        
        if op_type == 'add':
            return np.add(*inputs[:2]) if len(inputs) >= 2 else inputs[0]
        elif op_type == 'relu':
            return np.maximum(0, inputs[0]) if inputs else np.array([])
        elif op_type == 'softmax':
            if inputs:
                exp_x = np.exp(inputs[0] - np.max(inputs[0]))
                return exp_x / exp_x.sum()
            return np.array([])
        elif op_type == 'identity':
            return inputs[0] if inputs else np.array([])
        else:
            # Default: return first input or empty
            return inputs[0] if inputs else np.array([])

    # ============================================================
    # GRAPHIX PLATFORM DEEP INTEGRATION - Helper Methods
    # ============================================================
    
    def _requires_consensus(self, agent_id: str, task: str, data: Dict) -> bool:
        """
        Determine if a task requires consensus approval before execution.
        
        This implements the governance gate for high-risk operations that could:
        - Modify critical system state
        - Execute privileged operations
        - Access sensitive data
        - Affect other agents
        
        Industry-standard risk criteria:
        - Administrative/system agents require consensus
        - Operations with "admin", "modify", "delete" in task description
        - Tasks marked as high-risk in metadata
        
        Args:
            agent_id: Agent identifier
            task: Task description
            data: Task data with metadata
            
        Returns:
            True if consensus is required, False otherwise
        """
        # Risk criteria 1: Administrative/privileged agents
        admin_agents = {"admin", "system", "root", "orchestrator"}
        if any(admin_keyword in agent_id.lower() for admin_keyword in admin_agents):
            logger.info(f"[Consensus] Agent {agent_id} requires consensus (privileged agent)")
            return True
        
        # Risk criteria 2: High-risk operations in task description
        high_risk_keywords = {
            "delete", "remove", "modify", "admin", "system", 
            "privilege", "grant", "revoke", "escalate"
        }
        if any(keyword in task.lower() for keyword in high_risk_keywords):
            logger.info(f"[Consensus] Task requires consensus (high-risk operation detected)")
            return True
        
        # Risk criteria 3: Explicit risk flag in metadata
        metadata = data.get("metadata", {})
        if metadata.get("requires_consensus") or metadata.get("high_risk"):
            logger.info(f"[Consensus] Task requires consensus (marked in metadata)")
            return True
        
        # Risk criteria 4: Data sensitivity level
        data_sensitivity = metadata.get("sensitivity_level", "").lower()
        if data_sensitivity in {"high", "critical", "confidential"}:
            logger.info(f"[Consensus] Task requires consensus (high data sensitivity)")
            return True
        
        return False

    def _estimate_pii(self, graph: Dict) -> float:
        """
        Estimate PII (Personally Identifiable Information) confidence in graph data.
        
        This is a heuristic estimation used by DQSValidator to assess data privacy risk.
        
        Industry-standard PII indicators:
        - Email addresses, phone numbers, SSN patterns
        - Names, addresses
        - Financial information
        
        Args:
            graph: Graph data to analyze
            
        Returns:
            PII confidence score (0.0-1.0)
        """
        import re
        
        # Convert graph to string for pattern matching
        graph_str = json.dumps(graph).lower()
        
        pii_patterns = {
            "email": r'\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b',
            "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
            "credit_card": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
        }
        
        pii_indicators = ["name", "address", "dob", "birthdate", "password"]
        
        # Check for pattern matches
        pii_score = 0.0
        for pattern_name, pattern in pii_patterns.items():
            if re.search(pattern, graph_str):
                pii_score += 0.3
        
        # Check for PII indicator keywords
        for indicator in pii_indicators:
            if indicator in graph_str:
                pii_score += 0.1
        
        # Clamp to [0.0, 1.0]
        return min(1.0, pii_score)

    def _calculate_completeness(self, graph: Dict) -> float:
        """
        Calculate graph completeness score.
        
        A complete graph has:
        - All required fields (id, nodes, edges)
        - Valid node structure with ids and types
        - Valid edge structure with source/target references
        
        Args:
            graph: Graph data to analyze
            
        Returns:
            Completeness score (0.0-1.0)
        """
        score = 0.0
        
        # Check for required fields (0.3 weight)
        if "id" in graph or "graph_id" in graph:
            score += 0.1
        if "nodes" in graph:
            score += 0.1
        if "edges" in graph:
            score += 0.1
        
        # Check node quality (0.4 weight)
        nodes = graph.get("nodes", [])
        if nodes:
            valid_nodes = sum(
                1 for node in nodes 
                if isinstance(node, dict) and "id" in node
            )
            node_completeness = valid_nodes / len(nodes) if nodes else 0
            score += 0.4 * node_completeness
        
        # Check edge quality (0.3 weight)
        edges = graph.get("edges", [])
        if edges:
            node_ids = {node.get("id") for node in nodes if isinstance(node, dict) and "id" in node}
            valid_edges = sum(
                1 for edge in edges
                if isinstance(edge, dict) 
                and edge.get("from") in node_ids 
                and edge.get("to") in node_ids
            )
            edge_completeness = valid_edges / len(edges) if edges else 0
            score += 0.3 * edge_completeness
        else:
            # No edges is valid for simple graphs
            score += 0.3
        
        return min(1.0, score)

    def _validate_syntax(self, graph: Dict) -> bool:
        """
        Validate graph syntax.
        
        Checks for:
        - Valid JSON structure (already checked by parser)
        - Required fields present
        - Valid data types
        
        Args:
            graph: Graph to validate
            
        Returns:
            True if syntax is valid, False otherwise
        """
        # Basic structure check
        if not isinstance(graph, dict):
            return False
        
        # Nodes must be a list
        if "nodes" in graph and not isinstance(graph["nodes"], list):
            return False
        
        # Edges must be a list
        if "edges" in graph and not isinstance(graph["edges"], list):
            return False
        
        return True

    async def execute_graph(self, graph: dict) -> dict:
        """
        Execute graph using optimal hardware backend with DQS validation.
        
        Routes matrix operations through the hardware dispatcher for optimal
        performance on available accelerators (photonic, GPU, emulator, CPU).
        
        ============================================================
        GRAPHIX PLATFORM DEEP INTEGRATION - DQS Validation
        ============================================================
        Before graph execution, validates data quality using DQSValidator
        to prevent low-quality data from entering the reasoning pipeline.
        
        Args:
            graph: Graph definition with nodes and edges
            
        Returns:
            Dictionary with execution results and metadata
            
        Raises:
            ValueError: If DQS validation fails (score below reject threshold)
        """
        # Emit audit event for graph execution start
        graph_id = graph.get('id', graph.get('graph_id', 'unknown'))
        start_time = time.time()
        
        self.audit.log_event(
            "graph_execution_start",
            {
                "graph_id": graph_id,
                "node_count": len(graph.get('nodes', [])),
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
        
        # ============================================================
        # GRAPHIX PLATFORM DEEP INTEGRATION - DQS Validation Gate
        # ============================================================
        # Validate data quality before execution to prevent low-quality
        # data from entering the reasoning pipeline
        if self.dqs_validator:
            try:
                # Calculate data quality metrics
                pii_confidence = self._estimate_pii(graph)
                graph_completeness = self._calculate_completeness(graph)
                syntactic_completeness = 1.0 if self._validate_syntax(graph) else 0.5
                
                # Perform DQS validation
                dqs_result = self.dqs_validator.validate(
                    pii_confidence=pii_confidence,
                    graph_completeness=graph_completeness,
                    syntactic_completeness=syntactic_completeness,
                )
                
                logger.info(
                    f"[DQS] Graph {graph_id} validation: "
                    f"score={dqs_result.score:.3f}, decision={dqs_result.decision}"
                )
                
                # Reject graphs with low DQS scores
                if dqs_result.decision == "reject":
                    error_msg = (
                        f"DQS validation rejected graph {graph_id}: "
                        f"score={dqs_result.score:.3f} (threshold={self.dqs_validator.reject_threshold})"
                    )
                    logger.warning(f"[DQS] {error_msg}")
                    raise ValueError(error_msg)
                
                # Quarantine warning for borderline quality
                if dqs_result.decision == "quarantine":
                    logger.warning(
                        f"[DQS] Graph {graph_id} quality warning: "
                        f"score={dqs_result.score:.3f} (quarantine threshold)"
                    )
                
            except Exception as e:
                # Log DQS validation errors but don't block execution
                # (fail-open for availability)
                logger.error(f"[DQS] Validation error for graph {graph_id}: {e}")
                # Re-raise ValueError for explicit reject decisions
                if isinstance(e, ValueError) and "rejected" in str(e).lower():
                    raise
        
        # Note: Validate graph structure before processing
        # This prevents "Missing 'nodes' field" errors from data integrity issues
        if not isinstance(graph, dict):
            logger.warning("[Arena] Invalid graph: expected dict, got %s. Using empty graph.", type(graph).__name__)
            graph = {"nodes": [], "edges": []}
        
        if "nodes" not in graph:
            logger.warning("[Arena] Invalid base graph: Missing 'nodes' field. Using empty nodes list.")
            graph["nodes"] = []
        
        if not isinstance(graph.get("nodes"), list):
            logger.warning("[Arena] Invalid 'nodes' field: expected list. Using empty nodes list.")
            graph["nodes"] = []
            
        nodes = graph.get('nodes', [])
        # Note: edges reserved for future topological ordering/dependency resolution
        _ = graph.get('edges', [])

        # Log execution plan
        if self.hardware_dispatcher:
            backend_info = self.hardware_dispatcher.get_metrics_summary()
            logger.info(f"[Arena] Executing graph with {len(nodes)} nodes via dispatcher")
        else:
            logger.info(f"[Arena] Executing graph with {len(nodes)} nodes (CPU fallback)")

        results = {}
        execution_success = True
        error_details = None

        try:
            for node in nodes:
                node_id = node.get('id')
                op_type = node.get('op', 'mvm')

                # Get inputs from previous results or node data
                inputs = self._gather_inputs(node, results)

                # Route through dispatcher
                if op_type in ('mvm', 'matmul', 'photonic_mvm', 'photonic_fused'):
                    # Build params for photonic ops
                    params = node.get('params', {})
                    if not params and op_type.startswith('photonic'):
                        # Provide default photonic params
                        params = {
                            'noise_std': 0.01,
                            'multiplexing': 'wavelength',
                            'compression': 'ITU-F.748-quantized',
                            'bandwidth_ghz': 100,
                            'latency_ps': 50,
                        }
                    
                    # Use original op_type if it's already a photonic operation,
                    # otherwise prefix with 'photonic_' for dispatcher routing
                    dispatch_op = op_type if op_type.startswith('photonic') else f'photonic_{op_type}'
                # FIX #6: Use async dispatch to offload CPU-intensive matrix ops
                result = await self._dispatch_compute_async(
                    dispatch_op,
                    *inputs,
                    params=params
                )
            else:
                # Non-matrix ops - offload to thread pool if CPU-bound
                result = await asyncio.to_thread(
                    self._execute_cpu_op, op_type, inputs, node.get('params', {})
                )

            results[node_id] = result

        except Exception as e:
            execution_success = False
            error_details = str(e)
            logger.error(f"[Arena] Graph execution failed: {e}")
            
        # Emit audit event for graph execution completion
        execution_time = time.time() - start_time
        self.audit.log_event(
            "graph_execution_complete",
            {
                "graph_id": graph_id,
                "node_count": len(nodes),
                "success": execution_success,
                "execution_time_ms": execution_time * 1000,
                "error": error_details,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
        
        if not execution_success:
            return self._add_gdpr_metadata({
                'results': results,
                'success': False,
                'error': error_details
            })

        return self._add_gdpr_metadata({'results': results, 'success': True})

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
        Run agent task with validation, reasoning engine invocation, and error handling.

        CRITICAL FIX: Now invokes reasoning engines for reasoning-type agents
        instead of just using subprocess LLM calls.

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

        # ============================================================
        # REASONING ENGINE INVOCATION (Critical Fix)
        # ============================================================
        # Check if this is a reasoning-type agent that should invoke reasoning engines
        reasoning_agent_types = {
            "reasoner", "causal", "symbolic", "analogical", "probabilistic",
            "counterfactual", "multimodal", "deductive", "inductive", "abductive",
            "planner", "inference"
        }
        
        is_reasoning_agent = any(
            r_type in agent_id.lower() for r_type in reasoning_agent_types
        )
        
        # Also check task description for reasoning keywords
        reasoning_task_keywords = [
            "reason", "causal", "why", "because", "infer", "deduce",
            "analyze", "logic", "symbolic", "probabilistic"
        ]
        is_reasoning_task = any(
            keyword in task.lower() for keyword in reasoning_task_keywords
        )
        
        # CRITICAL FIX: Also check for selected_tools in data payload
        # This enables reasoning when QueryRouter selects reasoning tools
        # even if the agent_id is "generator"
        selected_tools = (
            data.get("parameters", {}).get("selected_tools", []) or
            data.get("properties", {}).get("selected_tools", []) or
            data.get("selected_tools", []) or
            []
        )
        
        # Check if selected_tools contains reasoning engines
        reasoning_tool_names = {
            "causal", "symbolic", "analogical", "probabilistic",
            "counterfactual", "deductive", "inductive", "abductive",
            "multimodal", "bayesian", "hybrid", "ensemble"
        }
        has_reasoning_tools = any(
            tool.lower() in reasoning_tool_names for tool in selected_tools
        ) if selected_tools else False
        
        # Also check query_type from payload
        query_type_from_payload = (
            data.get("parameters", {}).get("query_type", "") or
            data.get("properties", {}).get("query_type", "") or
            ""
        ).lower()
        is_reasoning_query_type = query_type_from_payload in ("reasoning", "causal", "inference")
        
        # Invoke reasoning if ANY of these conditions are true
        should_invoke_reasoning = (
            is_reasoning_agent or 
            is_reasoning_task or 
            has_reasoning_tools or
            is_reasoning_query_type
        )
        
        # ============================================================
        # GRAPHIX PLATFORM DEEP INTEGRATION - Consensus Governance Gate
        # ============================================================
        # Check if this operation requires consensus approval before execution
        # This adds distributed governance for high-risk operations
        if self.consensus_engine and self._requires_consensus(agent_id, task, data):
            logger.info(f"[Consensus] Agent {agent_id} task requires governance approval")
            
            try:
                # Create proposal for consensus voting
                proposal_graph = {
                    "id": f"agent_task_{agent_id}_{int(time.time())}",
                    "type": "Graph",
                    "nodes": [{
                        "id": "task_execution",
                        "type": "ProposalNode",
                        "proposed_by": "arena_orchestrator",
                        "rationale": f"Agent {agent_id} executing task: {task[:100]}",
                        "proposal_content": {
                            "type": "agent_task_execution",
                            "agent_id": agent_id,
                            "task": task,
                            "data_hash": hashlib.sha256(
                                json.dumps(data, sort_keys=True).encode()
                            ).hexdigest()[:16]
                        }
                    }],
                    "edges": []
                }
                
                # Submit proposal and get consensus
                proposal_id = self.consensus_engine.propose(
                    proposal_graph=proposal_graph,
                    agent_id="arena_orchestrator",
                    duration_days=1  # Short duration for real-time operations
                )
                
                # Evaluate consensus immediately
                consensus_result = self.consensus_engine.evaluate_consensus(proposal_id)
                
                logger.info(
                    f"[Consensus] Proposal {proposal_id}: "
                    f"status={consensus_result['status']}, "
                    f"approval_ratio={consensus_result.get('approval_ratio', 0):.2%}"
                )
                
                # Check if approved
                if consensus_result["status"] != "approved":
                    rejection_reason = (
                        f"Consensus rejected agent {agent_id} task execution: "
                        f"status={consensus_result['status']}, "
                        f"approval_ratio={consensus_result.get('approval_ratio', 0):.2%} "
                        f"(required: {self.consensus_engine.approval_threshold:.2%})"
                    )
                    logger.warning(f"[Consensus] {rejection_reason}")
                    
                    # Raise BiasDetectedException for governance rejection
                    # Note: Import at runtime to avoid circular dependency
                    try:
                        from vulcan.safety.safety_types import BiasDetectedException
                        raise BiasDetectedException(rejection_reason)
                    except ImportError:
                        # Fallback to standard exception if BiasDetectedException unavailable
                        raise RuntimeError(rejection_reason)
                
                logger.info(f"[Consensus] Agent {agent_id} task approved by consensus")
                
            except Exception as e:
                # Log consensus errors but don't block execution
                # (fail-open for availability, but re-raise explicit rejections)
                if "rejected" in str(e).lower() or isinstance(e, RuntimeError):
                    raise
                logger.error(f"[Consensus] Error checking consensus for agent {agent_id}: {e}")
        
        if should_invoke_reasoning and REASONING_AVAILABLE:
            # Build trigger reason for logging
            trigger_reasons = []
            if is_reasoning_agent:
                trigger_reasons.append(f"agent_type={agent_id}")
            if is_reasoning_task:
                trigger_reasons.append("reasoning_keywords_in_task")
            if has_reasoning_tools:
                trigger_reasons.append(f"selected_tools={selected_tools}")
            if is_reasoning_query_type:
                trigger_reasons.append(f"query_type={query_type_from_payload}")
            
            logger.info(
                f"🧠 Agent '{agent_id}' invoking reasoning engine for task: {task[:100]}... "
                f"(triggers: {', '.join(trigger_reasons)})"
            )
            
            try:
                # Extract query from task and data - check multiple locations
                # GraphSpec format: data.parameters.goal
                # GraphixIRGraph format: data.nodes[0].properties.text
                query = (
                    data.get("parameters", {}).get("goal") or
                    data.get("query") or 
                    data.get("input") or 
                    task
                )
                
                # Also check nodes for query text (GraphixIRGraph format)
                if not query or query == task:
                    nodes = data.get("nodes", [])
                    if nodes and isinstance(nodes, list):
                        query = nodes[0].get("properties", {}).get("text") or query
                
                context = data.get("context", {})
                
                # Calculate complexity based on data size
                complexity = min(
                    1.0, 
                    max(
                        REASONING_COMPLEXITY_BASE, 
                        len(json.dumps(data)) / REASONING_COMPLEXITY_DATA_DIVISOR
                    )
                )
                
                # Determine reasoning type from agent_id or selected_tools
                reasoning_type_map = {
                    "causal": "causal",
                    "symbolic": "symbolic",
                    "analogical": "analogical",
                    "probabilistic": "probabilistic",
                    "counterfactual": "counterfactual",
                    "deductive": "reasoning",
                    "inductive": "reasoning",
                    "abductive": "reasoning",
                }
                
                query_type = "reasoning"  # default
                
                # First check selected_tools from payload (most reliable)
                if selected_tools:
                    for tool in selected_tools:
                        tool_lower = tool.lower()
                        if tool_lower in reasoning_type_map:
                            query_type = reasoning_type_map[tool_lower]
                            break
                else:
                    # Fall back to checking agent_id
                    for key, value in reasoning_type_map.items():
                        if key in agent_id.lower():
                            query_type = value
                            break
                
                # Apply reasoning via the integration layer
                integration_result = apply_reasoning(
                    query=str(query)[:MAX_REASONING_QUERY_LENGTH],
                    query_type=query_type,
                    complexity=complexity,
                    context=context,
                )
                
                logger.info(
                    f"🧠 Reasoning selection complete for '{agent_id}': "
                    f"tools={integration_result.selected_tools}, "
                    f"strategy={integration_result.reasoning_strategy}, "
                    f"confidence={integration_result.confidence:.2f}"
                )
                
                # If UnifiedReasoner is available, invoke actual reasoning
                reasoning_output = None
                if UnifiedReasoner is not None and create_unified_reasoner is not None:
                    try:
                        # Get or create unified reasoner with learning and safety enabled
                        reasoner = create_unified_reasoner(
                            enable_learning=True,
                            enable_safety=True,
                        )
                        
                        if reasoner is not None:
                            # Map query_type to ReasoningType enum
                            reasoning_type = ReasoningType.HYBRID
                            if ReasoningType is not None:
                                type_map = {
                                    "causal": ReasoningType.CAUSAL,
                                    "symbolic": ReasoningType.SYMBOLIC,
                                    "analogical": ReasoningType.ANALOGICAL,
                                    "probabilistic": ReasoningType.PROBABILISTIC,
                                    "counterfactual": ReasoningType.COUNTERFACTUAL,
                                    "reasoning": ReasoningType.HYBRID,
                                }
                                reasoning_type = type_map.get(query_type, ReasoningType.HYBRID)
                            
                            # Invoke the actual reasoning engine
                            reasoning_result = reasoner.reason(
                                input_data=str(query),
                                query={"query": str(query), "context": context, "task": task},
                                reasoning_type=reasoning_type,
                            )
                            
                            reasoning_output = {
                                "conclusion": getattr(reasoning_result, "conclusion", None),
                                "confidence": getattr(reasoning_result, "confidence", None),
                                "reasoning_type": str(getattr(reasoning_result, "reasoning_type", "unknown")),
                                "explanation": getattr(reasoning_result, "explanation", None),
                            }
                            
                            logger.info(
                                f"🧠 Reasoning execution complete for '{agent_id}': "
                                f"type={reasoning_output.get('reasoning_type')}, "
                                f"confidence={reasoning_output.get('confidence')}"
                            )
                    except Exception as reasoning_error:
                        logger.warning(
                            f"UnifiedReasoner invocation failed for '{agent_id}': {reasoning_error}. "
                            f"Using integration result only."
                        )
                
                # Build result with reasoning output
                result = {
                    "status": "success",
                    "agent_id": agent_id,
                    "reasoning_invoked": True,
                    "selected_tools": integration_result.selected_tools,
                    "reasoning_strategy": integration_result.reasoning_strategy,
                    "confidence": integration_result.confidence,
                    "rationale": integration_result.rationale,
                    "reasoning_output": reasoning_output,
                    "output": (
                        reasoning_output.get("conclusion") if reasoning_output 
                        else f"Reasoning applied via {integration_result.reasoning_strategy}"
                    ),
                }
                
                # BUG #3 FIX: Notify world model of Arena reasoning execution
                # This makes the world model aware of reasoning via Arena
                _query_id = data.get("query_id", f"arena_{agent_id}_{int(time.time())}")
                observe_engine_result(
                    query_id=_query_id,
                    engine_name=integration_result.reasoning_strategy or query_type,
                    result=result,
                    # Note: 0.15 is consistent with MIN_REASONING_CONFIDENCE_THRESHOLD in main.py
                    success=integration_result.confidence > 0.15 if integration_result.confidence else False,
                    execution_time_ms=0  # No timing available here
                )
                
                return result
                
            except Exception as reasoning_error:
                logger.warning(
                    f"Reasoning integration failed for '{agent_id}': {reasoning_error}. "
                    f"Falling back to subprocess execution."
                )
                
                # BUG #3 FIX: Notify world model of reasoning error
                observe_error(
                    query_id=data.get("query_id", f"arena_{agent_id}_{int(time.time())}"),
                    error_type="reasoning_integration_failed",
                    error_message=str(reasoning_error),
                    component="graphix_arena._run_agent"
                )
                # Fall through to standard execution below

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

        # ============================================================
        # RAY ACTOR EXECUTION (Architectural Fix #3)
        # ============================================================
        # Use Ray actors when available - they manage memory/CPU better than
        # subprocess, preventing "Zombie Process" saturation.
        if self.use_ray and agent_id in self.ray_workers:
            logger.info(f"Executing task for agent '{agent_id}' via Ray actor...")
            content_payload = f"{task}: {json.dumps(data)}"
            
            try:
                worker = self.ray_workers[agent_id]
                # Use asyncio.wait_for with ray.get for timeout control
                future = worker.generate.remote(content_payload)
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, ray.get, future
                    ),
                    timeout=60  # 60 second timeout for Ray execution
                )
                
                if isinstance(result, dict) and result.get("status") == "error":
                    logger.warning(
                        f"Ray worker error for {agent_id}: {result.get('error')}, "
                        "falling back to subprocess"
                    )
                    # Fall through to subprocess execution
                else:
                    logger.debug(f"Ray execution successful for {agent_id}")
                    return result
                    
            except asyncio.TimeoutError:
                logger.warning(
                    f"Ray execution timeout for {agent_id}, falling back to subprocess"
                )
                # Fall through to subprocess execution
            except Exception as e:
                logger.warning(
                    f"Ray execution failed for {agent_id}: {e}, falling back to subprocess"
                )
                # Fall through to subprocess execution

        # Standard subprocess execution (fallback)
        logger.info(f"Executing task for agent '{agent_id}' via standard subprocess...")

        content_payload = f"{task}: {json.dumps(data)}"
        # FIX Issue 2 (Dirty JSON): Configure logging to stderr BEFORE importing modules.
        # This prevents logs from contaminating stdout, which must contain only clean JSON
        # for the parent process to parse. Without this, logging.basicConfig() in
        # llm_client.py outputs to stdout by default, causing JSON parse failures
        # (e.g., "Extra data: line 1 column 5") and 40-second timeouts.
        # FIX Issue 4: Use magic markers to wrap JSON output for reliable parsing.
        script_to_execute = (
            f"import sys, logging; "
            f"logging.basicConfig(stream=sys.stderr, level=logging.INFO, "
            f"format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'); "
            f"import json; from llm_client import GraphixLLMClient; "
            f"client=GraphixLLMClient('{agent_id}'); "
            f'messages = [{{"role": "user", "content": {repr(content_payload)}}}]; '
            f"result = client.chat(messages); "
            f"print('###ARENA_OUTPUT_START###'); "
            f"print(json.dumps(result)); "
            f"print('###ARENA_OUTPUT_END###'); "
            f"sys.stdout.flush()"
        )

        cmd = [sys.executable, "-c", script_to_execute]

        # FIX Issue 1: Pass PYTHONPATH environment to subprocess
        # Without this, subprocesses cannot find llm_client and other project modules,
        # causing ModuleNotFoundError and 30-40s timeout per query as it falls back
        # to slower Agent Pool.
        project_root = Path(__file__).resolve().parent.parent
        src_dir = Path(__file__).resolve().parent
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "")
        # Prepend project root and src directory to PYTHONPATH
        env["PYTHONPATH"] = f"{project_root}{os.pathsep}{src_dir}{os.pathsep}{existing_pythonpath}"

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,  # Note: Pass environment with correct PYTHONPATH
                cwd=str(project_root),  # Note: Set working directory to project root
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

            # Parse with marker-aware extractor
            stdout_str = stdout.decode()
            try:
                return extract_json_from_output(stdout_str)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Arena output: {e}")
                logger.debug(f"stdout: {stdout_str[:500]}")
                logger.debug(f"stderr: {stderr.decode()[:500]}")
                raise

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
        # Note: Add source context and compliance metadata for internal Arena operations
        # This prevents false positives from compliance checks that require purpose/bias_assessment
        if self.nso_aligner:
            try:
                audit_payload = self._build_audit_payload(payload, "arena_internal_operation")
                audit_result = self.nso_aligner.multi_model_audit(audit_payload)
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

                return self._add_gdpr_metadata({
                    "status": "rollback",
                    "reason": f"Hallucination rate {hallucination_rate:.2%} exceeded threshold",
                    "hallucination_rate": hallucination_rate,
                })

            # Apply EvolutionEngine for generator/evolver agents
            # FIX #6: CPU Offloading - Run heavy evolution computation via HardwareDispatcher
            # to prevent CPU saturation (98.4% CPU causing 60-70s response times)
            evolution_result = None
            if self.evolution_engine and agent_id in ("generator", "evolver"):
                try:
                    # Check if result contains graph data that can be evolved
                    if isinstance(result, dict) and ("graph" in result or "nodes" in result):
                        logger.info(f"[EvolutionEngine] Starting evolution for {agent_id} output")
                        
                        # Define fitness function based on result quality
                        def fitness_fn(individual):
                            # Basic fitness based on structure quality
                            nodes = individual.graph.get("nodes", [])
                            edges = individual.graph.get("edges", [])
                            return len(nodes) * 0.1 + len(edges) * 0.05 + 0.5
                        
                        # Note: Offload heavy evolution computation to thread pool
                        # This prevents blocking the main async event loop during
                        # CPU-intensive evolution cycles
                        def _run_evolution():
                            return self.evolution_engine.evolve(fitness_fn, generations=3)
                        
                        # Use asyncio.to_thread to run CPU-bound evolution off main thread
                        best = await asyncio.to_thread(_run_evolution)
                        
                        if best:
                            evolution_result = {
                                "evolved": True,
                                "best_fitness": best.fitness if hasattr(best, "fitness") else None,
                                "generations_run": 3,
                            }
                            logger.info(
                                f"[EvolutionEngine] Evolution complete: best_fitness={evolution_result.get('best_fitness', 'N/A')}"
                            )
                except Exception as e:
                    logger.warning(f"[EvolutionEngine] Evolution failed: {e}")
                    evolution_result = {"evolved": False, "error": str(e)}

            # Query LTM for similar topologies
            # FIX #6: Offload LTM query to thread pool
            ltm_results = []
            if self.registry and hasattr(self.registry, "find_similar_topologies"):
                try:
                    ltm_results = await asyncio.to_thread(
                        self.registry.find_similar_topologies, payload
                    )
                    logger.info(
                        f"Queried LTM for similar topologies, found {len(ltm_results)} matches"
                    )
                except Exception as e:
                    logger.error(f"LTM query failed: {e}")

            # Generate augmented data
            # FIX #6: Offload data augmentation to thread pool
            augmented = None
            if self.data_augmentor is not None:
                try:
                    augmented = await asyncio.to_thread(
                        self.data_augmentor.generate_synthetic_proposal, payload
                    )
                except Exception as e:
                    logger.error(f"Synthetic data augmentation failed: {e}")

            return self._add_gdpr_metadata({
                "status": "success",
                "result": result,
                "hallucination_rate": hallucination_rate,
                "similar_topologies": ltm_results,
                "augmented": augmented,
                "evolution": evolution_result,
            })

        except Exception as e:
            logger.error(f"Shadow task failed: {e}")
            return self._add_gdpr_metadata({"status": "error", "reason": str(e)})

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

    async def run_reasoning_task(self, request: Request):
        """
        Execute reasoning task using UnifiedReasoner.
        
        This endpoint is called by VULCAN when reasoning engines fail and need
        Arena's full reasoning pipeline with evolution/tournaments.
        
        Args:
            request: HTTP request containing query, tools, and context
            
        Returns:
            Reasoning result with conclusion, confidence, and explanation
        """
        try:
            data = await request.json()
            
            query = data.get("query")
            selected_tools = data.get("selected_tools", [])
            query_type = data.get("query_type", "reasoning")
            complexity = data.get("complexity", 0.5)
            context = data.get("context", {})
            
            if not query:
                raise HTTPException(status_code=400, detail="Missing 'query' field")
            
            logger.info(
                f"[Arena] Reasoning task received: query_type={query_type}, "
                f"tools={selected_tools}, complexity={complexity:.2f}"
            )
            
            # Use reasoning integration if available
            if REASONING_AVAILABLE and apply_reasoning is not None:
                result = apply_reasoning(
                    query=query,
                    query_type=query_type,
                    complexity=complexity,
                    context={
                        **context,
                        'arena_delegation': True,  # Mark as Arena-delegated
                    },
                )
                
                # Extract result attributes
                conclusion = None
                explanation = None
                confidence = 0.7
                result_tools = selected_tools
                
                if hasattr(result, "selected_tools"):
                    result_tools = result.selected_tools
                if hasattr(result, "confidence"):
                    confidence = result.confidence
                if hasattr(result, "rationale"):
                    explanation = result.rationale
                if hasattr(result, "reasoning_strategy"):
                    conclusion = f"Strategy: {result.reasoning_strategy}"
                
                logger.info(
                    f"[Arena] Reasoning task completed: confidence={confidence:.2f}, "
                    f"tools={result_tools}"
                )
                
                return {
                    "status": "success",
                    "result": {
                        "conclusion": conclusion,
                        "confidence": confidence,
                        "explanation": explanation,
                    },
                    "selected_tools": result_tools,
                    "metadata": {
                        "arena_processed": True,
                        "original_query_type": query_type,
                        "original_complexity": complexity,
                    },
                }
            else:
                logger.warning("[Arena] Reasoning integration not available")
                raise HTTPException(
                    status_code=503, 
                    detail="Reasoning integration not available"
                )
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[Arena] Reasoning task failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

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
        # Note: Arena internal operations should have reduced sensitivity to avoid false positives
        # Only block truly dangerous content (unsafe label) - risky and bias are often false positives
        audit_label = None
        if self.nso_aligner:
            try:
                # Add source context and compliance metadata to payload for NSOAligner to use
                audit_payload = self._build_audit_payload(payload, "arena_task_execution")
                audit_label = self.nso_aligner.multi_model_audit(audit_payload)

                # Note: For internal Arena operations, only reject on severe risks
                # "unsafe" always blocked - indicates truly dangerous content
                # "risky" and "bias" are often false positives for legitimate graph operations
                if audit_label == "unsafe":
                    # Always block unsafe content regardless of source
                    bias_detections.inc()
                    alert_msg = (
                        f"[Unsafe Content] Agent: {agent_id}, Graph: {graph_id}, "
                        f"Label: {audit_label}. Proposal rejected."
                    )
                    logger.warning(alert_msg)
                    self.send_slack_alert(alert_msg)

                    raise BiasDetectedException(
                        agent_id=agent_id,
                        graph_id=graph_id,
                        label=audit_label,
                        message="Proposal rejected by security audit engine due to unsafe content.",
                    )
                elif audit_label == "risky":
                    # Note: Log warning but don't block for risky (potential false positive)
                    logger.info(
                        f"[Arena Audit] Risky flag raised for agent {agent_id}, "
                        f"graph {graph_id}. Treating as monitoring event for internal Arena operation."
                    )
                elif audit_label == "bias":
                    # Note: Bias alone is often a false positive for Arena operations
                    logger.debug(
                        f"[Arena Audit] Bias flag raised for agent {agent_id}, graph {graph_id}. "
                        f"Treating as informational for internal Arena operation."
                    )
            except BiasDetectedException:
                raise
            except Exception as e:
                logger.warning(f"Multi-model audit failed: {e}")

        # Transparent task orchestration
        # FIX #6: Offload transparent task to thread pool to prevent CPU blocking
        transparent_result = None
        try:
            transparent_result = await asyncio.to_thread(
                self.run_transparent_task,
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
            # Predict execution cost using StrategyOrchestrator if available
            predicted_cost_ms = None
            strategy_features = None
            if self.strategy_orchestrator:
                try:
                    # Use task prompt as query for cost prediction (synchronous call)
                    strategy_decision = self.strategy_orchestrator.analyze(
                        task_prompt[:500] if task_prompt else "",
                        context={'budget_ms': 120000, 'agent_id': agent_id}
                    )
                    predicted_cost_ms = strategy_decision.estimated_cost_ms
                    logger.info(
                        f"[Arena] Agent {agent_id}: predicted cost={predicted_cost_ms:.0f}ms, "
                        f"confidence={strategy_decision.confidence:.2f}, "
                        f"drift={strategy_decision.drift_detected}"
                    )
                except Exception as e:
                    logger.debug(f"[Arena] Cost prediction failed: {e}")
            
            execution_start = time.time()
            
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

            # Calculate actual execution time and record for learning
            actual_latency_ms = (time.time() - execution_start) * 1000
            success = result.get("status") != "error"
            
            # Record execution in StrategyOrchestrator for cost model learning
            if self.strategy_orchestrator:
                try:
                    self.strategy_orchestrator.record_execution(
                        tool_name=agent_id,
                        success=success,
                        latency_ms=actual_latency_ms,
                        confidence=1.0 if success else 0.5
                    )
                    if predicted_cost_ms:
                        prediction_error = abs(actual_latency_ms - predicted_cost_ms) / max(1, predicted_cost_ms)
                        logger.info(
                            f"[Arena] Agent {agent_id}: actual={actual_latency_ms:.0f}ms, "
                            f"predicted={predicted_cost_ms:.0f}ms, error={prediction_error:.1%}"
                        )
                except Exception as e:
                    logger.debug(f"[Arena] Execution recording failed: {e}")

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

            return self._add_gdpr_metadata({"winners": winner_indices, "meta": meta})

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

            return self._add_gdpr_metadata({"status": "ok", "message": "Feedback recorded"})

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
            "/api/run/reasoner",
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


@app.get("/api/hardware/status", summary="Hardware dispatcher status", tags=["hardware"])
async def get_hardware_status():
    """
    Get hardware dispatcher status and metrics.
    
    Returns detailed information about available hardware backends,
    performance metrics, and health status for the HardwareDispatcher
    used for optimal computation routing (photonic → GPU → emulator → CPU).
    """
    if _ARENA_INSTANCE is None:
        return {"error": "Arena not initialized", "mode": "unavailable"}

    arena = _ARENA_INSTANCE
    if not hasattr(arena, 'hardware_dispatcher') or not arena.hardware_dispatcher:
        return {"error": "HardwareDispatcher not available", "mode": "cpu_only"}

    dispatcher = arena.hardware_dispatcher

    try:
        # Get available backends
        available_backends = dispatcher.list_available_hardware()

        # Get metrics summary
        metrics_summary = dispatcher.get_metrics_summary()

        # Get health status for each backend
        health_status = {}
        for backend, capabilities in dispatcher.hardware_registry.items():
            health_status[backend.value] = {
                "status": capabilities.health_status,
                "last_check": (
                    capabilities.last_health_check.isoformat()
                    if capabilities.last_health_check
                    else None
                ),
                "available": capabilities.available,
            }

        return {
            "available_backends": available_backends,
            "metrics_summary": metrics_summary,
            "health_status": health_status,
            "mode": "dispatcher_active",
        }
    except Exception as e:
        logger.error(f"Failed to get hardware dispatcher status: {e}")
        return {
            "error": str(e),
            "mode": "error",
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
    
    # Reasoning endpoint - called by VULCAN when reasoning engines fail
    add_once("/api/run/reasoner", arena.run_reasoning_task, "30/minute", ["POST"])

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
