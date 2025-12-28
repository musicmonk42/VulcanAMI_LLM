"""
VULCAN-Graphix Integration Module
Connects VULCAN World Model with Graphix Unified Runtime

This module provides seamless integration between:
- VULCAN's World Model (safety, meta-reasoning, semantic bridge)
- Graphix IR's execution engine, hardware dispatch, and node handlers

Integration capabilities:
1. Agent proposal validation through VULCAN meta-reasoning
2. Semantic concept transfer between domains
3. Safety validation before graph execution
4. Consensus voting for multi-agent systems
5. Hardware-aware execution with VULCAN guidance
"""

import asyncio
import inspect  # Added for shutdown checks
import logging
import os
import threading  # Added for new __init__
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from functools import wraps
from typing import Any, Dict, List, Optional, Set
from unittest.mock import MagicMock

# Define logger early
logger = logging.getLogger(__name__)

# Import VULCAN components - NO STUBS, 100% REAL
import sys
from pathlib import Path

# Ensure src is in the path for absolute imports
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

try:
    # WorldModel is now lazy-loaded to prevent circular imports
    from vulcan.semantic_bridge.semantic_bridge_core import SemanticBridge
    from vulcan.world_model.meta_reasoning.motivational_introspection import (
        MotivationalIntrospection,
    )

    VULCAN_AVAILABLE = True
except ImportError:
    try:
        # Try with src prefix
        from src.vulcan.semantic_bridge.semantic_bridge_core import SemanticBridge
        from src.vulcan.world_model.meta_reasoning.motivational_introspection import (
            MotivationalIntrospection,
        )

        VULCAN_AVAILABLE = True
    except ImportError as e:
        logger.error(f"CRITICAL: VULCAN components not available: {e}")
        raise ImportError("VULCAN components are required - no stubs allowed") from e

# Placeholder for lazy-loaded WorldModel
WorldModel = None

# Import Graphix components (relative imports within package) - NO STUBS
try:
    from .execution_engine import ExecutionContext, ExecutionMode, GraphExecutionResult
    from .execution_metrics import ExecutionMetrics  # Needed for on_run_complete
    from .graph_validator import ValidationResult
except ImportError as e:
    logger.error(f"CRITICAL: Could not import Graphix components: {e}")
    raise ImportError("Graphix components are required - no stubs allowed") from e


def _lazy_import_world_model():
    """Lazy imports WorldModel to avoid circular dependencies - NO STUBS."""
    try:
        from vulcan.world_model.world_model_core import WorldModel

        return WorldModel
    except ImportError:
        try:
            from src.vulcan.world_model.world_model_core import WorldModel

            return WorldModel
        except ImportError as e:
            logger.error(f"CRITICAL: Failed to import WorldModel: {e}")
            raise ImportError("WorldModel is required - no stubs allowed") from e


@dataclass
class VulcanIntegrationConfig:
    """Configuration for VULCAN-Graphix integration"""

    enable_validation: bool = True
    enable_consensus: bool = True
    enable_semantic_transfer: bool = True
    enable_safety_checks: bool = True
    consensus_threshold: float = 0.5
    safety_config: Optional[Dict[str, Any]] = None
    max_cache_size: int = 1000
    cache_ttl_seconds: int = 3600
    max_history_size: int = 1000
    validation_timeout_seconds: float = 30.0
    consensus_timeout_seconds: float = 60.0
    execution_timeout_seconds: float = 300.0
    enable_parallel_validation: bool = True
    enable_self_improvement: bool = field(
        default_factory=lambda: os.getenv("VULCAN_ENABLE_SELF_IMPROVEMENT", "1").lower()
        in ("1", "true", "yes", "on")
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationResponse:
    """Standardized validation response"""

    success: bool
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    conflicts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "reason": self.reason,
            "details": self.details,
            "confidence": self.confidence,
            "conflicts": self.conflicts,
        }


@dataclass
class ConceptTransferFailure:
    """Track failed concept transfers"""

    concept: str
    error: str
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)


def async_timeout(seconds: float):
    """Decorator to add timeout to async functions"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                logger.error(f"Timeout after {seconds}s in {func.__name__}")
                raise

        return wrapper

    return decorator


class VulcanGraphixBridge:
    """Bridge between VULCAN World Model and Graphix runtime"""

    def __init__(self, runtime: Any, config: Optional[VulcanIntegrationConfig] = None):
        self.runtime = runtime
        self.config = config or VulcanIntegrationConfig()
        self.vulcan_available = VULCAN_AVAILABLE
        self.world_model = None
        
        # BUG FIX Issue #48: Use singleton SemanticBridge to prevent per-query reinitialization.
        try:
            from vulcan.reasoning.singletons import get_semantic_bridge
            self.semantic_bridge = get_semantic_bridge()
            if self.semantic_bridge is None and SemanticBridge:
                self.semantic_bridge = SemanticBridge()
        except ImportError:
            self.semantic_bridge = SemanticBridge() if SemanticBridge else None

        # Initialize motivational driver - NO STUBS
        if VULCAN_AVAILABLE:
            self.motivational_driver = MotivationalIntrospection(
                world_model=self.world_model
            )
        else:
            raise RuntimeError("VULCAN components are required but not available")

        self._cache = {}
        self.cache_timestamps = {}  # Initialize cache timestamps
        self._lock = threading.Lock()
        self._history = deque(maxlen=self.config.max_history_size)
        self._stats = defaultdict(int)
        self.transfer_failures = deque(maxlen=100)  # Track failures
        self._pending_tasks: Set[asyncio.Task] = set()  # Track background tasks

        if self.vulcan_available:
            try:
                global WorldModel
                WorldModel = _lazy_import_world_model()
                # WorldModel import now raises error instead of returning mock
                # FIX: Pass explicit config including enable_self_improvement
                world_model_config = {
                    # Honor environment and integration config toggle
                    "enable_self_improvement": bool(
                        self.config.enable_self_improvement
                    ),
                    # Sensible defaults that match your profile_development.json
                    "enable_meta_reasoning": True,
                    "simulation_mode": True,
                    "bootstrap_mode": True,
                    # Also pass the file paths used elsewhere for consistency
                    "self_improvement_config": "configs/intrinsic_drives.json",
                    "self_improvement_state": "data/agent_state.json",
                    "meta_reasoning_config": "configs/intrinsic_drives.json",
                }
                # BUG FIX Issues #1-4, #45-46: Use singleton WorldModel to prevent
                # per-query reinitialization that causes ~10-15 second overhead.
                try:
                    from vulcan.reasoning.singletons import get_world_model
                    self.world_model = get_world_model(config=world_model_config)
                    if self.world_model is None:
                        # Fallback to direct creation if singleton fails
                        logger.warning("WorldModel singleton unavailable, creating directly")
                        self.world_model = WorldModel(config=world_model_config)
                except ImportError:
                    self.world_model = WorldModel(config=world_model_config)
                logger.info("Initializing VULCAN World Model...")
                if hasattr(self.world_model, "initialize"):
                    if asyncio.iscoroutinefunction(self.world_model.initialize):
                        asyncio.run(self.world_model.initialize())
                    else:
                        self.world_model.initialize()
                logger.info("✓ VULCAN World Model initialized")

                # BUG FIX Issues #9, #49: Use existing motivational_introspection from WorldModel
                # instead of creating a new one to prevent multiple initialization.
                # The WorldModel already creates MotivationalIntrospection during init.
                if VULCAN_AVAILABLE:
                    # Check if WorldModel already has MotivationalIntrospection initialized
                    wm_has_mi = (
                        hasattr(self.world_model, 'motivational_introspection') 
                        and self.world_model.motivational_introspection is not None
                    )
                    if wm_has_mi:
                        self.motivational_driver = self.world_model.motivational_introspection
                        logger.info(
                            "✓ VULCAN MotivationalIntrospection obtained from WorldModel (singleton)"
                        )
                    else:
                        # Only create if WorldModel doesn't have one
                        self.motivational_driver = MotivationalIntrospection(
                            world_model=self.world_model
                        )
                        logger.info(
                            "✓ VULCAN MotivationalIntrospection created (WorldModel had none)"
                        )
            except Exception as e:
                logger.error(f"❌ Failed to initialize VULCAN World Model: {e}")
                raise RuntimeError(
                    "VULCAN World Model initialization failed - no fallback allowed"
                ) from e

    def _lazy_import_world_model(self):
        """
        Lazy loader for WorldModel to fix circular dependencies - NO STUBS
        """
        global WorldModel  # Use global to cache the import
        if WorldModel is None:
            try:
                from vulcan.world_model.world_model_core import WorldModel as WM

                WorldModel = WM  # Assign to global
                logger.info("WorldModel lazy loaded successfully")
            except ImportError:
                try:
                    from src.vulcan.world_model.world_model_core import WorldModel as WM

                    WorldModel = WM
                    logger.info("WorldModel lazy loaded successfully (src prefix)")
                except ImportError as e:
                    logger.error(f"CRITICAL: Failed to import VULCAN World Model: {e}")
                    raise ImportError(
                        "WorldModel is required - no stubs allowed"
                    ) from e
        return WorldModel

    def _validate_graph_structure(self, graph: Dict[str, Any]) -> bool:
        """
        Validate that graph has required structure

        Args:
            graph: Graph to validate

        Returns:
            True if valid structure, False otherwise
        """
        if not isinstance(graph, dict):
            logger.error(f"Graph must be a dictionary, got {type(graph)}")
            return False

        required_fields = ["nodes"]
        for field in required_fields:
            if field not in graph:
                logger.error(f"Graph missing required field: {field}")
                return False

        nodes = graph.get("nodes", [])
        if not isinstance(nodes, list):
            logger.error(f"Graph 'nodes' must be a list, got {type(nodes)}")
            return False

        # Validate each node is a dictionary
        for i, node in enumerate(nodes):
            if not isinstance(node, dict):
                logger.error(
                    f"Node at index {i} must be a dictionary, got {type(node)}"
                )
                return False

            if "id" not in node:
                logger.error(f"Node at index {i} missing required 'id' field")
                return False

        return True

    def _clean_cache(self):
        """Remove expired entries from validation cache"""
        current_time = time.time()
        expired_keys = []

        # Use self._cache and self.cache_timestamps
        for key, timestamp in self.cache_timestamps.items():
            if current_time - timestamp > self.config.cache_ttl_seconds:
                expired_keys.append(key)

        for key in expired_keys:
            self._cache.pop(key, None)
            self.cache_timestamps.pop(key, None)

        if expired_keys:
            logger.debug(f"Cleaned {len(expired_keys)} expired cache entries")

    def _get_from_cache(self, key: str) -> Optional[Any]:
        """
        Get value from cache if not expired

        Args:
            key: Cache key

        Returns:
            Cached value or None if expired/missing
        """
        self._clean_cache()

        if key in self._cache:
            timestamp = self.cache_timestamps.get(key, 0)
            if time.time() - timestamp <= self.config.cache_ttl_seconds:
                self._stats["cache_hits"] += 1
                return self._cache[key]

        self._stats["cache_misses"] += 1
        return None

    def _put_in_cache(self, key: str, value: Any):
        """
        Put value in cache with timestamp

        Args:
            key: Cache key
            value: Value to cache
        """
        # Enforce max cache size
        if len(self._cache) >= self.config.max_cache_size:
            # Remove oldest entry
            oldest_key = min(
                self.cache_timestamps.keys(), key=lambda k: self.cache_timestamps[k]
            )
            self._cache.pop(oldest_key, None)
            self.cache_timestamps.pop(oldest_key, None)

        self._cache[key] = value
        self.cache_timestamps[key] = time.time()

    async def execute_graph_with_vulcan(
        self, graph: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute graph with full VULCAN validation pipeline (DEPRECATED - use runtime.execute_graph)

        Pipeline stages:
        1. Basic structure validation (Graphix)
        2. Extract agent proposals from graph
        3. VULCAN meta-reasoning validation
        4. Multi-agent consensus (if multiple proposals)
        5. Safety validation
        6. Semantic concept transfer
        7. Execute graph
        8. Post-execution analysis

        Args:
            graph: Graphix IR graph definition
            context: Optional execution context

        Returns:
            Execution result with VULCAN metadata
        """
        logger.warning(
            "execute_graph_with_vulcan is deprecated. Use runtime.execute_graph instead."
        )
        return await self.runtime.execute_graph(graph)  # Delegate to runtime's method

    async def _execute_graph_internal(
        self,
        graph: Dict[str, Any],
        context: Optional[Dict[str, Any]],
        start_time: float,
    ) -> Dict[str, Any]:
        """Internal graph execution with full pipeline (Kept for potential internal use)"""

        # Stage 1: Basic validation
        logger.info("🔍 Stage 1: Validating graph structure...")
        # Assume validator exists based on UnifiedRuntime logic
        if (
            not self.runtime
            or not hasattr(self.runtime, "validator")
            or not self.runtime.validator
        ):
            return {
                "status": "error",
                "errors": ["Validator not available"],
                "stage": "structure_validation",
            }

        validation_result_obj = self.runtime.validator.validate_graph(graph)
        validation_dict = validation_result_obj.to_dict()  # Convert to dict

        if not validation_dict.get("valid", False):
            return {
                "status": "validation_failed",
                "errors": validation_dict.get("errors", []),
                "warnings": validation_dict.get("warnings", []),
                "stage": "structure_validation",
            }

        self._stats["graphs_validated"] += 1

        # Stage 2: Extract agent proposals
        logger.info("🔍 Stage 2: Extracting agent proposals...")
        proposals = self._extract_agent_proposals(graph)

        # Stage 3: VULCAN validation (if available and enabled)
        vulcan_validations = []
        if self.vulcan_available and self.config.enable_validation and proposals:
            logger.info(f"🔍 Stage 3: VULCAN validating {len(proposals)} proposals...")

            try:
                if self.config.enable_parallel_validation and len(proposals) > 1:
                    # Parallel validation
                    self._stats["parallel_validations"] += 1
                    validation_tasks = [
                        self._validate_with_vulcan(proposal) for proposal in proposals
                    ]
                    vulcan_validations = await asyncio.gather(
                        *validation_tasks, return_exceptions=True
                    )

                    # Handle any exceptions from parallel execution
                    for i, result in enumerate(vulcan_validations):
                        if isinstance(result, Exception):
                            logger.error(
                                f"Validation failed for proposal {i}: {result}"
                            )
                            vulcan_validations[i] = ValidationResponse(
                                success=False,
                                reason=f"Validation error: {str(result)}",
                                details={"exception": str(result)},
                            )
                else:
                    # Sequential validation
                    for proposal in proposals:
                        validation_result = await self._validate_with_vulcan(proposal)
                        vulcan_validations.append(validation_result)

                self._stats["proposals_checked"] += len(proposals)

                # Block if critical proposal rejected
                for i, (proposal, validation) in enumerate(
                    zip(proposals, vulcan_validations)
                ):
                    if isinstance(validation, ValidationResponse):
                        validation_success = validation.success
                        validation_reason = validation.reason
                        validation_conflicts = validation.conflicts
                    else:
                        # Handle dict format for backward compatibility
                        validation_success = validation.get(
                            "success", validation.get("valid", True)
                        )
                        validation_reason = validation.get(
                            "reason", validation.get("reasoning", "")
                        )
                        validation_conflicts = validation.get("conflicts", [])

                    if not validation_success and proposal.get("critical", False):
                        self._stats["safety_blocks"] += 1
                        return {
                            "status": "proposal_rejected",
                            "proposal_id": proposal.get("id"),
                            "reason": validation_reason,
                            "conflicts": validation_conflicts,
                            "stage": "vulcan_validation",
                        }

            except asyncio.TimeoutError:
                self._stats["validation_timeouts"] += 1
                logger.error("Validation timed out")
                return {
                    "status": "validation_timeout",
                    "reason": f"Validation timed out after {self.config.validation_timeout_seconds}s",
                    "stage": "vulcan_validation",
                }
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                logger.error(f"Validation stage failed: {e}", exc_info=True)
                return {
                    "status": "validation_error",
                    "reason": f"Validation error: {str(e)}",
                    "stage": "vulcan_validation",
                }

        # Stage 4: Consensus voting (if multiple agents)
        consensus_result = None
        if self.config.enable_consensus and len(proposals) > 1:
            logger.info("🔍 Stage 4: Running consensus voting...")
            try:
                consensus_result = await self._run_consensus(
                    proposals, vulcan_validations
                )
                self._stats["consensus_votes"] += 1

                if not consensus_result["consensus_reached"]:
                    return {
                        "status": "consensus_failed",
                        "reason": "Agents could not reach consensus",
                        "voting_results": consensus_result,
                        "stage": "consensus",
                    }
            except asyncio.TimeoutError:
                self._stats["consensus_timeouts"] += 1
                logger.error("Consensus voting timed out")
                return {
                    "status": "consensus_timeout",
                    "reason": f"Consensus timed out after {self.config.consensus_timeout_seconds}s",
                    "stage": "consensus",
                }
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                logger.error(f"Consensus stage failed: {e}", exc_info=True)
                return {
                    "status": "consensus_error",
                    "reason": f"Consensus error: {str(e)}",
                    "stage": "consensus",
                }

        # Stage 5: Safety validation
        if self.vulcan_available and self.config.enable_safety_checks:
            logger.info("🔍 Stage 5: Safety validation...")
            try:
                safety_check = self._safety_validate_graph(graph)

                if not safety_check["safe"]:
                    self._stats["safety_blocks"] += 1
                    return {
                        "status": "safety_failed",
                        "reason": safety_check.get("reason"),
                        "violations": safety_check.get("violations", []),
                        "stage": "safety_validation",
                    }
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                logger.error(f"Safety validation failed: {e}", exc_info=True)
                # Continue execution but log the error
                logger.warning("Continuing execution despite safety validation error")

        # Stage 6: Semantic transfer (if enabled)
        transfer_failures = []
        if self.vulcan_available and self.config.enable_semantic_transfer:
            logger.info("🔍 Stage 6: Semantic concept transfer...")
            try:
                graph, transfer_failures = await self._apply_semantic_transfer(
                    graph, context or {}
                )
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                logger.error(f"Semantic transfer failed: {e}", exc_info=True)
                # Continue with original graph
                logger.warning(
                    "Continuing with original graph after semantic transfer failure"
                )

        # Stage 7: Execute graph with Graphix runtime
        logger.info("🚀 Stage 7: Executing graph...")
        try:
            # Delegate execution back to the main runtime method
            # This avoids duplicating the core execution logic
            exec_result = await self.runtime.execute_graph(graph)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.error(f"Graph execution failed: {e}", exc_info=True)
            return {
                "status": "execution_failed",
                "errors": [str(e)],
                "stage": "graph_execution",
                "metadata": {"execution_time_s": time.time() - start_time},
            }

        # Stage 8: Post-execution analysis
        execution_time = time.time() - start_time

        # Compile comprehensive result
        result = {
            "status": exec_result.get("status", "unknown"),
            "output": exec_result.get("output"),
            "errors": exec_result.get("errors"),
            "metadata": {
                "execution_time_s": execution_time,
                "vulcan_enabled": self.vulcan_available,
                "stages_completed": 8,
                "vulcan_validations": [
                    v.to_dict() if isinstance(v, ValidationResponse) else v
                    for v in vulcan_validations
                ],
                "consensus_result": consensus_result,
                "transfer_failures": [
                    {"concept": f.concept, "error": f.error} for f in transfer_failures
                ],
                "graphix_metadata": exec_result.get("metadata", {}),
            },
        }

        logger.info(f"✅ Graph execution complete in {execution_time:.2f}s")
        return result

    def _extract_agent_proposals(self, graph: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract agent proposals from graph nodes

        Looks for nodes with types:
        - AgentNode
        - ProposalNode
        - DecisionNode

        Args:
            graph: Graph to extract proposals from

        Returns:
            List of proposal dictionaries
        """
        proposals = []

        for node in graph.get("nodes", []):
            # Validate node is a dictionary
            if not isinstance(node, dict):
                logger.warning(f"Skipping non-dict node: {type(node)}")
                continue

            node_type = node.get("type", "")

            if node_type in ["AgentNode", "ProposalNode", "DecisionNode"]:
                params = node.get("params", {})

                if not isinstance(params, dict):
                    logger.warning(
                        f"Node {node.get('id', 'unknown')} has non-dict params: {type(params)}. Skipping."
                    )
                    continue

                if "proposal" in params:
                    proposal = params["proposal"]

                    # Validate proposal is a dictionary
                    if not isinstance(proposal, dict):
                        logger.warning(
                            f"Node {node['id']} has non-dict proposal: {type(proposal)}. Skipping."
                        )
                        continue

                    # Create a copy to avoid modifying original
                    proposal_copy = proposal.copy()
                    proposal_copy["node_id"] = node["id"]
                    proposal_copy["id"] = proposal_copy.get(
                        "id", f"proposal_{node['id']}"
                    )
                    proposal_copy["critical"] = node.get("critical", False)
                    proposals.append(proposal_copy)

        return proposals

    @async_timeout(30.0)
    async def _validate_with_vulcan(
        self, proposal: Dict[str, Any]
    ) -> ValidationResponse:
        """
        Validate proposal using VULCAN meta-reasoning

        Uses MotivationalIntrospection.validate_proposal_alignment

        Args:
            proposal: Proposal to validate

        Returns:
            ValidationResponse with validation results
        """
        if not self.world_model:
            return ValidationResponse(
                success=True,
                reason="VULCAN not available",
                details={"vulcan_available": False},
            )

        # Check cache first
        cache_key = f"proposal_{proposal.get('id', hash(str(proposal)))}"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            logger.debug(f"Cache hit for proposal {proposal.get('id')}")
            return cached

        # Check if method exists before calling
        if not hasattr(self.world_model, "evaluate_agent_proposal"):
            logger.warning("world_model.evaluate_agent_proposal not available")
            return ValidationResponse(
                success=True,
                reason="Method not available, skipping validation",
                details={"method_available": False},
            )

        try:
            # Use VULCAN's meta-reasoning for validation
            validation = self.world_model.evaluate_agent_proposal(proposal)

            # Convert to ValidationResponse if needed
            if isinstance(validation, ValidationResponse):
                result = validation
            elif isinstance(validation, dict):
                result = ValidationResponse(
                    success=validation.get("valid", validation.get("success", True)),
                    reason=validation.get("reasoning", validation.get("reason", "")),
                    details=validation.get("details", {}),
                    confidence=validation.get("confidence", 0.0),
                    conflicts=validation.get("conflicts", []),
                )
            else:
                result = ValidationResponse(
                    success=True,
                    reason="Unknown validation format",
                    details={"raw_validation": str(validation)},
                )

            # Cache the result
            self._put_in_cache(cache_key, result)

            return result

        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.error(f"VULCAN validation failed: {e}", exc_info=True)
            return ValidationResponse(
                success=False,
                reason=f"Validation error: {str(e)}",
                details={"exception": str(e)},
            )

    @async_timeout(60.0)
    async def _run_consensus(
        self, proposals: List[Dict[str, Any]], validations: List[Any]
    ) -> Dict[str, Any]:
        """
        Run consensus voting using VULCAN's transparency interface

        Args:
            proposals: List of proposals
            validations: List of validation results

        Returns:
            Dictionary with consensus results and voting details
        """
        if not self.world_model:
            return {
                "consensus_reached": True,
                "reason": "VULCAN not available",
                "votes": [],
            }

        # Validate list lengths match
        if len(proposals) != len(validations):
            error_msg = f"Proposal and validation counts do not match: {len(proposals)} vs {len(validations)}"
            logger.error(error_msg)
            return {"consensus_reached": False, "error": error_msg, "votes": []}

        try:
            # Get consensus context from VULCAN
            consensus_context = {}

            # Check if motivational_introspection exists
            if hasattr(self.world_model, "motivational_introspection"):
                mi = self.world_model.motivational_introspection

                # Check if transparency_interface and method exist
                if hasattr(mi, "transparency_interface"):
                    transparency_interface = mi.transparency_interface
                    if hasattr(transparency_interface, "export_for_consensus"):
                        try:
                            consensus_context = (
                                transparency_interface.export_for_consensus()
                            )
                        except (KeyboardInterrupt, SystemExit):
                            raise
                        except Exception as e:
                            logger.debug(f"Failed to export consensus context: {e}")
                    else:
                        logger.debug(
                            "transparency_interface.export_for_consensus not available"
                        )
                else:
                    logger.debug(
                        "motivational_introspection.transparency_interface not available"
                    )
            else:
                logger.debug("world_model.motivational_introspection not available")

            # Simple voting: approved if validated by VULCAN
            votes = []
            for proposal, validation in zip(proposals, validations):
                # Handle both ValidationResponse and dict formats
                if isinstance(validation, ValidationResponse):
                    approved = validation.success
                    confidence = validation.confidence
                    reasoning = validation.reason
                elif isinstance(validation, dict):
                    approved = validation.get("valid", validation.get("success", False))
                    confidence = validation.get("confidence", 0.0)
                    reasoning = validation.get(
                        "reasoning", validation.get("reason", "")
                    )
                else:
                    approved = False
                    confidence = 0.0
                    reasoning = "Unknown validation format"

                votes.append(
                    {
                        "proposal_id": proposal.get("id", "unknown"),
                        "approved": approved,
                        "confidence": confidence,
                        "reasoning": reasoning,
                    }
                )

            # Check for consensus
            approved_count = sum(1 for v in votes if v["approved"])
            approval_rate = approved_count / len(votes) if votes else 0

            consensus_reached = approval_rate >= self.config.consensus_threshold

            result = {
                "consensus_reached": consensus_reached,
                "votes": votes,
                "approval_rate": approval_rate,
                "threshold": self.config.consensus_threshold,
                "context": consensus_context,
                "timestamp": time.time(),
            }

            self._history.append(result)
            return result

        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.error(f"Consensus voting failed: {e}", exc_info=True)
            return {"consensus_reached": False, "error": str(e), "votes": []}

    def _safety_validate_graph(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform safety validation using VULCAN

        Checks for unsafe operations, constraints violations, etc.

        Args:
            graph: Graph to validate for safety

        Returns:
            Dictionary with safety validation results
        """
        if not self.world_model:
            return {"safe": True, "reason": "VULCAN not available", "violations": []}

        try:
            # Use VULCAN's safety validator
            if hasattr(self.world_model, "safety_validator"):
                try:
                    safety_result = self.world_model.safety_validator.validate(graph)

                    # Handle different result formats
                    if hasattr(safety_result, "safe"):
                        safe = safety_result.safe
                    elif isinstance(safety_result, dict):
                        safe = safety_result.get("safe", True)
                    else:
                        safe = True

                    if hasattr(safety_result, "violations"):
                        violations = safety_result.violations
                    elif isinstance(safety_result, dict):
                        violations = safety_result.get("violations", [])
                    else:
                        violations = []

                    if hasattr(safety_result, "reasoning"):
                        reason = safety_result.reasoning
                    elif isinstance(safety_result, dict):
                        reason = safety_result.get(
                            "reasoning", safety_result.get("reason", "")
                        )
                    else:
                        reason = ""

                    return {"safe": safe, "violations": violations, "reason": reason}
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception as e:
                    logger.error(
                        f"Safety validator threw exception: {e}", exc_info=True
                    )
                    return {
                        "safe": False,
                        "reason": f"Safety check error: {str(e)}",
                        "violations": [],
                    }

            return {
                "safe": True,
                "reason": "Safety validator not available",
                "violations": [],
            }

        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.error(f"Safety validation failed: {e}", exc_info=True)
            return {
                "safe": False,
                "reason": f"Safety check error: {str(e)}",
                "violations": [],
            }

    async def _apply_semantic_transfer(
        self, graph: Dict[str, Any], context: Dict[str, Any]
    ) -> tuple[Dict[str, Any], List[ConceptTransferFailure]]:
        """
        Apply semantic concept transfer using VULCAN semantic bridge

        Transfers concepts between agent planning and execution domains

        Args:
            graph: Graph to apply transfers to
            context: Execution context

        Returns:
            Tuple of (modified graph, list of transfer failures)
        """
        if not self.world_model or not hasattr(self.world_model, "semantic_bridge"):
            return graph, []

        failures = []

        try:
            semantic_bridge = self.world_model.semantic_bridge

            # Check if transfer_concept method exists
            if not hasattr(semantic_bridge, "transfer_concept"):
                logger.warning("semantic_bridge.transfer_concept not available")
                return graph, []

            # Extract concepts from graph
            concepts = self._extract_concepts(graph)

            # Transfer each concept
            for concept in concepts:
                try:
                    transferred = semantic_bridge.transfer_concept(
                        concept=concept,
                        source_domain="agent_planning",
                        target_domain="execution",
                        context=context,
                    )

                    if transferred:
                        graph = self._apply_transferred_concept(graph, transferred)

                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception as concept_error:
                    logger.warning(
                        f"Failed to transfer concept '{concept}': {concept_error}"
                    )

                    # Track the failure
                    failure = ConceptTransferFailure(
                        concept=concept,
                        error=str(concept_error),
                        timestamp=time.time(),
                        context={
                            "source_domain": "agent_planning",
                            "target_domain": "execution",
                        },
                    )
                    failures.append(failure)
                    self.transfer_failures.append(failure)

                    # Continue with other concepts
                    continue

            return graph, failures

        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.error(f"Semantic transfer failed: {e}", exc_info=True)
            return graph, failures

    def _extract_concepts(self, graph: Dict[str, Any]) -> List[str]:
        """
        Extract semantic concepts from graph for transfer

        Args:
            graph: Graph to extract concepts from

        Returns:
            List of unique concept strings
        """
        concepts = set()

        nodes = graph.get("nodes", [])
        for node in nodes:
            if not isinstance(node, dict):
                continue

            node_type = node.get("type", "")
            if node_type:
                concepts.add(node_type)

            # Extract concepts from node parameters
            params = node.get("params", {})
            if isinstance(params, dict):
                # Look for concept-related keys
                for key in ["concept", "semantic_type", "domain"]:
                    if key in params:
                        value = params[key]
                        if isinstance(value, str):
                            concepts.add(value)

        return list(concepts)

    def _apply_transferred_concept(
        self, graph: Dict[str, Any], transferred: Any
    ) -> Dict[str, Any]:
        """
        Apply transferred semantic concept to graph

        Args:
            graph: Graph to modify
            transferred: Transferred concept data

        Returns:
            Modified graph
        """
        # Placeholder: would modify graph based on transferred semantics
        # Real implementation would update node parameters, add hints, etc.

        # For now, just add metadata if it doesn't break the graph structure
        if isinstance(transferred, dict):
            if "metadata" not in graph:
                graph["metadata"] = {}
            if "_semantic_transfers" not in graph["metadata"]:
                graph["metadata"]["_semantic_transfers"] = []
            graph["metadata"]["_semantic_transfers"].append(transferred)

        return graph

    def get_vulcan_state(self) -> Dict[str, Any]:
        """
        Get current VULCAN system state

        Returns objective hierarchy, active goals, statistics

        Returns:
            Dictionary with VULCAN state information
        """
        if not self.world_model:
            return {"available": False, "reason": "VULCAN world model not initialized"}

        # Check if motivational_introspection exists
        if not hasattr(self.world_model, "motivational_introspection"):
            return {
                "available": False,
                "error": "MotivationalIntrospection not available",
            }

        try:
            mi = self.world_model.motivational_introspection

            # Check if method exists
            if not hasattr(mi, "explain_motivation_structure"):
                return {
                    "available": False,
                    "error": "explain_motivation_structure not available",
                }

            state = mi.explain_motivation_structure()

            # Add timestamp
            if isinstance(state, dict):
                state["timestamp"] = time.time()

            return state

        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.error(f"Failed to get VULCAN state: {e}", exc_info=True)
            return {"available": False, "error": str(e)}

    def get_integration_statistics(self) -> Dict[str, Any]:
        """
        Get integration performance statistics

        Returns:
            Dictionary with comprehensive statistics
        """
        stats = self._stats.copy()
        stats["vulcan_available"] = self.vulcan_available
        stats["config"] = self.config.to_dict()
        stats["cache_size"] = len(self._cache)
        stats["consensus_history_size"] = len(self._history)
        stats["transfer_failures_size"] = len(self.transfer_failures)
        stats["pending_tasks"] = len(self._pending_tasks)

        if self.world_model:
            try:
                # Check if method exists before calling
                if hasattr(self.world_model, "get_system_metrics"):
                    stats["vulcan_metrics"] = self.world_model.get_system_metrics()
                else:
                    stats["vulcan_metrics"] = {
                        "note": "get_system_metrics not available"
                    }
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                logger.debug(f"Could not get VULCAN metrics: {e}")
                stats["vulcan_metrics"] = {"error": str(e)}

        return stats

    def get_recent_consensus_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent consensus voting results

        Args:
            limit: Maximum number of results to return

        Returns:
            List of recent consensus results
        """
        # Get most recent entries up to limit
        history_list = list(self._history)
        return history_list[-limit:] if len(history_list) > limit else history_list

    def get_recent_transfer_failures(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent concept transfer failures

        Args:
            limit: Maximum number of results to return

        Returns:
            List of recent transfer failures
        """
        failures_list = list(self.transfer_failures)
        recent = failures_list[-limit:] if len(failures_list) > limit else failures_list

        return [
            {
                "concept": f.concept,
                "error": f.error,
                "timestamp": f.timestamp,
                "context": f.context,
            }
            for f in recent
        ]

    # ========================================================================
    # ADDED METHODS PER REQUEST
    # ========================================================================

    async def validate_graph_request(self, graph: Dict[str, Any]) -> ValidationResponse:
        """
        Ask VULCAN if executing this graph is allowed.
        This can do safety, alignment, or motivational checks.
        """
        if not self.config.enable_safety_checks or not self.vulcan_available:
            return ValidationResponse(
                success=True,
                reason="Bypassed (no safety / VULCAN offline)",
                confidence=0.0,
            )

        # Basic shape check (cheap)
        if not self._validate_graph_structure(graph):
            return ValidationResponse(
                success=False,
                reason="Invalid graph structure",
                confidence=1.0,
                details={"stage": "structure"},
            )

        # Integrate with VULCAN safety validation if available
        if self.world_model and hasattr(self.world_model, "safety_validator"):
            try:
                safety_result = self._safety_validate_graph(graph)
                if not safety_result.get("safe", True):
                    return ValidationResponse(
                        success=False,
                        reason=safety_result.get("reason", "Safety check failed"),
                        confidence=0.9,  # High confidence for safety block
                        details={
                            "stage": "safety",
                            "violations": safety_result.get("violations", []),
                        },
                    )
            except Exception as e:
                logger.error(f"Error during safety validation in pre-check: {e}")
                # Potentially block execution if safety check fails unexpectedly
                return ValidationResponse(
                    success=False,
                    reason=f"Safety check error: {e}",
                    confidence=0.5,
                    details={"stage": "safety_error"},
                )

        # Add richer policy checks here using self.world_model, self.motivational_driver, etc.
        # Example using motivational driver (if available)
        if self.motivational_driver and hasattr(
            self.motivational_driver, "validate_proposal_alignment"
        ):
            # Extract a pseudo-proposal from the graph intent
            pseudo_proposal = {
                "description": f"Execute graph {graph.get('id', 'unnamed')}",
                "graph": graph,
            }
            try:
                alignment = self.motivational_driver.validate_proposal_alignment(
                    pseudo_proposal
                )
                if isinstance(alignment, dict) and not alignment.get("aligned", True):
                    return ValidationResponse(
                        success=False,
                        reason=alignment.get("reasoning", "Motivational misalignment"),
                        confidence=alignment.get("confidence", 0.7),
                        details={"stage": "motivation"},
                    )
            except Exception as e:
                logger.warning(f"Motivational alignment check failed: {e}")

        # If all checks pass
        return ValidationResponse(
            success=True, reason="Approved", confidence=0.8, details={"stage": "policy"}
        )

    async def on_run_complete(
        self,
        graph: Dict[str, Any],
        result: GraphExecutionResult,  # Use the correct type hint
        metrics: Optional[ExecutionMetrics],  # Use the correct type hint
        extension_meta: Dict[str, Any],
    ):
        """
        Called by UnifiedRuntime after every run.
        Use this to update motivation/introspection, semantic memory, etc.
        """
        try:
            # Record basic run outcome in _history
            status_value = None
            if hasattr(result, "status") and hasattr(result.status, "value"):
                status_value = result.status.value
            elif hasattr(result, "status"):
                status_value = result.status

            metrics_summary = (
                metrics.to_dict() if metrics and hasattr(metrics, "to_dict") else {}
            )

            self._history.append(
                {
                    "ts": time.time(),
                    "status": status_value,
                    "nodes_executed": getattr(
                        result, "nodes_executed", metrics_summary.get("nodes_executed")
                    ),
                    "total_latency_ms": metrics_summary.get("total_latency_ms"),
                    "extension_meta": extension_meta,
                }
            )
            self._stats["runs_observed"] += 1

            # Optionally call world_model or motivational_driver with run summary
            if self.world_model:
                if hasattr(self.world_model, "observe_execution_outcome"):
                    # Create a summary payload
                    outcome_summary = {
                        "graph_id": graph.get("id"),
                        "status": status_value,
                        "metrics": metrics_summary,
                        "output_summary": str(getattr(result, "output", None))[
                            :200
                        ],  # Summary of output
                        "errors": getattr(result, "errors", []),
                    }
                    if asyncio.iscoroutinefunction(
                        self.world_model.observe_execution_outcome
                    ):
                        await self.world_model.observe_execution_outcome(
                            outcome_summary
                        )
                    else:
                        self.world_model.observe_execution_outcome(outcome_summary)

        except Exception as e:
            logger.error(f"Error in VULCAN on_run_complete hook: {e}")

    async def shutdown(self):
        """
        Cleanup integration resources, including VULCAN components.
        """
        logger.info("Shutting down VULCAN-Graphix bridge...")

        # Cancel pending tasks
        tasks_to_cancel = list(self._pending_tasks)
        self._pending_tasks.clear()
        if tasks_to_cancel:
            logger.info(f"Cancelling {len(tasks_to_cancel)} pending tasks...")
            for task in tasks_to_cancel:
                if not task.done():
                    task.cancel()
            # Wait for tasks to be cancelled
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)

        # Cleanup VULCAN world model if it has cleanup methods
        if self.world_model:
            logger.info("Shutting down VULCAN world model...")
            try:
                # Try standard shutdown/cleanup methods
                shutdown_method = None
                for method_name in ["shutdown", "cleanup", "close"]:
                    if hasattr(self.world_model, method_name):
                        shutdown_method = getattr(self.world_model, method_name)
                        break

                if shutdown_method:
                    if inspect.iscoroutinefunction(shutdown_method):
                        await shutdown_method()
                    else:
                        shutdown_method()
                    logger.debug(f"VULCAN world model {method_name}() called")
                else:
                    logger.debug(
                        "No standard cleanup method found on VULCAN world model"
                    )

            except Exception as e:
                logger.error(f"Error shutting down VULCAN world model: {e}")

        # Clear local state
        self._cache.clear()
        self.cache_timestamps.clear()
        self._history.clear()
        self.transfer_failures.clear()
        self._stats.clear()

        logger.info("✅ VULCAN-Graphix bridge shutdown complete")


# ============================================================================
# EXPORT HELPER
# ============================================================================


def enable_vulcan_integration(
    runtime: Any, config: Optional[VulcanIntegrationConfig] = None
) -> Optional[VulcanGraphixBridge]:
    """
    Enable VULCAN integration for a Graphix runtime

    Creates and configures a VulcanGraphixBridge instance that wraps
    the provided runtime with VULCAN capabilities.

    Args:
        runtime: UnifiedRuntime instance from Graphix
        config: Optional integration configuration. If None, uses defaults.

    Returns:
        Configured VulcanGraphixBridge instance or None if failed.

    Example:
        >>> from unified_runtime import UnifiedRuntime
        >>> runtime = UnifiedRuntime()
        >>> bridge = enable_vulcan_integration(runtime)
        >>> if bridge:
        >>>    # VULCAN integration enabled, runtime.vulcan_bridge is set
        >>>    pass
    """
    try:
        bridge = VulcanGraphixBridge(runtime=runtime, config=config)

        status = "enabled" if bridge.vulcan_available else "disabled"
        logger.info(f"VULCAN integration {status}")

        if not bridge.vulcan_available:
            logger.warning(
                "⚠️  VULCAN integration disabled - VULCAN components not available"
            )
            logger.warning("   Graph execution will proceed without VULCAN validation")
            # Return None or the bridge? Returning bridge allows partial functionality if desired.
            # Let's return the bridge instance even if VULCAN isn't fully available,
            # as the methods internally check `self.vulcan_available`.

        return bridge

    except Exception as e:
        logger.error(f"Failed to enable VULCAN integration: {e}")
        return None
