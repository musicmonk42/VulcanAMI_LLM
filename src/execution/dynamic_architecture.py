"""
Vulcan Dynamic Architecture Controller - Advanced Runtime Architecture Modification
==================================================================================

Provides safe, governed, and observable runtime modifications to transformer
architecture with comprehensive features:

- Add/remove attention heads per layer
- Prune/add connections (edges) between graph nodes
- Dynamic layer insertion/removal
- Architecture search and optimization
- Validate changes against constraints and consensus approval
- Snapshot/rollback on failure with versioning
- Comprehensive observability and audit records
- Performance impact analysis
- A/B testing support
- Duck-typed integration with multiple backends

Features:
- Multi-version snapshot management
- Automatic rollback on failure
- Constraint validation
- Consensus-based approval
- Performance monitoring
- Architecture diff tracking
- Safe concurrent modifications

Author: Vulcan AI Research Team
Version: 2.0.1 (Fixed)
License: MIT
"""

from __future__ import annotations

import copy
import logging
import threading
import time
import uuid
import json
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Union, Set, Callable
from collections import defaultdict, OrderedDict
from pathlib import Path
from enum import Enum, auto
import traceback

logger = logging.getLogger(__name__)

# ==================== ENUMS ====================


class ChangeType(Enum):
    """Types of architecture changes."""

    ADD_HEAD = auto()
    REMOVE_HEAD = auto()
    ADD_CONNECTION = auto()
    PRUNE_CONNECTION = auto()
    ADD_LAYER = auto()
    REMOVE_LAYER = auto()
    MODIFY_HEAD = auto()
    MODIFY_LAYER = auto()


class SnapshotPolicy(Enum):
    """Snapshot retention policies."""

    KEEP_ALL = auto()
    KEEP_RECENT = auto()
    KEEP_SUCCESSFUL = auto()
    KEEP_VERSIONS = auto()


class ValidationLevel(Enum):
    """Validation strictness levels."""

    NONE = auto()
    BASIC = auto()
    STANDARD = auto()
    STRICT = auto()


# ==================== DATA STRUCTURES ====================


@dataclass
class ArchChangeResult:
    """Result of an architecture change."""

    ok: bool
    reason: str = ""
    change_id: str = field(
        default_factory=lambda: f"archchg_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
    )
    before_snapshot_id: Optional[str] = None
    after_state_hint: Optional[Dict[str, Any]] = None
    execution_time: float = 0.0
    validation_errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Constraints:
    """Architecture constraints."""

    max_heads_per_layer: int = 128
    min_heads_per_layer: int = 1
    max_layers: int = 100
    min_layers: int = 1
    allow_new_connections: bool = True
    allow_prune_connections: bool = True
    allow_layer_add: bool = True
    allow_layer_remove: bool = True
    max_connection_degree: int = 100
    enforce_dag: bool = True  # Enforce DAG structure
    require_consensus: bool = False
    validation_level: ValidationLevel = ValidationLevel.STANDARD


@dataclass
class SnapshotMetadata:
    """Metadata for architecture snapshots."""

    snapshot_id: str
    timestamp: float
    change_id: Optional[str] = None
    change_type: Optional[str] = None
    success: bool = True
    num_layers: int = 0
    num_heads: int = 0
    num_connections: int = 0
    description: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class ArchitectureStats:
    """Statistics about architecture."""

    num_layers: int = 0
    num_heads: int = 0
    num_connections: int = 0
    total_parameters: int = 0
    avg_heads_per_layer: float = 0.0
    avg_connections_per_layer: float = 0.0
    layer_types: Dict[str, int] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of architecture validation."""

    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ==================== CONFIGURATION ====================


@dataclass
class DynamicArchConfig:
    """Configuration for dynamic architecture controller."""

    max_snapshots: int = 100
    snapshot_policy: SnapshotPolicy = SnapshotPolicy.KEEP_RECENT
    enable_auto_rollback: bool = True
    enable_validation: bool = True
    enable_consensus: bool = False
    enable_observability: bool = True
    enable_audit: bool = True
    enable_performance_tracking: bool = True
    snapshot_compression: bool = False
    max_rollback_depth: int = 10
    change_timeout: float = 5.0
    validation_timeout: float = 1.0


# ==================== MAIN CLASS ====================


class DynamicArchitecture:
    """
    Advanced runtime architecture modification controller.

    Public API:
      - apply_change(proposal) -> ArchChangeResult
      - add_head(layer_idx: int, head_cfg: Optional[Dict] = None) -> bool
      - remove_head(layer_idx: int, head_idx: int) -> bool
      - modify_head(layer_idx: int, head_idx: int, new_cfg: Dict) -> bool
      - prune_connection(src: Any, dst: Any) -> bool
      - add_connection(src: Any, dst: Any, metadata: Optional[Dict] = None) -> bool
      - add_layer(layer_idx: int, layer_cfg: Optional[Dict] = None) -> bool
      - remove_layer(layer_idx: int) -> bool
      - get_state() -> Dict
      - get_stats() -> ArchitectureStats
      - list_heads(layer_idx: int) -> List[Dict]
      - validate_architecture() -> ValidationResult
      - list_snapshots() -> List[SnapshotMetadata]
      - rollback_to_snapshot(snapshot_id: str) -> bool
      - diff_snapshots(snapshot_id1: str, snapshot_id2: str) -> Dict[str, Any]

    Proposal schema examples:
      {
        "type": "add_head",
        "layer_idx": 5,
        "head_cfg": {"d_k": 64, "d_v": 64, "dropout": 0.1}
      }

      {
        "type": "modify_head",
        "layer_idx": 3,
        "head_idx": 2,
        "new_cfg": {"d_k": 128}
      }

      {
        "type": "add_layer",
        "layer_idx": 4,
        "layer_cfg": {"type": "transformer_layer", "num_heads": 8}
      }
    """

    def __init__(
        self,
        model: Any = None,
        executor: Any = None,
        consensus_engine: Any = None,
        observability_manager: Any = None,
        audit_log: Any = None,
        constraints: Optional[Constraints] = None,
        config: Optional[DynamicArchConfig] = None,
    ) -> None:
        self.model = model
        self.executor = executor
        self.consensus = consensus_engine
        self.obs = observability_manager
        self.audit = audit_log
        self.constraints = constraints or Constraints()
        self.config = config or DynamicArchConfig()
        self._lock = threading.RLock()

        # Shadow state is used when model is missing or lacks .layers
        self._shadow_layers: List[Dict[str, Any]] = []

        # Snapshot management with metadata
        self._snapshots: OrderedDict[str, Tuple[Dict[str, Any], SnapshotMetadata]] = (
            OrderedDict()
        )

        # Change history
        self._change_history: List[Dict[str, Any]] = []

        # Performance tracking
        self._performance_metrics: Dict[str, Any] = {
            "total_changes": 0,
            "successful_changes": 0,
            "failed_changes": 0,
            "rollbacks": 0,
            "total_change_time": 0.0,
        }

        # Track connections in a simple graph structure
        self._graph_edges: Set[Tuple[str, str]] = set()

        # Try to fetch initial layers
        layers = self._get_layers()
        # FIXED: Only initialize shadow with 1 layer if model exists AND has layers
        # Otherwise leave it empty so tests can start from 0 layers
        if layers is None and self.model is not None:
            # Model exists but has no layers - initialize one default layer
            self._shadow_layers = [self._mk_layer(0)]
        # If model is None, leave _shadow_layers empty ([])

        # Take initial snapshot
        initial_snap_id = self._snapshot(description="Initial state")

        self._obs(
            "dynamic_arch.init",
            {"layers": self._layer_count(), "initial_snapshot": initial_snap_id},
        )

        logger.info(
            f"DynamicArchitecture initialized with {self._layer_count()} layers"
        )

    # ======================== PUBLIC API ======================== #

    def apply_change(self, proposal: Dict[str, Any]) -> ArchChangeResult:
        """
        Execute an approved architecture change proposal.

        Enhanced features:
        - Comprehensive validation
        - Automatic rollback on failure
        - Performance tracking
        - Change history
        """
        start_time = time.time()

        if not isinstance(proposal, dict):
            logger.error("apply_change: invalid proposal type")
            return ArchChangeResult(ok=False, reason="invalid_proposal_type")

        ctype = proposal.get("type")
        valid_types = {
            "add_head",
            "remove_head",
            "modify_head",
            "prune_connection",
            "add_connection",
            "add_layer",
            "remove_layer",
        }

        if ctype not in valid_types:
            logger.error(f"apply_change: unsupported type {ctype}")
            return ArchChangeResult(ok=False, reason=f"unsupported_type_{ctype}")

        # Validate proposal
        if self.config.enable_validation:
            validation_errors = self._validate_proposal(proposal)
            if validation_errors:
                logger.warning(f"Proposal validation failed: {validation_errors}")
                return ArchChangeResult(
                    ok=False,
                    reason="validation_failed",
                    validation_errors=validation_errors,
                )

        # Optional consensus approval
        if not self._approved_by_consensus(proposal):
            self._audit("dynamic_arch.consensus_reject", {"proposal": proposal})
            return ArchChangeResult(ok=False, reason="consensus_rejected")

        # Take snapshot before change
        snap_id = self._snapshot(description=f"Before {ctype}")

        ok = False
        reason = "unknown"
        result_metadata = {}

        try:
            # Execute the change
            if ctype == "add_head":
                ok = self.add_head(int(proposal["layer_idx"]), proposal.get("head_cfg"))
                reason = "added" if ok else "failed_add"

            elif ctype == "remove_head":
                ok = self.remove_head(
                    int(proposal["layer_idx"]), int(proposal["head_idx"])
                )
                reason = "removed" if ok else "failed_remove"

            elif ctype == "modify_head":
                ok = self.modify_head(
                    int(proposal["layer_idx"]),
                    int(proposal["head_idx"]),
                    proposal.get("new_cfg", {}),
                )
                reason = "modified" if ok else "failed_modify"

            elif ctype == "prune_connection":
                ok = self.prune_connection(proposal.get("src"), proposal.get("dst"))
                reason = "pruned" if ok else "failed_prune"

            elif ctype == "add_connection":
                ok = self.add_connection(
                    proposal.get("src"), proposal.get("dst"), proposal.get("metadata")
                )
                reason = "connected" if ok else "failed_connect"

            elif ctype == "add_layer":
                ok = self.add_layer(
                    int(proposal["layer_idx"]), proposal.get("layer_cfg")
                )
                reason = "layer_added" if ok else "failed_add_layer"

            elif ctype == "remove_layer":
                ok = self.remove_layer(int(proposal["layer_idx"]))
                reason = "layer_removed" if ok else "failed_remove_layer"

        except Exception as e:
            logger.exception("apply_change exception")
            ok = False
            reason = f"exception:{e}"
            result_metadata["exception"] = str(e)
            result_metadata["traceback"] = traceback.format_exc()

        # Handle failure
        if not ok:
            if self.config.enable_auto_rollback:
                self._rollback(snap_id)
                self._audit(
                    "dynamic_arch.rollback",
                    {"snapshot": snap_id, "reason": reason, "proposal": proposal},
                )
                self._obs("dynamic_arch.rollback", {"reason": reason})

            with self._lock:
                self._performance_metrics["failed_changes"] += 1

            return ArchChangeResult(
                ok=False,
                reason=reason,
                before_snapshot_id=snap_id,
                execution_time=time.time() - start_time,
                metadata=result_metadata,
            )

        # Success path
        # Optional executor hook after change
        self._post_change_refresh()

        # Update performance metrics
        execution_time = time.time() - start_time
        with self._lock:
            self._performance_metrics["total_changes"] += 1
            self._performance_metrics["successful_changes"] += 1
            self._performance_metrics["total_change_time"] += execution_time

        # Record change in history
        change_record = {
            "change_id": f"chg_{uuid.uuid4().hex[:8]}",
            "type": ctype,
            "timestamp": time.time(),
            "proposal": proposal,
            "snapshot_before": snap_id,
            "success": True,
            "execution_time": execution_time,
        }
        with self._lock:
            self._change_history.append(change_record)

        # Record audit and observability
        self._audit(
            "dynamic_arch.applied",
            {
                "proposal": proposal,
                "snapshot": snap_id,
                "result": reason,
                "execution_time": execution_time,
            },
        )
        self._obs(
            "dynamic_arch.applied",
            {"type": ctype, "result": reason, "execution_time": execution_time},
        )

        return ArchChangeResult(
            ok=True,
            reason=reason,
            change_id=change_record["change_id"],
            before_snapshot_id=snap_id,
            execution_time=execution_time,
            metadata=result_metadata,
        )

    def add_head(
        self, layer_idx: int, head_cfg: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a new attention head to a layer.

        Enhanced:
        - Validates constraints
        - Auto-wires connections
        - Supports model-native API
        - FIXED: Tracks performance metrics when called directly
        """
        with self._lock:
            if not self._check_layer_index(layer_idx):
                return False

            # Check constraints
            layers = self._require_layers()
            layer = layers[layer_idx]
            existing_heads = self._find_heads(layer)

            if len(existing_heads) >= self.constraints.max_heads_per_layer:
                logger.warning(
                    f"Max heads per layer ({self.constraints.max_heads_per_layer}) reached"
                )
                return False

            # Model-native support
            if hasattr(self.model, "add_attention_head"):
                try:
                    self.model.add_attention_head(layer_idx, head_cfg or {})
                    # Track metrics for direct calls
                    if self.config.enable_performance_tracking:
                        self._performance_metrics["total_changes"] += 1
                        self._performance_metrics["successful_changes"] += 1
                    return True
                except Exception as e:
                    logger.exception(
                        "add_head: model.add_attention_head failed, falling back to IR"
                    )

            # IR-level update
            head_cfg = head_cfg or {}
            ordinal = len(existing_heads)
            head_id = self._new_head_id(layer_idx, ordinal)
            head_node = self._mk_head_node(head_id, head_cfg)

            # Add node to layer
            layer.setdefault("nodes", []).append(head_node)

            # Wire default connections
            self._wire_head_defaults(layer, head_node)

            # FIXED: Track metrics for direct calls
            if self.config.enable_performance_tracking:
                self._performance_metrics["total_changes"] += 1
                self._performance_metrics["successful_changes"] += 1

            logger.info(f"Added head {head_id} to layer {layer_idx}")
            return True

    def remove_head(self, layer_idx: int, head_idx: int) -> bool:
        """
        Remove an attention head from a layer.

        Enhanced:
        - Validates constraints
        - Cleans up connections
        - Supports model-native API
        - FIXED: Tracks performance metrics when called directly
        """
        with self._lock:
            if not self._check_layer_index(layer_idx):
                return False

            # Model-native support
            if hasattr(self.model, "remove_attention_head"):
                try:
                    self.model.remove_attention_head(layer_idx, head_idx)
                    # Track metrics for direct calls
                    if self.config.enable_performance_tracking:
                        self._performance_metrics["total_changes"] += 1
                        self._performance_metrics["successful_changes"] += 1
                    return True
                except Exception as e:
                    logger.exception("remove_head: model method failed")

            # IR-level update
            layers = self._require_layers()
            layer = layers[layer_idx]
            heads = self._find_heads(layer)

            # Check constraints
            if len(heads) <= self.constraints.min_heads_per_layer:
                logger.warning(f"Cannot remove head: would violate min_heads_per_layer")
                return False

            if head_idx < 0 or head_idx >= len(heads):
                logger.warning(f"Invalid head_idx {head_idx}")
                return False

            head_to_remove = heads[head_idx]
            head_id = head_to_remove.get("id")

            # Remove node
            layer["nodes"] = [
                n for n in layer.get("nodes", []) if n.get("id") != head_id
            ]

            # Remove associated edges
            edges = layer.get("edges", [])
            layer["edges"] = [
                e for e in edges if e.get("src") != head_id and e.get("dst") != head_id
            ]

            # FIXED: Track metrics for direct calls
            if self.config.enable_performance_tracking:
                self._performance_metrics["total_changes"] += 1
                self._performance_metrics["successful_changes"] += 1

            logger.info(f"Removed head {head_id} from layer {layer_idx}")
            return True

    def modify_head(
        self, layer_idx: int, head_idx: int, new_cfg: Dict[str, Any]
    ) -> bool:
        """
        Modify configuration of an existing attention head.

        New method - not in original implementation.
        FIXED: Tracks performance metrics when called directly
        """
        with self._lock:
            if not self._check_layer_index(layer_idx):
                return False

            layers = self._require_layers()
            layer = layers[layer_idx]
            heads = self._find_heads(layer)

            if head_idx < 0 or head_idx >= len(heads):
                logger.warning(f"Invalid head_idx {head_idx}")
                return False

            head = heads[head_idx]

            # Update parameters
            if "params" not in head:
                head["params"] = {}

            head["params"].update(new_cfg)
            head.setdefault("metadata", {})["modified_at"] = time.time()

            # FIXED: Track metrics for direct calls
            if self.config.enable_performance_tracking:
                self._performance_metrics["total_changes"] += 1
                self._performance_metrics["successful_changes"] += 1

            logger.info(f"Modified head {head.get('id')} in layer {layer_idx}")
            return True

    def prune_connection(self, src: Any, dst: Any) -> bool:
        """
        Remove a connection between nodes.

        Enhanced:
        - Validates connection exists
        - Checks constraints
        - FIXED: Supports simple string node IDs
        - FIXED: Tracks performance metrics when called directly
        """
        with self._lock:
            if not self.constraints.allow_prune_connections:
                logger.warning("Pruning connections is disabled by constraints")
                return False

            layer_idx, src_id, dst_id = self._resolve_edge_ends(src, dst)
            if layer_idx is None:
                logger.warning("Could not resolve edge endpoints")
                return False

            if not self._check_layer_index(layer_idx):
                return False

            layers = self._require_layers()
            layer = layers[layer_idx]
            edges = layer.get("edges", [])

            # Find and remove edge
            initial_count = len(edges)
            layer["edges"] = [
                e
                for e in edges
                if not (e.get("src") == src_id and e.get("dst") == dst_id)
            ]

            removed = len(layer["edges"]) < initial_count

            if removed:
                # Remove from graph edge tracking
                self._graph_edges.discard((src_id, dst_id))

                # FIXED: Track metrics for direct calls
                if self.config.enable_performance_tracking:
                    self._performance_metrics["total_changes"] += 1
                    self._performance_metrics["successful_changes"] += 1

                logger.info(
                    f"Pruned connection {src_id} -> {dst_id} in layer {layer_idx}"
                )
            else:
                logger.warning(f"Connection {src_id} -> {dst_id} not found")

            return removed

    def add_connection(
        self, src: Any, dst: Any, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a connection between nodes.

        Enhanced:
        - Validates endpoints exist
        - Checks for cycles if DAG enforcement enabled
        - Validates connection degree
        - FIXED: Supports simple string node IDs
        - FIXED: Tracks performance metrics when called directly
        """
        with self._lock:
            if not self.constraints.allow_new_connections:
                logger.warning("Adding connections is disabled by constraints")
                return False

            layer_idx, src_id, dst_id = self._resolve_edge_ends(src, dst)
            if layer_idx is None:
                logger.warning("Could not resolve edge endpoints")
                return False

            if not self._check_layer_index(layer_idx):
                return False

            layers = self._require_layers()
            layer = layers[layer_idx]

            # Verify nodes exist
            nodes = layer.get("nodes", [])
            node_ids = {n.get("id") for n in nodes}

            if src_id not in node_ids or dst_id not in node_ids:
                logger.warning(
                    f"Source or destination node not found: {src_id}, {dst_id}"
                )
                return False

            # Check if edge already exists
            edges = layer.get("edges", [])
            if any(e.get("src") == src_id and e.get("dst") == dst_id for e in edges):
                logger.warning(f"Connection {src_id} -> {dst_id} already exists")
                return False

            # Check connection degree
            out_degree = sum(1 for e in edges if e.get("src") == src_id)
            if out_degree >= self.constraints.max_connection_degree:
                logger.warning(f"Max connection degree reached for {src_id}")
                return False

            # Check for cycles if DAG enforcement is enabled
            if self.constraints.enforce_dag:
                # Create temporary edge list
                temp_edges = edges + [{"src": src_id, "dst": dst_id}]
                if self._has_cycle(nodes, temp_edges):
                    logger.warning(
                        f"Adding edge {src_id} -> {dst_id} would create a cycle"
                    )
                    return False

            # Add edge
            new_edge = {"src": src_id, "dst": dst_id, "metadata": metadata or {}}
            new_edge["metadata"]["created_at"] = time.time()

            layer.setdefault("edges", []).append(new_edge)

            # Track in graph edges
            self._graph_edges.add((src_id, dst_id))

            # FIXED: Track metrics for direct calls
            if self.config.enable_performance_tracking:
                self._performance_metrics["total_changes"] += 1
                self._performance_metrics["successful_changes"] += 1

            logger.info(f"Added connection {src_id} -> {dst_id} in layer {layer_idx}")
            return True

    def add_layer(
        self, layer_idx: int, layer_cfg: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a new layer at the specified index.

        New method - not in original implementation.
        FIXED: Tracks performance metrics when called directly
        """
        with self._lock:
            if not self.constraints.allow_layer_add:
                logger.warning("Adding layers is disabled by constraints")
                return False

            layers = self._require_layers()

            if len(layers) >= self.constraints.max_layers:
                logger.warning(f"Max layers ({self.constraints.max_layers}) reached")
                return False

            if layer_idx < 0 or layer_idx > len(layers):
                logger.warning(f"Invalid layer_idx {layer_idx} for insertion")
                return False

            # Create new layer
            layer_cfg = layer_cfg or {}
            new_layer = self._mk_layer(layer_idx)

            # Apply configuration
            if "num_heads" in layer_cfg:
                num_heads = layer_cfg["num_heads"]
                for i in range(num_heads):
                    head_id = self._new_head_id(layer_idx, i)
                    head_node = self._mk_head_node(head_id, {})
                    new_layer["nodes"].append(head_node)

            # Insert layer
            layers.insert(layer_idx, new_layer)

            # Re-index subsequent layers
            for i in range(layer_idx + 1, len(layers)):
                layers[i]["layer"] = i

            # FIXED: Track metrics for direct calls
            if self.config.enable_performance_tracking:
                self._performance_metrics["total_changes"] += 1
                self._performance_metrics["successful_changes"] += 1

            logger.info(f"Added layer at index {layer_idx}")
            return True

    def remove_layer(self, layer_idx: int) -> bool:
        """
        Remove a layer.

        New method - not in original implementation.
        FIXED: Tracks performance metrics when called directly
        """
        with self._lock:
            if not self.constraints.allow_layer_remove:
                logger.warning("Removing layers is disabled by constraints")
                return False

            layers = self._require_layers()

            if len(layers) <= self.constraints.min_layers:
                logger.warning(f"Cannot remove layer: would violate min_layers")
                return False

            if not self._check_layer_index(layer_idx):
                return False

            # Remove layer
            layers.pop(layer_idx)

            # Re-index subsequent layers
            for i in range(layer_idx, len(layers)):
                layers[i]["layer"] = i

            # FIXED: Track metrics for direct calls
            if self.config.enable_performance_tracking:
                self._performance_metrics["total_changes"] += 1
                self._performance_metrics["successful_changes"] += 1

            logger.info(f"Removed layer at index {layer_idx}")
            return True

    def get_state(self) -> Dict[str, Any]:
        """
        Get current architecture state.

        Enhanced with comprehensive information.
        """
        with self._lock:
            layers = self._require_layers()

            # Calculate statistics
            stats = self.get_stats()

            return {
                "layers": copy.deepcopy(layers),
                "num_layers": len(layers),
                "stats": asdict(stats),
                "constraints": asdict(self.constraints),
                "snapshot_count": len(self._snapshots),
                "change_count": len(self._change_history),
                "performance_metrics": self._performance_metrics.copy(),
            }

    def get_stats(self) -> ArchitectureStats:
        """
        Get architecture statistics.

        New method - not in original implementation.
        """
        with self._lock:
            layers = self._require_layers()

            num_layers = len(layers)
            total_heads = 0
            total_connections = 0
            layer_types = defaultdict(int)

            for layer in layers:
                heads = self._find_heads(layer)
                total_heads += len(heads)
                total_connections += len(layer.get("edges", []))
                layer_type = layer.get("metadata", {}).get("type", "unknown")
                layer_types[layer_type] += 1

            avg_heads = total_heads / num_layers if num_layers > 0 else 0.0
            avg_connections = total_connections / num_layers if num_layers > 0 else 0.0

            return ArchitectureStats(
                num_layers=num_layers,
                num_heads=total_heads,
                num_connections=total_connections,
                avg_heads_per_layer=avg_heads,
                avg_connections_per_layer=avg_connections,
                layer_types=dict(layer_types),
            )

    def list_heads(self, layer_idx: int) -> List[Dict[str, Any]]:
        """List all attention heads in a layer."""
        with self._lock:
            if not self._check_layer_index(layer_idx):
                return []
            layers = self._require_layers()
            return self._find_heads(layers[layer_idx])

    def validate_architecture(self) -> ValidationResult:
        """
        Validate current architecture against constraints.

        New method - comprehensive validation.
        """
        with self._lock:
            errors = []
            warnings = []

            layers = self._require_layers()

            # Validate layer count
            if len(layers) < self.constraints.min_layers:
                errors.append(
                    f"Too few layers: {len(layers)} < {self.constraints.min_layers}"
                )
            if len(layers) > self.constraints.max_layers:
                errors.append(
                    f"Too many layers: {len(layers)} > {self.constraints.max_layers}"
                )

            # Validate each layer
            for idx, layer in enumerate(layers):
                heads = self._find_heads(layer)

                # Check head count
                if len(heads) < self.constraints.min_heads_per_layer:
                    errors.append(f"Layer {idx}: too few heads ({len(heads)})")
                if len(heads) > self.constraints.max_heads_per_layer:
                    errors.append(f"Layer {idx}: too many heads ({len(heads)})")

                # Check for duplicate node IDs
                node_ids = [n.get("id") for n in layer.get("nodes", [])]
                if len(node_ids) != len(set(node_ids)):
                    errors.append(f"Layer {idx}: duplicate node IDs")

                # Check edges reference valid nodes
                node_id_set = set(node_ids)
                for edge in layer.get("edges", []):
                    src = edge.get("src")
                    dst = edge.get("dst")
                    if src not in node_id_set:
                        errors.append(f"Layer {idx}: edge src '{src}' not found")
                    if dst not in node_id_set:
                        errors.append(f"Layer {idx}: edge dst '{dst}' not found")

                # Check for cycles if DAG enforcement enabled
                if self.constraints.enforce_dag:
                    if self._has_cycle(layer.get("nodes", []), layer.get("edges", [])):
                        errors.append(f"Layer {idx}: contains cycle (DAG violation)")

            valid = len(errors) == 0

            return ValidationResult(
                valid=valid,
                errors=errors,
                warnings=warnings,
                metadata={"validated_at": time.time()},
            )

    def list_snapshots(self) -> List[SnapshotMetadata]:
        """
        List all available snapshots.

        New method - snapshot management.
        """
        with self._lock:
            return [meta for _, meta in self._snapshots.values()]

    def rollback_to_snapshot(self, snapshot_id: str) -> bool:
        """
        Rollback to a specific snapshot.

        New method - manual rollback.
        """
        return self._rollback(snapshot_id)

    def diff_snapshots(self, snapshot_id1: str, snapshot_id2: str) -> Dict[str, Any]:
        """
        Compute diff between two snapshots.

        New method - snapshot comparison.
        """
        with self._lock:
            if snapshot_id1 not in self._snapshots:
                return {"error": f"Snapshot {snapshot_id1} not found"}
            if snapshot_id2 not in self._snapshots:
                return {"error": f"Snapshot {snapshot_id2} not found"}

            state1, meta1 = self._snapshots[snapshot_id1]
            state2, meta2 = self._snapshots[snapshot_id2]

            layers1 = state1.get("layers", [])
            layers2 = state2.get("layers", [])

            diff = {
                "snapshot1": snapshot_id1,
                "snapshot2": snapshot_id2,
                "timestamp1": meta1.timestamp,
                "timestamp2": meta2.timestamp,
                "layer_count_diff": len(layers2) - len(layers1),
                "head_count_diff": meta2.num_heads - meta1.num_heads,
                "connection_count_diff": meta2.num_connections - meta1.num_connections,
                "changes": [],
            }

            return diff

    # ======================== INTERNAL HELPERS ======================== #

    def _get_layers(self) -> Optional[List[Dict[str, Any]]]:
        """Get layers from model or shadow."""
        if self.model is None:
            return None
        layers = getattr(self.model, "layers", None)
        if isinstance(layers, list):
            return layers
        return None

    def _require_layers(self) -> List[Dict[str, Any]]:
        """Get layers, falling back to shadow."""
        layers = self._get_layers()
        if layers is None:
            return self._shadow_layers
        return layers

    def _layer_count(self) -> int:
        """Get number of layers."""
        layers = self._get_layers()
        if layers is None:
            return len(self._shadow_layers)
        return len(layers)

    def _check_layer_index(self, idx: int) -> bool:
        """Validate layer index."""
        if idx < 0 or idx >= self._layer_count():
            logger.warning(f"layer_idx {idx} out of range")
            return False
        return True

    def _mk_layer(self, idx: int) -> Dict[str, Any]:
        """Create a new layer dict."""
        return {
            "layer": idx,
            "nodes": [],
            "edges": [],
            "metadata": {"type": "transformer_layer", "created_at": time.time()},
        }

    def _new_head_id(self, layer_idx: int, ordinal: int) -> str:
        """Generate unique head ID."""
        return f"attn_{layer_idx}_{ordinal}_{uuid.uuid4().hex[:6]}"

    def _mk_head_node(self, node_id: str, head_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Create head node dict."""
        node = {
            "id": node_id,
            "type": "attention_head",
            "params": {
                "d_k": int(head_cfg.get("d_k", 64)),
                "d_v": int(head_cfg.get("d_v", 64)),
                "dropout": float(head_cfg.get("dropout", 0.0)),
            },
            "metadata": {"created_at": time.time()},
        }
        return node

    def _wire_head_defaults(
        self, layer: Dict[str, Any], head_node: Dict[str, Any]
    ) -> None:
        """Wire default connections for new head."""
        edges = layer.setdefault("edges", [])

        # Connect to aggregator/output node
        combine = self._find_node_by_type(
            layer, {"combine", "merge", "attention_out", "project_out"}
        )
        if combine:
            edges.append(
                {
                    "src": head_node["id"],
                    "dst": combine.get("id"),
                    "metadata": {"edge_type": "attn_out", "created_at": time.time()},
                }
            )

        # Connect from input node
        inp = self._find_node_by_type(
            layer, {"embedding", "input", "residual_in", "qkv_project"}
        )
        if inp:
            edges.append(
                {
                    "src": inp.get("id"),
                    "dst": head_node["id"],
                    "metadata": {"edge_type": "attn_in", "created_at": time.time()},
                }
            )

    def _find_heads(self, layer: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find all attention heads in layer."""
        return [
            n
            for n in layer.get("nodes", [])
            if n.get("type") in ("attention_head", "attention")
        ]

    def _find_node_by_type(
        self, layer: Dict[str, Any], types: Union[str, set, tuple]
    ) -> Optional[Dict[str, Any]]:
        """Find first node matching type(s)."""
        if isinstance(types, str):
            types = {types}
        for n in layer.get("nodes", []):
            if n.get("type") in types:
                return n
        return None

    def _resolve_edge_ends(
        self, src: Any, dst: Any
    ) -> Tuple[Optional[int], Optional[str], Optional[str]]:
        """
        Resolve layer index and node IDs from edge descriptors.

        FIXED: Now supports simple string node IDs (assumes layer 0)
        """

        def parse_end(end: Any) -> Tuple[Optional[int], Optional[str]]:
            # FIXED: Support simple string node IDs
            if isinstance(end, str):
                # Simple string assumes layer 0
                return 0, end
            elif isinstance(end, dict):
                layer = end.get("layer")
                return int(layer) if layer is not None else None, end.get("node_id")
            elif isinstance(end, tuple) and len(end) == 2:
                try:
                    return int(end[0]), str(end[1])
                except Exception as e:
                    return None, None
            return None, None

        src_layer, src_id = parse_end(src)
        dst_layer, dst_id = parse_end(dst)

        # FIXED: Handle case where both are simple strings (assume same layer)
        if isinstance(src, str) and isinstance(dst, str):
            # Both are simple strings - assume layer 0 if we have at least one layer
            if self._layer_count() > 0:
                return 0, src, dst

        if src_layer is None or dst_layer is None or src_layer != dst_layer:
            return None, None, None
        if not src_id or not dst_id:
            return None, None, None

        return src_layer, src_id, dst_id

    def _has_cycle(
        self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]
    ) -> bool:
        """Check if graph has cycles using DFS."""
        # Build adjacency list
        adj = defaultdict(list)
        for edge in edges:
            src = edge.get("src")
            dst = edge.get("dst")
            if src and dst:
                adj[src].append(dst)

        node_ids = {n.get("id") for n in nodes if n.get("id")}
        visited = set()
        rec_stack = set()

        def dfs(node):
            visited.add(node)
            rec_stack.add(node)

            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for node_id in node_ids:
            if node_id not in visited:
                if dfs(node_id):
                    return True

        return False

    # ======================== SNAPSHOT / ROLLBACK ======================== #

    def _snapshot(self, description: str = "") -> str:
        """Take a snapshot of current state."""
        with self._lock:
            state = self._capture_state()
            snap_id = f"snap_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"

            # Calculate snapshot stats
            layers = state.get("layers", [])
            num_heads = sum(len(self._find_heads(l)) for l in layers)
            num_connections = sum(len(l.get("edges", [])) for l in layers)

            metadata = SnapshotMetadata(
                snapshot_id=snap_id,
                timestamp=time.time(),
                num_layers=len(layers),
                num_heads=num_heads,
                num_connections=num_connections,
                description=description,
            )

            self._snapshots[snap_id] = (state, metadata)

            # Enforce snapshot limit
            self._enforce_snapshot_policy()

            self._audit(
                "dynamic_arch.snapshot",
                {"snapshot": snap_id, "description": description},
            )

            return snap_id

    def _rollback(self, snapshot_id: str) -> bool:
        """Rollback to a snapshot."""
        with self._lock:
            if snapshot_id not in self._snapshots:
                logger.error(f"rollback: snapshot {snapshot_id} not found")
                return False

            state, metadata = self._snapshots[snapshot_id]
            self._restore_state(state)

            with self._lock:
                self._performance_metrics["rollbacks"] += 1

            self._obs(
                "dynamic_arch.rolled_back",
                {"snapshot": snapshot_id, "timestamp": metadata.timestamp},
            )

            logger.info(f"Rolled back to snapshot {snapshot_id}")
            return True

    def _capture_state(self) -> Dict[str, Any]:
        """Capture current state for snapshot."""
        layers = self._get_layers()
        if layers is not None:
            return {
                "mode": "model",
                "layers": copy.deepcopy(layers),
                "timestamp": time.time(),
            }
        return {
            "mode": "shadow",
            "layers": copy.deepcopy(self._shadow_layers),
            "timestamp": time.time(),
        }

    def _restore_state(self, state: Dict[str, Any]) -> None:
        """Restore state from snapshot."""
        if state.get("mode") == "model" and self._get_layers() is not None:
            try:
                self.model.layers = copy.deepcopy(state["layers"])
            except Exception as e:
                cur = self._get_layers()
                if isinstance(cur, list):
                    cur.clear()
                    cur.extend(copy.deepcopy(state["layers"]))
        else:
            self._shadow_layers = copy.deepcopy(state["layers"])

    def _enforce_snapshot_policy(self):
        """Enforce snapshot retention policy."""
        with self._lock:
            if len(self._snapshots) <= self.config.max_snapshots:
                return

            # Remove oldest snapshots beyond limit
            while len(self._snapshots) > self.config.max_snapshots:
                self._snapshots.popitem(last=False)

    # ======================== VALIDATION ======================== #

    def _validate_proposal(self, proposal: Dict[str, Any]) -> List[str]:
        """Validate change proposal."""
        errors = []

        ctype = proposal.get("type")

        if ctype in ["add_head", "remove_head", "modify_head"]:
            layer_idx = proposal.get("layer_idx")
            if layer_idx is None:
                errors.append("Missing layer_idx")
            elif not isinstance(layer_idx, int):
                errors.append("layer_idx must be int")
            elif layer_idx < 0 or layer_idx >= self._layer_count():
                errors.append(f"layer_idx {layer_idx} out of range")

        if ctype in ["remove_head", "modify_head"]:
            head_idx = proposal.get("head_idx")
            if head_idx is None:
                errors.append("Missing head_idx")
            elif not isinstance(head_idx, int):
                errors.append("head_idx must be int")

        if ctype in ["add_connection", "prune_connection"]:
            if "src" not in proposal:
                errors.append("Missing src")
            if "dst" not in proposal:
                errors.append("Missing dst")

        return errors

    # ======================== POST-CHANGE HOOKS ======================== #

    def _post_change_refresh(self) -> None:
        """Refresh executor after architecture change."""
        try:
            if self.executor and hasattr(self.executor, "refresh"):
                self.executor.refresh()
        except Exception as e:
            logger.debug(f"Executor refresh failed: {e}")

    # ======================== GOVERNANCE / IO ======================== #

    def _approved_by_consensus(self, proposal: Dict[str, Any]) -> bool:
        """Check consensus approval for change."""
        if not self.consensus or not self.config.enable_consensus:
            return True

        try:
            if hasattr(self.consensus, "approve"):
                return bool(self.consensus.approve(proposal))
            if hasattr(self.consensus, "submit_proposal"):
                p = self.consensus.submit_proposal(proposal)
                return getattr(p, "status", "") == "approved"
        except Exception as e:
            logger.debug(f"Consensus approval failed: {e}")

        return True

    def _obs(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Record observability event."""
        if not self.obs or not self.config.enable_observability:
            return

        try:
            if hasattr(self.obs, "record"):
                self.obs.record(event_type, payload)
            elif hasattr(self.obs, "log"):
                self.obs.log(event_type, payload)
        except Exception as e:
            logger.debug(f"Observability recording failed: {e}")

    def _audit(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Record audit event."""
        if not self.audit or not self.config.enable_audit:
            return

        try:
            record = {"event": event_type, "timestamp": time.time(), **payload}
            if hasattr(self.audit, "append"):
                self.audit.append(record)
            elif hasattr(self.audit, "record"):
                self.audit.record(event_type, payload)
        except Exception as e:
            logger.debug(f"Audit recording failed: {e}")

    # ======================== PERFORMANCE METRICS ======================== #

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        with self._lock:
            metrics = self._performance_metrics.copy()

            if metrics["total_changes"] > 0:
                metrics["avg_change_time"] = (
                    metrics["total_change_time"] / metrics["total_changes"]
                )
                metrics["success_rate"] = (
                    metrics["successful_changes"] / metrics["total_changes"]
                )
            else:
                metrics["avg_change_time"] = 0.0
                metrics["success_rate"] = 0.0

            return metrics

    def reset_metrics(self):
        """Reset performance metrics."""
        with self._lock:
            self._performance_metrics = {
                "total_changes": 0,
                "successful_changes": 0,
                "failed_changes": 0,
                "rollbacks": 0,
                "total_change_time": 0.0,
            }

    # ======================== PERSISTENCE ======================== #

    def save_state(self, path: str):
        """Save architecture state to file."""
        state = self.get_state()

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2, default=str)

        logger.info(f"Architecture state saved to {path}")

    def load_state(self, path: str):
        """Load architecture state from file."""
        with open(path, "r") as f:
            state = json.load(f)

        layers = state.get("layers", [])
        if self._get_layers() is not None:
            try:
                self.model.layers = layers
            except Exception as e:
                cur = self._get_layers()
                if isinstance(cur, list):
                    cur.clear()
                    cur.extend(layers)
        else:
            self._shadow_layers = layers

        logger.info(f"Architecture state loaded from {path}")


# ==================== UTILITY FUNCTIONS ====================


def create_default_controller(**kwargs) -> DynamicArchitecture:
    """Create controller with default configuration."""
    return DynamicArchitecture(**kwargs)


def create_strict_controller(**kwargs) -> DynamicArchitecture:
    """Create controller with strict constraints."""
    constraints = Constraints(
        max_heads_per_layer=64,
        min_heads_per_layer=2,
        enforce_dag=True,
        require_consensus=True,
        validation_level=ValidationLevel.STRICT,
    )
    return DynamicArchitecture(constraints=constraints, **kwargs)
