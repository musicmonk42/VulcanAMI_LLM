"""
Parameter history management and checkpointing for auditing
FIXED: Proper cross-platform path handling using Path objects exclusively
FIXED: Handle both nn.Module and dict types in async_checkpoint
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict, Optional, List, Union
from collections import deque
from dataclasses import asdict
import logging
import time
import copy
import threading
import pickle
import hashlib
from pathlib import Path
import queue
from queue import Empty as QueueEmpty
import json
from datetime import datetime

from .learning_types import LearningConfig, LearningTrajectory
from ..security_fixes import safe_pickle_load

logger = logging.getLogger(__name__)

# ============================================================
# PARAMETER HISTORY MANAGER
# ============================================================


class ParameterHistoryManager:
    """Manages parameter history and checkpointing for auditing"""

    def __init__(
        self, base_path: str = "parameter_history", config: LearningConfig = None
    ):
        self.config = config or LearningConfig()

        # FIXED: Use Path object and resolve to absolute path
        self.base_path = Path(base_path).resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.checkpoint_counter = 0

        # FIXED: Validate and cap max_checkpoints
        max_len = getattr(self.config, "max_checkpoints", 100)
        if not isinstance(max_len, int) or max_len <= 0:
            logger.warning(f"Invalid max_checkpoints ({max_len}), defaulting to 100")
            max_len = 100
        max_len = min(max_len, 10000)  # Cap at reasonable limit
        self.parameter_history = deque(maxlen=max_len)

        self.trajectory_storage = {}
        self.current_trajectory = None

        # Compression for storage efficiency
        self.compress_checkpoints = True

        # Shutdown event for clean termination
        self._shutdown = threading.Event()
        self._running = True

        # FIXED: Background thread for async checkpointing with queue size limit
        self.checkpoint_queue = queue.Queue(maxsize=5)  # Limit to prevent OOM
        self.checkpoint_thread = threading.Thread(
            target=self._checkpoint_worker, daemon=True
        )
        self.checkpoint_thread.start()

        # Statistics tracking
        self.stats = {
            "total_checkpoints": 0,
            "total_trajectories": 0,
            "failed_saves": 0,
            "successful_loads": 0,
            "failed_loads": 0,
            "queue_full_events": 0,
        }

        # Lock for thread safety
        self._lock = threading.RLock()

        # Load existing checkpoint history if available
        self._load_checkpoint_history()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure trajectory is saved and cleanup"""
        if self.current_trajectory:
            self.end_trajectory(save=True)
        self.shutdown()
        return False

    def save_checkpoint(self, model: nn.Module, metadata: Dict[str, Any] = None) -> str:
        """Save model checkpoint with metadata"""
        with self._lock:
            self.checkpoint_counter += 1

            checkpoint_id = f"checkpoint_{self.checkpoint_counter}_{int(time.time())}"

            # FIXED: Use Path object throughout
            checkpoint_path = self.base_path / f"{checkpoint_id}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "timestamp": time.time(),
                "datetime": datetime.now().isoformat(),
                "metadata": metadata or {},
                "checkpoint_id": checkpoint_id,
                "counter": self.checkpoint_counter,
                "model_class": model.__class__.__name__,
            }

            # Calculate checksum for integrity
            state_bytes = pickle.dumps(model.state_dict())
            checkpoint["checksum"] = hashlib.sha256(state_bytes).hexdigest()

            try:
                if self.compress_checkpoints:
                    import gzip

                    # FIXED: Use Path.with_suffix() and ensure parent exists
                    compressed_path = checkpoint_path.with_suffix(".pt.gz")
                    compressed_path.parent.mkdir(parents=True, exist_ok=True)

                    with gzip.open(compressed_path, "wb") as f:
                        torch.save(checkpoint, f)
                    checkpoint_path = compressed_path
                else:
                    torch.save(checkpoint, checkpoint_path)

                # Update history
                self.parameter_history.append(
                    {
                        "checkpoint_id": checkpoint_id,
                        "path": str(checkpoint_path),
                        "timestamp": checkpoint["timestamp"],
                        "datetime": checkpoint["datetime"],
                        "checksum": checkpoint["checksum"],
                        "metadata": metadata,
                        "size_bytes": checkpoint_path.stat().st_size,
                    }
                )

                # Clean old checkpoints if needed
                if len(self.parameter_history) >= self.config.max_checkpoints:
                    self._cleanup_old_checkpoints()

                self.stats["total_checkpoints"] += 1

                # Save checkpoint history
                self._save_checkpoint_history()

                logger.info(f"Saved checkpoint: {checkpoint_id}")
                return str(checkpoint_path)

            except Exception as e:
                self.stats["failed_saves"] += 1
                logger.error(f"Failed to save checkpoint: {e}")
                raise

    def async_checkpoint(
        self, model: Union[nn.Module, Dict[str, Any]], metadata: Dict[str, Any] = None
    ):
        """
        FIXED: Queue checkpoint for asynchronous saving with backpressure
        Accepts either an nn.Module or a pre-built state dict
        """
        if self._shutdown.is_set():
            logger.warning("Cannot queue checkpoint: manager is shutting down")
            return

        # FIXED: Check queue size to prevent memory explosion
        if self.checkpoint_queue.qsize() >= 5:
            with self._lock:
                self.stats["queue_full_events"] += 1
            logger.warning("Checkpoint queue full, skipping async checkpoint")
            return

        try:
            # FIXED: Handle both nn.Module and dict types
            if isinstance(model, nn.Module):
                # Extract state_dict from module
                try:
                    state_dict = copy.deepcopy(model.state_dict())
                    model_class = model.__class__.__name__
                except Exception as e:
                    logger.error(f"Failed to extract state_dict from module: {e}")
                    return
            elif isinstance(model, dict):
                # Already a state dict, verify it's picklable
                try:
                    # Test if the dict is picklable
                    pickle.dumps(model)
                    state_dict = model
                    model_class = (
                        metadata.get("model_class", "Unknown")
                        if metadata
                        else "Unknown"
                    )
                except Exception as e:
                    logger.error(f"Provided state dict is not picklable: {e}")
                    return
            else:
                logger.error(f"async_checkpoint received invalid type: {type(model)}")
                return

            # Use timeout to avoid blocking forever
            self.checkpoint_queue.put((state_dict, model_class, metadata), timeout=0.1)
        except queue.Full:
            with self._lock:
                self.stats["queue_full_events"] += 1
            logger.warning("Failed to queue checkpoint - queue full")
        except Exception as e:
            logger.error(f"Failed to queue async checkpoint: {e}")

    def _checkpoint_worker(self):
        """FIXED: Background worker for async checkpointing with proper error handling"""
        while not self._shutdown.is_set():
            try:
                state_dict, model_class, metadata = self.checkpoint_queue.get(timeout=1)

                checkpoint_id = f"async_checkpoint_{int(time.time())}"

                # FIXED: Use Path object and ensure parent exists
                checkpoint_path = self.base_path / f"{checkpoint_id}.pt"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

                checkpoint = {
                    "model_state_dict": state_dict,
                    "timestamp": time.time(),
                    "datetime": datetime.now().isoformat(),
                    "metadata": metadata or {},
                    "checkpoint_id": checkpoint_id,
                    "model_class": model_class,
                }

                # Calculate checksum
                state_bytes = pickle.dumps(state_dict)
                checkpoint["checksum"] = hashlib.sha256(state_bytes).hexdigest()

                if self.compress_checkpoints:
                    import gzip

                    # FIXED: Use Path.with_suffix() and ensure parent exists
                    compressed_path = checkpoint_path.with_suffix(".pt.gz")
                    compressed_path.parent.mkdir(parents=True, exist_ok=True)

                    with gzip.open(compressed_path, "wb") as f:
                        torch.save(checkpoint, f)
                    checkpoint_path = compressed_path
                else:
                    torch.save(checkpoint, checkpoint_path)

                with self._lock:
                    self.parameter_history.append(
                        {
                            "checkpoint_id": checkpoint_id,
                            "path": str(checkpoint_path),
                            "timestamp": checkpoint["timestamp"],
                            "datetime": checkpoint["datetime"],
                            "checksum": checkpoint["checksum"],
                            "metadata": metadata,
                            "size_bytes": checkpoint_path.stat().st_size,
                            "async": True,
                        }
                    )
                    self.stats["total_checkpoints"] += 1

                logger.debug(f"Async checkpoint saved: {checkpoint_id}")

            except QueueEmpty:
                continue
            except Exception as e:
                if not self._shutdown.is_set():
                    with self._lock:
                        self.stats["failed_saves"] += 1
                    logger.error(f"Checkpoint worker error: {e}")

    def load_checkpoint(
        self, checkpoint_path: str, model: nn.Module, strict: bool = True
    ) -> Dict[str, Any]:
        """Load checkpoint into model with validation"""
        # FIXED: Convert to Path object and resolve
        path = Path(checkpoint_path).resolve()

        if not path.exists():
            with self._lock:
                self.stats["failed_loads"] += 1
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        try:
            if path.suffix == ".gz":
                import gzip

                with gzip.open(path, "rb") as f:
                    checkpoint = torch.load(f, map_location="cpu")
            else:
                checkpoint = torch.load(path, map_location="cpu")

            # Validate checksum
            if "checksum" in checkpoint:
                state_bytes = pickle.dumps(checkpoint["model_state_dict"])
                calculated_checksum = hashlib.sha256(state_bytes).hexdigest()
                if calculated_checksum != checkpoint["checksum"]:
                    logger.warning(
                        f"Checksum mismatch for checkpoint: {checkpoint_path}"
                    )

            # Load state dict
            model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

            with self._lock:
                self.stats["successful_loads"] += 1
            logger.info(f"Loaded checkpoint from: {checkpoint_path}")

            return checkpoint.get("metadata", {})

        except Exception as e:
            with self._lock:
                self.stats["failed_loads"] += 1
            logger.error(f"Failed to load checkpoint: {e}")
            raise

    def validate_checkpoint(self, checkpoint_path: str) -> bool:
        """Validate checkpoint integrity using checksum"""
        try:
            # FIXED: Convert to Path object and resolve
            path = Path(checkpoint_path).resolve()
            if not path.exists():
                return False

            if path.suffix == ".gz":
                import gzip

                with gzip.open(path, "rb") as f:
                    checkpoint = torch.load(f, map_location="cpu")
            else:
                checkpoint = torch.load(path, map_location="cpu")

            # Verify checksum
            if "checksum" not in checkpoint:
                logger.warning(f"No checksum found in checkpoint: {checkpoint_path}")
                return True  # Assume valid if no checksum

            state_bytes = pickle.dumps(checkpoint["model_state_dict"])
            calculated_checksum = hashlib.sha256(state_bytes).hexdigest()

            is_valid = calculated_checksum == checkpoint["checksum"]
            if not is_valid:
                logger.error(f"Invalid checksum for checkpoint: {checkpoint_path}")

            return is_valid

        except Exception as e:
            logger.error(f"Checkpoint validation failed: {e}")
            return False

    def list_checkpoints(
        self, sort_by: str = "timestamp", ascending: bool = False
    ) -> List[Dict[str, Any]]:
        """List all available checkpoints with metadata"""
        with self._lock:
            checkpoints = list(self.parameter_history)

            if sort_by in ["timestamp", "size_bytes", "checkpoint_id"]:
                try:
                    checkpoints.sort(
                        key=lambda x: x.get(sort_by, 0), reverse=not ascending
                    )
                except Exception as e:
                    logger.warning(f"Failed to sort checkpoints: {e}")

            return checkpoints

    def get_checkpoint_info(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific checkpoint"""
        with self._lock:
            for checkpoint in self.parameter_history:
                if checkpoint["checkpoint_id"] == checkpoint_id:
                    return checkpoint.copy()
        return None

    def find_checkpoints_by_metadata(self, **criteria) -> List[Dict[str, Any]]:
        """Find checkpoints matching metadata criteria"""
        matching = []

        with self._lock:
            for checkpoint in self.parameter_history:
                metadata = checkpoint.get("metadata", {})
                try:
                    if all(metadata.get(k) == v for k, v in criteria.items()):
                        matching.append(checkpoint.copy())
                except Exception as e:
                    logger.debug(
                        f"Skipping checkpoint due to metadata comparison error: {e}"
                    )
                    continue

        return matching

    def get_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Get the most recent checkpoint info"""
        with self._lock:
            if self.parameter_history:
                return self.parameter_history[-1].copy()
        return None

    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a specific checkpoint"""
        with self._lock:
            for i, checkpoint in enumerate(self.parameter_history):
                if checkpoint["checkpoint_id"] == checkpoint_id:
                    # Delete file - FIXED: Use Path object
                    path = Path(checkpoint["path"]).resolve()
                    if path.exists():
                        try:
                            path.unlink()
                        except Exception as e:
                            logger.error(f"Failed to delete checkpoint file: {e}")
                            return False

                    # Remove from history - use list() to create a copy for safe iteration
                    history_list = list(self.parameter_history)
                    del history_list[i]
                    self.parameter_history = deque(
                        history_list, maxlen=self.parameter_history.maxlen
                    )

                    logger.info(f"Deleted checkpoint: {checkpoint_id}")
                    return True
        return False

    def start_trajectory(
        self, task_id: str, agent_id: str, metadata: Dict[str, Any] = None
    ) -> str:
        """Start recording a new learning trajectory"""
        trajectory_id = f"trajectory_{task_id}_{agent_id}_{int(time.time())}"

        with self._lock:
            self.current_trajectory = LearningTrajectory(
                trajectory_id=trajectory_id,
                start_time=time.time(),
                end_time=None,
                task_id=task_id,
                agent_id=agent_id,
                states=[],
                actions=[],
                rewards=[],
                losses=[],
                parameter_snapshots=[],
                metadata=metadata or {},
            )

        logger.info(f"Started trajectory: {trajectory_id}")
        return trajectory_id

    def record_step(self, state: np.ndarray, action: Any, reward: float, loss: float):
        """Record a step in the current trajectory"""
        with self._lock:
            if self.current_trajectory:
                self.current_trajectory.states.append(state)
                self.current_trajectory.actions.append(action)
                self.current_trajectory.rewards.append(reward)
                self.current_trajectory.losses.append(loss)

    def add_trajectory_checkpoint(self, checkpoint_path: str):
        """Add a checkpoint reference to the current trajectory"""
        with self._lock:
            if self.current_trajectory:
                self.current_trajectory.parameter_snapshots.append(checkpoint_path)

    def end_trajectory(
        self, save: bool = True, metadata: Dict[str, Any] = None
    ) -> Optional[str]:
        """FIXED: End current trajectory and save if requested with proper locking"""
        with self._lock:
            if not self.current_trajectory:
                return None

            self.current_trajectory.end_time = time.time()
            trajectory_id = self.current_trajectory.trajectory_id

            # Update metadata
            if metadata:
                self.current_trajectory.metadata.update(metadata)

            # Add summary statistics
            if self.current_trajectory.rewards:
                self.current_trajectory.metadata["total_reward"] = sum(
                    self.current_trajectory.rewards
                )
                self.current_trajectory.metadata["avg_reward"] = np.mean(
                    self.current_trajectory.rewards
                )

            if self.current_trajectory.losses:
                self.current_trajectory.metadata["avg_loss"] = np.mean(
                    self.current_trajectory.losses
                )
                self.current_trajectory.metadata["final_loss"] = (
                    self.current_trajectory.losses[-1]
                )

            self.current_trajectory.metadata["duration"] = (
                self.current_trajectory.end_time - self.current_trajectory.start_time
            )

            if save:
                # FIXED: Use Path object and ensure parent exists
                save_path = self.base_path / f"{trajectory_id}.pkl"
                save_path.parent.mkdir(parents=True, exist_ok=True)

                try:
                    # FIXED: Use Path.open() for consistency
                    with save_path.open("wb") as f:
                        pickle.dump(asdict(self.current_trajectory), f)

                    self.trajectory_storage[trajectory_id] = str(save_path)
                    self.stats["total_trajectories"] += 1

                    logger.info(f"Saved trajectory: {trajectory_id}")
                except Exception as e:
                    logger.error(f"Failed to save trajectory: {e}")

            self.current_trajectory = None
            return trajectory_id

    def get_trajectory(self, trajectory_id: str) -> Optional[LearningTrajectory]:
        """Load a saved trajectory"""
        with self._lock:
            if trajectory_id not in self.trajectory_storage:
                return None

            trajectory_path = self.trajectory_storage[trajectory_id]

        try:
            # FIXED: Use Path object and pass string path to safe_pickle_load
            path = Path(trajectory_path).resolve()
            data = safe_pickle_load(str(path))
            return LearningTrajectory(**data)
        except Exception as e:
            logger.error(f"Failed to load trajectory: {e}")
            return None

    def list_trajectories(self) -> List[str]:
        """List all available trajectory IDs"""
        with self._lock:
            return list(self.trajectory_storage.keys())

    def analyze_trajectory(self, trajectory_id: str) -> Optional[Dict[str, Any]]:
        """Analyze a trajectory and return statistics"""
        trajectory = self.get_trajectory(trajectory_id)
        if not trajectory:
            return None

        analysis = {
            "trajectory_id": trajectory_id,
            "duration": trajectory.end_time - trajectory.start_time
            if trajectory.end_time
            else None,
            "num_steps": len(trajectory.states),
            "num_checkpoints": len(trajectory.parameter_snapshots),
        }

        if trajectory.rewards:
            analysis["reward_stats"] = {
                "total": sum(trajectory.rewards),
                "mean": np.mean(trajectory.rewards),
                "std": np.std(trajectory.rewards),
                "min": min(trajectory.rewards),
                "max": max(trajectory.rewards),
            }

        if trajectory.losses:
            analysis["loss_stats"] = {
                "mean": np.mean(trajectory.losses),
                "std": np.std(trajectory.losses),
                "min": min(trajectory.losses),
                "max": max(trajectory.losses),
                "initial": trajectory.losses[0],
                "final": trajectory.losses[-1],
                "improvement": trajectory.losses[0] - trajectory.losses[-1],
            }

        return analysis

    def cleanup_old_trajectories(self, days: int = 30):
        """Clean up trajectories older than specified days"""
        current_time = time.time()
        cutoff_time = current_time - (days * 24 * 3600)

        with self._lock:
            trajectories_to_remove = []
            for trajectory_id in list(self.trajectory_storage.keys()):
                trajectory = self.get_trajectory(trajectory_id)
                if trajectory and trajectory.start_time < cutoff_time:
                    trajectories_to_remove.append(trajectory_id)

            for trajectory_id in trajectories_to_remove:
                # FIXED: Use Path object
                path = Path(self.trajectory_storage[trajectory_id]).resolve()
                if path.exists():
                    try:
                        path.unlink()
                    except Exception as e:
                        logger.error(f"Failed to delete trajectory file: {e}")
                        continue
                del self.trajectory_storage[trajectory_id]
                logger.info(f"Cleaned up old trajectory: {trajectory_id}")

        return len(trajectories_to_remove)

    def export_checkpoint_history(self, export_path: str):
        """Export checkpoint history to JSON file"""
        with self._lock:
            export_data = {
                "checkpoints": list(self.parameter_history),
                "stats": self.stats,
                "export_time": datetime.now().isoformat(),
            }

        # FIXED: Use Path object and ensure parent exists
        path = Path(export_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # FIXED: Use Path.open() for consistency
            with path.open("w") as f:
                json.dump(export_data, f, indent=2, default=str)
            logger.info(f"Exported checkpoint history to: {export_path}")
        except Exception as e:
            logger.error(f"Failed to export checkpoint history: {e}")
            raise

    def import_checkpoint_history(self, import_path: str):
        """Import checkpoint history from JSON file"""
        # FIXED: Use Path object and resolve
        path = Path(import_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Import file not found: {import_path}")

        # FIXED: Use Path.open() for consistency
        with path.open("r") as f:
            import_data = json.load(f)

        with self._lock:
            # Merge with existing history
            for checkpoint in import_data.get("checkpoints", []):
                # Check if checkpoint already exists
                existing = any(
                    c["checkpoint_id"] == checkpoint["checkpoint_id"]
                    for c in self.parameter_history
                )
                if not existing:
                    self.parameter_history.append(checkpoint)

        logger.info(f"Imported checkpoint history from: {import_path}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get usage statistics"""
        with self._lock:
            stats = self.stats.copy()
            stats["current_checkpoints"] = len(self.parameter_history)
            stats["current_trajectories"] = len(self.trajectory_storage)
            stats["queue_size"] = self.checkpoint_queue.qsize()

            if self.parameter_history:
                total_size = sum(c.get("size_bytes", 0) for c in self.parameter_history)
                stats["total_checkpoint_size_mb"] = total_size / (1024 * 1024)

            return stats

    def shutdown(self):
        """FIXED: Clean shutdown of background threads and save state"""
        logger.info("Shutting down ParameterHistoryManager...")

        # Signal shutdown
        self._shutdown.set()
        self._running = False

        # Wait for checkpoint thread to finish
        if hasattr(self, "checkpoint_thread") and self.checkpoint_thread is not None:
            if self.checkpoint_thread.is_alive():
                logger.info("Waiting for checkpoint thread to complete...")
                self.checkpoint_thread.join(timeout=5)
                if self.checkpoint_thread.is_alive():
                    logger.warning("Checkpoint thread did not terminate in time")

        # FIXED: Drain queue with proper timeout and discard pending items
        logger.info("Draining checkpoint queue...")
        drained = 0
        timeout_time = time.time() + 2
        while not self.checkpoint_queue.empty() and time.time() < timeout_time:
            try:
                self.checkpoint_queue.get_nowait()
                drained += 1
            except QueueEmpty:
                break

        if drained > 0:
            logger.warning(f"Discarded {drained} pending checkpoints during shutdown")

        # Save checkpoint history
        try:
            self._save_checkpoint_history()
        except Exception as e:
            logger.error(f"Failed to save checkpoint history during shutdown: {e}")

        logger.info("ParameterHistoryManager shutdown complete")

    def _cleanup_old_checkpoints(self):
        """Clean up old checkpoints to save space (lock held by caller)"""
        while len(self.parameter_history) > self.config.max_checkpoints:
            old_checkpoint = self.parameter_history.popleft()
            # FIXED: Use Path object
            old_path = Path(old_checkpoint["path"]).resolve()
            if old_path.exists():
                try:
                    old_path.unlink()
                    logger.debug(
                        f"Deleted old checkpoint: {old_checkpoint['checkpoint_id']}"
                    )
                except Exception as e:
                    logger.error(f"Failed to delete old checkpoint: {e}")

    def _save_checkpoint_history(self):
        """
        Save checkpoint history to disk for persistence

        FIXED: Use Path objects exclusively for cross-platform compatibility
        """
        # FIXED: Use Path object with proper operator
        checkpoint_dir = self.base_path
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        history_file = checkpoint_dir / "checkpoint_history.json"

        try:
            with self._lock:
                history_data = list(self.parameter_history)

            # FIXED: Use Path.open() for consistency
            with history_file.open("w") as f:
                json.dump(history_data, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Failed to save checkpoint history: {e}")

    def _load_checkpoint_history(self):
        """
        Load checkpoint history from disk if available

        FIXED: Use Path objects exclusively for cross-platform compatibility
        """
        # FIXED: Use Path object with proper operator
        checkpoint_dir = self.base_path
        history_file = checkpoint_dir / "checkpoint_history.json"

        if history_file.exists():
            try:
                # FIXED: Use Path.open() for consistency
                with history_file.open("r") as f:
                    history = json.load(f)

                with self._lock:
                    self.parameter_history = deque(
                        history, maxlen=self.config.max_checkpoints
                    )

                logger.info(
                    f"Loaded {len(self.parameter_history)} checkpoints from history"
                )
            except Exception as e:
                logger.error(f"Failed to load checkpoint history: {e}")

    def __del__(self):
        """Destructor to ensure cleanup"""
        if hasattr(self, "_running") and self._running:
            try:
                self.shutdown()
            except Exception as e:
                pass  # Suppress errors in destructor
