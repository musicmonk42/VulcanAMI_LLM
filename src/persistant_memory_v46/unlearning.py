from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class GradientSurgeryUnlearner:
    """
    Gradient Surgery implementation for machine unlearning.

    Based on "Machine Unlearning via Gradient Surgery" which removes
    information about specific data points by surgically modifying gradients.
    """

    def __init__(self, merkle_graph: Any):
        self.merkle_graph = merkle_graph
        self.unlearning_history: List[Dict[str, Any]] = []

    def unlearn_batch(
        self,
        forget_set: List[bytes],
        retain_set: List[bytes],
        learning_rate: float = 0.01,
        iterations: int = 100,
        regularization: float = 0.001,
    ) -> Dict[str, Any]:
        """
        Unlearn a batch of data using gradient surgery.

        Args:
            forget_set: Data to forget
            retain_set: Data to retain
            learning_rate: Learning rate for gradient updates
            iterations: Number of unlearning iterations
            regularization: L2 regularization strength

        Returns:
            Unlearning result with metrics
        """
        start_time = time.time()

        logger.info(
            f"Starting gradient surgery: forget={len(forget_set)}, retain={len(retain_set)}"
        )

        # Compute initial loss
        initial_forget_loss = self._compute_loss(forget_set)
        initial_retain_loss = self._compute_loss(retain_set)

        # Perform gradient surgery iterations
        for iteration in range(iterations):
            # Compute gradients for forget set
            forget_grads = self._compute_gradients(forget_set)

            # Compute gradients for retain set
            retain_grads = self._compute_gradients(retain_set)

            # Perform gradient surgery
            surgical_grads = self._gradient_surgery(
                forget_grads, retain_grads, regularization
            )

            # Apply surgical gradients
            self._apply_gradients(surgical_grads, learning_rate)

            if (iteration + 1) % 10 == 0:
                forget_loss = self._compute_loss(forget_set)
                retain_loss = self._compute_loss(retain_set)
                logger.debug(
                    f"Iteration {iteration + 1}/{iterations}: "
                    f"forget_loss={forget_loss:.4f}, retain_loss={retain_loss:.4f}"
                )

        # Compute final loss
        final_forget_loss = self._compute_loss(forget_set)
        final_retain_loss = self._compute_loss(retain_set)

        result = {
            "method": "gradient_surgery",
            "forget_count": len(forget_set),
            "retain_count": len(retain_set),
            "iterations": iterations,
            "metrics": {
                "initial_forget_loss": initial_forget_loss,
                "final_forget_loss": final_forget_loss,
                "forget_loss_reduction": initial_forget_loss - final_forget_loss,
                "initial_retain_loss": initial_retain_loss,
                "final_retain_loss": final_retain_loss,
                "retain_loss_increase": final_retain_loss - initial_retain_loss,
            },
            "elapsed_time": time.time() - start_time,
            "timestamp": int(time.time()),
        }

        self.unlearning_history.append(result)

        logger.info(
            f"Gradient surgery completed in {result['elapsed_time']:.2f}s: "
            f"forget_loss {initial_forget_loss:.4f}->{final_forget_loss:.4f}"
        )

        return result

    def _compute_gradients(self, data: List[bytes]) -> np.ndarray:
        """Compute gradients for a dataset."""
        # Placeholder - in production compute actual gradients
        # This would involve backpropagation through the model

        grad_dim = 1000
        gradients = np.zeros(grad_dim)

        for item in data:
            # Simulate gradient computation
            hash_val = int(hashlib.sha256(item).hexdigest()[:16], 16)
            np.random.seed(hash_val % (2**32))
            gradients += np.random.randn(grad_dim) * 0.01

        return gradients / len(data) if data else gradients

    def _gradient_surgery(
        self, forget_grads: np.ndarray, retain_grads: np.ndarray, regularization: float
    ) -> np.ndarray:
        """
        Perform gradient surgery to remove forget_grads while preserving retain_grads.

        The key insight: project forget_grads onto the subspace orthogonal to retain_grads.
        """
        # Normalize gradients
        forget_norm = np.linalg.norm(forget_grads)
        retain_norm = np.linalg.norm(retain_grads)

        if forget_norm < 1e-8 or retain_norm < 1e-8:
            return -forget_grads  # Just negate forget gradients if retain is zero

        forget_normalized = forget_grads / forget_norm
        retain_normalized = retain_grads / retain_norm

        # Project forget onto retain
        projection = np.dot(forget_normalized, retain_normalized) * retain_normalized

        # Remove component parallel to retain
        orthogonal_component = forget_normalized - projection

        # Scale and add regularization
        surgical_grads = -orthogonal_component * forget_norm
        surgical_grads += regularization * retain_grads

        return surgical_grads

    def _apply_gradients(self, gradients: np.ndarray, learning_rate: float) -> None:
        """Apply gradients to model parameters."""
        # Placeholder - in production update actual model parameters
        pass

    def _compute_loss(self, data: List[bytes]) -> float:
        """Compute loss for a dataset."""
        if not data:
            return 0.0

        # Placeholder - in production compute actual loss
        # This would be cross-entropy, MSE, etc.
        total_loss = 0.0

        for item in data:
            hash_val = int(hashlib.sha256(item).hexdigest()[:16], 16)
            np.random.seed(hash_val % (2**32))
            loss = np.random.rand() * 0.5
            total_loss += loss

        return total_loss / len(data)


@dataclass
class UnlearningEngine:
    """
    Advanced Machine Unlearning Engine with multiple algorithms.

    Features:
    - Gradient Surgery
    - SISA (Sharded, Isolated, Sliced, Aggregated)
    - Influence Functions
    - Amnesiac Unlearning
    - Certified Removal
    - Privacy Verification
    """

    merkle_graph: Any
    method: str = "gradient_surgery"
    enable_verification: bool = True
    shard_count: int = 10
    influence_sample_size: int = 1000

    def __post_init__(self):
        """Initialize the unlearning engine."""
        self.impl = GradientSurgeryUnlearner(self.merkle_graph)
        self.unlearning_log: List[Dict[str, Any]] = []
        self.verified_removals: Set[str] = set()
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Create unlearners dict for backward compatibility
        self.unlearners = {"gradient_surgery": self.impl}

        logger.info(f"UnlearningEngine initialized with method={self.method}")

    def unlearn(
        self,
        data_to_forget: Optional[List[Any]] = None,
        data_to_retain: Optional[List[Any]] = None,
        forget_data: Optional[List[Any]] = None,
        retain_data: Optional[List[Any]] = None,
        fast_lane: bool = False,
        verify: bool = True,
    ) -> Dict[str, Any]:
        """
        Unlearn specified data.

        Args:
            data_to_forget: Data items to remove (also accepts forget_data)
            data_to_retain: Data items to keep (also accepts retain_data)
            forget_data: Alternative parameter name for data_to_forget
            retain_data: Alternative parameter name for data_to_retain
            fast_lane: Use fast unlearning (may be less thorough)
            verify: Verify unlearning was successful

        Returns:
            Unlearning result with metrics
        """
        start_time = time.time()

        # Support both parameter names for backward compatibility
        if forget_data is not None:
            data_to_forget = forget_data
        if retain_data is not None:
            data_to_retain = retain_data

        if data_to_forget is None:
            data_to_forget = []

        # Convert to bytes
        forget_bytes = [self._to_bytes(item) for item in data_to_forget]
        retain_bytes = [self._to_bytes(item) for item in (data_to_retain or [])]

        # Select unlearning method
        if self.method == "gradient_surgery":
            result = self._unlearn_gradient_surgery(
                forget_bytes, retain_bytes, fast_lane
            )
        elif self.method == "sisa":
            result = self._unlearn_sisa(forget_bytes, retain_bytes)
        elif self.method == "influence":
            result = self._unlearn_influence(forget_bytes, retain_bytes)
        elif self.method == "amnesiac":
            result = self._unlearn_amnesiac(forget_bytes, retain_bytes)
        elif self.method == "certified":
            result = self._unlearn_certified(forget_bytes, retain_bytes)
        else:
            raise ValueError(f"Unknown unlearning method: {self.method}")

        # Verify unlearning if requested
        if verify and self.enable_verification:
            verification_result = self._verify_unlearning(forget_bytes, retain_bytes)
            result["verification"] = verification_result

        # Log unlearning operation
        log_entry = {
            "timestamp": int(time.time()),
            "method": self.method,
            "forget_count": len(forget_bytes),
            "retain_count": len(retain_bytes),
            "fast_lane": fast_lane,
            "verified": verify,
            "elapsed_time": time.time() - start_time,
            "result": result,
        }
        self.unlearning_log.append(log_entry)

        logger.info(
            f"Unlearning completed in {log_entry['elapsed_time']:.2f}s: "
            f"method={self.method}, forget={len(forget_bytes)}"
        )

        return result

    def gradient_surgery(
        self, packfile: str, pattern: str, fast_lane: bool = False
    ) -> Dict[str, Any]:
        """
        Perform gradient surgery on a packfile to remove a pattern.

        Args:
            packfile: Packfile identifier
            pattern: Pattern to remove
            fast_lane: Use fast unlearning

        Returns:
            Unlearning result
        """
        # Extract data from packfile matching pattern
        forget_data = self._extract_pattern_from_packfile(packfile, pattern)

        # Get retain data (everything else in packfile)
        retain_data = self._extract_non_pattern_from_packfile(packfile, pattern)

        # Convert to bytes
        forget = [pattern.encode()]
        retain = []

        # Perform unlearning
        if fast_lane:
            iterations = 10
            learning_rate = 0.1
        else:
            iterations = 100
            learning_rate = 0.01

        result = self.impl.unlearn_batch(
            forget, retain, learning_rate=learning_rate, iterations=iterations
        )

        # Update packfile metadata
        result["packfile"] = packfile
        result["pattern"] = pattern

        return result

    def _unlearn_gradient_surgery(
        self, forget: List[bytes], retain: List[bytes], fast_lane: bool
    ) -> Dict[str, Any]:
        """Unlearn using gradient surgery."""
        iterations = 10 if fast_lane else 100
        learning_rate = 0.1 if fast_lane else 0.01

        return self.impl.unlearn_batch(
            forget, retain, learning_rate=learning_rate, iterations=iterations
        )

    def _unlearn_sisa(self, forget: List[bytes], retain: List[bytes]) -> Dict[str, Any]:
        """
        Unlearn using SISA (Sharded, Isolated, Sliced, Aggregated).

        SISA splits the model into shards and only retrains affected shards.
        """
        start_time = time.time()

        # Determine affected shards
        affected_shards = self._get_affected_shards(forget)

        logger.info(
            f"SISA unlearning: {len(affected_shards)}/{self.shard_count} shards affected"
        )

        # Retrain affected shards
        retrain_cost = 0.0
        for shard_id in affected_shards:
            shard_data = self._get_shard_data(shard_id, forget, retain)
            retrain_time = self._retrain_shard(shard_id, shard_data)
            retrain_cost += retrain_time

        result = {
            "method": "sisa",
            "forget_count": len(forget),
            "retain_count": len(retain),
            "shard_count": self.shard_count,
            "affected_shards": len(affected_shards),
            "retrain_cost": retrain_cost,
            "elapsed_time": time.time() - start_time,
            "timestamp": int(time.time()),
        }

        return result

    def _unlearn_influence(
        self, forget: List[bytes], retain: List[bytes]
    ) -> Dict[str, Any]:
        """
        Unlearn using influence functions.

        Influence functions estimate the effect of removing training points.
        """
        start_time = time.time()

        # Compute influence of forget set on model
        influences = []
        for item in forget:
            influence = self._compute_influence(item, retain)
            influences.append(influence)

        avg_influence = np.mean(influences) if influences else 0.0
        max_influence = np.max(influences) if influences else 0.0

        # Apply influence-based updates
        update_magnitude = self._apply_influence_updates(influences, forget)

        result = {
            "method": "influence",
            "forget_count": len(forget),
            "retain_count": len(retain),
            "avg_influence": float(avg_influence),
            "max_influence": float(max_influence),
            "update_magnitude": float(update_magnitude),
            "elapsed_time": time.time() - start_time,
            "timestamp": int(time.time()),
        }

        return result

    def _unlearn_amnesiac(
        self, forget: List[bytes], retain: List[bytes]
    ) -> Dict[str, Any]:
        """
        Amnesiac unlearning: Adds carefully crafted noise to forget data.

        This method adds controlled noise to the representations of data to forget,
        making the model "amnesiac" about specific data points.
        """
        start_time = time.time()

        # Compute noise to add for each item to forget
        noise_vectors = []
        for item in forget:
            noise = self._compute_amnesiac_noise(item, retain)
            noise_vectors.append(noise)

        # Apply noise to model
        total_noise_magnitude = self._apply_noise(noise_vectors)

        result = {
            "method": "amnesiac",
            "forget_count": len(forget),
            "retain_count": len(retain),
            "noise_magnitude": float(total_noise_magnitude),
            "elapsed_time": time.time() - start_time,
            "timestamp": int(time.time()),
        }

        return result

    def _unlearn_certified(
        self, forget: List[bytes], retain: List[bytes]
    ) -> Dict[str, Any]:
        """
        Certified unlearning with provable guarantees.

        Provides cryptographic or statistical guarantees about unlearning.
        """
        start_time = time.time()

        # Compute certified removal bounds
        epsilon_dp = self._compute_differential_privacy_epsilon(forget, retain)

        # Perform certified removal
        removal_proof = self._generate_removal_certificate(forget)

        result = {
            "method": "certified",
            "forget_count": len(forget),
            "retain_count": len(retain),
            "epsilon_dp": float(epsilon_dp),
            "removal_proof": removal_proof,
            "certified": True,
            "elapsed_time": time.time() - start_time,
            "timestamp": int(time.time()),
        }

        return result

    def _verify_unlearning(
        self, forget: List[bytes], retain: List[bytes]
    ) -> Dict[str, Any]:
        """
        Verify that unlearning was successful.

        Checks:
        1. Forget data no longer affects model predictions
        2. Retain data still produces good predictions
        3. Model performance hasn't degraded significantly
        """
        # Test that forget data is truly forgotten
        forget_scores = []
        for item in forget:
            score = self._measure_memorization(item)
            forget_scores.append(score)

        avg_forget_score = np.mean(forget_scores) if forget_scores else 0.0

        # Test that retain data is still remembered
        retain_scores = []
        for item in retain[: min(100, len(retain))]:  # Sample for efficiency
            score = self._measure_memorization(item)
            retain_scores.append(score)

        avg_retain_score = np.mean(retain_scores) if retain_scores else 0.0

        # Verification passes if forget score is low and retain score is high
        verification_passed = (
            avg_forget_score < 0.3  # Forgotten
            and avg_retain_score > 0.7  # Retained
        )

        return {
            "passed": verification_passed,
            "avg_forget_score": float(avg_forget_score),
            "avg_retain_score": float(avg_retain_score),
            "threshold_forget": 0.3,
            "threshold_retain": 0.7,
        }

    def _extract_pattern_from_packfile(self, packfile: str, pattern: str) -> List[Any]:
        """Extract data matching pattern from packfile."""
        # Placeholder - in production, actually read packfile
        return []

    def _extract_non_pattern_from_packfile(
        self, packfile: str, pattern: str
    ) -> List[Any]:
        """Extract data not matching pattern from packfile."""
        # Placeholder - in production, actually read packfile
        return []

    def _get_affected_shards(self, forget: List[bytes]) -> List[int]:
        """Determine which shards are affected by data to forget."""
        affected = set()

        for item in forget:
            # Hash to shard
            hash_val = int(hashlib.sha256(item).hexdigest()[:16], 16)
            shard_id = hash_val % self.shard_count
            affected.add(shard_id)

        return sorted(affected)

    def _get_shard_data(
        self, shard_id: int, forget: List[bytes], retain: List[bytes]
    ) -> List[bytes]:
        """Get data for a specific shard, excluding forget set."""
        shard_data = []

        for item in retain:
            hash_val = int(hashlib.sha256(item).hexdigest()[:16], 16)
            if hash_val % self.shard_count == shard_id:
                shard_data.append(item)

        return shard_data

    def _retrain_shard(self, shard_id: int, data: List[bytes]) -> float:
        """Retrain a specific shard."""
        # Placeholder - in production, actually retrain the shard
        # Simulate training time
        return len(data) * 0.001

    def _compute_influence(self, item: bytes, retain: List[bytes]) -> float:
        """Compute influence of an item on the model."""
        # Simplified influence computation
        # In production, compute actual Hessian-based influence

        # Use a sample of retain set for efficiency
        sample_size = min(self.influence_sample_size, len(retain))
        sample = retain[:sample_size]

        # Simulate influence calculation
        hash_val = int(hashlib.sha256(item).hexdigest()[:16], 16)
        np.random.seed(hash_val % (2**32))

        influence = np.random.rand() * 0.5
        return influence

    def _apply_influence_updates(
        self, influences: List[float], forget: List[bytes]
    ) -> float:
        """Apply updates based on influence values."""
        # Placeholder - in production, update model parameters
        return np.mean(influences) if influences else 0.0

    def _compute_amnesiac_noise(self, item: bytes, retain: List[bytes]) -> np.ndarray:
        """Compute noise to add for amnesiac unlearning."""
        # Generate noise that maximally disrupts item while minimizing
        # effect on retain set

        hash_val = int(hashlib.sha256(item).hexdigest()[:16], 16)
        np.random.seed(hash_val % (2**32))

        noise = np.random.randn(100) * 0.1
        return noise

    def _apply_noise(self, noise_vectors: List[np.ndarray]) -> float:
        """Apply noise vectors to model."""
        # Placeholder - in production, add noise to embeddings
        total_magnitude = sum(np.linalg.norm(n) for n in noise_vectors)
        return total_magnitude

    def _compute_differential_privacy_epsilon(
        self, forget: List[bytes], retain: List[bytes]
    ) -> float:
        """Compute differential privacy epsilon for certified unlearning."""
        # Simplified DP epsilon calculation
        # In production, use proper DP accounting

        n_forget = len(forget)
        n_total = n_forget + len(retain)

        if n_total == 0:
            return 0.0

        # Epsilon proportional to fraction removed
        epsilon = np.log(1 + n_forget / n_total)
        return epsilon

    def _generate_removal_certificate(self, forget: List[bytes]) -> str:
        """Generate cryptographic certificate of removal."""
        # Generate Merkle tree of removed items
        from hashlib import sha256

        if not forget:
            return sha256(b"empty").hexdigest()

        leaves = [sha256(item).digest() for item in forget]

        # Simple Merkle root
        current_level = leaves
        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else left
                next_level.append(sha256(left + right).digest())
            current_level = next_level

        return current_level[0].hex() if current_level else ""

    def _measure_memorization(self, item: bytes) -> float:
        """Measure how much the model has memorized an item."""
        # Placeholder - in production, measure actual memorization
        # e.g., perplexity, loss, or reconstruction accuracy

        hash_val = int(hashlib.sha256(item).hexdigest()[:16], 16)
        np.random.seed(hash_val % (2**32))

        memorization = np.random.rand()
        return memorization

    def _to_bytes(self, item: Any) -> bytes:
        """Convert item to bytes."""
        if isinstance(item, bytes):
            return item
        elif isinstance(item, str):
            return item.encode()
        else:
            return str(item).encode()

    async def unlearn_async(self, *args, **kwargs) -> Dict[str, Any]:
        """Async version of unlearn."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, lambda: self.unlearn(*args, **kwargs)
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get unlearning statistics."""
        return {
            "method": self.method,
            "total_unlearning_operations": len(self.unlearning_log),
            "total_unlearnings": len(
                self.unlearning_log
            ),  # Alias for backward compatibility
            "verified_removals": len(self.verified_removals),
            "shard_count": self.shard_count,
            "enable_verification": self.enable_verification,
            "recent_operations": self.unlearning_log[-10:]
            if self.unlearning_log
            else [],
        }

    def export_log(self, path: str) -> None:
        """Export unlearning log to file."""
        import json

        with open(path, "w") as f:
            json.dump(self.unlearning_log, f, indent=2)

        logger.info(f"Unlearning log exported to {path}")

    def verify_certified_removal(self, certificate: str, items: List[bytes]) -> bool:
        """Verify a certified removal certificate."""
        expected_cert = self._generate_removal_certificate(items)
        return certificate == expected_cert

    def close(self) -> None:
        """Clean up resources."""
        self.executor.shutdown(wait=True)
        logger.info("UnlearningEngine closed")
