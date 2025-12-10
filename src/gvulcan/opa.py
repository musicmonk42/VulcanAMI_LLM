"""
Open Policy Agent (OPA) Integration

This module provides comprehensive OPA integration with support for policy evaluation,
caching, audit logging, and multiple policy bundles.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class WriteBarrierInput:
    """
    Input data for write barrier policy evaluation

    Attributes:
        dqs: Data Quality Score (0-1)
        pii: PII detection results
    """

    dqs: float
    pii: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for policy evaluation"""
        return {"dqs": self.dqs, "pii": self.pii}


@dataclass
class WriteBarrierResult:
    """
    Result of write barrier policy evaluation

    Attributes:
        allow: Whether write is allowed
        quarantine: Whether data should be quarantined
        deny_reason: Optional reason for denial
    """

    allow: bool
    quarantine: bool
    deny_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "allow": self.allow,
            "quarantine": self.quarantine,
            "deny_reason": self.deny_reason,
        }


@dataclass
class PolicyEvaluation:
    """
    Detailed result of a policy evaluation

    Attributes:
        policy_name: Name of the evaluated policy
        input_data: Input data used for evaluation
        result: Evaluation result
        evaluation_time: When evaluation occurred
        cache_hit: Whether result was from cache
        metadata: Additional metadata
    """

    policy_name: str
    input_data: Dict[str, Any]
    result: Any
    evaluation_time: datetime = field(default_factory=datetime.now)
    cache_hit: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage"""
        return {
            "policy_name": self.policy_name,
            "input_data": self.input_data,
            "result": self.result
            if isinstance(self.result, dict)
            else str(self.result),
            "evaluation_time": self.evaluation_time.isoformat(),
            "cache_hit": self.cache_hit,
            "metadata": self.metadata,
        }


class LRUCache:
    """Least Recently Used cache implementation"""

    def __init__(self, capacity: int = 1000):
        """
        Initialize LRU cache.

        Args:
            capacity: Maximum number of items to cache
        """
        self.capacity = capacity
        self.cache: OrderedDict = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, key: str, value: Any) -> None:
        """Put value in cache"""
        if key in self.cache:
            # Update and move to end
            self.cache.move_to_end(key)
        self.cache[key] = value

        # Evict oldest if over capacity
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def clear(self) -> None:
        """Clear the cache"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0

        return {
            "size": len(self.cache),
            "capacity": self.capacity,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
        }


class OPAClient:
    """
    Enhanced OPA client with policy evaluation, caching, and audit logging.

    This class provides:
    - Policy evaluation
    - Result caching for performance
    - Audit logging
    - Multiple policy bundle support
    - Statistical tracking
    """

    def __init__(
        self,
        bundle_version: str,
        enable_cache: bool = True,
        cache_size: int = 1000,
        enable_audit: bool = True,
    ):
        """
        Initialize OPA client.

        Args:
            bundle_version: OPA policy bundle version
            enable_cache: Whether to enable result caching
            cache_size: Maximum cache size
            enable_audit: Whether to enable audit logging
        """
        self.bundle_version = bundle_version
        self.enable_cache = enable_cache
        self.enable_audit = enable_audit

        # Initialize cache
        self.cache = LRUCache(capacity=cache_size) if enable_cache else None

        # Audit log
        self.audit_log: List[PolicyEvaluation] = []

        # Statistics
        self.evaluations_count = 0
        self.policy_stats: Dict[str, int] = {}

        logger.info(
            f"Initialized OPA Client with bundle version {bundle_version}, "
            f"cache={'enabled' if enable_cache else 'disabled'}, "
            f"audit={'enabled' if enable_audit else 'disabled'}"
        )

    def _compute_cache_key(self, policy_name: str, input_data: Dict[str, Any]) -> str:
        """
        Compute cache key for policy evaluation.

        Args:
            policy_name: Name of policy
            input_data: Input data

        Returns:
            Cache key string
        """
        # Create deterministic hash of policy + input
        data_str = json.dumps(
            {"policy": policy_name, "input": input_data}, sort_keys=True
        )

        return hashlib.sha256(data_str.encode()).hexdigest()

    def evaluate_write_barrier(self, data: WriteBarrierInput) -> WriteBarrierResult:
        """
        Evaluate write barrier policy.

        This is a simplified policy that gates based on DQS thresholds.
        In production, this would make an HTTP request to OPA.

        Args:
            data: Write barrier input data

        Returns:
            WriteBarrierResult with decision
        """
        policy_name = "write_barrier"
        input_dict = data.to_dict()

        # Check cache
        cache_hit = False
        if self.enable_cache:
            cache_key = self._compute_cache_key(policy_name, input_dict)
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                cache_hit = True
                result = cached_result
                logger.debug(f"Cache hit for {policy_name}")
            else:
                result = self._evaluate_write_barrier_impl(data)
                self.cache.put(cache_key, result)
        else:
            result = self._evaluate_write_barrier_impl(data)

        # Update statistics
        self.evaluations_count += 1
        self.policy_stats[policy_name] = self.policy_stats.get(policy_name, 0) + 1

        # Audit logging
        if self.enable_audit:
            evaluation = PolicyEvaluation(
                policy_name=policy_name,
                input_data=input_dict,
                result=result.to_dict(),
                cache_hit=cache_hit,
            )
            self.audit_log.append(evaluation)

        return result

    def _evaluate_write_barrier_impl(
        self, data: WriteBarrierInput
    ) -> WriteBarrierResult:
        """
        Internal implementation of write barrier policy.

        Policy rules:
        - DQS < 0.30: Reject
        - 0.30 <= DQS < 0.40: Quarantine
        - DQS >= 0.40: Allow
        """
        if data.dqs < 0.30:
            return WriteBarrierResult(
                allow=False, quarantine=False, deny_reason="DQS below reject threshold"
            )

        if 0.30 <= data.dqs < 0.40:
            return WriteBarrierResult(allow=False, quarantine=True, deny_reason=None)

        return WriteBarrierResult(allow=True, quarantine=False, deny_reason=None)

    def evaluate_policy(
        self, policy_name: str, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate a custom policy.

        This is a generic policy evaluation interface that can be extended
        to support various policy types.

        Args:
            policy_name: Name of policy to evaluate
            input_data: Input data for policy

        Returns:
            Policy evaluation result
        """
        # Check cache
        cache_hit = False
        if self.enable_cache:
            cache_key = self._compute_cache_key(policy_name, input_data)
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                cache_hit = True
                result = cached_result
            else:
                result = self._evaluate_policy_impl(policy_name, input_data)
                self.cache.put(cache_key, result)
        else:
            result = self._evaluate_policy_impl(policy_name, input_data)

        # Update statistics
        self.evaluations_count += 1
        self.policy_stats[policy_name] = self.policy_stats.get(policy_name, 0) + 1

        # Audit logging
        if self.enable_audit:
            evaluation = PolicyEvaluation(
                policy_name=policy_name,
                input_data=input_data,
                result=result,
                cache_hit=cache_hit,
            )
            self.audit_log.append(evaluation)

        return result

    def _evaluate_policy_impl(
        self, policy_name: str, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Internal implementation of generic policy evaluation.

        In production, this would make an HTTP request to OPA server.
        """
        # Placeholder implementation
        logger.debug(f"Evaluating policy: {policy_name}")

        return {
            "allowed": True,
            "policy": policy_name,
            "bundle_version": self.bundle_version,
        }

    def evaluate_batch(
        self, policy_name: str, inputs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Evaluate policy for multiple inputs in batch.

        Args:
            policy_name: Name of policy to evaluate
            inputs: List of input data

        Returns:
            List of policy evaluation results
        """
        results = []
        for input_data in inputs:
            result = self.evaluate_policy(policy_name, input_data)
            results.append(result)

        logger.info(f"Batch evaluated {len(results)} inputs for policy {policy_name}")
        return results

    def clear_cache(self) -> None:
        """Clear the policy evaluation cache"""
        if self.cache:
            self.cache.clear()
            logger.info("Cleared policy evaluation cache")

    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get cache statistics"""
        if self.cache:
            return self.cache.get_stats()
        return None

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about policy evaluations.

        Returns:
            Dictionary with statistics
        """
        stats = {
            "bundle_version": self.bundle_version,
            "total_evaluations": self.evaluations_count,
            "policy_counts": self.policy_stats.copy(),
            "audit_log_size": len(self.audit_log),
        }

        if self.cache:
            stats["cache"] = self.get_cache_stats()

        return stats

    def get_audit_log(
        self, policy_name: Optional[str] = None, limit: Optional[int] = None
    ) -> List[PolicyEvaluation]:
        """
        Get audit log entries.

        Args:
            policy_name: Optional filter by policy name
            limit: Optional limit on number of entries

        Returns:
            List of policy evaluations
        """
        log = self.audit_log

        if policy_name:
            log = [e for e in log if e.policy_name == policy_name]

        if limit:
            log = log[-limit:]

        return log

    def export_audit_log(self, path: Path, policy_name: Optional[str] = None) -> None:
        """
        Export audit log to file.

        Args:
            path: Path to export to
            policy_name: Optional filter by policy name
        """
        log = self.get_audit_log(policy_name=policy_name)

        data = {
            "bundle_version": self.bundle_version,
            "export_time": datetime.now().isoformat(),
            "entries_count": len(log),
            "entries": [e.to_dict() for e in log],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported {len(log)} audit log entries to {path}")

    def generate_report(self) -> str:
        """
        Generate a text report of OPA client statistics.

        Returns:
            Formatted report string
        """
        stats = self.get_statistics()

        report = f"""
OPA Client Report
=================

Bundle Version: {stats["bundle_version"]}
Total Evaluations: {stats["total_evaluations"]}
Audit Log Entries: {stats["audit_log_size"]}

Policy Evaluation Counts:
"""

        for policy, count in sorted(
            stats["policy_counts"].items(), key=lambda x: x[1], reverse=True
        ):
            pct = (
                count / stats["total_evaluations"] * 100
                if stats["total_evaluations"] > 0
                else 0
            )
            report += f"  {policy}: {count} ({pct:.1f}%)\n"

        if "cache" in stats and stats["cache"]:
            cache = stats["cache"]
            report += f"""
Cache Statistics:
  Size: {cache["size"]}/{cache["capacity"]}
  Hits: {cache["hits"]}
  Misses: {cache["misses"]}
  Hit Rate: {cache["hit_rate"]:.2%}
"""

        return report


class PolicyRegistry:
    """
    Registry for managing multiple OPA policy bundles.

    This class allows managing multiple policy bundles and routing
    policy evaluations to the appropriate bundle.
    """

    def __init__(self):
        """Initialize policy registry"""
        self.clients: Dict[str, OPAClient] = {}
        self.default_client: Optional[str] = None

        logger.info("Initialized Policy Registry")

    def register_client(
        self, name: str, client: OPAClient, set_default: bool = False
    ) -> None:
        """
        Register an OPA client.

        Args:
            name: Name for the client
            client: OPA client instance
            set_default: Whether to set as default client
        """
        self.clients[name] = client

        if set_default or self.default_client is None:
            self.default_client = name

        logger.info(f"Registered OPA client: {name}")

    def get_client(self, name: Optional[str] = None) -> OPAClient:
        """
        Get an OPA client by name.

        Args:
            name: Client name (uses default if not specified)

        Returns:
            OPA client

        Raises:
            KeyError: If client not found
        """
        if name is None:
            name = self.default_client

        if name not in self.clients:
            raise KeyError(f"OPA client not found: {name}")

        return self.clients[name]

    def evaluate_policy(
        self,
        policy_name: str,
        input_data: Dict[str, Any],
        client_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a policy using specified or default client.

        Args:
            policy_name: Policy to evaluate
            input_data: Input data
            client_name: Optional client name

        Returns:
            Policy evaluation result
        """
        client = self.get_client(client_name)
        return client.evaluate_policy(policy_name, input_data)

    def get_all_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all registered clients.

        Returns:
            Dictionary mapping client names to statistics
        """
        return {name: client.get_statistics() for name, client in self.clients.items()}


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)

    print("=== Testing OPA Client ===")
    client = OPAClient(bundle_version="1.0.0", enable_cache=True)

    # Test write barrier evaluation
    input_data = WriteBarrierInput(dqs=0.45, pii={"detected": False})
    result = client.evaluate_write_barrier(input_data)
    print(
        f"Write Barrier Result: allow={result.allow}, "
        f"quarantine={result.quarantine}, reason={result.deny_reason}"
    )

    # Test with different DQS scores
    for dqs in [0.25, 0.35, 0.50]:
        test_input = WriteBarrierInput(dqs=dqs, pii={})
        test_result = client.evaluate_write_barrier(test_input)
        print(
            f"DQS {dqs:.2f} -> {test_result.allow}, quarantine={test_result.quarantine}"
        )

    # Test cache
    print("\n=== Testing Cache ===")
    # Evaluate same input again (should hit cache)
    result2 = client.evaluate_write_barrier(input_data)
    cache_stats = client.get_cache_stats()
    print(f"Cache stats: {cache_stats}")

    # Generate report
    print("\n=== OPA Client Report ===")
    print(client.generate_report())

    print("\n=== Testing Policy Registry ===")
    registry = PolicyRegistry()
    registry.register_client("production", client, set_default=True)

    # Evaluate through registry
    result = registry.evaluate_policy("custom_policy", {"data": "test"})
    print(f"Registry evaluation result: {result}")

    # Get all statistics
    all_stats = registry.get_all_statistics()
    print(f"\nRegistry statistics: {json.dumps(all_stats, indent=2)}")
