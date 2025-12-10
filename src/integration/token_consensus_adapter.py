from __future__ import annotations

import asyncio
import inspect
import logging
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Callable, Union

log = logging.getLogger(__name__)


@dataclass
class ConsensusAdapterConfig:
    """Configuration for the TokenConsensusAdapter."""

    fail_closed: bool = False  # If True, fail closed when consensus is unavailable
    timeout_seconds: float = 2.0  # Timeout for consensus calls
    max_retries: int = 3  # Maximum retry attempts
    enable_observability: bool = True  # Enable observability hooks


@dataclass
class ConsensusProposal:
    """Typed proposal structure."""

    type: str
    token: str
    position: int
    chosen_index: Optional[int] = None
    timestamp: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class TokenConsensusAdapter:
    """
    Asynchronous adapter for integrating consensus engines into token emission workflow.

    Provides async approve() method that:
    - Validates proposals for required fields and correct types
    - Handles both sync and async consensus engines
    - Implements retry logic with exponential backoff
    - Provides fail-open or fail-closed behavior
    - Tracks metrics and emits rich observability events
    """

    def __init__(
        self,
        engine: Optional[Any] = None,
        config: Optional[ConsensusAdapterConfig] = None,
        observability: Optional[
            Callable[[str, Dict[str, Any]], Union[None, Any]]
        ] = None,
    ):
        self.engine = engine
        self.config = config or ConsensusAdapterConfig()

        # Adapter configuration shortcuts
        self.fail_closed = self.config.fail_closed
        self.timeout = self.config.timeout_seconds
        self.max_retries = self.config.max_retries

        # Metrics tracking
        self._calls = 0
        self._successes = 0

        # FIX: Implement safe observability wrapping
        if observability:
            if inspect.iscoroutinefunction(observability):
                # Wrap async observability call in a fire-and-forget task
                self.obs = lambda e, p: asyncio.create_task(observability(e, p))
            else:
                # Use synchronous observability directly
                self.obs = observability
        else:
            self.obs = self._noop_obs

    async def approve(self, proposal: Dict[str, Any]) -> bool:
        """
        Approve a token emission proposal asynchronously.
        Returns True if approved or fallback (True for fail-open, False for fail-closed).
        Includes retry logic with exponential backoff and metrics tracking.
        """
        fallback_result = not self.fail_closed

        start_time = time.time()
        attempt_times = []

        try:
            validated = self._validate_proposal(proposal)
        except (ValueError, TypeError) as e:
            self.obs(
                "consensus.validation_error", {"error": str(e), "proposal": proposal}
            )
            if not fallback_result:
                log.warning(f"Validation failed for proposal (Fail-Closed): {e}")

            self._emit_final_obs(
                approved=fallback_result,
                proposal=proposal,
                attempt=0,
                start_time=start_time,
                attempt_times=[],
                error_type="ValidationFailed",
            )
            # ENHANCEMENT: Update global metrics on failure
            self._calls += 1
            return fallback_result

        proposal_dict = asdict(validated)
        approved = False

        for attempt in range(self.max_retries + 1):
            attempt_start_time = time.time()
            try:
                approved = await self._attempt_approval(proposal_dict, validated)

                attempt_times.append((time.time() - attempt_start_time) * 1000)

                # Success:
                self._calls += 1
                if approved:
                    self._successes += 1

                self._emit_final_obs(
                    approved=approved,
                    proposal=proposal_dict,
                    attempt=attempt,
                    start_time=start_time,
                    attempt_times=attempt_times,
                    error_type=None,
                )
                return approved

            except asyncio.TimeoutError:
                attempt_times.append((time.time() - attempt_start_time) * 1000)

                if attempt == self.max_retries:
                    self.obs(
                        "consensus.timeout_final",
                        {"proposal": proposal_dict, "max_retries": self.max_retries},
                    )
                    if not fallback_result:
                        log.error(
                            "Consensus failed after retries: Timeout"
                        )  # ENHANCEMENT: Error log severity

                    self._calls += 1  # Call counted on final attempt/failure

                    self._emit_final_obs(
                        approved=fallback_result,
                        proposal=proposal_dict,
                        attempt=attempt,
                        start_time=start_time,
                        attempt_times=attempt_times,
                        error_type="TimeoutFinal",
                    )
                    return fallback_result

                backoff_sleep = 0.1 * (2**attempt)
                self.obs(
                    "consensus.timeout_retry",
                    {"attempt": attempt, "backoff_sleep": backoff_sleep},
                )
                await asyncio.sleep(backoff_sleep)

            except Exception as e:
                attempt_times.append((time.time() - attempt_start_time) * 1000)

                self.obs(
                    "consensus.engine_error",
                    {
                        "error": str(e),
                        "attempt": attempt,
                        "proposal_type": validated.type,
                    },
                )
                if attempt == self.max_retries:
                    if not fallback_result:
                        log.error(
                            f"Consensus failed after retries: Engine Error: {e}"
                        )  # ENHANCEMENT: Error log severity

                    self._calls += 1  # Call counted on final attempt/failure

                    self._emit_final_obs(
                        approved=fallback_result,
                        proposal=proposal_dict,
                        attempt=attempt,
                        start_time=start_time,
                        attempt_times=attempt_times,
                        error_type="EngineErrorFinal",
                    )
                    return fallback_result

                # ENHANCEMENT: Optimized backoff for engine errors
                engine_backoff_sleep = 0.05 * (2**attempt)
                self.obs(
                    "consensus.engine_error_retry",
                    {"attempt": attempt, "backoff_sleep": engine_backoff_sleep},
                )
                await asyncio.sleep(engine_backoff_sleep)

        # Should only be reached if retries loop completes unexpectedly
        if not fallback_result:
            log.error("Consensus failed: Unexpected exit from retry loop")

        self._calls += 1

        self._emit_final_obs(
            approved=fallback_result,
            proposal=proposal_dict,
            attempt=self.max_retries + 1,
            start_time=start_time,
            attempt_times=attempt_times,
            error_type="UnexpectedExit",
        )
        return fallback_result

    def _emit_final_obs(
        self,
        approved: bool,
        proposal: Dict[str, Any],
        attempt: int,
        start_time: float,
        attempt_times: List[float],
        error_type: Optional[str],
    ):
        """Helper to emit the final, rich observability event."""
        total_time_ms = (time.time() - start_time) * 1000
        avg_attempt_time_ms = (
            sum(attempt_times) / len(attempt_times) if attempt_times else 0
        )

        # ENHANCEMENT: success_rate, retry_count metrics
        success_rate = self._successes / self._calls if self._calls > 0 else 0.0
        retry_count = attempt

        self.obs(
            "consensus.approve",
            {
                "approved": approved,
                "token": str(proposal.get("token")),
                "position": proposal.get("position"),
                "engine": type(self.engine).__name__ if self.engine else "None",
                "attempt_count": attempt + 1,
                "total_time_ms": total_time_ms,
                "avg_attempt_time_ms": avg_attempt_time_ms,
                "fail_closed_policy": self.fail_closed,
                "error_type": error_type,
                "retry_count": retry_count,
                "success_rate_global": success_rate,
                "success_rate_call": 1.0 if approved and error_type is None else 0.0,
            },
        )

    async def _attempt_approval(
        self, proposal_dict: Dict[str, Any], validated_proposal: ConsensusProposal
    ) -> bool:
        """Helper function to contact the engine, normalizing async/sync calls."""
        if not self.engine:
            self.obs(
                "consensus.engine_missing", {"proposal_type": validated_proposal.type}
            )
            # FIXED: Use fail_closed policy instead of always returning True
            return not self.fail_closed

        loop = asyncio.get_running_loop()

        # Try engine.approve(proposal)
        if hasattr(self.engine, "approve"):
            fn = self.engine.approve

            if inspect.iscoroutinefunction(fn):
                return await asyncio.wait_for(fn(proposal_dict), timeout=self.timeout)
            else:
                # Sync method: run in executor
                return await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: fn(proposal_dict)),
                    timeout=self.timeout,
                )

        # Try engine.submit_proposal(proposal)
        if hasattr(self.engine, "submit_proposal"):
            fn = self.engine.submit_proposal

            if inspect.iscoroutinefunction(fn):
                result = await asyncio.wait_for(fn(proposal_dict), timeout=self.timeout)
            else:
                # Sync method: run in executor
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: fn(proposal_dict)),
                    timeout=self.timeout,
                )

            # Check for common result patterns (e.g., dict with 'status' or object with 'status' attribute)
            if isinstance(result, dict) and "status" in result:
                return result.get("status", "").lower() == "approved"
            return getattr(result, "status", "").lower() == "approved"

        # Fallback if engine is present but lacks expected methods
        self.obs(
            "consensus.engine_no_method", {"proposal_type": validated_proposal.type}
        )
        return True

    def _validate_proposal(self, raw: Dict[str, Any]) -> ConsensusProposal:
        """
        Validates raw proposal dict and converts it into a typed ConsensusProposal.
        Raises ValueError for missing fields and TypeError for invalid types.
        """
        required = {"type", "token", "position"}
        if not isinstance(raw, dict):
            raise TypeError("Proposal must be a dictionary.")

        if not required.issubset(raw.keys()):
            missing = required - raw.keys()
            raise ValueError(f"Missing required fields in proposal: {missing}")

        # Basic type checks and ENHANCED checks
        if not raw["type"]:  # ENHANCEMENT: Empty type check
            raise ValueError("Type cannot be empty")
        if not isinstance(raw["type"], str):
            raise TypeError(
                f"Field 'type' must be a string, got {type(raw['type']).__name__}"
            )

        # Token validation - ENHANCED
        token = raw["token"]
        if not isinstance(token, (str, int)):
            raise TypeError(
                f"Field 'token' must be a string or int, got {type(token).__name__}"
            )
        if isinstance(token, str) and not token:
            raise ValueError("Token cannot be empty string")

        # Position validation - ENHANCED
        position = raw["position"]
        if not isinstance(position, int):
            raise TypeError(
                f"Field 'position' must be an int, got {type(position).__name__}"
            )
        if position < 0:
            raise ValueError(f"Position must be non-negative, got {position}")

        # Chosen index validation (optional field) - ENHANCED
        chosen_index = raw.get("chosen_index")
        if chosen_index is not None:
            if not isinstance(chosen_index, int):
                raise TypeError(
                    f"Field 'chosen_index' must be an int if provided, got {type(chosen_index).__name__}"
                )
            if chosen_index < 0:
                raise ValueError(
                    f"chosen_index must be non-negative if provided, got {chosen_index}"
                )

        return ConsensusProposal(
            type=raw["type"],
            token=str(token),  # Normalize to string
            position=position,
            chosen_index=chosen_index,
            timestamp=raw.get("timestamp"),
            metadata=raw.get("metadata"),
        )

    def _noop_obs(self, event: str, payload: Dict[str, Any]) -> None:
        """No-op observability function."""
        pass


# ------------------------ Test and Demo helper functions ------------------------ #


class MockConsensusEngine:
    """Mock consensus engine for testing and demonstration."""

    def __init__(self, approval_rate: float = 0.8, is_async: bool = True):
        self.approval_rate = approval_rate
        self.is_async = is_async
        self.call_count = 0

    async def approve(self, proposal: Dict[str, Any]) -> bool:
        """Async approve method."""
        self.call_count += 1
        await asyncio.sleep(0.01)  # Simulate processing

        # Simple approval logic
        if proposal.get("token") == "bad":
            return False

        # Random approval based on rate
        import random

        return random.random() < self.approval_rate

    def approve_sync(self, proposal: Dict[str, Any]) -> bool:
        """Sync approve method for testing sync engines."""
        self.call_count += 1

        if proposal.get("token") == "bad":
            return False

        import random

        return random.random() < self.approval_rate


async def demo_consensus_adapter():
    """Demonstration of the TokenConsensusAdapter."""
    print("=== TokenConsensusAdapter Demo ===")

    # 1. Demo with mock engine
    engine = MockConsensusEngine(approval_rate=0.7)
    adapter = TokenConsensusAdapter(
        engine=engine,
        config=ConsensusAdapterConfig(fail_closed=False, timeout_seconds=1.0),
    )

    # Test proposals
    proposals = [
        {"type": "token_emission", "token": "hello", "position": 0},
        {"type": "token_emission", "token": "world", "position": 1},
        {"type": "token_emission", "token": "bad", "position": 2},  # Will be rejected
        {"type": "token_emission", "token": "test", "position": 3},
    ]

    for proposal in proposals:
        approved = await adapter.approve(proposal)
        print(
            f"Proposal {proposal['token']}: {'✓ Approved' if approved else '✗ Rejected'}"
        )

    print(f"Engine calls: {engine.call_count}")
    print(f"Adapter successes: {adapter._successes}/{adapter._calls}")

    # 2. Demo fail-closed behavior
    print("\n=== Fail-Closed Demo (no engine) ===")
    fail_closed_adapter = TokenConsensusAdapter(
        engine=None, config=ConsensusAdapterConfig(fail_closed=True)
    )

    test_proposal = {"type": "token_emission", "token": "test", "position": 0}
    result = await fail_closed_adapter.approve(test_proposal)
    print(f"No engine, fail-closed: {'✓ Approved' if result else '✗ Rejected'}")

    # 3. Demo fail-open behavior
    fail_open_adapter = TokenConsensusAdapter(
        engine=None, config=ConsensusAdapterConfig(fail_closed=False)
    )

    result = await fail_open_adapter.approve(test_proposal)
    print(f"No engine, fail-open: {'✓ Approved' if result else '✗ Rejected'}")


if __name__ == "__main__":
    asyncio.run(demo_consensus_adapter())
