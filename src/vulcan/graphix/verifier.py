"""Private, release-bound epistemic warrant issuance for Phase B."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from fractions import Fraction
from types import MappingProxyType
from typing import Callable, Mapping

from vulcan.constitution.primitives import (
    Digest,
    canonical_json,
    canonical_json_loads,
    canonical_timestamp,
    parse_timestamp,
    require_utc,
)
from vulcan.graphix.epistemic import ClaimStatus, EpistemicCommit

VERIFIER_REGISTRY_SCHEMA = "graphix.verifier-registry/1"
WARRANT_RECEIPT_SCHEMA = "graphix.warrant-receipt/1"
SUPPORTED_STATUSES = frozenset(
    {
        ClaimStatus.COMPUTED,
        ClaimStatus.RETRIEVED,
        ClaimStatus.UNKNOWN,
        ClaimStatus.CONTESTED,
    }
)
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_REGISTRY_AUTHORITY = object()
_VERIFIER_RELEASE = "phase-b/1"


class VerificationError(ValueError):
    """The proposed warrant cannot be independently reproduced."""


class IntegrityState(str, Enum):
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"


class DurableCommitState(str, Enum):
    CANDIDATE = "CANDIDATE"
    COMMITTED = "COMMITTED"


class DeonticState(str, Enum):
    UNAUTHORIZED = "UNAUTHORIZED"
    AUTHORIZED = "AUTHORIZED"


class EffectState(str, Enum):
    NONE = "NONE"
    AUTHORIZED = "AUTHORIZED"
    RECEIPTED = "RECEIPTED"


@dataclass(frozen=True, slots=True)
class VerificationRequest:
    """Evidence needed by a verifier; never a caller assertion of truth."""

    claim_id: str
    status: ClaimStatus
    value: str | None
    operation_id: str | None = None
    operands: tuple[str, ...] = ()
    trace: tuple[str, ...] = ()
    material: bytes | None = None
    extraction_path: tuple[str, ...] = ()
    supported_values: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", ClaimStatus(self.status))
        for name in ("operands", "trace", "extraction_path", "supported_values"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not isinstance(self.claim_id, str) or not 3 <= len(self.claim_id) <= 128:
            raise VerificationError("invalid warrant claim id")
        if self.value is not None and (
            not isinstance(self.value, str) or len(self.value) > 4096
        ):
            raise VerificationError("invalid warrant value")
        if self.operation_id is not None and (
            not isinstance(self.operation_id, str) or len(self.operation_id) > 128
        ):
            raise VerificationError("invalid operation id")
        if self.material is not None and (
            not isinstance(self.material, bytes) or len(self.material) > 1_048_576
        ):
            raise VerificationError("invalid admitted material")
        for name in ("operands", "trace", "extraction_path"):
            values = getattr(self, name)
            if len(values) > 64 or any(
                not isinstance(value, str) or not value or len(value) > 4096
                for value in values
            ):
                raise VerificationError(f"invalid {name}")
        if len(self.supported_values) > 8 or any(
            not isinstance(item, tuple)
            or len(item) != 2
            or any(not isinstance(value, str) or not value for value in item)
            for item in self.supported_values
        ):
            raise VerificationError("invalid contested support")


@dataclass(frozen=True, slots=True)
class WarrantReceipt:
    schema_version: str
    registry_digest: str
    qualified_core_release: str
    verifier_id: str
    verifier_release: str
    episode_id: str
    snapshot_digest: str
    admitted_at: datetime
    candidate_digest: str
    claim_id: str
    status: ClaimStatus
    value_digest: str | None
    evidence_digest: str
    trace_digest: str
    evidence_material: bytes
    trace_material: bytes
    receipt_digest: str
    _issuer: object = field(repr=False, compare=False)

    def body(self) -> dict[str, object]:
        return {
            "schema": self.schema_version,
            "registry": self.registry_digest,
            "core": self.qualified_core_release,
            "verifier": self.verifier_id,
            "verifier_release": self.verifier_release,
            "episode": self.episode_id,
            "snapshot": self.snapshot_digest,
            "admitted_at": canonical_timestamp(self.admitted_at),
            "candidate": self.candidate_digest,
            "claim": self.claim_id,
            "status": self.status.value,
            "value": self.value_digest,
            "evidence": self.evidence_digest,
            "trace": self.trace_digest,
            "evidence_material": canonical_json_loads(self.evidence_material),
            "trace_material": canonical_json_loads(self.trace_material),
        }

    def canonical_bytes(self) -> bytes:
        body = self.body()
        if str(Digest.of_bytes(canonical_json(body))) != self.receipt_digest:
            raise VerificationError("warrant receipt digest mismatch")
        return canonical_json({**body, "receipt_digest": self.receipt_digest})


@dataclass(frozen=True, slots=True)
class VerifiedEpistemicCandidate:
    """Candidate plus private verifier receipts; still not a durable commit."""

    commit: EpistemicCommit
    receipts: tuple[WarrantReceipt, ...]
    integrity: IntegrityState = IntegrityState.VERIFIED
    durable_state: DurableCommitState = DurableCommitState.CANDIDATE
    publication: DeonticState = DeonticState.UNAUTHORIZED
    effect: EffectState = EffectState.NONE


class VerifierRegistry:
    """Closed registry. It deliberately exposes no runtime registration method."""

    def __init__(
        self,
        *,
        qualified_core_release: str,
        evaluators: Mapping[
            str, Callable[[tuple[str, ...]], tuple[str, tuple[str, ...]]]
        ],
        _authority: object = None,
    ) -> None:
        if _authority is not _REGISTRY_AUTHORITY:
            raise TypeError("VerifierRegistry is created only by the qualified core")
        if not _DIGEST.fullmatch(qualified_core_release):
            raise VerificationError("qualified core release must be digest-bound")
        if not evaluators or any(not key for key in evaluators):
            raise VerificationError(
                "closed verifier registry requires named evaluators"
            )
        self._release = qualified_core_release
        self._evaluators = MappingProxyType(dict(sorted(evaluators.items())))
        declaration = {
            "schema": VERIFIER_REGISTRY_SCHEMA,
            "qualified_core_release": qualified_core_release,
            "verifiers": tuple(self._evaluators),
        }
        self.registry_digest = str(Digest.of_bytes(canonical_json(declaration)))
        self._issuer = object()
        self._issued: dict[int, str] = {}

    @property
    def verifier_ids(self) -> tuple[str, ...]:
        return tuple(self._evaluators)

    def issue(
        self,
        *,
        commit: EpistemicCommit,
        requests: tuple[VerificationRequest, ...],
        admitted_at: datetime,
        context_digest: str,
    ) -> VerifiedEpistemicCandidate:
        require_utc(admitted_at, name="admitted_at")
        if commit.committed_at != admitted_at:
            raise VerificationError("candidate time differs from admitted time")
        if commit.snapshot_digest != context_digest:
            raise VerificationError("candidate snapshot differs from admitted context")
        indexed = {claim.claim_id: claim for claim in commit.claims}
        if set(indexed) != {request.claim_id for request in requests}:
            raise VerificationError("every claim requires exactly one warrant request")
        if len(requests) != len(indexed):
            raise VerificationError("duplicate warrant request")
        receipts = tuple(
            self._verify(commit, indexed[request.claim_id], request, admitted_at)
            for request in requests
        )
        return VerifiedEpistemicCandidate(commit, receipts)

    def validate(self, candidate: VerifiedEpistemicCandidate) -> EpistemicCommit:
        if not isinstance(candidate, VerifiedEpistemicCandidate):
            raise TypeError("privately verified candidate required")
        if candidate.integrity is not IntegrityState.VERIFIED:
            raise VerificationError("content integrity is not verified")
        if candidate.durable_state is not DurableCommitState.CANDIDATE:
            raise VerificationError("warrant cannot imply durable commit state")
        if candidate.publication is not DeonticState.UNAUTHORIZED:
            raise VerificationError("warrant cannot imply publication authority")
        if candidate.effect is not EffectState.NONE:
            raise VerificationError("warrant cannot imply effect authority")
        if len(candidate.receipts) != len(candidate.commit.claims):
            raise VerificationError("incomplete warrant set")
        claims = {claim.claim_id: claim for claim in candidate.commit.claims}
        if {receipt.claim_id for receipt in candidate.receipts} != set(claims):
            raise VerificationError("warrant set does not cover candidate claims")
        for receipt in candidate.receipts:
            if (
                not isinstance(receipt, WarrantReceipt)
                or receipt._issuer is not self._issuer
                or self._issued.get(id(receipt)) != receipt.receipt_digest
                or receipt.registry_digest != self.registry_digest
                or receipt.qualified_core_release != self._release
                or receipt.schema_version != WARRANT_RECEIPT_SCHEMA
                or receipt.verifier_release != _VERIFIER_RELEASE
                or receipt.status is not claims[receipt.claim_id].status
                or receipt.candidate_digest != candidate.commit.commit_digest
                or receipt.episode_id != candidate.commit.episode_id
                or receipt.snapshot_digest != candidate.commit.snapshot_digest
                or str(Digest.of_bytes(canonical_json(receipt.body())))
                != receipt.receipt_digest
            ):
                raise VerificationError("receipt was not issued by this registry")
        return candidate.commit

    def reverify_persisted(
        self, raw: bytes, *, commit: EpistemicCommit
    ) -> WarrantReceipt:
        """Cold-reconstruct and independently re-evaluate one journal receipt."""
        document = canonical_json_loads(raw)
        if not isinstance(document, dict) or canonical_json(document) != raw:
            raise VerificationError("persisted warrant is not canonical")
        expected = {
            "schema",
            "registry",
            "core",
            "verifier",
            "verifier_release",
            "episode",
            "snapshot",
            "admitted_at",
            "candidate",
            "claim",
            "status",
            "value",
            "evidence",
            "trace",
            "evidence_material",
            "trace_material",
            "receipt_digest",
        }
        if set(document) != expected:
            raise VerificationError("persisted warrant schema is not closed")
        evidence_material = canonical_json(document["evidence_material"])
        trace_material = canonical_json(document["trace_material"])
        try:
            admitted_at = parse_timestamp(str(document["admitted_at"]))
            status = ClaimStatus(str(document["status"]))
        except (TypeError, ValueError) as exc:
            raise VerificationError("persisted warrant types are invalid") from exc
        receipt = WarrantReceipt(
            str(document["schema"]),
            str(document["registry"]),
            str(document["core"]),
            str(document["verifier"]),
            str(document["verifier_release"]),
            str(document["episode"]),
            str(document["snapshot"]),
            admitted_at,
            str(document["candidate"]),
            str(document["claim"]),
            status,
            None if document["value"] is None else str(document["value"]),
            str(document["evidence"]),
            str(document["trace"]),
            evidence_material,
            trace_material,
            str(document["receipt_digest"]),
            self._issuer,
        )
        try:
            require_utc(receipt.admitted_at, name="admitted_at")
            claim = next(
                item for item in commit.claims if item.claim_id == receipt.claim_id
            )
        except (StopIteration, TypeError, ValueError) as exc:
            raise VerificationError(
                "persisted warrant claim binding is invalid"
            ) from exc
        if (
            receipt.schema_version != WARRANT_RECEIPT_SCHEMA
            or receipt.verifier_release != _VERIFIER_RELEASE
            or receipt.registry_digest != self.registry_digest
            or receipt.qualified_core_release != self._release
            or receipt.candidate_digest != commit.commit_digest
            or receipt.episode_id != commit.episode_id
            or receipt.snapshot_digest != commit.snapshot_digest
            or receipt.admitted_at != commit.committed_at
            or receipt.status is not claim.status
            or str(Digest.of_json(canonical_json_loads(evidence_material)))
            != receipt.evidence_digest
            or str(Digest.of_json(canonical_json_loads(trace_material)))
            != receipt.trace_digest
            or str(Digest.of_bytes(canonical_json(receipt.body())))
            != receipt.receipt_digest
        ):
            raise VerificationError("persisted warrant binding mismatch")
        evidence = canonical_json_loads(evidence_material)
        decoded_trace = canonical_json_loads(trace_material)
        if not isinstance(decoded_trace, list):
            raise VerificationError("persisted warrant trace is malformed")
        trace = tuple(decoded_trace)
        expected_value_digest = (
            None
            if receipt.status in {ClaimStatus.UNKNOWN, ClaimStatus.CONTESTED}
            else str(Digest.of_json(claim.proposition.object_value))
        )
        if receipt.value_digest != expected_value_digest:
            raise VerificationError("persisted warrant value binding mismatch")
        if receipt.verifier_id.startswith("computed:"):
            evaluator = self._evaluators.get(
                receipt.verifier_id.removeprefix("computed:")
            )
            if not isinstance(evidence, dict) or evaluator is None:
                raise VerificationError("persisted computation verifier unavailable")
            result, replay_trace = evaluator(tuple(evidence.get("operands", ())))
            if (
                result != claim.proposition.object_value
                or result != evidence.get("result")
                or replay_trace != trace
            ):
                raise VerificationError("persisted computation did not reproduce")
        elif receipt.verifier_id == "retrieved:canonical-json":
            if not isinstance(evidence, dict):
                raise VerificationError("persisted retrieval evidence malformed")
            value = evidence.get("document")
            for part in evidence.get("path", ()):
                if not isinstance(value, dict) or part not in value:
                    raise VerificationError("persisted retrieval path is unresolved")
                value = value[part]
            if (
                value != evidence.get("value")
                or str(value) != claim.proposition.object_value
            ):
                raise VerificationError("persisted retrieval did not reproduce")
        elif receipt.verifier_id == "unknown:closed-world":
            if (
                evidence != {"value": None}
                or trace
                or claim.proposition.object_value
                not in {"unknown", "unavailable", "unsupported"}
            ):
                raise VerificationError("persisted UNKNOWN warrant is malformed")
        elif receipt.verifier_id == "contested:multi-source":
            values = evidence.get("values") if isinstance(evidence, dict) else None
            if (
                not isinstance(values, list)
                or not 2 <= len(values) <= 8
                or trace
                or len(
                    {
                        item[0]
                        for item in values
                        if isinstance(item, list) and len(item) == 2
                    }
                )
                < 2
                or len(
                    {
                        item[1]
                        for item in values
                        if isinstance(item, list) and len(item) == 2
                    }
                )
                != len(values)
                or any(
                    not isinstance(item, list)
                    or len(item) != 2
                    or not isinstance(item[0], str)
                    or not _DIGEST.fullmatch(item[1])
                    for item in values
                )
            ):
                raise VerificationError("persisted CONTESTED warrant is malformed")
        else:
            raise VerificationError("persisted warrant verifier is unavailable")
        return receipt

    def _verify(self, commit, claim, request, admitted_at) -> WarrantReceipt:
        if claim.status is not request.status or claim.status not in SUPPORTED_STATUSES:
            raise VerificationError("status has no Phase-B verifier")
        if request.value != (
            None
            if claim.status in {ClaimStatus.UNKNOWN, ClaimStatus.CONTESTED}
            else claim.proposition.object_value
        ):
            raise VerificationError("request value does not match candidate")
        verifier_id: str
        evidence: object
        trace: tuple[str, ...]
        if claim.status is ClaimStatus.COMPUTED:
            evaluator = self._evaluators.get(request.operation_id or "")
            if evaluator is None:
                raise VerificationError("operation is not registered")
            recomputed, trace = evaluator(request.operands)
            if recomputed != request.value or (
                request.trace and trace != request.trace
            ):
                raise VerificationError("computation did not reproduce exactly")
            verifier_id = f"computed:{request.operation_id}"
            evidence = {
                "operands": request.operands,
                "result": recomputed,
                "trace": trace,
            }
        elif claim.status is ClaimStatus.RETRIEVED:
            if request.material is None or not request.extraction_path:
                raise VerificationError(
                    "retrieval requires exact material and extraction path"
                )
            try:
                value: object = canonical_json_loads(request.material)
            except (TypeError, ValueError) as exc:
                raise VerificationError(
                    "retrieval material is not canonical JSON"
                ) from exc
            if canonical_json(value) != request.material:
                raise VerificationError("retrieval material is not canonical JSON")
            for part in request.extraction_path:
                if not isinstance(value, dict) or part not in value:
                    raise VerificationError("retrieval extraction path is unresolved")
                value = value[part]
            if str(value) != request.value:
                raise VerificationError(
                    "retrieved value differs from admitted material"
                )
            verifier_id, trace = "retrieved:canonical-json", request.extraction_path
            evidence = {
                "document": canonical_json_loads(request.material),
                "path": trace,
                "value": value,
            }
        elif claim.status is ClaimStatus.UNKNOWN:
            if request.value is not None or claim.proposition.object_value not in {
                "unknown",
                "unavailable",
                "unsupported",
            }:
                raise VerificationError("UNKNOWN must carry no supported value")
            verifier_id, trace, evidence = "unknown:closed-world", (), {"value": None}
        else:
            values = request.supported_values
            if (
                request.value is not None
                or len(values) < 2
                or len({v for v, _ in values}) < 2
            ):
                raise VerificationError(
                    "CONTESTED requires incompatible supported values and no winner"
                )
            if len({support for _, support in values}) != len(values) or any(
                not _DIGEST.fullmatch(support) for _, support in values
            ):
                raise VerificationError(
                    "contested values require independent support digests"
                )
            verifier_id, trace, evidence = (
                "contested:multi-source",
                (),
                {"values": values},
            )
        body = {
            "schema": WARRANT_RECEIPT_SCHEMA,
            "registry": self.registry_digest,
            "core": self._release,
            "verifier": verifier_id,
            "verifier_release": _VERIFIER_RELEASE,
            "episode": commit.episode_id,
            "snapshot": commit.snapshot_digest,
            "admitted_at": canonical_timestamp(admitted_at),
            "candidate": commit.commit_digest,
            "claim": claim.claim_id,
            "status": claim.status.value,
            "value": (
                None if request.value is None else str(Digest.of_json(request.value))
            ),
            "evidence": str(Digest.of_json(evidence)),
            "trace": str(Digest.of_json(trace)),
            "evidence_material": evidence,
            "trace_material": trace,
        }
        digest = str(Digest.of_bytes(canonical_json(body)))
        receipt = WarrantReceipt(
            WARRANT_RECEIPT_SCHEMA,
            self.registry_digest,
            self._release,
            verifier_id,
            _VERIFIER_RELEASE,
            commit.episode_id,
            commit.snapshot_digest,
            admitted_at,
            commit.commit_digest,
            claim.claim_id,
            claim.status,
            body["value"],
            body["evidence"],
            body["trace"],
            canonical_json(evidence),
            canonical_json(trace),
            digest,
            self._issuer,
        )
        self._issued[id(receipt)] = digest
        return receipt


def _integer_add(operands: tuple[str, ...]) -> tuple[str, tuple[str, ...]]:
    if len(operands) != 2 or any(
        not re.fullmatch(r"-?[0-9]{1,38}", v) for v in operands
    ):
        raise VerificationError("integer.add/1 requires two bounded integers")
    result = str(int(operands[0]) + int(operands[1]))
    return result, (f"load:{operands[0]}", f"load:{operands[1]}", f"add:{result}")


def _bounded_rational_expression(
    operands: tuple[str, ...],
) -> tuple[str, tuple[str, ...]]:
    if len(operands) != 1 or len(operands[0]) > 256:
        raise VerificationError("bounded-rational-expression/1 requires one expression")
    try:
        tree = ast.parse(operands[0], mode="eval")
    except SyntaxError as exc:
        raise VerificationError("invalid arithmetic expression") from exc
    nodes = tuple(ast.walk(tree))
    if len(nodes) > 64:
        raise VerificationError("arithmetic expression exceeds verifier budget")

    def evaluate(node: ast.AST, depth: int = 0) -> Fraction:
        if depth > 16:
            raise VerificationError("arithmetic expression exceeds verifier depth")
        if isinstance(node, ast.Constant) and type(node.value) is int:
            if abs(node.value).bit_length() > 128:
                raise VerificationError("integer exceeds verifier budget")
            return Fraction(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand, depth + 1)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
        ):
            left, right = evaluate(node.left, depth + 1), evaluate(
                node.right, depth + 1
            )
            if isinstance(node.op, ast.Add):
                result = left + right
            elif isinstance(node.op, ast.Sub):
                result = left - right
            elif isinstance(node.op, ast.Mult):
                result = left * right
            else:
                if right == 0:
                    raise VerificationError("division by zero")
                result = left / right
            if (
                result.numerator.bit_length() > 256
                or result.denominator.bit_length() > 256
            ):
                raise VerificationError("result exceeds verifier budget")
            return result
        raise VerificationError("unregistered arithmetic syntax")

    value = evaluate(tree.body)
    result = (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )
    return result, (
        f"parse:{Digest.of_json(operands[0])}",
        f"evaluate:exact-rational:{result}",
    )


def phase_b_registry(*, qualified_core_release: str) -> VerifierRegistry:
    """Construct the immutable code-owned registry for one qualified release."""
    return VerifierRegistry(
        qualified_core_release=qualified_core_release,
        evaluators={
            "bounded-rational-expression/1": _bounded_rational_expression,
            "integer.add/1": _integer_add,
        },
        _authority=_REGISTRY_AUTHORITY,
    )
