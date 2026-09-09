from __future__ import annotations

import hashlib
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from vulcan.graphix.codec import canonical_json
from vulcan.graphix.core import AuthorityLevel
from vulcan.graphix.dialects import (
    EPISTEMIC_COMMIT_CANDIDATE,
    INTERPRETATION_CANDIDATE,
    PLAN_CANDIDATE,
    RESPONSE_PROJECTION,
)
from vulcan.graphix.operations import CANONICAL_OPERATIONS
from vulcan.graphix.validation import ValidatedGraphixArtifact


@dataclass(frozen=True, slots=True)
class CompilationRecord:
    source_validation_digest: str
    source_dialect: str
    target_dialect: str
    output_authority: AuthorityLevel
    output_digest: str
    audit_digest: str
    projection: Mapping[str, object]


class CompilationError(ValueError):
    pass


def compile_graphix(
    validated: ValidatedGraphixArtifact, *, target_dialect: str
) -> CompilationRecord:
    if validated.target_dialect != target_dialect:
        raise CompilationError("target dialect must be explicit and match validation")
    env = validated.envelope
    if env.authority_level is not AuthorityLevel.UNTRUSTED_PROPOSAL:
        raise CompilationError("compiler cannot elevate non-proposal authority")
    if target_dialect == INTERPRETATION_CANDIDATE:
        projection = {
            "kind": "interpretation_candidate",
            "source_artifact_id": env.node_artifact_id,
            "episode_id": env.episode_id,
            "authority_level": AuthorityLevel.VALIDATED_CANDIDATE.value,
            "private_reasoning": "redacted",
            "notes": "validated proposal only; kernel commit required",
        }
        authority = AuthorityLevel.VALIDATED_CANDIDATE
    elif target_dialect == PLAN_CANDIDATE:
        declaration = next(
            (item for item in env.extensions if item.namespace == "org.vulcan.plan"),
            None,
        )
        if declaration is None:
            raise CompilationError("plan candidate requires org.vulcan.plan extension")
        operation_name = declaration.value.get("operation")
        operands = declaration.value.get("operands")
        if not isinstance(operation_name, str) or not isinstance(operands, Mapping):
            raise CompilationError("invalid registered operation payload")
        try:
            operation = CANONICAL_OPERATIONS.require(operation_name)
        except ValueError as exc:
            raise CompilationError("operation is not registered") from exc
        if set(operands) != {operation.operand}:
            raise CompilationError("registered operation operands do not match")
        operand = operands[operation.operand]
        if not isinstance(operand, str) or not operand or len(operand) > 512:
            raise CompilationError("operation operand outside bounds")
        projection = {
            "kind": "arithmetic_domain_plan_candidate",
            "operation": operation.name,
            "operands": {operation.operand: operand},
            "source_artifact_id": env.node_artifact_id,
            "snapshot_bundle_digest": env.snapshot_bundle_digest,
            "authority_level": AuthorityLevel.VALIDATED_CANDIDATE.value,
        }
        authority = AuthorityLevel.VALIDATED_CANDIDATE
    elif target_dialect == EPISTEMIC_COMMIT_CANDIDATE:
        projection = {
            "kind": "epistemic_commit_candidate",
            "source_artifact_id": env.node_artifact_id,
            "source_content_digest": env.content_digest,
            "snapshot_bundle_digest": env.snapshot_bundle_digest,
            "authority_level": AuthorityLevel.VALIDATED_CANDIDATE.value,
            "commit_required": True,
        }
        authority = AuthorityLevel.VALIDATED_CANDIDATE
    elif target_dialect == RESPONSE_PROJECTION:
        projection = {
            "kind": "human_explanation",
            "source_artifact_id": env.node_artifact_id,
            "max_chars": 1024,
            "summary": "Graphix proposal passed compiler validation. It is not a committed belief or authorized plan.",
            "omitted": ["private_reasoning", "raw_chain_of_thought"],
        }
        authority = AuthorityLevel.VALIDATED_CANDIDATE
    else:
        raise CompilationError("unsupported target dialect")
    out_digest = "sha256:" + hashlib.sha256(canonical_json(projection)).hexdigest()
    audit = {
        "source_validation_digest": validated.validation_digest,
        "source_dialect": env.dialect,
        "target_dialect": target_dialect,
        "output_digest": out_digest,
        "stages": list(validated.stage_digests),
    }
    return CompilationRecord(
        validated.validation_digest,
        env.dialect,
        target_dialect,
        authority,
        out_digest,
        "sha256:" + hashlib.sha256(canonical_json(audit)).hexdigest(),
        MappingProxyType(projection),
    )
