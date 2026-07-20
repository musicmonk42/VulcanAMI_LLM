from __future__ import annotations
from dataclasses import dataclass
import hashlib
from types import MappingProxyType
from typing import Mapping
from vulcan.graphix.codec import canonical_json
from vulcan.graphix.core import AuthorityLevel
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

class CompilationError(ValueError): pass

def compile_graphix(validated: ValidatedGraphixArtifact, *, target_dialect: str) -> CompilationRecord:
    if validated.target_dialect != target_dialect:
        raise CompilationError("target dialect must be explicit and match validation")
    env = validated.envelope
    if env.authority_level is not AuthorityLevel.UNTRUSTED_PROPOSAL:
        raise CompilationError("compiler cannot elevate non-proposal authority")
    if target_dialect == "graphix.language.candidate":
        projection = {"kind":"interpretation_candidate","source_artifact_id":env.node_artifact_id,"episode_id":env.episode_id,"authority_level":AuthorityLevel.VALIDATED_CANDIDATE.value,"private_reasoning":"redacted","notes":"validated proposal only; kernel commit required"}
        authority = AuthorityLevel.VALIDATED_CANDIDATE
    elif target_dialect == "graphix.response.projection":
        projection = {"kind":"human_explanation","source_artifact_id":env.node_artifact_id,"max_chars":1024,"summary":"Graphix proposal passed compiler validation. It is not a committed belief or authorized plan.","omitted":["private_reasoning","raw_chain_of_thought"]}
        authority = AuthorityLevel.VALIDATED_CANDIDATE
    else:
        raise CompilationError("unsupported target dialect")
    out_digest = "sha256:"+hashlib.sha256(canonical_json(projection)).hexdigest()
    audit = {"source_validation_digest":validated.validation_digest,"source_dialect":env.dialect,"target_dialect":target_dialect,"output_digest":out_digest,"stages":list(validated.stage_digests)}
    return CompilationRecord(validated.validation_digest, env.dialect, target_dialect, authority, out_digest, "sha256:"+hashlib.sha256(canonical_json(audit)).hexdigest(), MappingProxyType(projection))
