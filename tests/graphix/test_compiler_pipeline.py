from datetime import datetime, timezone
from dataclasses import replace
import pytest

from vulcan.graphix.core import AuthorityLevel, EpistemicStatus, ExtensionDeclaration, GraphixEnvelope, PrincipalRelease, PrivacyClass, SourceKind, SourceReference
from vulcan.graphix.codec import extension_digest
from vulcan.graphix.validation import STAGES, ValidationError, ValidationPolicy, validate_graphix
from vulcan.graphix.compilers import CompilationError, compile_graphix
from vulcan.graphix.migrations import MigrationError, MigrationPlan

D = "sha256:" + "1"*64
NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)

def policy():
    return ValidationPolicy(now=lambda: NOW, trusted_principals=frozenset({"kernel.principal"}), allowed_dialects=frozenset({"graphix.language","graphix.language.candidate","graphix.response.projection"}))

def env(**kw):
    base = GraphixEnvelope(
        dialect="graphix.language", schema_version=1, node_artifact_id="artifact:001", episode_id="episode:001", content_digest=D,
        proposer=PrincipalRelease("kernel.principal","rel:001"), authority_level=AuthorityLevel.UNTRUSTED_PROPOSAL,
        source_references=(SourceReference(SourceKind.ARTIFACT,"source:001",D),), snapshot_bundle_digest=D,
        epistemic_status=EpistemicStatus.PROPOSED, privacy_class=PrivacyClass.INTERNAL, purpose="language proposal validation", consent_references=(), valid_from=NOW, valid_until=None,
    )
    return replace(base, **kw)

def test_valid_language_proposal_becomes_candidate_not_belief():
    validated = validate_graphix(env(), target_dialect="graphix.language.candidate", policy=policy())
    record = compile_graphix(validated, target_dialect="graphix.language.candidate")
    assert record.output_authority is AuthorityLevel.VALIDATED_CANDIDATE
    assert record.projection["authority_level"] == "VALIDATED_CANDIDATE"
    assert "kernel commit required" in record.projection["notes"]

def test_validation_rejects_skipped_or_reordered_stages():
    with pytest.raises(ValidationError, match="mandatory and ordered"):
        validate_graphix(env(), target_dialect="graphix.language.candidate", policy=policy(), stages=STAGES[:-1])
    with pytest.raises(ValidationError, match="mandatory and ordered"):
        validate_graphix(env(), target_dialect="graphix.language.candidate", policy=policy(), stages=tuple(reversed(STAGES)))

def test_authority_smuggling_is_rejected():
    with pytest.raises(ValidationError, match="untrusted proposals"):
        validate_graphix(env(authority_level=AuthorityLevel.COMMITTED_BELIEF), target_dialect="graphix.language.candidate", policy=policy())

def test_security_extension_claim_rejected():
    value = {"meaning":"policy override"}
    ext = ExtensionDeclaration("com.example.security", 1, extension_digest(value), value)
    with pytest.raises(ValidationError, match="reserved meaning"):
        validate_graphix(env(extensions=(ext,)), target_dialect="graphix.language.candidate", policy=policy())

def test_resource_bomb_rejected():
    p = ValidationPolicy(now=lambda: NOW, trusted_principals=frozenset({"kernel.principal"}), allowed_dialects=frozenset({"graphix.language","graphix.language.candidate"}), max_canonical_bytes=128)
    with pytest.raises(ValidationError, match="resource bounds"):
        validate_graphix(env(), target_dialect="graphix.language.candidate", policy=p)

def test_compile_requires_explicit_validated_target():
    validated = validate_graphix(env(), target_dialect="graphix.language.candidate", policy=policy())
    with pytest.raises(CompilationError, match="explicit"):
        compile_graphix(validated, target_dialect="graphix.response.projection")

def test_response_projection_is_bounded_and_redacts_reasoning():
    validated = validate_graphix(env(), target_dialect="graphix.response.projection", policy=policy())
    record = compile_graphix(validated, target_dialect="graphix.response.projection")
    assert record.projection["max_chars"] == 1024
    assert "raw_chain_of_thought" in record.projection["omitted"]

def test_migration_unsupported_and_authority_change_rejected():
    with pytest.raises(MigrationError, match="unsupported"):
        MigrationPlan({}).migrate(env(), to_version=2)
    def bad(e):
        return replace(e, schema_version=2, authority_level=AuthorityLevel.AUTHORIZED_PLAN)
    with pytest.raises(MigrationError, match="authority"):
        MigrationPlan({("graphix.language",1,2): bad}).migrate(env(), to_version=2)

def test_migration_pure_content_addressed():
    def good(e): return replace(e, schema_version=2)
    rec1 = MigrationPlan({("graphix.language",1,2): good}).migrate(env(), to_version=2)
    rec2 = MigrationPlan({("graphix.language",1,2): good}).migrate(env(), to_version=2)
    assert rec1.source_digest == rec2.source_digest
    assert rec1.target_digest == rec2.target_digest
    assert rec1.artifact.authority_level is AuthorityLevel.UNTRUSTED_PROPOSAL
