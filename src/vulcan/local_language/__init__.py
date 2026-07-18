"""Offline-only verification primitives for local language-interface releases.

This package deliberately contains no provider, model loader, network client, or
serving integration.  A release must be verified and independently promoted
before a future runtime integration can consider it.
"""

from .release import (
    LocalLanguageRelease,
    ReleaseRole,
    ReleaseVerificationError,
    verify_release,
)

__all__ = ["LocalLanguageRelease", "ReleaseRole", "ReleaseVerificationError", "verify_release", "VerifiedLocalSpanCompletion", "VerifiedAdapterMetadata", "parse_transformer_span_proposal", "SpanProposalError", "RUNTIME_ABI", "ImmutableTokenizerContract", "load_tokenizer_contract", "validate_tokenizer_contract", "decode_generated_suffix", "build_verified_adapter"]

from .governance import DatasetManifest, DatasetSource, ExampleRole, LanguageExample, ReleaseState
from .tokenizer import ImmutableTokenizerContract, load_tokenizer_contract, validate_tokenizer_contract, decode_generated_suffix
from .adapter import VerifiedLocalSpanCompletion, VerifiedAdapterMetadata, parse_transformer_span_proposal, SpanProposalError, RUNTIME_ABI, build_verified_adapter
