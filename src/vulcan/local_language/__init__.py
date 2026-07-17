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

__all__ = ["LocalLanguageRelease", "ReleaseRole", "ReleaseVerificationError", "verify_release"]

from .governance import DatasetManifest, DatasetSource, ExampleRole, LanguageExample, ReleaseState
from .tokenizer import ImmutableTokenizerContract, load_tokenizer_contract
