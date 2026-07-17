"""Strict immutable custom-tokenizer contract for offline artifact review."""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from .release import ReleaseVerificationError

TOKENIZER_SCHEMA = "local-tokenizer/1"
MAX_VOCABULARY = 65_536


@dataclass(frozen=True)
class ImmutableTokenizerContract:
    normalization: str
    vocabulary: tuple[str, ...]
    special_tokens: tuple[str, ...]
    max_length: int


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ReleaseVerificationError("duplicate tokenizer JSON key")
        result[key] = value
    return result


def load_tokenizer_contract(path: str | Path) -> ImmutableTokenizerContract:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_pairs)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseVerificationError("unreadable tokenizer contract") from exc
    expected = {"schema_version", "normalization", "vocabulary", "special_tokens", "max_length"}
    if not isinstance(raw, dict) or set(raw) != expected or raw["schema_version"] != TOKENIZER_SCHEMA:
        raise ReleaseVerificationError("invalid tokenizer contract schema")
    vocabulary, special = raw["vocabulary"], raw["special_tokens"]
    if (not isinstance(vocabulary, list) or not 0 < len(vocabulary) <= MAX_VOCABULARY or
            any(not isinstance(token, str) or not token for token in vocabulary) or len(set(vocabulary)) != len(vocabulary)):
        raise ReleaseVerificationError("invalid immutable vocabulary")
    if (not isinstance(special, list) or not special or any(token not in vocabulary for token in special) or
            len(set(special)) != len(special)):
        raise ReleaseVerificationError("invalid special token map")
    if raw["normalization"] != "NFC" or isinstance(raw["max_length"], bool) or not isinstance(raw["max_length"], int) or not 0 < raw["max_length"] <= 10_000:
        raise ReleaseVerificationError("invalid tokenizer bounds")
    return ImmutableTokenizerContract(raw["normalization"], tuple(vocabulary), tuple(special), raw["max_length"])
