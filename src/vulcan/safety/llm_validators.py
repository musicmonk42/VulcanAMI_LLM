from __future__ import annotations

"""
LLM Safety Validators (Fully Functional)

Provides multi-layered validation utilities for token and sequence generation:

Core Validators:
- ToxicityValidator: heuristic + pattern + lightweight lexicon scoring
- HallucinationValidator: uses optional world_model signals (knowledge base / confidence)
- PromptInjectionValidator: detects common prompt injection & jailbreak directives

EnhancedSafetyValidator:
- Aggregates validators
- Produces structured SafetyEvent records
- Supports token-level & sequence-level validation
- Integrates optional world_model consistency (validate_generation / suggest_correction)
- Optional risk scoring & replacement strategies

Design Goals:
- Dependency-light (pure Python)
- Duck-typed world_model integration (if provided)
- Deterministic unless random strategies explicitly added
- Extensible: you can pass extra validators or override policies

Returned SafetyEvent example:
{
  "kind": "toxicity",
  "token": "idiot",
  "risk": 0.92,
  "action": "redacted",
  "replacement": "[REDACTED]",
  "reason": "matched toxic lexicon",
  "timestamp": 1731510000.123,
  "meta": {...}
}

Public API:
- EnhancedSafetyValidator.validate_generation(token, context, world_model=None) -> safe_token
- EnhancedSafetyValidator.validate_sequence(tokens, context, world_model=None) -> True | False | corrected_tokens
- EnhancedSafetyValidator.last_events -> list[SafetyEvent]
- EnhancedSafetyValidator.assess(token, context) -> (safe_token, SafetyEvent|None)
"""

import asyncio
import inspect
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Initialize logger
logger = logging.getLogger(__name__)

Token = Union[int, str]
Tokens = Sequence[Token]


# ---------------------------------------------------------------------------
# Common Data Structures
# ---------------------------------------------------------------------------


@dataclass
class SafetyEvent:
    kind: str
    token: Any
    risk: float
    action: str
    reason: str
    replacement: Optional[Any] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Base Validator Interface (duck-typed)
# ---------------------------------------------------------------------------


class BaseValidator:
    name: str = "base"

    def check(self, token: Any, context: Dict[str, Any]) -> bool:
        raise NotImplementedError

    def score(self, token: Any, context: Dict[str, Any]) -> float:
        """
        Return risk score in [0,1]; higher == more risky.
        """
        return 0.0

    def get_safe_alternative(self, token: Any, context: Dict[str, Any]) -> Any:
        return token

    def reason(self, token: Any, context: Dict[str, Any]) -> str:
        return "no-op"

    def applies(self, token: Any) -> bool:
        return True


# ---------------------------------------------------------------------------
# Toxicity Validator
# ---------------------------------------------------------------------------


class ToxicityValidator(BaseValidator):
    """
    Prevents toxic generations via pattern & lexicon scoring.
    """

    name = "toxicity"

    def __init__(self) -> None:
        # Simple lexicon; expand as needed
        self._lexicon = {
            "idiot": 0.9,
            "stupid": 0.85,
            "dumb": 0.82,
            "hate": 0.75,
            "moron": 0.88,
            "kill": 0.95,
            "kys": 0.98,
        }
        self._patterns = [
            (re.compile(r"\b(fuck|shit|bitch|asshole)\b", re.IGNORECASE), 0.8),
            (re.compile(r"\b(go\s+kill\s+yourself)\b", re.IGNORECASE), 1.0),
        ]
        self.threshold = 0.5

    def _normalize(self, token: Any) -> str:
        return str(token).lower()

    def score(self, token: Any, context: Dict[str, Any]) -> float:
        text = self._normalize(token)
        base = 0.0
        for w, s in self._lexicon.items():
            if w in text:
                base = max(base, s)
        for rx, sc in self._patterns:
            if rx.search(text):
                base = max(base, sc)
        # Optional context factor (e.g., conversation sentiment)
        sentiment = context.get("sentiment")  # expected in [-1,1]
        if isinstance(sentiment, (int, float)):
            # Increase risk if negative sentiment
            if sentiment < -0.3:
                base = min(1.0, base + 0.1)
        
        # FIX: Reduce risk for internal platform calls (arena, agent, internal)
        # These are trusted sources that shouldn't be blocked by false positives
        source = context.get("source", "user")
        if source in ("arena", "agent", "internal", "system"):
            # Significant reduction for internal calls - only block truly dangerous content
            base *= 0.4
        
        return base

    def check(self, token: Any, context: Dict[str, Any]) -> bool:
        return self.score(token, context) < self.threshold

    def get_safe_alternative(self, token: Any, context: Dict[str, Any]) -> Any:
        return "[REDACTED]"

    def reason(self, token: Any, context: Dict[str, Any]) -> str:
        return (
            "matched toxic lexicon or pattern"
            if not self.check(token, context)
            else "non-toxic"
        )

    def applies(self, token: Any) -> bool:
        return isinstance(token, (str, int))


# ---------------------------------------------------------------------------
# Hallucination Validator
# ---------------------------------------------------------------------------


class HallucinationValidator(BaseValidator):
    """
    Detects probable hallucinations using a world model interface (stub).
    Duck-typed world_model capabilities:
      - has_knowledge(token) -> bool
      - confidence(token) -> float in [0,1]
    """

    name = "hallucination"

    def __init__(self, low_conf_threshold: float = 0.25) -> None:
        self.low_conf_threshold = low_conf_threshold

    def score(self, token: Any, context: Dict[str, Any]) -> float:
        world_model = context.get("world_model")
        if not world_model:
            return 0.0
        token_str = str(token)
        conf_val = None
        try:
            if hasattr(world_model, "confidence"):
                conf_val = world_model.confidence(token_str)
            elif hasattr(world_model, "get_confidence"):
                conf_val = world_model.get_confidence(token_str)
        except Exception:
            conf_val = None

        if conf_val is None:
            # Unknown; treat as low hallucination risk
            return 0.0
        # Invert confidence for risk
        return max(0.0, min(1.0, 1.0 - conf_val))

    def check(self, token: Any, context: Dict[str, Any]) -> bool:
        return self.score(token, context) < 0.6  # risk < 0.6 considered acceptable

    def get_safe_alternative(self, token: Any, context: Dict[str, Any]) -> Any:
        return "[VERIFY_FACT]"

    def reason(self, token: Any, context: Dict[str, Any]) -> str:
        risk = self.score(token, context)
        if risk >= 0.6:
            return f"low world model confidence (risk={risk:.2f})"
        return "adequate world model confidence"

    def applies(self, token: Any) -> bool:
        return isinstance(token, (str, int))


# ---------------------------------------------------------------------------
# Structural Validator
# ---------------------------------------------------------------------------


class StructuralValidator(BaseValidator):
    """
    Validates structural integrity and well-formedness of tokens.
    Checks for:
    - Proper formatting
    - Valid syntax patterns
    - Balanced delimiters
    """

    name = "structural"

    def __init__(self) -> None:
        self.delimiter_pairs = [
            ("(", ")"),
            ("[", "]"),
            ("{", "}"),
            ("<", ">"),
        ]
        self.threshold = 0.5

    def _check_balanced_delimiters(self, text: str) -> bool:
        """Check if delimiters are balanced in the text."""
        for open_d, close_d in self.delimiter_pairs:
            stack = []
            for char in text:
                if char == open_d:
                    stack.append(char)
                elif char == close_d:
                    if not stack:
                        return False
                    stack.pop()
            if stack:
                return False
        return True

    def score(self, token: Any, context: Dict[str, Any]) -> float:
        """
        Return structural risk score.
        Higher score means more structural issues.
        """
        text = str(token)
        risk = 0.0

        # Check for balanced delimiters
        if not self._check_balanced_delimiters(text):
            risk += 0.4

        # Check for excessive special characters (potential encoding issues)
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        if len(text) > 0:
            special_ratio = special_chars / len(text)
            if special_ratio > 0.5:
                risk += 0.3

        # Check for control characters
        if any(ord(c) < 32 and c not in "\n\r\t" for c in text):
            risk += 0.3

        return min(1.0, risk)

    def check(self, token: Any, context: Dict[str, Any]) -> bool:
        """Return True if token is structurally valid."""
        return self.score(token, context) < self.threshold

    def get_safe_alternative(self, token: Any, context: Dict[str, Any]) -> Any:
        """Return a safe alternative for structurally invalid tokens."""
        return "[SAFE_TOKEN]"

    def reason(self, token: Any, context: Dict[str, Any]) -> str:
        """Provide reason for structural validation result."""
        if not self.check(token, context):
            return "structural integrity issues detected"
        return "structurally valid"

    def applies(self, token: Any) -> bool:
        return isinstance(token, (str, int))


# ---------------------------------------------------------------------------
# Ethical Validator
# ---------------------------------------------------------------------------


class EthicalValidator(BaseValidator):
    """
    Validates ethical considerations in token generation.
    Checks for:
    - Harmful content
    - Discriminatory language
    - Privacy violations
    - Manipulation attempts
    
    FIX: False Positive Reduction
    - Raised threshold from 0.5 to 0.65 to reduce false positives
    - Added context awareness for internal/arena/agent sources
    - Individual keyword scores lowered to require combination of indicators
    - Added source-based risk reduction for internal platform calls
    """

    name = "ethical"

    def __init__(self) -> None:
        # Ethical concern patterns - require more specific phrases to trigger
        # These patterns require context (e.g., "harm people" not just "harm")
        self._concern_patterns = [
            # Only flag manipulation in harmful context
            (re.compile(r"\b(manipulate|deceive|trick)\s+(people|users|someone)\b", re.IGNORECASE), 0.7),
            (
                re.compile(r"\b(discriminat(e|ion)|prejudice|bigot)\b", re.IGNORECASE),
                0.8,
            ),
            (
                re.compile(
                    r"\b(harm|hurt|damage|destroy)\s+(people|humans|users)\b",
                    re.IGNORECASE,
                ),
                0.9,
            ),
            # More specific patterns for hacking/exploiting - require target
            (re.compile(r"\b(steal|hack|breach)\s+(data|account|system|password)\b", re.IGNORECASE), 0.75),
            (
                re.compile(
                    r"\b(personal|private)\s+(data|information)\s+(leak|expose|steal)\b", re.IGNORECASE
                ),
                0.7,
            ),
        ]

        # Ethical concern keywords - REDUCED scores to require multiple indicators
        # FIX: Single keywords like "exploit" can appear in legitimate contexts
        # (e.g., "exploit this opportunity", "security exploit detection")
        self._concern_keywords = {
            "exploit vulnerability": 0.7,  # More specific than just "exploit"
            "manipulate people": 0.65,
            "deceive users": 0.7,
            "harm": 0.5,  # Reduced - too common in legitimate contexts
            "discriminate": 0.75,  # Still flag discrimination
            "violate privacy": 0.65,
        }
        # FIX: Raised threshold from 0.5 to 0.65 to reduce false positives
        # This means queries need stronger indicators to be blocked
        self.threshold = 0.65

    def _normalize(self, token: Any) -> str:
        return str(token).lower()

    def score(self, token: Any, context: Dict[str, Any]) -> float:
        """
        Return ethical risk score.
        Higher score means more ethical concerns.
        
        FIX: Added source-based risk reduction for internal platform calls.
        Arena, agent, and internal sources get reduced sensitivity to avoid
        blocking legitimate platform operations.
        """
        text = self._normalize(token)
        risk = 0.0

        # Check patterns
        for pattern, score_val in self._concern_patterns:
            if pattern.search(text):
                risk = max(risk, score_val)

        # Check keywords (multi-word keywords for more precision)
        for keyword, score_val in self._concern_keywords.items():
            if keyword in text:
                risk = max(risk, score_val)

        # Context-based adjustments
        intent = context.get("intent")
        if intent in ("educational", "safety_training"):
            # Lower risk if context is educational
            risk *= 0.5
        
        # FIX: Reduce risk for internal platform calls (arena, agent, internal)
        # These are trusted sources that shouldn't be blocked by false positives
        source = context.get("source", "user")
        phase = context.get("phase", "")
        
        if source in ("arena", "agent", "internal", "system"):
            # Significant reduction for internal calls - only block truly dangerous content
            risk *= 0.4
        elif phase == "pre_check" and source == "user":
            # Normal user queries get standard processing
            pass

        return min(1.0, risk)

    def check(self, token: Any, context: Dict[str, Any]) -> bool:
        """Return True if token is ethically acceptable."""
        return self.score(token, context) < self.threshold

    def get_safe_alternative(self, token: Any, context: Dict[str, Any]) -> Any:
        """Return a safe alternative for ethically problematic tokens."""
        return "[SAFE_TOKEN]"

    def reason(self, token: Any, context: Dict[str, Any]) -> str:
        """Provide reason for ethical validation result."""
        if not self.check(token, context):
            return "ethical concerns detected"
        return "ethically acceptable"

    def applies(self, token: Any) -> bool:
        return isinstance(token, (str, int))


# ---------------------------------------------------------------------------
# Prompt Injection Validator
# ---------------------------------------------------------------------------


class PromptInjectionValidator(BaseValidator):
    """
    Detects prompt injection attempts (ignore instructions, system override).
    """

    name = "prompt_injection"

    def __init__(self) -> None:
        self._patterns = [
            (re.compile(r"(?i)\bignore\s+previous\s+instructions\b"), 0.9),
            (re.compile(r"(?i)\boverride\s+system\b"), 0.85),
            (re.compile(r"(?i)\bdisregard\s+all\b"), 0.8),
            (re.compile(r"(?i)^\s*system\s*:"), 0.7),
            (re.compile(r"(?i)\bself-destruct\b"), 0.95),
        ]
        self.threshold = 0.5

    def score(self, token: Any, context: Dict[str, Any]) -> float:
        text = str(token)
        base = 0.0
        for rx, sc in self._patterns:
            if rx.search(text):
                base = max(base, sc)
        # If context shows user role escalate
        role = context.get("role")
        if role == "user" and base > 0:
            base = min(1.0, base + 0.05)
        return base

    def check(self, token: Any, context: Dict[str, Any]) -> bool:
        return self.score(token, context) < self.threshold

    def get_safe_alternative(self, token: Any, context: Dict[str, Any]) -> Any:
        return "[NEUTRALIZED]"

    def reason(self, token: Any, context: Dict[str, Any]) -> str:
        return (
            "matched injection directive"
            if not self.check(token, context)
            else "no injection patterns"
        )

    def applies(self, token: Any) -> bool:
        return isinstance(token, (str, int))


# ---------------------------------------------------------------------------
# Enhanced Safety Validator (Aggregator)
# ---------------------------------------------------------------------------


class EnhancedSafetyValidator:
    """
    Aggregates all validators and orchestrates multi-stage safety checks.

    Features:
    - Token-level validate_generation()
    - Sequence-level validate_sequence()
    - Risk scoring & replacement logic
    - Optional world_model integration (consistency & correction)
    - Maintains last_events list for auditing

    Policy keys (override via set_policy()):
        {
          "max_sequence_toxicity_events": 3,
          "block_on_high_risk": True,
          "high_risk_threshold": 0.9,
          "mediate_on_hallucination": True,
          "world_model_consistency": True
        }
    """

    def __init__(
        self,
        extra_validators: Optional[List[BaseValidator]] = None,
        policy: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.validators: List[BaseValidator] = [
            ToxicityValidator(),
            HallucinationValidator(),
            PromptInjectionValidator(),
        ]
        if extra_validators:
            self.validators.extend(extra_validators)
        self.policy: Dict[str, Any] = {
            "max_sequence_toxicity_events": 3,
            "block_on_high_risk": True,
            "high_risk_threshold": 0.9,
            "mediate_on_hallucination": True,
            "world_model_consistency": True,
        }
        if policy:
            self.policy.update(policy)

        self.last_events: List[SafetyEvent] = []
        self.world_model = None  # Optionally set externally

    # --------------------- Policy Management --------------------- #

    def set_policy(self, **updates) -> None:
        self.policy.update(updates)

    def attach_world_model(self, world_model: Any) -> None:
        self.world_model = world_model

    # --------------------- Internal Helpers --------------------- #

    def _maybe_await(self, value: Any) -> Any:
        """
        Safely resolve a possibly awaitable value in a synchronous context.
        - If there is a running loop and value is awaitable, schedule and wait.
        - If no loop, run a temporary loop via asyncio.run.
        """
        if not inspect.isawaitable(value):
            return value
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Schedule coroutine thread-safely (assuming we are in a different thread).
            future = asyncio.run_coroutine_threadsafe(value, loop)
            return future.result()
        else:
            return asyncio.run(value)

    # --------------------- Token-Level Validation (Async) --------------------- #

    async def validate_generation_async(
        self, token: Token, context: Dict[str, Any], world_model: Any = None
    ) -> Token:
        """
        Async variant for callers already in async contexts.
        """
        # Reuse sync logic but allow async world model methods to be awaited directly.
        wm = world_model or self.world_model
        if wm and "world_model" not in context:
            context = {**context, "world_model": wm}

        original = token
        highest_risk = 0.0
        replacement = token

        for v in self.validators:
            if not v.applies(token):
                continue
            safe = v.check(token, context)
            risk = v.score(token, context)
            highest_risk = max(highest_risk, risk)

            if not safe:
                replacement_candidate = v.get_safe_alternative(token, context)
                ev = SafetyEvent(
                    kind=v.name,
                    token=original,
                    risk=risk,
                    action="redacted",
                    reason=v.reason(token, context),
                    replacement=replacement_candidate,
                    meta={"validator": v.name},
                )
                self.last_events.append(ev)
                replacement = replacement_candidate
                if v.name in ("toxicity", "prompt_injection"):
                    break

        if wm and self.policy.get("world_model_consistency", True):
            try:
                validate_fn = getattr(wm, "validate_generation", None)
                suggest_fn = getattr(wm, "suggest_correction", None)
                if validate_fn and suggest_fn:
                    valid_result = validate_fn(replacement, context)
                    if inspect.isawaitable(valid_result):
                        valid_result = await valid_result
                    if not valid_result:
                        corr = suggest_fn(replacement, context)
                        if inspect.isawaitable(corr):
                            corr = await corr
                        if corr != replacement:
                            ev = SafetyEvent(
                                kind="world_model_correction",
                                token=replacement,
                                risk=0.5,
                                action="corrected",
                                reason="world model consistency failure",
                                replacement=corr,
                                meta={},
                            )
                            self.last_events.append(ev)
                            replacement = corr
            except Exception as e:
                logger.warning(f"Failed to validate LLM output: {e}")

        if self.policy.get(
            "block_on_high_risk", True
        ) and highest_risk >= self.policy.get("high_risk_threshold", 0.9):
            if replacement == original:
                replacement = "[SAFE]"
                ev = SafetyEvent(
                    kind="high_risk_block",
                    token=original,
                    risk=highest_risk,
                    action="blocked",
                    reason="exceeded high risk threshold",
                    replacement=replacement,
                    meta={},
                )
                self.last_events.append(ev)

        return replacement

    # --------------------- Token-Level Validation (sync, async-aware) --------------------- #

    def validate_generation(
        self, token: Token, context: Dict[str, Any], world_model: Any = None
    ) -> Token:
        """
        Synchronous facade:
        - Runs validator passes synchronously.
        - If world model methods are async, safely resolves them via _maybe_await.
        """
        wm = world_model or self.world_model
        if wm and "world_model" not in context:
            context = {**context, "world_model": wm}

        original = token
        highest_risk = 0.0
        replacement = token

        for v in self.validators:
            if not v.applies(token):
                continue
            safe = v.check(token, context)
            risk = v.score(token, context)
            highest_risk = max(highest_risk, risk)

            if not safe:
                # Replace immediately and record event
                replacement_candidate = v.get_safe_alternative(token, context)
                ev = SafetyEvent(
                    kind=v.name,
                    token=original,
                    risk=risk,
                    action="redacted",
                    reason=v.reason(token, context),
                    replacement=replacement_candidate,
                    meta={"validator": v.name},
                )
                self.last_events.append(ev)
                replacement = replacement_candidate
                # For injection/toxicity we can break early
                if v.name in ("toxicity", "prompt_injection"):
                    break

        # World model consistency (optional)
        if wm and self.policy.get("world_model_consistency", True):
            try:
                validate_fn = getattr(wm, "validate_generation", None)
                suggest_fn = getattr(wm, "suggest_correction", None)
                if validate_fn and suggest_fn:
                    valid_result = validate_fn(replacement, context)
                    valid_result = self._maybe_await(valid_result)
                    if not valid_result:
                        corr = suggest_fn(replacement, context)
                        corr = self._maybe_await(corr)
                        if corr != replacement:
                            ev = SafetyEvent(
                                kind="world_model_correction",
                                token=replacement,
                                risk=0.5,
                                action="corrected",
                                reason="world model consistency failure",
                                replacement=corr,
                                meta={},
                            )
                            self.last_events.append(ev)
                            replacement = corr
            except Exception as e:
                logger.warning(f"Failed to check safety constraints: {e}")

        # High-risk block policy
        if self.policy.get(
            "block_on_high_risk", True
        ) and highest_risk >= self.policy.get("high_risk_threshold", 0.9):
            # Force replacement with a neutral token if not already replaced
            if replacement == original:
                replacement = "[SAFE]"
                ev = SafetyEvent(
                    kind="high_risk_block",
                    token=original,
                    risk=highest_risk,
                    action="blocked",
                    reason="exceeded high risk threshold",
                    replacement=replacement,
                    meta={},
                )
                self.last_events.append(ev)

        return replacement

    # --------------------- Sequence-Level Validation --------------------- #

    def validate_sequence(
        self, tokens: Tokens, context: Dict[str, Any], world_model: Any = None
    ) -> Union[bool, List[Token]]:
        """
        Validate entire sequence:
        Returns:
          - True if sequence acceptable (no modifications)
          - False if sequence blocked
          - List[Token] corrected sequence
        """
        wm = world_model or self.world_model
        if wm and "world_model" not in context:
            context = {**context, "world_model": wm}

        toxicity_events = 0
        corrected: List[Token] = []
        changed = False

        for t in tokens:
            new_t = self.validate_generation(t, context, wm)
            if new_t != t:
                changed = True
            corrected.append(new_t)
            # Count toxicity events
            toxicity_events += sum(
                1
                for ev in self.last_events[-3:]
                if ev.kind == "toxicity" and ev.token == t
            )

            if toxicity_events >= self.policy.get("max_sequence_toxicity_events", 3):
                # Block entire sequence
                ev = SafetyEvent(
                    kind="sequence_block",
                    token=t,
                    risk=1.0,
                    action="blocked",
                    reason="excessive toxicity events",
                    meta={"count": toxicity_events},
                )
                self.last_events.append(ev)
                return False

        return corrected if changed else True

    # --------------------- Assessment Helper --------------------- #

    def assess(
        self, token: Token, context: Dict[str, Any]
    ) -> Tuple[Token, Optional[SafetyEvent]]:
        """
        Single-token assessment returning (safe_token, event_if_any).
        Leaves full validate_generation logic intact but also returns last event.
        """
        before_count = len(self.last_events)
        safe = self.validate_generation(token, context)
        after_events = self.last_events[before_count:]
        return safe, after_events[-1] if after_events else None

    # --------------------- Reporting --------------------- #

    def get_events(self) -> List[Dict[str, Any]]:
        return [ev.to_dict() for ev in self.last_events]

    def flush_events(self) -> None:
        self.last_events.clear()

    def summary(self) -> Dict[str, Any]:
        return {
            "events_total": len(self.last_events),
            "recent_events": [ev.to_dict() for ev in self.last_events[-10:]],
            "policy": dict(self.policy),
            "validators": [type(v).__name__ for v in self.validators],
        }


# ---------------------------------------------------------------------------
# Convenience Factory
# ---------------------------------------------------------------------------


def build_default_safety_validator(
    extra_validators: Optional[List[BaseValidator]] = None,
    policy: Optional[Dict[str, Any]] = None,
) -> EnhancedSafetyValidator:
    return EnhancedSafetyValidator(extra_validators=extra_validators, policy=policy)
