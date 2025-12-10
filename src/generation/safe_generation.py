from __future__ import annotations

"""
Safe Generation - Enhanced Multi-Layered Safety System

A comprehensive, production-ready safety filtering system for LLM token generation with:
- Multi-tier validation pipeline (toxicity, hallucination, prompt injection, PII, bias, consistency)
- Advanced risk scoring with contextual awareness
- Adaptive safety thresholds based on domain and user context
- Real-time monitoring and alerting
- Comprehensive audit trails with provenance tracking
- Token-level and sequence-level safety validation
- Integration hooks for external safety systems
- Performance optimization with caching
- Anomaly detection and pattern recognition
- Content policy enforcement
- Explainable safety decisions

This module is dependency-light with fallback implementations while integrating
with VULCAN safety validators and external systems when available.

Input/Output shapes:
- List[Token] where Token is int (id) or str (piece)
- List[Dict] with optional fields: token, prob, logit, id, score, etc.
Returns same shape with validated tokens.
"""

import hashlib
import json
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import (Any, Deque, Dict, List, Optional, Tuple, Union)

Token = Union[int, str]
Candidate = Union[Token, Dict[str, Any]]


# ================================ Enums and Constants ================================ #


class RiskLevel(Enum):
    """Risk severity levels"""

    SAFE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class ValidationCategory(Enum):
    """Categories of safety validation"""

    TOXICITY = "toxicity"
    HALLUCINATION = "hallucination"
    PROMPT_INJECTION = "prompt_injection"
    PII = "pii"
    BIAS = "bias"
    CONSISTENCY = "consistency"
    PROFANITY = "profanity"
    VIOLENCE = "violence"
    HATE_SPEECH = "hate_speech"
    SEXUAL_CONTENT = "sexual_content"
    MEDICAL_HARM = "medical_harm"
    LEGAL_VIOLATION = "legal_violation"


# ================================ Fallback Validators ================================ #

try:
    # Prefer repository-provided LLM validators if available
    from src.vulcan.safety.llm_validators import (BiasValidator,
                                                  EnhancedSafetyValidator,
                                                  HallucinationValidator,
                                                  PIIValidator,
                                                  PromptInjectionValidator,
                                                  ToxicityValidator)
except Exception:
    # Comprehensive heuristic validators with enhanced detection

    class ToxicityValidator:  # type: ignore
        """Enhanced toxicity detection with severity levels"""

        def __init__(self) -> None:
            self._severe_words = {
                "kill",
                "murder",
                "genocide",
                "torture",
                "rape",
                "assault",
            }
            self._moderate_words = {
                "hate",
                "stupid",
                "idiot",
                "dumb",
                "moron",
                "loser",
                "worthless",
                "disgusting",
                "terrible",
                "awful",
            }
            self._mild_words = {"annoying", "boring", "lame", "silly"}

        def check(self, token: Any, context: Any) -> bool:
            s = str(token).lower()
            # Check severe first
            if any(w in s for w in self._severe_words):
                return False
            # Context-aware moderate check
            if any(w in s for w in self._moderate_words):
                # Allow in educational/analytical contexts
                ctx_str = str(context).lower() if context else ""
                if not any(
                    marker in ctx_str
                    for marker in ["discuss", "analyze", "explain", "research"]
                ):
                    return False
            return True

        def get_severity(self, token: Any) -> RiskLevel:
            s = str(token).lower()
            if any(w in s for w in self._severe_words):
                return RiskLevel.CRITICAL
            if any(w in s for w in self._moderate_words):
                return RiskLevel.MEDIUM
            if any(w in s for w in self._mild_words):
                return RiskLevel.LOW
            return RiskLevel.SAFE

        def get_safe_alternative(self, token: Any, context: Any) -> Any:
            severity = self.get_severity(token)
            if severity == RiskLevel.CRITICAL:
                return "[CONTENT_FILTERED]"
            return "[MODERATED]"

    class HallucinationValidator:  # type: ignore
        """Enhanced hallucination detection with confidence scoring"""

        def __init__(self) -> None:
            self._fact_patterns = [
                r"\d{4}\s*-\s*\d{4}",  # Date ranges
                r"\d+\s*(million|billion|trillion)",  # Large numbers
                r"\d+(\.\d+)?%",  # Percentages
            ]
            self._confidence_keywords = {
                "certainly",
                "definitely",
                "absolutely",
                "proven",
                "fact",
            }

        def check(self, token: Any, context: Any) -> bool:
            # Without knowledge base, check for suspicious patterns
            s = str(token).lower()

            # Flag high-confidence claims without context
            if any(kw in s for kw in self._confidence_keywords):
                if not self._has_supporting_context(s, context):
                    return False

            # Check for specific fact patterns
            for pattern in self._fact_patterns:
                if re.search(pattern, s):
                    # Verify against context if available
                    if not self._verify_fact(s, context):
                        return False

            return True

        def _has_supporting_context(self, text: str, context: Any) -> bool:
            if not context or not isinstance(context, dict):
                return False
            ctx_text = str(context.get("retrieved_knowledge", "")).lower()
            return len(ctx_text) > 50  # Has substantial context

        def _verify_fact(self, text: str, context: Any) -> bool:
            # Placeholder for fact verification logic
            return True

        def get_safe_alternative(self, token: Any, context: Any) -> Any:
            return "[CLAIM_UNVERIFIED]"

    class PromptInjectionValidator:  # type: ignore
        """Enhanced prompt injection detection with pattern evolution"""

        def __init__(self) -> None:
            self._patterns = [
                # Direct instruction override
                r"(?i)\bignore\s+(all\s+)?(previous|prior|above)\s+instructions?\b",
                r"(?i)\boverride\s+(system|instructions?)\b",
                r"(?i)\bdisregard\s+(all\s+)?(above|previous|prior)\b",
                r"(?i)\bforget\s+(all\s+)?(previous|prior)\s+(instructions?|prompts?)\b",
                # Role manipulation
                r"(?i)\bact\s+as\s+(system|admin|root|developer)\b",
                r"(?i)\byou\s+are\s+now\s+(a|an)\s+(admin|developer|root)\b",
                r"(?i)^system\s*:",
                r"(?i)^admin\s*:",
                # Delimiter injection
                r"(?i)```\s*system",
                r"(?i)<\s*system\s*>",
                r"(?i)\[\s*system\s*\]",
                # Special tokens
                r"<\|.*?\|>",
                r"\[INST\]",
                r"\[/INST\]",
                # Encoding tricks
                r"(?i)base64\s*:",
                r"(?i)hex\s*:",
                r"(?i)unicode\s*:",
            ]
            self._rx = [re.compile(p) for p in self._patterns]
            self._sequence_memory: Deque[str] = deque(maxlen=5)

        def check(self, token: Any, context: Any) -> bool:
            s = str(token)

            # Pattern matching
            if any(rx.search(s) for rx in self._rx):
                return False

            # Sequence-based detection
            self._sequence_memory.append(s.lower())
            sequence = " ".join(self._sequence_memory)
            if any(rx.search(sequence) for rx in self._rx):
                return False

            # Check for unusual repetition (possible injection attempt)
            if self._detect_injection_pattern(sequence):
                return False

            return True

        def _detect_injection_pattern(self, sequence: str) -> bool:
            """Detect injection patterns in token sequence"""
            # Look for repeated instruction keywords
            instruction_keywords = [
                "ignore",
                "override",
                "disregard",
                "system",
                "admin",
            ]
            count = sum(1 for kw in instruction_keywords if kw in sequence)
            return count >= 2

        def get_safe_alternative(self, token: Any, context: Any) -> Any:
            return "[INJECTION_BLOCKED]"

    class PIIValidator:  # type: ignore
        """Personal Identifiable Information detection"""

        def __init__(self) -> None:
            self._patterns = {
                "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                "phone": r"\b(\+\d{1,3}[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b",
                "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
                "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
                "ip_address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
            }
            self._compiled = {k: re.compile(p) for k, p in self._patterns.items()}

        def check(self, token: Any, context: Any) -> bool:
            s = str(token)
            for pii_type, pattern in self._compiled.items():
                if pattern.search(s):
                    # Check if context explicitly allows PII
                    if not self._is_pii_allowed(context, pii_type):
                        return False
            return True

        def _is_pii_allowed(self, context: Any, pii_type: str) -> bool:
            if not context or not isinstance(context, dict):
                return False
            allowed = context.get("allowed_pii", set())
            return pii_type in allowed

        def detect_pii_types(self, token: Any) -> List[str]:
            """Return list of PII types detected"""
            s = str(token)
            detected = []
            for pii_type, pattern in self._compiled.items():
                if pattern.search(s):
                    detected.append(pii_type)
            return detected

        def get_safe_alternative(self, token: Any, context: Any) -> Any:
            pii_types = self.detect_pii_types(token)
            if pii_types:
                return f"[PII_{pii_types[0].upper()}_REDACTED]"
            return "[PII_REDACTED]"

    class BiasValidator:  # type: ignore
        """Bias and stereotype detection"""

        def __init__(self) -> None:
            self._protected_attributes = {
                "race",
                "ethnicity",
                "gender",
                "age",
                "religion",
                "disability",
                "nationality",
                "sexual_orientation",
            }
            self._stereotype_patterns = [
                r"(?i)\ball\s+(men|women|blacks|whites|asians|muslims|christians|jews)\s+(are|should|must|always)",
                r"(?i)\b(men|women|blacks|whites|asians)\s+(can't|cannot|don't|never)",
                r"(?i)\btypical\s+(man|woman|black|white|asian|muslim|christian|jew)",
            ]
            self._bias_rx = [re.compile(p) for p in self._stereotype_patterns]

        def check(self, token: Any, context: Any) -> bool:
            s = str(token)

            # Check for stereotype patterns
            if any(rx.search(s) for rx in self._bias_rx):
                # Allow in educational context
                if not self._is_educational_context(context):
                    return False

            return True

        def _is_educational_context(self, context: Any) -> bool:
            if not context or not isinstance(context, dict):
                return False
            ctx_str = str(context).lower()
            return any(
                marker in ctx_str
                for marker in [
                    "discuss",
                    "analyze",
                    "study",
                    "research",
                    "examine",
                    "critique",
                ]
            )

        def get_safe_alternative(self, token: Any, context: Any) -> Any:
            return "[BIAS_FILTERED]"

    class EnhancedSafetyValidator:  # type: ignore
        """Orchestrates all validators with priority and weighting"""

        def __init__(self) -> None:
            self.validators = [
                ToxicityValidator(),
                HallucinationValidator(),
                PromptInjectionValidator(),
                PIIValidator(),
                BiasValidator(),
            ]
            self.validator_weights = {
                "ToxicityValidator": 1.0,
                "PromptInjectionValidator": 1.0,
                "PIIValidator": 0.9,
                "BiasValidator": 0.7,
                "HallucinationValidator": 0.6,
            }

        def validate_generation(
            self, token: Any, context: Any, world_model: Any = None
        ) -> Any:
            out = token
            violations = []

            # Run all validators
            for v in self.validators:
                try:
                    if not v.check(out, context):
                        violations.append(type(v).__name__)
                        out = v.get_safe_alternative(out, context)
                except Exception:
                    # Log but continue
                    continue

            # World model validation
            if world_model:
                out = self._world_model_check(out, context, world_model, violations)

            return out

        def _world_model_check(
            self, token: Any, context: Any, world_model: Any, violations: List[str]
        ) -> Any:
            """Apply world model consistency checks"""
            if hasattr(world_model, "validate_generation"):
                try:
                    if not world_model.validate_generation(token, context):
                        if hasattr(world_model, "suggest_correction"):
                            return world_model.suggest_correction(token, context)
                except Exception:
                    pass
            return token


# ================================ Data Structures ================================ #


@dataclass
class SafetyEvent:
    """Represents a safety-related event during validation"""

    kind: str
    category: ValidationCategory
    detail: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    severity: RiskLevel = RiskLevel.LOW
    validator: Optional[str] = None


@dataclass
class RiskAssessment:
    """Comprehensive risk assessment for a token"""

    token: Any
    risk: float
    risk_level: RiskLevel
    reasons: List[str]
    violations: Dict[ValidationCategory, float] = field(default_factory=dict)
    confidence: float = 1.0
    context_factor: float = 1.0


@dataclass
class SafetyMetrics:
    """Aggregated safety metrics for monitoring"""

    total_processed: int = 0
    total_filtered: int = 0
    total_modified: int = 0
    risk_distribution: Dict[RiskLevel, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    category_violations: Dict[ValidationCategory, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    avg_risk_score: float = 0.0
    last_updated: float = field(default_factory=time.time)


# ================================ SafeGeneration ================================ #


class SafeGeneration:
    """
    Production-ready multi-layered safety system for token generation.

    Features:
    - Multi-tier validation pipeline
    - Contextual risk assessment
    - Adaptive safety thresholds
    - Real-time monitoring and metrics
    - Comprehensive audit trails
    - Sequence-level validation
    - External system integration
    - Performance optimization with caching

    Usage:
        sg = SafeGeneration(
            world_model=wm,
            observability=obs,
            audit=audit,
            policy={"mode": "keep_safe", "max_k": 3}
        )
        safe = sg.filter(candidates, context={"prompt": "...", "domain": "medical"}, top_k=1)
    """

    def __init__(
        self,
        validators: Optional[List[Any]] = None,
        world_model: Optional[Any] = None,
        observability: Optional[Any] = None,
        audit: Optional[Any] = None,
        policy: Optional[Dict[str, Any]] = None,
        fallback_safe_token: Token = 0,
        vocab: Optional[Any] = None,
        enable_caching: bool = True,
        cache_size: int = 1000,
    ) -> None:
        # Build validator pipeline
        self._enhanced = EnhancedSafetyValidator()
        self.validators = (
            validators
            or getattr(self._enhanced, "validators", [)]
            or [
                ToxicityValidator(),
                HallucinationValidator(),
                PromptInjectionValidator(),
                PIIValidator(),
                BiasValidator(),
            ]
        )

        # External integrations
        self.world_model = world_model
        self.observability = observability
        self.audit = audit
        self.vocab = vocab

        # Policy configuration with advanced options
        self.policy = {
            "mode": "first_safe",  # first_safe | keep_safe | ranked
            "max_k": 1,
            "block_on_high_risk": True,
            "high_risk_threshold": 0.9,
            "medium_risk_threshold": 0.6,
            "low_risk_threshold": 0.3,
            "replacement_strategy": "redact",  # redact | suggest | eos | filter
            "enable_adaptive_thresholds": True,
            "context_aware_scoring": True,
            "sequence_validation": True,
            "anomaly_detection": True,
            "allow_explanation": True,
        }
        if isinstance(policy, dict):
            self.policy.update(policy)

        self.fallback_safe_token = fallback_safe_token

        # Internal state
        self.last_events: List[SafetyEvent] = []
        self.metrics = SafetyMetrics()
        self._context_history: Deque[Dict[str, Any]] = deque(maxlen=100)

        # Caching for performance
        self.enable_caching = enable_caching
        self._cache: Dict[str, Tuple[Token, float]] = {}
        self._cache_size = cache_size

        # Adaptive threshold learning
        self._threshold_history: Deque[Tuple[float, str]] = deque(maxlen=500)

    # ================================ Public API ================================ #

    def filter(
        self,
        candidates: List[Candidate],
        context: Optional[Dict[str, Any]] = None,
        world_model: Optional[Any] = None,
        top_k: Optional[int] = None,
    ) -> List[Candidate]:
        """
        Apply comprehensive safety validation to candidate tokens.

        Args:
            candidates: List of token candidates (int, str, or dict)
            context: Optional context dict with prompt, domain, user info, etc.
            world_model: Optional world model for consistency checks
            top_k: Number of candidates to return (overrides policy)

        Returns:
            Filtered and validated candidate list preserving input shape
        """
        start_time = time.time()
        context = context or {}
        wm = world_model or self.world_model
        self.last_events = []

        if not candidates:
            return candidates

        # Normalize candidates to uniform dict format
        norm, shape = self._normalize(candidates)

        # Update context history for adaptive learning
        if self.policy.get("enable_adaptive_thresholds"):
            self._context_history.append(
                {
                    "domain": context.get("domain"),
                    "user_type": context.get("user_type"),
                    "timestamp": start_time,
                }
            )

        # Validate and score each candidate
        assessments: List[RiskAssessment] = []
        repaired: List[Any] = []

        for idx, c in enumerate(norm):
            tok = c.get("token")

            # Check cache first
            cache_key = self._get_cache_key(tok, context)
            if self.enable_caching and cache_key in self._cache:
                cached_tok, cached_risk = self._cache[cache_key]
                safe_tok = cached_tok
                risk = cached_risk
                reasons = ["cached"]
            else:
                # Full validation
                safe_tok, reasons, violations = self._validate_and_repair(
                    tok, context, wm
                )
                risk = self._risk_score(safe_tok, reasons, violations, context)

                # Update cache
                if self.enable_caching:
                    self._update_cache(cache_key, safe_tok, risk)

            # Create risk assessment
            risk_level = self._risk_to_level(risk)
            assessment = RiskAssessment(
                token=safe_tok,
                risk=risk,
                risk_level=risk_level,
                reasons=reasons,
                violations=violations if "violations" in locals() else {},
                confidence=self._compute_confidence(reasons, context),
                context_factor=self._compute_context_factor(context),
            )

            assessments.append(assessment)
            repaired.append(safe_tok)

        # Ranking and selection strategy
        mode = self.policy["mode"]
        if top_k is None:
            top_k = int(self.policy.get("max_k", 1))

        # Pair assessments with original indices
        indexed = list(enumerate(assessments))

        # Sort by risk score (ascending), then original index for stability
        indexed.sort(key=lambda t: (t[1].risk, t[0]))

        # Select candidates based on mode
        selected_indices = self._select_candidates(indexed, mode, top_k)

        # Handle high-risk blocking
        best_idx = selected_indices[0] if selected_indices else 0
        best_ra = assessments[best_idx]

        if self.policy["block_on_high_risk"]:
            if best_ra.risk >= float(self.policy["high_risk_threshold"]):
                replacement = self._replacement_token(repaired[best_idx], context, wm)
                self._emit_event(
                    "block_high_risk",
                    ValidationCategory.TOXICITY,
                    {
                        "original": str(repaired[best_idx]),
                        "replacement": str(replacement),
                        "risk": best_ra.risk,
                        "reasons": best_ra.reasons,
                    },
                    RiskLevel.HIGH,
                )
                repaired[best_idx] = replacement

        # Sequence-level validation if enabled
        if self.policy.get("sequence_validation"):
            repaired = self._validate_sequence_coherence(repaired, context, wm)

        # Build output preserving input shape
        out_norm: List[Dict[str, Any]] = []
        for i in selected_indices:
            d = dict(norm[i])
            d["token"] = repaired[i]

            # Optionally attach risk metadata
            if self.policy.get("allow_explanation"):
                d["safety_assessment"] = {
                    "risk": assessments[i].risk,
                    "risk_level": assessments[i].risk_level.name,
                    "reasons": assessments[i].reasons,
                    "confidence": assessments[i].confidence,
                }

            out_norm.append(d)

        out = self._denormalize(out_norm, shape)

        # Update metrics
        self._update_metrics(assessments, selected_indices)

        # Emit telemetry
        processing_time = time.time() - start_time
        self._obs(
            "safe_generation.filter",
            {
                "mode": mode,
                "selected": len(out_norm),
                "original": len(norm),
                "best_risk": best_ra.risk if best_ra else None,
                "processing_time_ms": processing_time * 1000,
                "cache_hit_rate": self._compute_cache_hit_rate(),
            },
        )

        self._audit(
            "safe_generation.filter",
            {
                "assessments": [self._assessment_to_dict(a) for a in assessments],
                "selected_indices": selected_indices,
                "context_hash": self._hash_context(context),
                "timestamp": time.time(),
            },
        )

        return out

    def validate_token(
        self,
        token: Token,
        context: Optional[Dict[str, Any]] = None,
        world_model: Optional[Any] = None,
    ) -> Token:
        """
        Validate a single token and return a safe token (possibly replaced).

        Args:
            token: The token to validate
            context: Optional context information
            world_model: Optional world model for consistency checks

        Returns:
            Validated/repaired token
        """
        context = context or {}
        wm = world_model or self.world_model
        safe_tok, _, _ = self._validate_and_repair(token, context, wm)
        return safe_tok

    def validate_sequence(
        self,
        tokens: List[Token],
        context: Optional[Dict[str, Any]] = None,
        world_model: Optional[Any] = None,
    ) -> Union[bool, List[Token]]:
        """
        Sequence-level validation with coherence checking.

        Args:
            tokens: List of tokens to validate
            context: Optional context information
            world_model: Optional world model

        Returns:
            - True if sequence is safe
            - False to block emission
            - List[Token] with corrections
        """
        context = context or {}
        wm = world_model or self.world_model

        # Token-level validation first
        repaired: List[Token] = []
        changed = False
        for t in tokens:
            rt = self.validate_token(t, context, wm)
            changed = changed or (rt != t)
            repaired.append(rt)

        # Sequence-level checks
        if self.policy.get("sequence_validation"):
            # Check for injection across token boundaries
            sequence_str = " ".join(str(t) for t in repaired)
            if self._detect_sequence_injection(sequence_str):
                self._emit_event(
                    "sequence_injection_detected",
                    ValidationCategory.PROMPT_INJECTION,
                    {"sequence": sequence_str[:100]},
                    RiskLevel.HIGH,
                )
                return False

            # Check for accumulated toxicity
            if self._detect_accumulated_toxicity(repaired):
                self._emit_event(
                    "accumulated_toxicity",
                    ValidationCategory.TOXICITY,
                    {"length": len(repaired)},
                    RiskLevel.MEDIUM,
                )
                return False

        # World model sequence validation
        if wm and hasattr(wm, "validate_sequence"):
            try:
                ok = wm.validate_sequence(repaired, context)
                if ok is True:
                    return repaired if changed else True
                if ok is False:
                    return False
                if isinstance(ok, list):
                    return ok
            except Exception:
                pass

        return repaired if changed else True

    def get_metrics(self) -> SafetyMetrics:
        """Get current safety metrics for monitoring"""
        return self.metrics

    def reset_metrics(self) -> None:
        """Reset accumulated metrics"""
        self.metrics = SafetyMetrics()

    def update_policy(self, policy_updates: Dict[str, Any]) -> None:
        """Dynamically update safety policy"""
        self.policy.update(policy_updates)
        self._emit_event(
            "policy_updated",
            ValidationCategory.CONSISTENCY,
            {"updates": list(policy_updates.keys())},
            RiskLevel.SAFE,
        )

    def export_audit_log(self) -> List[Dict[str, Any]]:
        """Export comprehensive audit log"""
        return [
            {
                "event": e.kind,
                "category": e.category.value,
                "severity": e.severity.name,
                "timestamp": e.timestamp,
                "detail": e.detail,
                "validator": e.validator,
            }
            for e in self.last_events
        ]

    # ================================ Internal Validation ================================ #

    def _validate_and_repair(
        self, token: Token, context: Dict[str, Any], world_model: Optional[Any]
    ) -> Tuple[Token, List[str], Dict[ValidationCategory, float]]:
        """
        Run through validator pipeline and world model checks.

        Returns:
            (safe_token, reasons, violation_scores)
        """
        out = token
        reasons: List[str] = []
        violations: Dict[ValidationCategory, float] = {}

        # Run each validator
        for v in self.validators:
            validator_name = type(v).__name__
            try:
                safe = v.check(out, context)
                if not safe:
                    replacement = v.get_safe_alternative(out, context)
                    reasons.append(validator_name)

                    # Track violation category and severity
                    category = self._validator_to_category(validator_name)
                    severity = getattr(v, "get_severity", lambda t: RiskLevel.MEDIUM)(
                        out
                    )
                    violations[category] = self._risk_level_to_score(severity)

                    self._emit_event(
                        "validator_triggered",
                        category,
                        {
                            "validator": validator_name,
                            "original": str(out),
                            "replacement": str(replacement),
                        },
                        severity,
                    )

                    out = replacement
            except Exception as e:
                # Log error but continue pipeline
                self._emit_event(
                    "validator_error",
                    ValidationCategory.CONSISTENCY,
                    {"validator": validator_name, "error": str(e)},
                    RiskLevel.LOW,
                )
                continue

        # Enhanced validator final pass
        if isinstance(self._enhanced, EnhancedSafetyValidator):
            if hasattr(self._enhanced, "validate_generation"):
                try:
                    final = self._enhanced.validate_generation(
                        out, context, world_model
                    )
                    if final != out:
                        reasons.append("EnhancedSafetyValidator")
                        violations[ValidationCategory.CONSISTENCY] = 0.5
                    out = final
                except Exception:
                    pass

        # World model validation
        if world_model:
            if hasattr(world_model, "validate_generation") and hasattr(
                world_model, "suggest_correction"
            ):
                try:
                    if not world_model.validate_generation(out, context):
                        reasons.append("WorldModelCorrection")
                        violations[ValidationCategory.CONSISTENCY] = 0.4
                        out = world_model.suggest_correction(out, context)
                except Exception:
                    pass

        # Additional heuristic checks
        if isinstance(out, str):
            # Length clamping
            if len(out) > 400:
                reasons.append("LengthClamp")
                out = out[:400] + "…"

            # Repetition detection
            if self._detect_repetition(out):
                reasons.append("RepetitionDetected")
                violations[ValidationCategory.CONSISTENCY] = 0.3

        return out, reasons, violations

    def _risk_score(
        self,
        token: Token,
        reasons: List[str],
        violations: Dict[ValidationCategory, float],
        context: Dict[str, Any],
    ) -> float:
        """
        Compute comprehensive risk score [0, 1] with contextual awareness.
        """
        score = 0.0

        # Base score from validator violations
        for reason in reasons:
            if reason == "cached":
                continue
            elif reason in ("ToxicityValidator", "PromptInjectionValidator"):
                score += 0.6
            elif reason in ("PIIValidator", "BiasValidator"):
                score += 0.5
            elif reason in ("HallucinationValidator", "WorldModelCorrection"):
                score += 0.4
            elif reason == "EnhancedSafetyValidator":
                score += 0.2
            elif reason in ("LengthClamp", "RepetitionDetected"):
                score += 0.1

        # Add category-specific violation scores
        for category, cat_score in violations.items():
            score += cat_score * 0.3  # Weight category scores

        # Context-aware adjustment
        if self.policy.get("context_aware_scoring"):
            context_factor = self._compute_context_factor(context)
            score *= context_factor

        # Pattern-based risk enhancement
        s = str(token)

        # Injection patterns
        injection_patterns = [
            r"(?i)\bignore\s+previous\s+instructions\b",
            r"(?i)\bdisregard\s+all\s+above\b",
            r"(?i)^system\s*:",
            r"<\|.*?\|>",
        ]
        if any(re.search(p, s) for p in injection_patterns):
            score = max(score, 0.8)

        # Violence/harm patterns
        violence_patterns = [
            r"(?i)\b(kill|murder|attack|harm|hurt)\s+(people|person|someone)",
            r"(?i)\b(how|ways)\s+to\s+(kill|harm|hurt)",
        ]
        if any(re.search(p, s) for p in violence_patterns):
            score = max(score, 0.9)

        # Anomaly detection
        if self.policy.get("anomaly_detection"):
            anomaly_score = self._detect_anomaly(token, context)
            score = max(score, anomaly_score)

        # Adaptive threshold adjustment
        if self.policy.get("enable_adaptive_thresholds"):
            score = self._apply_adaptive_adjustment(score, context)

        # Clamp to [0, 1]
        return max(0.0, min(1.0, score))

    def _compute_confidence(self, reasons: List[str], context: Dict[str, Any]) -> float:
        """Compute confidence in the risk assessment"""
        if not reasons:
            return 1.0

        # More reasons = lower confidence (might be false positives)
        base_confidence = 1.0 / (1.0 + 0.1 * len(reasons))

        # Adjust based on context quality
        has_context = bool(context.get("retrieved_knowledge") or context.get("prompt"))
        context_boost = 1.1 if has_context else 0.9

        return min(1.0, base_confidence * context_boost)

    def _compute_context_factor(self, context: Dict[str, Any]) -> float:
        """Compute context-based risk adjustment factor"""
        factor = 1.0

        # Domain-specific adjustments
        domain = context.get("domain", "").lower()
        if domain in ("medical", "legal", "financial"):
            factor *= 1.2  # Higher scrutiny
        elif domain in ("creative", "entertainment"):
            factor *= 0.8  # More lenient

        # User type adjustments
        user_type = context.get("user_type", "").lower()
        if user_type in ("child", "minor"):
            factor *= 1.5  # Much stricter
        elif user_type == "researcher":
            factor *= 0.9  # Slightly more lenient

        # Intent-based adjustments
        intent = context.get("intent", "").lower()
        if intent in ("educational", "research", "analysis"):
            factor *= 0.85

        return factor

    def _detect_anomaly(self, token: Token, context: Dict[str, Any]) -> float:
        """Detect anomalous tokens based on statistical patterns"""
        s = str(token)
        anomaly_score = 0.0

        # Unusual length
        if len(s) > 200:
            anomaly_score += 0.2

        # Unusual character distribution
        if s:
            alpha_ratio = sum(c.isalpha() for c in s) / len(s)
            if alpha_ratio < 0.3 or alpha_ratio > 0.95:
                anomaly_score += 0.1

        # Excessive punctuation
        punct_count = sum(1 for c in s if not c.isalnum() and not c.isspace())
        if len(s) > 0 and punct_count / len(s) > 0.3:
            anomaly_score += 0.15

        # Unusual repetition
        if self._detect_repetition(s):
            anomaly_score += 0.2

        return min(1.0, anomaly_score)

    def _detect_repetition(self, text: str) -> bool:
        """Detect suspicious repetition patterns"""
        if len(text) < 10:
            return False

        # Check for character repetition
        for i in range(len(text) - 3):
            pattern = text[i : i + 3]
            if text.count(pattern) > 3:
                return True

        # Check for word repetition
        words = text.split()
        if len(words) >= 3:
            for i in range(len(words) - 2):
                if words[i] == words[i + 1] == words[i + 2]:
                    return True

        return False

    def _detect_sequence_injection(self, sequence: str) -> bool:
        """Detect injection attempts across token boundaries"""
        injection_keywords = [
            "ignore previous",
            "disregard above",
            "override system",
            "act as admin",
            "forget instructions",
        ]
        sequence_lower = sequence.lower()
        return any(keyword in sequence_lower for keyword in injection_keywords)

    def _detect_accumulated_toxicity(self, tokens: List[Token]) -> bool:
        """Detect toxicity that accumulates across multiple tokens"""
        # Join tokens and check cumulative toxicity
        combined = " ".join(str(t) for t in tokens)

        # Use toxicity validator on combined sequence
        for v in self.validators:
            if isinstance(v, ToxicityValidator):
                if not v.check(combined, {}):
                    return True

        return False

    def _validate_sequence_coherence(
        self, tokens: List[Token], context: Dict[str, Any], world_model: Optional[Any]
    ) -> List[Token]:
        """Validate and potentially repair sequence-level coherence"""
        if len(tokens) < 2:
            return tokens

        # Check for contradictions or incoherence
        if world_model and hasattr(world_model, "check_coherence"):
            try:
                is_coherent = world_model.check_coherence(tokens, context)
                if not is_coherent and hasattr(world_model, "repair_sequence"):
                    repaired = world_model.repair_sequence(tokens, context)
                    if repaired:
                        self._emit_event(
                            "sequence_repaired",
                            ValidationCategory.CONSISTENCY,
                            {
                                "original_length": len(tokens),
                                "repaired_length": len(repaired),
                            },
                            RiskLevel.LOW,
                        )
                        return repaired
            except Exception:
                pass

        return tokens

    # ================================ Selection & Replacement ================================ #

    def _select_candidates(
        self, indexed: List[Tuple[int, RiskAssessment]], mode: str, top_k: int
    ) -> List[int]:
        """Select candidates based on policy mode"""
        if mode == "first_safe":
            # Choose the first candidate under medium threshold
            med = float(self.policy["medium_risk_threshold"])
            for i, ra in indexed:
                if ra.risk <= med:
                    return [i]
            # If none safe, return safest
            return [indexed[0][0]]

        elif mode == "keep_safe":
            # Keep up to top_k under medium threshold
            med = float(self.policy["medium_risk_threshold"])
            safe_list = [i for i, ra in indexed if ra.risk <= med]
            if safe_list:
                return safe_list[: max(1, top_k)]
            else:
                return [indexed[0][0]]

        elif mode == "ranked":
            # Return top_k by lowest risk, regardless of threshold
            return [i for i, _ in indexed[: max(1, top_k)]]

        else:
            # Default to first_safe
            return self._select_candidates(indexed, "first_safe", top_k)

    def _replacement_token(
        self, original: Token, context: Dict[str, Any], world_model: Optional[Any]
    ) -> Token:
        """Get replacement token based on strategy"""
        strat = self.policy.get("replacement_strategy", "redact")

        if strat == "eos":
            return self.fallback_safe_token

        elif strat == "suggest" and world_model:
            if hasattr(world_model, "suggest_correction"):
                try:
                    return world_model.suggest_correction(original, context)
                except Exception:
                    pass

        elif strat == "filter":
            # Return empty/neutral token
            return "" if isinstance(original, str) else self.fallback_safe_token

        # Default: redact
        if isinstance(original, int):
            return self.fallback_safe_token
        return "[SAFE]"

    # ================================ Utilities ================================ #

    def _risk_to_level(self, risk: float) -> RiskLevel:
        """Convert numeric risk score to RiskLevel enum"""
        if risk >= self.policy["high_risk_threshold"]:
            return RiskLevel.CRITICAL
        elif risk >= self.policy["medium_risk_threshold"]:
            return RiskLevel.HIGH
        elif risk >= self.policy["low_risk_threshold"]:
            return RiskLevel.MEDIUM
        elif risk > 0:
            return RiskLevel.LOW
        else:
            return RiskLevel.SAFE

    def _risk_level_to_score(self, level: RiskLevel) -> float:
        """Convert RiskLevel to numeric score"""
        mapping = {
            RiskLevel.SAFE: 0.0,
            RiskLevel.LOW: 0.25,
            RiskLevel.MEDIUM: 0.5,
            RiskLevel.HIGH: 0.75,
            RiskLevel.CRITICAL: 1.0,
        }
        return mapping.get(level, 0.5)

    def _validator_to_category(self, validator_name: str) -> ValidationCategory:
        """Map validator name to ValidationCategory"""
        mapping = {
            "ToxicityValidator": ValidationCategory.TOXICITY,
            "HallucinationValidator": ValidationCategory.HALLUCINATION,
            "PromptInjectionValidator": ValidationCategory.PROMPT_INJECTION,
            "PIIValidator": ValidationCategory.PII,
            "BiasValidator": ValidationCategory.BIAS,
        }
        return mapping.get(validator_name, ValidationCategory.CONSISTENCY)

    def _apply_adaptive_adjustment(
        self, score: float, context: Dict[str, Any]
    ) -> float:
        """Apply adaptive threshold learning"""
        domain = context.get("domain", "general")

        # Track score distribution per domain
        self._threshold_history.append((score, domain))

        # Adjust based on historical patterns (simple implementation)
        # In production, use more sophisticated ML-based adaptation
        domain_scores = [s for s, d in self._threshold_history if d == domain]
        if len(domain_scores) > 10:
            avg_domain_score = sum(domain_scores) / len(domain_scores)
            # If consistently high/low scores in this domain, adjust
            if avg_domain_score > 0.7:
                score *= 0.9  # Slightly more lenient
            elif avg_domain_score < 0.2:
                score *= 1.1  # Slightly stricter

        return score

    def _normalize(
        self, candidates: List[Candidate]
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Convert candidates to uniform dict format.

        Returns:
            (normalized_list, shape_hint)
            shape_hint: "token_list" or "dict_list"
        """
        norm: List[Dict[str, Any]] = []
        if not candidates:
            return norm, "token_list"

        if isinstance(candidates[0], dict):
            for d in candidates:
                token = d.get("token")
                if token is None and "id" in d:
                    token = d.get("id")
                norm.append({"token": token, **d})
            return norm, "dict_list"
        else:
            for t in candidates:
                norm.append({"token": t})
            return norm, "token_list"

    def _denormalize(self, norm: List[Dict[str, Any]], shape: str) -> List[Candidate]:
        """Convert normalized dicts back to original shape"""
        if shape == "dict_list":
            return norm
        else:
            return [d.get("token") for d in norm]

    # ================================ Caching ================================ #

    def _get_cache_key(self, token: Token, context: Dict[str, Any]) -> str:
        """Generate cache key from token and context"""
        context_str = json.dumps(
            {
                "domain": context.get("domain"),
                "user_type": context.get("user_type"),
            },
            sort_keys=True,
        )
        combined = f"{token}:{context_str}"
        return hashlib.md5(combined.encode(), usedforsecurity=False).hexdigest()

    def _update_cache(self, key: str, token: Token, risk: float) -> None:
        """Update cache with LRU eviction"""
        if len(self._cache) >= self._cache_size:
            # Simple LRU: remove oldest entry
            # In production, use proper LRU implementation
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[key] = (token, risk)

    def _compute_cache_hit_rate(self) -> float:
        """Compute cache hit rate for monitoring"""
        # Simplified; in production, track hits/misses explicitly
        return 0.0 if not self._cache else len(self._cache) / self._cache_size

    # ================================ Metrics & Monitoring ================================ #

    def _update_metrics(
        self, assessments: List[RiskAssessment], selected_indices: List[int]
    ) -> None:
        """Update aggregated metrics"""
        self.metrics.total_processed += len(assessments)

        for i, assessment in enumerate(assessments):
            # Track risk distribution
            self.metrics.risk_distribution[assessment.risk_level] += 1

            # Track category violations
            for category in assessment.violations.keys():
                self.metrics.category_violations[category] += 1

            # Track modifications
            if assessment.reasons and "cached" not in assessment.reasons:
                if i in selected_indices:
                    self.metrics.total_modified += 1
                else:
                    self.metrics.total_filtered += 1

        # Update average risk score
        if assessments:
            total_risk = sum(a.risk for a in assessments)
            self.metrics.avg_risk_score = total_risk / len(assessments)

        self.metrics.last_updated = time.time()

    def _assessment_to_dict(self, assessment: RiskAssessment) -> Dict[str, Any]:
        """Convert RiskAssessment to dict for serialization"""
        return {
            "token": str(assessment.token),
            "risk": assessment.risk,
            "risk_level": assessment.risk_level.name,
            "reasons": assessment.reasons,
            "violations": {k.value: v for k, v in assessment.violations.items()},
            "confidence": assessment.confidence,
            "context_factor": assessment.context_factor,
        }

    # ================================ Telemetry ================================ #

    def _emit_event(
        self,
        kind: str,
        category: ValidationCategory,
        detail: Dict[str, Any],
        severity: RiskLevel = RiskLevel.LOW,
    ) -> None:
        """Emit a safety event for monitoring"""
        evt = SafetyEvent(
            kind=kind,
            category=category,
            detail=detail,
            severity=severity,
            validator=detail.get("validator"),
        )
        self.last_events.append(evt)
        self._obs(f"safe_generation.{kind}", detail)
        self._audit(f"safe_generation.{kind}", detail)

    def _obs(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Send to observability system"""
        if not self.observability:
            return
        try:
            if hasattr(self.observability, "record"):
                self.observability.record(event_type, payload)
            elif hasattr(self.observability, "log"):
                self.observability.log(event_type, payload)
        except Exception:
            pass

    def _audit(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Send to audit system"""
        if not self.audit:
            return
        try:
            if hasattr(self.audit, "append"):
                self.audit.append({"event": event_type, **payload})
            elif hasattr(self.audit, "record"):
                self.audit.record(event_type, payload)
        except Exception:
            pass

    def _hash_context(self, context: Dict[str, Any]) -> str:
        """Create hash of context for audit trail"""
        context_str = json.dumps(context, sort_keys=True, default=str)
        return hashlib.sha256(context_str.encode()).hexdigest()[:16]
