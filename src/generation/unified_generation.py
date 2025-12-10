from __future__ import annotations

"""
Unified Generation - Advanced Multi-Strategy Reasoning Ensemble

A production-ready, extensible generation system that combines multiple reasoning
strategies into a unified, weighted ensemble for next-token generation.

Enhanced Features:
- Extended reasoning strategies: symbolic, probabilistic, causal, analogical, language,
  meta-cognitive, evolutionary, adversarial, hierarchical
- Dynamic weight adaptation based on context and performance
- Cross-module interaction modeling
- Ensemble uncertainty quantification
- Performance profiling and optimization
- Advanced normalization strategies (temperature scaling, calibration)
- Caching for efficiency
- Confidence-aware fusion
- Diversity-aware sampling
- Provenance tracking for full explainability

Design Philosophy:
- Duck-typed modules: works with any subset of modules in `reasoning_modules`
- Zero heavy dependencies: pure-Python, list-based math
- Robust fallbacks when modules are missing or return unexpected formats
- Returns rich candidate dicts usable by safety/rerankers/explainers

Module Interface (duck-typing; any subset will work):
- propose_candidates(hidden_state, context) -> List[Token | {token, score?, logit?, confidence?}]
- generate_candidates(hidden_state, context) -> same
- select_next_token(hidden_state, context) -> Token | List[...]
- generate(hidden_state, context?) -> Token | List[...]
- score_candidates(candidates, context) -> List[float]
- get_confidence() -> float

Additional keys in `reasoning_modules`:
- "context": object passed into module calls
- "weights": Dict[str, float] for per-module weighting
- "max_candidates": int to override default cap
- "temperature": float for temperature scaling
- "diversity_penalty": float to encourage diverse candidates
"""

import hashlib
import json
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

Token = Union[int, str]


# ================================ Enums and Configuration ================================ #


class FusionStrategy(Enum):
    """Strategy for combining module outputs"""

    WEIGHTED_SUM = "weighted_sum"
    PRODUCT = "product"
    MAX = "max"
    RANK_FUSION = "rank_fusion"
    BORDA_COUNT = "borda_count"


class NormalizationMethod(Enum):
    """Normalization approach for scores"""

    SOFTMAX = "softmax"
    MIN_MAX = "min_max"
    Z_SCORE = "z_score"
    RANK = "rank"


@dataclass
class UnifiedGenConfig:
    """Comprehensive configuration for unified generation"""

    max_candidates: int = 10
    default_module_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "language": 1.0,
            "symbolic": 1.0,
            "probabilistic": 1.0,
            "causal": 1.2,  # Causal reasoning slightly favored
            "analogical": 0.8,  # Creative but risky, downweighted
            "meta_cognitive": 1.1,  # Meta-reasoning boosted
            "evolutionary": 0.7,  # Experimental strategies
            "adversarial": 0.6,  # Challenge-based reasoning
            "hierarchical": 0.9,  # Structured reasoning
        }
    )
    min_prob_floor: float = 1e-9
    normalization_eps: float = 1e-9
    dedupe: bool = True
    attach_logits: bool = True
    fallback_vocab_size: int = 64

    # Advanced options
    fusion_strategy: FusionStrategy = FusionStrategy.WEIGHTED_SUM
    normalization_method: NormalizationMethod = NormalizationMethod.SOFTMAX
    temperature: float = 1.0
    diversity_penalty: float = 0.0
    enable_cross_module_interaction: bool = True
    enable_dynamic_weights: bool = True
    enable_confidence_scaling: bool = True
    enable_caching: bool = True
    cache_size: int = 1000
    min_module_agreement: int = 1  # Min modules that must propose a candidate


@dataclass
class CandidateMetadata:
    """Rich metadata for each candidate"""

    token: Any
    score: float
    prob: float
    logit: float
    rank: int
    provenance: List[Dict[str, Any]]
    confidence: float = 1.0
    diversity_score: float = 0.0
    module_agreement: int = 1
    uncertainty: float = 0.0


# ================================ UnifiedGeneration ================================ #


class UnifiedGeneration:
    """
    Advanced unified reasoning system combining multiple strategies.

    Features:
    - Multi-strategy ensemble with pluggable modules
    - Dynamic weight adaptation
    - Cross-module interaction modeling
    - Uncertainty quantification
    - Performance optimization with caching
    - Rich provenance tracking

    Usage:
        config = UnifiedGenConfig(
            fusion_strategy=FusionStrategy.WEIGHTED_SUM,
            enable_dynamic_weights=True,
            temperature=0.8
        )
        gen = UnifiedGeneration(config)

        candidates = gen.generate_candidates(
            hidden_state=hidden,
            reasoning_modules={
                "symbolic": symbolic_reasoner,
                "causal": causal_reasoner,
                "language": language_model,
                "context": context_dict,
                "weights": {"causal": 1.5},  # Override weight
            }
        )
    """

    def __init__(self, config: Optional[UnifiedGenConfig] = None) -> None:
        self.cfg = config or UnifiedGenConfig()

        # Performance tracking
        self._module_performance: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=100)
        )
        self._generation_history: deque = deque(maxlen=500)

        # Caching
        self._cache: Dict[str, List[Dict[str, Any]]] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    # ================================ Public API ================================ #

    def generate_candidates(
        self, hidden_state: Any, reasoning_modules: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Combine multiple reasoning strategies into unified candidate list.

        Args:
            hidden_state: Model-specific hidden representation
            reasoning_modules: Dict of module name -> module object, plus:
              - "context": external context object
              - "weights": Dict[str, float] per-module weights
              - "max_candidates": int to override config cap
              - "temperature": float for temperature scaling
              - "diversity_penalty": float for diversity

        Returns:
            List[Dict] candidates with comprehensive metadata:
            - token: The token (int or str)
            - score: Ensemble score (unnormalized)
            - prob: Normalized probability
            - logit: Approximate logit
            - rank: 1-based rank
            - provenance: List of module contributions
            - confidence: Ensemble confidence
            - diversity_score: Diversity metric
            - module_agreement: Number of modules proposing this token
            - uncertainty: Uncertainty estimate
        """
        start_time = time.time()

        if not isinstance(reasoning_modules, dict):
            reasoning_modules = {}

        # Extract parameters
        context = reasoning_modules.get("context")
        weights = self._merge_weights(reasoning_modules.get("weights") or {})
        max_k = int(reasoning_modules.get("max_candidates") or self.cfg.max_candidates)
        temperature = float(
            reasoning_modules.get("temperature") or self.cfg.temperature
        )
        diversity_penalty = float(
            reasoning_modules.get("diversity_penalty") or self.cfg.diversity_penalty
        )

        # Check cache
        cache_key = self._get_cache_key(hidden_state, context, weights)
        if self.cfg.enable_caching and cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key][:max_k]

        self._cache_misses += 1

        # Detect available modules
        module_names = self._detect_modules(reasoning_modules)

        if not module_names:
            # Fallback if no modules available
            result = self._fallback_candidates(max_k)
            if self.cfg.enable_caching:
                self._update_cache(cache_key, result)
            return result

        # Collect proposals from each module
        per_module_proposals: Dict[str, List[Dict[str, Any]]] = {}
        module_confidences: Dict[str, float] = {}

        for name in module_names:
            mod = reasoning_modules.get(name)
            module_start = time.time()

            props = self._propose_from_module(name, mod, hidden_state, context)
            module_time = time.time() - module_start

            if props:
                # Normalize module-local scores/logits to probabilities
                per_module_proposals[name] = self._normalize_module_proposals(props)

                # Extract module confidence if available
                confidence = self._get_module_confidence(mod)
                module_confidences[name] = confidence

                # Track performance
                self._module_performance[name].append(
                    {
                        "time": module_time,
                        "num_proposals": len(props),
                        "timestamp": time.time(),
                    }
                )

        # If no proposals from any module, fallback
        if not per_module_proposals:
            result = self._fallback_candidates(max_k)
            if self.cfg.enable_caching:
                self._update_cache(cache_key, result)
            return result

        # Dynamic weight adaptation
        if self.cfg.enable_dynamic_weights:
            weights = self._adapt_weights(
                weights, per_module_proposals, module_confidences, context
            )

        # Cross-module interaction modeling
        if self.cfg.enable_cross_module_interaction:
            per_module_proposals = self._model_cross_module_interactions(
                per_module_proposals, weights
            )

        # Fuse proposals from all modules
        fused = self._fuse_proposals(
            per_module_proposals,
            weights,
            module_confidences,
            fusion_strategy=self.cfg.fusion_strategy,
        )

        # Apply diversity penalty if requested
        if diversity_penalty > 0:
            fused = self._apply_diversity_penalty(fused, diversity_penalty)

        # Rank candidates
        fused_sorted = sorted(fused.values(), key=lambda d: d["score"], reverse=True)
        fused_top = fused_sorted[:max_k]

        # Finalize probabilities and logits with temperature scaling
        self._finalize_probs_and_logits(fused_top, temperature)

        # Compute ensemble metrics
        for d in fused_top:
            d["confidence"] = self._compute_ensemble_confidence(d, module_confidences)
            d["uncertainty"] = self._compute_ensemble_uncertainty(d)
            d["diversity_score"] = self._compute_diversity_score(d, fused_top)

        # Add 1-based rank
        for i, d in enumerate(fused_top, start=1):
            d["rank"] = i

        # Update generation history
        self._generation_history.append(
            {
                "num_modules": len(module_names),
                "num_candidates": len(fused_top),
                "top_score": fused_top[0]["score"] if fused_top else 0,
                "processing_time": time.time() - start_time,
                "timestamp": time.time(),
            }
        )

        # Cache result
        if self.cfg.enable_caching:
            self._update_cache(cache_key, fused_top)

        return fused_top

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for monitoring"""
        stats = {
            "cache_hit_rate": self._cache_hits
            / max(1, self._cache_hits + self._cache_misses),
            "cache_size": len(self._cache),
            "module_stats": {},
            "recent_generations": len(self._generation_history),
        }

        # Per-module stats
        for name, history in self._module_performance.items():
            if history:
                times = [h["time"] for h in history]
                stats["module_stats"][name] = {
                    "avg_time_ms": sum(times) / len(times) * 1000,
                    "min_time_ms": min(times) * 1000,
                    "max_time_ms": max(times) * 1000,
                    "num_calls": len(history),
                }

        # Overall generation stats
        if self._generation_history:
            avg_time = sum(
                h["processing_time"] for h in self._generation_history
            ) / len(self._generation_history)
            stats["avg_generation_time_ms"] = avg_time * 1000

        return stats

    def reset_cache(self) -> None:
        """Clear the cache"""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    # ================================ Module Handling ================================ #

    def _detect_modules(self, modules: Dict[str, Any]) -> List[str]:
        """Detect available reasoning modules"""
        # Known module keys
        known = [
            "symbolic",
            "probabilistic",
            "causal",
            "analogical",
            "language",
            "meta_cognitive",
            "evolutionary",
            "adversarial",
            "hierarchical",
        ]
        present = [
            k for k in known if k in modules and self._looks_like_module(modules[k])
        ]

        # Include any additional keys that look like modules
        reserved = (
            "context",
            "weights",
            "max_candidates",
            "temperature",
            "diversity_penalty",
        )
        for k, v in modules.items():
            if k not in reserved and k not in present and self._looks_like_module(v):
                present.append(k)

        return present

    def _looks_like_module(self, obj: Any) -> bool:
        """Check if object has module-like interface"""
        if obj is None:
            return False

        # Any of these capabilities qualify
        return any(
            hasattr(obj, m)
            for m in (
                "propose_candidates",
                "generate_candidates",
                "select_next_token",
                "generate",
                "score_candidates",
            )
        )

    def _propose_from_module(
        self, name: str, module: Any, hidden_state: Any, context: Any
    ) -> List[Dict[str, Any]]:
        """
        Call module using best-effort duck-typing.
        Returns normalized list of dicts: {token, score?, logit?, confidence?}
        """
        out: Any = None

        # Try methods in order of most structured → least
        for method in ("propose_candidates", "generate_candidates"):
            if hasattr(module, method):
                try:
                    out = getattr(module, method)(hidden_state, context)
                    break
                except Exception:
                    out = None

        if out is None and hasattr(module, "select_next_token"):
            try:
                out = module.select_next_token(hidden_state, context)
            except Exception:
                out = None

        if out is None and hasattr(module, "generate"):
            try:
                try:
                    out = module.generate(hidden_state, context)
                except TypeError:
                    out = module.generate(hidden_state)
            except Exception:
                out = None

        # Normalize to list[dict]
        props = self._normalize_candidates(out)

        # Add provenance metadata
        for p in props:
            p.setdefault("provenance", [])
            p["provenance"].append(
                {
                    "module": name,
                    "raw": {
                        "score": p.get("score"),
                        "logit": p.get("logit"),
                        "confidence": p.get("confidence"),
                    },
                }
            )

        return props

    def _get_module_confidence(self, module: Any) -> float:
        """Extract confidence score from module if available"""
        if hasattr(module, "get_confidence"):
            try:
                conf = module.get_confidence()
                if isinstance(conf, (int, float)):
                    return float(conf)
            except Exception:
                pass
        return 1.0

    # ================================ Normalization & Fusion ================================ #

    def _normalize_candidates(self, proposals: Any) -> List[Dict[str, Any]]:
        """
        Convert module proposals to uniform dict format.

        Accepts:
        - Token
        - List[Token]
        - List[Dict] with 'token' or 'id'
        """
        if proposals is None:
            return []

        # Single token -> list
        if not isinstance(proposals, (list, tuple)):
            proposals = [proposals]

        out: List[Dict[str, Any]] = []
        for item in proposals:
            if isinstance(item, dict):
                token = item.get("token", item.get("id"))
                if token is None:
                    continue
                d = {"token": token}
                # Pass through optional fields
                for field in ("score", "logit", "confidence", "prob"):
                    if field in item and isinstance(item[field], (int, float)):
                        d[field] = float(item[field])
                out.append(d)
            else:
                # Assume token
                out.append({"token": item})

        return out

    def _normalize_module_proposals(
        self, props: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Ensure module-local probabilities by normalizing scores/logits.
        Adds 'module_prob' to each candidate.
        """
        if not props:
            return props

        # Prefer explicit 'score', else 'logit', else 'prob', else uniform
        scores = [p.get("score") for p in props]
        logits = [p.get("logit") for p in props]
        probs_raw = [p.get("prob") for p in props]

        if any(isinstance(s, (int, float)) for s in scores):
            # Use scores
            svec = [float(s) if isinstance(s, (int, float)) else 0.0 for s in scores]
            probs = self._softmax(svec)
        elif any(isinstance(l, (int, float)) for l in logits):
            # Use logits
            lvec = [float(l) if isinstance(l, (int, float)) else 0.0 for l in logits]
            probs = self._softmax(lvec)
        elif any(isinstance(p, (int, float)) for p in probs_raw):
            # Use existing probs
            pvec = [float(p) if isinstance(p, (int, float)) else 0.0 for p in probs_raw]
            total = sum(pvec)
            probs = [p / total for p in pvec] if total > 0 else self._uniform(len(pvec))
        else:
            # Uniform
            probs = self._uniform(len(props))

        for p, pr in zip(props, probs):
            p["module_prob"] = float(pr)

        return props

    def _fuse_proposals(
        self,
        per_module: Dict[str, List[Dict[str, Any]]],
        weights: Dict[str, float],
        module_confidences: Dict[str, float],
        fusion_strategy: FusionStrategy = FusionStrategy.WEIGHTED_SUM,
    ) -> Dict[Any, Dict[str, Any]]:
        """
        Fuse proposals from all modules using specified strategy.
        """
        if fusion_strategy == FusionStrategy.WEIGHTED_SUM:
            return self._fuse_weighted_sum(per_module, weights, module_confidences)
        elif fusion_strategy == FusionStrategy.PRODUCT:
            return self._fuse_product(per_module, weights, module_confidences)
        elif fusion_strategy == FusionStrategy.MAX:
            return self._fuse_max(per_module, weights)
        elif fusion_strategy == FusionStrategy.RANK_FUSION:
            return self._fuse_rank_fusion(per_module, weights)
        elif fusion_strategy == FusionStrategy.BORDA_COUNT:
            return self._fuse_borda_count(per_module, weights)
        else:
            return self._fuse_weighted_sum(per_module, weights, module_confidences)

    def _fuse_weighted_sum(
        self,
        per_module: Dict[str, List[Dict[str, Any]]],
        weights: Dict[str, float],
        module_confidences: Dict[str, float],
    ) -> Dict[Any, Dict[str, Any]]:
        """Weighted sum fusion"""
        fused: Dict[Any, Dict[str, Any]] = {}

        for mod_name, proposals in per_module.items():
            w = float(weights.get(mod_name, 1.0))
            mod_conf = module_confidences.get(mod_name, 1.0)

            # Confidence scaling
            if self.cfg.enable_confidence_scaling:
                w *= mod_conf

            for p in proposals:
                tok = p.get("token")
                if tok is None:
                    continue

                contrib = w * float(p.get("module_prob", 0.0))

                entry = fused.get(tok)
                if entry is None:
                    fused[tok] = {
                        "token": tok,
                        "score": contrib,
                        "provenance": [
                            {
                                "module": mod_name,
                                "prob": p.get("module_prob", 0.0),
                                "weight": w,
                                "confidence": mod_conf,
                            }
                        ],
                        "module_agreement": 1,
                    }
                else:
                    entry["score"] += contrib
                    entry.setdefault("provenance", []).append(
                        {
                            "module": mod_name,
                            "prob": p.get("module_prob", 0.0),
                            "weight": w,
                            "confidence": mod_conf,
                        }
                    )
                    entry["module_agreement"] += 1

        # Filter by minimum module agreement
        if self.cfg.min_module_agreement > 1:
            fused = {
                tok: data
                for tok, data in fused.items()
                if data["module_agreement"] >= self.cfg.min_module_agreement
            }

        return fused

    def _fuse_product(
        self,
        per_module: Dict[str, List[Dict[str, Any]]],
        weights: Dict[str, float],
        module_confidences: Dict[str, float],
    ) -> Dict[Any, Dict[str, Any]]:
        """Product fusion (geometric mean)"""
        fused: Dict[Any, Dict[str, Any]] = {}

        for mod_name, proposals in per_module.items():
            w = float(weights.get(mod_name, 1.0))
            module_confidences.get(mod_name, 1.0)

            for p in proposals:
                tok = p.get("token")
                if tok is None:
                    continue

                # Use probability with weight as exponent
                prob = float(p.get("module_prob", 1e-9))
                contrib = math.pow(prob, w)

                entry = fused.get(tok)
                if entry is None:
                    fused[tok] = {
                        "token": tok,
                        "score": contrib,
                        "provenance": [
                            {
                                "module": mod_name,
                                "prob": prob,
                                "weight": w,
                            }
                        ],
                        "module_agreement": 1,
                    }
                else:
                    entry["score"] *= contrib
                    entry.setdefault("provenance", []).append(
                        {
                            "module": mod_name,
                            "prob": prob,
                            "weight": w,
                        }
                    )
                    entry["module_agreement"] += 1

        # Convert product to geometric mean
        for data in fused.values():
            n = data["module_agreement"]
            if n > 0:
                data["score"] = math.pow(data["score"], 1.0 / n)

        return fused

    def _fuse_max(
        self,
        per_module: Dict[str, List[Dict[str, Any]]],
        weights: Dict[str, float],
    ) -> Dict[Any, Dict[str, Any]]:
        """Max fusion (take highest weighted score)"""
        fused: Dict[Any, Dict[str, Any]] = {}

        for mod_name, proposals in per_module.items():
            w = float(weights.get(mod_name, 1.0))

            for p in proposals:
                tok = p.get("token")
                if tok is None:
                    continue

                contrib = w * float(p.get("module_prob", 0.0))

                entry = fused.get(tok)
                if entry is None or contrib > entry["score"]:
                    fused[tok] = {
                        "token": tok,
                        "score": contrib,
                        "provenance": [
                            {
                                "module": mod_name,
                                "prob": p.get("module_prob", 0.0),
                                "weight": w,
                            }
                        ],
                        "module_agreement": 1
                        if entry is None
                        else entry["module_agreement"] + 1,
                    }

        return fused

    def _fuse_rank_fusion(
        self,
        per_module: Dict[str, List[Dict[str, Any]]],
        weights: Dict[str, float],
    ) -> Dict[Any, Dict[str, Any]]:
        """Reciprocal Rank Fusion"""
        fused: Dict[Any, Dict[str, Any]] = {}
        k = 60  # RRF constant

        for mod_name, proposals in per_module.items():
            w = float(weights.get(mod_name, 1.0))

            # Sort proposals by module_prob descending
            sorted_props = sorted(
                proposals, key=lambda p: p.get("module_prob", 0), reverse=True
            )

            for rank, p in enumerate(sorted_props, start=1):
                tok = p.get("token")
                if tok is None:
                    continue

                # RRF score: 1 / (k + rank)
                rrf_score = 1.0 / (k + rank)
                contrib = w * rrf_score

                entry = fused.get(tok)
                if entry is None:
                    fused[tok] = {
                        "token": tok,
                        "score": contrib,
                        "provenance": [
                            {
                                "module": mod_name,
                                "rank": rank,
                                "weight": w,
                            }
                        ],
                        "module_agreement": 1,
                    }
                else:
                    entry["score"] += contrib
                    entry.setdefault("provenance", []).append(
                        {
                            "module": mod_name,
                            "rank": rank,
                            "weight": w,
                        }
                    )
                    entry["module_agreement"] += 1

        return fused

    def _fuse_borda_count(
        self,
        per_module: Dict[str, List[Dict[str, Any]]],
        weights: Dict[str, float],
    ) -> Dict[Any, Dict[str, Any]]:
        """Borda count voting"""
        fused: Dict[Any, Dict[str, Any]] = {}

        for mod_name, proposals in per_module.items():
            w = float(weights.get(mod_name, 1.0))
            n = len(proposals)

            # Sort proposals by module_prob descending
            sorted_props = sorted(
                proposals, key=lambda p: p.get("module_prob", 0), reverse=True
            )

            for rank, p in enumerate(sorted_props, start=1):
                tok = p.get("token")
                if tok is None:
                    continue

                # Borda points: n - rank
                borda_points = n - rank
                contrib = w * borda_points

                entry = fused.get(tok)
                if entry is None:
                    fused[tok] = {
                        "token": tok,
                        "score": contrib,
                        "provenance": [
                            {
                                "module": mod_name,
                                "borda_points": borda_points,
                                "weight": w,
                            }
                        ],
                        "module_agreement": 1,
                    }
                else:
                    entry["score"] += contrib
                    entry.setdefault("provenance", []).append(
                        {
                            "module": mod_name,
                            "borda_points": borda_points,
                            "weight": w,
                        }
                    )
                    entry["module_agreement"] += 1

        return fused

    def _finalize_probs_and_logits(
        self, fused_list: List[Dict[str, Any]], temperature: float = 1.0
    ) -> None:
        """
        Normalize ensemble scores → probabilities, apply temperature, estimate logits.
        """
        if not fused_list:
            return

        # Extract scores
        scores = [max(0.0, float(d.get("score", 0.0))) for d in fused_list]

        # Temperature scaling on scores (before normalization)
        if temperature != 1.0 and temperature > 0:
            scores = [s / temperature for s in scores]

        # Normalize to probabilities
        if sum(scores) <= self.cfg.normalization_eps:
            n = len(scores)
            probs = [1.0 / n] * n
        else:
            if self.cfg.normalization_method == NormalizationMethod.SOFTMAX:
                probs = self._softmax(scores)
            elif self.cfg.normalization_method == NormalizationMethod.MIN_MAX:
                probs = self._min_max_normalize(scores)
            elif self.cfg.normalization_method == NormalizationMethod.Z_SCORE:
                probs = self._z_score_normalize(scores)
            else:
                probs = self._normalize(scores)

        # Assign probabilities and compute logits
        for d, pr in zip(fused_list, probs):
            d["prob"] = float(pr)
            if self.cfg.attach_logits:
                d["logit"] = float(math.log(max(pr, self.cfg.min_prob_floor)))

    def _apply_diversity_penalty(
        self, fused: Dict[Any, Dict[str, Any]], penalty: float
    ) -> Dict[Any, Dict[str, Any]]:
        """
        Apply diversity penalty to encourage varied candidates.
        """
        # Simple implementation: slightly boost less common tokens
        # In production, use semantic similarity

        # Count token frequencies (heuristic: token similarity)
        token_strs = [str(data["token"]) for data in fused.values()]
        freq = {}
        for ts in token_strs:
            freq[ts] = freq.get(ts, 0) + 1

        # Apply penalty
        for data in fused.values():
            ts = str(data["token"])
            if freq[ts] > 1:
                # Penalize common tokens
                data["score"] *= 1.0 - penalty * (freq[ts] - 1) / len(fused)

        return fused

    # ================================ Adaptive Weighting ================================ #

    def _adapt_weights(
        self,
        weights: Dict[str, float],
        per_module: Dict[str, List[Dict[str, Any]]],
        module_confidences: Dict[str, float],
        context: Any,
    ) -> Dict[str, float]:
        """
        Dynamically adapt module weights based on context and performance.
        """
        adapted = dict(weights)

        # Boost weights for confident modules
        for name, conf in module_confidences.items():
            if conf > 0.8:
                adapted[name] = adapted.get(name, 1.0) * 1.1
            elif conf < 0.5:
                adapted[name] = adapted.get(name, 1.0) * 0.9

        # Context-based adaptation
        if isinstance(context, dict):
            domain = context.get("domain", "").lower()

            # Domain-specific weight adjustments
            if domain in ("math", "logic", "reasoning"):
                adapted["symbolic"] = adapted.get("symbolic", 1.0) * 1.2
                adapted["causal"] = adapted.get("causal", 1.0) * 1.15
            elif domain in ("creative", "story", "narrative"):
                adapted["analogical"] = adapted.get("analogical", 1.0) * 1.3
                adapted["language"] = adapted.get("language", 1.0) * 1.2
            elif domain in ("factual", "knowledge"):
                adapted["language"] = adapted.get("language", 1.0) * 1.2
                adapted["probabilistic"] = adapted.get("probabilistic", 1.0) * 1.1

        # Performance-based adaptation
        for name in per_module.keys():
            if name in self._module_performance:
                history = self._module_performance[name]
                if len(history) >= 10:
                    # Boost reliable, fast modules
                    avg_time = sum(h["time"] for h in history) / len(history)
                    if avg_time < 0.01:  # Fast module
                        adapted[name] = adapted.get(name, 1.0) * 1.05

        return adapted

    def _model_cross_module_interactions(
        self, per_module: Dict[str, List[Dict[str, Any]]], weights: Dict[str, float]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Model interactions between modules to boost consensus candidates.
        """
        # Find tokens proposed by multiple modules
        all_tokens: Dict[Any, List[str]] = defaultdict(list)
        for mod_name, proposals in per_module.items():
            for p in proposals:
                tok = p.get("token")
                if tok is not None:
                    all_tokens[tok].append(mod_name)

        # Boost candidates with cross-module agreement
        for mod_name, proposals in per_module.items():
            for p in proposals:
                tok = p.get("token")
                if tok is None:
                    continue

                # Count how many other modules also proposed this
                other_modules = list(all_tokens[tok) if m != mod_name]
                if len(other_modules) > 0:
                    # Boost module_prob based on agreement
                    boost = 1.0 + 0.1 * len(other_modules)
                    p["module_prob"] = min(1.0, p.get("module_prob", 0.5) * boost)
                    p["cross_module_agreement"] = len(other_modules) + 1

        return per_module

    # ================================ Ensemble Metrics ================================ #

    def _compute_ensemble_confidence(
        self, candidate: Dict[str, Any], module_confidences: Dict[str, float]
    ) -> float:
        """
        Compute ensemble confidence for a candidate.
        """
        provenance = candidate.get("provenance", [])
        if not provenance:
            return 0.5

        # Weighted average of module confidences
        total_weight = 0.0
        weighted_conf = 0.0

        for prov in provenance:
            mod_name = prov.get("module")
            weight = prov.get("weight", 1.0)
            conf = module_confidences.get(mod_name, 1.0)

            weighted_conf += conf * weight
            total_weight += weight

        if total_weight > 0:
            base_conf = weighted_conf / total_weight
        else:
            base_conf = 0.5

        # Boost confidence for high module agreement
        agreement = candidate.get("module_agreement", 1)
        if agreement > 1:
            base_conf = min(1.0, base_conf * (1.0 + 0.05 * (agreement - 1)))

        return base_conf

    def _compute_ensemble_uncertainty(self, candidate: Dict[str, Any]) -> float:
        """
        Compute uncertainty estimate for a candidate.
        """
        prob = candidate.get("prob", 0.5)
        agreement = candidate.get("module_agreement", 1)

        # Base uncertainty from probability
        base_uncertainty = 1.0 - prob

        # Reduce uncertainty with high module agreement
        if agreement > 1:
            base_uncertainty *= 1.0 / math.sqrt(agreement)

        return max(0.0, min(1.0, base_uncertainty))

    def _compute_diversity_score(
        self, candidate: Dict[str, Any], all_candidates: List[Dict[str, Any]]
    ) -> float:
        """
        Compute diversity score (how different from other candidates).
        """
        # Simplified: based on score difference
        # In production, use semantic similarity

        score = candidate.get("score", 0)
        other_scores = [c.get("score", 0) for c in all_candidates if c is not candidate]

        if not other_scores:
            return 0.5

        # Average absolute difference
        avg_diff = sum(abs(score - s) for s in other_scores) / len(other_scores)

        # Normalize to [0, 1]
        max_possible_diff = max(other_scores) if other_scores else 1.0
        diversity = min(1.0, avg_diff / (max_possible_diff + 1e-9))

        return diversity

    # ================================ Weights & Math ================================ #

    def _merge_weights(self, overrides: Dict[str, float]) -> Dict[str, float]:
        """Merge default weights with overrides"""
        out = dict(self.cfg.default_module_weights)
        for k, v in (overrides or {}).items():
            try:
                out[k] = float(v)
            except Exception:
                continue
        return out

    def _softmax(self, xs: List[float]) -> List[float]:
        """Numerically stable softmax"""
        if not xs:
            return []
        m = max(xs)
        exps = [math.exp(x - m) for x in xs]
        s = sum(exps)
        if s <= 0:
            return self._uniform(len(xs))
        return [e / s for e in exps]

    def _normalize(self, xs: List[float]) -> List[float]:
        """Simple normalization"""
        s = sum(xs)
        if s <= 0:
            return self._uniform(len(xs))
        return [x / s for x in xs]

    def _min_max_normalize(self, xs: List[float]) -> List[float]:
        """Min-max normalization"""
        if not xs:
            return []
        min_x = min(xs)
        max_x = max(xs)
        if max_x - min_x < 1e-9:
            return self._uniform(len(xs))
        normalized = [(x - min_x) / (max_x - min_x) for x in xs]
        return self._normalize(normalized)

    def _z_score_normalize(self, xs: List[float]) -> List[float]:
        """Z-score normalization"""
        if not xs:
            return []
        mean = sum(xs) / len(xs)
        var = sum((x - mean) ** 2 for x in xs) / len(xs)
        std = math.sqrt(var) if var > 0 else 1.0

        z_scores = [(x - mean) / std for x in xs]
        # Shift to positive and normalize
        min_z = min(z_scores)
        shifted = [z - min_z + 1.0 for z in z_scores]
        return self._normalize(shifted)

    def _uniform(self, n: int) -> List[float]:
        """Uniform distribution"""
        if n <= 0:
            return []
        return [1.0 / n] * n

    # ================================ Caching ================================ #

    def _get_cache_key(
        self, hidden_state: Any, context: Any, weights: Dict[str, float]
    ) -> str:
        """Generate cache key from inputs"""
        # Simplified hashing
        # In production, use more sophisticated hashing

        components = [
            str(type(hidden_state).__name__),
            str(id(hidden_state)),
            json.dumps(weights, sort_keys=True),
        ]

        if isinstance(context, dict):
            context_str = json.dumps(
                {
                    "domain": context.get("domain"),
                    "type": context.get("type"),
                },
                sort_keys=True,
            )
            components.append(context_str)

        combined = ":".join(components)
        return hashlib.md5(combined.encode(), usedforsecurity=False).hexdigest()

    def _update_cache(self, key: str, result: List[Dict[str, Any]]) -> None:
        """Update cache with LRU eviction"""
        if len(self._cache) >= self.cfg.cache_size:
            # Simple eviction: remove oldest
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[key] = result

    # ================================ Fallbacks ================================ #

    def _fallback_candidates(self, max_k: int) -> List[Dict[str, Any]]:
        """
        Generate fallback candidates when no modules available.
        Returns uniform probability distribution over token IDs.
        """
        k = max(1, min(max_k, self.cfg.fallback_vocab_size))
        tokens = list(range(k))
        prob = 1.0 / k
        logit = math.log(max(prob, self.cfg.min_prob_floor))

        out = []
        for i, t in enumerate(tokens, start=1):
            d = {
                "token": t,
                "score": prob,
                "prob": prob,
                "logit": float(logit),
                "rank": i,
                "provenance": [{"module": "fallback", "prob": prob, "weight": 1.0}],
                "confidence": 0.5,
                "module_agreement": 0,
                "uncertainty": 0.5,
                "diversity_score": 0.5,
            }
            out.append(d)

        return out
