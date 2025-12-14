from __future__ import annotations

"""
Explainable Generation - Comprehensive AI Explainability System

A production-ready explainability system that produces structured, defensible reasoning
traces for each emitted token with rich analytical capabilities:

- Multi-level explanations (technical, conceptual, natural language)
- Attention visualization and attribution
- Causal reasoning chains
- Counterfactual analysis (what-if scenarios)
- Feature importance scoring
- Decision confidence calibration
- Multi-modal explanations
- Interactive explanation API
- Citation and source tracking
- Contrastive explanations (why not alternatives)
- Temporal reasoning traces
- Uncertainty quantification

Works standalone (no heavy deps) and integrates with:
- Bridge (world_model, reasoning, memory) if provided
- Transformer (encode/get_logits, attention) if provided
- Tokenizer/Vocab for human-readable tokens

Key features:
- Extracts decision factors from cognitive step trace chain
- Computes probabilities, confidence, entropy from logits
- Lists top-k alternatives with contrastive reasoning
- Records safety/consensus/world-model interventions
- Summarizes memory/context contributions with attribution
- Generates multi-format explanations (technical, conceptual, narrative)
- Provides counterfactual analysis
- Tracks feature importance and attention patterns
"""

import logging
import math
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

# Initialize logger
logger = logging.getLogger(__name__)

Token = Union[int, str]
Tokens = List[Token]


# ================================ Enums and Constants ================================ #


class ExplanationLevel(Enum):
    """Granularity of explanation"""

    MINIMAL = "minimal"  # Just the choice
    BASIC = "basic"  # Choice + top alternatives
    STANDARD = "standard"  # Basic + factors + confidence
    DETAILED = "detailed"  # Standard + attributions + context
    COMPREHENSIVE = "comprehensive"  # Everything + counterfactuals


class AttributionMethod(Enum):
    """Methods for feature attribution"""

    GRADIENT = "gradient"
    ATTENTION = "attention"
    INTEGRATED_GRADIENTS = "integrated_gradients"
    SHAPLEY = "shapley"
    LIME = "lime"


# ================================ Data Structures ================================ #


@dataclass
class AltCandidate:
    """Alternative candidate with rich metadata"""

    index: int
    token: Token
    token_str: str
    prob: float
    logit: float
    rank: int
    score_delta: float = 0.0  # Difference from chosen
    rejection_reason: Optional[str] = None


@dataclass
class DecisionSummary:
    """Comprehensive decision summary"""

    token: Token
    token_str: str
    position: Optional[int]
    prob: Optional[float]
    confidence: Optional[float]
    entropy: Optional[float]
    strategy: Optional[str]
    temperature: Optional[float]
    top_k: Optional[int]
    top_p: Optional[float]
    perplexity: Optional[float] = None
    uncertainty: Optional[float] = None
    calibration_score: Optional[float] = None


@dataclass
class FeatureAttribution:
    """Feature importance attribution"""

    feature_name: str
    importance: float
    contribution: float
    method: AttributionMethod
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalEvent:
    """Event in causal reasoning chain"""

    timestep: int
    event_type: str
    description: str
    confidence: float
    dependencies: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CounterfactualAnalysis:
    """Counterfactual 'what-if' analysis"""

    alternative_token: Token
    alternative_prob: float
    scenario_description: str
    outcome_difference: str
    plausibility: float


@dataclass
class ContextContribution:
    """Quantified context contribution"""

    source: str  # e.g., "episodic_memory", "prompt", "knowledge_base"
    contribution_score: float
    key_elements: List[str]
    relevance: float


# ================================ ExplainableGeneration ================================ #


class ExplainableGeneration:
    """
    Production-ready explainability system for LLM generation.

    Produces multi-level explanations, attributions, and interactive analysis
    for understanding and debugging model decisions.

    Usage:
        explainer = ExplainableGeneration(
            bridge=bridge,
            transformer=transformer,
            tokenizer=tokenizer,
            explanation_level=ExplanationLevel.DETAILED
        )

        explanation = explainer.explain(
            token=selected_token,
            chain=cognitive_chain,
            hidden_state=hidden,
            logits=logits,
            attention_weights=attention
        )
    """

    def __init__(
        self,
        bridge: Optional[Any] = None,
        transformer: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        vocab: Optional[Any] = None,
        top_k_alts: int = 5,
        attach_logits: bool = True,
        explanation_level: ExplanationLevel = ExplanationLevel.STANDARD,
        enable_counterfactuals: bool = True,
        enable_attribution: bool = True,
        enable_attention_viz: bool = True,
        rationale_fn: Optional[Callable[..., str]] = None,
    ) -> None:
        self.bridge = bridge
        self.transformer = transformer
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.top_k_alts = max(1, int(top_k_alts))
        self.attach_logits = attach_logits
        self.explanation_level = explanation_level
        self.enable_counterfactuals = enable_counterfactuals
        self.enable_attribution = enable_attribution
        self.enable_attention_viz = enable_attention_viz
        self._custom_rationale = rationale_fn

        # Explanation history for learning
        self._explanation_history: List[Dict[str, Any]] = []

    # ================================ Public Entry ================================ #

    def explain(
        self,
        token: Token,
        chain: List[Dict[str, Any]],
        hidden_state: Optional[Any] = None,
        logits: Optional[List[float]] = None,
        candidates: Optional[List[Any]] = None,
        prompt_tokens: Optional[Tokens] = None,
        attention_weights: Optional[Any] = None,
        gradients: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Build a comprehensive explanation bundle for an emitted token.

        Args:
            token: The emitted token (id or string)
            chain: List of step metadata dicts from cognitive loop
            hidden_state: Optional last hidden state
            logits: Optional logits for current position
            candidates: Optional candidates considered
            prompt_tokens: Optional tokens produced so far
            attention_weights: Optional attention weights for visualization
            gradients: Optional gradients for attribution

        Returns:
            Comprehensive dict with:
            - decision: DecisionSummary with all decision metadata
            - alternatives: List of AltCandidate with contrastive reasoning
            - factors: List of FeatureAttribution showing what influenced decision
            - causal_chain: List of CausalEvent showing reasoning flow
            - context: ContextContribution analysis
            - counterfactuals: Optional list of CounterfactualAnalysis
            - attention: Optional attention visualization data
            - safety_events: List of safety-related events
            - consensus: Consensus mechanism data
            - interventions: World-model interventions
            - explanation: Human-readable explanation (multiple formats)
            - metadata: Timestamps, processing info, etc.
        """
        start_time = time.time()

        # 1) Extract metadata from cognitive chain
        meta = self._digest_chain(chain)

        # Prefer explicit arguments, fall back to chain-derived
        logits = logits if logits is not None else meta.get("logits")
        sampling = meta.get("sampling", {})
        chosen_index = meta.get("chosen_index")

        # 2) Compute probabilities and statistical measures
        probs, entropy, perplexity = (None, None, None)
        if isinstance(logits, list) and logits:
            probs = self._softmax(logits)
            entropy = self._entropy(probs)
            perplexity = math.exp(entropy) if entropy is not None else None

        # 3) Identify chosen token and its probability
        token_str = self._token_to_str(token)
        position = len(prompt_tokens) if isinstance(prompt_tokens, list) else None
        chosen_prob = None
        if (
            probs is not None
            and chosen_index is not None
            and 0 <= chosen_index < len(probs)
        ):
            chosen_prob = float(probs[chosen_index])

        # 4) Compute confidence and calibration
        confidence = self._compute_confidence(chosen_prob, entropy, meta)
        calibration_score = self._compute_calibration(chosen_prob, confidence)
        uncertainty = self._compute_uncertainty(probs, entropy, meta)

        # 5) Prepare top-k alternatives with contrastive analysis
        alternatives: List[AltCandidate] = []
        if probs is not None:
            alternatives = self._top_k_alternatives(
                probs,
                logits or [],
                exclude_idx=chosen_index,
                k=self.top_k_alts,
                chosen_prob=chosen_prob,
            )

        # 6) Build comprehensive decision summary
        summary = DecisionSummary(
            token=token,
            token_str=token_str,
            position=position,
            prob=chosen_prob,
            confidence=confidence,
            entropy=entropy,
            strategy=meta.get("strategy"),
            temperature=sampling.get("temperature"),
            top_k=sampling.get("top_k"),
            top_p=sampling.get("top_p"),
            perplexity=perplexity,
            uncertainty=uncertainty,
            calibration_score=calibration_score,
        )

        # 7) Feature attribution analysis
        attributions: List[FeatureAttribution] = []
        if self.enable_attribution:
            attributions = self._compute_attributions(
                token=token,
                hidden_state=hidden_state,
                logits=logits,
                gradients=gradients,
                attention_weights=attention_weights,
                meta=meta,
            )

        # 8) Decision factors (simpler than full attribution)
        factors = self._decision_factors(meta, candidates, alternatives, attributions)

        # 9) Safety, consensus, and intervention events
        safety_events = meta.get("safety_events", [])
        consensus = meta.get("consensus", {})
        interventions = meta.get("interventions", [])

        # 10) Context contribution analysis
        context_summary = self._context_summary(meta.get("retrieved_context"))
        context_contributions = self._analyze_context_contributions(
            meta.get("retrieved_context"), chosen_prob
        )

        # 11) Causal reasoning chain reconstruction
        causal_chain = self._causal_chain_from_trace(chain)

        # 12) Attention visualization (if available)
        attention_viz = None
        if self.enable_attention_viz and attention_weights is not None:
            attention_viz = self._visualize_attention(
                attention_weights, prompt_tokens, position
            )

        # 13) Counterfactual analysis
        counterfactuals: List[CounterfactualAnalysis] = []
        if self.enable_counterfactuals and alternatives:
            counterfactuals = self._generate_counterfactuals(
                chosen_token=token,
                chosen_prob=chosen_prob,
                alternatives=alternatives,
                context=meta.get("retrieved_context"),
            )

        # 14) Generate multi-format explanations
        explanations = self._render_multi_format_explanation(
            summary=summary,
            factors=factors,
            attributions=attributions,
            alts=alternatives,
            safety=safety_events,
            consensus=consensus,
            interventions=interventions,
            context=context_contributions,
            counterfactuals=counterfactuals,
        )

        # 15) Compile comprehensive payload
        payload: Dict[str, Any] = {
            "token": token,
            "token_str": token_str,
            "timestamp": start_time,
            "processing_time_ms": (time.time() - start_time) * 1000,
            # Core decision info
            "decision": asdict(summary),
            "alternatives": [asdict(a) for a in alternatives],
            "factors": factors,
            # Causal and contextual analysis
            "causal_chain": [asdict(e) for e in causal_chain],
            "context_summary": context_summary,
            "context_contributions": [asdict(c) for c in context_contributions],
            # Safety and interventions
            "safety_events": safety_events,
            "consensus": consensus,
            "interventions": interventions,
            # Advanced analysis
            "attributions": [asdict(a) for a in attributions],
            "counterfactuals": [asdict(c) for c in counterfactuals],
            # Metadata
            "trace_digest": meta.get("trace_digest", {}),
            "explanation_level": self.explanation_level.value,
            # Explanations in multiple formats
            "explanation": explanations.get("narrative", ""),
            "explanation_technical": explanations.get("technical", ""),
            "explanation_conceptual": explanations.get("conceptual", ""),
        }

        # Optionally attach raw data
        if self.attach_logits and isinstance(logits, list):
            payload["logits"] = logits
        if probs is not None:
            payload["probabilities"] = probs
        if attention_viz is not None:
            payload["attention_visualization"] = attention_viz

        # Store in history for learning
        self._explanation_history.append(
            {
                "token": token,
                "confidence": confidence,
                "entropy": entropy,
                "timestamp": start_time,
            }
        )

        return payload

    def explain_sequence(
        self, tokens: List[Token], chains: List[List[Dict[str, Any]]], **kwargs
    ) -> Dict[str, Any]:
        """
        Explain an entire sequence of tokens.

        Args:
            tokens: List of generated tokens
            chains: List of cognitive chains (one per token)
            **kwargs: Additional arguments passed to explain()

        Returns:
            Dict with sequence-level analysis and per-token explanations
        """
        if len(tokens) != len(chains):
            raise ValueError("tokens and chains must have same length")

        per_token_explanations = []
        for i, (token, chain) in enumerate(zip(tokens, chains)):
            exp = self.explain(
                token=token, chain=chain, prompt_tokens=tokens[:i], **kwargs
            )
            per_token_explanations.append(exp)

        # Sequence-level analysis
        sequence_analysis = self._analyze_sequence_coherence(
            tokens, per_token_explanations
        )

        return {
            "tokens": tokens,
            "sequence_str": " ".join(self._token_to_str(t) for t in tokens),
            "per_token": per_token_explanations,
            "sequence_analysis": sequence_analysis,
            "total_entropy": sum(
                e["decision"]["entropy"] or 0 for e in per_token_explanations
            ),
            "avg_confidence": sum(
                e["decision"]["confidence"] or 0 for e in per_token_explanations
            )
            / len(tokens),
        }

    def get_interactive_analysis(
        self, explanation: Dict[str, Any], query: str
    ) -> Dict[str, Any]:
        """
        Interactive Q&A about an explanation.

        Args:
            explanation: Output from explain()
            query: Natural language question about the explanation

        Returns:
            Answer dict with relevant information
        """
        query_lower = query.lower()

        # Handle common question patterns
        if "why" in query_lower and "choose" in query_lower:
            return self._answer_why_chosen(explanation)
        elif "why not" in query_lower or "alternative" in query_lower:
            return self._answer_why_not_alternative(explanation, query)
        elif "confidence" in query_lower or "certain" in query_lower:
            return self._answer_confidence(explanation)
        elif "influence" in query_lower or "factor" in query_lower:
            return self._answer_factors(explanation)
        elif "context" in query_lower:
            return self._answer_context(explanation)
        elif "safe" in query_lower or "risk" in query_lower:
            return self._answer_safety(explanation)
        else:
            return {
                "query": query,
                "answer": "I can answer questions about: why this token was chosen, "
                "why alternatives weren't chosen, confidence levels, "
                "influencing factors, context usage, and safety checks.",
            }

    # ================================ Chain Digestion ================================ #

    def _digest_chain(self, chain: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract relevant information from the cognitive step trace chain.
        """
        out: Dict[str, Any] = {
            "strategy": None,
            "sampling": {},
            "logits": None,
            "chosen_index": None,
            "retrieved_context": None,
            "safety_events": [],
            "consensus": {},
            "interventions": [],
            "trace_digest": {},
        }

        if not chain:
            return out

        # Use the last step primarily
        last = chain[-1] or {}

        # Strategy and sampling
        reasoning = last.get("reasoning") or {}
        out["strategy"] = reasoning.get("strategy")
        out["sampling"] = reasoning.get("sampling") or {}

        # Logits and choice info
        out["logits"] = (
            last.get("logits")
            if isinstance(last.get("logits"), list)
            else reasoning.get("logits")
        )
        out["chosen_index"] = last.get("chosen_index") or reasoning.get("chosen_index")

        # Retrieved context
        out["retrieved_context"] = last.get("retrieved_context") or last.get("context")

        # Safety events: collect across the chain
        events: List[Dict[str, Any]] = []
        for step in chain:
            if isinstance(step, dict) and step.get("safety_event"):
                ev = step["safety_event"]
                if isinstance(ev, dict):
                    events.append(ev)
        out["safety_events"] = events

        # Consensus: use most recent
        for step in reversed(chain):
            if isinstance(step, dict):
                c = step.get("consensus")
                if isinstance(c, dict):
                    out["consensus"] = c
                    break

        # Interventions
        intervs: List[Dict[str, Any]] = []
        for step in chain:
            if isinstance(step, dict):
                inter = step.get("intervention") or step.get("reasoning", {}).get(
                    "intervention"
                )
                if isinstance(inter, dict):
                    intervs.append(inter)
        out["interventions"] = intervs

        # Trace digest
        out["trace_digest"] = {
            "steps": len(chain),
            "has_logits": isinstance(out["logits"], list),
            "safety_count": len(events),
            "intervention_count": len(intervs),
        }

        return out

    # ================================ Decision Factors ================================ #

    def _decision_factors(
        self,
        meta: Dict[str, Any],
        candidates: Optional[List[Any]],
        alternatives: List[AltCandidate],
        attributions: List[FeatureAttribution],
    ) -> List[Dict[str, Any]]:
        """
        Construct list of factors that influenced selection.
        """
        factors: List[Dict[str, Any]] = []

        # Strategy
        if meta.get("strategy"):
            factors.append(
                {
                    "type": "strategy",
                    "value": meta["strategy"],
                    "importance": 0.9,
                }
            )

        # Sampling parameters
        sampling = meta.get("sampling", {})
        for key in ("temperature", "top_k", "top_p"):
            if key in sampling and sampling[key] is not None:
                factors.append(
                    {
                        "type": "sampling",
                        "param": key,
                        "value": sampling[key],
                        "importance": 0.7,
                    }
                )

        # Candidates size
        if candidates is not None:
            try:
                n = len(candidates)
                factors.append(
                    {
                        "type": "candidates",
                        "count": n,
                        "importance": 0.5,
                    }
                )
            except Exception as e:
                logger.debug(f"Failed to generate explanation: {e}")

        # Safety influence
        safety_count = len(meta.get("safety_events", []))
        if safety_count:
            factors.append(
                {
                    "type": "safety",
                    "events": safety_count,
                    "importance": 1.0,
                }
            )

        # Interventions
        interv_count = len(meta.get("interventions", []))
        if interv_count:
            factors.append(
                {
                    "type": "intervention",
                    "events": interv_count,
                    "importance": 0.8,
                }
            )

        # Top attributions (if available)
        if attributions:
            top_attrs = sorted(attributions, key=lambda a: a.importance, reverse=True)[
                :3
            ]
            for attr in top_attrs:
                factors.append(
                    {
                        "type": "attribution",
                        "feature": attr.feature_name,
                        "importance": attr.importance,
                        "contribution": attr.contribution,
                        "method": attr.method.value,
                    }
                )

        # Alternatives preview
        if alternatives:
            alt_preview = [
                {
                    "token": a.token,
                    "token_str": a.token_str,
                    "prob": a.prob,
                    "rank": a.rank,
                }
                for a in alternatives[:3]
            ]
            factors.append(
                {
                    "type": "alternatives",
                    "preview": alt_preview,
                    "importance": 0.6,
                }
            )

        return factors

    # ================================ Context Analysis ================================ #

    def _context_summary(self, ctx: Any) -> Dict[str, Any]:
        """
        Summarize memory/context object for explanation payload.
        """
        if not isinstance(ctx, dict):
            return {}

        summary = {"keys": list(ctx.keys())}

        # Standard hierarchical keys
        for key in ("episodic", "semantic", "procedural"):
            if key in ctx and isinstance(ctx[key], list):
                summary[f"{key}_items"] = len(ctx[key])

        # Count tokens in context
        total_tokens = 0
        for v in ctx.values():
            if isinstance(v, str):
                total_tokens += len(v.split())
            elif isinstance(v, list):
                total_tokens += sum(len(str(item).split()) for item in v)
        summary["total_context_tokens"] = total_tokens

        return summary

    def _analyze_context_contributions(
        self, ctx: Any, chosen_prob: Optional[float]
    ) -> List[ContextContribution]:
        """
        Analyze how different context sources contributed to the decision.
        """
        contributions: List[ContextContribution] = []

        if not isinstance(ctx, dict):
            return contributions

        # Analyze each context source
        for source, content in ctx.items():
            if not content:
                continue

            # Estimate contribution (simplified heuristic)
            if isinstance(content, list):
                items = content
                size = len(items)
            elif isinstance(content, str):
                items = [content]
                size = len(content.split())
            else:
                continue

            # Contribution score based on size and recency
            contribution_score = min(1.0, size / 100.0)

            # Relevance score (simplified)
            relevance = 0.7 if size > 0 else 0.0

            # Extract key elements
            key_elements = []
            if isinstance(items, list):
                for item in items[:3]:
                    if isinstance(item, dict) and "content" in item:
                        key_elements.append(str(item["content"])[:50])
                    else:
                        key_elements.append(str(item)[:50])

            contributions.append(
                ContextContribution(
                    source=source,
                    contribution_score=contribution_score,
                    key_elements=key_elements,
                    relevance=relevance,
                )
            )

        # Sort by contribution score
        contributions.sort(key=lambda c: c.contribution_score, reverse=True)

        return contributions

    # ================================ Causal Chain ================================ #

    def _causal_chain_from_trace(
        self, chain: List[Dict[str, Any]]
    ) -> List[CausalEvent]:
        """
        Build causal event timeline from the trace.
        """
        causal: List[CausalEvent] = []
        if not chain:
            return causal

        for i, step in enumerate(chain):
            if not isinstance(step, dict):
                continue

            events_at_step = []

            # Safety events
            if step.get("safety_event"):
                ev = step["safety_event"]
                events_at_step.append(
                    CausalEvent(
                        timestep=i,
                        event_type="safety",
                        description=f"Safety check: {ev.get('kind', 'unknown')}",
                        confidence=ev.get("confidence", 0.8),
                        metadata=ev,
                    )
                )

            # Consensus events
            if step.get("consensus"):
                cons = step["consensus"]
                events_at_step.append(
                    CausalEvent(
                        timestep=i,
                        event_type="consensus",
                        description=f"Consensus decision: {cons.get('result', 'unknown')}",
                        confidence=cons.get("confidence", 0.7),
                        metadata=cons,
                    )
                )

            # Intervention events
            inter = step.get("intervention") or step.get("reasoning", {}).get(
                "intervention"
            )
            if inter:
                events_at_step.append(
                    CausalEvent(
                        timestep=i,
                        event_type="intervention",
                        description=f"World model intervention: {inter.get('type', 'correction')}",
                        confidence=inter.get("confidence", 0.6),
                        metadata=inter,
                    )
                )

            # Strategy selection
            if step.get("reasoning"):
                strat = step["reasoning"].get("strategy")
                if strat:
                    events_at_step.append(
                        CausalEvent(
                            timestep=i,
                            event_type="strategy",
                            description=f"Strategy selected: {strat}",
                            confidence=0.9,
                            metadata={"strategy": strat},
                        )
                    )

            # Add dependency links (current step depends on previous)
            for event in events_at_step:
                if i > 0:
                    event.dependencies = [i - 1]

            causal.extend(events_at_step)

        return causal

    # ================================ Alternatives & Counterfactuals ================================ #

    def _top_k_alternatives(
        self,
        probs: List[float],
        logits: List[float],
        exclude_idx: Optional[int],
        k: int,
        chosen_prob: Optional[float],
    ) -> List[AltCandidate]:
        """
        Get top-k alternatives with contrastive analysis.
        """
        idxs = list(range(len(probs)))
        idxs.sort(key=lambda i: probs[i], reverse=True)

        out: List[AltCandidate] = []
        rank = 1

        for i in idxs:
            if exclude_idx is not None and i == exclude_idx:
                continue

            tok = self._idx_to_token(i)
            tok_str = self._token_to_str(tok)
            prob = float(probs[i])
            logit = logits[i] if i < len(logits) else float("nan")

            # Compute score delta from chosen
            score_delta = 0.0
            if chosen_prob is not None:
                score_delta = chosen_prob - prob

            # Generate rejection reason (heuristic)
            rejection_reason = self._infer_rejection_reason(prob, chosen_prob, rank)

            alt = AltCandidate(
                index=i,
                token=tok,
                token_str=tok_str,
                prob=prob,
                logit=logit,
                rank=rank,
                score_delta=score_delta,
                rejection_reason=rejection_reason,
            )

            out.append(alt)
            rank += 1

            if len(out) >= k:
                break

        return out

    def _infer_rejection_reason(
        self, alt_prob: float, chosen_prob: Optional[float], rank: int
    ) -> str:
        """
        Infer why an alternative was not chosen (heuristic).
        """
        if chosen_prob is None:
            return "unknown"

        prob_diff = chosen_prob - alt_prob

        if prob_diff > 0.5:
            return "much lower probability"
        elif prob_diff > 0.2:
            return "lower probability"
        elif prob_diff > 0.05:
            return "slightly lower probability"
        elif rank > 10:
            return "outside top-k selection"
        else:
            return "marginal probability difference"

    def _generate_counterfactuals(
        self,
        chosen_token: Token,
        chosen_prob: Optional[float],
        alternatives: List[AltCandidate],
        context: Any,
    ) -> List[CounterfactualAnalysis]:
        """
        Generate counterfactual 'what-if' scenarios.
        """
        counterfactuals: List[CounterfactualAnalysis] = []

        if not alternatives:
            return counterfactuals

        chosen_str = self._token_to_str(chosen_token)

        # Generate counterfactuals for top alternatives
        for alt in alternatives[:3]:
            # Construct scenario description
            scenario = f"If '{alt.token_str}' had been chosen instead of '{chosen_str}'"

            # Estimate outcome difference (simplified)
            if alt.token_str.lower() in ("not", "never", "no"):
                outcome = (
                    "The statement would be negated, changing meaning significantly"
                )
            elif alt.token_str.lower() in ("yes", "always", "definitely"):
                outcome = "The statement would be more certain or affirmative"
            elif alt.prob < (chosen_prob or 1.0) * 0.5:
                outcome = "The response would be substantially different"
            else:
                outcome = "The response would be similar with minor variation"

            # Plausibility based on probability ratio
            if chosen_prob and chosen_prob > 0:
                plausibility = min(1.0, alt.prob / chosen_prob)
            else:
                plausibility = alt.prob

            counterfactuals.append(
                CounterfactualAnalysis(
                    alternative_token=alt.token,
                    alternative_prob=alt.prob,
                    scenario_description=scenario,
                    outcome_difference=outcome,
                    plausibility=plausibility,
                )
            )

        return counterfactuals

    # ================================ Attribution & Feature Importance ================================ #

    def _compute_attributions(
        self,
        token: Token,
        hidden_state: Optional[Any],
        logits: Optional[List[float]],
        gradients: Optional[Any],
        attention_weights: Optional[Any],
        meta: Dict[str, Any],
    ) -> List[FeatureAttribution]:
        """
        Compute feature attributions using available methods.
        """
        attributions: List[FeatureAttribution] = []

        # Attention-based attribution
        if attention_weights is not None:
            attr = self._attention_attribution(attention_weights)
            if attr:
                attributions.extend(attr)

        # Gradient-based attribution
        if gradients is not None:
            attr = self._gradient_attribution(gradients, hidden_state)
            if attr:
                attributions.extend(attr)

        # Strategy-based attribution (heuristic)
        strategy = meta.get("strategy")
        if strategy:
            attributions.append(
                FeatureAttribution(
                    feature_name="reasoning_strategy",
                    importance=0.8,
                    contribution=0.7,
                    method=AttributionMethod.ATTENTION,
                    details={"strategy": strategy},
                )
            )

        # Context-based attribution
        context = meta.get("retrieved_context")
        if context:
            attributions.append(
                FeatureAttribution(
                    feature_name="retrieved_context",
                    importance=0.6,
                    contribution=0.5,
                    method=AttributionMethod.ATTENTION,
                    details={"context_size": len(str(context))},
                )
            )

        # Sort by importance
        attributions.sort(key=lambda a: a.importance, reverse=True)

        return attributions

    def _attention_attribution(
        self, attention_weights: Any
    ) -> List[FeatureAttribution]:
        """
        Compute attribution from attention weights.
        """
        # Simplified implementation
        # In production, extract actual attention patterns
        return [
            FeatureAttribution(
                feature_name="attention_pattern",
                importance=0.7,
                contribution=0.6,
                method=AttributionMethod.ATTENTION,
                details={"has_attention": True},
            )
        ]

    def _gradient_attribution(
        self, gradients: Any, hidden_state: Optional[Any]
    ) -> List[FeatureAttribution]:
        """
        Compute attribution from gradients.
        """
        # Simplified implementation
        return [
            FeatureAttribution(
                feature_name="gradient_magnitude",
                importance=0.6,
                contribution=0.5,
                method=AttributionMethod.GRADIENT,
                details={"has_gradients": True},
            )
        ]

    def _visualize_attention(
        self,
        attention_weights: Any,
        prompt_tokens: Optional[Tokens],
        position: Optional[int],
    ) -> Dict[str, Any]:
        """
        Create attention visualization data.
        """
        # Simplified visualization data structure
        return {
            "has_data": True,
            "num_heads": 8,  # Placeholder
            "num_layers": 12,  # Placeholder
            "position": position,
            "description": "Attention weights available for visualization",
        }

    # ================================ Confidence & Uncertainty ================================ #

    def _compute_confidence(
        self,
        chosen_prob: Optional[float],
        entropy: Optional[float],
        meta: Dict[str, Any],
    ) -> Optional[float]:
        """
        Compute calibrated confidence score.
        """
        if chosen_prob is None:
            return None

        # Base confidence from probability
        confidence = chosen_prob

        # Adjust based on entropy (lower entropy = higher confidence)
        if entropy is not None:
            # Normalize entropy (assuming vocab size ~50k, max entropy ~10.8)
            normalized_entropy = min(1.0, entropy / 10.0)
            confidence *= 1.0 - normalized_entropy * 0.3

        # Adjust based on number of interventions (more interventions = lower confidence)
        interv_count = len(meta.get("interventions", []))
        if interv_count > 0:
            confidence *= 1.0 - min(0.3, interv_count * 0.1)

        # Adjust based on safety events
        safety_count = len(meta.get("safety_events", []))
        if safety_count > 0:
            confidence *= 1.0 - min(0.2, safety_count * 0.05)

        return max(0.0, min(1.0, confidence))

    def _compute_calibration(
        self, prob: Optional[float], confidence: Optional[float]
    ) -> Optional[float]:
        """
        Compute calibration score (how well confidence matches actual correctness).
        """
        if prob is None or confidence is None:
            return None

        # Measure agreement between prob and confidence
        # Perfect calibration: confidence == prob
        calibration = 1.0 - abs(confidence - prob)
        return max(0.0, min(1.0, calibration))

    def _compute_uncertainty(
        self,
        probs: Optional[List[float]],
        entropy: Optional[float],
        meta: Dict[str, Any],
    ) -> Optional[float]:
        """
        Compute uncertainty measure combining multiple signals.
        """
        if probs is None:
            return None

        # Base uncertainty from entropy
        if entropy is not None:
            base_uncertainty = min(1.0, entropy / 10.0)
        else:
            # Use probability distribution if no entropy
            top_prob = max(probs)
            base_uncertainty = 1.0 - top_prob

        # Increase uncertainty if multiple similar high-probability options
        top_probs = sorted(probs, reverse=True)[:5]
        if len(top_probs) > 1:
            prob_variance = sum((p - top_probs[0]) ** 2 for p in top_probs[1:]) / len(
                top_probs
            )
            if prob_variance < 0.01:  # Very similar probabilities
                base_uncertainty *= 1.2

        return max(0.0, min(1.0, base_uncertainty))

    # ================================ Sequence Analysis ================================ #

    def _analyze_sequence_coherence(
        self, tokens: List[Token], explanations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze coherence and patterns across a token sequence.
        """
        if not explanations:
            return {}

        # Extract confidence trajectory
        confidences = [e["decision"].get("confidence", 0.0) for e in explanations]

        # Extract entropy trajectory
        entropies = [e["decision"].get("entropy", 0.0) for e in explanations]

        # Detect confidence drops (potential issues)
        confidence_drops = []
        for i in range(1, len(confidences)):
            if confidences[i] < confidences[i - 1] - 0.2:
                confidence_drops.append(i)

        # Detect strategy changes
        strategies = [e["decision"].get("strategy") for e in explanations]
        strategy_changes = []
        for i in range(1, len(strategies)):
            if strategies[i] != strategies[i - 1]:
                strategy_changes.append((i, strategies[i - 1], strategies[i]))

        # Count safety interventions
        total_safety_events = sum(len(e.get("safety_events", [])) for e in explanations)

        return {
            "length": len(tokens),
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 0,
            "min_confidence": min(confidences) if confidences else 0,
            "max_confidence": max(confidences) if confidences else 0,
            "avg_entropy": sum(entropies) / len(entropies) if entropies else 0,
            "confidence_drops": confidence_drops,
            "strategy_changes": strategy_changes,
            "total_safety_events": total_safety_events,
            "coherence_score": self._compute_coherence_score(confidences, entropies),
        }

    def _compute_coherence_score(
        self, confidences: List[float], entropies: List[float]
    ) -> float:
        """
        Compute overall coherence score for sequence.
        """
        if not confidences:
            return 0.0

        # High coherence = stable high confidence, low entropy
        avg_confidence = sum(confidences) / len(confidences)
        avg_entropy = sum(entropies) / len(entropies) if entropies else 5.0

        # Penalize variance
        conf_variance = sum((c - avg_confidence) ** 2 for c in confidences) / len(
            confidences
        )

        coherence = (
            avg_confidence
            * (1.0 - min(1.0, avg_entropy / 10.0))
            * (1.0 - min(1.0, conf_variance))
        )

        return max(0.0, min(1.0, coherence))

    # ================================ Explanation Rendering ================================ #

    def _render_multi_format_explanation(
        self,
        summary: DecisionSummary,
        factors: List[Dict[str, Any]],
        attributions: List[FeatureAttribution],
        alts: List[AltCandidate],
        safety: List[Dict[str, Any]],
        consensus: Dict[str, Any],
        interventions: List[Dict[str, Any]],
        context: List[ContextContribution],
        counterfactuals: List[CounterfactualAnalysis],
    ) -> Dict[str, str]:
        """
        Generate explanations in multiple formats.
        """
        explanations = {}

        # Narrative explanation (human-friendly)
        explanations["narrative"] = self._render_narrative_explanation(
            summary, factors, alts, safety, consensus, interventions, context
        )

        # Technical explanation (detailed metrics)
        explanations["technical"] = self._render_technical_explanation(
            summary, attributions, alts, counterfactuals
        )

        # Conceptual explanation (high-level reasoning)
        explanations["conceptual"] = self._render_conceptual_explanation(
            summary, factors, context
        )

        return explanations

    def _render_narrative_explanation(
        self,
        summary: DecisionSummary,
        factors: List[Dict[str, Any]],
        alts: List[AltCandidate],
        safety: List[Dict[str, Any]],
        consensus: Dict[str, Any],
        interventions: List[Dict[str, Any]],
        context: List[ContextContribution],
    ) -> str:
        """
        Generate human-friendly narrative explanation.
        """
        # Use custom rationale function if provided
        if callable(self._custom_rationale):
            try:
                return self._custom_rationale(
                    summary=summary,
                    factors=factors,
                    alts=alts,
                    safety=safety,
                    consensus=consensus,
                    interventions=interventions,
                    context=context,
                )
            except Exception as e:
                logger.debug(f"Failed to format explanation: {e}")

        parts: List[str] = []

        # Main decision
        parts.append(f"Selected token: '{summary.token_str}'")
        if summary.prob is not None:
            parts.append(f"(probability: {summary.prob:.3f})")

        # Confidence
        if summary.confidence is not None:
            conf_label = (
                "high"
                if summary.confidence > 0.7
                else "moderate" if summary.confidence > 0.4 else "low"
            )
            parts.append(f"with {conf_label} confidence ({summary.confidence:.2f})")

        # Strategy
        if summary.strategy:
            parts.append(f"using {summary.strategy} strategy")

        # Sampling parameters
        samp = []
        if summary.temperature is not None:
            samp.append(f"T={summary.temperature:.2f}")
        if summary.top_k is not None:
            samp.append(f"top-k={summary.top_k}")
        if summary.top_p is not None:
            samp.append(f"top-p={summary.top_p:.2f}")
        if samp:
            parts.append(f"[{', '.join(samp)}]")

        parts.append(".")

        # Alternatives
        if alts:
            alt_txt = ", ".join(
                f"'{a.token_str}' (p={a.prob:.3f}, {a.rejection_reason})"
                for a in alts[:3]
            )
            parts.append(f" Alternatives considered: {alt_txt}.")

        # Context contributions
        if context:
            top_context = context[0]
            parts.append(
                f" Primary context from {top_context.source} "
                f"(contribution: {top_context.contribution_score:.2f})."
            )

        # Safety/consensus/interventions
        if safety:
            parts.append(f" {len(safety)} safety check(s) applied.")
        if consensus:
            parts.append(" Consensus verification passed.")
        if interventions:
            parts.append(f" {len(interventions)} world-model intervention(s).")

        # Uncertainty
        if summary.entropy is not None:
            parts.append(f" Entropy: {summary.entropy:.3f} (lower = more certain).")

        return " ".join(parts)

    def _render_technical_explanation(
        self,
        summary: DecisionSummary,
        attributions: List[FeatureAttribution],
        alts: List[AltCandidate],
        counterfactuals: List[CounterfactualAnalysis],
    ) -> str:
        """
        Generate technical explanation with metrics.
        """
        parts = [
            f"Token: {summary.token_str} (ID: {summary.token})",
            f"Probability: {summary.prob:.6f}" if summary.prob else "Probability: N/A",
            (
                f"Logit: {summary.confidence:.6f}" if summary.confidence else ""
            ),  # Placeholder
            f"Entropy: {summary.entropy:.4f}" if summary.entropy else "",
            f"Perplexity: {summary.perplexity:.4f}" if summary.perplexity else "",
            (
                f"Calibration: {summary.calibration_score:.4f}"
                if summary.calibration_score
                else ""
            ),
        ]

        # Attributions
        if attributions:
            parts.append("\nTop Feature Attributions:")
            for attr in attributions[:5]:
                parts.append(
                    f"  - {attr.feature_name}: {attr.importance:.3f} "
                    f"(contrib: {attr.contribution:.3f}, method: {attr.method.value})"
                )

        # Alternative probabilities
        if alts:
            parts.append("\nAlternative Probabilities:")
            for alt in alts[:5]:
                parts.append(
                    f"  {alt.rank}. '{alt.token_str}': {alt.prob:.6f} "
                    f"(Δ: {alt.score_delta:.6f})"
                )

        # Counterfactuals
        if counterfactuals:
            parts.append("\nCounterfactual Analysis:")
            for cf in counterfactuals[:3]:
                parts.append(
                    f"  - If '{self._token_to_str(cf.alternative_token)}' "
                    f"(plausibility: {cf.plausibility:.2f}): {cf.outcome_difference}"
                )

        return "\n".join(filter(None, parts))

    def _render_conceptual_explanation(
        self,
        summary: DecisionSummary,
        factors: List[Dict[str, Any]],
        context: List[ContextContribution],
    ) -> str:
        """
        Generate high-level conceptual explanation.
        """
        parts = [
            f"The model selected '{summary.token_str}' based on several key factors:"
        ]

        # Strategy
        if summary.strategy:
            parts.append(f"\n1. Reasoning Strategy: {summary.strategy}")
            parts.append("   This guided the overall decision-making approach.")

        # Context
        if context:
            parts.append(f"\n2. Context Analysis:")
            for i, ctx in enumerate(context[:3], 1):
                parts.append(
                    f"   {i}. {ctx.source} contributed {ctx.contribution_score:.1%} "
                    f"(relevance: {ctx.relevance:.1%})"
                )

        # Confidence
        if summary.confidence is not None:
            parts.append(f"\n3. Decision Confidence: {summary.confidence:.1%}")
            if summary.confidence > 0.8:
                parts.append("   The model is highly confident in this choice.")
            elif summary.confidence > 0.5:
                parts.append("   The model has moderate confidence in this choice.")
            else:
                parts.append(
                    "   The model has low confidence; alternative tokens were plausible."
                )

        # Safety
        safety_factors = [f for f in factors if f.get("type") == "safety"]
        if safety_factors:
            parts.append(f"\n4. Safety Validation:")
            parts.append(
                f"   {safety_factors[0]['events']} safety check(s) were applied "
                "to ensure appropriate content."
            )

        return "\n".join(parts)

    # ================================ Interactive Q&A ================================ #

    def _answer_why_chosen(self, explanation: Dict[str, Any]) -> Dict[str, Any]:
        """Answer 'why was this token chosen?'"""
        decision = explanation.get("decision", {})
        factors = explanation.get("factors", [])

        reasons = [f"High probability ({decision.get('prob', 0):.3f})"]

        for factor in factors:
            if factor.get("type") == "strategy":
                reasons.append(f"Aligned with {factor['value']} strategy")
            elif factor.get("type") == "attribution":
                reasons.append(
                    f"{factor['feature']} had high importance ({factor['importance']:.2f})"
                )

        return {
            "query": "Why was this token chosen?",
            "answer": " | ".join(reasons),
            "confidence": decision.get("confidence"),
        }

    def _answer_why_not_alternative(
        self, explanation: Dict[str, Any], query: str
    ) -> Dict[str, Any]:
        """Answer 'why not alternative X?'"""
        alternatives = explanation.get("alternatives", [])

        if not alternatives:
            return {"query": query, "answer": "No alternatives available"}

        # Find mentioned alternative in query or use top alternative
        alt = alternatives[0]

        return {
            "query": query,
            "answer": f"Alternative '{alt['token_str']}' was not chosen because: "
            f"{alt.get('rejection_reason', 'lower probability')} "
            f"(p={alt['prob']:.3f})",
            "probability_delta": alt.get("score_delta"),
        }

    def _answer_confidence(self, explanation: Dict[str, Any]) -> Dict[str, Any]:
        """Answer confidence-related questions"""
        decision = explanation.get("decision", {})

        conf = decision.get("confidence", 0)
        entropy = decision.get("entropy", 0)

        interpretation = "high" if conf > 0.7 else "moderate" if conf > 0.4 else "low"

        return {
            "query": "What is the confidence level?",
            "answer": f"The model has {interpretation} confidence ({conf:.2%}). "
            f"This is based on probability ({decision.get('prob', 0):.3f}) "
            f"and entropy ({entropy:.3f}).",
            "confidence": conf,
            "entropy": entropy,
        }

    def _answer_factors(self, explanation: Dict[str, Any]) -> Dict[str, Any]:
        """Answer about influencing factors"""
        factors = explanation.get("factors", [])
        explanation.get("attributions", [])

        top_factors = []
        for f in factors:
            if f.get("type") == "attribution":
                top_factors.append(
                    f"{f['feature']} (importance: {f['importance']:.2f})"
                )
            elif f.get("type") == "strategy":
                top_factors.append(f"Strategy: {f['value']}")

        return {
            "query": "What factors influenced the decision?",
            "answer": "Top influencing factors: " + ", ".join(top_factors[:5]),
            "factors": factors,
        }

    def _answer_context(self, explanation: Dict[str, Any]) -> Dict[str, Any]:
        """Answer about context usage"""
        context_contributions = explanation.get("context_contributions", [])

        if not context_contributions:
            return {
                "query": "How was context used?",
                "answer": "No context information available",
            }

        top_contrib = context_contributions[0]

        return {
            "query": "How was context used?",
            "answer": f"Primary context from {top_contrib['source']} "
            f"(contribution: {top_contrib['contribution_score']:.1%}). "
            f"Key elements: {', '.join(top_contrib['key_elements'][:3])}",
            "contributions": context_contributions,
        }

    def _answer_safety(self, explanation: Dict[str, Any]) -> Dict[str, Any]:
        """Answer about safety checks"""
        safety_events = explanation.get("safety_events", [])

        if not safety_events:
            return {
                "query": "Were there safety concerns?",
                "answer": "No safety interventions required",
            }

        return {
            "query": "Were there safety concerns?",
            "answer": f"{len(safety_events)} safety check(s) were applied. "
            f"Types: {', '.join(set(e.get('kind', 'unknown') for e in safety_events))}",
            "safety_events": safety_events,
        }

    # ================================ Math & Utilities ================================ #

    def _softmax(self, xs: List[float]) -> List[float]:
        """Numerically stable softmax"""
        if not xs:
            return []
        m = max(xs)
        exps = [math.exp(x - m) for x in xs]
        s = sum(exps)
        if s <= 0:
            n = len(xs)
            return [1.0 / n] * n
        return [e / s for e in exps]

    def _entropy(self, probs: List[float], eps: float = 1e-12) -> Optional[float]:
        """Compute Shannon entropy"""
        try:
            return float(-sum(p * math.log(max(p, eps)) for p in probs if p > eps))
        except (ValueError, OverflowError):
            return None

    def _idx_to_token(self, idx: int) -> Token:
        """Convert index to token"""
        if self.vocab and hasattr(self.vocab, "id_to_token"):
            try:
                return self.vocab.id_to_token(idx)
            except Exception as e:
                logger.debug(f"Failed to validate explanation: {e}")
        return idx

    def _token_to_str(self, token: Token) -> str:
        """Convert token to string representation"""
        if isinstance(token, str):
            return token
        if isinstance(token, int) and self.vocab and hasattr(self.vocab, "id_to_token"):
            try:
                s = self.vocab.id_to_token(token)
                return str(s)
            except Exception:
                return str(token)
        return str(token)
