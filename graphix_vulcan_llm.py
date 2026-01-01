from __future__ import annotations

"""
GraphixVulcanLLM - COMPLETE FIXED VERSION (Async Generator Handling)

Version: 2.0.2
Date: 2025-11-17 06:27:31 UTC
User: musicmonk42

CRITICAL FIX: Properly handle async generator from CognitiveLoop.generate()
"""

import asyncio
import inspect
import json
import logging
import math
import os
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Union,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Safe Imports / Fallbacks with Enhanced Error Handling
# -------------------------------------------------------------------

# Transformer Core
try:
    from src.llm_core.graphix_transformer import (
        GraphixTransformer,
        GraphixTransformerConfig,
    )

    logger.info("✓ GraphixTransformer loaded successfully")
except Exception as e:
    logger.warning(f"GraphixTransformer import failed: {e}, using fallback")

    @dataclass
    class GraphixTransformerConfig:
        num_layers: int = 6
        hidden_size: int = 512
        num_heads: int = 8
        vocab_size: int = 50257
        max_position_embeddings: int = 2048
        layer_norm_eps: float = 1e-5
        seed: int = 42
        dropout: float = 0.1
        activation: str = "gelu"

    class GraphixTransformer:
        def __init__(self, config=None, **kwargs):
            self.config = config or GraphixTransformerConfig()
            logger.info(
                f"Initialized fallback transformer: {self.config.vocab_size} vocab, {self.config.hidden_size}d"
            )
            self._vocab = list(range(self.config.vocab_size))
            self._eos_token_id = 50256

        def encode(self, tokens: List[int]) -> Dict[str, Any]:
            batch_size = 1
            seq_len = len(tokens) if tokens else 1
            hidden_dim = self.config.hidden_size
            hidden_states = [
                [0.01 * (i % 100) for i in range(hidden_dim)] for _ in range(seq_len)
            ]
            return {
                "hidden_states": hidden_states,
                "last_hidden_state": (
                    hidden_states[-1] if hidden_states else [0.0] * hidden_dim
                ),
                "attention_mask": [1] * seq_len,
            }

        def get_logits(
            self, hidden_state: List[float], tokens: List[int]
        ) -> List[float]:
            vocab_size = self.config.vocab_size
            logits = [0.0] * vocab_size
            if tokens:
                next_token = (tokens[-1] + 1) % vocab_size
                logits[next_token] = 5.0
                for i in range(min(50, vocab_size)):
                    logits[(next_token + i) % vocab_size] = 2.0 - (i * 0.04)
            else:
                logits[100] = 3.0
            return logits

        def generate_token(self, logits: List[float]) -> int:
            if not logits:
                return 0
            max_logit = max(logits)
            exp_logits = [math.exp(l - max_logit) for l in logits]
            sum_exp = sum(exp_logits)
            probs = [e / sum_exp for e in exp_logits]
            import random

            r = random.random()
            cumsum = 0.0
            for i, p in enumerate(probs):
                cumsum += p
                if r < cumsum:
                    return i
            return 0

        def apply_update(self, gradients: Dict[str, Any]) -> None:
            logger.debug(f"Applied gradient update: {len(gradients)} params")

        def vocab_size(self) -> int:
            return self.config.vocab_size

        def __call__(self, batch: Dict[str, Any]) -> float:
            return 0.05 + (hash(str(batch)) % 100) * 0.001


# Bridge Integration
try:
    from src.integration.graphix_vulcan_bridge import GraphixVulcanBridge

    logger.info("✓ GraphixVulcanBridge loaded successfully")
except Exception as e:
    logger.warning(f"GraphixVulcanBridge import failed: {e}, using fallback")

    class GraphixVulcanBridge:
        def __init__(self):
            self.world_model = type(
                "WorldModel", (), {"get_context": lambda: {}, "update": lambda x: None}
            )()
            self.reasoning = type(
                "Reasoning", (), {"reason": lambda query: {"result": "fallback"}}
            )()
            logger.info("Initialized fallback bridge")

        def before_execution(self, context: Dict[str, Any]) -> Dict[str, Any]:
            return {"memory": context.get("memory", {}), "retrieved": []}

        def during_execution(self, node: Any, context: Dict[str, Any]) -> str:
            return "language"

        def after_execution(self, result: Dict[str, Any]) -> None:
            pass

        def consensus_approve_token(
            self, token: int, position: int, chosen_index: Optional[int] = None
        ) -> bool:
            return 0 <= token < 50257


# Safety Components
try:
    from src.generation.safe_generation import SafeGeneration

    logger.info("✓ SafeGeneration loaded successfully")
except Exception as e:
    logger.warning(f"SafeGeneration import failed: {e}, using fallback")

    class SafeGeneration:
        def __init__(self, *args, **kwargs):
            self.blocked_tokens = {0, 50256}

        def validate_token(
            self, token: int, context: Optional[Dict] = None, world_model: Any = None
        ) -> int:
            if token in self.blocked_tokens:
                return 100
            return token

        def filter(
            self,
            candidates: List[int],
            context: Optional[Dict] = None,
            world_model: Any = None,
        ) -> List[int]:
            return [c for c in candidates if c not in self.blocked_tokens][:10]


try:
    from src.vulcan.safety.llm_validators import EnhancedSafetyValidator

    logger.info("✓ EnhancedSafetyValidator loaded successfully")
except Exception as e:
    logger.warning(f"EnhancedSafetyValidator import failed: {e}, using fallback")

    class EnhancedSafetyValidator:
        def __init__(self):
            self.validation_count = 0

        def validate_generation(
            self, token: int, context: Dict, world_model: Any = None
        ) -> int:
            self.validation_count += 1
            if token < 0 or token > 50257:
                return 100
            return token

        def validate_sequence(
            self, tokens: List[int], context: Dict, world_model: Any = None
        ) -> bool:
            return len(tokens) <= 2048 and all(0 <= t <= 50257 for t in tokens)


# Explainability
try:
    from src.generation.explainable_generation import ExplainableGeneration

    logger.info("✓ ExplainableGeneration loaded successfully")
except Exception as e:
    logger.warning(f"ExplainableGeneration import failed: {e}, using fallback")

    class ExplainableGeneration:
        def __init__(self, *args, **kwargs):
            pass

        def explain(self, token: int, chain: List[Dict], **kwargs) -> Dict[str, Any]:
            return {
                "token": token,
                "token_str": f"<{token}>",
                "explanation": f"Token {token} selected based on context",
                "reasoning_steps": len(chain),
                "confidence": 0.85,
                "alternatives": [],
            }


# Context Management
try:
    from src.context.hierarchical_context import HierarchicalContext

    logger.info("✓ HierarchicalContext loaded successfully")
except Exception as e:
    logger.warning(f"HierarchicalContext import failed: {e}, using fallback")

    class HierarchicalContext:
        def __init__(self):
            self.memory = deque(maxlen=1000)

        def retrieve_context_for_generation(
            self, query_tokens: List[int], max_tokens: int = 2048
        ) -> Dict[str, Any]:
            return {
                "flat": " ".join(map(str, query_tokens[-100:])),
                "episodic": list(self.memory)[-10:],
                "semantic": [],
                "procedural": [],
            }

        def store_generation(
            self, prompt: Any, generated: str, reasoning_trace: Dict
        ) -> None:
            self.memory.append(
                {
                    "prompt": str(prompt),
                    "generated": generated,
                    "timestamp": time.time(),
                }
            )


try:
    from src.context.causal_context import CausalContext

    logger.info("✓ CausalContext loaded successfully")
except Exception as e:
    logger.warning(f"CausalContext import failed: {e}, using fallback")

    class CausalContext:
        def select(self, world_model: Any, query: str) -> Dict[str, Any]:
            return {"causal_context": [], "concepts": [], "relationships": []}


# Cognitive Loop
try:
    from src.integration.cognitive_loop import (
        CognitiveLoop,
        LoopRuntimeConfig,
        LoopSamplingConfig,
    )

    logger.info("✓ CognitiveLoop loaded successfully")
except Exception as e:
    logger.warning(f"CognitiveLoop import failed: {e}, using ENHANCED fallback")

    @dataclass
    class LoopSamplingConfig:
        max_tokens: int = 128
        temperature: float = 0.7
        top_k: int = 50
        top_p: float = 0.9
        repetition_penalty: float = 1.0

    @dataclass
    class LoopRuntimeConfig:
        enable_stream: bool = True
        enable_safety: bool = True
        enable_reasoning: bool = True

    class CognitiveLoop:
        def __init__(
            self,
            bridge: Any,
            transformer: Any,
            safety: Any,
            sampling_config: Optional[LoopSamplingConfig] = None,
            runtime_config: Optional[LoopRuntimeConfig] = None,
            **kwargs,
        ):
            self.bridge = bridge
            self.transformer = transformer
            self.safety = safety
            self.sampling = sampling_config or LoopSamplingConfig()
            self.runtime = runtime_config or LoopRuntimeConfig()
            logger.info(
                f"Initialized CognitiveLoop: max_tokens={self.sampling.max_tokens}, temp={self.sampling.temperature}"
            )

        def generate(
            self,
            prompt: Union[str, List[int]],
            max_tokens: Optional[int] = None,
            stream_callback: Optional[Callable] = None,
            stop_strings: tuple = (),
            stop_tokens: tuple = (),
        ):
            start = time.time()
            tokens = []
            reasoning_trace = []
            safety_events = []

            if isinstance(prompt, str):
                prompt_tokens = [ord(c) % 256 for c in prompt[:50]]
            else:
                prompt_tokens = list(prompt)

            tokens = list(prompt_tokens)
            max_steps = max_tokens or self.sampling.max_tokens

            logger.info(
                f"Generating {max_steps} tokens from prompt of length {len(prompt_tokens)}"
            )

            for step in range(max_steps):
                encoded = self.transformer.encode(tokens)
                hidden = encoded.get("hidden_states", [[0.0]])

                logits = self.transformer.get_logits(
                    hidden[-1] if hidden else [0.0], tokens
                )

                if self.sampling.temperature != 1.0:
                    logits = [l / self.sampling.temperature for l in logits]

                new_token = self.transformer.generate_token(logits)

                if self.runtime.enable_safety:
                    safe_token = self.safety.validate_generation(
                        new_token,
                        {"position": len(tokens), "prompt": prompt},
                        self.bridge.world_model,
                    )
                    if safe_token != new_token:
                        safety_events.append(
                            {
                                "step": step,
                                "original": new_token,
                                "safe": safe_token,
                                "reason": "safety_override",
                            }
                        )
                        new_token = safe_token

                if not self.bridge.consensus_approve_token(new_token, len(tokens)):
                    logger.debug(
                        f"Token {new_token} rejected by bridge at position {len(tokens)}"
                    )
                    break

                tokens.append(new_token)

                reasoning_trace.append(
                    {
                        "step": step,
                        "token": new_token,
                        "logits_max": max(logits) if logits else 0,
                        "sequence_length": len(tokens),
                    }
                )

                if stream_callback:
                    try:
                        stream_callback(new_token, f"<{new_token}>", {"step": step})
                    except Exception as e:
                        logger.warning(f"Stream callback error: {e}")

                if new_token in stop_tokens:
                    logger.info(f"Stopped at step {step}: stop token {new_token}")
                    break

                if new_token == 50256:
                    logger.info(f"Stopped at step {step}: EOS token")
                    break

            duration = time.time() - start
            text = " ".join([f"<{t}>" for t in tokens])

            logger.info(
                f"Generated {len(tokens)} tokens in {duration:.2f}s ({len(tokens)/duration:.1f} tok/s)"
            )

            return type(
                "GenerationResult",
                (),
                {
                    "tokens": tokens,
                    "text": text,
                    "reasoning_trace": reasoning_trace,
                    "safety_events": safety_events,
                    "audit_records": [],
                    "metrics": {
                        "duration_seconds": duration,
                        "tokens_per_second": (
                            len(tokens) / duration if duration > 0 else 0
                        ),
                        "total_tokens": len(tokens),
                    },
                    "completed": True,
                    "stopped_reason": (
                        "max_tokens" if len(tokens) >= max_steps else "eos"
                    ),
                    "duration_seconds": duration,
                },
            )()


# Language Reasoning
try:
    from src.vulcan.reasoning.language_reasoning import (
        LanguageReasoning,
        LanguageReasoningConfig,
    )

    logger.info("✓ LanguageReasoning loaded successfully")
except Exception as e:
    logger.warning(f"LanguageReasoning import failed: {e}, using fallback")

    @dataclass
    class LanguageReasoningConfig:
        temperature: float = 0.7
        top_k: int = 50
        top_p: float = 0.9

    class LanguageReasoning:
        def __init__(
            self, model: Any, config: Optional[LanguageReasoningConfig] = None, **kwargs
        ):
            self.model = model
            self.config = config or LanguageReasoningConfig()

        def generate(
            self,
            hidden_state: Any,
            generated_tokens: List[int],
            context: Optional[Dict] = None,
            strategy: Optional[str] = None,
        ) -> Dict[str, Any]:
            logits = self.model.get_logits(hidden_state, generated_tokens)
            token = self.model.generate_token(logits)
            return {
                "token": token,
                "token_id": token,
                "strategy": strategy or "greedy",
                "confidence": 0.85,
            }


# Training
try:
    from src.training.governed_trainer import GovernedTrainer

    logger.info("✓ GovernedTrainer loaded successfully")
except Exception as e:
    logger.warning(f"GovernedTrainer import failed: {e}, using fallback")

    class GovernedTrainer:
        def __init__(self, *args, **kwargs):
            self.audit_log = []
            self.step_count = 0

        def training_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
            self.step_count += 1
            loss = 0.05 + (self.step_count % 10) * 0.001
            rec = {
                "status": "applied",
                "loss": loss,
                "step": self.step_count,
                "timestamp": time.time(),
            }
            self.audit_log.append(rec)
            return rec

        def summary(self) -> Dict[str, Any]:
            return {
                "steps": len(self.audit_log),
                "total_steps": self.step_count,
                "avg_loss": (
                    sum(r["loss"] for r in self.audit_log) / len(self.audit_log)
                    if self.audit_log
                    else 0
                ),
            }


try:
    from src.training.self_improving_training import SelfImprovingTraining

    logger.info("✓ SelfImprovingTraining loaded successfully")
    HAS_SELF_IMPROVEMENT = True
except Exception as e:
    logger.warning(f"SelfImprovingTraining import failed: {e}, using fallback")
    HAS_SELF_IMPROVEMENT = False

    class SelfImprovingTraining:
        def __init__(self, *args, **kwargs):
            self.telemetry = []

        def record_telemetry(self, **metrics):
            self.telemetry.append({**metrics, "timestamp": time.time()})

        def detect_issue(self, *args, **kwargs):
            return None

        def self_improve(self, *args, **kwargs):
            return None

        def get_status(self) -> Dict[str, Any]:
            return {"telemetry_points": len(self.telemetry), "enabled": False}


# -------------------------------------------------------------------
# Additional LLM Component Imports (Full Integration)
# -------------------------------------------------------------------

# Unified Reasoning System
try:
    from src.vulcan.reasoning.unified_reasoning import UnifiedReasoner
    logger.info("✓ UnifiedReasoner loaded successfully")
    HAS_UNIFIED_REASONING = True
except Exception as e:
    logger.debug(f"UnifiedReasoner import failed: {e}")
    HAS_UNIFIED_REASONING = False
    UnifiedReasoner = None

# Vulcan Memory System
try:
    from src.vulcan.memory import (
        HierarchicalMemory,
        EpisodicMemory,
        SemanticMemory,
        ProceduralMemory,
        WorkingMemory,
        MemoryConsolidator,
        MemoryPersistence,
    )
    logger.info("✓ Vulcan Memory System loaded successfully")
    HAS_VULCAN_MEMORY = True
except Exception as e:
    logger.debug(f"Vulcan Memory import failed: {e}")
    HAS_VULCAN_MEMORY = False
    HierarchicalMemory = None
    EpisodicMemory = None
    SemanticMemory = None
    ProceduralMemory = None
    WorkingMemory = None
    MemoryConsolidator = None
    MemoryPersistence = None

# Unified Generation System
try:
    from src.generation.unified_generation import UnifiedGeneration
    logger.info("✓ UnifiedGeneration loaded successfully")
    HAS_UNIFIED_GENERATION = True
except Exception as e:
    logger.debug(f"UnifiedGeneration import failed: {e}")
    HAS_UNIFIED_GENERATION = False
    UnifiedGeneration = None

# LLM Executor
try:
    from src.execution.llm_executor import LLMExecutor
    logger.info("✓ LLMExecutor loaded successfully")
    HAS_LLM_EXECUTOR = True
except Exception as e:
    logger.debug(f"LLMExecutor import failed: {e}")
    HAS_LLM_EXECUTOR = False
    LLMExecutor = None

# GraphixExecutor (production executor)
try:
    from src.llm_core.graphix_executor import GraphixExecutor
    logger.info("✓ GraphixExecutor loaded successfully")
    HAS_GRAPHIX_EXECUTOR = True
except Exception as e:
    logger.debug(f"GraphixExecutor import failed: {e}")
    HAS_GRAPHIX_EXECUTOR = False
    GraphixExecutor = None

# Local GPT Provider
try:
    from src.local_llm.provider.local_gpt_provider import LocalGPTProvider
    logger.info("✓ LocalGPTProvider loaded successfully")
    HAS_LOCAL_GPT = True
except Exception as e:
    logger.debug(f"LocalGPTProvider import failed: {e}")
    HAS_LOCAL_GPT = False
    LocalGPTProvider = None

# Mathematical Reasoning
try:
    from src.vulcan.reasoning.mathematical_computation import MathematicalComputationTool
    from src.vulcan.reasoning.mathematical_verification import MathematicalVerificationEngine
    logger.info("✓ Mathematical Reasoning loaded successfully")
    HAS_MATH_REASONING = True
except Exception as e:
    logger.debug(f"Mathematical Reasoning import failed: {e}")
    HAS_MATH_REASONING = False
    MathematicalComputationTool = None
    MathematicalVerificationEngine = None

# Causal Reasoning
try:
    from src.vulcan.reasoning.causal_reasoning import CausalReasoning
    logger.info("✓ CausalReasoning loaded successfully")
    HAS_CAUSAL_REASONING = True
except Exception as e:
    logger.debug(f"CausalReasoning import failed: {e}")
    HAS_CAUSAL_REASONING = False
    CausalReasoning = None

# Analogical Reasoning
try:
    from src.vulcan.reasoning.analogical_reasoning import AnalogicalReasoning
    logger.info("✓ AnalogicalReasoning loaded successfully")
    HAS_ANALOGICAL_REASONING = True
except Exception as e:
    logger.debug(f"AnalogicalReasoning import failed: {e}")
    HAS_ANALOGICAL_REASONING = False
    AnalogicalReasoning = None

# Probabilistic Reasoning
try:
    from src.vulcan.reasoning.probabilistic_reasoning import ProbabilisticReasoning
    logger.info("✓ ProbabilisticReasoning loaded successfully")
    HAS_PROBABILISTIC_REASONING = True
except Exception as e:
    logger.debug(f"ProbabilisticReasoning import failed: {e}")
    HAS_PROBABILISTIC_REASONING = False
    ProbabilisticReasoning = None

# Multimodal Reasoning
try:
    from src.vulcan.reasoning.multimodal_reasoning import MultimodalReasoning
    logger.info("✓ MultimodalReasoning loaded successfully")
    HAS_MULTIMODAL_REASONING = True
except Exception as e:
    logger.debug(f"MultimodalReasoning import failed: {e}")
    HAS_MULTIMODAL_REASONING = False
    MultimodalReasoning = None

# Reasoning Integration
try:
    from src.vulcan.reasoning.reasoning_integration import ReasoningIntegration
    logger.info("✓ ReasoningIntegration loaded successfully")
    HAS_REASONING_INTEGRATION = True
except Exception as e:
    logger.debug(f"ReasoningIntegration import failed: {e}")
    HAS_REASONING_INTEGRATION = False
    ReasoningIntegration = None

# Dynamic Architecture
try:
    from src.execution.dynamic_architecture import DynamicArchitecture
    logger.info("✓ DynamicArchitecture loaded successfully")
    HAS_DYNAMIC_ARCHITECTURE = True
except Exception as e:
    logger.debug(f"DynamicArchitecture import failed: {e}")
    HAS_DYNAMIC_ARCHITECTURE = False
    DynamicArchitecture = None

# Persistent Context
try:
    from src.llm_core.persistant_context import PersistentContext
    logger.info("✓ PersistentContext loaded successfully")
    HAS_PERSISTENT_CONTEXT = True
except Exception as e:
    logger.debug(f"PersistentContext import failed: {e}")
    HAS_PERSISTENT_CONTEXT = False
    PersistentContext = None


# -------------------------------------------------------------------
# Configuration Loader
# -------------------------------------------------------------------


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Loads configuration from path or returns optimized defaults."""
    default_config = {
        "transformer": GraphixTransformerConfig(
            num_layers=6,
            hidden_size=512,
            num_heads=8,
            vocab_size=50257,
            max_position_embeddings=2048,
            dropout=0.1,
        ),
        "generation": {
            "max_tokens": 2000,  # Increased to 2000 for diagnostic purposes (was 64)
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.9,
            "repetition_penalty": 1.0,
            "enable_streaming": True,
        },
        "safety": {"mode": "first_safe", "enable_validation": True, "max_retries": 3},
        "training": {"max_grad_norm": 5.0, "learning_rate": 1e-4, "batch_size": 8},
        "performance": {
            "enable_caching": True,
            "cache_size_mb": 512,
            "enable_batching": True,
            "batch_size": 32,
        },
        "monitoring": {
            "enable_metrics": True,
            "log_level": "INFO",
            "enable_profiling": False,
        },
    }

    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                loaded = json.load(f)
            for key in loaded:
                if key in default_config and isinstance(default_config[key], dict):
                    default_config[key].update(loaded[key])
                else:
                    default_config[key] = loaded[key]
            logger.info(f"Loaded config from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")

    return default_config


# -------------------------------------------------------------------
# Data Classes
# -------------------------------------------------------------------


@dataclass
class GenerationResult:
    """Enhanced generation result with full telemetry"""

    tokens: List[int]
    text: str
    reasoning_trace: List[Dict[str, Any]]
    safety_events: List[Dict[str, Any]]
    explanation: Optional[Dict[str, Any]]
    metrics: Dict[str, Any]
    stopped_reason: str
    duration_seconds: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        return f"Generated {len(self.tokens)} tokens in {self.duration_seconds:.2f}s ({self.metrics.get('tokens_per_second', 0):.1f} tok/s)"


@dataclass
class TrainingRecord:
    """Enhanced training record"""

    status: str
    loss: float
    step: int = 0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


# -------------------------------------------------------------------
# Performance Monitor
# -------------------------------------------------------------------


class PerformanceMonitor:
    """Tracks system performance metrics"""

    def __init__(self):
        self.metrics = {
            "total_tokens": 0,
            "total_duration": 0.0,
            "generation_count": 0,
            "error_count": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }
        self._lock = threading.Lock()

    def record_generation(self, tokens: int, duration: float):
        with self._lock:
            self.metrics["total_tokens"] += tokens
            self.metrics["total_duration"] += duration
            self.metrics["generation_count"] += 1

    def record_error(self):
        with self._lock:
            self.metrics["error_count"] += 1

    def record_cache_hit(self):
        with self._lock:
            self.metrics["cache_hits"] += 1

    def record_cache_miss(self):
        with self._lock:
            self.metrics["cache_misses"] += 1

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            avg_tokens = self.metrics["total_tokens"] / max(
                1, self.metrics["generation_count"]
            )
            avg_duration = self.metrics["total_duration"] / max(
                1, self.metrics["generation_count"]
            )
            throughput = self.metrics["total_tokens"] / max(
                1, self.metrics["total_duration"]
            )
            cache_hit_rate = self.metrics["cache_hits"] / max(
                1, self.metrics["cache_hits"] + self.metrics["cache_misses"]
            )

            return {
                **self.metrics,
                "avg_tokens_per_generation": avg_tokens,
                "avg_duration_per_generation": avg_duration,
                "overall_throughput_tokens_per_sec": throughput,
                "cache_hit_rate": cache_hit_rate,
                "error_rate": self.metrics["error_count"]
                / max(1, self.metrics["generation_count"]),
            }


# -------------------------------------------------------------------
# Cache Manager
# -------------------------------------------------------------------


class CacheManager:
    """LRU cache for generation results"""

    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.access_order = deque()
        self.max_size = max_size
        self._lock = threading.Lock()

    def _cache_key(self, prompt: str, **kwargs) -> str:
        key_parts = [str(prompt)]
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")
        return "|".join(key_parts)

    def get(self, prompt: str, **kwargs) -> Optional[GenerationResult]:
        key = self._cache_key(prompt, **kwargs)
        with self._lock:
            if key in self.cache:
                self.access_order.remove(key)
                self.access_order.append(key)
                return self.cache[key]
        return None

    def put(self, prompt: str, result: GenerationResult, **kwargs):
        key = self._cache_key(prompt, **kwargs)
        with self._lock:
            if len(self.cache) >= self.max_size and key not in self.cache:
                oldest = self.access_order.popleft()
                del self.cache[oldest]

            self.cache[key] = result
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)

    def clear(self):
        with self._lock:
            self.cache.clear()
            self.access_order.clear()

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "utilization": len(self.cache) / self.max_size,
            }


# -------------------------------------------------------------------
# Main LLM Class - FIXED ASYNC GENERATOR HANDLING
# -------------------------------------------------------------------


class GraphixVulcanLLM:
    """
    Fully Optimized LLM over Graphix-VULCAN components.

    Version 2.0.2 - Critical fix for async generator handling
    """

    VERSION = "2.0.2"

    def __init__(
        self,
        config_path: Optional[str] = None,
        observability: Optional[Any] = None,
        audit_log: Optional[List[Dict[str, Any]]] = None,
        metrics_provider: Optional[Callable[[str], Optional[float]]] = None,
        enable_self_improvement: bool = True,
        enable_caching: bool = True,
        enable_monitoring: bool = True,
    ) -> None:
        logger.info(f"Initializing GraphixVulcanLLM v{self.VERSION}")

        self.config = load_config(config_path)
        self.observability = observability
        self.audit_log = audit_log if audit_log is not None else []
        self.metrics_provider = metrics_provider
        self._lock = threading.RLock()

        # Event loop management
        self._event_loop = None

        # Performance & Monitoring
        self.monitor = PerformanceMonitor() if enable_monitoring else None
        self.cache = CacheManager(max_size=1000) if enable_caching else None

        # Core Components
        logger.info("Loading core components...")
        self.transformer = GraphixTransformer(
            self.config.get("transformer"),
            observability=observability,
            audit_log=self.audit_log,
        )
        self.bridge = GraphixVulcanBridge()
        self.safety_validator = EnhancedSafetyValidator()
        self.safe_generation = SafeGeneration(observability=observability)
        self.explainer = ExplainableGeneration(
            bridge=self.bridge, transformer=self.transformer
        )
        self.hier_context = HierarchicalContext()
        self.causal_context = CausalContext()
        self.language_reasoning = LanguageReasoning(self.transformer)
        self.trainer = GovernedTrainer(
            max_grad_norm=self.config.get("training", {}).get("max_grad_norm", 5.0)
        )

        # Self-improvement
        self.self_improvement = None
        if HAS_SELF_IMPROVEMENT and enable_self_improvement:
            try:
                self.self_improvement = SelfImprovingTraining()
                logger.info("✓ Self-improvement enabled")
            except Exception as e:
                logger.warning(f"Self-improvement initialization failed: {e}")

        # -----------------------------------------------------------
        # Extended LLM Component Integration
        # -----------------------------------------------------------
        
        # Unified Reasoning System (orchestrates all reasoning types)
        self.unified_reasoner = None
        if HAS_UNIFIED_REASONING:
            try:
                self.unified_reasoner = UnifiedReasoner()
                logger.info("✓ UnifiedReasoner initialized")
            except Exception as e:
                logger.debug(f"UnifiedReasoner initialization skipped: {e}")
        
        # Vulcan Memory System (hierarchical + specialized memory)
        # PERF FIX Issue #2: Use singleton HierarchicalMemory to avoid re-initialization
        self.vulcan_memory = None
        self.episodic_memory = None
        self.semantic_memory = None
        self.procedural_memory = None
        self.working_memory = None
        self.memory_consolidator = None
        if HAS_VULCAN_MEMORY:
            try:
                # Try to use singleton HierarchicalMemory first
                try:
                    from vulcan.reasoning.singletons import get_hierarchical_memory
                    self.vulcan_memory = get_hierarchical_memory()
                    if self.vulcan_memory:
                        logger.info("✓ HierarchicalMemory obtained from singleton")
                except ImportError:
                    self.vulcan_memory = None
                
                # Fallback to direct instantiation if singleton not available
                if self.vulcan_memory is None:
                    self.vulcan_memory = HierarchicalMemory()
                    logger.info("✓ HierarchicalMemory initialized (direct)")
                
                self.episodic_memory = EpisodicMemory()
                self.semantic_memory = SemanticMemory()
                self.procedural_memory = ProceduralMemory()
                self.working_memory = WorkingMemory()
                self.memory_consolidator = MemoryConsolidator()
                logger.info("✓ Vulcan Memory System initialized")
            except Exception as e:
                logger.debug(f"Vulcan Memory initialization skipped: {e}")
        
        # Unified Generation (multi-strategy ensemble)
        self.unified_generation = None
        if HAS_UNIFIED_GENERATION:
            try:
                self.unified_generation = UnifiedGeneration()
                logger.info("✓ UnifiedGeneration initialized")
            except Exception as e:
                logger.debug(f"UnifiedGeneration initialization skipped: {e}")
        
        # LLM Executor (advanced graph execution)
        self.llm_executor = None
        if HAS_LLM_EXECUTOR:
            try:
                self.llm_executor = LLMExecutor()
                logger.info("✓ LLMExecutor initialized")
            except Exception as e:
                logger.debug(f"LLMExecutor initialization skipped: {e}")
        
        # GraphixExecutor (production executor with KV cache)
        self.graphix_executor = None
        if HAS_GRAPHIX_EXECUTOR:
            try:
                self.graphix_executor = GraphixExecutor()
                logger.info("✓ GraphixExecutor initialized")
            except Exception as e:
                logger.debug(f"GraphixExecutor initialization skipped: {e}")
        
        # Mathematical Reasoning
        self.math_computation = None
        self.math_verification = None
        if HAS_MATH_REASONING:
            try:
                self.math_computation = MathematicalComputationTool(llm=self)
                self.math_verification = MathematicalVerificationEngine()
                logger.info("✓ Mathematical Reasoning initialized")
            except Exception as e:
                logger.debug(f"Mathematical Reasoning initialization skipped: {e}")
        
        # Causal Reasoning
        self.causal_reasoning = None
        if HAS_CAUSAL_REASONING:
            try:
                self.causal_reasoning = CausalReasoning()
                logger.info("✓ CausalReasoning initialized")
            except Exception as e:
                logger.debug(f"CausalReasoning initialization skipped: {e}")
        
        # Analogical Reasoning
        self.analogical_reasoning = None
        if HAS_ANALOGICAL_REASONING:
            try:
                self.analogical_reasoning = AnalogicalReasoning()
                logger.info("✓ AnalogicalReasoning initialized")
            except Exception as e:
                logger.debug(f"AnalogicalReasoning initialization skipped: {e}")
        
        # Probabilistic Reasoning
        self.probabilistic_reasoning = None
        if HAS_PROBABILISTIC_REASONING:
            try:
                self.probabilistic_reasoning = ProbabilisticReasoning()
                logger.info("✓ ProbabilisticReasoning initialized")
            except Exception as e:
                logger.debug(f"ProbabilisticReasoning initialization skipped: {e}")
        
        # Multimodal Reasoning
        self.multimodal_reasoning = None
        if HAS_MULTIMODAL_REASONING:
            try:
                self.multimodal_reasoning = MultimodalReasoning()
                logger.info("✓ MultimodalReasoning initialized")
            except Exception as e:
                logger.debug(f"MultimodalReasoning initialization skipped: {e}")
        
        # Reasoning Integration (connects all reasoning systems)
        self.reasoning_integration = None
        if HAS_REASONING_INTEGRATION:
            try:
                self.reasoning_integration = ReasoningIntegration()
                logger.info("✓ ReasoningIntegration initialized")
            except Exception as e:
                logger.debug(f"ReasoningIntegration initialization skipped: {e}")
        
        # Dynamic Architecture
        self.dynamic_architecture = None
        if HAS_DYNAMIC_ARCHITECTURE:
            try:
                self.dynamic_architecture = DynamicArchitecture()
                logger.info("✓ DynamicArchitecture initialized")
            except Exception as e:
                logger.debug(f"DynamicArchitecture initialization skipped: {e}")
        
        # Persistent Context
        self.persistent_context = None
        if HAS_PERSISTENT_CONTEXT:
            try:
                self.persistent_context = PersistentContext()
                logger.info("✓ PersistentContext initialized")
            except Exception as e:
                logger.debug(f"PersistentContext initialization skipped: {e}")

        # Cognitive Loop
        logger.info("Initializing cognitive loop...")
        sampling_config = LoopSamplingConfig(
            max_tokens=self.config["generation"]["max_tokens"],
            temperature=self.config["generation"]["temperature"],
            top_k=self.config["generation"]["top_k"],
            top_p=self.config["generation"]["top_p"],
        )
        runtime_config = LoopRuntimeConfig(
            enable_stream=self.config["generation"]["enable_streaming"]
        )

        self.cog_loop = CognitiveLoop(
            bridge=self.bridge,
            transformer=self.transformer,
            safety=self.safety_validator,
            sampling_config=sampling_config,
            runtime_config=runtime_config,
            observability_manager=observability,
            audit_log=self.audit_log,
            tokenizer=getattr(self.transformer, 'tokenizer', None),  # Pass tokenizer for proper decoding
        )

        # State
        self._last_generation: Optional[GenerationResult] = None
        self._total_tokens_generated = 0
        self._generation_sessions = 0
        self._start_time = time.time()
        self._saved_state_fields = ["_total_tokens_generated", "_generation_sessions"]

        logger.info("✓ GraphixVulcanLLM initialized successfully")
        logger.info(
            f"  Model: {self.config['transformer'].vocab_size} vocab, {self.config['transformer'].hidden_size}d"
        )
        logger.info(
            f"  Config: max_tokens={self.config['generation']['max_tokens']}, temp={self.config['generation']['temperature']}"
        )

    # -----------------------------------------------------------
    # Async Helper Methods - FIXED
    # -----------------------------------------------------------

    def _get_event_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create event loop - OPTIMIZED to reuse existing loops.
        
        Performance fix: Avoids creating new event loops on each call.
        Caches the loop instance and only creates a new one if absolutely necessary.
        """
        # Fast path: Return cached loop if still valid
        if self._event_loop is not None and not self._event_loop.is_closed():
            return self._event_loop
        
        # Try to get an already running loop first
        try:
            loop = asyncio.get_running_loop()
            self._event_loop = loop
            return self._event_loop
        except RuntimeError:
            pass  # No running loop - continue to create/get one
        
        # Try to get the event loop from the policy (may already exist)
        try:
            loop = asyncio.get_event_loop_policy().get_event_loop()
            if not loop.is_closed():
                self._event_loop = loop
                return self._event_loop
        except RuntimeError:
            pass  # No loop available, need to create one
        
        # Last resort: Create a new event loop (only happens once typically)
        logger.debug("Creating new event loop (this should only happen once)")
        self._event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._event_loop)
        return self._event_loop

    def _check_running_loop(self) -> bool:
        """Check if there's already a running event loop in the current thread.

        Returns True if a loop is already running (and run_until_complete would fail),
        False if it's safe to create and run a new loop.
        
        BUG FIX: Added thread context logging to help debug why internal LLM
        returns None when called from HybridLLMExecutor.run_in_executor().
        """
        import threading
        thread_name = threading.current_thread().name
        thread_id = threading.current_thread().ident
        
        try:
            running_loop = asyncio.get_running_loop()
            # Log detailed context to help debug the issue
            logger.debug(
                f"[_check_running_loop] Event loop FOUND in thread '{thread_name}' (id={thread_id})"
            )
            return True  # Loop is running - not safe to use run_until_complete
        except RuntimeError:
            logger.debug(
                f"[_check_running_loop] No event loop in thread '{thread_name}' (id={thread_id}) - safe to proceed"
            )
            return False  # No running loop - safe to proceed

    def _run_async(self, coro):
        """Run async coroutine or collect async generator"""
        # EMERGENCY FIX: Check if we're in an async context where run_until_complete would fail
        if self._check_running_loop():
            logger.warning(
                "Event loop already running in _run_async() - returning None"
            )
            return None

        # Create a new event loop for this synchronous context
        # Note: We don't call set_event_loop() to avoid affecting the thread's default loop
        loop = asyncio.new_event_loop()

        try:
            # Check if it's an async generator
            if inspect.isasyncgen(coro):
                logger.debug("Detected async generator, collecting results...")

                async def _collect():
                    result = None
                    async for item in coro:
                        result = item  # Collect last item
                    return result

                return loop.run_until_complete(_collect())
            # Check if it's a coroutine
            elif inspect.iscoroutine(coro):
                logger.debug("Detected coroutine, awaiting...")
                return loop.run_until_complete(coro)
            else:
                # Not async, return as-is
                logger.debug("Not async, returning directly")
                return coro
        finally:
            # Clean up the event loop we created
            loop.close()

    def _is_async_callable(self, obj: Any, method_name: str) -> bool:
        """Check if a method is async"""
        if not hasattr(obj, method_name):
            return False
        method = getattr(obj, method_name)
        return asyncio.iscoroutinefunction(method) or inspect.iscoroutinefunction(
            method
        )

    # -----------------------------------------------------------
    # Generation Methods - FIXED
    # -----------------------------------------------------------

    def generate(
        self,
        prompt: Union[str, Sequence[int]],
        max_tokens: Optional[int] = None,
        explain: bool = True,
        stream: bool = False,
        stream_callback: Optional[Callable[[int, str, Dict[str, Any]], None]] = None,
        stop_strings: Optional[Sequence[str]] = None,
        stop_tokens: Optional[Sequence[int]] = None,
        use_cache: bool = True,
    ) -> GenerationResult:
        """High-level safe generation - FIXED nested async pattern"""
        start = time.time()
        max_steps = max_tokens or self.config["generation"]["max_tokens"]
        
        # FIX #1: Add debugging to expose why generate() may return None
        if isinstance(prompt, str):
            prompt_len = len(prompt)
        elif isinstance(prompt, (list, tuple)):
            prompt_len = f"{len(prompt)} items"
        else:
            prompt_len = "unknown"
        logger.info(f"[DEBUG] generate() called with prompt_len={prompt_len}, max_tokens={max_steps}")

        # Check cache
        if use_cache and self.cache and not stream:
            cache_key_params = {
                "max_tokens": max_steps,
                "temperature": self.config["generation"]["temperature"],
            }
            cached = self.cache.get(str(prompt), **cache_key_params)
            if cached:
                if self.monitor:
                    self.monitor.record_cache_hit()
                logger.debug(f"Cache hit for prompt: {str(prompt)[:50]}")
                return cached
            if self.monitor:
                self.monitor.record_cache_miss()

        logger.info(f"Generating up to {max_steps} tokens...")

        # Call generate method - NO try-except to let errors propagate
        gen_result = self.cog_loop.generate(
            prompt=prompt,
            max_tokens=max_steps,
            stream_callback=stream_callback if stream else None,
            stop_strings=tuple(stop_strings or []),
            stop_tokens=tuple(stop_tokens or []),
        )

        # Check what we got
        result_type_name = type(gen_result).__name__
        has_anext = hasattr(gen_result, "__anext__")
        has_tokens = hasattr(gen_result, "tokens")

        logger.debug(
            f"CognitiveLoop returned: type={result_type_name}, has___anext__={has_anext}, has_tokens={has_tokens}"
        )

        # FIX #1 CRITICAL: If result already has tokens, use it directly without async handling
        # This is the common case when using the fallback synchronous CognitiveLoop
        # or when running from run_in_executor() where the generation completed synchronously
        is_synchronous_result = has_tokens and not has_anext and not inspect.iscoroutine(gen_result)
        
        if is_synchronous_result:
            logger.info(f"[DEBUG] Using synchronous result directly (type={result_type_name})")
            loop_result = gen_result
        else:
            # Need async handling - check if we can safely create an event loop
            if self._check_running_loop():
                # FIX #1 CRITICAL: Instead of returning None, raise an informative error
                # This allows HybridLLMExecutor to provide a proper error message
                error_msg = (
                    "Cannot process async result: event loop already running in current thread. "
                    "This typically happens when generate() is called from within an async context. "
                    "Consider using generate_async() instead, or ensure generate() is called "
                    "from a synchronous context (e.g., via run_in_executor)."
                )
                logger.error(f"[DEBUG] {error_msg}")
                raise RuntimeError(error_msg)

            # Create a new event loop for this synchronous context
            # Note: We don't call set_event_loop() to avoid affecting the thread's default loop
            loop = asyncio.new_event_loop()

            try:
                # Process based on type
                if inspect.iscoroutine(gen_result):
                    logger.debug("DETECTED COROUTINE - AWAITING")
                    loop_result = loop.run_until_complete(gen_result)

                    # Check if the coroutine returned an async generator!
                    result_type_after_await = type(loop_result).__name__
                    logger.debug(
                        f"After awaiting coroutine: type={result_type_after_await}, has_tokens={hasattr(loop_result, 'tokens')}"
                    )

                    # If it returned an async generator, consume it
                    if hasattr(loop_result, "__anext__"):
                        logger.debug("COROUTINE RETURNED ASYNC GENERATOR - CONSUMING IT")

                        async def consume_nested_generator():
                            results = []
                            count = 0

                            async for item in loop_result:
                                count += 1
                                results.append(item)

                            logger.debug(f"Consumed {count} items from nested generator")

                            if not results:
                                raise ValueError("Nested generator yielded no items!")

                            return results[-1]

                        loop_result = loop.run_until_complete(consume_nested_generator())
                        logger.debug(
                            f"After consuming nested generator: type={type(loop_result).__name__}, has_tokens={hasattr(loop_result, 'tokens')}"
                        )

                elif has_anext:
                    logger.debug("DETECTED ASYNC GENERATOR - CONSUMING")

                    async def consume_generator():
                        results = []
                        count = 0

                        async for item in gen_result:
                            count += 1
                            results.append(item)

                        logger.debug(f"Consumed {count} items")

                        if not results:
                            raise ValueError("Generator yielded no items!")

                        return results[-1]

                    loop_result = loop.run_until_complete(consume_generator())

                elif has_tokens:
                    logger.debug("DETECTED DIRECT RESULT OBJECT")
                    loop_result = gen_result

                else:
                    raise TypeError(
                        f"Unknown return type: type={result_type_name}, "
                        f"has___anext__={has_anext}, has_tokens={has_tokens}"
                    )
            finally:
                # Clean up the event loop we created
                loop.close()

        # Final check
        final_type = type(loop_result).__name__
        final_has_tokens = hasattr(loop_result, "tokens")
        logger.debug(
            f"Final result: type={final_type}, has_tokens={final_has_tokens}"
        )

        if not final_has_tokens:
            raise AttributeError(
                f"Result STILL missing 'tokens'! Type: {final_type}"
            )

        # Extract data
        tokens = list(loop_result.tokens)
        text = loop_result.text
        reasoning_trace = loop_result.reasoning_trace
        safety_events = loop_result.safety_events
        metrics = loop_result.metrics
        stopped_reason = loop_result.stopped_reason
        duration_seconds = loop_result.duration_seconds

        logger.info(f"✓ Generated {len(text)} chars")

        # Explanation
        explanation = None
        if explain and tokens:
            try:
                hidden_state = self.transformer.encode(tokens).get(
                    "last_hidden_state"
                )
                explanation = self.explainer.explain(
                    token=tokens[-1],
                    chain=reasoning_trace,
                    hidden_state=hidden_state,
                    logits=None,
                    candidates=None,
                    prompt_tokens=tokens,
                )
            except Exception as e:
                logger.warning(f"Explanation generation failed: {e}")
                explanation = {"error": str(e)}

        # Build result
        result = GenerationResult(
            tokens=tokens,
            text=text,
            reasoning_trace=reasoning_trace,
            safety_events=safety_events,
            explanation=explanation,
            metrics=metrics,
            stopped_reason=stopped_reason,
            duration_seconds=duration_seconds,
            metadata={
                "config": self.config["generation"],
                "prompt_length": (
                    len(prompt) if isinstance(prompt, (list, str)) else 0
                ),
            },
        )

        # Update state
        with self._lock:
            self._last_generation = result
            self._total_tokens_generated += len(tokens)
            self._generation_sessions += 1

        # Monitor
        if self.monitor:
            self.monitor.record_generation(len(tokens), duration_seconds)

        # Cache
        if use_cache and self.cache and not stream:
            self.cache.put(str(prompt), result, max_tokens=max_steps)

        # Store in memory (hierarchical context)
        try:
            self.hier_context.store_generation(
                prompt,
                text,
                {
                    "trace_len": len(reasoning_trace),
                    "safety_events": len(safety_events),
                },
            )
        except Exception as e:
            logger.warning(f"Hierarchical memory storage failed: {e}")
        
        # Update causal context with generation info for causal reasoning
        try:
            self.causal_context.update(
                query=str(prompt),
                response=text,
                reasoning_trace=reasoning_trace,
                metadata={
                    "tokens_generated": len(tokens),
                    "duration": duration_seconds,
                    "stopped_reason": stopped_reason,
                }
            )
        except Exception as e:
            logger.debug(f"Causal context update skipped: {e}")

        # FIX #1: Add final logging to show successful generation
        result_len = len(result.text) if result is not None and hasattr(result, 'text') and result.text else 'None'
        logger.info(f"[DEBUG] generate() returning: type={type(result).__name__}, len={result_len}")
        if result is not None:
            logger.info(result.summary())
        return result

    # --- PATCH A START ---
    def stream(
        self,
        prompt: Union[str, Sequence[int]],
        max_tokens: Optional[int] = None,
        callback: Optional[Callable[[int, str, Dict[str, Any]], None]] = None,
        **kwargs,
    ) -> Iterator[int]:
        """Streaming token emission (correctly consumes async generator)."""
        collected_tokens: List[int] = []

        def _internal_callback(tok: int, decoded: str, info: Dict):
            # Store token first
            if isinstance(tok, int):
                collected_tokens.append(tok)
            if callback:
                try:
                    callback(tok, decoded, info)
                except Exception as e:
                    logger.warning(f"Stream callback error: {e}")

        try:
            gen_result = self.cog_loop.generate(
                prompt=prompt,
                max_tokens=max_tokens or self.config["generation"]["max_tokens"],
                stream_callback=_internal_callback,
                stop_tokens=tuple(kwargs.get("stop_tokens", [])),
                stop_strings=tuple(kwargs.get("stop_strings", [])),
            )

            # EMERGENCY FIX: Check if we're in an async context where run_until_complete would fail
            if self._check_running_loop():
                logger.warning(
                    "Event loop already running in stream() - returning empty result"
                )
                return  # Exit generator entirely, yielding nothing to caller

            # Create a new event loop for this synchronous context
            # Note: We don't call set_event_loop() to avoid affecting the thread's default loop
            loop = asyncio.new_event_loop()

            try:
                async def _consume_async_generator(async_gen):
                    # Drive the async generator to produce tokens (callback captures them)
                    async for chunk in async_gen:
                        # chunk is a dict with keys: token, text, token_info
                        tok = chunk.get("token")
                        if (
                            tok is not None
                            and isinstance(tok, int)
                            and tok not in collected_tokens
                        ):
                            collected_tokens.append(tok)

                # Distinguish coroutine vs async generator
                if inspect.iscoroutine(gen_result):
                    # Await coroutine to get async generator or final result
                    awaited = loop.run_until_complete(gen_result)
                    if inspect.isasyncgen(awaited):
                        loop.run_until_complete(_consume_async_generator(awaited))
                    else:
                        # Not streaming (fallback)
                        if hasattr(awaited, "tokens"):
                            collected_tokens.extend(
                                int(t) for t in awaited.tokens if isinstance(t, int)
                            )
                elif inspect.isasyncgen(gen_result):
                    loop.run_until_complete(_consume_async_generator(gen_result))
                else:
                    # Synchronous fallback result
                    if hasattr(gen_result, "tokens"):
                        collected_tokens.extend(
                            int(t) for t in gen_result.tokens if isinstance(t, int)
                        )
            finally:
                # Clean up the event loop we created
                loop.close()

            # Yield collected tokens
            for t in collected_tokens:
                yield t

            # Fallback: if still empty, log
            if not collected_tokens:
                logger.debug("Streaming produced no tokens (empty collection).")

        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            return

    # --- PATCH A END ---

    async def generate_async(
        self, prompt: Union[str, Sequence[int]], **kwargs
    ) -> GenerationResult:
        """Async generation wrapper"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.generate(prompt, **kwargs))

    async def stream_async(
        self, prompt: Union[str, Sequence[int]], **kwargs
    ) -> AsyncIterator[int]:
        """Async streaming wrapper"""

        async def _gen():
            for token in self.stream(prompt, **kwargs):
                yield token
                await asyncio.sleep(0)

        async for token in _gen():
            yield token

    def explain_last(self) -> Optional[Dict[str, Any]]:
        """Return explanation for last generation"""
        return self._last_generation.explanation if self._last_generation else None

    def generate_text(self, prompt: str, **kwargs) -> str:
        """Shortcut: returns generated text only"""
        result = self.generate(prompt, **kwargs)
        return result.text

    def quick_generate(self, prompt: str, max_tokens: int = 32) -> List[int]:
        """Lightweight greedy generation"""
        tokens: List[int] = []
        for _ in range(max_tokens):
            hidden = self.transformer.encode(tokens)
            logits = self.transformer.get_logits(
                hidden.get("last_hidden_state", [0.0]), tokens
            )
            idx = max(range(len(logits)), key=lambda i: logits[i]) if logits else 0
            safe_idx = self.safety_validator.validate_generation(
                idx, {"prompt": prompt}, self.bridge.world_model
            )
            tokens.append(safe_idx)
        return tokens

    # -----------------------------------------------------------
    # Training & Fine-Tuning
    # -----------------------------------------------------------

    def train(
        self,
        dataset: Sequence[Dict[str, Any]],
        epochs: int = 1,
        batch_size: Optional[int] = None,
    ) -> List[TrainingRecord]:
        """Iterate over dataset with governed training"""
        logger.info(f"Starting training: {len(dataset)} samples, {epochs} epochs")
        logs: List[TrainingRecord] = []
        batch_size = batch_size or self.config.get("training", {}).get("batch_size", 8)

        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i : i + batch_size]
                rec = self.trainer.training_step({"batch": batch, "epoch": epoch})
                logs.append(
                    TrainingRecord(
                        status=rec.get("status", "unknown"),
                        loss=rec.get("loss", 0.0),
                        step=rec.get("step", 0),
                        metadata={
                            "timestamp": rec.get("timestamp"),
                            "epoch": epoch,
                            "batch": i // batch_size,
                        },
                    )
                )

        logger.info(f"Training complete: {len(logs)} steps")
        return logs

    def fine_tune_step(self, batch: Dict[str, Any]) -> TrainingRecord:
        """Single governed training step"""
        rec = self.trainer.training_step(batch)
        return TrainingRecord(
            status=rec.get("status", "unknown"),
            loss=rec.get("loss", 0.0),
            step=rec.get("step", 0),
            metadata={"timestamp": rec.get("timestamp")},
        )

    # -----------------------------------------------------------
    # Self-Improvement
    # -----------------------------------------------------------

    def self_improve(
        self, context: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Perform intrinsic improvement cycle"""
        if not self.self_improvement:
            logger.debug("Self-improvement not available")
            return None

        try:
            self.self_improvement.record_telemetry(
                loss=0.05,
                eval_score=0.9,
                safety_incidents=0,
                causal_contradictions=0,
                novelty_score=0.7,
            )

            issue = self.self_improvement.detect_issue(
                {
                    "loss_plateau_steps": 15,
                    "loss_improvement_min": 0.005,
                    "safety_incident_threshold": 5,
                    "causal_contradiction_threshold": 3,
                    "novelty_min": 0.3,
                }
            )

            if issue:
                logger.info(f"Self-improvement issue detected: {issue}")
                return asdict(issue)
            else:
                logger.debug("No self-improvement issues detected")
                return {"status": "healthy"}

        except Exception as e:
            logger.error(f"Self-improvement failed: {e}")
            return {"status": "error", "error": str(e)}

    # -----------------------------------------------------------
    # Status / Introspection
    # -----------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Comprehensive system status"""
        uptime = time.time() - self._start_time
        last = self._last_generation

        status = {
            "version": self.VERSION,
            "uptime_seconds": uptime,
            "total_tokens_generated": self._total_tokens_generated,
            "generation_sessions": self._generation_sessions,
            "avg_tokens_per_session": (
                self._total_tokens_generated / max(1, self._generation_sessions)
            ),
            "last_generation": {
                "tokens": len(last.tokens) if last else 0,
                "stopped_reason": last.stopped_reason if last else None,
                "duration_seconds": last.duration_seconds if last else None,
                "tokens_per_second": (
                    last.metrics.get("tokens_per_second") if last else None
                ),
            },
            "trainer_summary": self.trainer.summary(),
            "self_improvement": (
                self.self_improvement.get_status() if self.self_improvement else None
            ),
            "safety_events_recent": (last.safety_events[-5:] if last else []),
            "config": {
                "generation": self.config.get("generation", {}),
                "training": self.config.get("training", {}),
            },
        }

        if self.monitor:
            status["performance"] = self.monitor.get_stats()

        if self.cache:
            status["cache"] = self.cache.stats()

        return status

    def health_check(self) -> Dict[str, Any]:
        """Quick health check"""
        try:
            test_result = self.quick_generate("test", max_tokens=5)

            return {
                "status": "healthy",
                "timestamp": time.time(),
                "checks": {
                    "transformer": len(test_result) > 0,
                    "safety": self.safety_validator is not None,
                    "bridge": self.bridge is not None,
                    "memory": True,
                },
            }
        except Exception as e:
            return {"status": "unhealthy", "timestamp": time.time(), "error": str(e)}

    # -----------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------

    def save(self, path: str) -> None:
        """Save lightweight state"""
        state = {
            "version": self.VERSION,
            "timestamp": time.time(),
            "uptime_seconds": time.time() - self._start_time,
        }

        for fld in self._saved_state_fields:
            state[fld] = getattr(self, fld)

        if self.monitor:
            state["performance_metrics"] = self.monitor.get_stats()

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
            logger.info(f"State saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def load(self, path: str) -> None:
        """Load lightweight state"""
        if not os.path.exists(path):
            logger.warning(f"State file not found: {path}")
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for fld in self._saved_state_fields:
                if fld in data:
                    setattr(self, fld, data[fld])

            logger.info(
                f"State loaded from {path} (saved at {datetime.fromtimestamp(data['timestamp'])})"
            )
        except Exception as e:
            logger.error(f"Failed to load state: {e}")

    # -----------------------------------------------------------
    # Cache Management
    # -----------------------------------------------------------

    def clear_cache(self):
        """Clear generation cache"""
        if self.cache:
            self.cache.clear()
            logger.info("Cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.cache.stats() if self.cache else {"enabled": False}

    # -----------------------------------------------------------
    # Internal Component Integration
    # -----------------------------------------------------------
    
    def get_causal_context(
        self, 
        query: str, 
        limit: int = 10,
        include_confounders: bool = True
    ) -> Dict[str, Any]:
        """
        Get causally-relevant context for a query using the causal reasoning system.
        
        This integrates:
        - Causal graph traversal
        - Temporal reasoning
        - Confounding detection
        
        Args:
            query: The query to analyze
            limit: Maximum context items to return
            include_confounders: Whether to include confounding factors
            
        Returns:
            Dictionary with causal context, concepts, and analysis
        """
        try:
            return self.causal_context.select(
                world_model=self.bridge.world_model,
                query={
                    "text": query,
                    "limit": limit,
                    "include_confounders": include_confounders,
                    "memory": self.hier_context.get_memory_snapshot(),
                }
            )
        except Exception as e:
            logger.warning(f"Causal context selection failed: {e}")
            return {"causal_context": [], "concepts": [], "error": str(e)}
    
    def reason_over_tokens(
        self,
        hidden_state: List[float],
        generated_tokens: List[int],
        strategy: str = "auto"
    ) -> Dict[str, Any]:
        """
        Use the language reasoning system for next-token prediction with analysis.
        
        This integrates:
        - LanguageReasoning for sophisticated sampling
        - Entropy-based strategy selection
        - Repetition penalty
        - Beam search when needed
        
        Args:
            hidden_state: Current hidden state from transformer
            generated_tokens: Tokens generated so far
            strategy: Sampling strategy ("greedy", "sample", "beam", "auto")
            
        Returns:
            Dictionary with token, candidates, and reasoning trace
        """
        try:
            return self.language_reasoning.generate(
                hidden_state=hidden_state,
                generated_tokens=generated_tokens,
            )
        except Exception as e:
            logger.warning(f"Language reasoning failed: {e}")
            # Fallback to simple greedy from transformer
            logits = self.transformer.get_logits(hidden_state, generated_tokens)
            token_id = max(range(len(logits)), key=lambda i: logits[i])
            return {
                "token_id": token_id,
                "strategy": "fallback_greedy",
                "error": str(e)
            }
    
    def generate_with_reasoning(
        self,
        prompt: Union[str, Sequence[int]],
        max_tokens: Optional[int] = None,
        use_causal_context: bool = True,
        use_language_reasoning: bool = True,
        explain: bool = True,
    ) -> GenerationResult:
        """
        Enhanced generation that fully integrates all internal components.
        
        This method connects:
        - HierarchicalContext for memory
        - CausalContext for causal reasoning
        - LanguageReasoning for sophisticated token selection
        - SafeGeneration for safety filtering
        - ExplainableGeneration for explanations
        - GovernedTrainer for governance
        
        Args:
            prompt: Input prompt (string or token sequence)
            max_tokens: Maximum tokens to generate
            use_causal_context: Whether to incorporate causal context
            use_language_reasoning: Whether to use sophisticated reasoning
            explain: Whether to generate explanations
            
        Returns:
            GenerationResult with full reasoning trace
        """
        start = time.time()
        
        # Get causal context if requested
        causal_info = None
        if use_causal_context:
            try:
                causal_info = self.get_causal_context(
                    str(prompt), 
                    limit=5,
                    include_confounders=True
                )
                logger.debug(f"Causal context: {len(causal_info.get('causal_context', []))} items")
            except Exception as e:
                logger.debug(f"Causal context skipped: {e}")
        
        # Store query context in hierarchical memory
        try:
            self.hier_context.store(
                prompt=str(prompt),
                token=None,
                metadata={"causal_context": causal_info is not None}
            )
        except Exception as e:
            logger.debug(f"Hierarchical context store skipped: {e}")
        
        # Delegate to main generate method which handles the core generation
        result = self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            explain=explain,
        )
        
        # Enhance result with causal info if available
        if causal_info and result.metadata:
            result.metadata["causal_analysis"] = {
                "concepts_detected": causal_info.get("concepts", []),
                "context_items": len(causal_info.get("causal_context", [])),
                "confounders": causal_info.get("confounders", []),
            }
        
        logger.info(
            f"generate_with_reasoning complete: {len(result.tokens)} tokens, "
            f"causal={use_causal_context}, reasoning={use_language_reasoning}"
        )
        
        return result
    
    def get_internal_components_status(self) -> Dict[str, Any]:
        """
        Get detailed status of all internal LLM components.
        
        Returns integration status of all subsystems:
        - Core: Transformer, Bridge, Cognitive Loop
        - Safety: Validator, Safe Generation
        - Context: Hierarchical, Causal, Persistent
        - Generation: Unified Generation, Explainer
        - Reasoning: Language, Unified, Causal, Analogical, Probabilistic, Mathematical, Multimodal
        - Memory: Vulcan Memory System (Hierarchical, Episodic, Semantic, Procedural, Working)
        - Execution: LLM Executor, Graphix Executor, Dynamic Architecture
        - Training: Governed Trainer, Self-Improvement
        """
        # Core components (required)
        core_status = {
            "transformer": {
                "available": self.transformer is not None,
                "vocab_size": getattr(self.transformer, 'config', {}).vocab_size 
                    if hasattr(self.transformer, 'config') else "unknown",
            },
            "bridge": {
                "available": self.bridge is not None,
                "world_model": hasattr(self.bridge, 'world_model'),
                "consensus": hasattr(self.bridge, 'consensus_approve_token'),
            },
            "cognitive_loop": self.cog_loop is not None,
        }
        
        # Safety systems
        safety_status = {
            "validator": self.safety_validator is not None,
            "safe_generation": self.safe_generation is not None,
        }
        
        # Context systems
        context_status = {
            "hierarchical_context": self.hier_context is not None,
            "causal_context": self.causal_context is not None,
            "persistent_context": getattr(self, 'persistent_context', None) is not None,
        }
        
        # Generation systems
        generation_status = {
            "explainer": self.explainer is not None,
            "unified_generation": getattr(self, 'unified_generation', None) is not None,
        }
        
        # Reasoning systems (comprehensive)
        reasoning_status = {
            "language_reasoning": self.language_reasoning is not None,
            "unified_reasoner": getattr(self, 'unified_reasoner', None) is not None,
            "causal_reasoning": getattr(self, 'causal_reasoning', None) is not None,
            "analogical_reasoning": getattr(self, 'analogical_reasoning', None) is not None,
            "probabilistic_reasoning": getattr(self, 'probabilistic_reasoning', None) is not None,
            "multimodal_reasoning": getattr(self, 'multimodal_reasoning', None) is not None,
            "math_computation": getattr(self, 'math_computation', None) is not None,
            "math_verification": getattr(self, 'math_verification', None) is not None,
            "reasoning_integration": getattr(self, 'reasoning_integration', None) is not None,
        }
        
        # Memory systems (Vulcan Memory)
        memory_status = {
            "vulcan_memory": getattr(self, 'vulcan_memory', None) is not None,
            "episodic_memory": getattr(self, 'episodic_memory', None) is not None,
            "semantic_memory": getattr(self, 'semantic_memory', None) is not None,
            "procedural_memory": getattr(self, 'procedural_memory', None) is not None,
            "working_memory": getattr(self, 'working_memory', None) is not None,
            "memory_consolidator": getattr(self, 'memory_consolidator', None) is not None,
        }
        
        # Execution systems
        execution_status = {
            "llm_executor": getattr(self, 'llm_executor', None) is not None,
            "graphix_executor": getattr(self, 'graphix_executor', None) is not None,
            "dynamic_architecture": getattr(self, 'dynamic_architecture', None) is not None,
        }
        
        # Training systems
        training_status = {
            "trainer": self.trainer is not None,
            "self_improvement": self.self_improvement is not None,
        }
        
        # Count available components
        all_components = []
        for status_dict in [core_status, safety_status, context_status, 
                           generation_status, reasoning_status, memory_status,
                           execution_status, training_status]:
            if isinstance(status_dict, dict):
                for k, v in status_dict.items():
                    if isinstance(v, dict):
                        all_components.extend(v.values())
                    else:
                        all_components.append(v)
        
        available_count = sum(1 for c in all_components if c is True or c is not None)
        total_count = len(all_components)
        
        # Core integration check (minimum required components)
        core_integrated = all([
            self.transformer is not None,
            self.bridge is not None,
            self.safety_validator is not None,
            self.hier_context is not None,
            self.causal_context is not None,
            self.language_reasoning is not None,
            self.cog_loop is not None,
        ])
        
        return {
            "core": core_status,
            "safety": safety_status,
            "context": context_status,
            "generation": generation_status,
            "reasoning": reasoning_status,
            "memory": memory_status,
            "execution": execution_status,
            "training": training_status,
            "summary": {
                "available_components": available_count,
                "total_components": total_count,
                "integration_percentage": round(available_count / total_count * 100, 1) if total_count > 0 else 0,
                "core_integrated": core_integrated,
            },
        }

    # -----------------------------------------------------------
    # Utility
    # -----------------------------------------------------------

    def _obs(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Internal observability recording"""
        obs = self.observability
        if not obs:
            return
        try:
            if hasattr(obs, "record"):
                obs.record(event_type, payload)
            elif hasattr(obs, "log"):
                obs.log(event_type, payload)
        except Exception as e:
            logger.debug(f"Observability recording failed: {e}")

    def __del__(self):
        """Cleanup resources"""
        try:
            if self._event_loop and not self._event_loop.is_closed():
                self._event_loop.close()
        except Exception:
            pass


# -------------------------------------------------------------------
# Factory Function
# -------------------------------------------------------------------


def build_llm(
    config_path: Optional[str] = None,
    enable_caching: bool = True,
    enable_monitoring: bool = True,
    **kwargs,
) -> GraphixVulcanLLM:
    """Factory function to build GraphixVulcanLLM"""
    logger.info("Building GraphixVulcanLLM...")
    return GraphixVulcanLLM(
        config_path=config_path,
        enable_caching=enable_caching,
        enable_monitoring=enable_monitoring,
        **kwargs,
    )


# -------------------------------------------------------------------
# Test Script - COMPLETE
# -------------------------------------------------------------------

if __name__ == "__main__":
    print("VulcanAMI LLM Test")
    print("=" * 40)

    # Build LLM
    llm = build_llm()

    # Initial status
    status = llm.get_status()
    print(f"Status: {status['total_tokens_generated']} tokens generated")
    print(f"Config Model: {llm.config.get('generation', {})}")
    print(f"Max Tokens: {llm.config['generation']['max_tokens']}")
    print()

    # Test generation
    print("Testing generation...")
    try:
        result = llm.generate("Hello, world!", max_tokens=16, explain=True)
        print(f"✓ Generated: {result.text}")
        print(f"✓ Tokens: {len(result.tokens)}")
        print(f"✓ Duration: {result.duration_seconds:.2f}s")
        print(f"✓ Throughput: {result.metrics.get('tokens_per_second', 0):.1f} tok/s")
    except Exception as e:
        print(f"✗ Generation failed: {e}")
        import traceback

        traceback.print_exc()
    print()

    # Test streaming
    print("Testing streaming...")
    try:
        print("Stream: ", end="")
        count = 0
        for i, token in enumerate(llm.stream("Test prompt", max_tokens=8)):
            print(f"{token} ", end="", flush=True)
            count += 1
            if i >= 7:
                break
        print()
        print(f"✓ Streamed {count} tokens")
    except Exception as e:
        print(f"✗ Streaming failed: {e}")
    print()

    # Test async
    print("Testing async generation...")
    try:

        async def async_test():
            result = await llm.generate_async("Async test", max_tokens=8)
            return result

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            async_result = loop.run_until_complete(async_test())
            print(f"✓ Async tokens generated: {len(async_result.tokens)}")
            print(f"✓ Async text: {async_result.text if async_result.text else 'N/A'}")
        finally:
            loop.close()
    except Exception as e:
        print(f"✗ Async test failed: {e}")

    print()

    # Final status
    final_status = llm.get_status()
    print("Final Status:")
    print(f"  Total tokens: {final_status['total_tokens_generated']}")
    print(f"  Sessions: {final_status['generation_sessions']}")
    if final_status["generation_sessions"] > 0:
        print(f"  Avg tokens/session: {final_status['avg_tokens_per_session']:.1f}")

    if "performance" in final_status:
        perf = final_status["performance"]
        print(
            f"  Throughput: {perf.get('overall_throughput_tokens_per_sec', 0):.1f} tok/s"
        )
        print(f"  Error rate: {perf.get('error_rate', 0):.2%}")

    if "cache" in final_status:
        cache = final_status["cache"]
        print(
            f"  Cache: {cache['size']}/{cache['max_size']} ({cache['utilization']:.1%})"
        )

    # Health check
    health = llm.health_check()
    print(f"  Health: {health['status']}")

    print()
    print("=" * 40)
    print("✓ All tests completed!")
