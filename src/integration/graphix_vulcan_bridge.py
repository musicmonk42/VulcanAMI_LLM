from __future__ import annotations
"""
Graphix–VULCAN Bridge (2025 Production Core)

Orchestrates EXAMINE → SELECT → APPLY → REMEMBER cognitive phases asynchronously.
Features:
- Fully asynchronous phase methods.
- Functional HierarchicalMemory (in-memory vector store) with caching.
- Functional WorldModelCore (enhanced state/intervention logic).
- Robust async error handling with retry (_safe_call_async).
- Comprehensive Audit and Observability hooks with KL divergence tracking.
"""

import asyncio
import time
import logging
import inspect
import threading
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
import math
import random 
import torch
import torch.nn as nn
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor

log = logging.getLogger(__name__)

# --- Configuration Constants (Defaults) ---
_ASYNC_TIMEOUT = 2.0
_EMBEDDING_DIM = 256 
_MEMORY_CAPACITY = 100 
_KL_GUARD_THRESHOLD = 0.05 
_MAX_RETRIES = 3 
_VOCAB_SIZE = 5000 
_CACHE_TTL_SECONDS = 60.0
DEVICE = torch.device("cpu") # Use CPU for thread-based embedding

# A. FIX: Expose configuration instead of hard-coding
@dataclass
class BridgeConfig:
    async_timeout: float = _ASYNC_TIMEOUT
    embedding_dim: int = _EMBEDDING_DIM
    memory_capacity: int = _MEMORY_CAPACITY
    kl_guard_threshold: float = _KL_GUARD_THRESHOLD
    max_retries: int = _MAX_RETRIES
    vocab_size: int = _VOCAB_SIZE
    cache_ttl_seconds: float = _CACHE_TTL_SECONDS
    consensus_timeout_seconds: float = 2.0  # Added consensus timeout configuration
    
    def __post_init__(self):
        """Validate configuration parameters."""
        # Define validation rules: (field_name, allow_zero, allow_negative)
        validations = [
            ('async_timeout', False, False),
            ('embedding_dim', False, False),
            ('memory_capacity', False, False),
            ('kl_guard_threshold', True, False),  # Can be zero
            ('max_retries', True, False),  # Can be zero (no retries)
            ('vocab_size', False, False),
            ('cache_ttl_seconds', False, False),
            ('consensus_timeout_seconds', False, False),
        ]
        
        for field_name, allow_zero, allow_negative in validations:
            value = getattr(self, field_name)
            if not allow_negative and value < 0:
                raise ValueError(f"{field_name} must be non-negative, got {value}")
            if not allow_zero and value <= 0:
                raise ValueError(f"{field_name} must be positive, got {value}")


# ------------------------ Functional VULCAN Components ------------------------ #

def _normalize_vector(v: List[float]) -> List[float]:
    # Handles list input for non-PyTorch operations
    norm = math.sqrt(sum(x * x for x in v))
    return [x / (norm + 1e-6) for x in v]

class WorldModelCore:
    """Implements core world model functionality with safety/intervention hooks."""
    def __init__(self):
        self._state: Dict[str, Any] = {"status": "nominal"}
        self._concept_registry: Dict[str, int] = {} 
        self._state["prev_logits"]: Optional[List[float]] = None # KL tracking

    async def update(self, obs: Any) -> None:
        """Async update model state based on observation."""
        await asyncio.sleep(0.01) # Simulate async I/O
        self._state["last_obs"] = obs
        self._state["timestamp"] = time.time()

    async def update_from_text(self, tokens: List[Any], predictions: Optional[Dict[str, Any]] = None) -> None:
        """
        Async update based on generated text and predictions.
        Enhancement: KL tracking using previous logits if available.
        """
        await asyncio.sleep(0.01)
        self._state["last_tokens"] = tokens
        
        # Update concept frequency
        for token in tokens:
            t_str = str(token).lower()
            if len(t_str) > 3:
                self._concept_registry[t_str] = self._concept_registry.get(t_str, 0) + 1

        # KL Tracking (ENHANCEMENT)
        kl_value = None
        if predictions and predictions.get("logits") and self._state.get("prev_logits"):
            try:
                # Assuming logits are lists of floats
                current_logits = predictions["logits"]
                prev_logits = self._state["prev_logits"]
                
                # Convert to tensors
                P = F.softmax(torch.tensor(prev_logits, device=DEVICE), dim=-1)
                Q = F.log_softmax(torch.tensor(current_logits, device=DEVICE), dim=-1)
                
                # Compute KL divergence (KL(P || Q) - D_KL(P || Q) = sum(P * log(P/Q)))
                kl_tensor = F.kl_div(Q, P, reduction='sum', log_target=True) 
                kl_value = kl_tensor.item()
                
                # OBS for KL tracking
                bridge = WorldModelCore._get_bridge() # Requires getting bridge instance
                if bridge and bridge._observability:
                    await bridge._obs("worldmodel.kl_tracking", {"kl": kl_value, "token_count": len(tokens)})
                
            except Exception as e:
                log.debug(f"KL computation failed: {e}")

        # Store current logits for the next step's comparison (assuming single token prediction logits)
        if predictions and predictions.get("logits"):
             self._state["prev_logits"] = predictions["logits"]
        else:
             self._state["prev_logits"] = None


    async def validate_generation(self, token: Any, context: Any) -> bool:
        """Async check for semantic or factual consistency."""
        await asyncio.sleep(0.005)
        
        token_str = str(token).lower()
        
        # Repetition check (ENHANCEMENT)
        prompt_tokens = context.get("prompt_tokens", [])
        if token_str in [str(t).lower() for t in prompt_tokens[-5:]]:
            log.warning(f"WorldModel flag: Repetition ('{token_str}') detected in last 5 tokens.")
            return False

        if token_str.endswith(("ing", "ed")) and "noun" in context.get("strategy", "").lower():
            log.warning(f"WorldModel flag: Verb form ('{token_str}') used in noun strategy context.")
            return False

        if token_str in ["error", "incorrect"]:
            return False
            
        return True

    async def suggest_correction(self, token: Any, context: Any) -> Any:
        """Async suggest replacement for a validated failure."""
        await asyncio.sleep(0.005)
        
        token_str = str(token).lower()
        
        # Suggest correction based on most frequent concept (ENHANCEMENT)
        if self._concept_registry:
            top_concept = max(self._concept_registry, key=self._concept_registry.get)
            
            # Suggest the top concept only if it starts with the same letter
            if token_str.startswith(top_concept[0]):
                return top_concept.upper()
            
        return f"CORRECTED_{token_str.upper()}"

    async def intervene_before_emit(self, token: Any, context: Any, hidden_state: Any) -> Optional[Dict[str, Any]]:
        """Async final safety check, returns modification dict if intervention is needed."""
        await asyncio.sleep(0.005)
        if str(token) == "UPPER":
            return {"modified_token": "upper", "notes": "Forced lowercase for tone consistency."}
        return None

    def explain(self, concept: str) -> str:
        return f"Explanation for {concept} (world model). Registry size: {len(self._concept_registry)}"
    
    @staticmethod
    def _get_bridge():
        """Helper to get the bridge instance from the global module context (unsafe but common for shared state access)."""
        # This assumes the bridge instance is stored somewhere accessible, e.g., a shared list or global variable
        # In this context, we rely on external systems to link the WorldModelCore back to the Bridge,
        # but for simulation, we return None if not externally managed.
        return GraphixVulcanBridge._instance


class HierarchicalMemory:
    """Implements in-memory vector store mimicking HierarchicalMemory."""
    def __init__(self, config: BridgeConfig):
        self.config = config
        
        # Store as: (text, vector - stored as a torch tensor)
        # To handle batch retrieval, episodic will store (text, index) and embeddings will be in a tensor.
        self.episodic: List[Tuple[str, int]] = []
        self._embedding_tensor: Optional[torch.Tensor] = None # Stores all embeddings [N, D]
        
        # A. FIX: Initialize nn.Embedding layer
        self.embedder = nn.Embedding(self.config.vocab_size, self.config.embedding_dim, device=DEVICE)
        
        # Caching: {query: (context_dict, timestamp)}
        self._cache: Dict[str, Tuple[Dict[str, Any], float]] = {}
        self._cache_capacity = 10 # Retain existing capacity constant

    def _tokenize_to_ids(self, text: str) -> torch.Tensor:
        """Simple hash-based tokenizer/indexer for simulation."""
        ids = [sum(ord(c) for c in word) % (self.config.vocab_size - 1) + 1 for word in text.split() if word]
        # Pad or truncate to a fixed length if necessary, here we just return the sequence
        return torch.tensor(ids, dtype=torch.long, device=DEVICE)

    @torch.no_grad()
    def _embed_text(self, text: str) -> torch.Tensor:
        """Torch-based text embedding (embedding -> average pooling)."""
        if not text:
            return torch.zeros(self.config.embedding_dim, device=DEVICE)

        ids_tensor = self._tokenize_to_ids(text).unsqueeze(0) # [1, S]
        
        embeds = self.embedder(ids_tensor) # [1, S, D]
        
        # Average pooling (ENHANCEMENT)
        avg_embed = embeds.mean(dim=1).squeeze(0) # [D]
        
        # Normalize and return
        norm_embed = avg_embed / (torch.linalg.norm(avg_embed) + 1e-6)
        return norm_embed

    async def aretrieve_context(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Async retrieval using PyTorch cosine similarity and TTL cache."""
        await asyncio.sleep(0.05) # Simulate retrieval latency
        
        # 1. Check TTL cache (ENHANCEMENT)
        cache_hit = False
        if query in self._cache:
            context, timestamp = self._cache[query]
            if (time.time() - timestamp) < self.config.cache_ttl_seconds:
                cache_hit = True
                # Update timestamp to refresh TTL
                self._cache[query] = (context, time.time())
                return context, cache_hit
            # Cache expired
            del self._cache[query]

        query_vec = self._embed_text(query).unsqueeze(0) # [1, D]
        context_items = []

        if self._embedding_tensor is not None and len(self.episodic) > 0:
            # Enhance retrieval: Use torch.matmul for batch cosine similarity
            episodic_embs = self._embedding_tensor # [N, D]
            
            # Cosine similarity (dot product of normalized vectors)
            # score_tensor [1, N]
            score_tensor = torch.matmul(query_vec, episodic_embs.T) 
            
            # Get top_k scores and indices
            top_k_scores, top_k_indices = torch.topk(score_tensor, min(top_k, episodic_embs.shape[0]), dim=1)
            
            # Extract retrieved texts
            for idx in top_k_indices.squeeze(0).tolist():
                 context_items.append(self.episodic[idx][0])

        retrieved_context = {"episodic": context_items, "semantic": [], "procedural": []}
        
        # 2. Update cache (ENHANCEMENT)
        if len(self._cache) >= self._cache_capacity:
            # Simple eviction: remove oldest
            oldest_query = min(self._cache, key=lambda q: self._cache[q][1])
            del self._cache[oldest_query]
            
        self._cache[query] = (retrieved_context, time.time())
        
        return retrieved_context, cache_hit

    async def astore_generation(self, prompt: str, generated: str, reasoning_trace: Any) -> None:
        """Async storage of generated content."""
        full_text = f"Prompt: {prompt}\nGeneration: {generated}"
        vector = self._embed_text(full_text) # [D]
        
        # Update episodic list and embedding tensor (synchronously, in a thread if needed)
        
        # NOTE: Using synchronous operation inside the async store is standard practice if the operation is fast or wrapped in to_thread, 
        # but here we wrap the storage mechanism in a thread to simulate potentially heavy vector database writes.
        
        def _sync_store():
            nonlocal vector
            
            # 1. Update episodic list
            new_index = len(self.episodic)
            self.episodic.append((full_text, new_index))
            
            # 2. Update embedding tensor (ENHANCEMENT)
            vector = vector.unsqueeze(0) # [1, D]
            if self._embedding_tensor is None:
                self._embedding_tensor = vector
            else:
                self._embedding_tensor = torch.cat([self._embedding_tensor, vector], dim=0) # [N+1, D]
            
            # 3. Enforce capacity
            if len(self.episodic) > self.config.memory_capacity:
                self.episodic.pop(0)
                # Rebuild embedding tensor to maintain consistency
                self._embedding_tensor = self._embedding_tensor[1:]

        await asyncio.to_thread(_sync_store)
        await asyncio.sleep(0.01)

    async def store(self, result: Any) -> None:
        """Fallback async storage."""
        if isinstance(result, dict) and "tokens" in result and isinstance(result["tokens"], str):
            await self.astore_generation(result.get("prompt", ""), result["tokens"], result.get("reasoning_trace", {}))
        else:
            await asyncio.sleep(0.01) # Simulate work


class UnifiedReasoning:
    """Minimal shim, assume methods are async/to_thread handled by the bridge."""
    async def select_strategy(self, node: Any, context: Any) -> str:
        await asyncio.sleep(0.001)
        return "language"
        
    async def select_next_token(self, hidden_state: Any, context: Any) -> List[Any]:
        await asyncio.sleep(0.001)
        if isinstance(hidden_state, (int, float, str)):
            return [hidden_state, f"{hidden_state}_alt"]
        return [hidden_state]
        
    async def explain_choice(self, token: Any, hidden_state: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        await asyncio.sleep(0.005)
        return {"reason": "deep_reasoning_explanation", "token": token, "context_hash": hash(str(context)[:20])}

# ------------------------ Data structures ------------------------ #

@dataclass
class BridgeContext:
    """
    Context bundle returned by before_execution and threaded through the cycle.
    """
    raw_input: Any
    memory: Dict[str, Any] = field(default_factory=dict)
    world_state: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


# ------------------------ Bridge implementation ------------------------ #

class GraphixVulcanBridge:
    """
    Connects Graphix execution with VULCAN cognitive control phases.
    Implements Singleton pattern to ensure only one instance exists.
    """
    # Store the instance reference for the WorldModelCore to access for logging/bridge calls
    _instance: Optional[GraphixVulcanBridge] = None
    _lock: threading.Lock = threading.Lock()
    _initialized: bool = False
    
    def __new__(cls, *args, **kwargs):
        """
        Thread-safe Singleton implementation.
        Ensures only one instance of the bridge exists in the runtime.
        """
        if not cls._instance:
            with cls._lock:
                # Double-checked locking to ensure thread safety
                if not cls._instance:
                    cls._instance = super(GraphixVulcanBridge, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, config: Optional[BridgeConfig] = None) -> None:
        """
        Initialize the bridge.
        Note: The check for _initialized prevents re-running the setup logic
        if the object is retrieved a second time.
        """
        if getattr(self, "_initialized", False):
            log.warning("GraphixVulcanBridge accessed (already initialized). Returning existing instance.")
            return
             
        self.config = config or BridgeConfig()
        
        self.world_model: WorldModelCore = WorldModelCore()
        self.reasoning: UnifiedReasoning = UnifiedReasoning()
        self.memory: HierarchicalMemory = HierarchicalMemory(self.config)

        self._safety: Optional[Any] = None
        self._observability: Optional[Any] = None
        self._audit_log: Optional[Any] = None
        self._consensus: Optional[Any] = None
        self._executor: Optional[ThreadPoolExecutor] = None # For sync obs

        self._last_context: Optional[BridgeContext] = None
        
        # Link WorldModelCore back to this bridge for logging purposes
        WorldModelCore._get_bridge = lambda: self
        
        # Mark as initialized to prevent re-execution of this block
        self._initialized = True
        
        log.info("GraphixVulcanBridge initialized successfully (Singleton instance)") 

    # ------------------------ Setup/injection helpers ------------------------ #

    def attach_safety(self, safety_validator: Any) -> None:
        self._safety = safety_validator

    def set_observability_manager(self, observability_manager: Any) -> None:
        self._observability = observability_manager

    def set_audit_log(self, audit_log: Any) -> None:
        self._audit_log = audit_log

    def set_consensus_engine(self, consensus_engine: Any) -> None:
        self._consensus = consensus_engine
        
    def _get_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=2) # Minimal executor for obs/audit
        return self._executor

    # ------------------------ Async Safe Call Helper (ENHANCED with Retry) ------------------------ #

    async def _safe_call_async(
        self, 
        fn: Optional[Union[Callable, Any]], 
        args: Any, 
        default: Any, 
        timeout: Optional[float] = None, 
        max_retries: Optional[int] = None
    ) -> Any:
        
        # Use provided timeout/retries or fall back to config defaults
        timeout = timeout if timeout is not None else self.config.async_timeout
        max_retries = max_retries if max_retries is not None else self.config.max_retries
        
        if fn is None:
            return default
        
        args_tuple = args if isinstance(args, tuple) else (args,)
        
        for attempt in range(max_retries + 1):
            try:
                # Determine awaitable
                if asyncio.iscoroutinefunction(fn):
                    awaitable = fn(*args_tuple)
                elif hasattr(fn, '__call__'):
                    # Assume synchronous function that needs to be run in a thread
                    awaitable = asyncio.to_thread(fn, *args_tuple)
                else:
                    raise TypeError(f"Function {fn} is not callable.")

                return await asyncio.wait_for(awaitable, timeout)
            
            except asyncio.TimeoutError:
                if attempt == max_retries:
                    await self._obs("bridge.timeout_final", {"function": fn.__name__ if hasattr(fn, '__name__') else "unknown", "attempts": attempt + 1})
                    return default
                
                # ENHANCEMENT: Retry on timeout with exponential backoff
                log.warning(f"Timeout on {fn.__name__ if hasattr(fn, '__name__') else 'unknown'}. Retrying in {0.05 * (2 ** attempt):.2f}s...")
                await asyncio.sleep(0.05 * (2 ** attempt))
            
            except Exception as e:
                # FIX: Retry on all exceptions
                await self._obs("bridge.exception", {
                    "function": fn.__name__ if hasattr(fn, '__name__') else str(fn),
                    "error": str(e),
                    "attempt": attempt,
                })
                
                if attempt == max_retries:
                    return default

                # Small backoff for transient, non-timeout errors
                await asyncio.sleep(0.05 * (attempt + 1))


        return default

    # ------------------------ Phase: EXAMINE (ASYNC) ------------------------ #

    async def before_execution(self, graph_or_tokens: Any) -> Dict[str, Any]:
        observation = self._normalize_observation(graph_or_tokens)

        # 1. World model update
        await self._safe_call_async(self.world_model.update, observation, default=None)

        # 2. Retrieve memory context 
        tokens = observation.get("tokens") or []
        query = observation.get("prompt") or " ".join(tokens)
        
        mem_tuple, cache_hit = await self._safe_call_async(
            self.memory.aretrieve_context, 
            (query, 5), 
            default=({}, False), 
        )
        mem = mem_tuple or {"episodic": [], "semantic": [], "procedural": []}
        
        # 3. Compose world_state snapshot
        world_state = {"timestamp": time.time(), "status": self.world_model._state.get("status")}

        ctx = BridgeContext(raw_input=observation, memory=mem, world_state=world_state)
        self._last_context = ctx

        await self._obs("bridge.before_execution", {
            "mem_keys": list(mem.keys()) if isinstance(mem, dict) else [], 
            "tokens": len(tokens), 
            "cache_hit": cache_hit,
            "episodic_size": len(self.memory.episodic), 
        })
        await self._audit("bridge.before_execution", {
            "query": query, 
            "retrieved_count": len(mem.get("episodic", [])), 
            "memory_hit_rate": cache_hit, 
        })
        return asdict(ctx)

    # ------------------------ Phase: SELECT (ASYNC) ------------------------ #

    async def during_execution(self, node: Any, context: Any) -> str:
        strategy = await self._safe_call_async(self.reasoning.select_strategy, (node, context), default="language")
        await self._obs("bridge.during_execution", {"strategy": strategy})
        await self._audit("bridge.during_execution", {"node_id": node.get("id", "n/a"), "strategy": strategy})
        return strategy

    # ------------------------ Phase: REMEMBER (ASYNC) ------------------------ #

    async def after_execution(self, result: Any) -> None:
        # 1. Store to hierarchical memory
        if isinstance(result, dict):
            prompt = result.get("prompt")
            generated = result.get("token") or result.get("tokens") or result.get("text")
            reasoning_trace = result.get("reasoning_trace")
            
            if generated and isinstance(generated, str):
                await self._safe_call_async(
                    self.memory.astore_generation,
                    (prompt, generated, reasoning_trace),
                    default=None
                )
            else:
                await self._safe_call_async(self.memory.store, result, default=None)
        else:
            await self._safe_call_async(self.memory.store, result, default=None)


        # 2. World model update from text
        tokens = []
        predictions = None
        if isinstance(result, dict):
            tokens = result.get("tokens") or result.get("prompt_tokens")
            predictions = result.get("predictions")
            
        if isinstance(tokens, list):
            await self._safe_call_async(self.world_model.update_from_text, (tokens, predictions), default=None)

        # 3. Observability: Compute KL if predictions are available
        kl_div = 0.0
        if predictions and isinstance(predictions, dict) and "draft_logits" in predictions and "main_logits" in predictions:
            draft_logits = predictions["draft_logits"]
            main_logits = predictions["main_logits"]
            
            if draft_logits and main_logits:
                kl_div = abs(math.log(sum(draft_logits) / len(draft_logits) + 1e-6) - math.log(sum(main_logits) / len(main_logits) + 1e-6))
        
        # E. FIX: KL Guard Enforcement
        if kl_div > self.config.kl_guard_threshold:
            await self._obs("bridge.kl_guard_triggered", {"kl_div": kl_div, "threshold": self.config.kl_guard_threshold})
            self.world_model._state["status"] = "kl_guard"
        
        await self._audit("bridge.after_execution", {"status": "memory_updated", "kl_div_sim": kl_div})
        await self._obs("bridge.after_execution", {"status": self.world_model._state["status"], "kl_div_sim": kl_div})

    # ------------------------ Reasoning/Safety helpers (ASYNC) ------------------------ #

    async def reason_next_token(self, hidden_state: Any, context: Dict[str, Any]) -> List[Any]:
        cands = await self._safe_call_async(self.reasoning.select_next_token, (hidden_state, context), default=None)
        if cands is None:
            cands = [hidden_state]
        elif not isinstance(cands, (list, tuple)):
            cands = [cands]
            
        await self._audit("bridge.reason_next_token", {"num_cands": len(cands)})
        return list(cands)

    async def validate_token(self, token: Any, context: Dict[str, Any], hidden_state: Any) -> Tuple[Any, Optional[Dict[str, Any]]]:
        original = token
        notes: Optional[Dict[str, Any]] = None

        if self._safety and hasattr(self._safety, "validate_generation"):
            token = await self._safe_call_async(self._safety.validate_generation, (token, context, self.world_model), default=token)

        ok = await self._safe_call_async(self.world_model.validate_generation, (token, context), default=True)
        if not ok:
            token = await self._safe_call_async(self.world_model.suggest_correction, (token, context), default=token)
            await self._obs("bridge.token_inconsistency", {"original": str(original), "reason": "WorldModel"})
            
        intervention = await self._safe_call_async(self.world_model.intervene_before_emit, (token, context, hidden_state), default=None)
        if intervention and isinstance(intervention, dict):
            if 'modified_token' in intervention:
                token = intervention['modified_token']
            notes = intervention

        if original != token:
            await self._obs("bridge.token_replaced", {"original": str(original), "replacement": str(token)})
            await self._audit("bridge.token_replaced", {"original": str(original), "replacement": str(token), "notes": notes})
            
        return token, notes

    async def consensus_approve_token(self, token: Any, position: int, chosen_index: Optional[int] = None) -> bool:
        if not self._consensus or not hasattr(self._consensus, "approve"):
            await self._obs("bridge.consensus_skipped", {"position": position})
            return True
        
        proposal = {
            "type": "token_emission",
            "token": token,
            "position": position,
            "chosen_index": chosen_index,
            "timestamp": time.time(),
            "world_state": self.world_model._state.copy(),
            "bridge_context": asdict(self._last_context) if self._last_context else {},
        }
        
        # Use configured consensus timeout
        approved = await self._safe_call_async(
            self._consensus.approve, 
            proposal, 
            default=True, 
            timeout=self.config.consensus_timeout_seconds
        ) 
        
        await self._obs("bridge.consensus", {"approved": bool(approved)})
        return bool(approved)

    # ------------------------ Utilities (Async and Sync methods) ------------------------ #

    async def validate_sequence(self, tokens: List[Any], context: Dict[str, Any]) -> Union[bool, List[Any]]:
        if self._safety and hasattr(self._safety, "validate_sequence"):
            return await self._safe_call_async(self._safety.validate_sequence, (tokens, context, self.world_model), default=True)
        return True

    async def explain_choice(self, token: Any, hidden_state: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        exp = await self._safe_call_async(self.reasoning.explain_choice, (token, hidden_state, context), default=None)
        
        if isinstance(exp, dict):
            return exp
        return {"explanation": exp if exp is not None else "unavailable"}

    def _normalize_observation(self, inp: Any) -> Dict[str, Any]:
        if isinstance(inp, dict):
            obs = dict(inp)
            if "prompt_tokens" in obs and "tokens" not in obs:
                obs["tokens"] = obs["prompt_tokens"]
            return obs
        if isinstance(inp, str):
            return {"prompt": inp, "tokens": inp.split()}
        if isinstance(inp, (list, tuple)):
            return {"tokens": list(inp)}
        return {"input": inp}

    # ASYNC OBSERVABILITY (ENHANCEMENT)
    async def _obs(self, event_type: str, payload: Dict[str, Any]) -> None:
        if not self._observability:
            return
        fn = getattr(self._observability, "record", getattr(self._observability, "log", None))
        if fn:
            try:
                if inspect.iscoroutinefunction(fn):
                    await fn(event_type, payload)
                else:
                    # Run sync observation in a thread
                    await asyncio.to_thread(fn, event_type, payload)
            except Exception as e:
                # Log observability failures for debugging (non-critical)
                log.debug(f"Observability recording failed for {event_type}: {e}")

    # ASYNC AUDIT (ENHANCEMENT)
    async def _audit(self, event_type: str, payload: Dict[str, Any]) -> None:
        if not self._audit_log:
            return
        fn = getattr(self._audit_log, "append", getattr(self._audit_log, "record", None))
        
        if fn:
            record_payload = {"event": event_type, **payload} if getattr(self._audit_log, "append", None) else payload
            try:
                if inspect.iscoroutinefunction(fn):
                    await fn(record_payload if getattr(self._audit_log, "append", None) else event_type, record_payload if not getattr(self._audit_log, "append", None) else None)
                else:
                    # Run sync audit in a thread
                    await asyncio.to_thread(fn, record_payload if getattr(self._audit_log, "append", None) else event_type, record_payload if not getattr(self._audit_log, "append", None) else None)
            except Exception as e:
                # Log audit failures for debugging (non-critical)
                log.debug(f"Audit logging failed for {event_type}: {e}")