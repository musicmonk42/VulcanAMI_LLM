"""
Memory-based Bayesian Prior for Tool Selection

Computes prior probabilities for tool selection based on historical performance
stored in the memory system, using similarity-weighted Bayesian inference.

Fixed version with proper resource management and cache eviction.
"""

import logging
import os
import pickle
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ============================================================
# BUG #4 FIX: Global singleton for embeddings
# ============================================================
# This ensures the embedding model is loaded ONCE per process,
# preventing 50ms→6400ms performance degradation from repeated loading.
# ============================================================

_GLOBAL_EMBEDDING_MODEL = None
_EMBEDDING_LOCK = threading.Lock()


def get_global_embedding_model():
    """
    Get or create the global embedding model singleton.
    
    BUG #4 FIX: This function ensures the SentenceTransformer model
    is loaded only ONCE per process, preventing expensive reloads that
    cause performance degradation.
    
    Returns:
        SentenceTransformer model instance, or None if unavailable.
    """
    global _GLOBAL_EMBEDDING_MODEL
    if _GLOBAL_EMBEDDING_MODEL is None:
        with _EMBEDDING_LOCK:
            # Double-check after acquiring lock
            if _GLOBAL_EMBEDDING_MODEL is None:
                try:
                    from sentence_transformers import SentenceTransformer
                    _GLOBAL_EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
                    logger.info("[Embeddings] Loaded model ONCE (global singleton)")
                except ImportError as e:
                    logger.warning(f"[Embeddings] sentence-transformers not available: {e}")
                    return None
                except Exception as e:
                    logger.error(f"[Embeddings] Failed to load model: {e}")
                    return None
    return _GLOBAL_EMBEDDING_MODEL


# Import semantic tool matcher for query-based tool selection
try:
    from .semantic_tool_matcher import SemanticToolMatcher
    SEMANTIC_MATCHER_AVAILABLE = True
except ImportError:
    SEMANTIC_MATCHER_AVAILABLE = False
    SemanticToolMatcher = None
    logger.warning("SemanticToolMatcher not available")

# Configuration constants for semantic tool matching
MIN_QUERY_LENGTH_FOR_SEMANTIC_BOOST = 10  # Minimum query text length to apply semantic boost
MULTIMODAL_CONTENT_BOOST = 0.4  # Strong boost for explicit multimodal content

# PERFORMANCE FIX Issue #4: Re-enable semantic matching now that caching is optimized
# 
# DEFAULT: ENABLED (semantic matching on) - safe now that model caching works properly
# 
# Set VULCAN_DISABLE_SEMANTIC_MATCHING=1 to DISABLE embedding-based tool selection
# 
# Background: Semantic matching was previously causing 6-30 second delays per query
# due to repeated model loading. With the following fixes now in place, it's safe to enable:
#   - Issue #2: HierarchicalMemory singleton pattern (models loaded once)
#   - Issue #3: Batch processing optimized with show_progress_bar=False
#   - Model registry caching ensures SentenceTransformer loads once per process
#
# With these fixes, semantic matching adds only ~0.1-0.5s per query while providing
# more accurate tool selection than keyword-only matching.
#
# To disable semantic matching (if you encounter performance issues):
#   export VULCAN_DISABLE_SEMANTIC_MATCHING=1
DISABLE_SEMANTIC_MATCHING = os.environ.get("VULCAN_DISABLE_SEMANTIC_MATCHING", "0").lower() in ("1", "true", "yes")


class SimilarityMetric(Enum):
    """Similarity metrics for memory retrieval"""

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    JACCARD = "jaccard"
    SEMANTIC = "semantic"
    WEIGHTED = "weighted"


class PriorType(Enum):
    """Types of priors"""

    UNIFORM = "uniform"
    BETA = "beta"
    DIRICHLET = "dirichlet"
    EMPIRICAL = "empirical"
    HIERARCHICAL = "hierarchical"


@dataclass
class MemoryEntry:
    """Entry in memory system"""

    entry_id: str
    problem_features: np.ndarray
    tool_used: str
    success: bool
    confidence: float
    execution_time: float
    energy_used: float
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PriorDistribution:
    """Prior distribution for tool selection"""

    tool_probs: Dict[str, float]
    confidence: float
    support_count: int
    entropy: float
    most_likely_tool: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class MemoryIndex:
    """Fast similarity search index for memory entries"""

    def __init__(self, metric: SimilarityMetric = SimilarityMetric.COSINE):
        self.metric = metric
        self.entries = []
        self.features = None
        self.index_built = False

        # CRITICAL FIX: Add max entries limit to prevent unbounded growth
        self.max_entries = 10000

        # Try to import faiss for fast similarity search
        try:
            from src.utils.faiss_config import initialize_faiss

            # Initialize once and store results
            faiss_module, is_available, _ = initialize_faiss()
            self.use_faiss = is_available
            self.faiss = faiss_module
            self.faiss_index = None

            if not self.use_faiss:
                logger.info("FAISS not available, using numpy for similarity search")
        except ImportError:
            logger.info(
                "FAISS configuration not available, using numpy for similarity search"
            )
            self.use_faiss = False
            self.faiss = None

    def add(self, entry: MemoryEntry):
        """Add entry to index with size limit"""

        # CRITICAL FIX: Limit index size
        if len(self.entries) >= self.max_entries:
            # Remove oldest entries
            self.entries = self.entries[-(self.max_entries - 1) :]
            self.index_built = False

        self.entries.append(entry)
        self.index_built = False

    def build_index(self):
        """Build search index - CRITICAL: Prevent memory leak"""

        if not self.entries:
            return

        try:
            # CRITICAL FIX: Clear old index first to prevent memory leak
            if self.use_faiss and self.faiss_index is not None:
                # Explicitly delete old FAISS index
                del self.faiss_index
                self.faiss_index = None

            # Clear old features array
            if self.features is not None:
                del self.features
                self.features = None

            # Stack features
            # FIX: Cast to float32 at creation time for faiss compatibility
            self.features = np.vstack(
                [e.problem_features for e in self.entries]
            ).astype(np.float32)

            if self.use_faiss:
                import faiss

                d = self.features.shape[1]

                if self.metric == SimilarityMetric.COSINE:
                    # Normalize for cosine similarity (operates in-place)
                    faiss.normalize_L2(self.features)
                    self.faiss_index = faiss.IndexFlatIP(d)  # Inner product
                else:
                    self.faiss_index = faiss.IndexFlatL2(d)  # L2 distance

                self.faiss_index.add(self.features)

            self.index_built = True
        except Exception as e:
            logger.error(f"Index building failed: {e}")
            self.index_built = False

    def search(
        self, query_features: np.ndarray, k: int = 10
    ) -> List[Tuple[MemoryEntry, float]]:
        """Search for k most similar entries"""

        if not self.index_built:
            self.build_index()

        if not self.entries:
            return []

        try:
            query = query_features.reshape(1, -1)

            if self.use_faiss and self.faiss_index:
                import faiss

                # FIX: Create a float32 copy for normalization to avoid modifying input
                query_copy = query.copy().astype(np.float32)

                if self.metric == SimilarityMetric.COSINE:
                    faiss.normalize_L2(query_copy)

                distances, indices = self.faiss_index.search(
                    query_copy, min(k, len(self.entries))
                )

                results = []
                for i, dist in zip(indices[0], distances[0]):
                    if i >= 0:  # FAISS returns -1 for not found
                        # CRITICAL FIX: Handle distance/similarity conversion properly
                        if self.metric == SimilarityMetric.COSINE:
                            similarity = float(
                                dist
                            )  # Already similarity for inner product
                        else:
                            similarity = 1.0 / (
                                1.0 + float(dist)
                            )  # Convert distance to similarity

                        # CRITICAL FIX: Clamp similarity to valid range
                        similarity = np.clip(similarity, 0.0, 1.0)
                        results.append((self.entries[i], similarity))

                return results
            else:
                # Numpy fallback
                # FIX: Implement a more robust numpy-based cosine similarity
                if self.metric == SimilarityMetric.COSINE:
                    # Normalize query and feature vectors for cosine similarity
                    query_norm = query / (np.linalg.norm(query) + 1e-10)
                    features_norm = self.features / (
                        np.linalg.norm(self.features, axis=1, keepdims=True) + 1e-10
                    )

                    # Compute dot product for similarities
                    similarities = np.dot(features_norm, query_norm.T).flatten()
                elif self.metric == SimilarityMetric.EUCLIDEAN:
                    dists = np.linalg.norm(self.features - query, axis=1)
                    similarities = 1.0 / (1.0 + dists)
                else:
                    similarities = np.ones(len(self.features))

                # Get top k
                # Ensure k is not larger than the number of available similarities
                actual_k = min(k, len(similarities))
                top_indices = np.argsort(similarities)[-actual_k:][::-1]
                return [(self.entries[i], float(similarities[i])) for i in top_indices]
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def __del__(self):
        """Cleanup FAISS index on deletion"""
        try:
            if self.use_faiss and self.faiss_index is not None:
                del self.faiss_index
                self.faiss_index = None
        except Exception as e:
            logger.debug(f"Failed to validate memory prior: {e}")


class BayesianMemoryPrior:
    """
    Computes Bayesian priors for tool selection based on memory
    """

    def __init__(
        self,
        memory_system: Optional[Any] = None,
        similarity_metric: SimilarityMetric = SimilarityMetric.COSINE,
        prior_type: PriorType = PriorType.BETA,
        recency_weight: float = 0.95,
    ):
        self.memory_system = memory_system
        self.similarity_metric = similarity_metric
        self.prior_type = prior_type
        self.recency_weight = recency_weight

        # Memory index for fast search
        self.memory_index = MemoryIndex(similarity_metric)

        # Tool performance statistics
        self.tool_stats = defaultdict(
            lambda: {
                "successes": 1,  # Start with pseudo-count
                "failures": 1,
                "total_time": 0.0,
                "total_energy": 0.0,
                "count": 0,
            }
        )

        # CRITICAL FIX: Cache with eviction policy
        self.prior_cache = {}
        self.cache_ttl = 60  # seconds
        self.max_cache_size = 1000

        # Hyperparameters for hierarchical Bayes
        self.alpha_prior = 1.0  # Dirichlet concentration
        self.beta_prior = (2, 2)  # Beta prior for success rate

        # CRITICAL FIX: Use RLock for better thread safety
        self.lock = threading.RLock()

        # Check if semantic matching is disabled via environment variable
        # When disabled, use keyword-only matching (fast, no embedding computation)
        self.semantic_matcher = None
        
        if DISABLE_SEMANTIC_MATCHING:
            logger.info(
                "[BayesianMemoryPrior] Semantic matching DISABLED (VULCAN_DISABLE_SEMANTIC_MATCHING=1). "
                "Using keyword-only tool selection."
            )
        elif SEMANTIC_MATCHER_AVAILABLE:
            # BUG #4 FIX: Use singleton pattern ONLY - never create new SemanticToolMatcher instances
            # Creating new instances causes embedding model to reload (50ms → 6400ms degradation)
            try:
                from vulcan.reasoning.singletons import get_semantic_matcher
                self.semantic_matcher = get_semantic_matcher()
                if self.semantic_matcher is not None:
                    logger.info("[BayesianMemoryPrior] Semantic matching ENABLED (using singleton from registry)")
                else:
                    # BUG #4 FIX: Do NOT fallback to direct creation - this causes model reload
                    # Instead, log warning and continue without semantic matching
                    logger.warning(
                        "[BayesianMemoryPrior] Singleton SemanticToolMatcher not available. "
                        "Using keyword-only matching to prevent embedding model reload."
                    )
            except ImportError as e:
                # BUG #4 FIX: Do NOT create new SemanticToolMatcher - causes model reload
                logger.warning(
                    f"[BayesianMemoryPrior] Singletons module not available: {e}. "
                    "Using keyword-only matching to prevent embedding model reload."
                )
            except Exception as e:
                logger.warning(f"Failed to get semantic matcher singleton: {e}")

        # Load historical data if available
        self._load_from_memory_system()

    def _load_from_memory_system(self):
        """Load historical data from memory system"""

        if not self.memory_system:
            return

        try:
            # Query memory for tool selection history
            from ...memory.base import MemoryQuery, MemoryType

            query = MemoryQuery(
                content="tool_selection", type=MemoryType.SEMANTIC, limit=1000
            )

            memories = self.memory_system.retrieve(query)

            for memory in memories:
                if "tool_selection" in memory.metadata:
                    entry = self._memory_to_entry(memory)
                    if entry:
                        self.memory_index.add(entry)
                        self._update_tool_stats(entry)

            self.memory_index.build_index()
            logger.info(f"Loaded {len(self.memory_index.entries)} entries from memory")

        except Exception as e:
            logger.warning(f"Could not load from memory system: {e}")

    def _memory_to_entry(self, memory: Any) -> Optional[MemoryEntry]:
        """Convert memory system entry to MemoryEntry"""

        try:
            metadata = memory.metadata.get("tool_selection", {})

            features = metadata.get("features", [])
            if not features:
                return None

            return MemoryEntry(
                entry_id=str(memory.id),
                problem_features=np.array(features),
                tool_used=metadata.get("tool", ""),
                success=metadata.get("success", False),
                confidence=metadata.get("confidence", 0.5),
                execution_time=metadata.get("time", 0.0),
                energy_used=metadata.get("energy", 0.0),
                timestamp=metadata.get("timestamp", time.time()),
                context=metadata.get("context", {}),
            )
        except Exception as e:
            logger.warning(f"Memory conversion failed: {e}")
            return None

    def _update_tool_stats(self, entry: MemoryEntry):
        """Update tool performance statistics"""

        try:
            stats = self.tool_stats[entry.tool_used]

            if entry.success:
                stats["successes"] += 1
            else:
                stats["failures"] += 1

            stats["total_time"] += entry.execution_time
            stats["total_energy"] += entry.energy_used
            stats["count"] += 1
        except Exception as e:
            logger.warning(f"Stats update failed: {e}")

    def compute_prior(
        self,
        features: np.ndarray,
        available_tools: List[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> PriorDistribution:
        """
        Compute prior distribution over tools - CRITICAL: With cache eviction

        Args:
            features: Problem feature vector
            available_tools: List of available tool names
            context: Additional context

        Returns:
            PriorDistribution with tool probabilities
        """

        # CRITICAL FIX: Check cache with proper key generation
        try:
            cache_key = f"{features.tobytes()}_{str(sorted(available_tools))}"
        except Exception:
            cache_key = f"{hash(str(features))}_{hash(str(sorted(available_tools)))}"

        current_time = time.time()

        if cache_key in self.prior_cache:
            cached, timestamp = self.prior_cache[cache_key]
            if current_time - timestamp < self.cache_ttl:
                return cached

        with self.lock:
            try:
                if self.prior_type == PriorType.UNIFORM:
                    prior = self._uniform_prior(available_tools)
                elif self.prior_type == PriorType.BETA:
                    prior = self._beta_prior(features, available_tools)
                elif self.prior_type == PriorType.DIRICHLET:
                    prior = self._dirichlet_prior(features, available_tools)
                elif self.prior_type == PriorType.HIERARCHICAL:
                    prior = self._hierarchical_prior(features, available_tools)
                else:
                    prior = self._empirical_prior(features, available_tools)
            except Exception as e:
                logger.error(f"Prior computation failed: {e}")
                prior = self._uniform_prior(available_tools)

        # ================================================================
        # BUG #0 FIX EXTENSION: Check if semantic boost should be skipped
        # When the LLM classifier has made an authoritative decision about
        # the query category (UNKNOWN, CREATIVE, CONVERSATIONAL, etc.),
        # we should NOT override it with semantic embedding matching.
        # The LLM has better language understanding than embedding similarity.
        # ================================================================
        skip_semantic_boost = False
        if isinstance(context, dict):
            skip_semantic_boost = context.get('skip_semantic_boost', False)
        
        if skip_semantic_boost:
            logger.info(
                "[SemanticBoost] SKIPPED: LLM classifier is authoritative for this query category"
            )
            prior.metadata['semantic_boost_applied'] = False
            prior.metadata['semantic_boost_skipped_reason'] = 'classifier_is_authoritative'
        elif self.semantic_matcher is not None and context:
            query_text = None
            if isinstance(context, dict):
                query_text = context.get('query') or context.get('problem') or context.get('text')
                if query_text is None and 'problem' in context:
                    problem = context['problem']
                    if isinstance(problem, str):
                        query_text = problem
                    elif isinstance(problem, dict):
                        query_text = problem.get('text') or problem.get('query') or str(problem)
            elif isinstance(context, str):
                query_text = context
            
            # DEBUG: Log extracted query text details
            logger.debug(f"[SemanticBoost] Extracted query_text: type={type(query_text).__name__}, length={len(query_text) if query_text else 0}, min_required={MIN_QUERY_LENGTH_FOR_SEMANTIC_BOOST}")
            
            if query_text and isinstance(query_text, str) and len(query_text) > MIN_QUERY_LENGTH_FOR_SEMANTIC_BOOST:
                try:
                    original_top = max(prior.tool_probs.items(), key=lambda x: x[1]) if prior.tool_probs else None
                    prior.tool_probs = self.semantic_matcher.boost_prior(
                        prior.tool_probs,
                        query_text,
                        available_tools
                    )
                    if prior.tool_probs:
                        prior.most_likely_tool = max(
                            prior.tool_probs.items(),
                            key=lambda x: x[1]
                        )[0]
                    prior.metadata['semantic_boost_applied'] = True
                    # Also set flag in context so SafetyGovernor can see it
                    if isinstance(context, dict):
                        context['semantic_boost_applied'] = True
                    # Log semantic boost for debugging tool selection issues
                    new_top = max(prior.tool_probs.items(), key=lambda x: x[1]) if prior.tool_probs else None
                    if new_top:
                        changed_msg = ''
                        if original_top and original_top[0] != new_top[0]:
                            changed_msg = f', changed from {original_top[0]}'
                        logger.info(
                            f"[SemanticBoost] Applied to query ({len(query_text)} chars), "
                            f"top tool: {new_top[0]} ({new_top[1]:.3f}){changed_msg}"
                        )
                except Exception as e:
                    logger.warning(f"Semantic boost failed: {e}")

        # Additional boost for multimodal when multimodal content is detected
        # BUG #2 FIX: Only apply boost when actual multimodal data is present
        # (detected via stricter checks in tool_selector.py), NOT just keyword matches
        if context and context.get('is_multimodal') and 'multimodal' in available_tools:
            # Double-check: Only boost if we have evidence of actual multimodal data
            # This prevents false positives from text queries about images
            has_actual_multimodal_data = (
                context.get('has_binary_data') or
                context.get('has_image_url') or
                context.get('is_multimodal')  # Set by stricter check in tool_selector
            )
            if has_actual_multimodal_data:
                try:
                    original_multimodal_prob = prior.tool_probs.get('multimodal', 0)
                    prior.tool_probs['multimodal'] = original_multimodal_prob + MULTIMODAL_CONTENT_BOOST
                    
                    # Renormalize
                    total = sum(prior.tool_probs.values())
                    if total > 0:
                        prior.tool_probs = {k: v / total for k, v in prior.tool_probs.items()}
                    
                    # Update most likely tool
                    prior.most_likely_tool = max(prior.tool_probs.items(), key=lambda x: x[1])[0]
                    prior.metadata['multimodal_content_boost_applied'] = True
                    
                    logger.info(f"[MultimodalBoost] multimodal: {original_multimodal_prob:.3f} -> {prior.tool_probs['multimodal']:.3f}")
                except Exception as e:
                    logger.warning(f"Multimodal boost failed: {e}")

        # CRITICAL FIX: Evict old cache entries before adding new one
        if len(self.prior_cache) >= self.max_cache_size:
            # Remove entries older than TTL
            expired_keys = [
                k
                for k, (_, ts) in self.prior_cache.items()
                if current_time - ts > self.cache_ttl
            ]
            for k in expired_keys:
                del self.prior_cache[k]

            # If still too large, remove oldest half
            if len(self.prior_cache) >= self.max_cache_size:
                sorted_items = sorted(
                    self.prior_cache.items(),
                    key=lambda x: x[1][1],  # Sort by timestamp
                )
                # Keep only newest half
                self.prior_cache = dict(sorted_items[-self.max_cache_size // 2 :])

        # Cache result
        self.prior_cache[cache_key] = (prior, current_time)

        return prior

    def _uniform_prior(self, tools: List[str]) -> PriorDistribution:
        """Uniform prior over tools"""

        if not tools:
            return PriorDistribution(
                tool_probs={},
                confidence=0.0,
                support_count=0,
                entropy=0.0,
                most_likely_tool="",
            )

        n = len(tools)
        uniform_prob = 1.0 / n

        return PriorDistribution(
            tool_probs={tool: uniform_prob for tool in tools},
            confidence=0.0,  # No confidence in uniform
            support_count=0,
            entropy=float(np.log(n)),  # Maximum entropy
            most_likely_tool=tools[0],
        )

    def _beta_prior(self, features: np.ndarray, tools: List[str]) -> PriorDistribution:
        """Beta prior based on success rates"""

        if not tools:
            return self._uniform_prior(tools)

        try:
            # Find similar past problems
            similar_entries = self.memory_index.search(features, k=50)

            if not similar_entries:
                return self._uniform_prior(tools)

            # Aggregate successes/failures per tool with similarity weighting
            tool_data = defaultdict(
                lambda: {"successes": 1, "failures": 1, "weight": 0}
            )

            current_time = time.time()

            for entry, similarity in similar_entries:
                if entry.tool_used in tools:
                    # Apply recency weight
                    # CRITICAL FIX: Handle very old entries
                    days_old = (current_time - entry.timestamp) / 86400
                    days_old = min(days_old, 365)  # Cap at 1 year

                    recency = self.recency_weight**days_old
                    weight = similarity * recency

                    if entry.success:
                        tool_data[entry.tool_used]["successes"] += weight
                    else:
                        tool_data[entry.tool_used]["failures"] += weight

                    tool_data[entry.tool_used]["weight"] += weight

            # Compute Beta posterior mean for each tool
            tool_probs = {}
            for tool in tools:
                data = tool_data[tool]
                alpha = self.beta_prior[0] + data["successes"]
                beta_param = self.beta_prior[1] + data["failures"]

                # Posterior mean
                # CRITICAL FIX: Handle edge case
                total = alpha + beta_param
                if total > 0:
                    tool_probs[tool] = alpha / total
                else:
                    tool_probs[tool] = 1.0 / len(tools)

            # Normalize
            total = sum(tool_probs.values())
            if total > 0:
                tool_probs = {k: v / total for k, v in tool_probs.items()}
            else:
                tool_probs = {tool: 1.0 / len(tools) for tool in tools}

            # Calculate entropy
            probs = np.array(list(tool_probs.values()))
            probs = probs[probs > 1e-10]  # Filter very small probabilities

            if len(probs) > 0:
                entropy = -np.sum(probs * np.log(probs + 1e-10))
            else:
                entropy = 0.0

            # CRITICAL FIX: Handle empty tool_probs
            if tool_probs:
                most_likely = max(tool_probs, key=tool_probs.get)
                confidence = (
                    1.0 - entropy / np.log(len(tools)) if len(tools) > 1 else 1.0
                )
            else:
                most_likely = tools[0] if tools else ""
                confidence = 0.0

            return PriorDistribution(
                tool_probs=tool_probs,
                confidence=float(np.clip(confidence, 0.0, 1.0)),
                support_count=len(similar_entries),
                entropy=float(entropy),
                most_likely_tool=most_likely,
                metadata={"similar_problems": len(similar_entries)},
            )
        except Exception as e:
            logger.error(f"Beta prior computation failed: {e}")
            return self._uniform_prior(tools)

    def _dirichlet_prior(
        self, features: np.ndarray, tools: List[str]
    ) -> PriorDistribution:
        """Dirichlet prior for multinomial tool selection"""

        if not tools:
            return self._uniform_prior(tools)

        try:
            # Find similar past problems
            similar_entries = self.memory_index.search(features, k=50)

            if not similar_entries:
                return self._uniform_prior(tools)

            # Count tool usage weighted by similarity and recency
            tool_counts = defaultdict(float)
            total_weight = 0
            current_time = time.time()

            for entry, similarity in similar_entries:
                if entry.tool_used in tools:
                    # CRITICAL FIX: Cap days_old
                    days_old = min((current_time - entry.timestamp) / 86400, 365)
                    recency = self.recency_weight**days_old

                    # Weight successful tools more
                    weight = similarity * recency * (1.0 if entry.success else 0.5)
                    tool_counts[entry.tool_used] += weight
                    total_weight += weight

            # Set Dirichlet parameters
            alpha = np.array(
                [self.alpha_prior + tool_counts.get(tool, 0) for tool in tools]
            )

            # Sample from Dirichlet or use mean
            # CRITICAL FIX: Handle zero alpha
            alpha_sum = np.sum(alpha)
            if alpha_sum > 0:
                # Use mean of Dirichlet
                probs = alpha / alpha_sum
            else:
                probs = np.ones(len(tools)) / len(tools)

            tool_probs = {tool: float(prob) for tool, prob in zip(tools, probs)}

            # Calculate entropy
            probs_positive = probs[probs > 1e-10]
            if len(probs_positive) > 0:
                entropy = -np.sum(probs_positive * np.log(probs_positive + 1e-10))
            else:
                entropy = 0.0

            # CRITICAL FIX: Safe confidence calculation
            confidence = (
                total_weight / (total_weight + len(tools))
                if (total_weight + len(tools)) > 0
                else 0.0
            )

            return PriorDistribution(
                tool_probs=tool_probs,
                confidence=float(np.clip(confidence, 0.0, 1.0)),
                support_count=len(similar_entries),
                entropy=float(entropy),
                most_likely_tool=tools[np.argmax(probs)],
                metadata={"total_weight": float(total_weight)},
            )
        except Exception as e:
            logger.error(f"Dirichlet prior computation failed: {e}")
            return self._uniform_prior(tools)

    def _empirical_prior(
        self, features: np.ndarray, tools: List[str]
    ) -> PriorDistribution:
        """Empirical prior from direct frequency counts"""

        if not tools:
            return self._uniform_prior(tools)

        try:
            # Find similar past problems
            similar_entries = self.memory_index.search(features, k=100)

            if not similar_entries:
                return self._uniform_prior(tools)

            # Count successful tool uses
            tool_success_counts = defaultdict(float)
            tool_total_counts = defaultdict(float)

            for entry, similarity in similar_entries:
                if entry.tool_used in tools:
                    weight = similarity
                    tool_total_counts[entry.tool_used] += weight
                    if entry.success:
                        tool_success_counts[entry.tool_used] += weight

            # Compute empirical probabilities
            tool_probs = {}
            total_counts_sum = sum(tool_total_counts.values())

            for tool in tools:
                if tool in tool_total_counts and tool_total_counts[tool] > 0:
                    success_rate = tool_success_counts[tool] / tool_total_counts[tool]
                    # Weight by number of observations
                    # CRITICAL FIX: Handle division by zero
                    if total_counts_sum > 0:
                        weight = tool_total_counts[tool] / total_counts_sum
                        tool_probs[tool] = success_rate * weight
                    else:
                        tool_probs[tool] = success_rate / len(tools)
                else:
                    tool_probs[tool] = 1.0 / len(tools)  # Uniform for unseen

            # Normalize
            total = sum(tool_probs.values())
            if total > 0:
                tool_probs = {k: v / total for k, v in tool_probs.items()}
            else:
                tool_probs = {tool: 1.0 / len(tools) for tool in tools}

            # Calculate entropy
            probs = np.array(list(tool_probs.values()))
            probs = probs[probs > 1e-10]

            if len(probs) > 0:
                entropy = -np.sum(probs * np.log(probs + 1e-10))
            else:
                entropy = 0.0

            # CRITICAL FIX: Safe confidence calculation
            confidence = (
                min(1.0, total_counts_sum / 10) if total_counts_sum > 0 else 0.0
            )

            if tool_probs:
                most_likely = max(tool_probs, key=tool_probs.get)
            else:
                most_likely = tools[0] if tools else ""

            return PriorDistribution(
                tool_probs=tool_probs,
                confidence=float(np.clip(confidence, 0.0, 1.0)),
                support_count=len(similar_entries),
                entropy=float(entropy),
                most_likely_tool=most_likely,
            )
        except Exception as e:
            logger.error(f"Empirical prior computation failed: {e}")
            return self._uniform_prior(tools)

    def _hierarchical_prior(
        self, features: np.ndarray, tools: List[str]
    ) -> PriorDistribution:
        """Hierarchical Bayesian prior combining global and local evidence"""

        if not tools:
            return self._uniform_prior(tools)

        try:
            # Level 1: Global prior from all data
            global_prior = self._compute_global_prior(tools)

            # Level 2: Local prior from similar problems
            local_prior = self._beta_prior(features, tools)

            # Combine with hierarchical weighting
            n_local = local_prior.support_count
            n_global = sum(self.tool_stats[t]["count"] for t in tools)

            # Weight local evidence more as we have more similar examples
            # CRITICAL FIX: Handle division by zero
            denominator = n_local + 10
            local_weight = n_local / denominator if denominator > 0 else 0.0
            global_weight = 1 - local_weight

            # Combine probabilities
            combined_probs = {}
            for tool in tools:
                local_prob = local_prior.tool_probs.get(tool, 0)
                global_prob = global_prior.tool_probs.get(tool, 0)

                combined_probs[tool] = (
                    local_weight * local_prob + global_weight * global_prob
                )

            # Normalize
            total = sum(combined_probs.values())
            if total > 0:
                combined_probs = {k: v / total for k, v in combined_probs.items()}
            else:
                combined_probs = {tool: 1.0 / len(tools) for tool in tools}

            # Calculate entropy
            probs = np.array(list(combined_probs.values()))
            probs = probs[probs > 1e-10]

            if len(probs) > 0:
                entropy = -np.sum(probs * np.log(probs + 1e-10))
            else:
                entropy = 0.0

            # CRITICAL FIX: Safe confidence calculation
            combined_confidence = (
                local_weight * local_prior.confidence
                + global_weight * global_prior.confidence
            )

            if combined_probs:
                most_likely = max(combined_probs, key=combined_probs.get)
            else:
                most_likely = tools[0] if tools else ""

            return PriorDistribution(
                tool_probs=combined_probs,
                confidence=float(np.clip(combined_confidence, 0.0, 1.0)),
                support_count=n_local + n_global,
                entropy=float(entropy),
                most_likely_tool=most_likely,
                metadata={
                    "local_weight": float(local_weight),
                    "global_weight": float(global_weight),
                },
            )
        except Exception as e:
            logger.error(f"Hierarchical prior computation failed: {e}")
            return self._uniform_prior(tools)

    def _compute_global_prior(self, tools: List[str]) -> PriorDistribution:
        """Compute global prior from all historical data"""

        if not tools:
            return self._uniform_prior(tools)

        try:
            tool_probs = {}
            total_count = 0

            for tool in tools:
                stats = self.tool_stats[tool]

                # CRITICAL FIX: Handle division by zero
                total_attempts = stats["successes"] + stats["failures"]
                if total_attempts > 0:
                    success_rate = stats["successes"] / total_attempts
                else:
                    success_rate = 0.5

                count = stats["count"]

                # Weight by count and success rate
                tool_probs[tool] = success_rate * np.log(count + 2)
                total_count += count

            # Normalize
            total = sum(tool_probs.values())
            if total > 0:
                tool_probs = {k: v / total for k, v in tool_probs.items()}
            else:
                tool_probs = {tool: 1.0 / len(tools) for tool in tools}

            # Calculate entropy
            probs = np.array(list(tool_probs.values()))
            probs = probs[probs > 1e-10]

            if len(probs) > 0:
                entropy = -np.sum(probs * np.log(probs + 1e-10))
            else:
                entropy = 0.0

            if tool_probs:
                most_likely = max(tool_probs, key=tool_probs.get)
            else:
                most_likely = tools[0] if tools else ""

            return PriorDistribution(
                tool_probs=tool_probs,
                confidence=float(np.clip(min(1.0, total_count / 100), 0.0, 1.0)),
                support_count=total_count,
                entropy=float(entropy),
                most_likely_tool=most_likely,
            )
        except Exception as e:
            logger.error(f"Global prior computation failed: {e}")
            return self._uniform_prior(tools)

    def update(
        self,
        features: np.ndarray,
        tool_used: str,
        success: bool,
        confidence: float,
        execution_time: float,
        energy_used: float,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Update memory with new execution result"""

        with self.lock:
            try:
                # Create memory entry
                entry = MemoryEntry(
                    entry_id=f"entry_{time.time()}_{np.random.randint(1000)}",
                    problem_features=features,
                    tool_used=tool_used,
                    success=success,
                    confidence=confidence,
                    execution_time=execution_time,
                    energy_used=energy_used,
                    timestamp=time.time(),
                    context=context or {},
                )

                # Add to index
                self.memory_index.add(entry)

                # Update statistics
                self._update_tool_stats(entry)

                # Store in memory system if available
                if self.memory_system:
                    self._store_in_memory_system(entry)

                # Invalidate cache
                self.prior_cache.clear()
            except Exception as e:
                logger.error(f"Update failed: {e}")

    def _store_in_memory_system(self, entry: MemoryEntry):
        """Store entry in memory system"""

        try:
            from ...memory.base import MemoryType

            self.memory_system.store(
                content={
                    "features": entry.problem_features.tolist(),
                    "tool": entry.tool_used,
                    "success": entry.success,
                    "confidence": entry.confidence,
                },
                type=MemoryType.SEMANTIC,
                importance=0.7 if entry.success else 0.3,
                metadata={
                    "tool_selection": {
                        "tool": entry.tool_used,
                        "success": entry.success,
                        "confidence": entry.confidence,
                        "time": entry.execution_time,
                        "energy": entry.energy_used,
                        "timestamp": entry.timestamp,
                        "features": entry.problem_features.tolist(),
                        "context": entry.context,
                    }
                },
            )
        except Exception as e:
            logger.warning(f"Could not store in memory system: {e}")

    def get_tool_statistics(self) -> Dict[str, Any]:
        """Get performance statistics for all tools"""

        stats = {}

        try:
            for tool, data in self.tool_stats.items():
                if data["count"] > 0:
                    # CRITICAL FIX: Handle division by zero
                    total_attempts = data["successes"] + data["failures"]

                    stats[tool] = {
                        "success_rate": (
                            data["successes"] / total_attempts
                            if total_attempts > 0
                            else 0.0
                        ),
                        "avg_time": data["total_time"] / data["count"],
                        "avg_energy": data["total_energy"] / data["count"],
                        "usage_count": data["count"],
                    }
        except Exception as e:
            logger.error(f"Statistics computation failed: {e}")

        return stats

    def save_state(self, path: str):
        """Save prior state to disk"""

        try:
            save_path = Path(path)
            save_path.mkdir(parents=True, exist_ok=True)

            state = {
                "tool_stats": dict(self.tool_stats),
                "entries": self.memory_index.entries,
                "alpha_prior": self.alpha_prior,
                "beta_prior": self.beta_prior,
                "recency_weight": self.recency_weight,
                "max_cache_size": self.max_cache_size,
            }

            with open(save_path / "memory_prior.pkl", "wb") as f:
                pickle.dump(state, f)

            logger.info(f"Memory prior state saved to {save_path}")
        except Exception as e:
            logger.error(f"State saving failed: {e}")

    def load_state(self, path: str):
        """Load prior state from disk"""

        try:
            load_path = Path(path) / "memory_prior.pkl"

            if not load_path.exists():
                logger.warning(f"No saved state found at {load_path}")
                return

            with open(load_path, "rb") as f:
                state = pickle.load(f)  # nosec B301 - Internal data structure

            self.tool_stats = defaultdict(
                lambda: {
                    "successes": 1,
                    "failures": 1,
                    "total_time": 0.0,
                    "total_energy": 0.0,
                    "count": 0,
                },
                state["tool_stats"],
            )

            # Rebuild index
            self.memory_index = MemoryIndex(self.similarity_metric)
            for entry in state["entries"]:
                self.memory_index.add(entry)
            self.memory_index.build_index()

            self.alpha_prior = state.get("alpha_prior", 1.0)
            self.beta_prior = state.get("beta_prior", (2, 2))
            self.recency_weight = state.get("recency_weight", 0.95)
            self.max_cache_size = state.get("max_cache_size", 1000)

            logger.info(f"Memory prior state loaded from {load_path}")
        except Exception as e:
            logger.error(f"State loading failed: {e}")

    def clear_cache(self):
        """Clear the prior cache"""
        with self.lock:
            self.prior_cache.clear()
            logger.info("Prior cache cleared")

    def __del__(self):
        """Cleanup on deletion"""
        try:
            # Clear cache
            self.prior_cache.clear()

            # Cleanup index
            if hasattr(self, "memory_index"):
                del self.memory_index
        except Exception as e:
            logger.debug(f"Failed to update memory statistics: {e}")


class AdaptivePriorSelector:
    """
    Selects best prior type based on data availability
    """

    def __init__(self, memory_system: Optional[Any] = None):
        self.priors = {}

        # CRITICAL FIX: Initialize priors lazily to avoid memory overhead
        self.memory_system = memory_system
        self._initialized_priors = set()

        self.selection_history = deque(maxlen=100)

        # CRITICAL FIX: Add lock for thread safety
        self.lock = threading.RLock()

    def _get_prior(self, prior_type: PriorType) -> BayesianMemoryPrior:
        """Get or create prior of specified type"""

        if prior_type not in self.priors:
            with self.lock:
                # Double-check after acquiring lock
                if prior_type not in self.priors:
                    self.priors[prior_type] = BayesianMemoryPrior(
                        self.memory_system, prior_type=prior_type
                    )
                    self._initialized_priors.add(prior_type)

        return self.priors[prior_type]

    def select_prior_type(
        self, features: np.ndarray, available_tools: List[str]
    ) -> PriorType:
        """Select best prior type based on available data"""

        try:
            # Check data availability using Beta prior as reference
            beta_prior = self._get_prior(PriorType.BETA)
            n_entries = len(beta_prior.memory_index.entries)

            if n_entries < 10:
                return PriorType.UNIFORM
            elif n_entries < 50:
                return PriorType.EMPIRICAL
            elif n_entries < 200:
                return PriorType.BETA
            elif n_entries < 500:
                return PriorType.DIRICHLET
            else:
                return PriorType.HIERARCHICAL
        except Exception as e:
            logger.error(f"Prior type selection failed: {e}")
            return PriorType.UNIFORM

    def compute_adaptive_prior(
        self, features: np.ndarray, available_tools: List[str]
    ) -> PriorDistribution:
        """Compute prior with adaptive selection"""

        try:
            prior_type = self.select_prior_type(features, available_tools)
            prior_instance = self._get_prior(prior_type)
            prior = prior_instance.compute_prior(features, available_tools)

            # Track selection
            self.selection_history.append(
                {
                    "prior_type": prior_type,
                    "confidence": prior.confidence,
                    "entropy": prior.entropy,
                    "timestamp": time.time(),
                }
            )

            prior.metadata["prior_type"] = prior_type.value

            return prior
        except Exception as e:
            logger.error(f"Adaptive prior computation failed: {e}")
            # Fallback to uniform prior
            uniform_prior = self._get_prior(PriorType.UNIFORM)
            return uniform_prior.compute_prior(features, available_tools)

    def update_all(
        self,
        features: np.ndarray,
        tool_used: str,
        success: bool,
        confidence: float,
        execution_time: float,
        energy_used: float,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Update all initialized prior types"""

        try:
            with self.lock:
                for prior_type in self._initialized_priors:
                    if prior_type in self.priors:
                        self.priors[prior_type].update(
                            features,
                            tool_used,
                            success,
                            confidence,
                            execution_time,
                            energy_used,
                            context,
                        )
        except Exception as e:
            logger.error(f"Update all failed: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics across all priors"""

        stats = {
            "initialized_priors": [pt.value for pt in self._initialized_priors],
            "selection_history_size": len(self.selection_history),
        }

        try:
            # Get recent selections
            if self.selection_history:
                recent = list(self.selection_history)[-10:]
                stats["recent_selections"] = [
                    {
                        "type": s["prior_type"].value,
                        "confidence": s["confidence"],
                        "entropy": s["entropy"],
                    }
                    for s in recent
                ]
        except Exception as e:
            logger.warning(f"Statistics computation failed: {e}")

        return stats

    def __del__(self):
        """Cleanup on deletion"""
        try:
            for prior in self.priors.values():
                del prior
            self.priors.clear()
        except Exception as e:
            logger.debug(f"Failed to clear memory cache: {e}")
