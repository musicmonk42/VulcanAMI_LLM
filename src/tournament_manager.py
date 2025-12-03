"""
Tournament Manager for Graphix
==============================

Implements adaptive tournaments with dynamic diversity penalties to drive innovation.
Includes distributed scaling support, comprehensive validation, and robust error handling.

FIXES APPLIED:
- Input validation for proposals/fitness length match
- Embedding function return type validation
- Bounds checking on similarity_threshold
- Sharder output format validation
- Configurable winner percentage (no hardcoding)
- Exception handling around state updates
- Comprehensive error handling throughout
"""

import numpy as np
from typing import List, Dict, Any, Callable, Optional, Union
from prometheus_client import Histogram, Counter, Gauge
import logging
import uuid
import time
import json

# --- Distributed Scaling Integration ---
try:
    from distributed_sharder import sharder as DistributedSharder
except ImportError:
    DistributedSharder = None

# --- Observability metrics ---
# Use try-except to handle duplicate registration gracefully
# This prevents errors when module is imported multiple times (e.g., via test runner aliases)
# Note: Prometheus automatically adds suffixes (_total for Counter, _seconds for Histogram, etc.)
# so we need to match on the base name without suffix when retrieving metrics
tournament_latency = None
try:
    tournament_latency = Histogram('tournament_latency_seconds', 'Time for tournament selection (seconds)')
except ValueError:
    # Metric already registered, retrieve it from the registry
    from prometheus_client import REGISTRY
    for collector in REGISTRY._collector_to_names.keys():
        if hasattr(collector, '_name') and collector._name == 'tournament_latency_seconds':
            tournament_latency = collector
            break
    if tournament_latency is None:
        raise RuntimeError("Failed to retrieve existing tournament_latency metric from registry")

tournament_invocations = None
try:
    tournament_invocations = Counter('tournament_invocations_total', 'Total tournaments run')
except ValueError:
    from prometheus_client import REGISTRY
    for collector in REGISTRY._collector_to_names.keys():
        # Counter strips _total suffix, so match on base name
        if hasattr(collector, '_name') and collector._name == 'tournament_invocations':
            tournament_invocations = collector
            break
    if tournament_invocations is None:
        raise RuntimeError("Failed to retrieve existing tournament_invocations metric from registry")

diversity_penalty_events = None
try:
    diversity_penalty_events = Counter('tournament_diversity_penalty_total', 'Total diversity penalties applied')
except ValueError:
    from prometheus_client import REGISTRY
    for collector in REGISTRY._collector_to_names.keys():
        # Counter strips _total suffix, so match on base name
        if hasattr(collector, '_name') and collector._name == 'tournament_diversity_penalty':
            diversity_penalty_events = collector
            break
    if diversity_penalty_events is None:
        raise RuntimeError("Failed to retrieve existing diversity_penalty_events metric from registry")

innovation_score_gauge = None
try:
    innovation_score_gauge = Gauge('tournament_innovation_score', 'Mean diversity among selected winners')
except ValueError:
    from prometheus_client import REGISTRY
    for collector in REGISTRY._collector_to_names.keys():
        if hasattr(collector, '_name') and collector._name == 'tournament_innovation_score':
            innovation_score_gauge = collector
            break
    if innovation_score_gauge is None:
        raise RuntimeError("Failed to retrieve existing innovation_score_gauge metric from registry")

coherence_score_gauge = None
try:
    coherence_score_gauge = Gauge('tournament_coherence_score', 'Mean pairwise similarity among winners')
except ValueError:
    from prometheus_client import REGISTRY
    for collector in REGISTRY._collector_to_names.keys():
        if hasattr(collector, '_name') and collector._name == 'tournament_coherence_score':
            coherence_score_gauge = collector
            break
    if coherence_score_gauge is None:
        raise RuntimeError("Failed to retrieve existing coherence_score_gauge metric from registry")

adaptive_penalty_gauge = None
try:
    adaptive_penalty_gauge = Gauge('tournament_adaptive_penalty_value', 'The current value of the adaptive diversity penalty')
except ValueError:
    from prometheus_client import REGISTRY
    for collector in REGISTRY._collector_to_names.keys():
        if hasattr(collector, '_name') and collector._name == 'tournament_adaptive_penalty_value':
            adaptive_penalty_gauge = collector
            break
    if adaptive_penalty_gauge is None:
        raise RuntimeError("Failed to retrieve existing adaptive_penalty_gauge metric from registry")

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TournamentManager")


def trace_id() -> str:
    """Generate a unique trace ID for logging."""
    return uuid.uuid4().hex[:8]


class TournamentError(Exception):
    """Base exception for tournament errors."""
    pass


class ValidationError(TournamentError):
    """Raised when input validation fails."""
    pass


class TournamentManager:
    """
    Implements adaptive tournaments with dynamic diversity penalties to drive innovation.
    This manager can scale its embedding calculations using a distributed sharder and
    adjusts its diversity penalty based on the outcomes of previous tournaments.
    
    FIXES:
    - Comprehensive input validation
    - Configurable winner percentage
    - Safe state updates with exception handling
    - Sharder output validation
    - Bounds checking on all parameters
    """

    def __init__(
        self,
        similarity_threshold: float = 0.8,
        diversity_penalty: float = 0.2,
        min_winners: int = 1,
        max_winners: Optional[int] = None,
        winner_percentage: float = 0.1,  # FIXED: Configurable instead of hardcoded
        adaptive: bool = True,
        target_innovation: float = 0.6,
        log_penalties: bool = True,
        sharder_threshold: int = 50  # FIXED: Configurable sharder threshold
    ):
        """
        Args:
            similarity_threshold: Cosine similarity above which diversity penalty is applied (0-1).
            diversity_penalty: The initial fraction (e.g., 0.2 = 20%) to penalize fitness (0-1).
            min_winners: Minimum winners to select.
            max_winners: Maximum winners to select. None = no cap.
            winner_percentage: Percentage of proposals to select as winners (0-1).
            adaptive: If True, dynamically adjust the diversity penalty based on outcomes.
            target_innovation: The desired diversity score (0 to 1) for the adaptive mechanism.
            log_penalties: Log events when diversity penalty occurs.
            sharder_threshold: Minimum number of proposals to use distributed sharder.
        """
        # FIXED: Validate all parameters
        self._validate_init_params(
            similarity_threshold, diversity_penalty, min_winners, max_winners,
            winner_percentage, target_innovation, sharder_threshold
        )
        
        self.similarity_threshold = similarity_threshold
        self.min_winners = min_winners
        self.max_winners = max_winners
        self.winner_percentage = winner_percentage
        self.adaptive = adaptive
        self.log_penalties = log_penalties
        self.sharder_threshold = sharder_threshold
        
        # --- Adaptive Penalty State ---
        self.initial_diversity_penalty = diversity_penalty
        self.current_diversity_penalty = diversity_penalty
        self.target_innovation_score = target_innovation
        self.last_innovation_score = 0.5  # Start with a neutral assumption of past performance

        # --- Distributed Sharder Integration ---
        self.sharder = DistributedSharder() if DistributedSharder is not None else None
        if self.sharder:
            logger.info("TournamentManager initialized with distributed sharder support.")
        
        logger.info(
            f"TournamentManager initialized: similarity_threshold={similarity_threshold}, "
            f"diversity_penalty={diversity_penalty}, winner_percentage={winner_percentage}"
        )

    def _validate_init_params(
        self,
        similarity_threshold: float,
        diversity_penalty: float,
        min_winners: int,
        max_winners: Optional[int],
        winner_percentage: float,
        target_innovation: float,
        sharder_threshold: int
    ) -> None:
        """Validate initialization parameters."""
        # FIXED: Bounds checking on similarity_threshold
        if not 0 <= similarity_threshold <= 1:
            raise ValidationError(
                f"similarity_threshold must be in [0, 1], got {similarity_threshold}"
            )
        
        if not 0 < diversity_penalty <= 1:
            raise ValidationError(
                f"diversity_penalty must be in (0, 1], got {diversity_penalty}"
            )
        
        if min_winners < 1:
            raise ValidationError(f"min_winners must be >= 1, got {min_winners}")
        
        if max_winners is not None and max_winners < min_winners:
            raise ValidationError(
                f"max_winners ({max_winners}) must be >= min_winners ({min_winners})"
            )
        
        if not 0 < winner_percentage <= 1:
            raise ValidationError(
                f"winner_percentage must be in (0, 1], got {winner_percentage}"
            )
        
        if not 0 <= target_innovation <= 1:
            raise ValidationError(
                f"target_innovation must be in [0, 1], got {target_innovation}"
            )
        
        if sharder_threshold < 1:
            raise ValidationError(
                f"sharder_threshold must be >= 1, got {sharder_threshold}"
            )

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings to unit length."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
        return embeddings / norms

    def _pairwise_cosine(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute pairwise cosine similarity matrix."""
        embs = self._normalize(embeddings)
        return np.dot(embs, embs.T)

    def _diversity_score(self, sim_matrix: np.ndarray) -> float:
        """
        Calculate diversity score from similarity matrix.
        Diversity: 1 - mean pairwise similarity (excluding diagonal).
        """
        n = sim_matrix.shape[0]
        if n <= 1:
            return 1.0
        
        # FIXED: Already has guard against division by zero
        mean_sim = (np.sum(sim_matrix) - n) / (n * (n - 1))
        return 1.0 - mean_sim

    def _coherence_score(self, sim_matrix: np.ndarray) -> float:
        """
        Calculate coherence score from similarity matrix.
        Coherence: mean pairwise similarity (excluding diagonal).
        """
        n = sim_matrix.shape[0]
        if n <= 1:
            return 1.0
        
        mean_sim = (np.sum(sim_matrix) - n) / (n * (n - 1))
        return mean_sim
    
    def _adapt_penalty(self, trace: str) -> None:
        """
        Adjusts the diversity penalty based on the last run's innovation score.
        
        FIXED: Exception safe - wrapped in try/except to prevent state corruption.
        """
        if not self.adaptive:
            return

        try:
            # Use a simple proportional controller to nudge the penalty toward the target
            error = self.target_innovation_score - self.last_innovation_score
            adjustment_factor = 1.0 + (error * 0.1)  # Small, gentle adjustment (10% of error)
            
            old_penalty = self.current_diversity_penalty
            self.current_diversity_penalty *= adjustment_factor
            # Clamp the penalty to a reasonable range [0.05, 0.5] to prevent extreme values
            self.current_diversity_penalty = max(0.05, min(self.current_diversity_penalty, 0.5))
            
            logger.info(
                f"[{trace}] Adapting penalty. Last innovation: {self.last_innovation_score:.3f}, "
                f"Target: {self.target_innovation_score:.3f}. "
                f"Penalty adjusted from {old_penalty:.3f} to {self.current_diversity_penalty:.3f}"
            )
            adaptive_penalty_gauge.set(self.current_diversity_penalty)
        except Exception as e:
            logger.error(f"[{trace}] Error adapting penalty: {e}. Using previous penalty.")

    def _validate_inputs(
        self,
        proposals: List[Dict[str, Any]],
        fitness: List[float],
        trace: str
    ) -> None:
        """
        Validate tournament inputs.
        
        FIXED: Comprehensive input validation.
        """
        if not proposals or not fitness:
            raise ValidationError(
                f"[{trace}] Proposals and fitness cannot be empty. "
                f"Got {len(proposals)} proposals and {len(fitness)} fitness values."
            )
        
        # FIXED: Check length match
        if len(proposals) != len(fitness):
            raise ValidationError(
                f"[{trace}] Proposals and fitness length mismatch: "
                f"{len(proposals)} proposals vs {len(fitness)} fitness values"
            )
        
        # Validate fitness values
        for i, f in enumerate(fitness):
            if not isinstance(f, (int, float)) or not np.isfinite(f):
                raise ValidationError(
                    f"[{trace}] Invalid fitness value at index {i}: {f}"
                )

    def _validate_embeddings(
        self,
        embeddings: np.ndarray,
        expected_count: int,
        trace: str
    ) -> None:
        """
        Validate embedding array.
        
        FIXED: Embedding validation.
        """
        if not isinstance(embeddings, np.ndarray):
            raise ValidationError(
                f"[{trace}] Embeddings must be numpy array, got {type(embeddings)}"
            )
        
        if embeddings.ndim != 2:
            raise ValidationError(
                f"[{trace}] Embeddings must be 2D array, got shape {embeddings.shape}"
            )
        
        if embeddings.shape[0] != expected_count:
            raise ValidationError(
                f"[{trace}] Expected {expected_count} embeddings, got {embeddings.shape[0]}"
            )
        
        if not np.all(np.isfinite(embeddings)):
            raise ValidationError(
                f"[{trace}] Embeddings contain non-finite values (NaN or Inf)"
            )

    def _compute_embeddings(
        self,
        proposals: List[Dict[str, Any]],
        embedding_func: Callable[[Dict[str, Any]], np.ndarray],
        trace: str
    ) -> np.ndarray:
        """
        Compute embeddings with optional distributed processing.
        
        FIXED: Validates sharder output format.
        """
        n = len(proposals)
        
        # --- Distributed Embedding Calculation ---
        # Use sharder for large batches to improve performance
        if self.sharder and n > self.sharder_threshold:
            logger.info(f"[{trace}] Using distributed sharder for embedding {n} proposals.")
            try:
                # This assumes the sharder has a `map` method for parallel execution
                embeddings_list = self.sharder.map(embedding_func, proposals)
                
                # FIXED: Validate sharder output before vstacking
                if not isinstance(embeddings_list, list):
                    raise ValidationError(
                        f"[{trace}] Sharder.map returned {type(embeddings_list)}, expected list"
                    )
                
                if len(embeddings_list) != n:
                    raise ValidationError(
                        f"[{trace}] Sharder returned {len(embeddings_list)} embeddings, expected {n}"
                    )
                
                # Validate each embedding
                for i, emb in enumerate(embeddings_list):
                    if not isinstance(emb, np.ndarray):
                        raise ValidationError(
                            f"[{trace}] Embedding {i} is {type(emb)}, expected numpy array"
                        )
                
                embeddings = np.vstack(embeddings_list)
                
            except ValidationError:
                raise
            except Exception as e:
                logger.error(f"[{trace}] Distributed sharding failed: {e}. Falling back to local computation.")
                embeddings = np.vstack([embedding_func(p) for p in proposals])
        else:
            # Local computation
            embeddings = np.vstack([embedding_func(p) for p in proposals])
        
        # FIXED: Validate final embeddings
        self._validate_embeddings(embeddings, n, trace)
        
        return embeddings

    @tournament_latency.time()
    def run_adaptive_tournament(
        self,
        proposals: List[Dict[str, Any]],
        fitness: List[float],
        embedding_func: Callable[[Dict[str, Any]], np.ndarray],
        meta: Optional[Dict[str, Any]] = None
    ) -> List[int]:
        """
        Selects top proposals by penalized fitness, using adaptive penalties and distributed computation.
        
        FIXED: Comprehensive validation and error handling.
        """
        tournament_invocations.inc()
        trace = trace_id()
        start = time.time()
        
        try:
            # FIXED: Comprehensive input validation
            self._validate_inputs(proposals, fitness, trace)
            
            # Adapt penalty based on previous run
            self._adapt_penalty(trace)
            
            n = len(proposals)
            
            # Compute embeddings with validation
            embeddings = self._compute_embeddings(proposals, embedding_func, trace)
            
            # Calculate cosine similarity
            cosine_sim = self._pairwise_cosine(embeddings)
            
            # Apply diversity penalties
            penalized_fitness = np.array(fitness, dtype=np.float32).copy()
            penalties_applied = np.zeros(n, dtype=bool)

            for i in range(n):
                for j in range(i + 1, n):  # Only need to check each pair once
                    if cosine_sim[i, j] > self.similarity_threshold:
                        # Apply penalty to both similar items to break ties fairly
                        if not penalties_applied[i]:
                            penalized_fitness[i] *= (1.0 - self.current_diversity_penalty)
                            penalties_applied[i] = True
                        if not penalties_applied[j]:
                            penalized_fitness[j] *= (1.0 - self.current_diversity_penalty)
                            penalties_applied[j] = True

            num_penalties = np.sum(penalties_applied)
            if num_penalties > 0:
                diversity_penalty_events.inc(num_penalties)
                if self.log_penalties:
                    logger.info(
                        f"[{trace}] Applied {num_penalties} diversity penalties "
                        f"with factor {self.current_diversity_penalty:.3f}."
                    )

            # FIXED: Configurable winner selection instead of hardcoded 10%
            num_winners = max(self.min_winners, int(np.ceil(self.winner_percentage * n)))
            if self.max_winners is not None:
                num_winners = min(num_winners, self.max_winners)
            
            winner_indices = np.argsort(-penalized_fitness)[:num_winners]

            # Calculate metrics for winners
            winner_embs = embeddings[winner_indices]
            winner_sim = self._pairwise_cosine(winner_embs)
            innovation_score = self._diversity_score(winner_sim)
            coherence_score = self._coherence_score(winner_sim)
            
            # FIXED: Safe state update with try/except
            try:
                self.last_innovation_score = innovation_score
            except Exception as e:
                logger.error(f"[{trace}] Failed to update innovation score: {e}")
            
            innovation_score_gauge.set(innovation_score)
            coherence_score_gauge.set(coherence_score)

            elapsed = time.time() - start
            logger.info(
                f"[{trace}] Tournament complete: {n} proposals -> {num_winners} winners. "
                f"Innovation={innovation_score:.3f}, Coherence={coherence_score:.3f}, "
                f"Penalties Applied={num_penalties}, Elapsed={elapsed:.3f}s"
            )

            if meta is not None:
                meta.update({
                    "trace_id": trace, 
                    "num_proposals": n, 
                    "num_winners": num_winners,
                    "innovation_score": innovation_score, 
                    "diversity": innovation_score,  # For test compatibility
                    "coherence_score": coherence_score,
                    "penalties_applied": int(num_penalties), 
                    "adaptive_penalty_used": self.current_diversity_penalty,
                    "elapsed_seconds": elapsed,
                })
            
            return winner_indices.tolist()
            
        except ValidationError as e:
            logger.error(f"[{trace}] Validation error: {e}")
            raise
        except Exception as e:
            logger.error(f"[{trace}] Unexpected error in tournament: {e}", exc_info=True)
            raise TournamentError(f"Tournament failed: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get tournament manager statistics."""
        return {
            "similarity_threshold": self.similarity_threshold,
            "current_diversity_penalty": self.current_diversity_penalty,
            "initial_diversity_penalty": self.initial_diversity_penalty,
            "target_innovation_score": self.target_innovation_score,
            "last_innovation_score": self.last_innovation_score,
            "min_winners": self.min_winners,
            "max_winners": self.max_winners,
            "winner_percentage": self.winner_percentage,
            "adaptive": self.adaptive,
            "sharder_available": self.sharder is not None
        }


# --- Example usage to demonstrate adaptation ---
if __name__ == "__main__":
    print("\n" + "="*70)
    print("TournamentManager Production Demo")
    print("="*70 + "\n")
    
    def embed(proposal: Dict[str, Any]) -> np.ndarray:
        """Simulate embeddings where some are clustered."""
        cluster = proposal.get("cluster", 0)
        base = np.zeros(16)
        base[cluster] = 1
        return base + np.random.normal(0, 0.1, 16)

    proposals = [{"id": i, "cluster": i % 3} for i in range(30)]  # 3 distinct clusters
    fitness = np.random.uniform(0.5, 1, size=30).tolist()
    
    # Initialize with configurable parameters
    tm = TournamentManager(
        diversity_penalty=0.1,
        target_innovation=0.7,
        winner_percentage=0.15,  # Select top 15%
        sharder_threshold=20  # Use sharder for >20 proposals
    )
    
    print("Initial stats:")
    print(json.dumps(tm.get_stats(), indent=2))
    
    print("\n--- Running 3 tournament rounds to show penalty adaptation ---")
    for i in range(3):
        print(f"\n--- Round {i+1} ---")
        meta = {}
        
        try:
            winners = tm.run_adaptive_tournament(proposals, fitness, embed, meta=meta)
            print(f"Winners ({len(winners)}): {winners}")
            print("Meta:", json.dumps(meta, indent=2))
            
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n--- Final Stats ---")
    print(json.dumps(tm.get_stats(), indent=2))
    
    print("\n--- Testing Error Handling ---")
    
    # Test length mismatch
    try:
        tm.run_adaptive_tournament(proposals[:10], fitness, embed)
    except ValidationError as e:
        print(f"Caught expected error: {e}")
    
    # Test empty inputs
    try:
        tm.run_adaptive_tournament([], [], embed)
    except ValidationError as e:
        print(f"Caught expected error: {e}")
    
    print("\n" + "="*70)
    print("Demo Complete")
    print("="*70 + "\n")