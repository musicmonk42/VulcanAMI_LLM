"""Hierarchical memory implementation with multiple levels and tool selection history"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Persistent memory imports with fallback
# FIX Issue #8: Add import guard for persistant_memory_v46
try:
    from persistant_memory_v46 import (
        GraphRAG,
        MerkleLSM,
        PackfileStore,
        UnlearningEngine,
        ZKProver,
    )
    PERSISTENT_MEMORY_AVAILABLE = True
except ImportError:
    # Fallback: Create placeholder classes if module not available
    logging.warning("persistant_memory_v46 not available, using fallback implementations")
    PERSISTENT_MEMORY_AVAILABLE = False
    GraphRAG = None
    MerkleLSM = None
    PackfileStore = None
    UnlearningEngine = None
    ZKProver = None

from .base import (
    BaseMemorySystem,
    Memory,
    MemoryConfig,
    MemoryQuery,
    MemoryType,
    RetrievalResult,
)
from .consolidation import MemoryConsolidator
from .retrieval import AttentionMechanism, MemorySearch

# Enhanced embedding support
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available, using fallback embedding")

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

# ============================================================
# TOOL SELECTION MEMORY TYPES
# ============================================================


@dataclass
class ToolSelectionRecord:
    """Record of a tool selection decision."""

    record_id: str
    timestamp: float
    problem_features: np.ndarray
    problem_description: str
    selected_tools: List[str]
    execution_strategy: str
    performance_metrics: Dict[str, float]  # latency, accuracy, energy, etc.
    context: Dict[str, Any]
    success: bool
    utility_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProblemPattern:
    """Pattern representing a type of problem."""

    pattern_id: str
    feature_signature: np.ndarray
    typical_tools: List[str]
    success_rate: float
    avg_utility: float
    occurrence_count: int
    examples: List[str]  # Example problem IDs
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# MEMORY LEVELS
# ============================================================


@dataclass
class MemoryLevel:
    """Single level in memory hierarchy."""

    name: str
    capacity: int
    decay_rate: float
    consolidation_threshold: float
    memories: Dict[str, Memory] = field(default_factory=dict)
    access_queue: deque = field(default_factory=lambda: deque(maxlen=1000))

    def add(self, memory: Memory) -> bool:
        """Add memory to this level."""
        if len(self.memories) >= self.capacity:
            return False

        self.memories[memory.id] = memory
        self.access_queue.append(memory.id)
        return True

    def remove_least_salient(self, n: int = 1) -> List[Memory]:
        """Remove n least salient memories."""
        current_time = time.time()

        # Calculate salience for all memories
        saliences = {
            mid: mem.compute_salience(current_time)
            for mid, mem in self.memories.items()
        }

        # Sort by salience
        sorted_mids = sorted(saliences.keys(), key=lambda x: saliences[x])

        # Remove least salient
        removed = []
        for mid in sorted_mids[:n]:
            if mid in self.memories:
                removed.append(self.memories.pop(mid))

        return removed

    def get_candidates_for_consolidation(self) -> List[Memory]:
        """Get memories ready for consolidation."""
        current_time = time.time()
        candidates = []

        for memory in self.memories.values():
            salience = memory.compute_salience(current_time)
            # FIX: Use >= for threshold comparison
            if salience >= self.consolidation_threshold:
                candidates.append(memory)

        return candidates


# ============================================================
# EMBEDDING MIGRATION
# ============================================================


class EmbeddingMigration:
    """Manages incremental migration of embeddings to new models."""
    
    def __init__(
        self,
        old_model_name: str,
        new_model_name: str,
        batch_size: int = 100,
    ):
        """Initialize embedding migration.
        
        Args:
            old_model_name: Name of the old embedding model
            new_model_name: Name of the new embedding model
            batch_size: Number of memories to migrate per batch
        """
        self.old_model_name = old_model_name
        self.new_model_name = new_model_name
        self.batch_size = batch_size
        
        # Track migration progress
        self.total_memories = 0
        self.migrated_count = 0
        self.failed_count = 0
        self.migration_status = "not_started"  # not_started, in_progress, completed, failed
        
        # Thread safety
        self._lock = threading.RLock()
        self._migration_thread = None
        self._stop_migration = False
        
        # Load new model if available
        self.new_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.new_model = SentenceTransformer(new_model_name)
                logger.info(f"Loaded new embedding model: {new_model_name}")
            except Exception as e:
                logger.error(f"Failed to load new embedding model: {e}")
        
        logger.info(
            f"Initialized embedding migration: {old_model_name} -> {new_model_name}"
        )
    
    def start_migration(
        self,
        memories: List[Memory],
        callback: Optional[callable] = None,
    ) -> bool:
        """Start background migration in batches.
        
        Args:
            memories: List of memories to migrate
            callback: Optional callback function called after each batch
            
        Returns:
            True if migration started successfully
        """
        with self._lock:
            if self.migration_status == "in_progress":
                logger.warning("Migration already in progress")
                return False
            
            if not self.new_model:
                logger.error("New embedding model not available")
                return False
            
            self.total_memories = len(memories)
            self.migrated_count = 0
            self.failed_count = 0
            self.migration_status = "in_progress"
            self._stop_migration = False
            
            # Start migration thread
            self._migration_thread = threading.Thread(
                target=self._migrate_background,
                args=(memories, callback),
                daemon=True,
            )
            self._migration_thread.start()
            
            logger.info(f"Started migration of {self.total_memories} memories")
            return True
    
    def _migrate_background(
        self,
        memories: List[Memory],
        callback: Optional[callable],
    ):
        """Background migration worker.
        
        Args:
            memories: List of memories to migrate
            callback: Optional callback function
        """
        try:
            # Process in batches
            for i in range(0, len(memories), self.batch_size):
                if self._stop_migration:
                    logger.info("Migration stopped by user")
                    with self._lock:
                        self.migration_status = "stopped"
                    return
                
                batch = memories[i:i + self.batch_size]
                batch_success, batch_failed = self._migrate_batch(batch)
                
                with self._lock:
                    self.migrated_count += batch_success
                    self.failed_count += batch_failed
                
                # Call callback if provided
                if callback:
                    try:
                        callback(self.get_progress())
                    except Exception as e:
                        logger.error(f"Migration callback error: {e}")
                
                # Log progress
                progress_pct = (self.migrated_count / self.total_memories) * 100
                logger.info(
                    f"Migration progress: {self.migrated_count}/{self.total_memories} "
                    f"({progress_pct:.1f}%), failed: {self.failed_count}"
                )
            
            # Migration complete
            with self._lock:
                if self.failed_count == 0:
                    self.migration_status = "completed"
                    logger.info("Migration completed successfully")
                else:
                    self.migration_status = "completed_with_errors"
                    logger.warning(
                        f"Migration completed with {self.failed_count} failures"
                    )
        
        except Exception as e:
            logger.error(f"Migration failed: {e}", exc_info=True)
            with self._lock:
                self.migration_status = "failed"
    
    def _migrate_batch(self, batch: List[Memory]) -> Tuple[int, int]:
        """Migrate a batch of memories.
        
        Args:
            batch: Batch of memories to migrate
            
        Returns:
            Tuple of (success_count, failure_count)
        """
        success_count = 0
        failure_count = 0
        
        for memory in batch:
            try:
                # Check if already migrated
                if memory.metadata.get("embedding_model") == self.new_model_name:
                    success_count += 1
                    continue
                
                # Generate new embedding
                if isinstance(memory.content, str):
                    new_embedding = self.new_model.encode(
                        memory.content,
                        show_progress_bar=False
                    )
                    
                    # Update memory
                    memory.embedding = new_embedding
                    memory.metadata["embedding_model"] = self.new_model_name
                    memory.metadata["embedding_version"] = "migrated"
                    memory.metadata["migration_timestamp"] = time.time()
                    
                    success_count += 1
                else:
                    logger.warning(f"Cannot migrate non-string content: {memory.id}")
                    failure_count += 1
            
            except Exception as e:
                logger.error(f"Failed to migrate memory {memory.id}: {e}")
                failure_count += 1
        
        return success_count, failure_count
    
    def stop_migration(self):
        """Stop the migration process."""
        with self._lock:
            if self.migration_status != "in_progress":
                return
            
            self._stop_migration = True
            logger.info("Stopping migration...")
    
    def get_progress(self) -> Dict[str, Any]:
        """Get migration progress.
        
        Returns:
            Dict with progress information
        """
        with self._lock:
            progress_pct = (
                (self.migrated_count / self.total_memories * 100)
                if self.total_memories > 0 else 0
            )
            
            return {
                "status": self.migration_status,
                "total_memories": self.total_memories,
                "migrated_count": self.migrated_count,
                "failed_count": self.failed_count,
                "progress_percentage": progress_pct,
                "old_model": self.old_model_name,
                "new_model": self.new_model_name,
            }
    
    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """Wait for migration to complete.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if migration completed, False if timeout
        """
        if self._migration_thread and self._migration_thread.is_alive():
            self._migration_thread.join(timeout=timeout)
            return not self._migration_thread.is_alive()
        return True


# ============================================================
# HIERARCHICAL MEMORY WITH TOOL SELECTION
# ============================================================


class HierarchicalMemory(BaseMemorySystem):
    """Multi-level hierarchical memory system with tool selection history."""

    def __init__(self, config: MemoryConfig, embedding_model: Optional[str] = None):
        super().__init__(config)

        # Initialize memory levels
        self.levels = self._init_levels()

        # Initialize embedding model
        self.embedding_model = self._init_embedding_model(embedding_model)
        self.embedding_dimension = self._get_embedding_dimension()

        # Components - FIX: Pass actual embedding dimension to attention mechanism
        self.search_engine = MemorySearch()
        self.attention = AttentionMechanism(
            hidden_dim=min(256, self.embedding_dimension),
            input_dim=self.embedding_dimension,  # FIX: Use actual dimension
        )
        self.consolidator = MemoryConsolidator()

        # Tool selection specific storage
        self.tool_selection_history = deque(maxlen=10000)
        self.problem_patterns = {}
        self.pattern_index = {}  # Maps features to pattern IDs

        # Caches
        self.embedding_cache = {}
        self.retrieval_cache = {}
        self.pattern_cache = {}

        # Background tasks
        self._shutdown_event = threading.Event()  # ADDED
        self.consolidation_thread = None
        self.pattern_mining_thread = None
        self.start_background_tasks()

    # --- START NEW LLM CONTEXT METHODS ---

    def _embed(self, query_tokens: str) -> np.ndarray:
        """MOCK/Helper: Embeds tokens using the internal model."""
        if not query_tokens:
            return np.zeros(self.embedding_dimension)
        return self._generate_embedding(query_tokens)

    def _extract_concepts(self, prompt: str, generated: str) -> List[str]:
        """MOCK/Helper: Extracts key concepts from prompt and response."""
        # Mock logic: look for capitalized words
        text = prompt + " " + generated
        return [
            word
            for word in text.split()
            if word and word[0].isupper() and len(word) > 3
        ][:5]

    def _extract_pattern(self, reasoning_trace: Dict[str, Any]) -> str:
        """MOCK/Helper: Extracts a procedural pattern from a reasoning trace."""
        if "chain" in reasoning_trace and reasoning_trace["chain"]:
            return "Pattern: " + " -> ".join(
                step.get("type", "step") for step in reasoning_trace["chain"]
            )
        return "Generic Pattern"

    def _merge(
        self, recent: List[Any], relevant: List[Any], patterns: List[Any]
    ) -> Dict[str, Any]:
        """MOCK/Helper: Merges retrieved contexts."""
        return {
            "recent_context": [r.content for r in recent],
            "semantic_concepts": [r.content for r in relevant],
            "procedural_patterns": patterns,
            "source": "merged_context",
        }

    def retrieve_context_for_generation(
        self, query_tokens: str, max_tokens: int = 2048
    ) -> Dict[str, Any]:
        """
        Retrieve relevant context from memory for LLM generation.
        Maps episodic to 'short_term', semantic to 'long_term', procedural to 'long_term'.
        """

        # 1. Episodic: Recent conversation (from short_term)
        # Assuming get_recent is equivalent to retrieving by timestamp/access queue
        with self._lock:
            recent_level = self.levels.get("short_term")
            if recent_level:
                # Use retrieval based on recent access, limiting by approx token count
                recent_memories = [
                    recent_level.memories[mid]
                    for mid in list(recent_level.access_queue)
                ][-20:]
            else:
                recent_memories = []

        # 2. Semantic: Relevant concepts (from long_term)
        query_embedding = self._embed(query_tokens)

        semantic_query = MemoryQuery(
            query_type="semantic_search",
            embedding=query_embedding,
            memory_type=MemoryType.SEMANTIC,
            limit=10,
        )
        relevant_result = self.retrieve(semantic_query)
        relevant_memories = relevant_result.memories

        # 3. Procedural: Relevant patterns (using stored patterns)
        patterns = self.get_problem_patterns(min_occurrences=1, min_success_rate=0.0)

        # Filter patterns based on query embedding similarity to pattern features
        relevant_patterns = []
        for pattern in patterns:
            similarity = self._compute_feature_similarity(
                query_embedding, pattern.feature_signature
            )
            if similarity > 0.6:
                relevant_patterns.append(
                    {
                        "tools": pattern.typical_tools,
                        "utility": pattern.avg_utility,
                        "similarity": similarity,
                    }
                )

        return self._merge(recent_memories, relevant_memories, relevant_patterns)

    def store_generation(
        self, prompt: str, generated: str, reasoning_trace: Dict[str, Any]
    ):
        """Store generation in memory, updating episodic, semantic, and procedural levels."""

        # Store procedural pattern first for immediate use
        pattern_content = self._extract_pattern(reasoning_trace)

        # Store as procedural memory (long_term)
        self.store(
            content=pattern_content,
            memory_type=MemoryType.PROCEDURAL,
            importance=0.8,
            metadata={"trace_type": reasoning_trace.get("type", "generation_trace")},
        )

        # Episodic memory (short_term)
        episodic_content = {
            "prompt": prompt,
            "response": generated,
            "timestamp": time.time(),
            "trace_summary": pattern_content,
        }
        self.store(
            content=episodic_content,
            memory_type=MemoryType.EPISODIC,
            importance=0.7,
            metadata={"interaction_type": "generation"},
        )

        # Semantic memory (long_term) - extract concepts
        concepts = self._extract_concepts(prompt, generated)
        for concept in concepts:
            self.store(
                content=concept,
                memory_type=MemoryType.SEMANTIC,
                importance=0.6,
                metadata={"source_prompt": prompt[:30]},
            )

    # --- END NEW LLM CONTEXT METHODS ---

    def store_tool_selection(
        self,
        problem_features: np.ndarray,
        problem_description: str,
        selected_tools: List[str],
        execution_strategy: str,
        performance_metrics: Dict[str, float],
        context: Dict[str, Any] = None,
        success: bool = True,
        utility_score: float = 0.0,
    ) -> ToolSelectionRecord:
        """Store a tool selection decision in memory."""

        record = ToolSelectionRecord(
            record_id=self._generate_memory_id(f"{problem_description}_{time.time()}"),
            timestamp=time.time(),
            problem_features=(
                problem_features.copy()
                if problem_features is not None
                else np.zeros(self.embedding_dimension)
            ),
            problem_description=problem_description,
            selected_tools=selected_tools.copy(),
            execution_strategy=execution_strategy,
            performance_metrics=performance_metrics.copy(),
            context=context or {},
            success=success,
            utility_score=utility_score,
            metadata={"stored_at": time.time(), "level": "tool_selection"},
        )

        # Add to tool selection history
        with self._lock:
            self.tool_selection_history.append(record)

            # Store as regular memory for retrieval
            memory = Memory(
                id=record.record_id,
                type=MemoryType.PROCEDURAL,
                content={
                    "type": "tool_selection",
                    "problem": problem_description,
                    "tools": selected_tools,
                    "strategy": execution_strategy,
                    "performance": performance_metrics,
                    "success": success,
                    "utility": utility_score,
                },
                embedding=problem_features,
                importance=0.5
                + (utility_score * 0.5),  # Higher utility = higher importance
                metadata={
                    "record_type": "tool_selection",
                    "success": success,
                    "tools": selected_tools,
                },
            )

            # Store in appropriate level based on success and utility
            if success and utility_score > 0.7:
                level_name = "long_term"
            elif success:
                level_name = "short_term"
            else:
                level_name = "working"

            level = self.levels[level_name]
            if len(level.memories) >= level.capacity:
                level.remove_least_salient(1)
            level.add(memory)

            # Update pattern mining
            self._update_problem_patterns(record)

        return record

    def retrieve_similar_problems(
        self,
        problem_features: Optional[np.ndarray] = None,
        problem_description: Optional[str] = None,
        limit: int = 10,
        min_similarity: float = 0.5,
        success_only: bool = False,
    ) -> List[Tuple[ToolSelectionRecord, float]]:
        """Retrieve similar problems and their tool selection decisions."""

        # Generate features if only description provided
        if problem_features is None and problem_description is not None:
            problem_features = self._generate_embedding(problem_description)

        if problem_features is None:
            return []

        similar_problems = []

        with self._lock:
            for record in self.tool_selection_history:
                # Filter by success if requested
                if success_only and not record.success:
                    continue

                # Compute similarity
                if record.problem_features is not None:
                    similarity = self._compute_feature_similarity(
                        problem_features, record.problem_features
                    )

                    if similarity >= min_similarity:
                        similar_problems.append((record, similarity))

            # Sort by similarity
            similar_problems.sort(key=lambda x: x[1], reverse=True)

            return similar_problems[:limit]

    def get_problem_patterns(
        self, min_occurrences: int = 5, min_success_rate: float = 0.6
    ) -> List[ProblemPattern]:
        """Get discovered problem patterns."""

        patterns = []

        with self._lock:
            for pattern in self.problem_patterns.values():
                if (
                    pattern.occurrence_count >= min_occurrences
                    and pattern.success_rate >= min_success_rate
                ):
                    patterns.append(pattern)

        # Sort by utility and occurrence count
        patterns.sort(key=lambda p: (p.avg_utility, p.occurrence_count), reverse=True)

        return patterns

    def find_matching_pattern(
        self, problem_features: np.ndarray, threshold: float = 0.7
    ) -> Optional[ProblemPattern]:
        """Find a matching problem pattern for given features."""

        best_pattern = None
        best_similarity = 0

        with self._lock:
            for pattern in self.problem_patterns.values():
                similarity = self._compute_feature_similarity(
                    problem_features, pattern.feature_signature
                )

                if similarity >= threshold and similarity > best_similarity:
                    best_pattern = pattern
                    best_similarity = similarity

        return best_pattern

    def get_recommended_tools(
        self,
        problem_features: np.ndarray,
        problem_description: Optional[str] = None,
        max_recommendations: int = 5,
    ) -> List[Dict[str, Any]]:
        """Get tool recommendations based on similar problems and patterns."""

        recommendations = []
        tool_scores = defaultdict(
            lambda: {"count": 0, "total_utility": 0, "successes": 0}
        )

        # Find similar problems
        similar_problems = self.retrieve_similar_problems(
            problem_features=problem_features,
            problem_description=problem_description,
            limit=20,
            success_only=True,
        )

        # Aggregate tool performance from similar problems
        for record, similarity in similar_problems:
            weight = similarity * record.utility_score

            for tool in record.selected_tools:
                tool_scores[tool]["count"] += 1
                tool_scores[tool]["total_utility"] += weight
                if record.success:
                    tool_scores[tool]["successes"] += 1

        # Find matching pattern
        pattern = self.find_matching_pattern(problem_features)
        if pattern:
            # Boost scores for pattern's typical tools
            for tool in pattern.typical_tools:
                tool_scores[tool]["count"] += 5
                tool_scores[tool]["total_utility"] += pattern.avg_utility * 2
                tool_scores[tool]["pattern_match"] = True

        # Calculate final scores and create recommendations
        for tool, scores in tool_scores.items():
            if scores["count"] > 0:
                avg_utility = scores["total_utility"] / scores["count"]
                success_rate = (
                    scores["successes"] / scores["count"] if scores["count"] > 0 else 0
                )

                recommendation = {
                    "tool": tool,
                    "confidence": min(1.0, avg_utility),
                    "success_rate": success_rate,
                    "occurrence_count": scores["count"],
                    "pattern_match": scores.get("pattern_match", False),
                    "score": avg_utility * (1 + success_rate),  # Combined score
                }
                recommendations.append(recommendation)

        # Sort by score
        recommendations.sort(key=lambda x: x["score"], reverse=True)

        return recommendations[:max_recommendations]

    def _update_problem_patterns(self, record: ToolSelectionRecord):
        """Update problem patterns with new record."""

        # Hold lock for ENTIRE operation
        with self._lock:
            # Find if this matches an existing pattern
            matched_pattern = None
            best_similarity = 0

            for pattern_id, pattern in self.problem_patterns.items():
                similarity = self._compute_feature_similarity(
                    record.problem_features, pattern.feature_signature
                )

                if similarity > 0.8 and similarity > best_similarity:
                    matched_pattern = pattern
                    best_similarity = similarity

            if matched_pattern:
                # Update existing pattern (all inside lock)
                matched_pattern.occurrence_count += 1
                matched_pattern.examples.append(record.record_id)

                # Update success rate
                success_count = sum(
                    1
                    for rid in matched_pattern.examples[-100:]
                    if self._get_record_success(rid)
                )
                matched_pattern.success_rate = success_count / min(
                    100, len(matched_pattern.examples)
                )

                # Update average utility
                total_utility = sum(
                    self._get_record_utility(rid)
                    for rid in matched_pattern.examples[-100:]
                )
                matched_pattern.avg_utility = total_utility / min(
                    100, len(matched_pattern.examples)
                )

                # Update typical tools
                tool_counts = defaultdict(int)
                for rid in matched_pattern.examples[-50:]:
                    rec = self._get_record_by_id(rid)
                    if rec and rec.success:
                        for tool in rec.selected_tools:
                            tool_counts[tool] += 1

                if tool_counts:
                    sorted_tools = sorted(
                        tool_counts.items(), key=lambda x: x[1], reverse=True
                    )
                    matched_pattern.typical_tools = [
                        tool for tool, _ in sorted_tools[:5]
                    ]

            else:
                # Create new pattern if enough similar records
                similar_records = self._find_similar_records(
                    record.problem_features, threshold=0.8
                )

                if len(similar_records) >= 3:
                    pattern_id = self._generate_memory_id(f"pattern_{time.time()}")

                    # Calculate pattern statistics
                    success_count = sum(1 for r in similar_records if r.success)
                    success_rate = success_count / len(similar_records)

                    avg_utility = sum(r.utility_score for r in similar_records) / len(
                        similar_records
                    )

                    # Find common tools
                    tool_counts = defaultdict(int)
                    for r in similar_records:
                        if r.success:
                            for tool in r.selected_tools:
                                tool_counts[tool] += 1

                    typical_tools = []
                    if tool_counts:
                        sorted_tools = sorted(
                            tool_counts.items(), key=lambda x: x[1], reverse=True
                        )
                        typical_tools = [tool for tool, _ in sorted_tools[:5]]

                    # Create new pattern
                    pattern = ProblemPattern(
                        pattern_id=pattern_id,
                        feature_signature=record.problem_features,
                        typical_tools=typical_tools,
                        success_rate=success_rate,
                        avg_utility=avg_utility,
                        occurrence_count=len(similar_records),
                        examples=[r.record_id for r in similar_records],
                        metadata={"created_at": time.time()},
                    )

                    self.problem_patterns[pattern_id] = pattern

    def _find_similar_records(
        self, features: np.ndarray, threshold: float = 0.8
    ) -> List[ToolSelectionRecord]:
        """Find records with similar features."""
        similar = []

        for record in self.tool_selection_history:
            if record.problem_features is not None:
                similarity = self._compute_feature_similarity(
                    features, record.problem_features
                )
                if similarity >= threshold:
                    similar.append(record)

        return similar

    def _get_record_by_id(self, record_id: str) -> Optional[ToolSelectionRecord]:
        """Get tool selection record by ID."""
        for record in self.tool_selection_history:
            if record.record_id == record_id:
                return record
        return None

    def _get_record_success(self, record_id: str) -> bool:
        """Get success status of a record."""
        record = self._get_record_by_id(record_id)
        return record.success if record else False

    def _get_record_utility(self, record_id: str) -> float:
        """Get utility score of a record."""
        record = self._get_record_by_id(record_id)
        return record.utility_score if record else 0.0

    def _compute_feature_similarity(
        self, features1: np.ndarray, features2: np.ndarray
    ) -> float:
        """Compute similarity between two feature vectors."""
        if features1 is None or features2 is None:
            return 0.0

        # FIX: Validate dimensions match
        if len(features1) != len(features2):
            logger.warning(
                f"Feature dimension mismatch: {len(features1)} vs {len(features2)}"
            )
            return 0.0

        # Normalize vectors
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        normalized1 = features1 / norm1
        normalized2 = features2 / norm2

        # Compute cosine similarity
        similarity = np.dot(normalized1, normalized2)

        # Ensure in [0, 1] range
        return float(max(0.0, min(1.0, (similarity + 1.0) / 2.0)))

    def mine_patterns(self):
        """Mine patterns from tool selection history (background task)."""
        with self._lock:
            # Clear old patterns
            self.problem_patterns.clear()

            # Group records by similarity
            processed = set()

            for i, record in enumerate(self.tool_selection_history):
                if record.record_id in processed:
                    continue

                # Find all similar records
                similar_records = []
                for other_record in self.tool_selection_history:
                    if other_record.record_id not in processed:
                        similarity = self._compute_feature_similarity(
                            record.problem_features, other_record.problem_features
                        )

                        if similarity > 0.8:
                            similar_records.append(other_record)
                            processed.add(other_record.record_id)

                # Create pattern if enough similar records
                if len(similar_records) >= 3:
                    self._create_pattern_from_records(similar_records)

    def _create_pattern_from_records(self, records: List[ToolSelectionRecord]):
        """Create a problem pattern from similar records."""
        if not records:
            return

        # Calculate centroid of features
        feature_sum = np.zeros_like(records[0].problem_features)
        for record in records:
            if record.problem_features is not None:
                feature_sum += record.problem_features

        feature_signature = feature_sum / len(records)

        # Calculate statistics
        success_count = sum(1 for r in records if r.success)
        success_rate = success_count / len(records)

        avg_utility = sum(r.utility_score for r in records) / len(records)

        # Find common tools
        tool_counts = defaultdict(int)
        for r in records:
            if r.success:
                for tool in r.selected_tools:
                    tool_counts[tool] += 1

        typical_tools = []
        if tool_counts:
            sorted_tools = sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)
            typical_tools = [tool for tool, _ in sorted_tools[:5]]

        # Create pattern
        pattern_id = self._generate_memory_id(f"pattern_{time.time()}")

        pattern = ProblemPattern(
            pattern_id=pattern_id,
            feature_signature=feature_signature,
            typical_tools=typical_tools,
            success_rate=success_rate,
            avg_utility=avg_utility,
            occurrence_count=len(records),
            examples=[r.record_id for r in records],
            metadata={"created_at": time.time()},
        )

        self.problem_patterns[pattern_id] = pattern

    def get_tool_selection_stats(self) -> Dict[str, Any]:
        """Get statistics about tool selection history."""
        with self._lock:
            total_selections = len(self.tool_selection_history)

            if total_selections == 0:
                return {
                    "total_selections": 0,
                    "success_rate": 0,
                    "avg_utility": 0,
                    "unique_tools": [],
                    "patterns_discovered": 0,
                }

            success_count = sum(1 for r in self.tool_selection_history if r.success)
            success_rate = success_count / total_selections

            avg_utility = (
                sum(r.utility_score for r in self.tool_selection_history)
                / total_selections
            )

            # Get unique tools
            unique_tools = set()
            tool_usage = defaultdict(int)

            for record in self.tool_selection_history:
                for tool in record.selected_tools:
                    unique_tools.add(tool)
                    tool_usage[tool] += 1

            # Sort tools by usage
            sorted_tools = sorted(tool_usage.items(), key=lambda x: x[1], reverse=True)

            return {
                "total_selections": total_selections,
                "success_rate": success_rate,
                "avg_utility": avg_utility,
                "unique_tools": list(unique_tools),
                "top_tools": sorted_tools[:10],
                "patterns_discovered": len(self.problem_patterns),
                "recent_performance": self._get_recent_performance(),
            }

    def _get_recent_performance(self, window: int = 100) -> Dict[str, float]:
        """Get performance metrics for recent tool selections."""
        recent = list(self.tool_selection_history)[-window:]

        if not recent:
            return {"success_rate": 0, "avg_utility": 0}

        success_count = sum(1 for r in recent if r.success)
        success_rate = success_count / len(recent)

        avg_utility = sum(r.utility_score for r in recent) / len(recent)

        return {
            "success_rate": success_rate,
            "avg_utility": avg_utility,
            "window_size": len(recent),
        }

    def _init_embedding_model(self, model_name: Optional[str] = None):
        """Initialize the embedding model with fallback options.
        
        PERFORMANCE FIX: Uses global model registry to ensure SentenceTransformer
        is loaded exactly ONCE per process and shared across all components.
        """
        # First, try to use global model registry (process-wide singleton)
        try:
            from vulcan.models.model_registry import get_sentence_transformer
            model = get_sentence_transformer(model_name or "all-MiniLM-L6-v2")
            if model is not None:
                logger.info(f"[HierarchicalMemory] Using model from global registry")
                return model
        except ImportError:
            logger.debug("[HierarchicalMemory] Model registry not available")
        
        # Fallback to direct loading
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            if model_name is None:
                # Use a lightweight but effective model by default
                model_name = "all-MiniLM-L6-v2"

            try:
                # Try to load the specified model
                model = SentenceTransformer(model_name)
                logger.info(f"Loaded embedding model: {model_name} (fallback)")
                return model
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")

                # Try fallback models
                fallback_models = [
                    "all-MiniLM-L6-v2",  # Fast and lightweight
                    "all-mpnet-base-v2",  # Better quality
                    "paraphrase-MiniLM-L6-v2",  # Alternative lightweight
                ]

                for fallback in fallback_models:
                    if fallback != model_name:
                        try:
                            model = SentenceTransformer(fallback)
                            logger.info(f"Loaded fallback embedding model: {fallback}")
                            return model
                        except Exception:
                            continue

                logger.warning(
                    "No sentence transformer models available, using fallback"
                )

        return None

    def _get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by the model."""
        if self.embedding_model and hasattr(
            self.embedding_model, "get_sentence_embedding_dimension"
        ):
            return self.embedding_model.get_sentence_embedding_dimension()

        # Default dimension for hash-based fallback
        return 128

    def _init_levels(self) -> Dict[str, MemoryLevel]:
        """Initialize memory hierarchy levels with safe config access."""
        # FIX: Use getattr with defaults to handle missing attributes
        max_working = getattr(self.config, "max_working_memory", 20)
        max_short_term = getattr(self.config, "max_short_term", 1000)
        max_long_term = getattr(self.config, "max_long_term", 100000)

        return {
            "sensory": MemoryLevel(
                name="sensory", capacity=50, decay_rate=0.5, consolidation_threshold=0.7
            ),
            "working": MemoryLevel(
                name="working",
                capacity=max_working,
                decay_rate=0.1,
                consolidation_threshold=0.6,
            ),
            "short_term": MemoryLevel(
                name="short_term",
                capacity=max_short_term,
                decay_rate=0.05,
                consolidation_threshold=0.5,
            ),
            "long_term": MemoryLevel(
                name="long_term",
                capacity=max_long_term,
                decay_rate=0.001,
                consolidation_threshold=0.8,
            ),
        }

    def store(
        self,
        content: Any,
        memory_type: MemoryType = MemoryType.SENSORY,
        importance: float = 0.5,
        **kwargs,
    ) -> Memory:
        """Store content in appropriate memory level."""
        with self._lock:
            # Create memory
            memory_id = self._generate_memory_id(content)
            memory = Memory(
                id=memory_id,
                type=memory_type,
                content=content,
                importance=importance,
                **kwargs,
            )

            # Generate embedding if needed
            if memory.embedding is None:
                memory.embedding = self._generate_embedding(content)

            # FIX: Ensure embedding has correct dimension
            if (
                memory.embedding is not None
                and len(memory.embedding) != self.embedding_dimension
            ):
                logger.warning(
                    f"Embedding dimension mismatch: {len(memory.embedding)} vs {self.embedding_dimension}, regenerating"
                )
                memory.embedding = self._generate_embedding(content)

            # Store in appropriate level
            level_name = self._get_level_for_type(memory_type)
            level = self.levels[level_name]

            # Make space if needed
            if len(level.memories) >= level.capacity:
                if level_name != "long_term":
                    # Consolidate to next level
                    self._promote_memories(level_name)
                else:
                    # Remove least salient from long-term
                    level.remove_least_salient(1)

            # Add to level
            success = level.add(memory)

            # FIX: If add failed due to capacity, force removal and try again
            if not success:
                level.remove_least_salient(1)
                success = level.add(memory)

            # Update stats
            if success:
                self.stats.total_memories += 1
                self.stats.by_type[memory_type] = (
                    self.stats.by_type.get(memory_type, 0) + 1
                )
                self.stats.total_stores += 1

            # Clear caches
            self.retrieval_cache.clear()

            return memory

    def retrieve(self, query: MemoryQuery) -> RetrievalResult:
        """Retrieve memories matching query."""
        start_time = time.time()

        # Check cache
        cache_key = self._get_cache_key(query)
        if cache_key in self.retrieval_cache:
            self.stats.cache_hit_rate += 0.01
            return self.retrieval_cache[cache_key]

        with self._lock:
            # Collect all memories
            all_memories = []
            for level in self.levels.values():
                all_memories.extend(level.memories.values())

            # Apply filters
            filtered = self._apply_filters(all_memories, query)

            # Compute similarities
            if query.embedding is not None:
                scores = self._compute_similarities(filtered, query.embedding)
            elif query.content is not None:
                # Generate embedding for query content
                query_embedding = self._generate_embedding(query.content)
                scores = self._compute_similarities(filtered, query_embedding)
            else:
                scores = [m.compute_salience() for m in filtered]

            # Apply attention if we have embeddings and valid scores
            if (
                len(filtered) > 0
                and len(scores) > 0
                and (query.embedding is not None or query.content is not None)
            ):
                # Get the embedding used for search
                final_query_embedding = (
                    query.embedding
                    if query.embedding is not None
                    else (query_embedding if "query_embedding" in locals() else None)
                )

                memory_embeddings = [
                    m.embedding for m in filtered if m.embedding is not None
                ]
                if (
                    memory_embeddings
                    and len(memory_embeddings) == len(filtered)
                    and final_query_embedding is not None
                ):
                    try:
                        attention_weights = self.attention.compute_attention(
                            final_query_embedding, memory_embeddings
                        )

                        # Combine scores with attention
                        if attention_weights is not None and len(
                            attention_weights
                        ) == len(scores):
                            scores = [s * a for s, a in zip(scores, attention_weights)]
                    except Exception as e:
                        # FIX: Handle attention mechanism errors gracefully
                        logger.warning(
                            f"Attention mechanism failed: {e}, using raw scores"
                        )

            # Sort and limit
            sorted_pairs = sorted(
                zip(filtered, scores), key=lambda x: x[1], reverse=True
            )[: query.limit]

            memories = [m for m, _ in sorted_pairs]
            scores = [s for _, s in sorted_pairs]

            # Update access counts
            for memory in memories:
                memory.access()

            # Create result
            result = RetrievalResult(
                memories=memories,
                scores=scores,
                query_time_ms=(time.time() - start_time) * 1000,
                total_matches=len(filtered),
            )

            # Cache result
            self.retrieval_cache[cache_key] = result

            # Update stats
            self.stats.total_queries += 1
            self.stats.avg_retrieval_time_ms = (
                self.stats.avg_retrieval_time_ms * (self.stats.total_queries - 1)
                + result.query_time_ms
            ) / self.stats.total_queries

            return result

    def forget(self, memory_id: str) -> bool:
        """Remove memory by ID."""
        with self._lock:
            for level in self.levels.values():
                if memory_id in level.memories:
                    memory = level.memories[memory_id]
                    del level.memories[memory_id]

                    # Update stats
                    self.stats.total_memories -= 1
                    if memory.type in self.stats.by_type:
                        self.stats.by_type[memory.type] = max(
                            0, self.stats.by_type[memory.type] - 1
                        )

                    self.retrieval_cache.clear()
                    return True
            return False

    def consolidate(self) -> int:
        """Consolidate memories between levels."""
        with self._lock:
            consolidated_count = 0

            # FIX: Lower thresholds or boost importance for consolidation
            # Consolidate from sensory to working
            candidates = self.levels["sensory"].get_candidates_for_consolidation()
            for memory in candidates:
                if self._promote_memory(memory, "sensory", "working"):
                    consolidated_count += 1

            # Consolidate from working to short-term
            candidates = self.levels["working"].get_candidates_for_consolidation()
            for memory in candidates:
                if self._promote_memory(memory, "working", "short_term"):
                    consolidated_count += 1

            # Consolidate from short-term to long-term
            candidates = self.levels["short_term"].get_candidates_for_consolidation()
            for memory in candidates:
                if self._promote_memory(memory, "short_term", "long_term"):
                    consolidated_count += 1

            self.stats.total_consolidations += 1

            return consolidated_count

    def _promote_memory(self, memory: Memory, from_level: str, to_level: str) -> bool:
        """Promote memory to higher level."""
        from_lvl = self.levels[from_level]
        to_lvl = self.levels[to_level]

        # Check capacity
        if len(to_lvl.memories) >= to_lvl.capacity:
            # Make space
            to_lvl.remove_least_salient(1)

        # Move memory
        if memory.id in from_lvl.memories:
            del from_lvl.memories[memory.id]
            success = to_lvl.add(memory)

            # FIX: Ensure memory is actually added
            if not success:
                to_lvl.remove_least_salient(1)
                success = to_lvl.add(memory)

            if success:
                # Update memory properties
                memory.decay_rate = to_lvl.decay_rate
                return True

        return False

    def _promote_memories(self, level_name: str):
        """Promote memories from one level to next."""
        level_order = ["sensory", "working", "short_term", "long_term"]

        if level_name not in level_order[:-1]:
            return

        current_idx = level_order.index(level_name)
        next_level = level_order[current_idx + 1]

        candidates = self.levels[level_name].get_candidates_for_consolidation()

        for memory in candidates[:5]:  # Promote up to 5 at a time
            self._promote_memory(memory, level_name, next_level)

    def _get_level_for_type(self, memory_type: MemoryType) -> str:
        """Get level name for memory type."""
        mapping = {
            MemoryType.SENSORY: "sensory",
            MemoryType.WORKING: "working",
            MemoryType.EPISODIC: "short_term",
            MemoryType.SEMANTIC: "long_term",
            MemoryType.PROCEDURAL: "long_term",
            MemoryType.LONG_TERM: "long_term",
            MemoryType.CACHE: "working",
        }
        return mapping.get(memory_type, "sensory")

    def _generate_memory_id(self, content: Any) -> str:
        """Generate unique memory ID."""
        content_str = str(content)[:1000]
        timestamp = str(time.time())
        combined = f"{content_str}_{timestamp}"
        return hashlib.md5(combined.encode(), usedforsecurity=False).hexdigest()

    def _generate_embedding(self, content: Any) -> np.ndarray:
        """Generate embedding for content using sentence transformers or fallback."""
        # Check cache
        content_key = str(content)[:500]
        if content_key in self.embedding_cache:
            return self.embedding_cache[content_key]

        embedding = None

        # Try to use sentence transformer model
        if self.embedding_model is not None:
            try:
                # Convert content to string if necessary
                if isinstance(content, str):
                    text = content
                elif isinstance(content, dict):
                    # Handle dictionary content
                    text = json.dumps(content, default=str)
                elif isinstance(content, (list, tuple)):
                    # Handle list/tuple content
                    text = " ".join(str(item) for item in content)
                else:
                    text = str(content)

                # Truncate if too long (most models have token limits)
                max_length = 512
                if len(text) > max_length:
                    text = text[:max_length]

                # Generate embedding
                if TORCH_AVAILABLE and torch is not None:
                    with torch.no_grad():
                        embedding = self.embedding_model.encode(
                            text, convert_to_numpy=True, show_progress_bar=False
                        )
                else:
                    embedding = self.embedding_model.encode(
                        text, convert_to_numpy=True, show_progress_bar=False
                    )

                # Ensure it's a numpy array
                if not isinstance(embedding, np.ndarray):
                    embedding = np.array(embedding)

                # Normalize
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm

            except Exception as e:
                logger.warning(f"Failed to generate embedding with model: {e}")
                embedding = None

        # Fallback to hash-based embedding if model fails
        if embedding is None:
            content_str = str(content)
            hash_obj = hashlib.sha256(content_str.encode())
            hash_bytes = hash_obj.digest()

            # Convert to normalized vector with correct dimension
            raw_embedding = np.frombuffer(hash_bytes, dtype=np.uint8)

            # Extend or truncate to match expected dimension
            if len(raw_embedding) < self.embedding_dimension:
                # Pad with zeros
                embedding = np.zeros(self.embedding_dimension, dtype=np.float32)
                embedding[: len(raw_embedding)] = raw_embedding.astype(np.float32)
            else:
                # Truncate
                embedding = raw_embedding[: self.embedding_dimension].astype(np.float32)

            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

        # Check cache size BEFORE adding new entry
        if len(self.embedding_cache) >= 1000:
            # Remove oldest 10% of entries
            keys_to_remove = list(self.embedding_cache.keys())[:100]
            for key in keys_to_remove:
                del self.embedding_cache[key]

        # Now add the new embedding
        self.embedding_cache[content_key] = embedding

        return embedding

    def _apply_filters(
        self, memories: List[Memory], query: MemoryQuery
    ) -> List[Memory]:
        """Apply query filters to memories."""
        filtered = memories

        # Time range filter
        if query.time_range:
            start, end = query.time_range
            filtered = [m for m in filtered if start <= m.timestamp <= end]

        # Type filter
        if "type" in query.filters:
            memory_type = query.filters["type"]
            filtered = [m for m in filtered if m.type == memory_type]

        # Importance filter
        if "min_importance" in query.filters:
            min_imp = query.filters["min_importance"]
            filtered = [m for m in filtered if m.importance >= min_imp]

        # Content-based filter
        if "content_contains" in query.filters:
            search_text = query.filters["content_contains"].lower()
            filtered = [m for m in filtered if search_text in str(m.content).lower()]

        # Metadata filter
        if "metadata" in query.filters:
            meta_filters = query.filters["metadata"]
            filtered = [
                m
                for m in filtered
                if all(m.metadata.get(k) == v for k, v in meta_filters.items())
            ]

        return filtered

    def _compute_similarities(
        self, memories: List[Memory], query_embedding: np.ndarray
    ) -> List[float]:
        """Compute similarity scores using cosine similarity."""
        scores = []

        # Ensure query embedding is normalized
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm

        # FIX: Validate query embedding dimension
        query_dim = len(query_embedding)

        for memory in memories:
            if memory.embedding is not None:
                mem_dim = len(memory.embedding)

                # FIX: Check dimension match
                if mem_dim != query_dim:
                    logger.warning(
                        f"Dimension mismatch: query={query_dim}, memory={mem_dim}"
                    )
                    scores.append(0.0)  # No match for incompatible dimensions
                    continue

                # Ensure memory embedding is normalized
                mem_norm = np.linalg.norm(memory.embedding)
                if mem_norm > 0:
                    normalized_mem = memory.embedding / mem_norm
                else:
                    normalized_mem = memory.embedding

                # Compute cosine similarity (dot product of normalized vectors)
                similarity = np.dot(normalized_mem, query_embedding)

                # Ensure similarity is in [-1, 1] range
                similarity = np.clip(similarity, -1.0, 1.0)

                # Convert to [0, 1] range for scoring
                score = (similarity + 1.0) / 2.0
                scores.append(float(score))
            else:
                # No embedding available, use low default score
                scores.append(0.0)

        return scores

    def _get_cache_key(self, query: MemoryQuery) -> str:
        """Generate cache key for query."""
        key_parts = [
            query.query_type,
            json.dumps(query.filters, sort_keys=True),
            str(query.time_range),
            str(query.limit),
            str(query.threshold),
        ]

        # Add content hash if present
        if query.content is not None:
            key_parts.append(str(hash(str(query.content))))

        key_str = "_".join(key_parts)
        return hashlib.md5(key_str.encode(), usedforsecurity=False).hexdigest()

    def start_background_tasks(self):
        """Start background consolidation and pattern mining tasks."""
        # Use shorter intervals in CI environment
        is_ci = os.environ.get('CI') or os.environ.get('GITHUB_ACTIONS') or os.environ.get('VULCAN_CI_MODE')
        
        # Consolidation thread
        consolidation_interval = getattr(self.config, "consolidation_interval", 1000)
        
        # Override with shorter interval in CI to prevent hanging
        if is_ci:
            consolidation_interval = min(consolidation_interval, 5)  # Max 5 seconds in CI
        
        if consolidation_interval > 0:

            def consolidation_loop():
                while not self._shutdown_event.is_set():  # CHANGED
                    if self._shutdown_event.wait(
                        timeout=consolidation_interval
                    ):  # CHANGED
                        break  # Shutdown signaled
                    try:
                        self.consolidate()
                    except Exception as e:
                        logger.error(f"Consolidation error: {e}")

            self.consolidation_thread = threading.Thread(
                target=consolidation_loop, daemon=True
            )
            self.consolidation_thread.start()

        # Pattern mining thread - also use shorter interval in CI
        pattern_interval = 300 if not is_ci else 5
        
        def pattern_mining_loop():
            while not self._shutdown_event.is_set():  # CHANGED
                if self._shutdown_event.wait(timeout=pattern_interval):  # CHANGED
                    break  # Shutdown signaled
                try:
                    if len(self.tool_selection_history) > 10:
                        self.mine_patterns()
                except Exception as e:
                    logger.error(f"Pattern mining error: {e}")

        self.pattern_mining_thread = threading.Thread(
            target=pattern_mining_loop, daemon=True
        )
        self.pattern_mining_thread.start()

    # ADD THIS ENTIRE METHOD to the HierarchicalMemory class
    def shutdown(self):
        """Shuts down background threads."""
        logger.info("Shutting down HierarchicalMemory background tasks...")
        self._shutdown_event.set()

        if self.consolidation_thread and self.consolidation_thread.is_alive():
            self.consolidation_thread.join(timeout=5)

        if self.pattern_mining_thread and self.pattern_mining_thread.is_alive():
            self.pattern_mining_thread.join(timeout=5)

        logger.info("HierarchicalMemory shutdown complete.")

    def update_embedding_model(self, model_name: str):
        """Update the embedding model at runtime."""
        with self._lock:
            old_model = self.embedding_model
            old_cache = self.embedding_cache.copy()
            old_dimension = self.embedding_dimension

            try:
                # Try to load new model
                self.embedding_model = self._init_embedding_model(model_name)
                self.embedding_dimension = self._get_embedding_dimension()

                # FIX: Update attention mechanism with new dimension
                self.attention = AttentionMechanism(
                    hidden_dim=min(256, self.embedding_dimension),
                    input_dim=self.embedding_dimension,
                )

                # Clear embedding cache as dimensions might have changed
                self.embedding_cache.clear()

                # Regenerate embeddings for all memories
                for level in self.levels.values():
                    for memory in level.memories.values():
                        memory.embedding = self._generate_embedding(memory.content)

                logger.info(f"Successfully updated embedding model to {model_name}")

            except Exception as e:
                # Rollback on failure
                logger.error(f"Failed to update embedding model: {e}")
                self.embedding_model = old_model
                self.embedding_dimension = old_dimension
                self.embedding_cache = old_cache
                # Restore old attention mechanism
                self.attention = AttentionMechanism(
                    hidden_dim=min(256, self.embedding_dimension),
                    input_dim=self.embedding_dimension,
                )
                raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current embedding model."""
        info = {
            "has_model": self.embedding_model is not None,
            "embedding_dimension": self.embedding_dimension,
            "cache_size": len(self.embedding_cache),
        }

        if self.embedding_model:
            info["model_type"] = "sentence_transformer"
            if hasattr(self.embedding_model, "model_name_or_path"):
                info["model_name"] = self.embedding_model.model_name_or_path
        else:
            info["model_type"] = "hash_fallback"

        return info

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.shutdown()  # ADDED
            if hasattr(super(), "cleanup"):
                super().cleanup()
        except Exception as e:
            logger.debug(
                f"{self.__class__.__name__ if hasattr(self, '__class__') else 'Operation'} error: {e}"
            )


# ============================================================
# PERSISTENT HIERARCHICAL MEMORY IMPLEMENTATION
# ============================================================


class EpisodicMemory:
    """Simple episodic memory store for recent items."""

    def __init__(self):
        self.items = []

    def add(self, x):
        self.items.append(x)

    def search(self, q, k=10):
        return self.items[:k]


class SemanticMemory:
    """Semantic memory store for facts and concepts."""

    def __init__(self):
        self.facts = {}  # id -> fact content
        self.concepts = defaultdict(set)  # concept -> set of fact ids

    def add(self, fact_id: str, content: Any, concepts: List[str] = None):
        """Add a fact to semantic memory."""
        self.facts[fact_id] = content
        if concepts:
            for concept in concepts:
                self.concepts[concept].add(fact_id)

    def search(self, query: str, k: int = 10) -> List[Any]:
        """Search semantic memory."""
        results = []
        
        # Simple keyword-based search
        query_lower = query.lower()
        for fact_id, content in self.facts.items():
            if query_lower in str(content).lower():
                results.append(content)
                if len(results) >= k:
                    break
        
        return results

    def get_related_concepts(self, concept: str) -> List[str]:
        """Get fact IDs related to a concept."""
        return list(self.concepts.get(concept, set()))


class ProceduralMemory:
    """Procedural memory store for skills and procedures."""

    def __init__(self):
        self.procedures = {}  # id -> procedure
        self.skills = {}  # skill_name -> proficiency_level

    def add(self, procedure_id: str, procedure: Any, skill_name: str = None):
        """Add a procedure to memory."""
        self.procedures[procedure_id] = procedure
        if skill_name:
            # Track skill proficiency (increases with practice)
            current_level = self.skills.get(skill_name, 0.0)
            self.skills[skill_name] = min(1.0, current_level + 0.1)

    def search(self, query: str = None, k: int = 10) -> List[Any]:
        """Search procedural memory."""
        # Return recent procedures
        results = list(self.procedures.values())[:k]
        return results

    def get_skill_level(self, skill_name: str) -> float:
        """Get proficiency level for a skill."""
        return self.skills.get(skill_name, 0.0)


@dataclass
class PersistentMemoryConfig:
    """Configuration for persistent hierarchical memory."""

    memory_bucket: str
    cdn_url: str
    max_context: int = 8192


class PersistentHierarchicalMemory:
    """
    Hierarchical memory with persistent storage backend.

    This implementation uses:
    - EpisodicMemory for recent, frequently accessed items
    - Local disk for intermediate storage
    - PackfileStore with S3/CloudFront for long-term persistent storage
    - MerkleLSM for compaction
    - GraphRAG for semantic retrieval
    - UnlearningEngine for controlled forgetting
    - ZKProver for cryptographic proofs
    - LearningStatePersistence for learning state recovery
    """

    # Class-level singleton for learning persistence (shared across instances)
    _learning_persistence_instance: Optional["LearningStatePersistence"] = None

    def __init__(self, config: PersistentMemoryConfig):
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory()
        self.procedural = ProceduralMemory()
        
        # FIX Issue #8: Add guards for persistent memory components
        if PERSISTENT_MEMORY_AVAILABLE:
            # Persistent storage backend (S3Store aliased as PackfileStore)
            self.persistent_store = PackfileStore(
                bucket=config.memory_bucket,
            )

            # LSM tree for efficient compaction
            self.lsm_compactor = MerkleLSM(32, "adaptive", True)

            # Graph-based retrieval augmented generation
            # FIX: Changed from invalid "llm_embeddings" to valid model "all-MiniLM-L6-v2"
            self.graph_rag = GraphRAG("all-MiniLM-L6-v2", "disk_based_tier_c", True)

            # Unlearning engine for privacy and data deletion
            self.unlearning_engine = UnlearningEngine(merkle_graph=self.lsm_compactor)

            # Zero-knowledge proofs for verification
            self.zk_prover = ZKProver()
            
            logging.info("PersistentHierarchicalMemory initialized with full persistent backend")
        else:
            # Fallback: Use local memory only
            self.persistent_store = None
            self.lsm_compactor = None
            self.graph_rag = None
            self.unlearning_engine = None
            self.zk_prover = None
            
            logging.warning(
                "PersistentHierarchicalMemory initialized without persistent backend "
                "(persistant_memory_v46 not available). Using local memory only."
            )
        
        # Initialize learning persistence singleton
        self._init_learning_persistence()

    @classmethod
    def _init_learning_persistence(cls) -> None:
        """Initialize the learning persistence singleton if not already done."""
        if cls._learning_persistence_instance is None:
            try:
                from .learning_persistence import LearningStatePersistence
                cls._learning_persistence_instance = LearningStatePersistence()
                logger.info("LearningStatePersistence initialized for PersistentHierarchicalMemory")
            except Exception as e:
                logger.warning(f"Failed to initialize LearningStatePersistence: {e}")

    @classmethod
    def get_learning_persistence(cls) -> Optional["LearningStatePersistence"]:
        """Get the learning persistence singleton instance."""
        if cls._learning_persistence_instance is None:
            cls._init_learning_persistence()
        return cls._learning_persistence_instance

    def _is_recent(self, item) -> bool:
        """Check if item is recent enough for episodic memory."""
        if not hasattr(item, 'timestamp'):
            return True  # Default to episodic if no timestamp
        
        # Recent = last 24 hours
        age_hours = (time.time() - item.timestamp) / 3600
        return age_hours < 24

    def _is_frequently_accessed(self, item) -> bool:
        """Check if item is frequently accessed."""
        # Check access count if available
        access_count = getattr(item, 'access_count', 0)
        return access_count > 10  # Threshold for "frequent"

    def _serialize_to_local(self, item) -> None:
        """Serialize item to local disk storage."""
        import pickle
        from pathlib import Path
        
        # Create local cache directory
        storage_dir = Path("./local_memory_cache")
        storage_dir.mkdir(exist_ok=True)
        
        # Generate unique filename
        item_id = getattr(item, 'id', str(hash(str(item))))
        filepath = storage_dir / f"{item_id}.pkl"
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(item, f)
        except Exception as e:
            logger.error(f"Failed to serialize item to local disk: {e}")

    def _search_local_disk(self, q, k=10) -> list:
        """Search local disk storage."""
        import pickle
        from pathlib import Path
        
        storage_dir = Path("./local_memory_cache")
        if not storage_dir.exists():
            return []
        
        results = []
        try:
            # Iterate through cached files
            for filepath in storage_dir.glob("*.pkl"):
                try:
                    with open(filepath, 'rb') as f:
                        item = pickle.load(f)  # nosec B301 - Internal data structure
                        results.append(item)
                        
                        if len(results) >= k:
                            break
                            
                except Exception as e:
                    logger.debug(f"Failed to load {filepath}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error searching local disk: {e}")
        
        return results

    def store(self, memory_item):
        """
        Store a memory item in the appropriate tier.

        Storage tiers:
        1. Episodic memory (hot tier) - for recent items
        2. Local disk (warm tier) - for frequently accessed items
        3. Persistent store (cold tier) - for long-term storage
        """
        if self._is_recent(memory_item):
            self.episodic.add(memory_item)
        elif self._is_frequently_accessed(memory_item):
            self._serialize_to_local(memory_item)
        else:
            # Compact and upload to persistent storage (if available)
            if PERSISTENT_MEMORY_AVAILABLE and self.lsm_compactor and self.persistent_store:
                packfile = self.lsm_compactor.compact([memory_item])
                pack_id = str(uuid.uuid4())
                self.persistent_store.put_object(key=f"packs/{pack_id}.pack", data=packfile)
            else:
                # Fallback: Store locally
                self._serialize_to_local(memory_item)

    def retrieve(self, query, k=10):
        """
        Retrieve memories matching the query.

        Retrieval strategy:
        1. Search episodic memory first (fastest)
        2. Search local disk if needed
        3. Search persistent store via GraphRAG if still needed
        """
        # Search episodic memory
        results = self.episodic.search(query, k=k)

        # Search local disk if more results needed
        if len(results) < k:
            results.extend(self._search_local_disk(query, k=k - len(results)))

        # Search persistent store via GraphRAG if still more needed
        if len(results) < k and PERSISTENT_MEMORY_AVAILABLE and self.graph_rag:
            try:
                results.extend(
                    self.graph_rag.retrieve(
                        query, k=k - len(results), parent_child_context=True
                    )
                )
            except Exception as e:
                logging.debug(f"GraphRAG retrieve failed: {e}")

        return results

    def store_interaction(
        self,
        query_id: str,
        query: str,
        answer: str,
        tools_used: Optional[List[str]] = None,
        success: bool = True,
        latency_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Store a query/answer interaction in both memory tiers and learning persistence.
        
        This ensures interactions are:
        1. Available for semantic search via GraphRAG
        2. Persisted to disk for recovery via LearningStatePersistence
        3. Used for learning (tool weight updates)
        
        Args:
            query_id: Unique identifier for the query.
            query: The user's query text.
            answer: The system's response text.
            tools_used: List of tools used to generate the answer.
            success: Whether the interaction was successful.
            latency_ms: Response latency in milliseconds.
            metadata: Additional metadata about the interaction.
        """
        # Create interaction record
        interaction_content = {
            "query_id": query_id,
            "query": query,
            "answer": answer,
            "tools_used": tools_used or [],
            "success": success,
            "timestamp": time.time(),
            "latency_ms": latency_ms,
            "metadata": metadata or {},
        }
        
        # Store in GraphRAG for semantic retrieval (if available)
        if PERSISTENT_MEMORY_AVAILABLE and self.graph_rag:
            try:
                doc_id = f"interaction_{query_id}"
                combined_text = f"Query: {query}\n\nAnswer: {answer}"
                self.graph_rag.add_document(
                    doc_id=doc_id,
                    content=combined_text,
                    metadata={
                        "type": "interaction",
                        "query_id": query_id,
                        "tools_used": tools_used or [],
                        "success": success,
                        "timestamp": interaction_content["timestamp"],
                    },
                    auto_chunk=True,
                )
                logger.debug(f"Stored interaction {query_id} in GraphRAG")
            except Exception as e:
                logger.warning(f"Failed to store interaction in GraphRAG: {e}")
        
        # Store in episodic memory for quick access
        try:
            from .base import Memory, MemoryType
            memory_item = Memory(
                id=f"interaction_{query_id}",
                type=MemoryType.EPISODIC,
                content=interaction_content,
                importance=0.7 if success else 0.4,
                metadata={"interaction": True, "tools": tools_used or []},
            )
            self.episodic.add(memory_item)
        except Exception as e:
            logger.debug(f"Failed to store interaction in episodic memory: {e}")
        
        # Store in learning persistence for recovery (using singleton)
        try:
            persistence = self.get_learning_persistence()
            if persistence:
                persistence.add_interaction(
                    query_id=query_id,
                    query=query,
                    answer=answer,
                    tools_used=tools_used,
                    success=success,
                    latency_ms=latency_ms,
                    metadata=metadata,
                )
        except Exception as e:
            logger.debug(f"Failed to store interaction in learning persistence: {e}")

    def search_interactions(
        self,
        query: str,
        k: int = 10,
        success_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar past interactions.
        
        Args:
            query: Search query text.
            k: Maximum number of results to return.
            success_only: If True, only return successful interactions.
        
        Returns:
            List of matching interaction records with keys:
            - query_id: Unique identifier of the interaction
            - content: The query/answer text content
            - score: Semantic similarity score
            - tools_used: List of tools used
            - success: Whether the interaction was successful
        """
        results: List[Dict[str, Any]] = []
        
        # Check if GraphRAG is available
        if not PERSISTENT_MEMORY_AVAILABLE or not self.graph_rag:
            logging.warning("GraphRAG not available, cannot search past interactions")
            return results
        
        try:
            # Search GraphRAG for semantic matches
            # RetrievalResult has: node_id, content, score, metadata, source, hop
            rag_results = self.graph_rag.retrieve(
                query=query,
                k=k,
                parent_child_context=True,
            )
            
            for result in rag_results:
                # RetrievalResult is a dataclass with metadata dict
                result_metadata = result.metadata if hasattr(result, 'metadata') else {}
                if result_metadata.get('type') == 'interaction':
                    if success_only and not result_metadata.get('success', True):
                        continue
                    results.append({
                        'query_id': result_metadata.get('query_id'),
                        'content': result.content if hasattr(result, 'content') else '',
                        'score': result.score if hasattr(result, 'score') else 0.0,
                        'tools_used': result_metadata.get('tools_used', []),
                        'success': result_metadata.get('success', True),
                    })
        except Exception as e:
            logger.warning(f"Failed to search interactions in GraphRAG: {e}")
        
        return results[:k]
