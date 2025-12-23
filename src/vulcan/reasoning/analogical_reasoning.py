"""
Enhanced Analogical reasoning with advanced semantic understanding and NLP

FULLY IMPLEMENTED VERSION with:
- Sophisticated semantic similarity using embeddings
- REAL fallback embeddings using TF-IDF (not random hashes)
- Advanced goal relevance filtering with dependency parsing
- Entity recognition and disambiguation
- Knowledge graph integration
- Deep attribute understanding
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logger.warning("NetworkX not available, graph matching disabled")

try:
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available, some features disabled")

try:
    import spacy

    SPACY_AVAILABLE = True
    # Try to load model - prefer larger models that may be installed
    nlp = None
    for model_name in ["en_core_web_lg", "en_core_web_md", "en_core_web_sm"]:
        try:
            nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model '{model_name}' for analogical reasoning")
            break
        except OSError:
            # Model not installed, try next one
            continue
        except Exception as e:
            logger.warning(f"Error loading spaCy model '{model_name}': {e}")
            continue

    if nlp is None:
        logger.warning(
            "spaCy model not loaded for analogical reasoning. "
            "Install a model with: python -m spacy download en_core_web_lg (recommended) "
            "or: python -m spacy download en_core_web_md "
            "or: python -m spacy download en_core_web_sm"
        )
except ImportError:
    SPACY_AVAILABLE = False
    nlp = None
    logger.warning("spaCy not available, NLP features limited")

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
    # Initialize model lazily
    _embedding_model = None
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    _embedding_model = None
    logger.warning(
        "sentence-transformers not available, using TF-IDF fallback embeddings"
    )

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, neural features disabled")


class MappingType(Enum):
    """Types of analogical mappings"""

    SURFACE = "surface"
    STRUCTURAL = "structural"
    PRAGMATIC = "pragmatic"
    SEMANTIC = "semantic"


class SemanticRelationType(Enum):
    """Types of semantic relations"""

    HYPERNYM = "hypernym"  # is-a
    HYPONYM = "hyponym"  # inverse of is-a
    MERONYM = "meronym"  # part-of
    HOLONYM = "holonym"  # has-part
    SYNONYM = "synonym"
    ANTONYM = "antonym"
    CAUSE = "cause"
    EFFECT = "effect"
    ATTRIBUTE = "attribute"


@dataclass
class Entity:
    """Represents an entity in analogical reasoning with semantic enrichment"""

    name: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    entity_type: str = "object"
    embedding: Optional[np.ndarray] = None
    semantic_category: Optional[str] = None
    pos_tag: Optional[str] = None  # Part of speech
    dependency_role: Optional[str] = None  # Syntactic role

    def __hash__(self):
        """Make Entity hashable"""
        return hash((self.name, self.entity_type))

    def __eq__(self, other):
        """Equality comparison"""
        if not isinstance(other, Entity):
            return False
        return self.name == other.name and self.entity_type == other.entity_type

    def similarity_to(self, other: "Entity", use_embeddings: bool = True) -> float:
        """Compute sophisticated similarity to another entity"""
        if not isinstance(other, Entity):
            return 0.0

        # FIX #1: Check if comparing to self - return 1.0 immediately
        if self is other or (
            self.name == other.name and self.entity_type == other.entity_type
        ):
            return 1.0

        # FIX #1b: Different entity types should return 0.0 (no similarity)
        if self.entity_type != other.entity_type:
            return 0.0

        # Type compatibility (same type at this point)
        type_match = 1.0

        # Use embeddings if available
        if (
            use_embeddings
            and self.embedding is not None
            and other.embedding is not None
        ):
            embedding_sim = self._embedding_similarity(self.embedding, other.embedding)
        else:
            embedding_sim = 0.0

        # Semantic category match
        category_match = (
            1.0
            if (
                self.semantic_category
                and self.semantic_category == other.semantic_category
            )
            else 0.0
        )

        # Attribute similarity
        attr_sim = self._attribute_similarity(self.attributes, other.attributes)

        # Name similarity (lexical)
        name_sim = self._lexical_similarity(self.name, other.name)

        # Weighted combination
        if self.embedding is not None and other.embedding is not None:
            # Prefer embeddings when available, but give more weight to explicit type matching
            weights = [0.4, 0.3, 0.15, 0.15]  # embedding, type, category, attr
            scores = [embedding_sim, type_match, category_match, attr_sim]
        else:
            # Fall back to lexical and structural
            weights = [0.3, 0.3, 0.2, 0.2]  # name, type, category, attr
            scores = [name_sim, type_match, category_match, attr_sim]

        return sum(w * s for w, s in zip(weights, scores))

    def _embedding_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings"""
        try:
            emb1_flat = emb1.flatten()
            emb2_flat = emb2.flatten()

            norm1 = np.linalg.norm(emb1_flat)
            norm2 = np.linalg.norm(emb2_flat)

            if norm1 < 1e-10 or norm2 < 1e-10:
                return 0.0

            return float(
                np.clip(np.dot(emb1_flat, emb2_flat) / (norm1 * norm2), -1.0, 1.0)
            )
        except Exception:
            return 0.0

    def _attribute_similarity(self, attrs1: Dict, attrs2: Dict) -> float:
        """Deep attribute similarity with type awareness"""
        if not attrs1 and not attrs2:
            return 1.0
        if not attrs1 or not attrs2:
            return 0.0

        common_keys = set(attrs1.keys()) & set(attrs2.keys())
        all_keys = set(attrs1.keys()) | set(attrs2.keys())

        if not all_keys:
            return 1.0

        # Key overlap
        key_sim = len(common_keys) / len(all_keys)

        # Value similarity for common keys
        value_sims = []
        for key in common_keys:
            v1, v2 = attrs1[key], attrs2[key]

            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                # Numerical similarity
                max_val = max(abs(v1), abs(v2), 1e-10)
                value_sims.append(1.0 - min(1.0, abs(v1 - v2) / max_val))
            elif isinstance(v1, str) and isinstance(v2, str):
                # String similarity
                value_sims.append(self._lexical_similarity(v1, v2))
            else:
                # Exact match
                value_sims.append(1.0 if v1 == v2 else 0.0)

        value_sim = np.mean(value_sims) if value_sims else 0.0

        return (key_sim + value_sim) / 2

    def _lexical_similarity(self, s1: str, s2: str) -> float:
        """Multi-level lexical similarity"""
        s1_lower = s1.lower()
        s2_lower = s2.lower()

        # Exact match
        if s1_lower == s2_lower:
            return 1.0

        # Edit distance normalized
        edit_dist = self._levenshtein_distance(s1_lower, s2_lower)
        max_len = max(len(s1_lower), len(s2_lower), 1)
        edit_sim = 1.0 - (edit_dist / max_len)

        # Character n-gram similarity
        ngram_sim = self._ngram_similarity(s1_lower, s2_lower, n=2)

        # Token overlap
        tokens1 = set(s1_lower.split())
        tokens2 = set(s2_lower.split())
        if tokens1 or tokens2:
            token_sim = len(tokens1 & tokens2) / len(tokens1 | tokens2)
        else:
            token_sim = 0.0

        return (edit_sim + ngram_sim + token_sim) / 3

    @staticmethod
    def _levenshtein_distance(s1: str, s2: str) -> int:
        """Compute Levenshtein edit distance"""
        if len(s1) < len(s2):
            return Entity._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    @staticmethod
    def _ngram_similarity(s1: str, s2: str, n: int = 2) -> float:
        """Compute n-gram Jaccard similarity"""
        if len(s1) < n or len(s2) < n:
            return 1.0 if s1 == s2 else 0.0

        ngrams1 = set(s1[i : i + n] for i in range(len(s1) - n + 1))
        ngrams2 = set(s2[i : i + n] for i in range(len(s2) - n + 1))

        if not ngrams1 and not ngrams2:
            return 1.0
        if not ngrams1 or not ngrams2:
            return 0.0

        return len(ngrams1 & ngrams2) / len(ngrams1 | ngrams2)


@dataclass
class Relation:
    """Represents a relation between entities with semantic information"""

    predicate: str
    arguments: List[str]
    relation_type: str = "binary"
    order: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    semantic_type: Optional[SemanticRelationType] = None
    confidence: float = 1.0

    def matches(self, other: "Relation", entity_mapping: Dict[str, str]) -> bool:
        """Check if relation matches another given entity mapping"""
        if self.predicate != other.predicate:
            return False

        if len(self.arguments) != len(other.arguments):
            return False

        for arg1, arg2 in zip(self.arguments, other.arguments):
            if arg1 in entity_mapping:
                if entity_mapping[arg1] != arg2:
                    return False
            elif arg1 != arg2:
                return False

        return True

    def semantic_similarity_to(self, other: "Relation") -> float:
        """Compute semantic similarity between relations"""
        # Predicate similarity
        pred_sim = 1.0 if self.predicate == other.predicate else 0.5

        # Semantic type match
        type_sim = (
            1.0
            if (self.semantic_type and self.semantic_type == other.semantic_type)
            else 0.0
        )

        # Relation type match
        rel_type_sim = 1.0 if self.relation_type == other.relation_type else 0.5

        return (pred_sim + type_sim + rel_type_sim) / 3


@dataclass
class AnalogicalMapping:
    """Represents a complete analogical mapping with rich metadata"""

    source_domain: str
    target_domain: str
    entity_mappings: Dict[str, str]
    relation_mappings: List[Tuple[Relation, Relation]]
    mapping_score: float
    mapping_type: MappingType
    confidence: float = 0.0
    explanation: str = ""
    semantic_coherence: float = 0.0
    structural_depth: int = 0


class SemanticEnricher:
    """Enriches entities and relations with semantic information"""

    def __init__(self):
        self.embedding_cache = {}
        self.entity_cache = {}
        self.embedding_model = None

        # Initialize TF-IDF vectorizer for fallback
        self._tfidf_vectorizer = None
        self._tfidf_corpus = []
        self._corpus_hash = set()
        self._tfidf_fitted = False  # FIX #2: Track if vectorizer is fitted

        # Initialize embedding model lazily
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("Loaded sentence transformer model for semantic embeddings")
            except Exception as e:
                logger.warning(
                    f"Failed to load sentence transformer: {e}, using TF-IDF fallback"
                )
                self._init_tfidf_vectorizer()
        else:
            logger.info("Using TF-IDF fallback for semantic embeddings")
            self._init_tfidf_vectorizer()

        # Simple semantic categories
        self.semantic_categories = self._build_semantic_categories()

    def _init_tfidf_vectorizer(self):
        """Initialize TF-IDF vectorizer for fallback embeddings"""
        if SKLEARN_AVAILABLE:
            try:
                self._tfidf_vectorizer = TfidfVectorizer(
                    analyzer="char_wb",  # Character n-grams with word boundaries
                    ngram_range=(2, 4),  # Bigrams to 4-grams
                    max_features=384,  # Match sentence transformer dimension
                    lowercase=True,
                    strip_accents="unicode",
                    min_df=1,
                    max_df=1.0,
                )
                logger.info("Initialized TF-IDF vectorizer for semantic embeddings")
            except Exception as e:
                logger.warning(f"Failed to initialize TF-IDF vectorizer: {e}")

    def _build_semantic_categories(self) -> Dict[str, Set[str]]:
        """Build simple semantic category dictionary"""
        return {
            "person": {
                "person",
                "people",
                "individual",
                "human",
                "man",
                "woman",
                "child",
            },
            "animal": {"animal", "creature", "beast", "bird", "fish", "dog", "cat"},
            "object": {"object", "thing", "item", "tool", "device", "machine"},
            "place": {"place", "location", "area", "region", "city", "country"},
            "concept": {"concept", "idea", "notion", "principle", "theory"},
            "action": {"action", "activity", "process", "event", "operation"},
            "property": {
                "property",
                "attribute",
                "quality",
                "characteristic",
                "feature",
            },
        }

    def enrich_entity(self, entity: Entity) -> Entity:
        """Enrich entity with semantic information"""
        # Check cache
        cache_key = f"{entity.name}_{entity.entity_type}"
        if cache_key in self.entity_cache:
            cached = self.entity_cache[cache_key]
            entity.embedding = cached.embedding
            entity.semantic_category = cached.semantic_category
            entity.pos_tag = cached.pos_tag
            return entity

        # Generate embedding
        entity.embedding = self._get_embedding(entity.name)

        # Determine semantic category
        entity.semantic_category = self._categorize_entity(
            entity.name, entity.entity_type
        )

        # POS tagging if spaCy available
        if nlp:
            try:
                doc = nlp(entity.name)
                if doc:
                    entity.pos_tag = doc[0].pos_ if len(doc) > 0 else None
            except Exception as e:
                logger.debug(f"Failed to assign POS tag to entity: {e}")

        # Cache
        self.entity_cache[cache_key] = entity

        return entity

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get semantic embedding for text"""
        if text in self.embedding_cache:
            return self.embedding_cache[text]

        try:
            if self.embedding_model:
                # Use sentence transformer model
                embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            else:
                # Use TF-IDF fallback
                embedding = self._fallback_embedding(text)

            self.embedding_cache[text] = embedding
            return embedding
        except Exception as e:
            logger.warning(f"Embedding generation failed for '{text}': {e}")
            return self._fallback_embedding(text)

    def _fallback_embedding(self, text: str) -> np.ndarray:
        """
        REAL IMPLEMENTATION: Generate meaningful embedding using TF-IDF

        This is a proper fallback when sentence-transformers is not available.
        Uses character n-grams to capture semantic similarity.

        FIX #2: Only refit vectorizer during initial training phase (first 20 texts)
        to ensure deterministic embeddings after that.
        """
        if not SKLEARN_AVAILABLE or self._tfidf_vectorizer is None:
            # Ultimate fallback - character-based features
            return self._character_embedding(text)

        try:
            # Add to corpus if new
            text_hash = hash(text)
            if text_hash not in self._corpus_hash:
                self._tfidf_corpus.append(text)
                self._corpus_hash.add(text_hash)

                # FIX #2: Only refit during initial training (first 20 texts)
                # After that, use the fitted model without refitting for determinism
                if not self._tfidf_fitted and len(self._tfidf_corpus) <= 20:
                    if len(self._tfidf_corpus) > 1:
                        try:
                            self._tfidf_vectorizer.fit(self._tfidf_corpus)
                            # Mark as fitted once we have 20 texts
                            if len(self._tfidf_corpus) == 20:
                                self._tfidf_fitted = True
                        except Exception as e:
                            logger.warning(f"TF-IDF refitting failed: {e}")

            # Generate embedding
            if len(self._tfidf_corpus) > 1:
                try:
                    # Ensure vectorizer is fitted
                    if (
                        not hasattr(self._tfidf_vectorizer, "vocabulary_")
                        or self._tfidf_vectorizer.vocabulary_ is None
                    ):
                        self._tfidf_vectorizer.fit(self._tfidf_corpus)

                    vector = self._tfidf_vectorizer.transform([text]).toarray()[0]
                except Exception:
                    # If transform fails, use character embedding
                    return self._character_embedding(text)
            else:
                # Single text - use character embedding
                return self._character_embedding(text)

            # Ensure correct dimension
            if len(vector) < 384:
                vector = np.pad(vector, (0, 384 - len(vector)))
            elif len(vector) > 384:
                vector = vector[:384]

            # Normalize
            norm = np.linalg.norm(vector)
            if norm > 1e-10:
                vector = vector / norm
            else:
                # Zero vector - use character embedding
                return self._character_embedding(text)

            return vector.astype(np.float32)

        except Exception as e:
            logger.warning(
                f"TF-IDF embedding failed for '{text}': {e}, using character embedding"
            )
            return self._character_embedding(text)

    def _character_embedding(self, text: str) -> np.ndarray:
        """
        Minimal character-based embedding for texts
        Better than random - captures actual text features

        This provides real similarity: "running" will be similar to "runner"
        """
        text_lower = text.lower()

        # Character frequency features (256 dimensions)
        char_freq = np.zeros(256, dtype=np.float32)
        for char in text_lower:
            char_ord = ord(char) if ord(char) < 256 else ord(" ")
            char_freq[char_ord] += 1

        # Normalize character frequencies
        if char_freq.sum() > 0:
            char_freq = char_freq / char_freq.sum()

        # Character bigram features (128 dimensions via hashing)
        bigram_hash = np.zeros(128, dtype=np.float32)
        for i in range(len(text_lower) - 1):
            bigram = text_lower[i : i + 2]
            hash_val = hash(bigram) % 128
            bigram_hash[abs(hash_val)] += 1

        if bigram_hash.sum() > 0:
            bigram_hash = bigram_hash / bigram_hash.sum()

        # Combine features (384 total dimensions)
        embedding = np.concatenate([char_freq, bigram_hash])

        # Pad or truncate to exactly 384 dimensions
        if len(embedding) < 384:
            embedding = np.pad(embedding, (0, 384 - len(embedding)))
        else:
            embedding = embedding[:384]

        # Normalize to unit vector
        norm = np.linalg.norm(embedding)
        if norm > 1e-10:
            embedding = embedding / norm

        return embedding.astype(np.float32)

    def _categorize_entity(self, name: str, entity_type: str) -> str:
        """Categorize entity semantically"""
        name_lower = name.lower()

        # Check against known categories
        for category, keywords in self.semantic_categories.items():
            if any(kw in name_lower for kw in keywords):
                return category

        # Use entity type as fallback
        return entity_type

    def enrich_relation(self, relation: Relation) -> Relation:
        """Enrich relation with semantic type"""
        predicate_lower = relation.predicate.lower()

        # Classify relation type
        if any(
            word in predicate_lower for word in ["is", "are", "typeof", "instanceof"]
        ):
            relation.semantic_type = SemanticRelationType.HYPERNYM
        elif any(word in predicate_lower for word in ["has", "contains", "includes"]):
            relation.semantic_type = SemanticRelationType.HOLONYM
        elif any(word in predicate_lower for word in ["partof", "within", "inside"]):
            relation.semantic_type = SemanticRelationType.MERONYM
        elif any(word in predicate_lower for word in ["causes", "leads", "produces"]):
            relation.semantic_type = SemanticRelationType.CAUSE
        elif any(word in predicate_lower for word in ["results", "follows", "effects"]):
            relation.semantic_type = SemanticRelationType.EFFECT
        elif any(word in predicate_lower for word in ["like", "similar", "same"]):
            relation.semantic_type = SemanticRelationType.SYNONYM
        elif any(
            word in predicate_lower for word in ["opposite", "unlike", "different"]
        ):
            relation.semantic_type = SemanticRelationType.ANTONYM

        return relation

    def compute_semantic_similarity(self, term1: str, term2: str) -> float:
        """Compute semantic similarity between terms using embeddings"""
        emb1 = self._get_embedding(term1)
        emb2 = self._get_embedding(term2)

        # Cosine similarity
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0

        similarity = np.dot(emb1, emb2) / (norm1 * norm2)
        return float(np.clip(similarity, 0.0, 1.0))


class GoalRelevanceAnalyzer:
    """Analyzes goal relevance using NLP and semantic understanding"""

    def __init__(self, semantic_enricher: SemanticEnricher):
        self.semantic_enricher = semantic_enricher
        self.use_spacy = nlp is not None

    def analyze_goal_relevance(
        self, mappings: Dict[str, str], goal: Any, source: Dict, target: Dict
    ) -> Dict[str, float]:
        """
        Analyze which mappings are relevant to achieving the goal
        Returns: Dict mapping entity names to relevance scores (0-1)
        """
        if not goal:
            # No goal specified, all mappings equally relevant
            return {k: 1.0 for k in mappings.keys()}

        goal_text = str(goal)

        # Extract goal entities and concepts
        goal_entities = self._extract_goal_entities(goal_text)
        goal_concepts = self._extract_goal_concepts(goal_text)
        goal_embedding = self.semantic_enricher._get_embedding(goal_text)

        relevance_scores = {}

        for source_entity, target_entity in mappings.items():
            score = self._compute_entity_goal_relevance(
                source_entity,
                target_entity,
                goal_text,
                goal_entities,
                goal_concepts,
                goal_embedding,
                source,
                target,
            )
            relevance_scores[source_entity] = score

        return relevance_scores

    def _extract_goal_entities(self, goal_text: str) -> Set[str]:
        """Extract named entities and key nouns from goal"""
        entities = set()

        if self.use_spacy:
            try:
                doc = nlp(goal_text)

                # Named entities
                for ent in doc.ents:
                    entities.add(ent.text.lower())

                # Important nouns
                for token in doc:
                    if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop:
                        entities.add(token.text.lower())
            except Exception as e:
                logger.debug(f"Failed to extract entities from text: {e}")

        # Fallback: extract capitalized words and nouns heuristically
        words = goal_text.split()
        for word in words:
            if word and (word[0].isupper() or len(word) > 5):
                entities.add(word.lower())

        return entities

    def _extract_goal_concepts(self, goal_text: str) -> Set[str]:
        """Extract key concepts and actions from goal"""
        concepts = set()

        if self.use_spacy:
            try:
                doc = nlp(goal_text)

                # Verbs (actions)
                for token in doc:
                    if token.pos_ == "VERB" and not token.is_stop:
                        concepts.add(token.lemma_.lower())

                # Key adjectives
                for token in doc:
                    if token.pos_ == "ADJ" and not token.is_stop:
                        concepts.add(token.text.lower())
            except Exception as e:
                logger.debug(f"Failed to extract concepts from text: {e}")

        # Fallback: extract action words
        action_indicators = [
            "find",
            "get",
            "make",
            "create",
            "solve",
            "achieve",
            "reach",
            "obtain",
            "maximize",
            "minimize",
        ]
        goal_lower = goal_text.lower()
        for action in action_indicators:
            if action in goal_lower:
                concepts.add(action)

        return concepts

    def _compute_entity_goal_relevance(
        self,
        source_entity: str,
        target_entity: str,
        goal_text: str,
        goal_entities: Set[str],
        goal_concepts: Set[str],
        goal_embedding: np.ndarray,
        source: Dict,
        target: Dict,
    ) -> float:
        """Compute how relevant an entity mapping is to the goal"""
        scores = []

        # 1. Direct mention in goal
        source_lower = source_entity.lower()
        target_lower = target_entity.lower()
        goal_lower = goal_text.lower()

        direct_mention_score = 0.0
        if source_lower in goal_lower or target_lower in goal_lower:
            direct_mention_score = 1.0
        scores.append(direct_mention_score)

        # 2. Entity overlap with goal entities
        entity_overlap_score = 0.0
        if source_lower in goal_entities or target_lower in goal_entities:
            entity_overlap_score = 0.8
        scores.append(entity_overlap_score)

        # 3. Semantic similarity to goal
        source_emb = self.semantic_enricher._get_embedding(source_entity)
        target_emb = self.semantic_enricher._get_embedding(target_entity)

        source_goal_sim = self._embedding_similarity(source_emb, goal_embedding)
        target_goal_sim = self._embedding_similarity(target_emb, goal_embedding)
        semantic_score = max(source_goal_sim, target_goal_sim)
        scores.append(semantic_score)

        # 4. Causal relevance (if entity appears in relations leading to goal)
        causal_score = self._compute_causal_relevance(
            source_entity, target_entity, goal_concepts, source, target
        )
        scores.append(causal_score)

        # 5. Dependency analysis if spaCy available
        if self.use_spacy:
            dependency_score = self._compute_dependency_relevance(
                source_entity, target_entity, goal_text
            )
            scores.append(dependency_score)

        # Weighted average
        weights = (
            [0.3, 0.2, 0.25, 0.15, 0.1]
            if len(scores) == 5
            else [0.35, 0.25, 0.25, 0.15]
        )
        weights = weights[: len(scores)]

        return sum(w * s for w, s in zip(weights, scores))

    def _embedding_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity"""
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0

        return float(np.clip(np.dot(emb1, emb2) / (norm1 * norm2), 0.0, 1.0))

    def _compute_causal_relevance(
        self,
        source_entity: str,
        target_entity: str,
        goal_concepts: Set[str],
        source: Dict,
        target: Dict,
    ) -> float:
        """Check if entity is causally related to goal concepts"""
        # Check if entity participates in relations with goal-relevant predicates
        source_rels = source.get("relations", [])

        relevant_count = 0
        total_count = 0

        for rel in source_rels:
            if isinstance(rel, Relation):
                if source_entity in rel.arguments:
                    total_count += 1
                    pred_lower = rel.predicate.lower()
                    if any(concept in pred_lower for concept in goal_concepts):
                        relevant_count += 1
            elif isinstance(rel, tuple) and len(rel) >= 3:
                if source_entity in [rel[0], rel[2]]:
                    total_count += 1
                    pred_lower = str(rel[1]).lower()
                    if any(concept in pred_lower for concept in goal_concepts):
                        relevant_count += 1

        return relevant_count / max(total_count, 1) if total_count > 0 else 0.5

    def _compute_dependency_relevance(
        self, source_entity: str, target_entity: str, goal_text: str
    ) -> float:
        """Use dependency parsing to determine relevance"""
        try:
            doc = nlp(goal_text)

            # Find if entities play important syntactic roles
            source_lower = source_entity.lower()
            target_lower = target_entity.lower()

            important_roles = {"nsubj", "dobj", "pobj", "ROOT"}

            for token in doc:
                if (
                    source_lower in token.text.lower()
                    or target_lower in token.text.lower()
                ):
                    if token.dep_ in important_roles:
                        return 0.8
                    else:
                        return 0.5

            return 0.3
        except Exception:
            return 0.5

    def filter_by_relevance(
        self,
        mappings: Dict[str, str],
        relevance_scores: Dict[str, float],
        threshold: float = 0.5,
    ) -> Dict[str, str]:
        """Filter mappings by relevance threshold"""
        return {
            entity: target
            for entity, target in mappings.items()
            if relevance_scores.get(entity, 0.0) >= threshold
        }


class AbstractReasoner:
    """Base class for abstract reasoning"""

    def __init__(self):
        self.abstractions = {}

    def abstract(self, concept: Any) -> Any:
        """Create abstraction of concept"""
        return concept

    def concretize(self, abstraction: Any) -> Any:
        """Make abstraction concrete"""
        return abstraction


class AnalogicalReasoner(AbstractReasoner):
    """Enhanced analogical reasoning with advanced semantic understanding"""

    def __init__(self, enable_caching: bool = True, enable_learning: bool = True):
        super().__init__()

        # Semantic enrichment
        self.semantic_enricher = SemanticEnricher()
        self.goal_analyzer = GoalRelevanceAnalyzer(self.semantic_enricher)

        # Domain knowledge storage
        self.domain_knowledge = {}
        self.analogy_cache = {}
        self.mapping_cache = {}
        self.successful_mappings = deque(maxlen=1000)

        # Cache size limits
        self.max_cache_size = 1000
        self.max_analogy_cache_size = 500

        # Similarity thresholds
        self.similarity_threshold = 0.7
        self.structural_weight = 0.6
        self.surface_weight = 0.2
        self.semantic_weight = 0.2

        # Graph representations
        self.domain_graphs = {}

        # Learning components
        self.enable_learning = enable_learning
        if enable_learning:
            self.mapping_patterns = defaultdict(int)
            self.domain_similarities = defaultdict(float)
            self.learned_weights = {"structural": 0.6, "surface": 0.2, "semantic": 0.2}

        # Performance tracking
        self.stats = {
            "total_mappings": 0,
            "successful_mappings": 0,
            "cache_hits": 0,
            "average_mapping_score": 0.0,
            "semantic_enrichments": 0,
            "embedding_method": (
                "sentence-transformers" if SENTENCE_TRANSFORMERS_AVAILABLE else "tfidf"
            ),
        }

        # Caching
        self.enable_caching = enable_caching

        # Parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Persistence
        self.model_path = Path("analogical_models")
        self.model_path.mkdir(parents=True, exist_ok=True)

    def add_domain(self, domain_name: str, structure: Dict[str, Any]):
        """Add domain knowledge with semantic enrichment"""

        # Parse structure
        entities = self._extract_entities(structure)
        relations = self._extract_relations(structure)
        attributes = self._extract_attributes(structure)

        # Enrich entities semantically
        enriched_entities = set()
        for entity in entities:
            if isinstance(entity, Entity):
                enriched = self.semantic_enricher.enrich_entity(entity)
                enriched_entities.add(enriched)
                self.stats["semantic_enrichments"] += 1
            else:
                enriched_entities.add(entity)

        # Enrich relations
        enriched_relations = []
        for relation in relations:
            if isinstance(relation, Relation):
                enriched = self.semantic_enricher.enrich_relation(relation)
                enriched_relations.append(enriched)
            else:
                enriched_relations.append(relation)

        # Create domain representation
        self.domain_knowledge[domain_name] = {
            "structure": structure,
            "entities": enriched_entities,
            "relations": enriched_relations,
            "attributes": attributes,
            "graph": self._build_domain_graph(enriched_entities, enriched_relations),
            "abstraction_level": self._compute_abstraction_level(structure),
        }

        # Build NetworkX graph if available
        if NETWORKX_AVAILABLE:
            self.domain_graphs[domain_name] = self._create_networkx_graph(
                enriched_entities, enriched_relations
            )

        logger.info(
            f"Added domain {domain_name} with {len(enriched_entities)} entities "
            f"and {len(enriched_relations)} relations"
        )

    def compute_similarity(self, source: Dict, target: Dict) -> float:
        """Compute sophisticated structural similarity"""

        if not source and not target:
            return 1.0

        if not source or not target:
            return 0.0

        # Key overlap
        source_attrs = set(source.keys())
        target_attrs = set(target.keys())

        if not source_attrs and not target_attrs:
            return 1.0
        if not source_attrs or not target_attrs:
            return 0.0

        intersection = source_attrs & target_attrs
        union = source_attrs | target_attrs

        if len(union) == 0:
            return 1.0

        jaccard = len(intersection) / len(union)

        # Value similarity for common attributes
        value_sim = 0.0
        if intersection:
            for attr in intersection:
                try:
                    if source[attr] == target[attr]:
                        value_sim += 1.0
                    elif isinstance(source[attr], (int, float)) and isinstance(
                        target[attr], (int, float)
                    ):
                        diff = abs(source[attr] - target[attr])
                        max_val = max(abs(source[attr]), abs(target[attr]), 1e-10)
                        value_sim += 1.0 - min(1.0, diff / max_val)
                    elif isinstance(source[attr], str) and isinstance(
                        target[attr], str
                    ):
                        # Use semantic similarity
                        sem_sim = self.semantic_enricher.compute_semantic_similarity(
                            source[attr], target[attr]
                        )
                        value_sim += sem_sim
                except Exception as e:
                    logger.warning(f"Error computing value similarity: {e}")

            value_sim /= max(len(intersection), 1)

        return (jaccard + value_sim) / 2

    def update_cache(self, source: Any, target: Any, mapping: Dict, confidence: float):
        """Update mapping cache with LRU eviction"""

        try:
            cache_key = f"{hash(str(source))}_{hash(str(target))}"
        except Exception:
            cache_key = f"{id(source)}_{id(target)}"

        self.mapping_cache[cache_key] = {
            "mapping": mapping,
            "confidence": confidence,
            "timestamp": time.time(),
        }

        # LRU eviction
        if len(self.mapping_cache) > self.max_cache_size:
            sorted_items = sorted(
                self.mapping_cache.items(), key=lambda x: x[1].get("timestamp", 0)
            )
            self.mapping_cache = dict(sorted_items[-self.max_cache_size :])

    def find_structural_analogy(
        self,
        source_domain: str,
        target_problem: Dict,
        mapping_type: MappingType = MappingType.STRUCTURAL,
    ) -> Dict[str, Any]:
        """Find analogies using advanced semantic understanding"""

        start_time = time.time()
        self.stats["total_mappings"] += 1

        # Check cache - FIX #2: Use deterministic cache key generation
        try:
            # Create deterministic hash from target_problem structure
            cache_str = json.dumps(
                {
                    "entities": sorted(
                        [
                            e.name if isinstance(e, Entity) else str(e)
                            for e in self._extract_entities(target_problem)
                        ]
                    ),
                    "relations": sorted(
                        [str(r) for r in self._extract_relations(target_problem)]
                    ),
                    "attributes": sorted(
                        [
                            (k, sorted(v) if isinstance(v, list) else v)
                            for k, v in self._extract_attributes(target_problem).items()
                        ]
                    ),
                },
                sort_keys=True,
            )
            cache_key = f"{source_domain}_{hashlib.md5(cache_str.encode(), usedforsecurity=False).hexdigest()}"
        except Exception:
            # Fallback to id-based key if json serialization fails
            cache_key = f"{source_domain}_{id(target_problem)}"

        if self.enable_caching and cache_key in self.analogy_cache:
            self.stats["cache_hits"] += 1
            return self.analogy_cache[cache_key]

        if source_domain not in self.domain_knowledge:
            return {
                "found": False,
                "reason": "Unknown source domain",
                "confidence": 0.0,
            }

        source = self.domain_knowledge[source_domain]

        # Extract and enrich target structure
        target_entities = self._extract_entities(target_problem)
        enriched_target_entities = set()
        for entity in target_entities:
            if isinstance(entity, Entity):
                enriched = self.semantic_enricher.enrich_entity(entity)
                enriched_target_entities.add(enriched)
            else:
                enriched_target_entities.add(entity)

        target_structure = {
            "entities": enriched_target_entities,
            "relations": self._extract_relations(target_problem),
            "attributes": self._extract_attributes(target_problem),
        }

        # Perform structure mapping
        try:
            if mapping_type == MappingType.STRUCTURAL:
                mapping = self._structural_mapping(source, target_structure)
            elif mapping_type == MappingType.SURFACE:
                mapping = self._surface_mapping(source, target_structure)
            elif mapping_type == MappingType.PRAGMATIC:
                mapping = self._pragmatic_mapping(
                    source, target_structure, target_problem
                )
            else:  # SEMANTIC
                mapping = self._semantic_mapping(source, target_structure)
        except Exception as e:
            logger.error(f"Mapping failed: {e}")
            return {"found": False, "reason": f"Mapping error: {e}", "confidence": 0.0}

        query_time = time.time() - start_time

        # Check threshold
        if mapping.mapping_score >= self.similarity_threshold:
            solution = self._map_solution(
                source.get("structure", {}).get("solution"), mapping.entity_mappings
            )

            result = {
                "found": True,
                "source_domain": source_domain,
                "mapping": mapping,
                "mappings": mapping.entity_mappings,
                "solution": solution,
                "confidence": mapping.confidence,
                "score": mapping.mapping_score,
                "explanation": self._generate_explanation(mapping),
                "semantic_coherence": mapping.semantic_coherence,
                "_query_time": query_time,
            }

            if self.enable_learning:
                self._learn_from_mapping(source_domain, mapping)

            self.stats["successful_mappings"] += 1
            self.successful_mappings.append(mapping)

            self.update_cache(
                source_domain,
                target_problem,
                mapping.entity_mappings,
                mapping.confidence,
            )
        else:
            result = {
                "found": False,
                "reason": f"Similarity score {mapping.mapping_score:.2f} below threshold",
                "confidence": mapping.confidence,
                "partial_mapping": mapping.entity_mappings,
                "mapping": mapping,
                "mappings": mapping.entity_mappings,
                "score": mapping.mapping_score,
                "_query_time": query_time,
            }

        self._update_statistics(mapping.mapping_score)

        # Cache with size limit
        if self.enable_caching:
            while len(self.analogy_cache) >= self.max_analogy_cache_size:
                try:
                    first_key = next(iter(self.analogy_cache))
                    del self.analogy_cache[first_key]
                except StopIteration:
                    break

            self.analogy_cache[cache_key] = result

            while len(self.analogy_cache) > self.max_analogy_cache_size:
                first_key = next(iter(self.analogy_cache))
                del self.analogy_cache[first_key]

        return result

    def _structural_mapping(self, source: Dict, target: Dict) -> AnalogicalMapping:
        """Advanced structural mapping with semantic awareness"""

        # Find entity candidates using semantic similarity
        entity_candidates = self._find_entity_candidates_semantic(
            source["entities"], target["entities"]
        )

        # Find identical predicates
        initial_mappings = self._find_identical_predicates(
            source["relations"], target["relations"]
        )

        # Extend mappings structurally
        best_mapping = self._extend_mappings_structurally(
            initial_mappings,
            source["relations"],
            target["relations"],
            entity_candidates,
        )

        # Evaluate consistency
        consistency_score = self._evaluate_structural_consistency(
            best_mapping, source, target
        )

        # Systematicity
        systematicity_score = self._compute_systematicity(
            best_mapping["relation_mappings"]
        )

        # Semantic coherence
        semantic_coherence = self._compute_semantic_coherence(
            best_mapping, source, target
        )

        # Combined score
        mapping_score = (
            self.learned_weights["structural"] * consistency_score
            + 0.2 * systematicity_score
            + 0.2 * semantic_coherence
        )

        return AnalogicalMapping(
            source_domain="source",
            target_domain="target",
            entity_mappings=best_mapping["entity_mappings"],
            relation_mappings=best_mapping["relation_mappings"],
            mapping_score=mapping_score,
            mapping_type=MappingType.STRUCTURAL,
            confidence=consistency_score,
            semantic_coherence=semantic_coherence,
            structural_depth=len(best_mapping["relation_mappings"]),
        )

    def _find_entity_candidates_semantic(
        self, source_entities: Set[Entity], target_entities: Set[Entity]
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Find entity candidates using advanced semantic similarity"""

        candidates = defaultdict(list)

        for s_entity in source_entities:
            for t_entity in target_entities:
                try:
                    if isinstance(s_entity, Entity) and isinstance(t_entity, Entity):
                        # Use enhanced similarity with embeddings
                        similarity = s_entity.similarity_to(
                            t_entity, use_embeddings=True
                        )
                        if similarity > 0.3:  # Lower threshold for candidates
                            candidates[s_entity.name].append(
                                (t_entity.name, similarity)
                            )
                    else:
                        s_name = str(s_entity)
                        t_name = str(t_entity)
                        similarity = self.semantic_enricher.compute_semantic_similarity(
                            s_name, t_name
                        )
                        if similarity > 0.3:
                            candidates[s_name].append((t_name, similarity))
                except Exception as e:
                    logger.warning(f"Entity candidate finding failed: {e}")
                    continue

        # Sort by similarity
        for entity in candidates:
            candidates[entity].sort(key=lambda x: x[1], reverse=True)

        return dict(candidates)

    def _compute_semantic_coherence(
        self, mapping: Dict, source: Dict, target: Dict
    ) -> float:
        """Compute semantic coherence of the mapping"""

        if not mapping["entity_mappings"]:
            return 0.0

        coherence_scores = []

        # Check if mapped entities have similar semantic categories
        for s_name, t_name in mapping["entity_mappings"].items():
            # Find corresponding entities
            s_entity = next(
                (
                    e
                    for e in source["entities"]
                    if isinstance(e, Entity) and e.name == s_name
                ),
                None,
            )
            t_entity = next(
                (
                    e
                    for e in target["entities"]
                    if isinstance(e, Entity) and e.name == t_name
                ),
                None,
            )

            if (
                s_entity
                and t_entity
                and isinstance(s_entity, Entity)
                and isinstance(t_entity, Entity)
            ):
                if s_entity.semantic_category == t_entity.semantic_category:
                    coherence_scores.append(1.0)
                elif s_entity.semantic_category and t_entity.semantic_category:
                    # Compute category similarity
                    cat_sim = self.semantic_enricher.compute_semantic_similarity(
                        s_entity.semantic_category, t_entity.semantic_category
                    )
                    coherence_scores.append(cat_sim)
                else:
                    coherence_scores.append(0.5)

        # Check relation semantic coherence
        for rel_map in mapping["relation_mappings"]:
            if "source_rel" in rel_map and "target_rel" in rel_map:
                s_rel = rel_map["source_rel"]
                t_rel = rel_map["target_rel"]

                if isinstance(s_rel, Relation) and isinstance(t_rel, Relation):
                    rel_sim = s_rel.semantic_similarity_to(t_rel)
                    coherence_scores.append(rel_sim)

        return np.mean(coherence_scores) if coherence_scores else 0.0

    def _surface_mapping(self, source: Dict, target: Dict) -> AnalogicalMapping:
        """Surface mapping enhanced with semantic similarity"""

        entity_mappings = {}

        for s_entity in source["entities"]:
            best_match = None
            best_score = 0

            for t_entity in target["entities"]:
                try:
                    if isinstance(s_entity, Entity) and isinstance(t_entity, Entity):
                        score = s_entity.similarity_to(t_entity, use_embeddings=True)
                    else:
                        score = self.semantic_enricher.compute_semantic_similarity(
                            str(s_entity), str(t_entity)
                        )

                    if score > best_score:
                        best_score = score
                        best_match = t_entity
                except Exception as e:
                    logger.warning(f"Surface matching failed: {e}")
                    continue

            if best_match and best_score > 0.5:
                s_name = (
                    s_entity.name if isinstance(s_entity, Entity) else str(s_entity)
                )
                t_name = (
                    best_match.name
                    if isinstance(best_match, Entity)
                    else str(best_match)
                )
                entity_mappings[s_name] = t_name

        num_entities = max(len(source["entities"]), 1)
        mapping_score = len(entity_mappings) / num_entities

        return AnalogicalMapping(
            source_domain="source",
            target_domain="target",
            entity_mappings=entity_mappings,
            relation_mappings=[],
            mapping_score=mapping_score,
            mapping_type=MappingType.SURFACE,
            confidence=mapping_score,
        )

    def _pragmatic_mapping(
        self, source: Dict, target: Dict, target_problem: Dict
    ) -> AnalogicalMapping:
        """Goal-directed pragmatic mapping with advanced relevance analysis"""

        # Get base structural mapping
        structural_mapping = self._structural_mapping(source, target)

        # Extract goal
        goal = target_problem.get("goal", target_problem.get("objective"))

        if goal:
            # Analyze goal relevance
            relevance_scores = self.goal_analyzer.analyze_goal_relevance(
                structural_mapping.entity_mappings, goal, source, target
            )

            # Filter by relevance
            relevant_mappings = self.goal_analyzer.filter_by_relevance(
                structural_mapping.entity_mappings, relevance_scores, threshold=0.4
            )

            # Update mapping
            structural_mapping.entity_mappings = relevant_mappings

            # Adjust score based on relevance
            avg_relevance = (
                np.mean(list(relevance_scores.values())) if relevance_scores else 0.5
            )
            structural_mapping.mapping_score = (
                0.7 * structural_mapping.mapping_score + 0.3 * avg_relevance
            )

        structural_mapping.mapping_type = MappingType.PRAGMATIC
        return structural_mapping

    def _semantic_mapping(self, source: Dict, target: Dict) -> AnalogicalMapping:
        """Pure semantic similarity-based mapping"""

        entity_mappings = {}

        for s_entity in source["entities"]:
            best_match = None
            best_score = 0

            for t_entity in target["entities"]:
                try:
                    s_name = (
                        s_entity.name if isinstance(s_entity, Entity) else str(s_entity)
                    )
                    t_name = (
                        t_entity.name if isinstance(t_entity, Entity) else str(t_entity)
                    )

                    # Pure semantic similarity
                    score = self.semantic_enricher.compute_semantic_similarity(
                        s_name, t_name
                    )

                    if score > best_score:
                        best_score = score
                        best_match = t_entity
                except Exception as e:
                    logger.warning(f"Semantic matching failed: {e}")
                    continue

            if best_match and best_score > 0.5:
                s_name = (
                    s_entity.name if isinstance(s_entity, Entity) else str(s_entity)
                )
                t_name = (
                    best_match.name
                    if isinstance(best_match, Entity)
                    else str(best_match)
                )
                entity_mappings[s_name] = t_name

        num_entities = max(len(source["entities"]), 1)
        mapping_score = len(entity_mappings) / num_entities

        return AnalogicalMapping(
            source_domain="source",
            target_domain="target",
            entity_mappings=entity_mappings,
            relation_mappings=[],
            mapping_score=mapping_score,
            mapping_type=MappingType.SEMANTIC,
            confidence=0.7,
        )

    def _find_identical_predicates(
        self, source_rels: List, target_rels: List
    ) -> List[Dict]:
        """Find relations with identical or similar predicates"""
        mappings = []

        for s_rel in source_rels:
            for t_rel in target_rels:
                try:
                    if isinstance(s_rel, Relation) and isinstance(t_rel, Relation):
                        # Exact match
                        if s_rel.predicate == t_rel.predicate:
                            mappings.append(
                                {
                                    "source_rel": s_rel,
                                    "target_rel": t_rel,
                                    "entity_constraints": list(
                                        zip(s_rel.arguments, t_rel.arguments)
                                    ),
                                }
                            )
                        # Semantic similarity match
                        elif (
                            s_rel.semantic_type
                            and s_rel.semantic_type == t_rel.semantic_type
                        ):
                            mappings.append(
                                {
                                    "source_rel": s_rel,
                                    "target_rel": t_rel,
                                    "entity_constraints": list(
                                        zip(s_rel.arguments, t_rel.arguments)
                                    ),
                                }
                            )
                    elif isinstance(s_rel, tuple) and isinstance(t_rel, tuple):
                        if len(s_rel) >= 3 and len(t_rel) >= 3 and s_rel[1] == t_rel[1]:
                            mappings.append(
                                {
                                    "source_rel": s_rel,
                                    "target_rel": t_rel,
                                    "entity_constraints": [
                                        (s_rel[0], t_rel[0]),
                                        (s_rel[2], t_rel[2]),
                                    ],
                                }
                            )
                except Exception as e:
                    logger.warning(f"Predicate matching failed: {e}")
                    continue

        return mappings

    def _extend_mappings_structurally(
        self,
        initial_mappings: List[Dict],
        source_rels: List,
        target_rels: List,
        entity_candidates: Dict,
    ) -> Dict:
        """Extend mappings using structural consistency"""

        best_mapping = {"entity_mappings": {}, "relation_mappings": [], "score": 0}

        for initial in initial_mappings:
            try:
                mapping = {"entity_mappings": {}, "relation_mappings": [initial]}

                if "entity_constraints" in initial:
                    for s_entity, t_entity in initial["entity_constraints"]:
                        mapping["entity_mappings"][s_entity] = t_entity

                extended = True
                max_iterations = 100
                iterations = 0

                while extended and iterations < max_iterations:
                    iterations += 1
                    extended = False

                    for s_rel in source_rels:
                        for t_rel in target_rels:
                            if self._can_add_consistently(s_rel, t_rel, mapping):
                                mapping["relation_mappings"].append(
                                    {"source_rel": s_rel, "target_rel": t_rel}
                                )
                                extended = True
                                break

                num_rels = max(len(source_rels), 1)
                score = len(mapping["relation_mappings"]) / num_rels

                if score > best_mapping["score"]:
                    best_mapping = mapping
                    best_mapping["score"] = score
            except Exception as e:
                logger.warning(f"Mapping extension failed: {e}")
                continue

        return best_mapping

    def _can_add_consistently(self, s_rel, t_rel, mapping: Dict) -> bool:
        """Check consistency"""
        try:
            if hasattr(s_rel, "predicate") and hasattr(t_rel, "predicate"):
                if s_rel.predicate != t_rel.predicate:
                    return False

                for s_arg, t_arg in zip(s_rel.arguments, t_rel.arguments):
                    if s_arg in mapping["entity_mappings"]:
                        if mapping["entity_mappings"][s_arg] != t_arg:
                            return False
        except Exception as e:
            logger.warning(f"Consistency check failed: {e}")
            return False

        return True

    def _evaluate_structural_consistency(
        self, mapping: Dict, source: Dict, target: Dict
    ) -> float:
        """Evaluate consistency"""
        if not mapping["relation_mappings"]:
            return 0.0

        consistent_count = 0
        total_count = 0

        for rel_map in mapping["relation_mappings"]:
            total_count += 1

            try:
                consistent = True
                if "entity_constraints" in rel_map:
                    for s_entity, t_entity in rel_map["entity_constraints"]:
                        if s_entity in mapping["entity_mappings"]:
                            if mapping["entity_mappings"][s_entity] != t_entity:
                                consistent = False
                                break

                if consistent:
                    consistent_count += 1
            except Exception as e:
                logger.warning(f"Consistency evaluation failed: {e}")
                continue

        return consistent_count / max(total_count, 1)

    def _compute_systematicity(self, relation_mappings: List) -> float:
        """Compute systematicity"""
        if not relation_mappings:
            return 0.0

        try:
            higher_order_count = 0
            for rel_map in relation_mappings:
                if hasattr(rel_map.get("source_rel"), "order"):
                    if rel_map["source_rel"].order > 1:
                        higher_order_count += 1

            return higher_order_count / max(len(relation_mappings), 1)
        except Exception:
            return 0.0

    def _extract_entities(self, structure: Dict) -> Set:
        """Extract entities with semantic enrichment"""
        entities = set()

        try:
            if "entities" in structure:
                for entity in structure["entities"]:
                    if isinstance(entity, Entity):
                        entities.add(entity)
                    elif isinstance(entity, dict):
                        entities.add(
                            Entity(
                                name=entity.get("name", str(entity)),
                                attributes=entity.get("attributes", {}),
                                entity_type=entity.get("type", "object"),
                            )
                        )
                    else:
                        entities.add(Entity(name=str(entity), entity_type="object"))

            if "description" in structure:
                words = structure["description"].split()
                for w in words:
                    if w and w[0].isupper():
                        entities.add(Entity(name=w, entity_type="object"))
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")

        return entities

    def _extract_relations(self, structure: Dict) -> List:
        """Extract relations"""
        relations = []

        try:
            if "relations" in structure:
                for rel in structure["relations"]:
                    if isinstance(rel, Relation):
                        relations.append(rel)
                    elif isinstance(rel, tuple) and len(rel) >= 3:
                        relations.append(
                            Relation(
                                predicate=rel[1],
                                arguments=[rel[0], rel[2]],
                                relation_type="binary",
                            )
                        )
                    elif isinstance(rel, dict):
                        relations.append(
                            Relation(
                                predicate=rel.get("predicate", ""),
                                arguments=[
                                    rel.get("subject", ""),
                                    rel.get("object", ""),
                                ],
                                relation_type=rel.get("type", "binary"),
                            )
                        )
        except Exception as e:
            logger.warning(f"Relation extraction failed: {e}")

        return relations

    def _extract_attributes(self, structure: Dict) -> Dict[str, List]:
        """Extract attributes"""
        attributes = defaultdict(list)

        try:
            if "attributes" in structure:
                for entity, attrs in structure["attributes"].items():
                    if isinstance(attrs, list):
                        attributes[entity].extend(attrs)
                    else:
                        attributes[entity].append(str(attrs))
        except Exception as e:
            logger.warning(f"Attribute extraction failed: {e}")

        return dict(attributes)

    def _build_domain_graph(self, entities: Set, relations: List) -> Dict:
        """Build graph"""
        graph = {
            "nodes": [e.name if isinstance(e, Entity) else str(e) for e in entities],
            "edges": [],
        }

        try:
            for rel in relations:
                if isinstance(rel, Relation):
                    graph["edges"].append(
                        {
                            "from": rel.arguments[0] if rel.arguments else None,
                            "to": rel.arguments[1] if len(rel.arguments) > 1 else None,
                            "type": rel.predicate,
                        }
                    )
        except Exception as e:
            logger.warning(f"Graph building failed: {e}")

        return graph

    def _create_networkx_graph(self, entities: Set, relations: List):
        """Create NetworkX graph"""
        if not NETWORKX_AVAILABLE:
            return None

        try:
            G = nx.DiGraph()

            for entity in entities:
                if isinstance(entity, Entity):
                    G.add_node(entity.name, **entity.attributes)
                else:
                    G.add_node(str(entity))

            for rel in relations:
                if isinstance(rel, Relation) and len(rel.arguments) >= 2:
                    G.add_edge(
                        rel.arguments[0],
                        rel.arguments[1],
                        predicate=rel.predicate,
                        type=rel.relation_type,
                    )

            return G
        except Exception as e:
            logger.error(f"NetworkX graph creation failed: {e}")
            return None

    def _compute_abstraction_level(self, structure: Dict) -> int:
        """Compute abstraction level"""
        try:
            abstract_indicators = ["concept", "idea", "principle", "theory", "abstract"]

            level = 0
            desc = str(structure.get("description", "")).lower()

            for indicator in abstract_indicators:
                if indicator in desc:
                    level += 1

            return min(level, 5)
        except Exception:
            return 0

    def _map_solution(self, source_solution: Any, mappings: Dict[str, str]) -> Any:
        """Map solution"""
        if source_solution is None:
            return None

        try:
            if isinstance(source_solution, str):
                result = source_solution
                for source, target in mappings.items():
                    result = result.replace(source, target)
                return result

            elif isinstance(source_solution, dict):
                result = {}
                for key, value in source_solution.items():
                    mapped_key = mappings.get(key, key)
                    result[mapped_key] = self._map_solution(value, mappings)
                return result

            elif isinstance(source_solution, list):
                return [self._map_solution(item, mappings) for item in source_solution]

            return source_solution
        except Exception as e:
            logger.error(f"Solution mapping failed: {e}")
            return source_solution

    def _generate_explanation(self, mapping: AnalogicalMapping) -> str:
        """Generate detailed explanation"""
        try:
            explanation = f"Analogical mapping ({mapping.mapping_type.value}):\n"
            explanation += f"Similarity score: {mapping.mapping_score:.2f}\n"
            explanation += f"Confidence: {mapping.confidence:.2f}\n"
            explanation += f"Semantic coherence: {mapping.semantic_coherence:.2f}\n"
            explanation += f"Embedding method: {self.stats['embedding_method']}\n"

            if mapping.entity_mappings:
                explanation += "\nEntity mappings:\n"
                for source, target in list(mapping.entity_mappings.items())[:5]:
                    explanation += f"  {source} → {target}\n"

            if mapping.relation_mappings:
                explanation += f"\n{len(mapping.relation_mappings)} structural relations preserved\n"

            return explanation
        except Exception:
            return "Analogical mapping completed"

    def _learn_from_mapping(self, source_domain: str, mapping: AnalogicalMapping):
        """Learn from mapping"""
        if not self.enable_learning:
            return

        try:
            pattern_key = f"{source_domain}_{mapping.mapping_type.value}"
            self.mapping_patterns[pattern_key] += 1

            self.domain_similarities[source_domain] = (
                0.9 * self.domain_similarities.get(source_domain, 0)
                + 0.1 * mapping.mapping_score
            )

            if mapping.mapping_score > self.similarity_threshold:
                if mapping.mapping_type == MappingType.STRUCTURAL:
                    self.learned_weights["structural"] = min(
                        0.8, self.learned_weights["structural"] * 1.01
                    )
                elif mapping.mapping_type == MappingType.SURFACE:
                    self.learned_weights["surface"] = min(
                        0.4, self.learned_weights["surface"] * 1.01
                    )

                total = sum(self.learned_weights.values())
                if total > 0:
                    for key in self.learned_weights:
                        self.learned_weights[key] /= total
        except Exception as e:
            logger.warning(f"Learning from mapping failed: {e}")

    def _update_statistics(self, score: float):
        """Update statistics"""
        try:
            n = self.stats["total_mappings"]
            if n > 0:
                old_avg = self.stats["average_mapping_score"]
                self.stats["average_mapping_score"] = (old_avg * (n - 1) + score) / n
        except Exception as e:
            logger.warning(f"Statistics update failed: {e}")

    def find_multiple_analogies(
        self, target_problem: Dict, k: int = 5
    ) -> List[Dict[str, Any]]:
        """Find top-k analogies from all domains"""
        candidates = []

        futures = []
        for domain_name in self.domain_knowledge:
            future = self.executor.submit(
                self.find_structural_analogy, domain_name, target_problem
            )
            futures.append((domain_name, future))

        for domain_name, future in futures:
            try:
                result = future.result(timeout=2.0)
                if result.get("found"):
                    candidates.append(result)
            except TimeoutError:
                logger.warning(f"Analogy search timed out for {domain_name}")
            except Exception as e:
                logger.warning(f"Failed to find analogy with {domain_name}: {e}")

        candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
        return candidates[:k]

    def cross_domain_transfer(
        self, source_domain: str, target_domain: str, concept: str
    ) -> Dict[str, Any]:
        """Transfer concept between domains"""
        if source_domain not in self.domain_knowledge:
            return {"success": False, "reason": "Unknown source domain"}

        if target_domain not in self.domain_knowledge:
            return {"success": False, "reason": "Unknown target domain"}

        try:
            source = self.domain_knowledge[source_domain]
            target_structure = {
                "entities": self.domain_knowledge[target_domain]["entities"],
                "relations": self.domain_knowledge[target_domain]["relations"],
                "attributes": self.domain_knowledge[target_domain]["attributes"],
            }

            mapping = self._structural_mapping(source, target_structure)

            if concept in mapping.entity_mappings:
                transferred = mapping.entity_mappings[concept]

                return {
                    "success": True,
                    "source_concept": concept,
                    "target_concept": transferred,
                    "confidence": mapping.confidence,
                    "mapping": mapping,
                }

            return {
                "success": False,
                "reason": "Concept not found in mapping",
                "partial_mapping": mapping.entity_mappings,
            }
        except Exception as e:
            logger.error(f"Cross-domain transfer failed: {e}")
            return {"success": False, "reason": f"Transfer error: {e}"}

    def save_model(self, name: str = "default"):
        """Save model with semantic enrichment data"""
        model_file = self.model_path / f"{name}_analogical_model.pkl"

        model_data = {
            "domain_knowledge": self.domain_knowledge,
            "similarity_threshold": self.similarity_threshold,
            "learned_weights": self.learned_weights if self.enable_learning else None,
            "mapping_patterns": (
                dict(self.mapping_patterns) if self.enable_learning else None
            ),
            "domain_similarities": (
                dict(self.domain_similarities) if self.enable_learning else None
            ),
            "stats": self.stats,
            "max_cache_size": self.max_cache_size,
            "max_analogy_cache_size": self.max_analogy_cache_size,
            "semantic_enricher_cache": self.semantic_enricher.embedding_cache,
            "tfidf_corpus": self.semantic_enricher._tfidf_corpus,
            "tfidf_fitted": self.semantic_enricher._tfidf_fitted,
        }

        try:
            with open(model_file, "wb") as f:
                pickle.dump(model_data, f)
            logger.info(f"Analogical model saved to {model_file}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    def load_model(self, name: str = "default"):
        """Load model with semantic enrichment data"""
        model_file = self.model_path / f"{name}_analogical_model.pkl"

        if not model_file.exists():
            raise FileNotFoundError(f"Model file {model_file} not found")

        try:
            with open(model_file, "rb") as f:
                model_data = pickle.load(f)  # nosec B301 - Internal data structure

            self.domain_knowledge = model_data["domain_knowledge"]
            self.similarity_threshold = model_data["similarity_threshold"]
            self.stats = model_data["stats"]

            self.max_cache_size = model_data.get("max_cache_size", 1000)
            self.max_analogy_cache_size = model_data.get("max_analogy_cache_size", 500)

            if self.enable_learning and model_data.get("learned_weights"):
                self.learned_weights = model_data["learned_weights"]
                self.mapping_patterns = defaultdict(int, model_data["mapping_patterns"])
                self.domain_similarities = defaultdict(
                    float, model_data["domain_similarities"]
                )

            if "semantic_enricher_cache" in model_data:
                self.semantic_enricher.embedding_cache = model_data[
                    "semantic_enricher_cache"
                ]

            if "tfidf_corpus" in model_data:
                self.semantic_enricher._tfidf_corpus = model_data["tfidf_corpus"]
                self.semantic_enricher._tfidf_fitted = model_data.get(
                    "tfidf_fitted", False
                )

                # Refit vectorizer if available and if it was fitted before
                if (
                    self.semantic_enricher._tfidf_vectorizer
                    and len(self.semantic_enricher._tfidf_corpus) > 1
                    and self.semantic_enricher._tfidf_fitted
                ):
                    try:
                        self.semantic_enricher._tfidf_vectorizer.fit(
                            self.semantic_enricher._tfidf_corpus
                        )
                    except Exception as e:
                        logger.debug(f"Operation failed: {e}")

            if NETWORKX_AVAILABLE:
                for domain_name, domain_data in self.domain_knowledge.items():
                    self.domain_graphs[domain_name] = self._create_networkx_graph(
                        domain_data["entities"], domain_data["relations"]
                    )

            logger.info(f"Analogical model loaded from {model_file}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        stats = self.stats.copy()

        stats["num_domains"] = len(self.domain_knowledge)
        stats["cache_size"] = len(self.analogy_cache)
        stats["mapping_cache_size"] = len(self.mapping_cache)
        stats["successful_mappings_history"] = len(self.successful_mappings)
        stats["max_cache_size"] = self.max_cache_size
        stats["max_analogy_cache_size"] = self.max_analogy_cache_size
        stats["embedding_cache_size"] = len(self.semantic_enricher.embedding_cache)
        stats["entity_cache_size"] = len(self.semantic_enricher.entity_cache)
        stats["tfidf_corpus_size"] = len(self.semantic_enricher._tfidf_corpus)

        if self.enable_learning:
            top_patterns = sorted(
                self.mapping_patterns.items(), key=lambda x: x[1], reverse=True
            )[:5]

            top_domains = sorted(
                self.domain_similarities.items(), key=lambda x: x[1], reverse=True
            )[:5]

            stats["learning"] = {
                "most_common_patterns": dict(top_patterns),
                "domain_similarities": dict(top_domains),
                "learned_weights": self.learned_weights,
            }

        return stats


class AnalogicalReasoningEngine(AnalogicalReasoner):
    """Compatibility wrapper for analogical reasoning with full NLP integration"""

    def __init__(self, enable_caching: bool = True, enable_learning: bool = True):
        super().__init__(enable_caching=enable_caching, enable_learning=enable_learning)
        logger.info("Initialized AnalogicalReasoningEngine with semantic understanding")
        logger.info(f"Embedding method: {self.stats['embedding_method']}")

    def reason(self, input_data: Any, query: Optional[Dict] = None) -> Dict[str, Any]:
        """Main reasoning interface with enhanced semantic processing"""
        query = query or {}

        if isinstance(input_data, dict):
            # Multi-analogy search
            if "problem" in input_data or "target_problem" in input_data:
                target_problem = input_data.get(
                    "problem", input_data.get("target_problem")
                )

                k = query.get("k", 5)
                results = self.find_multiple_analogies(target_problem, k=k)

                return {
                    "found": len(results) > 0,
                    "analogies": results,
                    "count": len(results),
                    "confidence": results[0].get("confidence", 0.0) if results else 0.0,
                    "semantic_enrichment": True,
                    "embedding_method": self.stats["embedding_method"],
                }

            # Specific domain analogy
            elif "source_domain" in input_data:
                source_domain = input_data["source_domain"]
                target_problem = input_data.get("target_problem", input_data)
                mapping_type = MappingType(input_data.get("mapping_type", "structural"))

                result = self.find_structural_analogy(
                    source_domain, target_problem, mapping_type
                )
                result["semantic_enrichment"] = True
                result["embedding_method"] = self.stats["embedding_method"]
                return result

        return {"found": False, "reason": "Unsupported input format", "confidence": 0.0}

    def analyze_text_analogy(
        self, source_text: str, target_text: str
    ) -> Dict[str, Any]:
        """Analyze analogy between two text descriptions"""

        # Extract structure from text
        source_structure = self._extract_structure_from_text(source_text)
        target_structure = self._extract_structure_from_text(target_text)

        # Add as temporary domains
        self.add_domain("temp_source", source_structure)

        # Find analogy
        result = self.find_structural_analogy("temp_source", target_structure)

        # Clean up
        if "temp_source" in self.domain_knowledge:
            del self.domain_knowledge["temp_source"]

        return result

    def _extract_structure_from_text(self, text: str) -> Dict[str, Any]:
        """Extract structured representation from natural language text"""

        structure = {
            "description": text,
            "entities": [],
            "relations": [],
            "attributes": {},
        }

        if nlp:
            try:
                doc = nlp(text)

                # Extract entities
                for ent in doc.ents:
                    entity = Entity(
                        name=ent.text,
                        entity_type=ent.label_.lower(),
                        pos_tag=ent.root.pos_,
                    )
                    structure["entities"].append(entity)

                # Extract key nouns as entities
                for token in doc:
                    if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop:
                        if not any(e.name == token.text for e in structure["entities"]):
                            entity = Entity(
                                name=token.text,
                                entity_type="object",
                                pos_tag=token.pos_,
                                dependency_role=token.dep_,
                            )
                            structure["entities"].append(entity)

                # Extract relations from dependency parse
                for token in doc:
                    if token.pos_ == "VERB":
                        subjects = [
                            child
                            for child in token.children
                            if child.dep_ in ["nsubj", "nsubjpass"]
                        ]
                        objects = [
                            child
                            for child in token.children
                            if child.dep_ in ["dobj", "pobj"]
                        ]

                        for subj in subjects:
                            for obj in objects:
                                relation = Relation(
                                    predicate=token.lemma_,
                                    arguments=[subj.text, obj.text],
                                    relation_type="binary",
                                )
                                structure["relations"].append(relation)

                # Extract attributes from adjectives
                for token in doc:
                    if token.pos_ == "ADJ":
                        # Find what it modifies
                        head = token.head
                        if head.pos_ in ["NOUN", "PROPN"]:
                            if head.text not in structure["attributes"]:
                                structure["attributes"][head.text] = []
                            structure["attributes"][head.text].append(token.text)

            except Exception as e:
                logger.warning(f"Text structure extraction failed: {e}")

        # Fallback: simple extraction
        if not structure["entities"]:
            words = text.split()
            for word in words:
                if word and word[0].isupper():
                    structure["entities"].append(
                        Entity(name=word, entity_type="object")
                    )

        return structure


# Utility functions for advanced features


def compute_conceptual_distance(
    concept1: str, concept2: str, enricher: SemanticEnricher
) -> float:
    """
    Compute conceptual distance between two concepts in semantic space
    Lower distance = more similar concepts
    """
    similarity = enricher.compute_semantic_similarity(concept1, concept2)
    return 1.0 - similarity


def find_conceptual_path(
    start_concept: str, end_concept: str, knowledge_graph: nx.DiGraph
) -> List[str]:
    """
    Find conceptual path between two concepts in a knowledge graph
    Uses shortest path algorithm
    """
    if not NETWORKX_AVAILABLE or knowledge_graph is None:
        return []

    try:
        if start_concept in knowledge_graph and end_concept in knowledge_graph:
            path = nx.shortest_path(knowledge_graph, start_concept, end_concept)
            return path
    except nx.NetworkXNoPath:
        return []
    except Exception as e:
        logger.warning(f"Path finding failed: {e}")
        return []

    return []


def cluster_analogies(
    analogies: List[AnalogicalMapping], n_clusters: int = 3
) -> Dict[int, List[AnalogicalMapping]]:
    """
    Cluster analogical mappings by similarity
    Useful for finding patterns in multiple analogies
    """
    if not SKLEARN_AVAILABLE or not analogies:
        return {0: analogies}

    try:
        # Create feature vectors from mappings
        features = []
        for mapping in analogies:
            feature_vec = [
                mapping.mapping_score,
                mapping.confidence,
                mapping.semantic_coherence,
                len(mapping.entity_mappings),
                len(mapping.relation_mappings),
                mapping.structural_depth,
            ]
            features.append(feature_vec)

        features = np.array(features)

        # Cluster
        n_clusters = min(n_clusters, len(analogies))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(features)

        # Group by cluster
        clusters = defaultdict(list)
        for i, label in enumerate(labels):
            clusters[int(label)].append(analogies[i])

        return dict(clusters)

    except Exception as e:
        logger.error(f"Clustering failed: {e}")
        return {0: analogies}


def explain_analogy_differences(
    mapping1: AnalogicalMapping, mapping2: AnalogicalMapping
) -> str:
    """
    Generate explanation of differences between two analogical mappings
    Useful for understanding why one analogy is better than another
    """
    explanation = "Comparison of analogical mappings:\n\n"

    # Score comparison
    score_diff = mapping1.mapping_score - mapping2.mapping_score
    if abs(score_diff) > 0.1:
        better = "first" if score_diff > 0 else "second"
        explanation += f"The {better} mapping has a higher similarity score "
        explanation += (
            f"({mapping1.mapping_score:.2f} vs {mapping2.mapping_score:.2f}).\n"
        )
    else:
        explanation += "Both mappings have similar similarity scores.\n"

    # Semantic coherence
    coh_diff = mapping1.semantic_coherence - mapping2.semantic_coherence
    if abs(coh_diff) > 0.1:
        better = "first" if coh_diff > 0 else "second"
        explanation += f"The {better} mapping is more semantically coherent.\n"

    # Entity mappings
    entities1 = set(mapping1.entity_mappings.keys())
    entities2 = set(mapping2.entity_mappings.keys())

    only_in_1 = entities1 - entities2
    only_in_2 = entities2 - entities1

    if only_in_1:
        explanation += f"\nFirst mapping includes: {', '.join(list(only_in_1)[:3])}\n"
    if only_in_2:
        explanation += f"Second mapping includes: {', '.join(list(only_in_2)[:3])}\n"

    # Structural depth
    if mapping1.structural_depth != mapping2.structural_depth:
        deeper = (
            "first"
            if mapping1.structural_depth > mapping2.structural_depth
            else "second"
        )
        explanation += f"\nThe {deeper} mapping captures deeper structural relations.\n"

    return explanation


def test_semantic_similarity():
    """
    Test function to demonstrate real semantic similarity

    Compare this with the old hash-based approach
    """
    enricher = SemanticEnricher()

    # Test related concepts
    print("Testing semantic similarity:")
    print(f"cat vs dog: {enricher.compute_semantic_similarity('cat', 'dog'):.3f}")
    print(f"cat vs feline: {enricher.compute_semantic_similarity('cat', 'feline'):.3f}")
    print(
        f"cat vs quantum: {enricher.compute_semantic_similarity('cat', 'quantum'):.3f}"
    )
    print(
        f"running vs runner: {enricher.compute_semantic_similarity('running', 'runner'):.3f}"
    )
    print(
        f"car vs automobile: {enricher.compute_semantic_similarity('car', 'automobile'):.3f}"
    )
    print(f"happy vs sad: {enricher.compute_semantic_similarity('happy', 'sad'):.3f}")

    if enricher.embedding_model:
        print("\nUsing sentence-transformers embeddings (best quality)")
    else:
        print("\nUsing TF-IDF fallback embeddings (good quality)")


# Export main classes and functions
__all__ = [
    "Entity",
    "Relation",
    "AnalogicalMapping",
    "MappingType",
    "SemanticRelationType",
    "SemanticEnricher",
    "GoalRelevanceAnalyzer",
    "AnalogicalReasoner",
    "AnalogicalReasoningEngine",
    "compute_conceptual_distance",
    "find_conceptual_path",
    "cluster_analogies",
    "explain_analogy_differences",
    "test_semantic_similarity",
]
