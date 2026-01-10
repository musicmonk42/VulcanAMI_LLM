"""
Semantic enrichment for entities and relations in analogical reasoning.

This module provides sophisticated semantic enrichment capabilities using
embeddings, TF-IDF fallbacks, and NLP processing. It enables entities and
relations to be enriched with semantic information for improved analogy matching.

Key Features:
    - Singleton pattern for embedding model (prevents reloading)
    - TF-IDF fallback when sentence-transformers unavailable
    - Character-based embedding as ultimate fallback
    - Semantic categorization
    - POS tagging integration
    - Thread-safe model loading

Performance Optimizations:
    - Shared embedding model across instances (singleton)
    - Caching for embeddings and enriched entities
    - Lazy loading of NLP models
    - Efficient vectorization

Module: vulcan.reasoning.analogical.semantic_enricher
Author: Vulcan AI Team
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, Optional, Set

import numpy as np
import numpy.typing as npt

from .types import Entity, Relation, SemanticRelationType

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available, TF-IDF fallback disabled")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available, using TF-IDF fallback")

try:
    import spacy
    SPACY_AVAILABLE = True
    
    # Lazy loading of spaCy model
    _nlp = None
    _nlp_lock = threading.Lock()
    _nlp_loaded = False
    
    def get_nlp():
        """Lazy-load spaCy model on first use (thread-safe)."""
        global _nlp, _nlp_loaded
        if _nlp_loaded:
            return _nlp
        with _nlp_lock:
            if _nlp_loaded:
                return _nlp
            for model_name in ["en_core_web_lg", "en_core_web_md", "en_core_web_sm"]:
                try:
                    _nlp = spacy.load(model_name)
                    logger.info(f"Loaded spaCy model '{model_name}' for NLP")
                    break
                except OSError:
                    continue
                except Exception as e:
                    logger.warning(f"Error loading spaCy model '{model_name}': {e}")
                    continue
            if _nlp is None:
                logger.warning("spaCy model not available")
            _nlp_loaded = True
        return _nlp
except ImportError:
    SPACY_AVAILABLE = False
    
    def get_nlp():
        """Return None when spaCy unavailable."""
        return None


# Constants
EMBEDDING_DIMENSION = 384  # Standard dimension for compatibility
TFIDF_MAX_FEATURES = 384  # Match embedding dimension
TFIDF_NGRAM_RANGE = (2, 4)  # Character n-grams
TFIDF_TRAINING_SIZE = 20  # Number of texts before fitting completes


class SemanticEnricher:
    """
    Enriches entities and relations with semantic information.
    
    This class provides sophisticated semantic enrichment using embeddings,
    TF-IDF fallbacks, and NLP processing. It uses a singleton pattern for
    the embedding model to prevent expensive reloading.
    
    Performance Features:
        - Singleton embedding model (loaded once, shared across instances)
        - Caching for embeddings and enriched entities
        - Thread-safe model initialization
        - Multiple fallback strategies
    
    Attributes:
        embedding_model: Shared sentence transformer model (singleton)
        embedding_cache: Cache of text → embedding mappings
        entity_cache: Cache of enriched entities
        semantic_categories: Predefined semantic categories
    
    Examples:
        >>> enricher = SemanticEnricher()
        >>> entity = Entity(name="car", entity_type="vehicle")
        >>> enriched = enricher.enrich_entity(entity)
        >>> print(enriched.semantic_category)  # "object"
        >>> similarity = enricher.compute_semantic_similarity("car", "automobile")
        >>> print(f"Similarity: {similarity:.2f}")  # High similarity
    
    Note:
        The embedding model is loaded only once (singleton pattern) to avoid
        3-5 second loading delays on each instantiation. All instances share
        the same model through class-level storage.
    """
    
    # Singleton pattern for embedding model (class-level)
    _shared_embedding_model: Optional[SentenceTransformer] = None
    _shared_model_lock = threading.Lock()
    _model_load_attempted = False
    
    @classmethod
    def _get_shared_model(cls) -> Optional[SentenceTransformer]:
        """
        Get or create the shared embedding model (singleton pattern).
        
        Uses double-checked locking for thread-safe lazy initialization.
        The model is loaded only once and shared across all instances.
        
        Returns:
            SentenceTransformer model if available, None otherwise.
        """
        if cls._shared_embedding_model is None and not cls._model_load_attempted:
            with cls._shared_model_lock:
                # Double-checked locking pattern
                if cls._shared_embedding_model is None and not cls._model_load_attempted:
                    cls._model_load_attempted = True
                    if SENTENCE_TRANSFORMERS_AVAILABLE:
                        try:
                            logger.info(
                                "[TIMING] Loading SentenceTransformer model "
                                "(singleton, will load ONCE)..."
                            )
                            start = time.perf_counter()
                            cls._shared_embedding_model = SentenceTransformer(
                                "all-MiniLM-L6-v2"
                            )
                            elapsed = time.perf_counter() - start
                            logger.info(
                                f"[TIMING] SentenceTransformer model loaded "
                                f"in {elapsed:.2f}s (singleton)"
                            )
                        except Exception as e:
                            logger.warning(
                                f"Failed to load sentence transformer: {e}, "
                                "using TF-IDF fallback"
                            )
        
        return cls._shared_embedding_model
    
    def __init__(self):
        """
        Initialize the semantic enricher.
        
        Sets up caching, loads the shared embedding model, and initializes
        fallback mechanisms (TF-IDF) if needed.
        """
        self.embedding_cache: Dict[str, npt.NDArray[np.float32]] = {}
        self.entity_cache: Dict[str, Entity] = {}
        
        # Use shared singleton model
        self.embedding_model = SemanticEnricher._get_shared_model()
        
        # Initialize TF-IDF vectorizer for fallback
        self._tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self._tfidf_corpus: list[str] = []
        self._corpus_hash: Set[int] = set()
        self._tfidf_fitted = False
        
        # Initialize TF-IDF if embedding model not available
        if self.embedding_model is None:
            logger.info("Using TF-IDF fallback for semantic embeddings")
            self._init_tfidf_vectorizer()
        
        # Build semantic categories dictionary
        self.semantic_categories = self._build_semantic_categories()
    
    def _init_tfidf_vectorizer(self) -> None:
        """
        Initialize TF-IDF vectorizer for fallback embeddings.
        
        Uses character n-grams to capture semantic similarity when
        sentence transformers are unavailable.
        """
        if SKLEARN_AVAILABLE:
            try:
                self._tfidf_vectorizer = TfidfVectorizer(
                    analyzer="char_wb",  # Character n-grams with word boundaries
                    ngram_range=TFIDF_NGRAM_RANGE,
                    max_features=TFIDF_MAX_FEATURES,
                    lowercase=True,
                    strip_accents="unicode",
                    min_df=1,
                    max_df=1.0,
                )
                logger.info("Initialized TF-IDF vectorizer for semantic embeddings")
            except Exception as e:
                logger.warning(f"Failed to initialize TF-IDF vectorizer: {e}")
    
    def _build_semantic_categories(self) -> Dict[str, Set[str]]:
        """
        Build semantic category dictionary with keyword mappings.
        
        Returns:
            Dictionary mapping category names to sets of keywords.
        """
        return {
            "person": {
                "person", "people", "individual", "human",
                "man", "woman", "child",
            },
            "animal": {
                "animal", "creature", "beast", "bird", "fish", "dog", "cat",
            },
            "object": {
                "object", "thing", "item", "tool", "device", "machine",
            },
            "place": {
                "place", "location", "area", "region", "city", "country",
            },
            "concept": {
                "concept", "idea", "notion", "principle", "theory",
            },
            "action": {
                "action", "activity", "process", "event", "operation",
            },
            "property": {
                "property", "attribute", "quality", "characteristic", "feature",
            },
        }
    
    def enrich_entity(self, entity: Entity) -> Entity:
        """
        Enrich an entity with semantic information.
        
        Adds embedding, semantic category, and POS tag to the entity.
        Uses caching to avoid redundant processing.
        
        Args:
            entity: Entity to enrich (modified in-place)
            
        Returns:
            The enriched entity (same object).
            
        Examples:
            >>> entity = Entity(name="car", entity_type="vehicle")
            >>> enriched = enricher.enrich_entity(entity)
            >>> print(enriched.semantic_category)  # "object"
            >>> print(enriched.embedding.shape)  # (384,)
        """
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
        nlp = get_nlp()
        if nlp:
            try:
                doc = nlp(entity.name)
                if doc:
                    entity.pos_tag = doc[0].pos_ if len(doc) > 0 else None
            except Exception as e:
                logger.debug(f"Failed to assign POS tag to entity: {e}")
        
        # Cache the enriched entity
        self.entity_cache[cache_key] = entity
        
        return entity
    
    def _get_embedding(self, text: str) -> npt.NDArray[np.float32]:
        """
        Get semantic embedding for text.
        
        Uses sentence transformers if available, otherwise falls back to
        TF-IDF or character-based embeddings.
        
        Args:
            text: Text to embed
            
        Returns:
            384-dimensional embedding vector.
        """
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        try:
            if self.embedding_model:
                # Use sentence transformer
                embedding = self.embedding_model.encode(
                    text, convert_to_numpy=True, show_progress_bar=False
                )
            else:
                # Use TF-IDF fallback
                embedding = self._fallback_embedding(text)
            
            self.embedding_cache[text] = embedding
            return embedding
        except Exception as e:
            logger.warning(f"Embedding generation failed for '{text}': {e}")
            return self._fallback_embedding(text)
    
    def _fallback_embedding(self, text: str) -> npt.NDArray[np.float32]:
        """
        Generate meaningful embedding using TF-IDF.
        
        Uses character n-grams to capture semantic similarity. Falls back
        to character-based features if TF-IDF is unavailable.
        
        Args:
            text: Text to embed
            
        Returns:
            384-dimensional embedding vector.
        """
        if not SKLEARN_AVAILABLE or self._tfidf_vectorizer is None:
            return self._character_embedding(text)
        
        try:
            # Add to corpus if new
            text_hash = hash(text)
            if text_hash not in self._corpus_hash:
                self._tfidf_corpus.append(text)
                self._corpus_hash.add(text_hash)
                
                # Only refit during initial training phase
                if not self._tfidf_fitted and len(self._tfidf_corpus) <= TFIDF_TRAINING_SIZE:
                    if len(self._tfidf_corpus) > 1:
                        try:
                            self._tfidf_vectorizer.fit(self._tfidf_corpus)
                            if len(self._tfidf_corpus) == TFIDF_TRAINING_SIZE:
                                self._tfidf_fitted = True
                        except Exception as e:
                            logger.warning(f"TF-IDF refitting failed: {e}")
            
            # Generate embedding
            if len(self._tfidf_corpus) > 1:
                try:
                    # Ensure vectorizer is fitted
                    if (not hasattr(self._tfidf_vectorizer, "vocabulary_") or
                        self._tfidf_vectorizer.vocabulary_ is None):
                        self._tfidf_vectorizer.fit(self._tfidf_corpus)
                    
                    vector = self._tfidf_vectorizer.transform([text]).toarray()[0]
                except Exception:
                    return self._character_embedding(text)
            else:
                return self._character_embedding(text)
            
            # Ensure correct dimension
            if len(vector) < EMBEDDING_DIMENSION:
                vector = np.pad(vector, (0, EMBEDDING_DIMENSION - len(vector)))
            elif len(vector) > EMBEDDING_DIMENSION:
                vector = vector[:EMBEDDING_DIMENSION]
            
            # Normalize
            norm = np.linalg.norm(vector)
            if norm > 1e-10:
                vector = vector / norm
            else:
                return self._character_embedding(text)
            
            return vector.astype(np.float32)
        
        except Exception as e:
            logger.warning(
                f"TF-IDF embedding failed for '{text}': {e}, "
                "using character embedding"
            )
            return self._character_embedding(text)
    
    def _character_embedding(self, text: str) -> npt.NDArray[np.float32]:
        """
        Generate character-based embedding for text.
        
        Uses character frequencies and bigrams to create a meaningful
        embedding that captures lexical similarity.
        
        Args:
            text: Text to embed
            
        Returns:
            384-dimensional embedding vector.
        """
        text_lower = text.lower()
        
        # Character frequency features (256 dimensions)
        char_freq = np.zeros(256, dtype=np.float32)
        for char in text_lower:
            char_ord = ord(char) if ord(char) < 256 else ord(" ")
            char_freq[char_ord] += 1
        
        # Normalize
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
        
        # Ensure exactly 384 dimensions
        if len(embedding) < EMBEDDING_DIMENSION:
            embedding = np.pad(embedding, (0, EMBEDDING_DIMENSION - len(embedding)))
        else:
            embedding = embedding[:EMBEDDING_DIMENSION]
        
        # Normalize to unit vector
        norm = np.linalg.norm(embedding)
        if norm > 1e-10:
            embedding = embedding / norm
        
        return embedding.astype(np.float32)
    
    def _categorize_entity(self, name: str, entity_type: str) -> str:
        """
        Categorize an entity semantically.
        
        Args:
            name: Entity name
            entity_type: Entity type
            
        Returns:
            Semantic category string.
        """
        name_lower = name.lower()
        
        # Check against known categories
        for category, keywords in self.semantic_categories.items():
            if any(kw in name_lower for kw in keywords):
                return category
        
        # Use entity type as fallback
        return entity_type
    
    def enrich_relation(self, relation: Relation) -> Relation:
        """
        Enrich a relation with semantic type.
        
        Analyzes the predicate to determine the semantic type of the relation
        (e.g., HYPERNYM, CAUSE, etc.).
        
        Args:
            relation: Relation to enrich (modified in-place)
            
        Returns:
            The enriched relation (same object).
        """
        predicate_lower = relation.predicate.lower()
        
        # Classify relation type based on predicate
        if any(word in predicate_lower for word in ["is", "are", "typeof", "instanceof"]):
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
        elif any(word in predicate_lower for word in ["opposite", "unlike", "different"]):
            relation.semantic_type = SemanticRelationType.ANTONYM
        
        return relation
    
    def compute_semantic_similarity(self, term1: str, term2: str) -> float:
        """
        Compute semantic similarity between two terms.
        
        Uses embeddings and cosine similarity to measure how semantically
        similar two terms are.
        
        Args:
            term1: First term
            term2: Second term
            
        Returns:
            Similarity score in range [0.0, 1.0].
            
        Examples:
            >>> enricher = SemanticEnricher()
            >>> sim = enricher.compute_semantic_similarity("car", "automobile")
            >>> print(f"{sim:.2f}")  # High similarity (e.g., 0.85)
            >>> sim = enricher.compute_semantic_similarity("car", "banana")
            >>> print(f"{sim:.2f}")  # Low similarity (e.g., 0.15)
        """
        emb1 = self._get_embedding(term1)
        emb2 = self._get_embedding(term2)
        
        # Cosine similarity
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0
        
        similarity = np.dot(emb1, emb2) / (norm1 * norm2)
        return float(np.clip(similarity, 0.0, 1.0))


# Export public API
__all__ = [
    "SemanticEnricher",
    "get_nlp",
    "SKLEARN_AVAILABLE",
    "SENTENCE_TRANSFORMERS_AVAILABLE",
    "SPACY_AVAILABLE",
]
