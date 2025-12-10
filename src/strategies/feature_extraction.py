"""
Multi-Tier Feature Extraction for Tool Selection System

Implements hierarchical feature extraction with increasing complexity and cost,
from simple syntactic features to deep semantic and multimodal analysis.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from enum import Enum
import logging
import time
import re
import json
from pathlib import Path
import pickle
import hashlib
from abc import ABC, abstractmethod

try:
    import nltk

    HAS_NLTK = True
    # Download NLTK data if needed
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception as e:
            logger.debug(
                f"{self.__class__.__name__ if hasattr(self, '__class__') else 'Operation'} error: {e}"
            )
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        try:
            nltk.download("punkt", quiet=True)
        except Exception as e:
            logger.debug(
                f"{self.__class__.__name__ if hasattr(self, '__class__') else 'Operation'} error: {e}"
            )
    try:
        nltk.data.find("taggers/averaged_perceptron_tagger")
    except LookupError:
        try:
            nltk.download("averaged_perceptron_tagger", quiet=True)
        except Exception as e:
            logger.debug(
                f"{self.__class__.__name__ if hasattr(self, '__class__') else 'Operation'} error: {e}"
            )
    try:
        nltk.data.find("taggers/averaged_perceptron_tagger_eng")
    except LookupError:
        try:
            nltk.download("averaged_perceptron_tagger_eng", quiet=True)
        except Exception as e:
            logger.debug(
                f"{self.__class__.__name__ if hasattr(self, '__class__') else 'Operation'} error: {e}"
            )
except ImportError:
    HAS_NLTK = False
    nltk = None

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
import networkx as nx

logger = logging.getLogger(__name__)


class FeatureTier(Enum):
    """Feature extraction tiers with increasing complexity"""

    TIER1_SYNTACTIC = 1  # Basic syntactic features (fast)
    TIER2_STRUCTURAL = 2  # Structural and pattern features (medium)
    TIER3_SEMANTIC = 3  # Semantic and reasoning features (slow)
    TIER4_MULTIMODAL = 4  # Multi-modal and deep features (very slow)


@dataclass
class ExtractionResult:
    """Result of feature extraction"""

    features: np.ndarray
    tier: FeatureTier
    extraction_time_ms: float
    feature_names: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProblemStructure:
    """Structured representation of a problem"""

    text: Optional[str] = None
    tokens: Optional[List[str]] = None
    graph: Optional[nx.Graph] = None
    rules: Optional[List[str]] = None
    data: Optional[np.ndarray] = None
    modalities: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class FeatureExtractor(ABC):
    """Abstract base class for feature extractors"""

    @abstractmethod
    def extract(self, problem: Any) -> np.ndarray:
        """Extract features from problem"""
        pass

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Get feature names"""
        pass


class SyntacticFeatureExtractor(FeatureExtractor):
    """Tier 1: Basic syntactic features"""

    def __init__(self):
        self.feature_names = [
            "length",
            "num_tokens",
            "num_unique_tokens",
            "avg_token_length",
            "num_numbers",
            "num_symbols",
            "num_uppercase",
            "num_punctuation",
            "has_equation",
            "has_logic",
            "has_probability",
            "has_graph",
            "entropy",
            "complexity_score",
            "type_indicator",
        ]

    def extract(self, problem: Any) -> np.ndarray:
        """Extract syntactic features"""

        # Convert to string representation
        if isinstance(problem, str):
            text = problem
        elif isinstance(problem, dict):
            text = json.dumps(problem)
        else:
            text = str(problem) if problem is not None else ""

        features = []

        # Basic length features
        features.append(len(text))

        # Tokenization
        if HAS_NLTK and text:
            try:
                tokens = nltk.word_tokenize(text.lower())
            except Exception as e:  # Fallback to simple tokenization if NLTK fails
                tokens = text.lower().split()
        else:
            # Fallback to simple tokenization
            tokens = text.lower().split() if text else []

        features.append(len(tokens))
        features.append(len(set(tokens)))
        features.append(np.mean([len(t) for t in tokens]) if tokens else 0)

        # Character type counts
        features.append(sum(c.isdigit() for c in text))
        features.append(sum(not c.isalnum() and not c.isspace() for c in text))
        features.append(sum(c.isupper() for c in text))
        features.append(sum(c in ".,;:!?" for c in text))

        # Problem type indicators
        features.append(
            1.0 if any(op in text for op in ["=", "+", "-", "*", "/", "^"]) else 0.0
        )
        features.append(
            1.0
            if any(op in text for op in ["∧", "∨", "¬", "→", "⊢", "and", "or", "not"])
            else 0.0
        )
        features.append(
            1.0
            if any(
                term in text.lower()
                for term in ["probability", "likely", "chance", "random"]
            )
            else 0.0
        )
        features.append(
            1.0
            if any(
                term in text.lower()
                for term in ["node", "edge", "graph", "network", "connected"]
            )
            else 0.0
        )

        # Text entropy
        if tokens:
            token_counts = Counter(tokens)
            total = sum(token_counts.values())
            entropy = -sum(
                (count / total) * np.log2(count / total)
                for count in token_counts.values()
            )
            features.append(entropy)
        else:
            features.append(0.0)

        # Simple complexity score
        complexity = (len(text) / 100 + len(tokens) / 20 + len(set(tokens)) / 10) / 3
        features.append(min(1.0, complexity / 10))

        # Problem type classification (simplified)
        type_score = 0.0
        if "if" in text.lower() or "then" in text.lower():
            type_score = 0.2  # Logic
        elif any(term in text.lower() for term in ["probability", "distribution"]):
            type_score = 0.4  # Probabilistic
        elif any(term in text.lower() for term in ["cause", "effect", "influence"]):
            type_score = 0.6  # Causal
        elif any(term in text.lower() for term in ["similar", "like", "analogy"]):
            type_score = 0.8  # Analogical
        features.append(type_score)

        return np.array(features, dtype=np.float32)

    def get_feature_names(self) -> List[str]:
        """Get feature names"""
        return self.feature_names


class StructuralFeatureExtractor(FeatureExtractor):
    """Tier 2: Structural and pattern features"""

    def __init__(self):
        self.feature_names = []
        self.tfidf = TfidfVectorizer(max_features=50, stop_words="english")
        self.pos_tagger = None

    def extract(self, problem: Any) -> np.ndarray:
        """Extract structural features"""

        # Parse problem structure
        structure = self._parse_structure(problem)

        features = []

        # Structural complexity features
        features.extend(self._extract_structural_complexity(structure))

        # Pattern features
        features.extend(self._extract_pattern_features(structure))

        # Dependency features
        features.extend(self._extract_dependency_features(structure))

        # Graph features if applicable
        if structure.graph:
            features.extend(self._extract_graph_features(structure.graph))
        else:
            features.extend([0.0] * 10)  # Placeholder for graph features

        # TF-IDF features
        if structure.text:
            try:
                tfidf_features = self.tfidf.fit_transform([structure.text]).toarray()[0]
                features.extend(tfidf_features[:20])  # Top 20 TF-IDF features
            except Exception as e:
                features.extend([0.0] * 20)
        else:
            features.extend([0.0] * 20)

        return np.array(features, dtype=np.float32)

    def _parse_structure(self, problem: Any) -> ProblemStructure:
        """Parse problem into structured representation"""

        structure = ProblemStructure()

        if isinstance(problem, str):
            structure.text = problem
            if HAS_NLTK and problem:
                try:
                    structure.tokens = nltk.word_tokenize(problem)
                except Exception as e:
                    structure.tokens = problem.split()
            else:
                structure.tokens = problem.split() if problem else []
        elif isinstance(problem, dict):
            structure.text = json.dumps(problem)
            structure.metadata = problem

            # Check for specific structures
            if "graph" in problem:
                structure.graph = self._dict_to_graph(problem["graph"])
            if "rules" in problem:
                structure.rules = problem["rules"]
            if "data" in problem:
                structure.data = np.array(problem["data"])
        else:
            structure.text = str(problem) if problem is not None else ""
            if HAS_NLTK and structure.text:
                try:
                    structure.tokens = nltk.word_tokenize(structure.text)
                except Exception as e:
                    structure.tokens = structure.text.split()
            else:
                structure.tokens = structure.text.split() if structure.text else []

        return structure

    def _extract_structural_complexity(
        self, structure: ProblemStructure
    ) -> List[float]:
        """Extract structural complexity features"""

        features = []

        if structure.text:
            # Sentence complexity
            if HAS_NLTK:
                try:
                    sentences = nltk.sent_tokenize(structure.text)
                except Exception as e:  # Fallback: split on periods
                    sentences = structure.text.split(".")
            else:
                # Fallback: split on periods
                sentences = structure.text.split(".")

            features.append(float(len(sentences)))
            features.append(
                float(np.mean([len(s.split()) for s in sentences]) if sentences else 0)
            )

            # Parse tree depth (simplified)
            max_depth = self._estimate_parse_depth(structure.text)
            features.append(float(max_depth))

            # Clause complexity
            clause_markers = ["if", "then", "when", "where", "which", "that"]
            clause_count = sum(
                marker in structure.text.lower() for marker in clause_markers
            )
            features.append(float(clause_count))
        else:
            features.extend([0.0] * 4)

        # FIXED: Rule complexity with safe mean calculation and float conversion
        if structure.rules:
            features.append(float(len(structure.rules)))
            rule_lengths = [len(r) for r in structure.rules]
            features.append(float(np.mean(rule_lengths) if rule_lengths else 0.0))
        else:
            features.extend([0.0] * 2)

        return features

    def _extract_pattern_features(self, structure: ProblemStructure) -> List[float]:
        """Extract pattern-based features"""

        features = []

        if structure.text:
            text = structure.text.lower()

            # Logical patterns
            logic_patterns = [
                r"\bif\b.*\bthen\b",
                r"\bfor all\b",
                r"\bthere exists\b",
                r"\bnot\b.*\band\b",
                r"\beither\b.*\bor\b",
            ]
            for pattern in logic_patterns:
                features.append(1.0 if re.search(pattern, text) else 0.0)

            # Mathematical patterns
            math_patterns = [
                r"\d+\s*[+\-*/]\s*\d+",
                r"[a-z]\s*=\s*\d+",
                r"\bsolve\b",
                r"\bequation\b",
                r"\bfunction\b",
            ]
            for pattern in math_patterns:
                features.append(1.0 if re.search(pattern, text) else 0.0)
        else:
            features.extend([0.0] * 10)

        return features

    def _extract_dependency_features(self, structure: ProblemStructure) -> List[float]:
        """Extract dependency-based features"""

        features = []

        # FIXED: Added check for empty pos_tags to prevent division by zero
        if structure.tokens:
            # POS tag distribution
            if HAS_NLTK:
                try:
                    pos_tags = nltk.pos_tag(structure.tokens)
                except Exception as e:  # Fallback: no POS tagging
                    pos_tags = []
            else:
                # Fallback: no POS tagging
                pos_tags = []

            if pos_tags:
                pos_counts = Counter(tag for _, tag in pos_tags)

                # Major POS categories
                noun_ratio = sum(
                    pos_counts.get(tag, 0) for tag in ["NN", "NNS", "NNP", "NNPS"]
                ) / len(pos_tags)
                verb_ratio = sum(
                    pos_counts.get(tag, 0)
                    for tag in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
                ) / len(pos_tags)
                adj_ratio = sum(
                    pos_counts.get(tag, 0) for tag in ["JJ", "JJR", "JJS"]
                ) / len(pos_tags)

                features.extend([noun_ratio, verb_ratio, adj_ratio])

                # Dependency chain length (simplified)
                chain_length = self._estimate_dependency_chain(structure.tokens)
                features.append(chain_length)
            else:
                features.extend([0.0] * 4)
        else:
            features.extend([0.0] * 4)

        return features

    def _extract_graph_features(self, graph: nx.Graph) -> List[float]:
        """Extract graph-based features"""

        features = []

        # Basic graph statistics
        features.append(graph.number_of_nodes())
        features.append(graph.number_of_edges())
        features.append(nx.density(graph) if graph.number_of_nodes() > 1 else 0)

        # Connectivity
        features.append(1.0 if nx.is_connected(graph) else 0.0)
        features.append(nx.number_connected_components(graph))

        # Centrality measures
        if graph.number_of_nodes() > 0:
            degree_centrality = list(nx.degree_centrality(graph).values())
            features.append(np.mean(degree_centrality))
            features.append(np.std(degree_centrality))
        else:
            features.extend([0.0] * 2)

        # Cycles
        try:
            features.append(len(nx.cycle_basis(graph)))
        except Exception as e:
            features.append(0.0)

        # Diameter and radius
        if nx.is_connected(graph) and graph.number_of_nodes() > 1:
            features.append(nx.diameter(graph))
            features.append(nx.radius(graph))
        else:
            features.extend([0.0] * 2)

        return features

    def _dict_to_graph(self, graph_dict: Dict) -> nx.Graph:
        """Convert dictionary to NetworkX graph"""

        G = nx.Graph()

        if "nodes" in graph_dict:
            G.add_nodes_from(graph_dict["nodes"])

        if "edges" in graph_dict:
            G.add_edges_from(graph_dict["edges"])

        return G

    def _estimate_parse_depth(self, text: str) -> float:
        """Estimate parse tree depth (simplified)"""

        # Count nested parentheses/brackets as proxy
        max_depth = 0
        current_depth = 0

        for char in text:
            if char in "([{":
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char in ")]}":
                current_depth = max(0, current_depth - 1)

        return float(max_depth)

    def _estimate_dependency_chain(self, tokens: List[str]) -> float:
        """Estimate dependency chain length (simplified)"""

        # Use distance between related words as proxy
        connectives = ["and", "or", "but", "if", "then", "because"]
        distances = []

        for i, token in enumerate(tokens):
            if token.lower() in connectives:
                # Find next connective or end
                for j in range(i + 1, min(i + 20, len(tokens))):
                    if tokens[j].lower() in connectives or j == len(tokens) - 1:
                        distances.append(j - i)
                        break

        return np.mean(distances) if distances else 0.0

    def get_feature_names(self) -> List[str]:
        """Get feature names"""
        # This would be dynamically generated based on extraction
        return self.feature_names


class SemanticFeatureExtractor(FeatureExtractor):
    """Tier 3: Semantic and reasoning features"""

    def __init__(self):
        self.feature_names = []
        self.word_embeddings = {}  # Would load pre-trained embeddings
        self.reasoning_patterns = self._load_reasoning_patterns()

    def extract(self, problem: Any) -> np.ndarray:
        """Extract semantic features"""

        structure = self._parse_structure(problem)
        features = []

        # Semantic similarity features
        features.extend(self._extract_semantic_similarity(structure))

        # Reasoning pattern features
        features.extend(self._extract_reasoning_patterns(structure))

        # Conceptual features
        features.extend(self._extract_conceptual_features(structure))

        # Inference complexity
        features.extend(self._extract_inference_complexity(structure))

        # Embedding-based features
        features.extend(self._extract_embedding_features(structure))

        return np.array(features, dtype=np.float32)

    def _parse_structure(self, problem: Any) -> ProblemStructure:
        """Parse problem into structured representation"""

        structure = ProblemStructure()

        if isinstance(problem, str):
            structure.text = problem
            if HAS_NLTK and problem:
                try:
                    structure.tokens = nltk.word_tokenize(problem)
                except Exception as e:
                    structure.tokens = problem.split()
            else:
                structure.tokens = problem.split() if problem else []
        elif isinstance(problem, dict):
            structure.text = json.dumps(problem)
            structure.metadata = problem
        else:
            structure.text = str(problem) if problem is not None else ""
            if HAS_NLTK and structure.text:
                try:
                    structure.tokens = nltk.word_tokenize(structure.text)
                except Exception as e:
                    structure.tokens = structure.text.split()
            else:
                structure.tokens = structure.text.split() if structure.text else []

        return structure

    def _extract_semantic_similarity(self, structure: ProblemStructure) -> List[float]:
        """Extract semantic similarity features"""

        features = []

        # Similarity to known problem types
        problem_types = {
            "logical": ["premise", "conclusion", "valid", "argument", "proof"],
            "probabilistic": [
                "probability",
                "random",
                "distribution",
                "likelihood",
                "chance",
            ],
            "causal": ["cause", "effect", "influence", "result", "because"],
            "analogical": ["similar", "like", "compare", "analogy", "mapping"],
            "optimization": ["minimize", "maximize", "optimal", "best", "constraint"],
        }

        if structure.tokens:
            token_set = set(t.lower() for t in structure.tokens)

            for problem_type, keywords in problem_types.items():
                similarity = len(token_set.intersection(keywords)) / len(keywords)
                features.append(similarity)
        else:
            features.extend([0.0] * len(problem_types))

        return features

    def _extract_reasoning_patterns(self, structure: ProblemStructure) -> List[float]:
        """Extract reasoning pattern features"""

        features = []

        patterns = {
            "deductive": [r"if.*then", r"therefore", r"implies", r"follows that"],
            "inductive": [r"pattern", r"generalize", r"observe", r"conclude"],
            "abductive": [r"best explanation", r"hypothesis", r"likely cause"],
            "analogical": [r"similar to", r"like", r"corresponds to", r"maps to"],
            "causal": [r"causes", r"leads to", r"results in", r"because of"],
        }

        if structure.text:
            text_lower = structure.text.lower()

            for pattern_type, pattern_list in patterns.items():
                matches = sum(1 for p in pattern_list if re.search(p, text_lower))
                features.append(matches / len(pattern_list))
        else:
            features.extend([0.0] * len(patterns))

        return features

    def _extract_conceptual_features(self, structure: ProblemStructure) -> List[float]:
        """Extract conceptual complexity features"""

        features = []

        # Concept categories
        concepts = {
            "quantification": ["all", "some", "none", "every", "exists"],
            "modality": ["must", "might", "possible", "necessary", "could"],
            "temporal": ["before", "after", "during", "when", "while"],
            "spatial": ["above", "below", "beside", "between", "within"],
            "comparison": ["more", "less", "equal", "same", "different"],
        }

        if structure.tokens:
            token_set = set(t.lower() for t in structure.tokens)

            for concept_type, concept_words in concepts.items():
                presence = len(token_set.intersection(concept_words)) / len(
                    concept_words
                )
                features.append(presence)
        else:
            features.extend([0.0] * len(concepts))

        # Abstract vs concrete
        abstract_indicators = ["concept", "theory", "principle", "abstract", "general"]
        concrete_indicators = [
            "specific",
            "example",
            "instance",
            "particular",
            "concrete",
        ]

        if structure.tokens:
            token_set = set(t.lower() for t in structure.tokens)
            abstract_score = len(token_set.intersection(abstract_indicators)) / len(
                abstract_indicators
            )
            concrete_score = len(token_set.intersection(concrete_indicators)) / len(
                concrete_indicators
            )
            features.extend([abstract_score, concrete_score])
        else:
            features.extend([0.0, 0.0])

        return features

    def _extract_inference_complexity(self, structure: ProblemStructure) -> List[float]:
        """Extract inference complexity features"""

        features = []

        if structure.text:
            # Number of inference steps (estimated)
            inference_markers = ["therefore", "thus", "hence", "so", "consequently"]
            inference_steps = sum(
                1 for marker in inference_markers if marker in structure.text.lower()
            )
            features.append(inference_steps)

            # Logical connectives
            connectives = ["and", "or", "not", "if", "then", "iff"]
            connective_count = sum(
                1 for conn in connectives if conn in structure.text.lower()
            )
            features.append(connective_count)

            # Quantifier complexity
            quantifiers = ["all", "some", "none", "exists", "every"]
            quantifier_count = sum(
                1 for q in quantifiers if q in structure.text.lower()
            )
            features.append(quantifier_count)

            # Negation complexity
            negations = ["not", "no", "none", "neither", "never"]
            negation_count = sum(
                1 for neg in negations if neg in structure.text.lower()
            )
            features.append(negation_count)
        else:
            features.extend([0.0] * 4)

        return features

    def _extract_embedding_features(self, structure: ProblemStructure) -> List[float]:
        """Extract embedding-based features"""

        # Simplified - would use real embeddings in production
        # For now, using random projections as placeholder

        if structure.tokens:
            # Average word embedding (simulated)
            embedding_dim = 50
            embeddings = []

            for token in structure.tokens[:100]:  # Limit to first 100 tokens
                # Simulate word embedding lookup
                if token in self.word_embeddings:
                    embeddings.append(self.word_embeddings[token])
                else:
                    # Random projection as fallback
                    np.random.seed(hash(token) % 2**32)
                    embeddings.append(np.random.randn(embedding_dim) * 0.1)

            if embeddings:
                # Average pooling
                avg_embedding = np.mean(embeddings, axis=0)
                # Reduce dimensionality
                features = list(avg_embedding[:20])
            else:
                features = [0.0] * 20
        else:
            features = [0.0] * 20

        return features

    def _load_reasoning_patterns(self) -> Dict[str, List[str]]:
        """Load reasoning patterns (placeholder)"""

        return {
            "logical": ["modus_ponens", "modus_tollens", "syllogism"],
            "probabilistic": ["bayes", "likelihood", "marginal"],
            "causal": ["direct_cause", "indirect_effect", "confounder"],
        }

    def get_feature_names(self) -> List[str]:
        """Get feature names"""
        return self.feature_names


class MultimodalFeatureExtractor(FeatureExtractor):
    """Tier 4: Multimodal and deep features"""

    def __init__(self):
        self.feature_names = []
        self.modality_extractors = {
            "text": SemanticFeatureExtractor(),
            "graph": self._extract_graph_features,
            "table": self._extract_table_features,
            "formula": self._extract_formula_features,
            "image": self._extract_image_features,
        }

    def extract(self, problem: Any) -> np.ndarray:
        """Extract multimodal features"""

        # Detect modalities
        modalities = self._detect_modalities(problem)

        features = []

        # Extract features for each modality
        for modality in modalities:
            if modality in self.modality_extractors:
                modality_features = self._extract_modality_features(problem, modality)
                features.extend(modality_features)

        # Cross-modal features
        if len(modalities) > 1:
            cross_modal = self._extract_cross_modal_features(problem, modalities)
            features.extend(cross_modal)
        else:
            features.extend([0.0] * 10)  # Placeholder for cross-modal

        # Deep reasoning features
        deep_features = self._extract_deep_reasoning_features(problem)
        features.extend(deep_features)

        return np.array(features, dtype=np.float32)

    def _detect_modalities(self, problem: Any) -> List[str]:
        """Detect problem modalities"""

        modalities = []

        if isinstance(problem, str):
            modalities.append("text")
        elif isinstance(problem, dict):
            if "text" in problem or "description" in problem:
                modalities.append("text")
            if "graph" in problem or "network" in problem:
                modalities.append("graph")
            if "table" in problem or "data" in problem:
                modalities.append("table")
            if "formula" in problem or "equation" in problem:
                modalities.append("formula")
            if "image" in problem or "visual" in problem:
                modalities.append("image")

        if not modalities:
            modalities.append("text")  # Default to text

        return modalities

    def _extract_modality_features(self, problem: Any, modality: str) -> List[float]:
        """Extract features for specific modality"""

        if modality == "text":
            if isinstance(problem, str):
                return list(self.modality_extractors["text"].extract(problem)[:50])
            elif isinstance(problem, dict) and "text" in problem:
                return list(
                    self.modality_extractors["text"].extract(problem["text"])[:50]
                )

        elif modality == "graph":
            return self._extract_graph_features(problem)

        elif modality == "table":
            return self._extract_table_features(problem)

        elif modality == "formula":
            return self._extract_formula_features(problem)

        elif modality == "image":
            return self._extract_image_features(problem)

        return [0.0] * 50  # Default features

    def _extract_graph_features(self, problem: Any) -> List[float]:
        """Extract advanced graph features"""

        features = []

        if isinstance(problem, dict) and "graph" in problem:
            graph_data = problem["graph"]

            # Convert to NetworkX graph
            if isinstance(graph_data, dict):
                G = nx.Graph()
                if "nodes" in graph_data:
                    G.add_nodes_from(graph_data["nodes"])
                if "edges" in graph_data:
                    G.add_edges_from(graph_data["edges"])
            else:
                G = nx.Graph()

            # Advanced graph metrics
            features.append(G.number_of_nodes())
            features.append(G.number_of_edges())

            if G.number_of_nodes() > 0:
                # Clustering coefficient
                features.append(nx.average_clustering(G))

                # Transitivity
                features.append(nx.transitivity(G))

                # Assortativity (with variance check to avoid division by zero)
                try:
                    # Check if there's variance in the degrees before computing assortativity
                    degrees = [G.degree(n) for n in G.nodes()]
                    if len(degrees) > 1 and np.var(degrees) > 0:
                        features.append(nx.degree_assortativity_coefficient(G))
                    else:
                        features.append(0.0)
                except Exception as e:
                    features.append(0.0)

                # Spectral properties
                try:
                    laplacian_eigenvalues = nx.laplacian_spectrum(G)
                    features.append(np.mean(laplacian_eigenvalues))
                    features.append(np.std(laplacian_eigenvalues))
                except Exception as e:
                    features.extend([0.0] * 2)
            else:
                features.extend([0.0] * 5)
        else:
            features.extend([0.0] * 7)

        # Pad to fixed size
        while len(features) < 50:
            features.append(0.0)

        return features[:50]

    def _extract_table_features(self, problem: Any) -> List[float]:
        """Extract table/data features"""

        features = []

        if isinstance(problem, dict) and ("table" in problem or "data" in problem):
            data = problem.get("table", problem.get("data"))

            if isinstance(data, (list, np.ndarray)):
                data_array = np.array(data)

                # Shape features
                features.extend(list(data_array.shape))

                # Statistical features
                if data_array.size > 0:
                    features.append(np.mean(data_array))
                    features.append(np.std(data_array))
                    features.append(np.min(data_array))
                    features.append(np.max(data_array))

                    # Correlation structure
                    if len(data_array.shape) == 2 and data_array.shape[1] > 1:
                        corr_matrix = np.corrcoef(data_array.T)
                        features.append(np.mean(corr_matrix))
                        features.append(np.std(corr_matrix))
                    else:
                        features.extend([0.0] * 2)
                else:
                    features.extend([0.0] * 6)
            else:
                features.extend([0.0] * 8)
        else:
            features.extend([0.0] * 8)

        # Pad to fixed size
        while len(features) < 50:
            features.append(0.0)

        return features[:50]

    def _extract_formula_features(self, problem: Any) -> List[float]:
        """Extract formula/equation features"""

        features = []

        if isinstance(problem, dict) and "formula" in problem:
            formula = problem["formula"]
        elif isinstance(problem, str):
            formula = problem
        else:
            formula = ""

        if formula:
            # Operator counts
            operators = ["+", "-", "*", "/", "^", "=", "<", ">", "≤", "≥"]
            for op in operators:
                features.append(formula.count(op))

            # Function usage
            functions = ["sin", "cos", "tan", "log", "exp", "sqrt"]
            for func in functions:
                features.append(1.0 if func in formula.lower() else 0.0)

            # Complexity metrics
            features.append(len(formula))
            features.append(formula.count("("))  # Nesting depth proxy
            features.append(len(re.findall(r"[a-zA-Z]+", formula)))  # Variable count
        else:
            features.extend([0.0] * 19)

        # Pad to fixed size
        while len(features) < 50:
            features.append(0.0)

        return features[:50]

    def _extract_image_features(self, problem: Any) -> List[float]:
        """Extract image features (placeholder)"""

        # In production, would use CNN features
        features = [0.0] * 50

        if isinstance(problem, dict) and "image" in problem:
            # Simulate some basic features
            features[0] = 1.0  # Has image
            features[1] = 0.5  # Complexity estimate

        return features

    def _extract_cross_modal_features(
        self, problem: Any, modalities: List[str]
    ) -> List[float]:
        """Extract cross-modal interaction features"""

        features = []

        # Modality count
        features.append(len(modalities))

        # Modality combinations
        modal_pairs = [
            ("text", "graph"),
            ("text", "table"),
            ("text", "formula"),
            ("graph", "table"),
            ("graph", "formula"),
        ]

        for m1, m2 in modal_pairs:
            features.append(1.0 if m1 in modalities and m2 in modalities else 0.0)

        # Alignment features (simplified)
        if "text" in modalities and "graph" in modalities:
            features.append(0.7)  # Text-graph alignment
        else:
            features.append(0.0)

        if "text" in modalities and "formula" in modalities:
            features.append(0.8)  # Text-formula alignment
        else:
            features.append(0.0)

        # Information redundancy estimate
        features.append(1.0 / len(modalities) if modalities else 0.0)

        return features

    def _extract_deep_reasoning_features(self, problem: Any) -> List[float]:
        """Extract deep reasoning features"""

        features = []

        # Reasoning depth estimate
        if isinstance(problem, str):
            text = problem
        elif isinstance(problem, dict):
            text = json.dumps(problem)
        else:
            text = str(problem) if problem is not None else ""

        # Inference chain length
        inference_markers = ["therefore", "thus", "because", "since", "implies"]
        inference_count = sum(
            1 for marker in inference_markers if marker in text.lower()
        )
        features.append(min(1.0, inference_count / 5))

        # Abstraction level
        abstract_terms = ["concept", "principle", "theory", "general", "abstract"]
        abstraction = sum(1 for term in abstract_terms if term in text.lower())
        features.append(min(1.0, abstraction / 3))

        # Problem complexity score
        complexity = (len(text) / 1000 + inference_count / 5 + abstraction / 3) / 3
        features.append(min(1.0, complexity))

        # Reasoning type distribution (simplified)
        features.extend([0.2, 0.3, 0.2, 0.2, 0.1])  # Placeholder distribution

        # Meta-reasoning indicators
        meta_terms = ["think about", "consider", "analyze", "examine", "evaluate"]
        meta_score = sum(1 for term in meta_terms if term in text.lower())
        features.append(min(1.0, meta_score / 3))

        # Uncertainty indicators
        uncertainty_terms = ["maybe", "possibly", "might", "could", "uncertain"]
        uncertainty = sum(1 for term in uncertainty_terms if term in text.lower())
        features.append(min(1.0, uncertainty / 3))

        return features

    def get_feature_names(self) -> List[str]:
        """Get feature names"""
        return self.feature_names


class MultiTierFeatureExtractor:
    """
    Main multi-tier feature extraction system
    """

    def __init__(self):
        # Initialize tier extractors
        self.tier1_extractor = SyntacticFeatureExtractor()
        self.tier2_extractor = StructuralFeatureExtractor()
        self.tier3_extractor = SemanticFeatureExtractor()
        self.tier4_extractor = MultimodalFeatureExtractor()

        # Feature cache
        self.feature_cache = {}
        self.cache_size = 1000

        # Statistics
        self.extraction_stats = defaultdict(
            lambda: {"count": 0, "total_time": 0, "avg_time": 0}
        )

    def extract_tier1(self, problem: Any) -> np.ndarray:
        """Extract Tier 1 features (fast)"""

        start_time = time.time()
        features = self.tier1_extractor.extract(problem)

        self._update_stats(FeatureTier.TIER1_SYNTACTIC, time.time() - start_time)

        return features

    def extract_tier2(self, problem: Any) -> np.ndarray:
        """Extract Tier 2 features (medium)"""

        start_time = time.time()

        # Include Tier 1 features
        tier1_features = self.extract_tier1(problem)
        tier2_features = self.tier2_extractor.extract(problem)

        features = np.concatenate([tier1_features, tier2_features])

        self._update_stats(FeatureTier.TIER2_STRUCTURAL, time.time() - start_time)

        return features

    def extract_tier3(self, problem: Any) -> np.ndarray:
        """Extract Tier 3 features (slow)"""

        start_time = time.time()

        # Include lower tier features
        tier2_features = self.extract_tier2(problem)
        tier3_features = self.tier3_extractor.extract(problem)

        features = np.concatenate([tier2_features, tier3_features])

        self._update_stats(FeatureTier.TIER3_SEMANTIC, time.time() - start_time)

        return features

    def extract_tier4(self, problem: Any) -> np.ndarray:
        """Extract Tier 4 features (very slow)"""

        start_time = time.time()

        # Include all features
        tier3_features = self.extract_tier3(problem)
        tier4_features = self.tier4_extractor.extract(problem)

        features = np.concatenate([tier3_features, tier4_features])

        self._update_stats(FeatureTier.TIER4_MULTIMODAL, time.time() - start_time)

        return features

    def extract_adaptive(self, problem: Any, time_budget_ms: float) -> np.ndarray:
        """Extract features adaptively based on time budget"""

        # Check cache first
        cache_key = self._compute_cache_key(problem)
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]

        start_time = time.time()

        # Tier timing estimates (ms)
        tier_times = {
            FeatureTier.TIER1_SYNTACTIC: 10,
            FeatureTier.TIER2_STRUCTURAL: 50,
            FeatureTier.TIER3_SEMANTIC: 200,
            FeatureTier.TIER4_MULTIMODAL: 500,
        }

        # Select highest tier within budget
        features = None

        for tier in [
            FeatureTier.TIER4_MULTIMODAL,
            FeatureTier.TIER3_SEMANTIC,
            FeatureTier.TIER2_STRUCTURAL,
            FeatureTier.TIER1_SYNTACTIC,
        ]:
            if tier_times[tier] <= time_budget_ms:
                if tier == FeatureTier.TIER1_SYNTACTIC:
                    features = self.extract_tier1(problem)
                elif tier == FeatureTier.TIER2_STRUCTURAL:
                    features = self.extract_tier2(problem)
                elif tier == FeatureTier.TIER3_SEMANTIC:
                    features = self.extract_tier3(problem)
                elif tier == FeatureTier.TIER4_MULTIMODAL:
                    features = self.extract_tier4(problem)
                break

        if features is None:
            # Even Tier 1 is too expensive, use minimal features
            features = np.random.randn(10) * 0.1  # Placeholder

        # Cache features
        self._cache_features(cache_key, features)

        elapsed_ms = (time.time() - start_time) * 1000
        logger.debug(f"Adaptive extraction completed in {elapsed_ms:.1f}ms")

        return features

    def _compute_cache_key(self, problem: Any) -> str:
        """Compute cache key for problem"""

        if isinstance(problem, str):
            problem_str = problem
        else:
            problem_str = json.dumps(problem, sort_keys=True, default=str)

        return hashlib.md5(problem_str.encode()).hexdigest()

    def _cache_features(self, key: str, features: np.ndarray):
        """Cache extracted features"""

        if len(self.feature_cache) >= self.cache_size:
            # Remove oldest entry (simplified LRU)
            oldest_key = next(iter(self.feature_cache))
            del self.feature_cache[oldest_key]

        self.feature_cache[key] = features

    def _update_stats(self, tier: FeatureTier, elapsed_time: float):
        """Update extraction statistics"""

        stats = self.extraction_stats[tier]
        stats["count"] += 1
        stats["total_time"] += elapsed_time
        stats["avg_time"] = stats["total_time"] / stats["count"]

    def get_statistics(self) -> Dict[str, Any]:
        """Get extraction statistics"""

        return {
            tier.value: {
                "count": stats["count"],
                "avg_time_ms": stats["avg_time"] * 1000,
            }
            for tier, stats in self.extraction_stats.items()
        }
