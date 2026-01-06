"""
Semantic Tool Matcher for VulcanAMI

This module adds semantic matching between queries and tool descriptions
to fix the tool selection failure where symbolic and analogical tools
are never selected.

ROOT CAUSE: The tool_selector.py has NO mechanism to match query content
to tool purposes. It relies entirely on:
1. Historical memory (empty at start)
2. Contextual bandit (untrained at start)
3. Uniform prior (equal probability = random selection)

SOLUTION: Add semantic matching that computes cosine similarity between
query embeddings and pre-computed tool description embeddings, then
boost prior probabilities for semantically matching tools.

INSTALLATION:
1. Copy this file to: src/vulcan/reasoning/selection/semantic_tool_matcher.py
2. Apply the patch to memory_prior.py (see PATCH section below)
3. Restart the server

Author: Claude (diagnostic session)
Date: 2024-12-26
"""

import hashlib
import logging
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Import circuit breaker for embedding performance protection
try:
    from .embedding_circuit_breaker import (
        get_embedding_circuit_breaker,
    )
    CIRCUIT_BREAKER_AVAILABLE = True
except ImportError:
    CIRCUIT_BREAKER_AVAILABLE = False
    get_embedding_circuit_breaker = None
    logger.debug("[SemanticToolMatcher] Circuit breaker not available")


# =============================================================================
# RICH TOOL DESCRIPTIONS - The key to semantic matching
# =============================================================================

TOOL_DESCRIPTIONS = {
    "symbolic": """
        Formal logical reasoning and deductive proof using first-order logic.
        
        USE THIS TOOL FOR:
        - Syllogisms: "All X are Y. Z is X. Therefore Z is Y."
        - Classic logic puzzles: "All philosophers are mortal. Socrates is a philosopher."
        - Deductive proofs with premises and conclusions
        - If-then logical statements and implications
        - Modus ponens: "If P then Q. P is true. Therefore Q."
        - Modus tollens: "If P then Q. Q is false. Therefore P is false."
        - Universal quantifiers: forall, all, every, each, any
        - Existential quantifiers: exists, some, there is, there exists
        - Logical connectives: and, or, not, implies, if and only if
        - Mathematical proofs and theorem proving
        - Constraint satisfaction problems
        - Rule-based inference
        
        TRIGGER KEYWORDS: premise, conclude, deduce, prove, therefore, implies,
        if-then, syllogism, logic, forall, exists, mortal, philosopher, valid,
        invalid, entailment, inference, axiom, theorem, hypothesis, derivation,
        logical, reasoning, deduction, modus ponens, modus tollens
    """,
    "analogical": """
        Reasoning by analogy, metaphor, and cross-domain mapping.
        
        USE THIS TOOL FOR:
        - Analogies: "X is like Y", "X is to Y as A is to B"
        - Proportional analogies: "Doctor is to patient as teacher is to student"
        - Metaphorical reasoning: "The atom is like a solar system"
        - Cross-domain transfer: "Apply this concept from biology to economics"
        - Structural mapping between domains
        - "Just as... so too..." reasoning patterns
        - Finding similarities between different systems
        - Explaining one thing in terms of another
        - Pattern matching across contexts
        - Learning from examples by similarity
        
        TRIGGER KEYWORDS: analogy, analogous, like, similar to, corresponds to,
        is to, as, metaphor, compare, comparison, mapping, transfer, pattern,
        resembles, parallels, equivalent, mirrors, reflects, models, represents,
        just as, so too, in the same way, similarly, likewise, reminiscent
    """,
    "causal": """
        Causal reasoning, intervention analysis, and counterfactual thinking.
        
        USE THIS TOOL FOR:
        - Cause and effect relationships
        - "What would happen if..." questions
        - Intervention analysis: "What if we change X?"
        - Counterfactual reasoning: "What if X had been different?"
        - Root cause analysis
        - Causal chains and mechanisms
        - Confounding variable identification
        - Treatment effect estimation
        - Do-calculus and causal inference
        - Why questions about causation
        
        TRIGGER KEYWORDS: cause, effect, because, due to, leads to, results in,
        consequence, counterfactual, intervention, what if, would have, impact,
        influence, mechanism, pathway, mediation, confound, treatment, outcome,
        why, how come, reason, factor, determinant
    """,
    "probabilistic": """
        Probabilistic reasoning, uncertainty quantification, and Bayesian inference.
        
        USE THIS TOOL FOR:
        - Probability calculations and estimates
        - Uncertainty quantification
        - Bayesian inference and updating
        - Risk assessment
        - Likelihood estimation
        - Statistical reasoning
        - Confidence intervals
        - Prediction with uncertainty
        - Evidence combination
        - Belief updates
        
        TRIGGER KEYWORDS: probability, likely, unlikely, chance, odds, risk,
        uncertain, confidence, Bayesian, posterior, prior, evidence, belief,
        estimate, predict, expect, frequency, distribution, random, stochastic,
        percentage, ratio, rate
    """,
    "multimodal": """
        Multi-modal reasoning combining visual, textual, and other information types.
        
        USE THIS TOOL FOR:
        - Image analysis and understanding
        - Describing what is in a picture or photo
        - Reading text from images (OCR)
        - Analyzing charts, graphs, diagrams, and figures
        - Understanding screenshots and visual documents
        - Video frame analysis
        - Audio transcription and analysis
        - Combining information from images and text
        - Document analysis with visual elements
        - Any query that references an uploaded file or image
        - Questions like "what is this?", "describe this image", "what do you see?"
        - Analyzing PDFs with images or complex layouts
        - Cross-modal reasoning between different data types
        
        TRIGGER KEYWORDS: image, picture, photo, video, audio, visual, diagram,
        chart, graph, table, figure, document, PDF, screenshot, see, look,
        describe, analyze, what is this, uploaded, attached, file, scan, read
    """,
    "general": """
        General-purpose handler for lightweight queries that don't need complex reasoning.
        
        USE THIS TOOL FOR:
        - Simple greetings and conversational exchanges
        - Identity/attribution queries (who created this, what are you)
        - Philosophical paradoxes and thought experiments (no deep mathematical analysis needed)
        - Ethical dilemmas and moral philosophy questions (discussion, not calculation)
        - Simple factual lookups that don't require reasoning
        - Acknowledging user input (thanks, okay, goodbye)
        - Meta-questions about the system itself
        
        BYPASS COMPLEX REASONING FOR:
        - Paradoxes like "This sentence is false" (direct response, no logic solving)
        - Philosophical thought experiments like "The Experience Machine", "Trolley Problem"
        - Ethical dilemmas involving hedonism, utilitarianism, virtue ethics
        - Questions about consciousness, free will, meaning of life
        - Greetings like "Hello", "How are you?"
        - Identity queries like "Who made you?", "Who created you?"
        - Simple facts like "What is the capital of France?"
        
        TRIGGER KEYWORDS: hello, hi, hey, thanks, goodbye, who created, who made,
        what are you, paradox, this sentence is false, greeting, conversation,
        simple, basic, general, help, philosophical, ethical, moral, dilemma,
        hedonism, utilitarianism, virtue, consciousness, free will, experience machine,
        thought experiment, existential, meaning of life
    """,
    "philosophical": """
        Philosophical and ethical reasoning using deontic logic and moral frameworks.
        
        USE THIS TOOL FOR:
        - Deontic logic: obligations, permissions, prohibitions
        - Moral permissibility analysis (is action X morally permissible?)
        - Ethical dilemmas and moral philosophy questions
        - Moral uncertainty and decision theory under ethical ambiguity
        - Pareto dominance and multi-criteria ethical comparisons
        - Deontological reasoning (Kantian ethics, duty-based)
        - Consequentialist reasoning (utilitarianism, outcome-based)
        - Virtue ethics and character-based evaluation
        - Contractualist reasoning (fairness, consent-based)
        - Care ethics and relationship-based moral reasoning
        - Trolley problems and ethical thought experiments
        - Value conflicts and moral trade-offs
        - Normative reasoning about what one ought to do
        - Philosophical paradoxes requiring ethical analysis
        
        TRIGGER KEYWORDS: ethical, moral, permissible, obligatory, forbidden,
        duty, right, wrong, virtue, value, harm, benefit, justice, fairness,
        rights, autonomy, consent, welfare, utility, deontological, kantian,
        consequentialist, utilitarian, categorical imperative, trolley problem,
        dilemma, philosophical, deontic, normative, ought, should, must not,
        permissibility, obligation, prohibition
    """,
    "mathematical": """
        Mathematical computation, numerical calculation, and symbolic mathematics.
        
        USE THIS TOOL FOR:
        - Numerical calculations and arithmetic
        - Calculus: derivatives, integrals, limits
        - Linear algebra: matrices, vectors, eigenvalues
        - Solving equations and systems of equations
        - Optimization problems
        - Statistical calculations
        - Numerical methods and approximations
        - Formula evaluation and simplification
        - Graph theory calculations
        - Combinatorics and counting
        
        TRIGGER KEYWORDS: calculate, compute, solve, evaluate, simplify,
        derivative, integral, limit, matrix, vector, eigenvalue, optimize,
        minimize, maximize, equation, formula, sum, product, factorial,
        logarithm, exponential, root, sqrt, polynomial, function, graph
    """,
    "language": """
        Natural language processing, parsing, and linguistic reasoning.
        
        USE THIS TOOL FOR:
        - Quantifier scope disambiguation (every, some, most, few)
        - Semantic parsing and formalization
        - Syntactic analysis and parse trees
        - Coreference resolution (pronouns, anaphora)
        - Word sense disambiguation
        - Sentiment analysis
        - Named entity recognition
        - Dependency parsing
        - Constituent parsing
        - First-order logic formalization of natural language
        - Scope ambiguity resolution
        - Linguistic entailment
        
        TRIGGER KEYWORDS: quantifier, scope, parse, parsing, syntax, semantic,
        pronoun, coreference, anaphora, formalize, formalization, linguistic,
        ambiguity, entailment, sentiment, entity, dependency, constituent,
        first-order logic, FOL, every, some, most, few, disambiguate
    """,
}


# =============================================================================
# KEYWORD BOOSTING - Fast pattern matching without embeddings
# =============================================================================

TOOL_KEYWORDS = {
    "symbolic": [
        # Syllogism patterns
        "all",
        "every",
        "each",
        "any",
        "no",
        "some",
        "mortal",
        "philosopher",
        "socrates",
        "human",
        "animal",
        # Logic keywords
        "premise",
        "conclusion",
        "therefore",
        "hence",
        "thus",
        "deduce",
        "deduction",
        "infer",
        "inference",
        "prove",
        "proof",
        "theorem",
        "axiom",
        "valid",
        "invalid",
        "sound",
        "unsound",
        "implies",
        "if then",
        "if-then",
        "implication",
        "forall",
        "exists",
        "∀",
        "∃",
        "modus ponens",
        "modus tollens",
        "syllogism",
        "logic",
        "logical",
        "entailment",
        "entails",
        "formal",
        "first-order",
        "fol",
        # BUG #2 FIX: SAT/satisfiability problem detection keywords
        # These keywords strongly indicate symbolic logic problems that
        # MUST NOT be routed to analogical (which was happening due to
        # uniform semantic similarity scores of ~0.35)
        "satisfiable",
        "sat",
        "unsatisfiable",
        "cnf",
        "dnf",
        "→",
        "∧",
        "∨",
        "¬",
        "⊢",
        "⊨",
        "contradiction",
        "tautology",
        "propositional",
        "quantifier",
        # ENHANCED: Mathematical proof keywords
        "q.e.d.",
        "qed",
        "lemma",
        "corollary",
        "proposition",
        "definition",
        "postulate",
        "conjecture",
        "by induction",
        "by contradiction",
        "contrapositive",
        "if and only if",
        "iff",
        "necessary",
        "sufficient",
        "assume",
        "suppose",
        "let",
        "given",
        # ENHANCED: Mathematical derivation
        "derive",
        "derivation",
        "show that",
        "demonstrate",
        "equation",
        "formula",
        "expression",
        "solve",
        "solution",
        "compute",
        "calculate",
        # ENHANCED: Advanced math
        "integral",
        "derivative",
        "differential",
        "partial",
        "gradient",
        "laplacian",
        "limit",
        "convergence",
        "series",
        "matrix",
        "vector",
        "tensor",
        "eigenvalue",
        "lagrangian",
        "hamiltonian",
        "euler-lagrange",
    ],
    "analogical": [
        # Analogy patterns
        "is like",
        "is to",
        "as a",
        "analogous",
        "analogy",
        "analogies",
        "metaphor",
        "similar to",
        "corresponds to",
        "compare",
        "comparison",
        "just as",
        "so too",
        "likewise",
        "resembles",
        "mirrors",
        "parallels",
        "mapping",
        "transfer",
        "pattern",
        # Common analogy domains
        "relationship",
        "proportion",
        "example",
        "instance",
        # Structure mapping keywords (CRITICAL #1 FIX)
        "map",
        "domain",
        "source domain",
        "target domain",
        "correspondence",
        "structural",
        "structure mapping",
    ],
    "causal": [
        "cause",
        "causes",
        "caused",
        "effect",
        "effects",
        "affect",
        "affects",
        "because",
        "due to",
        "leads to",
        "results in",
        "consequence",
        "consequences",
        "counterfactual",
        "what if",
        "would have",
        "intervention",
        "intervene",
        "mechanism",
        "pathway",
        "why",
        "reason",
        "factor",
        "impact",
        "influence",
        # BUG #2 FIX: Causal inference and experiment design keywords
        # These keywords strongly indicate causal reasoning problems that
        # were incorrectly routing to analogical (scoring 0.350).
        "confound",
        "confounder",
        "confounding",
        "do(",
        "pearl",
        "dag",
        "backdoor",
        "frontdoor",
        "randomize",
        "randomized",
        "experiment",
        "treatment",
        "control group",
        "observational",
        "rct",
        "correlation",
        "correlates",
        "causation",
    ],
    "probabilistic": [
        "probability",
        "probable",
        "probabilistic",
        "likely",
        "unlikely",
        "likelihood",
        "chance",
        "chances",
        "odds",
        "risk",
        "risky",
        "uncertain",
        "uncertainty",
        "bayesian",
        "bayes",
        "prior",
        "posterior",
        "estimate",
        "predict",
        "prediction",
        "confidence",
        "confident",
        "percent",
        "percentage",
        # BUG #2 FIX: Bayes theorem and medical test keywords
        # These keywords strongly indicate Bayesian calculations that
        # MUST NOT be routed to analogical. The problem statement shows
        # Bayes problems incorrectly scoring analogical (0.339) highest.
        "p(",
        "sensitivity",
        "specificity",
        "prevalence",
        "conditional",
        "given",
        "positive predictive",
        "negative predictive",
        "false positive",
        "false negative",
        "true positive",
        "true negative",
        # ENHANCED: Quantum physics and statistical mechanics
        "quantum",
        "wave function",
        "eigenstate",
        "superposition",
        "entropy",
        "thermodynamic",
        "statistical mechanics",
        "density matrix",
        "expectation value",
        "stochastic",
        "markov",
        "poisson",
        "gaussian",
        "distribution",
        "random variable",
        "variance",
        # ENHANCED: Mathematical probability notation
        "conditional probability",
        "joint probability",
        "marginal",
        "independence",
        "correlation",
        "expected value",
        "mean",
        "standard deviation",
    ],
    "multimodal": [
        # Image keywords
        "image",
        "images",
        "picture",
        "pictures",
        "photo",
        "photograph",
        "photos",
        "img",
        "jpeg",
        "jpg",
        "png",
        "gif",
        "bitmap",
        "pixel",
        # Visual keywords
        "visual",
        "visually",
        "see",
        "look",
        "looking",
        "view",
        "viewing",
        "show",
        "display",
        "appearance",
        "looks like",
        "what is this",
        "identify",
        "recognize",
        # Document/diagram keywords
        "diagram",
        "chart",
        "graph",
        "table",
        "figure",
        "figures",
        "plot",
        "document",
        "pdf",
        "scan",
        "scanned",
        "screenshot",
        "screen",
        # Video/audio keywords
        "video",
        "videos",
        "clip",
        "footage",
        "frame",
        "frames",
        "audio",
        "sound",
        "voice",
        "speech",
        "listen",
        "hear",
        # Analysis keywords
        "analyze this image",
        "describe this",
        "what do you see",
        "extract from",
        "read this",
        "ocr",
        "text in image",
        "caption",
        "describe",
        # File references
        "attached",
        "uploaded",
        "this file",
        "the file",
        "this document",
    ],
    # PERFORMANCE FIX: General/lightweight tool for simple queries
    # These queries should bypass complex reasoning entirely
    # ISSUE FIX: Added comprehensive philosophical/ethical keywords that were causing
    # queries to be misrouted to mathematical tools instead of general handler
    "general": [
        # Greetings and conversational
        "hello",
        "hi",
        "hey",
        "howdy",
        "greetings",
        "good morning",
        "good afternoon",
        "good evening",
        "thanks",
        "thank you",
        "goodbye",
        "bye",
        "okay",
        "ok",
        "sure",
        "yes",
        "no",
        "please",
        "sorry",
        # Identity/attribution patterns
        "who created",
        "who made",
        "who built",
        "who designed",
        "who developed",
        "your creator",
        "your maker",
        "created by",
        "made by",
        "what are you",
        "who are you",
        "introduce yourself",
        "tell me about yourself",
        # Philosophical paradoxes (should NOT use complex reasoning)
        "paradox",
        "this sentence is false",
        "liar paradox",
        "ship of theseus",
        "brain in a vat",
        "experience machine",
        "thought experiment",
        "philosophical",
        "dilemma",
        # Simple acknowledgments
        "i see",
        "got it",
        "understood",
        "makes sense",
        "interesting",
        "cool",
        "nice",
        "great",
        "awesome",
        # ISSUE FIX: Comprehensive philosophical/ethical terms
        # These were causing queries to route to math tools
        "ethical",
        "ethics",
        "moral",
        "morality",
        "philosophy",
        "hedonism",
        "hedonistic",
        "utilitarianism",
        "utilitarian",
        "consequentialism",
        "consequentialist",
        "deontological",
        "kantian",
        "virtue ethics",
        "virtue",
        "categorical imperative",
        "free will",
        "determinism",
        "compatibilism",
        "consciousness",
        "sentience",
        "qualia",
        "existential",
        "existentialism",
        "nihilism",
        "absurdism",
        "stoicism",
        "meaning of life",
        "trolley problem",
        "pleasure machine",
        "nozick",
        "rawlsian",
        "social contract",
        "veil of ignorance",
        "greatest good",
        "greatest happiness",
        "metaphysics",
        "epistemology",
        "ontology",
        "phenomenology",
        "dualism",
        "materialism",
        "physicalism",
        "panpsychism",
        "solipsism",
        "mind-body problem",
        "hard problem of consciousness",
        "philosophical zombie",
        "chinese room",
        "mary's room",
        "twin earth",
        "simulation",
        "omelas",
        "utility monster",
        "repugnant conclusion",
    ],
    # PHILOSOPHICAL REASONING - Deontic logic and ethical analysis
    # Routes queries to PhilosophicalReasoner for proper ethical/moral reasoning
    "philosophical": [
        # Deontic operators
        "permissible",
        "permissibility",
        "obligatory",
        "obligation",
        "forbidden",
        "prohibition",
        "duty",
        "ought",
        "should",
        "must not",
        # Ethical concepts
        "ethical",
        "ethics",
        "moral",
        "morality",
        "right",
        "wrong",
        "virtue",
        "virtuous",
        "vice",
        "value",
        "values",
        # Ethical frameworks
        "deontological",
        "deontology",
        "kantian",
        "kant",
        "categorical imperative",
        "consequentialist",
        "consequentialism",
        "utilitarian",
        "utilitarianism",
        "virtue ethics",
        "contractualist",
        "contractualism",
        "care ethics",
        # Common ethical problems
        "trolley problem",
        "trolley",
        "dilemma",
        "moral dilemma",
        "ethical dilemma",
        # Value concepts
        "harm",
        "benefit",
        "justice",
        "fairness",
        "rights",
        "autonomy",
        "consent",
        "welfare",
        "well-being",
        "dignity",
        "respect",
        # Deontic logic terms
        "deontic",
        "normative",
        "prescriptive",
        "imperative",
    ],
    # BUG #2 FIX: Mathematical tool keywords for numeric computation
    # These queries need numerical/computational tools, not analogical reasoning
    "mathematical": [
        # Basic computation
        "proof",
        "theorem",
        "lemma",
        "differentiate",
        "integrate",
        "derivative",
        "integral",
        "limit",
        "continuous",
        "differentiable",
        "converge",
        "series",
        "equation",
        "solve",
        "calculate",
        "compute",
        "=",
        "+",
        "-",
        "*",
        "/",
        "sum",
        "product",
        "factorial",
        "exponent",
        "logarithm",
        "log",
        "sqrt",
        "root",
        # Advanced math
        "matrix",
        "vector",
        "linear algebra",
        "eigenvalue",
        "eigenvector",
        "determinant",
        "optimization",
        "minimize",
        "maximize",
        "constraint",
    ],
    # CRITICAL #1 FIX: Language reasoning tool keywords for NLP tasks
    "language": [
        # Quantifier and scope
        "quantifier",
        "scope",
        "scope ambiguity",
        "quantifier scope",
        "every",
        "some",
        "most",
        "few",
        "all",
        "none",
        "any",
        # Parsing and syntax
        "parse",
        "parsing",
        "syntax",
        "syntactic",
        "semantic",
        "grammar",
        "constituent",
        "dependency",
        "parse tree",
        # Coreference and anaphora
        "pronoun",
        "coreference",
        "anaphora",
        "antecedent",
        "reference",
        # Formalization
        "formalize",
        "formalization",
        "first-order logic",
        "fol",
        "predicate logic",
        # NLP tasks
        "sentiment",
        "entity",
        "named entity",
        "ner",
        "disambiguation",
        "disambiguate",
        "word sense",
        "entailment",
        "linguistic",
        "natural language",
    ],
}


@dataclass
class SemanticMatch:
    """Result of semantic matching"""

    tool_name: str
    similarity_score: float
    keyword_matches: List[str] = field(default_factory=list)
    keyword_boost: float = 0.0
    combined_score: float = 0.0

    def __post_init__(self):
        # Combine embedding similarity with keyword boost
        self.combined_score = min(1.0, self.similarity_score + self.keyword_boost * 0.3)


class SemanticToolMatcher:
    """
    Matches queries to tools using semantic similarity and keyword patterns.

    This fixes the core issue where symbolic/analogical tools are never selected
    because there's no mechanism to understand what queries they're designed for.

    ISSUE #2 FIX: Added query embedding cache to prevent repeated computation
    of embeddings for the same query. Previously, query embeddings were computed
    inside the tool loop, causing N embedding computations per query (where N is
    the number of tools). Now embeddings are cached and computed once per query.
    """

    # Class-level singleton for embedding model (same as MultiTierFeatureExtractor)
    _shared_model = None
    _shared_model_lock = threading.Lock()
    _model_load_attempted = False

    # Pre-computed tool embeddings
    _tool_embeddings: Dict[str, np.ndarray] = {}
    _embeddings_computed = False

    # ISSUE #2 FIX: Query embedding cache to prevent repeated computations
    # Uses OrderedDict for proper LRU eviction - oldest accessed entries removed first
    _query_embedding_cache: OrderedDict = OrderedDict()
    _query_cache_lock = threading.Lock()
    _query_cache_max_size = 1000
    _query_cache_hits = 0
    _query_cache_misses = 0

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}

        self.embedding_weight = config.get("embedding_weight", 0.6)
        self.keyword_weight = config.get("keyword_weight", 0.4)
        self.min_similarity_threshold = config.get("min_similarity_threshold", 0.15)
        self.boost_factor = config.get("boost_factor", 0.5)  # How much to boost prior

        # Learning system integration (set externally for meta-learning boost)
        self.learning_system: Optional[Any] = None

        # Get or create shared embedding model
        self.embedding_model = self._get_shared_model()

        # Pre-compute tool embeddings on first use
        if self.embedding_model and not SemanticToolMatcher._embeddings_computed:
            self._compute_tool_embeddings()

    @classmethod
    def _get_shared_model(cls):
        """Get or create the shared embedding model (singleton pattern).

        PERFORMANCE FIX: Uses global model registry to ensure SentenceTransformer
        is loaded exactly ONCE per process and shared across all components.
        
        BUG #4 FIX: Added prominent logging to track model loads.
        If you see "Loading embedding model" more than ONCE in logs, 
        there's a singleton violation causing performance degradation.
        """
        if cls._shared_model is None and not cls._model_load_attempted:
            with cls._shared_model_lock:
                if cls._shared_model is None and not cls._model_load_attempted:
                    cls._model_load_attempted = True
                    
                    # BUG #4 FIX: Log prominently that we're loading the model
                    logger.info("=" * 60)
                    logger.info("[SemanticToolMatcher] Loading embedding model (ONE TIME ONLY)")
                    logger.info("=" * 60)

                    # First, try to use global model registry (process-wide singleton)
                    try:
                        from vulcan.models.model_registry import (
                            get_sentence_transformer,
                        )

                        cls._shared_model = get_sentence_transformer("all-MiniLM-L6-v2")
                        if cls._shared_model is not None:
                            logger.info(
                                "[SemanticToolMatcher] ✓ Model obtained from global registry"
                            )
                    except ImportError as e:
                        logger.debug(
                            f"[SemanticToolMatcher] Model registry not available: {e}"
                        )

                    # Fallback: Try MultiTierFeatureExtractor's model if registry didn't work
                    if cls._shared_model is None:
                        MultiTierFeatureExtractor = None
                        try:
                            from .tool_selector import MultiTierFeatureExtractor as MTFE

                            MultiTierFeatureExtractor = MTFE
                        except ImportError as e:
                            logger.debug(
                                f"[SemanticToolMatcher] Cannot import MultiTierFeatureExtractor: {e}"
                            )

                        if MultiTierFeatureExtractor is not None:
                            try:
                                shared = MultiTierFeatureExtractor._get_shared_model()
                                if shared is not None:
                                    cls._shared_model = shared
                                    logger.info(
                                        "[SemanticToolMatcher] ✓ Using shared model from MultiTierFeatureExtractor"
                                    )
                            except AttributeError as e:
                                logger.debug(
                                    f"[SemanticToolMatcher] MultiTierFeatureExtractor._get_shared_model() not available: {e}"
                                )
                            except Exception as e:
                                logger.debug(
                                    f"[SemanticToolMatcher] Error getting shared model: {e}"
                                )

                    # Last resort: load our own model directly if all else failed
                    if cls._shared_model is None:
                        try:
                            from sentence_transformers import SentenceTransformer
                            
                            # BUG #4 FIX: Log this prominently - it means registry failed
                            logger.warning(
                                "[SemanticToolMatcher] WARNING: Loading model directly (registry failed)"
                            )
                            cls._shared_model = SentenceTransformer("all-MiniLM-L6-v2")
                            logger.info(
                                "[SemanticToolMatcher] ✓ Embedding model loaded (direct fallback)"
                            )
                        except ImportError:
                            logger.warning(
                                "sentence-transformers not available for semantic matching"
                            )
                        except Exception as e:
                            logger.error(f"Failed to load embedding model: {e}")
                    
                    # BUG #4 FIX: Log completion prominently
                    if cls._shared_model is not None:
                        logger.info("=" * 60)
                        logger.info("[SemanticToolMatcher] ✓ Model loaded and cached")
                        logger.info("=" * 60)
                    else:
                        logger.error("=" * 60)
                        logger.error("[SemanticToolMatcher] ✗ FAILED to load embedding model")
                        logger.error("=" * 60)

        return cls._shared_model

    def _compute_tool_embeddings(self):
        """Pre-compute embeddings for all tool descriptions"""
        if not self.embedding_model:
            return

        with SemanticToolMatcher._shared_model_lock:
            if SemanticToolMatcher._embeddings_computed:
                return

            try:
                logger.info(
                    "[SemanticToolMatcher] Computing tool description embeddings..."
                )
                for tool_name, description in TOOL_DESCRIPTIONS.items():
                    embedding = self.embedding_model.encode(
                        description, show_progress_bar=False, normalize_embeddings=True
                    )
                    SemanticToolMatcher._tool_embeddings[tool_name] = embedding

                SemanticToolMatcher._embeddings_computed = True
                logger.info(
                    f"[SemanticToolMatcher] Computed embeddings for {len(TOOL_DESCRIPTIONS)} tools"
                )
            except Exception as e:
                logger.error(f"Failed to compute tool embeddings: {e}")

    @classmethod
    def _normalize_text(cls, text: str) -> str:
        """
        Normalize text for consistent cache key generation.

        CRITICAL FIX: Without normalization, "hello world" and "Hello World "
        generate different cache keys despite being semantically identical.
        This was causing 0% cache hit rate.

        Args:
            text: Text to normalize.

        Returns:
            Normalized text string.
        """
        # Strip whitespace and convert to lowercase
        normalized = text.strip().lower()
        # Collapse multiple whitespaces to single space
        normalized = " ".join(normalized.split())
        return normalized

    @classmethod
    def _get_query_embedding_cached(cls, query: str, model) -> Optional[np.ndarray]:
        """
        Get query embedding from cache or compute and cache it.

        ISSUE #2 FIX: Caches query embeddings to prevent repeated computation.
        This addresses the tool selection performance degradation where
        the same query was being embedded multiple times (once per tool).

        CRITICAL FIX: Now normalizes text before hashing to ensure cache hits.
        Without this, queries with different whitespace/casing would miss cache.
        
        PERFORMANCE FIX: Integrates circuit breaker to skip embeddings when
        performance degrades, preventing 30-second delays.

        Args:
            query: The query text to embed
            model: The embedding model to use

        Returns:
            Cached or newly computed embedding, or None on failure/circuit open
        """
        # Get circuit breaker once for reuse throughout method
        circuit_breaker = None
        if CIRCUIT_BREAKER_AVAILABLE:
            circuit_breaker = get_embedding_circuit_breaker()
        
        # PERFORMANCE FIX: Check circuit breaker first - skip embedding if circuit is open
        if circuit_breaker is not None and circuit_breaker.should_skip_embedding():
            logger.warning(
                "[SemanticToolMatcher] Circuit breaker OPEN - skipping embedding, "
                "using keyword-only matching"
            )
            return None
        
        # CRITICAL FIX: Normalize text before hashing for consistent cache keys
        normalized_query = cls._normalize_text(query)
        # Create cache key from normalized query hash (SHA-256 provides good collision resistance)
        cache_key = hashlib.sha256(
            normalized_query.encode(), usedforsecurity=False
        ).hexdigest()

        with cls._query_cache_lock:
            # Check cache first - move_to_end for LRU ordering
            if cache_key in cls._query_embedding_cache:
                cls._query_cache_hits += 1
                # Move to end to mark as recently used (proper LRU behavior)
                cls._query_embedding_cache.move_to_end(cache_key)
                logger.debug(
                    f"[SemanticToolMatcher] Cache HIT: {cache_key[:8]}... "
                    f"(hits={cls._query_cache_hits}, misses={cls._query_cache_misses})"
                )
                return cls._query_embedding_cache[cache_key]

            cls._query_cache_misses += 1
            logger.debug(
                f"[SemanticToolMatcher] Cache MISS: {cache_key[:8]}... "
                f"(hits={cls._query_cache_hits}, misses={cls._query_cache_misses})"
            )

        # Compute embedding outside lock - track time for circuit breaker
        start_time = time.perf_counter()
        try:
            embedding = model.encode(
                query,  # Use original query for embedding (preserves semantic nuances)
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            
            # Record successful latency to circuit breaker
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            if circuit_breaker is not None:
                circuit_breaker.record_latency(elapsed_ms)

            # Cache the result
            with cls._query_cache_lock:
                # LRU eviction: remove least recently used entry if at capacity
                if len(cls._query_embedding_cache) >= cls._query_cache_max_size:
                    # popitem(last=False) removes the oldest (first) entry for LRU
                    cls._query_embedding_cache.popitem(last=False)

                cls._query_embedding_cache[cache_key] = embedding
                logger.debug(
                    f"[SemanticToolMatcher] Cached embedding: {cache_key[:8]}... "
                    f"(cache_size={len(cls._query_embedding_cache)}/{cls._query_cache_max_size}, "
                    f"time={elapsed_ms:.0f}ms)"
                )

            return embedding
        except Exception as e:
            # Record failure to circuit breaker
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            if circuit_breaker is not None:
                circuit_breaker.record_failure()
            
            logger.warning(
                f"[SemanticToolMatcher] Failed to compute query embedding after {elapsed_ms:.0f}ms: {e}"
            )
            return None

    @classmethod
    def get_cache_stats(cls) -> Dict[str, Any]:
        """Get query embedding cache statistics for monitoring."""
        with cls._query_cache_lock:
            total_requests = cls._query_cache_hits + cls._query_cache_misses
            hit_rate = (
                cls._query_cache_hits / total_requests if total_requests > 0 else 0.0
            )
            return {
                "size": len(cls._query_embedding_cache),
                "max_size": cls._query_cache_max_size,
                "hits": cls._query_cache_hits,
                "misses": cls._query_cache_misses,
                "hit_rate": hit_rate,
            }

    @classmethod
    def clear_query_cache(cls) -> None:
        """Clear the query embedding cache."""
        with cls._query_cache_lock:
            cls._query_embedding_cache.clear()
            logger.info("[SemanticToolMatcher] Query embedding cache cleared")

    def match_query(
        self, query: str, available_tools: Optional[List[str]] = None
    ) -> Dict[str, SemanticMatch]:
        """
        Match query to tools using semantic similarity and keyword patterns.

        ISSUE #2 FIX: Query embedding is now computed ONCE using the cache,
        instead of being recomputed for every tool in the loop.

        BUG #1 FIX (0.500 Bug): Added explicit greeting detection to ensure
        simple greetings like "Hello" route to 'general' tool instead of 
        probabilistic, which would return confusing "mean prediction 0.500".

        Args:
            query: The input query/problem text
            available_tools: Optional list of available tools (defaults to all)

        Returns:
            Dictionary mapping tool names to SemanticMatch objects
        """
        if available_tools is None:
            available_tools = list(TOOL_DESCRIPTIONS.keys())

        results = {}
        query_lower = query.lower().strip()
        
        # BUG #1 FIX: Fast-path for simple greetings
        # These patterns MUST route to 'general' tool - NOT probabilistic
        # Otherwise users get absurd responses like "mean prediction 0.500"
        SIMPLE_GREETING_PATTERNS = {
            'hello', 'hi', 'hey', 'howdy', 'greetings', 'yo',
            'good morning', 'good afternoon', 'good evening', 'good night',
            'thanks', 'thank you', 'thx', 'bye', 'goodbye', 'see you',
            "what's up", 'sup', 'how are you', "how's it going",
        }
        
        # Check if query is a simple greeting:
        # 1. Exact match with known patterns
        # 2. Starts with a greeting and total length is short (greeting + up to 15 extra chars)
        #    e.g., "hello there" (11 chars) = "hello" (5) + 6 extra chars < 15
        is_simple_greeting = (
            query_lower in SIMPLE_GREETING_PATTERNS or
            any(query_lower.startswith(p + ' ') and len(query_lower) <= len(p) + 15 
                for p in SIMPLE_GREETING_PATTERNS) or
            any(query_lower.startswith(p + '!') and len(query_lower) <= len(p) + 5
                for p in SIMPLE_GREETING_PATTERNS) or
            any(query_lower.startswith(p + '.') and len(query_lower) <= len(p) + 5
                for p in SIMPLE_GREETING_PATTERNS) or
            any(query_lower.startswith(p + ',') and len(query_lower) <= len(p) + 15
                for p in SIMPLE_GREETING_PATTERNS)
        )
        
        # =====================================================================
        # BUG #9 FIX: Task type detection takes PRIORITY over domain keywords
        # Problem: "Write a sonnet about quantum entanglement" was routing to
        #          mathematical/symbolic due to "quantum" keyword, but the
        #          TASK TYPE is creative writing ("Write a sonnet").
        # Fix: Detect task type first (verbs like "write", "compose", etc.)
        #      and override domain-based routing when task type is clear.
        # =====================================================================
        
        # Creative task indicators (verbs and nouns that indicate creative output)
        CREATIVE_TASK_VERBS = {
            'write', 'compose', 'create', 'craft', 'generate', 'draft',
            'author', 'pen', 'produce', 'design', 'invent', 'imagine',
        }
        CREATIVE_TASK_NOUNS = {
            'poem', 'sonnet', 'haiku', 'story', 'essay', 'article',
            'song', 'lyrics', 'script', 'novel', 'play', 'limerick',
            'verse', 'prose', 'fiction', 'narrative', 'tale',
        }
        
        # Check if this is a creative task (task type takes priority over domain)
        is_creative_task = False
        words = query_lower.split()
        if words:
            # Check if query starts with a creative verb
            first_word = words[0].rstrip(',.!?')
            is_creative_task = first_word in CREATIVE_TASK_VERBS
            
            # Also check for creative nouns in the query
            if not is_creative_task:
                is_creative_task = any(noun in query_lower for noun in CREATIVE_TASK_NOUNS)
        
        # BUG #10 FIX: Mathematical task detection to prevent ethical override
        # Problem: "Optimize welfare function" was routing to philosophical due
        #          to "welfare" keyword, but this is an optimization problem.
        MATHEMATICAL_TASK_VERBS = {
            'optimize', 'maximize', 'minimize', 'calculate', 'compute',
            'solve', 'derive', 'integrate', 'differentiate', 'evaluate',
            'factor', 'simplify', 'expand', 'reduce',
        }
        is_math_task = False
        if words:
            first_word = words[0].rstrip(',.!?')
            is_math_task = first_word in MATHEMATICAL_TASK_VERBS
        
        # =====================================================================
        # CRITICAL #1 FIX: Language/NLP task detection
        # Problem: "Every engineer reviewed a document" was routing to symbolic
        #          because both have "every" keyword. But this is a scope/parsing task.
        # Fix: Detect NLP tasks that need language reasoning (parsing, scope, etc.)
        # =====================================================================
        NLP_TASK_KEYWORDS = {
            'parse', 'parsing', 'scope', 'quantifier', 'formalize', 'formalization',
            'disambiguate', 'disambiguation', 'coreference', 'anaphora', 'pronoun',
            'semantic', 'sentiment', 'entailment', 'linguistic',
        }
        NLP_SENTENCE_PATTERNS = {
            'reviewed', 'document', 'engineer', 'student', 'professor',
            # Common NLP benchmark sentence patterns
        }
        is_language_task = any(kw in query_lower for kw in NLP_TASK_KEYWORDS)
        
        # Also detect if this looks like a sentence parsing/scope task
        # (sentences with quantifiers + verbs + objects)
        if not is_language_task:
            has_quantifier = any(q in query_lower for q in ['every', 'some', 'most', 'few', 'all', 'none'])
            has_verb_object = any(v in query_lower for v in ['reviewed', 'read', 'met', 'saw', 'visited'])
            if has_quantifier and has_verb_object and 'mortal' not in query_lower:
                # This looks like a scope/parsing task, not a syllogism
                is_language_task = True
                logger.debug(
                    f"[SemanticToolMatcher] CRITICAL#1 FIX: Detected LANGUAGE task - "
                    f"quantifier+verb pattern suggests NLP parsing task: {query_lower[:50]}..."
                )
        
        if is_creative_task:
            logger.debug(
                f"[SemanticToolMatcher] BUG#9 FIX: Detected CREATIVE task - "
                f"task type overrides domain keywords for query: {query_lower[:50]}..."
            )
        
        if is_math_task:
            logger.debug(
                f"[SemanticToolMatcher] BUG#10 FIX: Detected MATHEMATICAL task - "
                f"task type overrides ethical keywords for query: {query_lower[:50]}..."
            )

        # ISSUE #2 FIX: Pre-compute query embedding ONCE before the tool loop
        # Previously this was done inside the loop, causing N embedding computations
        query_embedding = None
        if self.embedding_model and SemanticToolMatcher._embeddings_computed:
            query_embedding = SemanticToolMatcher._get_query_embedding_cached(
                query, self.embedding_model
            )

        for tool_name in available_tools:
            # 1. Keyword matching (fast, always available)
            keyword_matches = []
            keyword_boost = 0.0

            if tool_name in TOOL_KEYWORDS:
                for keyword in TOOL_KEYWORDS[tool_name]:
                    # BUG FIX: Use word boundary matching to prevent false positives
                    # like "each" matching in "teacher" or "or" matching in "doctor"
                    # For multi-word keywords (e.g., "is to", "if then"), check if 
                    # keyword appears as a complete phrase
                    keyword_lower = keyword.lower()
                    if ' ' in keyword_lower:
                        # Multi-word keyword: simple substring match is fine
                        if keyword_lower in query_lower:
                            keyword_matches.append(keyword)
                            keyword_boost += 0.1
                    else:
                        # Single word: require word boundary to prevent false positives
                        pattern = r'\b' + re.escape(keyword_lower) + r'\b'
                        if re.search(pattern, query_lower):
                            keyword_matches.append(keyword)
                            keyword_boost += 0.1

                # Cap keyword boost at 0.5
                keyword_boost = min(0.5, keyword_boost)
            
            # BUG #1 FIX: Apply dominant boost for 'general' tool on greetings
            # This prevents greetings from being sent to probabilistic (which returns 0.5)
            if is_simple_greeting and tool_name == 'general':
                keyword_boost = 0.8  # Very strong boost - must win over probabilistic
                keyword_matches.append('greeting_fast_path')
                logger.debug(f"[SemanticToolMatcher] Applied greeting boost to 'general' for query: {query_lower[:30]}...")
            
            # BUG #1 FIX: Penalize probabilistic for simple greetings
            # Probabilistic tool should NOT handle greetings - it returns "mean prediction 0.500"
            if is_simple_greeting and tool_name == 'probabilistic':
                keyword_boost = max(0.0, keyword_boost - 0.3)  # Significant penalty
            
            # =====================================================================
            # BUG #9 FIX: Creative task type overrides domain-based routing
            # "Write a sonnet about quantum" should go to general/creative, not symbolic
            # =====================================================================
            if is_creative_task:
                if tool_name == 'general':
                    # Boost general tool significantly for creative tasks
                    keyword_boost = max(keyword_boost, 0.7)
                    keyword_matches.append('creative_task_override')
                elif tool_name in ('symbolic', 'probabilistic', 'causal'):
                    # Penalize reasoning tools for creative tasks
                    keyword_boost = max(0.0, keyword_boost - 0.4)
            
            # =====================================================================
            # BUG #10 FIX: Mathematical task type overrides ethical keyword routing
            # "Calculate the derivative" should go to mathematical tool for computation
            # "Prove the theorem" should go to symbolic tool for formal proofs
            # =====================================================================
            if is_math_task:
                if tool_name == 'mathematical':
                    # Boost mathematical tool significantly for computation tasks
                    keyword_boost = max(keyword_boost, 0.65)  # Higher than symbolic
                    keyword_matches.append('math_computation_override')
                elif tool_name == 'symbolic':
                    # Also boost symbolic but slightly less for math proofs
                    keyword_boost = max(keyword_boost, 0.55)
                    keyword_matches.append('math_task_override')
                elif tool_name in ('philosophical', 'analogical'):
                    # Penalize philosophical for math tasks
                    keyword_boost = max(0.0, keyword_boost - 0.3)
            
            # =====================================================================
            # CRITICAL #1 FIX: Language task type overrides symbolic keyword matching
            # "Every engineer reviewed a document" should go to language tool
            # not symbolic (which has "every" for syllogisms)
            # =====================================================================
            if is_language_task:
                if tool_name == 'language':
                    # Boost language tool significantly for NLP tasks
                    keyword_boost = max(keyword_boost, 0.7)
                    keyword_matches.append('language_task_override')
                elif tool_name == 'symbolic':
                    # Penalize symbolic for NLP tasks (unless it's a syllogism)
                    if 'mortal' not in query_lower and 'therefore' not in query_lower:
                        keyword_boost = max(0.0, keyword_boost - 0.3)

            # 2. Embedding similarity (now uses pre-computed query embedding)
            similarity_score = 0.0

            if (
                query_embedding is not None
                and tool_name in SemanticToolMatcher._tool_embeddings
            ):
                try:
                    tool_embedding = SemanticToolMatcher._tool_embeddings[tool_name]

                    # Cosine similarity (embeddings are normalized)
                    similarity_score = float(np.dot(query_embedding, tool_embedding))
                    similarity_score = max(0.0, similarity_score)  # Clamp to [0, 1]
                except Exception as e:
                    logger.warning(f"Embedding similarity failed for {tool_name}: {e}")

            # 3. Create match result
            results[tool_name] = SemanticMatch(
                tool_name=tool_name,
                similarity_score=similarity_score,
                keyword_matches=keyword_matches,
                keyword_boost=keyword_boost,
            )

        return results

    def boost_prior(
        self,
        prior_probs: Dict[str, float],
        query: str,
        available_tools: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Boost prior probabilities based on semantic matching.

        This is the key integration point with BayesianMemoryPrior.
        
        PERFORMANCE FIX: If circuit breaker is open (embeddings too slow),
        falls back to keyword-only matching for fast routing.

        Args:
            prior_probs: Original prior probabilities from BayesianMemoryPrior
            query: The input query/problem text
            available_tools: Optional list of available tools

        Returns:
            Boosted prior probabilities
        """
        if available_tools is None:
            available_tools = list(prior_probs.keys())

        # PERFORMANCE FIX: Check circuit breaker - use keyword-only if embeddings are slow
        if CIRCUIT_BREAKER_AVAILABLE:
            circuit_breaker = get_embedding_circuit_breaker()
            if circuit_breaker is not None and circuit_breaker.should_skip_embedding():
                logger.info(
                    "[SemanticToolMatcher] Circuit breaker OPEN - using keyword-only boost"
                )
                return self.boost_prior_keywords_only(prior_probs, query, available_tools)

        # Get semantic matches (uses embeddings)
        matches = self.match_query(query, available_tools)

        # Compute boost for each tool
        boosted = {}
        total_boost = 0.0

        for tool_name in available_tools:
            original_prob = prior_probs.get(tool_name, 0.0)

            if tool_name in matches:
                match = matches[tool_name]

                # Only boost if there's meaningful similarity
                if match.combined_score >= self.min_similarity_threshold:
                    boost = match.combined_score * self.boost_factor
                    boosted[tool_name] = original_prob + boost
                    total_boost += boost

                    logger.debug(
                        f"[SemanticToolMatcher] {tool_name}: "
                        f"sim={match.similarity_score:.3f}, "
                        f"kw={match.keyword_boost:.3f}, "
                        f"boost={boost:.3f}, "
                        f"keywords={match.keyword_matches[:3]}"
                    )
                else:
                    boosted[tool_name] = original_prob
            else:
                boosted[tool_name] = original_prob

        # Meta-learning boost: Use learned task patterns if available
        if self.learning_system:
            try:
                if self.learning_system.continual_learner:
                    # Get task embedding for this query
                    task_id = self.learning_system.continual_learner.task_detector.detect_task(
                        {"text": query, "type": "tool_selection"}
                    )

                    # If we've seen similar tasks, boost based on historical performance
                    if task_id in self.learning_system.continual_learner.task_info:
                        task_info = self.learning_system.continual_learner.task_info[
                            task_id
                        ]
                        if task_info.metadata and "best_tools" in task_info.metadata:
                            for tool in task_info.metadata["best_tools"]:
                                if tool in boosted:
                                    boosted[
                                        tool
                                    ] *= 1.2  # 20% boost for historically good tools
                            logger.debug(
                                f"[MetaBoost] Applied historical boost for task {task_id}"
                            )
            except Exception as e:
                logger.debug(f"[MetaBoost] Skipped: {e}")

        # Normalize to sum to 1.0
        total = sum(boosted.values())
        if total > 0:
            boosted = {k: v / total for k, v in boosted.items()}

        return boosted

    def boost_prior_keywords_only(
        self,
        prior_probs: Dict[str, float],
        query: str,
        available_tools: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        FAST keyword-only boost - NO EMBEDDINGS.
        
        This method bypasses the slow embedding computation (6-30 seconds)
        and uses only keyword matching for tool selection. Use this when:
        - Circuit breaker is open
        - VULCAN_DISABLE_SEMANTIC_MATCHING=1
        - Need fast fallback routing
        
        Args:
            prior_probs: Original prior probabilities
            query: The input query/problem text
            available_tools: Optional list of available tools
            
        Returns:
            Boosted prior probabilities (keyword-only, no embeddings)
        """
        if available_tools is None:
            available_tools = list(prior_probs.keys())
        
        query_lower = query.lower()
        boosted = {}
        
        for tool_name in available_tools:
            original_prob = prior_probs.get(tool_name, 0.0)
            keyword_boost = 0.0
            
            # Get keywords for this tool
            if tool_name in TOOL_KEYWORDS:
                keywords = TOOL_KEYWORDS[tool_name]
                matches = []
                for kw in keywords:
                    kw_lower = kw.lower()
                    if ' ' in kw_lower:
                        # Multi-word keyword: simple substring match is fine
                        if kw_lower in query_lower:
                            matches.append(kw)
                    else:
                        # Single word: require word boundary to prevent false positives
                        pattern = r'\b' + re.escape(kw_lower) + r'\b'
                        if re.search(pattern, query_lower):
                            matches.append(kw)
                
                if matches:
                    # Scale boost by number of keyword matches
                    keyword_boost = min(0.3, len(matches) * 0.1)
                    logger.debug(
                        f"[KeywordBoost] {tool_name}: matches={matches}, boost={keyword_boost:.3f}"
                    )
            
            boosted[tool_name] = original_prob + keyword_boost
        
        # Normalize
        total = sum(boosted.values())
        if total > 0:
            boosted = {k: v / total for k, v in boosted.items()}
        
        return boosted

    def get_best_match(
        self, query: str, available_tools: Optional[List[str]] = None
    ) -> Tuple[str, SemanticMatch]:
        """Get the best matching tool for a query"""
        matches = self.match_query(query, available_tools)

        if not matches:
            return ("general", SemanticMatch(tool_name="general", similarity_score=0.0))

        best_tool = max(matches.items(), key=lambda x: x[1].combined_score)
        return best_tool

    def diagnose_query(self, query: str) -> str:
        """
        Diagnostic method to explain why a query matches certain tools.
        Useful for debugging tool selection issues.
        """
        matches = self.match_query(query)

        lines = [
            f"Query: {query[:100]}...",
            "",
            "Tool Matches:",
            "-" * 60,
        ]

        sorted_matches = sorted(
            matches.items(), key=lambda x: x[1].combined_score, reverse=True
        )

        for tool_name, match in sorted_matches:
            lines.append(
                f"  {tool_name:15s}: "
                f"combined={match.combined_score:.3f} "
                f"(embed={match.similarity_score:.3f}, "
                f"kw={match.keyword_boost:.3f})"
            )
            if match.keyword_matches:
                lines.append(
                    f"                   keywords: {match.keyword_matches[:5]}"
                )

        return "\n".join(lines)


# =============================================================================
# INTEGRATION HELPER
# =============================================================================


def create_semantic_matcher(
    config: Optional[Dict[str, Any]] = None,
) -> SemanticToolMatcher:
    """Factory function to create a SemanticToolMatcher"""
    return SemanticToolMatcher(config)


# =============================================================================
# PATCH INSTRUCTIONS FOR memory_prior.py
# =============================================================================

PATCH_INSTRUCTIONS = """
================================================================================
PATCH INSTRUCTIONS FOR memory_prior.py
================================================================================

Add the following code to integrate semantic matching with the prior:

1. Add import at the top of memory_prior.py:
   
   from .semantic_tool_matcher import SemanticToolMatcher

2. Modify BayesianMemoryPrior.__init__() to add:

   # Semantic tool matcher for query-to-tool matching
   self.semantic_matcher = SemanticToolMatcher()

3. Modify BayesianMemoryPrior.compute_prior() to apply semantic boost:

   Replace the final section (after computing the prior) with:

   # Apply semantic boost based on query content
   if hasattr(self, 'semantic_matcher') and context and 'query' in context:
       query_text = str(context.get('query', ''))
       if query_text and len(query_text) > 10:
           prior.tool_probs = self.semantic_matcher.boost_prior(
               prior.tool_probs,
               query_text,
               available_tools
           )
           # Update most likely tool after boost
           if prior.tool_probs:
               prior.most_likely_tool = max(
                   prior.tool_probs.items(),
                   key=lambda x: x[1]
               )[0]

   return prior

================================================================================
"""


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    # Quick test of the semantic matcher
    print("Testing SemanticToolMatcher...")

    matcher = SemanticToolMatcher()

    test_queries = [
        "All philosophers are mortal. Socrates is a philosopher. Therefore Socrates is mortal.",
        "Given the premises: All humans are mortal. Aristotle is human. Conclude whether Aristotle is mortal.",
        "The immune system is like a military defense network. How can we extend this analogy?",
        "Hospital is to patients as school is to what?",
        "What is the probability of rain tomorrow given the forecast?",
        "What caused the economic recession in 2008?",
    ]

    for query in test_queries:
        print("\n" + "=" * 70)
        print(matcher.diagnose_query(query))
