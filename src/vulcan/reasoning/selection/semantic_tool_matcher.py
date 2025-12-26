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

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


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
    """
}


# =============================================================================
# KEYWORD BOOSTING - Fast pattern matching without embeddings
# =============================================================================

TOOL_KEYWORDS = {
    "symbolic": [
        # Syllogism patterns
        "all", "every", "each", "any", "no", "some",
        "mortal", "philosopher", "socrates", "human", "animal",
        # Logic keywords
        "premise", "conclusion", "therefore", "hence", "thus",
        "deduce", "deduction", "infer", "inference",
        "prove", "proof", "theorem", "axiom",
        "valid", "invalid", "sound", "unsound",
        "implies", "if then", "if-then", "implication",
        "forall", "exists", "∀", "∃",
        "modus ponens", "modus tollens",
        "syllogism", "logic", "logical",
        "entailment", "entails",
        "formal", "first-order", "fol",
    ],
    
    "analogical": [
        # Analogy patterns
        "is like", "is to", "as a", "analogous",
        "analogy", "analogies", "metaphor",
        "similar to", "corresponds to",
        "compare", "comparison",
        "just as", "so too", "likewise",
        "resembles", "mirrors", "parallels",
        "mapping", "transfer", "pattern",
        # Common analogy domains
        "relationship", "proportion",
        "example", "instance",
    ],
    
    "causal": [
        "cause", "causes", "caused",
        "effect", "effects", "affect", "affects",
        "because", "due to", "leads to", "results in",
        "consequence", "consequences",
        "counterfactual", "what if", "would have",
        "intervention", "intervene",
        "mechanism", "pathway",
        "why", "reason", "factor",
        "impact", "influence",
    ],
    
    "probabilistic": [
        "probability", "probable", "probabilistic",
        "likely", "unlikely", "likelihood",
        "chance", "chances", "odds",
        "risk", "risky",
        "uncertain", "uncertainty",
        "bayesian", "bayes",
        "prior", "posterior",
        "estimate", "predict", "prediction",
        "confidence", "confident",
        "percent", "percentage",
    ],
    
    "multimodal": [
        # Image keywords
        "image", "images", "picture", "pictures", "photo", "photograph", "photos",
        "img", "jpeg", "jpg", "png", "gif", "bitmap", "pixel",
        # Visual keywords  
        "visual", "visually", "see", "look", "looking", "view", "viewing", "show", "display",
        "appearance", "looks like", "what is this", "identify", "recognize",
        # Document/diagram keywords
        "diagram", "chart", "graph", "table", "figure", "figures", "plot",
        "document", "pdf", "scan", "scanned", "screenshot", "screen",
        # Video/audio keywords
        "video", "videos", "clip", "footage", "frame", "frames",
        "audio", "sound", "voice", "speech", "listen", "hear",
        # Analysis keywords
        "analyze this image", "describe this", "what do you see", "extract from",
        "read this", "ocr", "text in image", "caption", "describe",
        # File references
        "attached", "uploaded", "this file", "the file", "this document",
    ]
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
    """
    
    # Class-level singleton for embedding model (same as MultiTierFeatureExtractor)
    _shared_model = None
    _shared_model_lock = threading.Lock()
    _model_load_attempted = False
    
    # Pre-computed tool embeddings
    _tool_embeddings: Dict[str, np.ndarray] = {}
    _embeddings_computed = False
    
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
        """Get or create the shared embedding model (singleton pattern)"""
        if cls._shared_model is None and not cls._model_load_attempted:
            with cls._shared_model_lock:
                if cls._shared_model is None and not cls._model_load_attempted:
                    cls._model_load_attempted = True
                    try:
                        from sentence_transformers import SentenceTransformer
                        logger.info("[SemanticToolMatcher] Loading embedding model...")
                        cls._shared_model = SentenceTransformer("all-MiniLM-L6-v2")
                        logger.info("[SemanticToolMatcher] Embedding model loaded")
                    except ImportError:
                        logger.warning("sentence-transformers not available for semantic matching")
                    except Exception as e:
                        logger.error(f"Failed to load embedding model: {e}")
        
        return cls._shared_model
    
    def _compute_tool_embeddings(self):
        """Pre-compute embeddings for all tool descriptions"""
        if not self.embedding_model:
            return
        
        with SemanticToolMatcher._shared_model_lock:
            if SemanticToolMatcher._embeddings_computed:
                return
            
            try:
                logger.info("[SemanticToolMatcher] Computing tool description embeddings...")
                for tool_name, description in TOOL_DESCRIPTIONS.items():
                    embedding = self.embedding_model.encode(
                        description, 
                        show_progress_bar=False,
                        normalize_embeddings=True
                    )
                    SemanticToolMatcher._tool_embeddings[tool_name] = embedding
                
                SemanticToolMatcher._embeddings_computed = True
                logger.info(f"[SemanticToolMatcher] Computed embeddings for {len(TOOL_DESCRIPTIONS)} tools")
            except Exception as e:
                logger.error(f"Failed to compute tool embeddings: {e}")
    
    def match_query(
        self,
        query: str,
        available_tools: Optional[List[str]] = None
    ) -> Dict[str, SemanticMatch]:
        """
        Match query to tools using semantic similarity and keyword patterns.
        
        Args:
            query: The input query/problem text
            available_tools: Optional list of available tools (defaults to all)
        
        Returns:
            Dictionary mapping tool names to SemanticMatch objects
        """
        if available_tools is None:
            available_tools = list(TOOL_DESCRIPTIONS.keys())
        
        results = {}
        query_lower = query.lower()
        
        for tool_name in available_tools:
            # 1. Keyword matching (fast, always available)
            keyword_matches = []
            keyword_boost = 0.0
            
            if tool_name in TOOL_KEYWORDS:
                for keyword in TOOL_KEYWORDS[tool_name]:
                    if keyword.lower() in query_lower:
                        keyword_matches.append(keyword)
                        keyword_boost += 0.1  # Accumulate boost per match
                
                # Cap keyword boost at 0.5
                keyword_boost = min(0.5, keyword_boost)
            
            # 2. Embedding similarity (slower, higher quality)
            similarity_score = 0.0
            
            if (self.embedding_model and 
                SemanticToolMatcher._embeddings_computed and
                tool_name in SemanticToolMatcher._tool_embeddings):
                try:
                    query_embedding = self.embedding_model.encode(
                        query,
                        show_progress_bar=False,
                        normalize_embeddings=True
                    )
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
        available_tools: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Boost prior probabilities based on semantic matching.
        
        This is the key integration point with BayesianMemoryPrior.
        
        Args:
            prior_probs: Original prior probabilities from BayesianMemoryPrior
            query: The input query/problem text
            available_tools: Optional list of available tools
        
        Returns:
            Boosted prior probabilities
        """
        if available_tools is None:
            available_tools = list(prior_probs.keys())
        
        # Get semantic matches
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
                        {'text': query, 'type': 'tool_selection'}
                    )
                    
                    # If we've seen similar tasks, boost based on historical performance
                    if task_id in self.learning_system.continual_learner.task_info:
                        task_info = self.learning_system.continual_learner.task_info[task_id]
                        if task_info.metadata and 'best_tools' in task_info.metadata:
                            for tool in task_info.metadata['best_tools']:
                                if tool in boosted:
                                    boosted[tool] *= 1.2  # 20% boost for historically good tools
                            logger.debug(f"[MetaBoost] Applied historical boost for task {task_id}")
            except Exception as e:
                logger.debug(f"[MetaBoost] Skipped: {e}")
        
        # Normalize to sum to 1.0
        total = sum(boosted.values())
        if total > 0:
            boosted = {k: v / total for k, v in boosted.items()}
        
        return boosted
    
    def get_best_match(
        self,
        query: str,
        available_tools: Optional[List[str]] = None
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
            matches.items(),
            key=lambda x: x[1].combined_score,
            reverse=True
        )
        
        for tool_name, match in sorted_matches:
            lines.append(
                f"  {tool_name:15s}: "
                f"combined={match.combined_score:.3f} "
                f"(embed={match.similarity_score:.3f}, "
                f"kw={match.keyword_boost:.3f})"
            )
            if match.keyword_matches:
                lines.append(f"                   keywords: {match.keyword_matches[:5]}")
        
        return "\n".join(lines)


# =============================================================================
# INTEGRATION HELPER
# =============================================================================

def create_semantic_matcher(config: Optional[Dict[str, Any]] = None) -> SemanticToolMatcher:
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
