"""
Tool Selector - Main Orchestrator for Tool Selection System

Integrates all components to provide intelligent, safe, and efficient
tool selection for reasoning problems.

This version has been upgraded with full implementations for all previously
stubbed components, providing a complete, functional system.

Fixed with interruptible background threads.
"""

import json
import logging
import pickle  # SECURITY: Internal data only, never deserialize untrusted data
import re
import threading
import time
import uuid
from collections import defaultdict, deque, OrderedDict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import hashlib

import numpy as np

# Import routing keywords to avoid duplication
from vulcan.routing.llm_router import (
    ANALOGICAL_KEYWORDS,
    CAUSAL_KEYWORDS,
    MATHEMATICAL_KEYWORDS,
    LOGIC_KEYWORDS,
)

# CRITICAL FIX: Define logger BEFORE any imports that might fail
logger = logging.getLogger(__name__)

# --- Dependencies for Full Implementations ---
try:
    import lightgbm as lgb

    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    logger.warning(
        "LightGBM not available. StochasticCostModel will use a simple average."
    )

try:
    from sentence_transformers import SentenceTransformer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning(
        "sentence-transformers not available. MultiTierFeatureExtractor will have limited semantic capabilities."
    )

try:
    from sklearn.isotonic import IsotonicRegression

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning(
        "scikit-learn not available. ToolConfidenceCalibrator will be disabled."
    )

try:
    from scipy.stats import ks_2samp

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("SciPy not available. DistributionMonitor will be disabled.")

# CRITICAL FIX: Complete import section with proper module references
try:
    # Use relative imports within the selection package
    from .admission_control import AdmissionControlIntegration, RequestPriority
    from .memory_prior import BayesianMemoryPrior, PriorType
    from .portfolio_executor import (
        ExecutionMonitor,
        ExecutionStrategy,
        PortfolioExecutor,
    )
    from .safety_governor import SafetyContext, SafetyGovernor, SafetyLevel
    from .selection_cache import SelectionCache
    from .utility_model import UtilityModel
    from .warm_pool import WarmStartPool

    IMPORTS_SUCCESSFUL = True
    SELECTION_IMPORTS_SUCCESSFUL = True
    logger.info("Selection support components imported successfully")
except ImportError as e:
    logger.error(f"Selection support components not available: {e}")
    IMPORTS_SUCCESSFUL = False
    SELECTION_IMPORTS_SUCCESSFUL = False
    # Create placeholders
    AdmissionControlIntegration = None
    RequestPriority = None
    BayesianMemoryPrior = None
    PriorType = None
    PortfolioExecutor = None
    ExecutionStrategy = None
    ExecutionMonitor = None
    SafetyGovernor = None
    SafetyContext = None
    SafetyLevel = None
    SelectionCache = None
    WarmStartPool = None
    UtilityModel = None

# CRITICAL FIX: Bandit import is separate - it might not exist
try:
    from ..contextual_bandit import (
        AdaptiveBanditOrchestrator,
        BanditAction,
        BanditContext,
        BanditFeedback,
    )

    BANDIT_AVAILABLE = True
    logger.info("Contextual bandit imported successfully")
except ImportError as e:
    logger.warning(f"Contextual bandit not available: {e}")
    BANDIT_AVAILABLE = False
    # Create placeholders
    AdaptiveBanditOrchestrator = None
    BanditContext = None
    BanditFeedback = None
    BanditAction = None


# Import outcome bridge for implicit feedback recording
# This enables learning from tool selection outcomes
try:
    from ...curiosity_engine.outcome_bridge import record_query_outcome
    OUTCOME_BRIDGE_AVAILABLE = True
    logger.info("Outcome bridge imported for implicit feedback recording")
except ImportError:
    try:
        from vulcan.curiosity_engine.outcome_bridge import record_query_outcome
        OUTCOME_BRIDGE_AVAILABLE = True
        logger.info("Outcome bridge imported for implicit feedback recording")
    except ImportError:
        record_query_outcome = None
        OUTCOME_BRIDGE_AVAILABLE = False
        logger.debug("Outcome bridge not available - implicit feedback disabled")


# Import LLM Router for tool classification
try:
    from vulcan.routing.llm_router import get_llm_router, RoutingDecision
    LLM_ROUTER_AVAILABLE = True
    logger.info("LLMQueryRouter imported for LLM-based tool selection")
except ImportError as e:
    logger.warning(f"LLMQueryRouter not available: {e}")
    LLM_ROUTER_AVAILABLE = False
    get_llm_router = None
    RoutingDecision = None


# Import mathematical verification for accuracy feedback
# This enables learning from mathematical reasoning accuracy
try:
    from ..mathematical_verification import (
        MathematicalVerificationEngine,
        MathErrorType,
        MathVerificationStatus,
        BayesianProblem,
    )
    MATH_VERIFICATION_AVAILABLE = True
    logger.info("Mathematical verification imported for accuracy feedback")
except ImportError:
    try:
        from vulcan.reasoning.mathematical_verification import (
            MathematicalVerificationEngine,
            MathErrorType,
            MathVerificationStatus,
            BayesianProblem,
        )
        MATH_VERIFICATION_AVAILABLE = True
        logger.info("Mathematical verification imported for accuracy feedback")
    except ImportError:
        MathematicalVerificationEngine = None
        MathErrorType = None
        MathVerificationStatus = None
        BayesianProblem = None
        MATH_VERIFICATION_AVAILABLE = False
        logger.debug("Mathematical verification not available")


# Import embedding circuit breaker for latency protection
try:
    from .embedding_circuit_breaker import (
        EmbeddingCircuitBreaker,
        get_embedding_circuit_breaker,
        get_circuit_breaker_stats,
    )
    CIRCUIT_BREAKER_AVAILABLE = True
    logger.info("Embedding circuit breaker imported successfully")
except ImportError as e:
    logger.warning(f"Embedding circuit breaker not available: {e}")
    CIRCUIT_BREAKER_AVAILABLE = False
    EmbeddingCircuitBreaker = None
    get_embedding_circuit_breaker = None
    get_circuit_breaker_stats = None


# ==============================================================================
# EXTRACTED MODULE IMPORTS
# Classes extracted for modularity - imported back for backward compatibility
# ==============================================================================
from .feature_extraction import MultiTierFeatureExtractor
from .confidence import (
    ToolConfidenceCalibrator,
    CalibratedDecisionMaker,
    ValueOfInformationGate,
    DistributionMonitor,
)
from .bandit import ToolSelectionBandit
from .selection_types import SelectionMode, SelectionRequest, SelectionResult
from .tools import (
    CausalToolWrapper,
    AnalogicalToolWrapper,
    MultimodalToolWrapper,
    CryptographicToolWrapper,
    PhilosophicalToolWrapper,
    SymbolicToolWrapper,
    ProbabilisticToolWrapper,
    WorldModelToolWrapper,
    MathematicalToolWrapper,
)
from .stochastic_cost import StochasticCostModel


# ==============================================================================
# Constants for Implicit Feedback Recording
# ==============================================================================
SUCCESS_CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for success
MAX_SUCCESS_TIME_MS = 10000  # Maximum execution time (ms) for success

# ==============================================================================
# ==============================================================================
# Learned Weight Thresholds
# ==============================================================================
# Threshold below which tools are considered "too unreliable" based on learned weights.
# Tools with weight below this are skipped in classifier suggestions.
NEGATIVE_WEIGHT_THRESHOLD = -0.05

# ==============================================================================
# Learning Reward Penalties
# ==============================================================================
# Penalty factor for unverified high-confidence results.
# Prevents learning from potentially wrong but confident answers.
UNVERIFIED_QUALITY_PENALTY = 0.7  # Reduce to 70% of claimed confidence

# Penalty factor for fallback results.
# Heavily penalizes fallback paths to prevent reinforcing failures.
FALLBACK_QUALITY_PENALTY = 0.3  # Reduce to 30% of quality

# ==============================================================================
# Semantic Context Keywords for Ethics/Philosophy Detection (Issue #3 Fix)
# ==============================================================================
# Keywords indicating ethics/philosophy context.
# Used to prevent routing ethics queries to mathematical engine based solely
# on symbol detection. When 2+ of these keywords are present, the query is
# considered to have an ethics/philosophy context.
ETHICS_PHILOSOPHY_KEYWORDS: Tuple[str, ...] = (
    'ethics', 'ethical', 'policy', 'moral', 'morality', 'philosophy',
    'philosophical', 'value', 'values', 'constraint', 'constraints',
    'multimodal reasoning', 'cross-constraints', 'cross-domain',
    'reasoning about', 'ethical implications', 'policy implications',
)

# ==============================================================================
# QueryRouter Tool Selection
# ==============================================================================
# Default available tools when not specified in class instance.
# These represent all reasoning tools that can be selected by the QueryRouter.
# Includes 'world_model' for self-introspection queries (routing queries about
# Vulcan's capabilities, goals, and identity to WorldModel's meta-reasoning).
# Includes 'language' for NLP tasks (quantifier scope, parsing, etc.).
DEFAULT_AVAILABLE_TOOLS = (
    'symbolic', 'probabilistic', 'causal', 'analogical', 'multimodal',
    'mathematical', 'philosophical', 'language', 'world_model'
)

# ==============================================================================
# Multimodal Detection
# ==============================================================================
# Minimum string length to be considered as potential URL or file path.
MULTIMODAL_MIN_URL_LENGTH = 50
# Minimum string length to be considered as potential base64 data.
MULTIMODAL_MIN_BASE64_LENGTH = 100

# ==============================================================================
# Embedding Timeout Configuration
# ==============================================================================
# PERFORMANCE FIX: Reduced from 30s to 5s to prevent query routing cascade delays
# Issue: With decomposition path, each step calls tool selection which calls embeddings
# Multiple 30s timeouts per query caused 48+ second delays (evidenced in logs)
# 5 seconds is sufficient for cached embeddings; fallback to Tier 1 features otherwise
#
# CONFIGURABLE: Set VULCAN_EMBEDDING_TIMEOUT environment variable to override
# Example: VULCAN_EMBEDDING_TIMEOUT=10.0 for slower environments
import os
try:
    EMBEDDING_TIMEOUT = float(os.environ.get("VULCAN_EMBEDDING_TIMEOUT", "5.0"))
except (ValueError, TypeError):
    logger.warning("Invalid VULCAN_EMBEDDING_TIMEOUT, using default 5.0")
    EMBEDDING_TIMEOUT = 5.0


# ==============================================================================
# 1. StochasticCostModel - now imported from .stochastic_cost
# ==============================================================================


# ==============================================================================
# 2. MultiTierFeatureExtractor - now imported from .feature_extraction
# ==============================================================================

# Memory cleanup thresholds for embedding cache
# These values are tuned based on production observations of memory degradation
CLEANUP_CACHE_CAPACITY_THRESHOLD = 0.9  # Trigger cleanup at 90% cache capacity
CLEANUP_MISS_INTERVAL = 100  # Trigger cleanup every N cache misses

# Multimodal tool configuration
# CPU OPTIMIZATION: Increased from 1.5 to 3.0 to allow multimodal operations
# sufficient time headroom under CPU-only execution
MULTIMODAL_TIME_BUDGET_MULTIPLIER = 3.0  # Allow multimodal more time headroom

# Quality penalty for meta tools when domain-specific tools are available
# This ensures symbolic/math/probabilistic/causal engines preferred over meta-reasoning
META_TOOL_QUALITY_PENALTY = 0.85  # 15% reduction in quality estimate

# ==============================================================================
# Candidate Filtering Configuration
# ==============================================================================
CANDIDATE_PRIOR_THRESHOLD = 0.20  # Minimum prior probability to be a candidate (increased from 0.15)
CANDIDATE_MAX_COUNT = 1  # Maximum number of candidates (reduced from 2 to strongly prefer single tool)
CANDIDATE_DOMINANCE_RATIO = 1.8  # If top tool has 1.8x the prior, use only that tool (reduced from 2.0)

# ==============================================================================
# LLM Classification Integration Configuration
# ==============================================================================
LLM_CLASSIFICATION_ENABLED = True  # Feature flag to enable/disable LLM classification
LLM_CLASSIFICATION_CONFIDENCE_THRESHOLD = 0.8  # Minimum confidence to use LLM result
LLM_CLASSIFICATION_TIMEOUT = 3.0  # Seconds - timeout for LLM classification call


# ==============================================================================
# 3-6. ToolConfidenceCalibrator, CalibratedDecisionMaker, ValueOfInformationGate,
#      DistributionMonitor, ToolSelectionBandit - now imported from
#      .confidence and .bandit
# 7-9. SelectionMode, SelectionRequest, SelectionResult - now imported from
#      .selection_types
# ==============================================================================



# ==============================================================================
# TOOL WRAPPER CLASSES - now imported from .tools subpackage
# ==============================================================================
# SymbolicToolWrapper -> .tools.symbolic + .tools.symbolic_helpers
# ProbabilisticToolWrapper -> .tools.probabilistic + .tools.probabilistic_inference
# WorldModelToolWrapper -> .tools.world_model_queries + world_model_creative + world_model_domain + world_model_helpers
# MathematicalToolWrapper -> .tools.mathematical
# StochasticCostModel -> .stochastic_cost
# (CausalToolWrapper, AnalogicalToolWrapper, MultimodalToolWrapper,
#  CryptographicToolWrapper, PhilosophicalToolWrapper were already extracted)


class ToolSelector:
    """
    Main tool selector orchestrating all components
    """
    
    # REMOVED (Jan 21 2026): Regex patterns for keyword-based routing
    # These patterns were bypassing LLM classification and causing misrouting.
    # Evidence: SAT queries routed to probabilistic engine, causal queries to symbolic.
    # The LLM router provides accurate semantic classification - trust it.

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize tool selector with configuration

        Args:
            config: Configuration dictionary
        """
        config = config or {}

        # Load configuration
        self.config = self._load_config(config)

        # Available tools
        self.tools = {}
        self.tool_names = []
        self._initialize_tools()

        # Core components
        self.admission_control = AdmissionControlIntegration(
            config.get("admission_config", {})
        )

        self.memory_prior = BayesianMemoryPrior(
            memory_system=config.get("memory_system"), prior_type=PriorType.HIERARCHICAL
        )

        self.portfolio_executor = PortfolioExecutor(
            tools=self.tools, max_workers=config.get("max_workers", 4)
        )

        self.safety_governor = SafetyGovernor(config.get("safety_config", {}))

        self.cache = SelectionCache(config.get("cache_config", {}))

        # Use singleton WarmStartPool to prevent
        # "Warm pool initialized with 5 tool pools" appearing multiple times.
        # The singleton is shared across all ToolSelector instances.
        try:
            from vulcan.reasoning.singletons import get_warm_pool
            self.warm_pool = get_warm_pool(
                tools=self.tools,
                config=config.get("warm_pool_config", {})
            )
            if self.warm_pool is None:
                # Fallback: Create directly if singleton fails.
                # Note: This may cause duplicate initialization if called multiple times,
                # but is necessary for robustness when singletons module has issues.
                logger.warning(
                    "WarmStartPool singleton unavailable, creating instance directly. "
                    "This may result in duplicate initialization if ToolSelector is created multiple times."
                )
                self.warm_pool = WarmStartPool(
                    tools=self.tools, config=config.get("warm_pool_config", {})
                )
        except ImportError:
            # Fallback: Create directly if singletons module not available
            self.warm_pool = WarmStartPool(
                tools=self.tools, config=config.get("warm_pool_config", {})
            )

        # Decision components
        self.utility_model = UtilityModel()
        self.cost_model = StochasticCostModel(config.get("cost_model_config", {}))
        self.feature_extractor = MultiTierFeatureExtractor(
            config.get("feature_config", {})
        )
        self.calibrator = ToolConfidenceCalibrator(config.get("calibration_config", {}))
        self.voi_gate = ValueOfInformationGate(config.get("voi_config", {}))
        self.distribution_monitor = DistributionMonitor(
            config.get("monitor_config", {})
        )

        # Learning component
        self.bandit = ToolSelectionBandit(config.get("bandit_config", {}))
        
        # Learning system integration (set externally)
        self.learning_system: Optional[Any] = None
        
        # Mathematical verification engine for accuracy feedback
        # This enables learning from mathematical reasoning accuracy
        # CACHING FIX: Use singleton to prevent repeated initialization
        self.math_verifier: Optional["MathematicalVerificationEngine"] = None
        if MATH_VERIFICATION_AVAILABLE and config.get("enable_math_verification", True):
            self.math_verifier = self._init_math_verifier()

        # Execution statistics
        self.execution_history = deque(maxlen=1000)
        self.performance_metrics = defaultdict(
            lambda: {
                "count": 0,
                "successes": 0,
                "avg_time": 0.0,
                "avg_energy": 0.0,
                "avg_confidence": 0.0,
            }
        )
        
        # Mathematical accuracy statistics
        self.math_accuracy_metrics = defaultdict(
            lambda: {
                "verifications": 0,
                "verified_correct": 0,
                "errors_detected": 0,
                "error_types": defaultdict(int),
            }
        )

        # CRITICAL FIX: Add locks and shutdown event for thread safety and interruptible threads
        self.stats_lock = threading.RLock()
        self.shutdown_lock = threading.Lock()
        self._shutdown_event = threading.Event()
        self.is_shutdown = False

        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=2)

        # Start background processes
        self._start_background_processes()

        logger.info("Tool Selector initialized with {} tools".format(len(self.tools)))

    def _load_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Load and validate configuration"""

        default_config = {
            "max_workers": 4,
            "cache_enabled": True,
            "safety_enabled": True,
            "learning_enabled": True,
            "warm_pool_enabled": True,
            "default_timeout_ms": 10000,  # Increased to allow multimodal processing
            "default_energy_budget_mj": 1000,
            "min_confidence": 0.5,
            "enable_calibration": True,
            "enable_voi": True,
            "enable_distribution_monitoring": True,
        }

        # Merge with provided config
        merged_config = {**default_config, **config}

        # Load from file if specified
        if "config_file" in merged_config:
            try:
                config_path = Path(merged_config["config_file"])
                if config_path.exists():
                    with open(config_path, "r", encoding="utf-8") as f:
                        file_config = json.load(f)
                        merged_config.update(file_config)
            except Exception as e:
                logger.error(f"Failed to load config file: {e}")

        return merged_config

    def _init_math_verifier(self) -> Optional["MathematicalVerificationEngine"]:
        """
        Initialize MathematicalVerificationEngine with singleton pattern.
        
        CACHING FIX: Uses singleton to prevent repeated initialization
        that was causing "MathematicalVerificationEngine initialized" to
        appear 4+ times per query.
        
        Returns:
            MathematicalVerificationEngine instance or None on failure
        """
        try:
            # Try singleton first
            from vulcan.reasoning.singletons import get_math_verification_engine
            verifier = get_math_verification_engine()
            if verifier is not None:
                logger.info("Mathematical verification engine obtained from singleton")
                return verifier
        except ImportError:
            pass  # singletons module not available, continue to fallback
        except Exception as e:
            logger.debug(f"Singleton access failed: {e}")
        
        # Fallback to direct creation
        try:
            verifier = MathematicalVerificationEngine()
            logger.info("Mathematical verification engine initialized (fallback)")
            return verifier
        except Exception as e:
            logger.warning(f"Failed to initialize math verifier: {e}")
            return None

    def _detect_math_symbols(self, query: str) -> bool:
        """
        Detect MATHEMATICAL symbols (NOT logic symbols).
        
        This is separate from formal logic detection because symbols like ∑ (summation),
        ∫ (integral), ∂ (partial derivative) are MATH symbols, not logic symbols.
        
        Previously, ∑ was incorrectly grouped with logic symbols, causing math
        queries like "Compute ∑(k=1 to n) k" to be routed to the symbolic reasoner,
        which correctly rejected them, but then math engine never got a chance.
        
        Math symbols: ∑ ∫ ∂ ∇ ∏ √ ≤ ≥ ≠ ≈ ± × ÷ ∞
        Logic symbols: → ∧ ∨ ¬ ∀ ∃ ⊢ ⊨ ↔ ⇒ ⇔
        
        FIX (Issue #3): Added semantic context check to prevent routing ethics/philosophical
        queries to mathematical engine just because they contain mathematical notation.
        Example: "Multimodal Reasoning (cross-constraints) MM1 — Math + logic + ethics + policy"
        contains mathematical notation but is fundamentally an ethics/policy question.
        
        Args:
            query: The query text
            
        Returns:
            True if query contains math symbols AND is not semantically ethics/philosophical
        """
        if not query or not isinstance(query, str):
            return False
        
        query_lower = query.lower()
        
        # =================================================================
        # FIX (Issue #3): Check semantic context BEFORE symbol detection
        # =================================================================
        # Queries about ethics, policy, philosophy, or cross-domain reasoning
        # should NOT be routed to mathematical engine even if they contain
        # mathematical symbols or notation. The presence of symbols in an
        # academic/philosophical context doesn't make it a math problem.
        #
        # Example that was broken:
        #   "Multimodal Reasoning (cross-constraints) MM1 — Math + logic + ethics + policy"
        #   - Contains mathematical notation (𝐸, 𝑢(𝑡), Greek letters)
        #   - BUT is fundamentally about ethics/policy reasoning
        #   - Should route to world_model/philosophical, NOT mathematical
        # Uses module-level ETHICS_PHILOSOPHY_KEYWORDS for better performance.
        # =================================================================
        
        # Count ethics/philosophy keywords using module-level constant
        ethics_count = sum(1 for kw in ETHICS_PHILOSOPHY_KEYWORDS if kw in query_lower)
        
        # If query has 2+ ethics/philosophy keywords, it's likely NOT a pure math problem
        if ethics_count >= 2:
            logger.debug(
                f"[ToolSelector] Query has {ethics_count} ethics/philosophy keywords - "
                f"NOT detecting as math despite symbols"
            )
            return False
        
        # Pure math operators (NOT logic)
        math_symbols = ['∑', '∫', '∂', '∇', '∏', '√', '≤', '≥', '≠', '≈', '±', '×', '÷', '∞']
        if any(symbol in query for symbol in math_symbols):
            logger.debug("[ToolSelector] Detected Unicode math symbol")
            return True
        
        # Math-specific keywords (not shared with logic)
        math_keywords = [
            'summation', 'integral', 'derivative', 'differential', 'limit',
            'compute exactly', 'calculate', 'evaluate the sum', 'closed form',
            'by induction', 'sigma', 'sigma notation', 'series', 'convergent',
            'arithmetic progression', 'geometric series'
        ]
        if any(keyword in query_lower for keyword in math_keywords):
            logger.debug("[ToolSelector] Detected math keyword")
            return True
        
        # Summation notation patterns: ∑(k=1 to n) or sum from k=1 to n
        sum_patterns = [
            r'∑.*=.*\d+',  # ∑...=...n
            r'sum\s+(?:from|for)\s+\w+\s*=\s*\d+',  # sum from k=1
            r'\bsum\s+\w+\s*=\s*\d+\s+to\b',  # sum k=1 to
        ]
        for pattern in sum_patterns:
            if re.search(pattern, query_lower):
                logger.debug(f"[ToolSelector] Detected summation pattern: {pattern}")
                return True
        
        # Integral notation patterns
        if re.search(r'∫.*d[xyz]', query):
            logger.debug("[ToolSelector] Detected integral pattern")
            return True
        
        # Limit notation patterns
        if re.search(r'lim.*→|limit.*as.*→', query_lower):
            logger.debug("[ToolSelector] Detected limit pattern")
            return True
        
        return False
    
    def _detect_formal_logic(self, query: str) -> bool:
        """
        Detect formal logic notation to route to symbolic engine.
        
        This prevents SAT/FOL problems from being misrouted to probabilistic
        engine by the LLM classifier.
        
        NO LONGER detects math symbols (∑, ∫, etc.). Those are
        handled separately by _detect_math_symbols() for mathematical routing.
        
        Note: NO LONGER triggers on ethical/philosophical queries that
        contain natural language choice structures like "option A or B".
        
        Detects:
        - Logic symbols: →, ∧, ∨, ¬, ∀, ∃, ⊢, ⊨ (NOT ∑, ∫, ∂ - those are math!)
        - SAT problem keywords: satisfiable, SAT, CNF, prove, theorem
        - Propositional variables with constraints (A, B, C)
        - First-order logic patterns
        
        Args:
            query: The query text
            
        Returns:
            True if query appears to be a formal logic problem
        """
        if not query or not isinstance(query, str):
            return False
        
        # First check if this is a math query - math takes priority
        # over logic symbol detection because math queries may contain both
        if self._detect_math_symbols(query):
            logger.debug("[ToolSelector] Query is mathematical, not formal logic")
            return False
        
        query_lower = query.lower()
        
        # Note: Check if this is an ethical/philosophical query FIRST
        # Ethical queries contain natural language choice structures ("A or B",
        # "not pulling the lever") that should NOT trigger formal logic routing.
        # The symbolic engine cannot parse natural language ethical dilemmas.
        ethical_indicators = [
            'trolley', 'dilemma', 'ethical', 'moral', 'ethics', 'morality',
            'should you', 'must choose', 'lives', 'death', 'kill', 'save',
            'sacrifice', 'utilitarian', 'deontological', 'virtue', 'duty',
            'right thing', 'wrong to', 'permissible', 'obligation',
            'conscience', 'harm', 'benefit', 'consequent', 'rights',
        ]
        ethical_count = sum(1 for ind in ethical_indicators if ind in query_lower)
        
        if ethical_count >= 2:
            # Multiple ethical indicators = likely philosophical query
            logger.debug(
                f"[ToolSelector] Detected {ethical_count} ethical indicators - "
                f"NOT routing to symbolic engine (ethical queries need philosophical reasoning)"
            )
            return False
        
        # Check for Unicode logic symbols (optimized using any())
        # Note: REMOVED ∑, ∫, ∂, ∇ from this list - they are MATH symbols!
        logic_symbols = ['→', '∧', '∨', '¬', '∀', '∃', '⊢', '⊨', '↔', '⇒', '⇔']
        if any(symbol in query for symbol in logic_symbols):
            logger.debug("[ToolSelector] TASK 3: Detected Unicode logic symbol")
            return True
        
        # Note: More restrictive ASCII logic detection
        # Don't match natural language patterns like "option A or B" or "not pulling"
        # Only match patterns that look like actual formal logic: "A -> B", "P && Q"
        # The check for 'not ', 'and ', 'or ' is too aggressive for natural language.
        ascii_logic_strict = ['->', '<->', '&&', '||']  # Removed 'not ', 'and ', 'or '
        has_proposition = re.search(r'\b[A-Z]\b', query) is not None  # Cache this check
        if has_proposition and any(pattern in query_lower for pattern in ascii_logic_strict):
            logger.debug("[ToolSelector] TASK 3: Detected ASCII logic with propositions")
            return True
        
        # Check for SAT-style keywords (optimized using any())
        sat_keywords = [
            'satisfiable', 'satisfiability', 'sat', 'cnf', 'dnf',
            'prove', 'theorem', 'proof', 'valid', 'tautology', 
            'contradiction', 'unsatisfiable', 'entailment', 'entails',
            'contrapositive', 'modus ponens', 'modus tollens',
        ]
        if any(keyword in query_lower for keyword in sat_keywords):
            logger.debug("[ToolSelector] TASK 3: Detected SAT keyword")
            return True
        
        # Check for "Propositions: A, B, C" or "Variables: A, B, C" pattern
        if re.search(r'(?:propositions?|variables?)\s*:?\s*[A-Z](?:\s*,\s*[A-Z])+', query, re.IGNORECASE):
            logger.debug("[ToolSelector] TASK 3: Detected proposition list")
            return True
        
        # Check for constraint patterns: "A → B", "B → C", "¬C"
        # This catches: "Constraints: A→B, B→C, ¬C"
        if 'constraint' in query_lower and re.search(r'[A-Z]\s*[→\-−>]+\s*[A-Z]', query):
            logger.debug("[ToolSelector] TASK 3: Detected constraint pattern")
            return True
        
        # Check for first-order logic quantifiers in natural language
        # Note: Only trigger if BOTH quantifier AND logic keywords present
        fol_patterns = [
            r'\bfor\s+all\b',
            r'\bthere\s+exists?\b',
            r'\bfor\s+every\b',
            r'\bfor\s+some\b',
            r'\bfor\s+any\b',
        ]
        # More restrictive: require logic-specific words, not just 'if/then'
        logic_keywords = ['implies', 'therefore', 'conclude', 'entails', 'logically']
        has_logic_keyword = any(w in query_lower for w in logic_keywords)
        
        if has_logic_keyword:
            for pattern in fol_patterns:
                if re.search(pattern, query_lower):
                    logger.debug(f"[ToolSelector] TASK 3: Detected FOL pattern '{pattern}'")
                    return True
        
        return False

    def _initialize_tools(self):
        """
        Initialize reasoning tools with ACTUAL reasoning engines.
        
        Note: Previously this method created MockTool placeholders that just
        returned canned responses. This caused the selected tools to never
        actually execute any reasoning logic - OpenAI answered everything.
        
        Now this method:
        1. Tries to import the real reasoning engines (SymbolicReasoner, etc.)
        2. Creates wrapper classes that adapt engine interfaces to reason() method
        3. Falls back to mock tools ONLY if imports fail
        
        The wrapper classes ensure that when tool.reason(problem) is called,
        the actual engine's query/inference logic is executed (SAT solving,
        Bayesian inference, causal analysis, etc.)
        
        Note: Added world_model tool for self-introspection queries.
        Note: Added cryptographic tool for hash/encoding computations.
        Note: Added philosophical and mathematical tools (FIX #1 - Missing Engine Registration).
        """
        tool_configs = {
            "symbolic": {"speed": "medium", "accuracy": "high", "energy": "medium"},
            "probabilistic": {"speed": "fast", "accuracy": "medium", "energy": "low"},
            "causal": {"speed": "slow", "accuracy": "high", "energy": "high"},
            "analogical": {"speed": "fast", "accuracy": "low", "energy": "low"},
            "multimodal": {"speed": "slow", "accuracy": "high", "energy": "very_high"},
            "world_model": {"speed": "fast", "accuracy": "high", "energy": "low"},  # Note: world_model tool
            "cryptographic": {"speed": "fast", "accuracy": "perfect", "energy": "low"},  # Note: cryptographic tool
            # FIX #1: Register philosophical and mathematical tools
            # These engines were being routed to but not available, causing fallback to wrong tools
            "philosophical": {"speed": "medium", "accuracy": "high", "energy": "medium"},  # FIX #1: philosophical tool
            "mathematical": {"speed": "medium", "accuracy": "high", "energy": "medium"},   # FIX #1: mathematical tool
        }

        # Try to initialize real reasoning engines
        engines_initialized = self._initialize_real_engines()
        
        for tool_name, config in tool_configs.items():
            if tool_name in engines_initialized and engines_initialized[tool_name] is not None:
                # Use the real engine wrapper
                self.tools[tool_name] = engines_initialized[tool_name]
                logger.info(f"[ToolSelector] Initialized REAL {tool_name} engine")
            else:
                # Fall back to mock tool
                self.tools[tool_name] = self._create_mock_tool(tool_name, config)
                logger.warning(f"[ToolSelector] Using MOCK {tool_name} tool (real engine unavailable)")
            self.tool_names.append(tool_name)

    def _initialize_real_engines(self) -> Dict[str, Any]:
        """
        Initialize real reasoning engines with proper adapters.
        
        Returns:
            Dictionary mapping tool names to engine wrapper instances
        """
        engines = {}
        
        # ============================================================
        # SYMBOLIC ENGINE (SAT solver, FOL theorem proving)
        # ============================================================
        try:
            from ..symbolic.reasoner import SymbolicReasoner
            engines["symbolic"] = SymbolicToolWrapper(SymbolicReasoner())
            logger.info("[ToolSelector] SymbolicReasoner loaded successfully")
        except ImportError as e:
            logger.warning(f"[ToolSelector] SymbolicReasoner not available: {e}")
            engines["symbolic"] = None
        except Exception as e:
            logger.error(f"[ToolSelector] SymbolicReasoner initialization failed: {e}")
            engines["symbolic"] = None
        
        # ============================================================
        # PROBABILISTIC ENGINE (Bayesian inference)
        # ============================================================
        try:
            from ..symbolic.reasoner import ProbabilisticReasoner
            engines["probabilistic"] = ProbabilisticToolWrapper(ProbabilisticReasoner())
            logger.info("[ToolSelector] ProbabilisticReasoner loaded successfully")
        except ImportError as e:
            logger.warning(f"[ToolSelector] ProbabilisticReasoner not available: {e}")
            engines["probabilistic"] = None
        except Exception as e:
            logger.error(f"[ToolSelector] ProbabilisticReasoner initialization failed: {e}")
            engines["probabilistic"] = None
        
        # ============================================================
        # CAUSAL ENGINE (Causal DAG analysis, counterfactuals)
        # ============================================================
        try:
            from ..causal_reasoning import CausalReasoner
            engines["causal"] = CausalToolWrapper(CausalReasoner())
            logger.info("[ToolSelector] CausalReasoner loaded successfully")
        except ImportError as e:
            logger.warning(f"[ToolSelector] CausalReasoner not available: {e}")
            engines["causal"] = None
        except Exception as e:
            logger.error(f"[ToolSelector] CausalReasoner initialization failed: {e}")
            engines["causal"] = None
        
        # ============================================================
        # ANALOGICAL ENGINE (Pattern matching, analogy reasoning)
        # ============================================================
        try:
            from ..analogical import AnalogicalReasoner
            engines["analogical"] = AnalogicalToolWrapper(AnalogicalReasoner())
            logger.info("[ToolSelector] AnalogicalReasoner loaded successfully")
        except ImportError as e:
            logger.warning(f"[ToolSelector] AnalogicalReasoner not available: {e}")
            engines["analogical"] = None
        except Exception as e:
            logger.error(f"[ToolSelector] AnalogicalReasoner initialization failed: {e}")
            engines["analogical"] = None
        
        # ============================================================
        # MULTIMODAL ENGINE (Multi-modal reasoning - images, etc.)
        # ============================================================
        try:
            from ..multimodal_reasoning import MultimodalReasoner
            engines["multimodal"] = MultimodalToolWrapper(MultimodalReasoner())
            logger.info("[ToolSelector] MultimodalReasoner loaded successfully")
        except ImportError as e:
            logger.warning(f"[ToolSelector] MultimodalReasoner not available: {e}")
            engines["multimodal"] = None
        except Exception as e:
            logger.error(f"[ToolSelector] MultimodalReasoner initialization failed: {e}")
            engines["multimodal"] = None
        
        # ============================================================
        # WORLD MODEL ENGINE (Self-introspection queries)
        # ============================================================
        # This enables queries about Vulcan's capabilities, goals, and limitations
        # to be routed to the World Model's SelfModel instead of reasoning engines.
        try:
            # Try to get the world model instance from the global context
            # The WorldModelToolWrapper is DESIGNED to work without a live world model
            # using its static self-model data as a fallback. The live world model
            # should be injected via the orchestrator/main.py at runtime when available.
            # This initialization creates a functional wrapper that can serve
            # self-introspection queries even during standalone testing.
            world_model_instance = None
            try:
                from ...world_model.world_model_core import WorldModel
                # Note: We don't instantiate WorldModel here - that should be done
                # at application startup. The wrapper can function without it.
                # When a live world model is available, it will be passed to the
                # wrapper via the orchestrator or main.py.
                logger.info("[ToolSelector] WorldModel module available for future injection")
            except ImportError:
                logger.debug("[ToolSelector] WorldModel module not available, using static self-model")
            
            engines["world_model"] = WorldModelToolWrapper(world_model=world_model_instance)
            logger.info("[ToolSelector] WorldModelToolWrapper loaded successfully")
        except Exception as e:
            logger.error(f"[ToolSelector] WorldModelToolWrapper initialization failed: {e}")
            engines["world_model"] = None
        
        # ============================================================
        # Note: CRYPTOGRAPHIC ENGINE (Hash/encoding computations)
        # ============================================================
        # This enables deterministic cryptographic computations (SHA-256, MD5, etc.)
        # instead of relying on LLM fallback which hallucinates incorrect values.
        try:
            from ..cryptographic_engine import CryptographicEngine
            engines["cryptographic"] = CryptographicToolWrapper(CryptographicEngine())
            logger.info("[ToolSelector] CryptographicEngine loaded successfully")
        except ImportError as e:
            logger.warning(f"[ToolSelector] CryptographicEngine not available: {e}")
            engines["cryptographic"] = None
        except Exception as e:
            logger.error(f"[ToolSelector] CryptographicEngine initialization failed: {e}")
            engines["cryptographic"] = None
        
        # ============================================================
        # ============================================================
        # PHILOSOPHICAL ENGINE - Now routes to World Model
        # ============================================================
        # PhilosophicalReasoner has been removed. The PhilosophicalToolWrapper
        # now delegates to World Model's _philosophical_reasoning method.
        # World Model has full meta-reasoning machinery for ethical reasoning.
        try:
            engines["philosophical"] = PhilosophicalToolWrapper()  # Delegates to World Model
            logger.info("[ToolSelector] Philosophical reasoning: Routed to World Model")
        except Exception as e:
            logger.error(f"[ToolSelector] PhilosophicalToolWrapper initialization failed: {e}")
            engines["philosophical"] = None
        
        # ============================================================
        # FIX #1: MATHEMATICAL ENGINE (Symbolic math computation)
        # ============================================================
        # This enables routing of mathematical queries to the proper
        # computation engine instead of falling back to symbolic or LLM.
        # Evidence from logs: "Tool 'mathematical' not available, using fallback: symbolic"
        try:
            from ..mathematical_computation import MathematicalComputationTool
            engines["mathematical"] = MathematicalToolWrapper(MathematicalComputationTool())
            logger.info("[ToolSelector] MathematicalComputationTool loaded successfully")
        except ImportError as e:
            logger.warning(f"[ToolSelector] MathematicalComputationTool not available: {e}")
            engines["mathematical"] = None
        except Exception as e:
            logger.error(f"[ToolSelector] MathematicalComputationTool initialization failed: {e}")
            engines["mathematical"] = None
        
        return engines

    def _create_mock_tool(self, name: str, config: Dict[str, Any]) -> Any:
        """
        Create mock tool as fallback when real engines are unavailable.
        
        NOTE: This is a FALLBACK only. In production, real engines should be used.
        Mock tools do NOT perform actual reasoning - they just return placeholder results.
        """

        class MockTool:
            def __init__(self, tool_name, tool_config):
                self.name = tool_name
                self.config = tool_config

            def reason(self, problem):
                # Log that we're using a mock (helps debugging)
                logger.warning(f"[MockTool:{self.name}] Using MOCK reasoning (real engine unavailable)")
                
                # Simulate execution
                time.sleep(0.1)  # Simulate work

                # Deterministic confidence based on tool name and config
                import zlib

                # SECURITY NOTE: Using CRC32 instead of MD5 for deterministic hashing
                # CRC32 is appropriate here because:
                # 1. This is NOT cryptographic use (just deterministic mock simulation)
                # 2. No security properties required (collision resistance not needed)
                # 3. Better performance than MD5 (4-8x faster)
                # 4. Clearer intent (CRC32 is explicitly non-cryptographic)
                # Mask with 0xffffffff to ensure unsigned 32-bit value for cross-platform consistency
                tool_hash = zlib.crc32(f"{name}{str(config)}".encode()) & 0xffffffff
                confidence = 0.5 + (tool_hash % 500) / 1000.0  # Range: 0.5 to 1.0

                return {
                    "tool": self.name,
                    "result": f"[MOCK] Result from {self.name} - real engine not available",
                    "confidence": confidence,
                    "is_mock": True,  # Flag to indicate this is a mock result
                }

        return MockTool(name, config)


    def _start_background_processes(self):
        """Start background processes"""

        # Periodic cache warming
        if self.config.get("warm_pool_enabled"):
            self.executor.submit(self._warm_cache_loop)

        # Periodic statistics update
        self.executor.submit(self._update_statistics_loop)

    # CRITICAL FIX: Interruptible cache warming thread
    def _warm_cache_loop(self):
        """Background cache warming - CRITICAL: With error handling and interruptible sleep"""

        consecutive_errors = 0
        max_errors = 5

        while not self._shutdown_event.is_set():
            try:
                # CRITICAL FIX: Interruptible sleep - 5 minutes can be interrupted
                if self._shutdown_event.wait(timeout=300):
                    break

                with self.shutdown_lock:
                    if self.is_shutdown:
                        break

                self.cache.warm_cache()
                consecutive_errors = 0

            except Exception as e:
                consecutive_errors += 1
                logger.error(
                    f"Cache warming error ({consecutive_errors}/{max_errors}): {e}"
                )

                if consecutive_errors >= max_errors:
                    logger.critical("Cache warming failed repeatedly, stopping")
                    break

                # CRITICAL FIX: Interruptible error sleep
                if self._shutdown_event.wait(timeout=30):
                    break

    # CRITICAL FIX: Interruptible statistics update thread
    def _update_statistics_loop(self):
        """Background statistics update - CRITICAL: With error handling and interruptible sleep"""

        consecutive_errors = 0
        max_errors = 5

        while not self._shutdown_event.is_set():
            try:
                # CRITICAL FIX: Interruptible sleep - 1 minute can be interrupted
                if self._shutdown_event.wait(timeout=60):
                    break

                with self.shutdown_lock:
                    if self.is_shutdown:
                        break

                stats = self.get_statistics()
                logger.debug(
                    f"System statistics: {json.dumps(stats, default=str)[:500]}"
                )
                consecutive_errors = 0

            except Exception as e:
                consecutive_errors += 1
                logger.error(
                    f"Statistics update error ({consecutive_errors}/{max_errors}): {e}"
                )

                if consecutive_errors >= max_errors:
                    logger.critical("Statistics update failed repeatedly, stopping")
                    break

                # CRITICAL FIX: Interruptible error sleep
                if self._shutdown_event.wait(timeout=30):
                    break

    def select_and_execute(self, request: SelectionRequest) -> SelectionResult:
        """
        Main entry point for tool selection and execution

        Args:
            request: Selection request

        Returns:
            SelectionResult with execution details
        """

        try:
            start_time = time.time()

            # ================================================================
            # CRITICAL FIX (Jan 6 2026): Check for world model delegation FIRST
            # ================================================================
            # Note: Previously routing was overriding world model delegation because
            # it detected "formal logic" keywords in cryptocurrency questions.
            #
            # Evidence from diagnostic report:
            #   Line 2854: [WorldModel] DELEGATION RECOMMENDED: 'mathematical'
            #   Line 2855: [ToolSelector] Formal logic detected - routing to symbolic
            #   ^ CONTRADICTION: Delegation ignored, symbolic used instead
            #
            # Check if delegation is active BEFORE applying special routing.
            # If delegation context is set, skip the early detection overrides.
            # 
            # Note: delegation_active and skip_task3 are used in the conditional below
            # to determine whether to skip special routing.
            # ================================================================
            delegation_active = False
            if hasattr(request, 'context') and isinstance(request.context, dict):
                delegation_active = request.context.get('world_model_delegation', False)
                skip_task3 = request.context.get('skip_task3_fix', False)
                
                # Update delegation_active to include skip_task3 flag
                if skip_task3:
                    delegation_active = True
                
                if delegation_active:
                    delegated_tool = request.context.get('world_model_recommended_tool', 'unknown')
                    logger.info(
                        f"[ToolSelector] Delegation check: Delegation active to '{delegated_tool}' - "
                        f"NOT overriding with formal logic detection"
                    )

            # ================================================================
            # REMOVED (Jan 21 2026): Keyword override logic
            # ================================================================
            # Previously, regex patterns (_MATH_PATTERN, _SAT_PATTERN, _CAUSAL_PATTERN)
            # bypassed LLM classification with keyword matching. This caused misrouting:
            # - SAT queries → probabilistic engine (wrong!)
            # - Causal queries → symbolic engine (wrong!)
            # - Multimodal queries → probabilistic engine (wrong!)
            #
            # The problem: Pattern matching cannot distinguish semantic context.
            # Example: "S→T" could be analogical mapping OR symbolic proof.
            #
            # Solution: Trust the LLM router's semantic classification.
            # The query_router.py sets 'selected_tools' based on LLM understanding.
            # We map 'selected_tools' → 'classifier_suggested_tools' below.
            # ================================================================

            # ================================================================
            # Note: REMOVED formal logic pattern override (Jan 9 2026)
            # patterns were detected, routing everything to symbolic engine.
            # 
            # This was WRONG because pattern matching CANNOT distinguish between:
            # - "Map structure S→T" (analogical reasoning)
            # - "Prove S→T→C" (symbolic reasoning)
            # - "Confounding vs causation" (causal reasoning)
            # 
            # Evidence from production logs:
            #   Query: "Structure mapping (not surface similarity)... Domain S→T"
            #   Classifier: CRYPTOGRAPHIC ❌ (should be ANALOGICAL)
            #   Route: symbolic engine ❌
            #   Result: Parser failed, 20% confidence
            #
            #   Query: "Confounding vs causation (Pearl-style)..."
            #   Classifier: SELF_INTROSPECTION ❌ (should be CAUSAL)
            #   Override: "Formal logic detected - routing to symbolic" ❌
            #   Result: Parser failed, 20% confidence
            #
            # The LLM classifier uses semantic understanding and is smarter than
            # pattern matching. Trust it to identify the correct reasoning type:
            # - ANALOGICAL queries → tools=['analogical']
            # - CAUSAL queries → tools=['causal']
            # - LOGICAL queries → tools=['symbolic']
            # - PROBABILISTIC queries → tools=['probabilistic']
            #
            # The override has been REMOVED. The LLM classifier path below
            # will now handle all queries without bypass.
            # ================================================================

            # ================================================================
            # Note: REMOVED mathematical symbols pattern override (Jan 9 2026)
            # ================================================================
            # The mathematical symbols detection code has been REMOVED.
            # 
            # Problem: Symbols like "→" are ambiguous and appear in:
            # - "Map structure S→T" (analogical reasoning)
            # - "Intervention→outcome" (causal reasoning)  
            # - "∑(2k-1)" (mathematical computation)
            #
            # Pattern matching CANNOT distinguish between these cases.
            # The LLM classifier uses semantic understanding to identify
            # the correct reasoning type based on query intent.
            #
            # Evidence from production logs:
            #   Query: "Compute ∑(2k-1), verify by induction"
            #   Pattern override: "Mathematical symbols detected"
            #   Mathematical engine: SyntaxError "invalid syntax at '-'"
            #   Result: confidence=0.1
            #
            # The LLM classifier is smarter than pattern matching. Trust it.
            # Mathematical queries are correctly classified as MATHEMATICAL.
            # The classifier path below handles all queries.
            # ================================================================
            # NOTE: The _detect_math_symbols() method still exists for other uses
            # but it no longer bypasses the LLM classifier here.

            # ================================================================
            # FIX (Jan 21 2026): Map selected_tools to classifier_suggested_tools
            # ================================================================
            # The query_router.py sets 'selected_tools' in telemetry_data based on
            # LLM classification. However, the code below checks for 
            # 'classifier_suggested_tools'. This mismatch caused the LLM's routing
            # decision to be ignored, falling through to regex patterns.
            #
            # Solution: Map 'selected_tools' → 'classifier_suggested_tools' so the
            # existing classifier path (lines 5067-5173) works correctly.
            # ================================================================
            if hasattr(request, 'context') and isinstance(request.context, dict):
                # Check if selected_tools was set by query_router
                selected_tools = request.context.get('selected_tools')
                
                # If selected_tools exists but classifier_suggested_tools doesn't, map it
                if selected_tools and not request.context.get('classifier_suggested_tools'):
                    request.context['classifier_suggested_tools'] = selected_tools
                    logger.info(
                        f"[ToolSelector] Mapped selected_tools={selected_tools} to "
                        f"classifier_suggested_tools (from query_router)"
                    )

            # ================================================================
            # Note: Check if QueryClassifier already suggested tools
            # The classifier uses LLM-based language understanding to identify
            # the correct tool based on query intent (not heuristics).
            # This is the PRIMARY tool selection path.
            #
            # CRITICAL: Skip classifier if this is a fallback attempt.
            # When a tool fails and we're trying a fallback (fallback_attempt=True),
            # the classifier must NOT override the explicit fallback tool selection.
            # The classifier would just re-select the same failed tool (e.g., symbolic
            # for logic queries), causing an infinite retry loop.
            # ================================================================
            if hasattr(request, 'context') and isinstance(request.context, dict):
                # Note: Check if this is a fallback attempt - skip classifier if so
                is_fallback_attempt = request.context.get('fallback_attempt', False)
                
                if is_fallback_attempt:
                    logger.info(
                        f"[ToolSelector] Fallback attempt detected - skipping "
                        f"classifier to allow direct tool override via router_tools"
                    )
                    # Fall through to router_tools check below
                    classifier_tools = None
                else:
                    classifier_tools = request.context.get('classifier_suggested_tools')
                
                classifier_category = request.context.get('classifier_category')
                
                if classifier_tools and isinstance(classifier_tools, (list, tuple)) and len(classifier_tools) > 0:
                    logger.info(
                        f"[ToolSelector] Using LLM classifier's suggested tools: {classifier_tools} "
                        f"for category={classifier_category} (LLM understands query intent)"
                    )
                    
                    # ================================================================
                    # FIX (Jan 10 2026): Handle 'general' tool specially
                    # ================================================================
                    # 'general' is not a reasoning engine - it means "use LLM directly".
                    # When classifier suggests ['general'], we should return immediately
                    # with a high-confidence "skip reasoning" result. This fixes the bug
                    # where CREATIVE queries with introspection themes (e.g., "write a 
                    # poem about becoming self-aware") were being routed to symbolic
                    # reasoning because 'general' is not in DEFAULT_AVAILABLE_TOOLS.
                    # ================================================================
                    if classifier_tools == ['general'] and classifier_category in (
                        'CREATIVE', 'CONVERSATIONAL', 'FACTUAL', 'GREETING', 'CHITCHAT',
                        'creative', 'conversational', 'factual', 'greeting', 'chitchat',
                    ):
                        logger.info(
                            f"[ToolSelector] FIX: Classifier suggests ['general'] for "
                            f"category={classifier_category} - returning early (no reasoning needed)"
                        )
                        # Return a result that indicates "use LLM directly, no reasoning"
                        return SelectionResult(
                            selected_tool='general',
                            execution_result={
                                'tool': 'general',
                                'skip_reasoning': True,
                                'result': None,  # No reasoning result - use LLM
                            },
                            confidence=0.85,
                            calibrated_confidence=0.85,
                            execution_time_ms=0.0,
                            energy_used_mj=0.0,
                            strategy_used=ExecutionStrategy.SINGLE,
                            all_results={'general': {'skip_reasoning': True}},
                            metadata={
                                'classifier_category': classifier_category,
                                'classifier_tools': classifier_tools,
                                'skip_reasoning': True,
                                'fast_path': True,
                            },
                        )
                    
                    # Filter to only include available tools
                    available_tools = getattr(self, 'available_tools', None) or DEFAULT_AVAILABLE_TOOLS
                    valid_classifier_tools = [t for t in classifier_tools if t in available_tools]
                    
                    # ================================================================
                    # Note: Respect learned weights - skip tools with very negative weights
                    # The learning system punishes failing tools, but previously the
                    # classifier/router bypassed this. Now we filter out tools that have
                    # been learned to be unreliable (weight < NEGATIVE_WEIGHT_THRESHOLD).
                    # ================================================================
                    if self.learning_system:
                        filtered_tools = []
                        for tool in valid_classifier_tools:
                            weight = self.learning_system.get_tool_weight_adjustment(tool)
                            if weight < NEGATIVE_WEIGHT_THRESHOLD:
                                logger.info(
                                    f"[ToolSelector] Skipping '{tool}' - learned weight "
                                    f"too low ({weight:.3f}), suggesting alternative"
                                )
                                # Don't add this tool - it has been learned to be unreliable
                            else:
                                filtered_tools.append(tool)
                        
                        # If all classifier tools were filtered out, suggest fallback
                        if not filtered_tools and valid_classifier_tools:
                            logger.warning(
                                f"[ToolSelector] All classifier tools rejected by "
                                f"learned weights, using world_model as fallback"
                            )
                            filtered_tools = ['world_model']
                        
                        valid_classifier_tools = filtered_tools
                    
                    if valid_classifier_tools:
                        # Execute with classifier's selected tools directly
                        candidates = [
                            {'tool': tool, 'utility': 1.0 - (i * 0.1), 'source': 'llm_classifier'}
                            for i, tool in enumerate(valid_classifier_tools)
                        ]
                        
                        features = self._extract_features(request)
                        request.features = features
                        
                        strategy = self._select_strategy(request, candidates)
                        execution_result = self._execute_portfolio(request, candidates, strategy)
                        final_result = self._postprocess_result(request, execution_result, start_time)
                        
                        if self.config.get("learning_enabled"):
                            self._update_learning(request, final_result)
                        
                        if self.config.get("cache_enabled"):
                            self._cache_result(request, final_result)
                        
                        self._update_statistics(final_result)
                        
                        logger.info(f"[ToolSelector] Executed with classifier's tools: {valid_classifier_tools}")
                        return final_result

            # ================================================================
            # BUG #6 FIX: Router suggestions should be INPUT to selection, not override
            # ================================================================
            # Previously: Router pre-selected tools completely bypassed SemanticBoost
            # Now: Router suggestions are HIGH-WEIGHT candidates that still go through
            # the normal selection flow (SemanticBoost, bandit, etc.)
            # 
            # Priority order (BUG #6 FIX):
            # 1. SemanticBoost (learned from success patterns) - HIGHEST
            # 2. LLM Classifier (understands query semantics)
            # 3. Router keywords (suggestion only, not override) - LOWEST
            # ================================================================
            router_suggestions = []
            if hasattr(request, 'context') and isinstance(request.context, dict):
                # Try multiple sources for router-selected tools:
                routing_plan = request.context.get('routing_plan', {})
                routing_tools = None
                task_type = request.context.get('task_type') or request.context.get('query_type')
                
                # Source 1: routing_plan dict with 'tools' key
                if isinstance(routing_plan, dict) and routing_plan.get('tools'):
                    routing_tools = routing_plan.get('tools')
                    logger.debug(f"[ToolSelector] Found tools in routing_plan dict: {routing_tools}")
                
                # Source 2: Direct routing_plan_tools, router_tools, or selected_tools keys
                if not routing_tools:
                    routing_tools = (
                        request.context.get('routing_plan_tools') or 
                        request.context.get('router_tools') or
                        request.context.get('selected_tools')
                    )
                
                # Source 3: routing_plan object with telemetry_data attribute
                if not routing_tools and hasattr(routing_plan, 'telemetry_data'):
                    routing_tools = routing_plan.telemetry_data.get('selected_tools', [])
                
                # Source 4: routing_plan object with selected_tools attribute
                if not routing_tools and hasattr(routing_plan, 'selected_tools'):
                    routing_tools = routing_plan.selected_tools
                
                if routing_tools and isinstance(routing_tools, (list, tuple)) and len(routing_tools) > 0:
                    # BUG #6 FIX: Store router suggestions as hints, don't bypass selection
                    available_tools = getattr(self, 'available_tools', None) or DEFAULT_AVAILABLE_TOOLS
                    router_suggestions = [t for t in routing_tools if t in available_tools]
                    
                    if router_suggestions:
                        logger.info(
                            f"[ToolSelector] BUG #6 FIX: Router suggests tools: {router_suggestions} "
                            f"for task_type={task_type} (will be used as weighted hints, not bypassing selection)"
                        )
                        # Store in context for use during candidate generation
                        if not hasattr(request, 'context') or not isinstance(request.context, dict):
                            request.context = {}
                        request.context['router_suggestions'] = router_suggestions
                        request.context['router_suggestion_boost'] = 0.3  # Moderate boost, not override

            # Step 1: Admission control
            admitted, admission_info = self._check_admission(request)
            if not admitted:
                return self._create_rejection_result(
                    admission_info.get("reason", "Unknown")
                )

            # Step 2: Check cache
            cached_result = self._check_cache(request)
            if cached_result:
                return cached_result

            # Step 3: Feature extraction
            features = self._extract_features(request)
            request.features = features
            
            # DIAGNOSTIC LOGGING: Log extracted features (first 5 elements for arrays)
            try:
                # Industry Standard: Simple type checking with early exit
                if hasattr(features, '__len__') and len(features) > 5:
                    features_preview = str(features[:5]) + '...'
                else:
                    features_preview = str(features)
                logger.info(f"[ToolSelector] Query features: {features_preview}")
            except Exception:
                # Defensive: Don't let logging errors break the flow
                logger.info(f"[ToolSelector] Query features: <unavailable>")
            
            logger.info(f"[ToolSelector] Available tools: {self.tool_names}")

            # Step 4: Safety pre-check
            safety_context = self._create_safety_context(request)
            safe_candidates = self._safety_precheck(safety_context)
            if not safe_candidates:
                return self._create_safety_veto_result()

            # Step 5: Value of Information check
            should_refine, voi_action = self._check_voi(request, features)
            if should_refine:
                features = self._refine_features(features, voi_action)
                request.features = features

            # Step 6: Compute prior probabilities
            # CRITICAL: Include query text for semantic tool matching
            prior_context = {}
            if hasattr(request, 'context') and request.context:
                prior_context = request.context.copy() if isinstance(request.context, dict) else {}
            
            # ================================================================
            # Note: Skip SemanticBoost if LLM classifier is authoritative
            # When classifier identifies category with high confidence, its tool selection
            # is authoritative and should not be overridden by semantic matching
            # (which uses embeddings, not language understanding).
            #
            # FIX: Added ANALOGICAL, PHILOSOPHICAL, CAUSAL categories to prevent
            # semantic boost from overriding the classifier's decision. The classifier
            # uses keyword patterns and language understanding to identify these categories,
            # which is more reliable than embedding similarity for distinguishing between:
            # - "Quantum physics is like a symphony" (ANALOGICAL - uses "quantum" but is analogy)
            # - "Calculate quantum probability" (PROBABILISTIC - actual math query)
            # ================================================================
            skip_semantic_boost = False
            classifier_category = None
            
            if hasattr(request, 'context') and isinstance(request.context, dict):
                classifier_category = request.context.get('classifier_category')
                classifier_is_authoritative = request.context.get('classifier_is_authoritative', False)
                prevent_router_override = request.context.get('prevent_router_tool_override', False)
                # Note: Default to None to distinguish "not provided" from "provided as 0.0"
                # This allows the confidence check to be skipped when no confidence is available
                classifier_confidence = request.context.get('classifier_confidence')
                
                # For these categories, the LLM's language understanding is more reliable
                # than semantic embedding similarity. Normalize to uppercase for comparison.
                # FIX: Added ANALOGICAL, PHILOSOPHICAL, CAUSAL, PROBABILISTIC to prevent
                # domain keywords (quantum, welfare) from overriding correct classification.
                AUTHORITATIVE_CATEGORIES = frozenset([
                    'UNKNOWN', 'CREATIVE', 'CONVERSATIONAL', 'GENERAL',
                    'GREETING', 'FACTUAL', 'SELF_INTROSPECTION',
                    # FIX: These categories should also be authoritative when classifier is confident
                    'ANALOGICAL', 'PHILOSOPHICAL', 'CAUSAL', 'PROBABILISTIC',
                    'MATHEMATICAL', 'LOGICAL', 'CRYPTOGRAPHIC',
                ])
                
                # Threshold for confidence-based semantic boost skip
                # When classifier confidence is at or above this threshold, trust the classifier
                CONFIDENCE_THRESHOLD_FOR_SKIP = 0.8
                
                # Normalize category to uppercase for comparison
                category_upper = classifier_category.upper() if classifier_category else None
                
                # Skip semantic boost if:
                # 1. Category is in authoritative list, OR
                # 2. Classifier explicitly marked as authoritative, OR
                # 3. Router override is prevented, OR
                # 4. Classifier confidence is high (>= threshold) - only check if confidence was provided
                confidence_is_high = (
                    classifier_confidence is not None and 
                    classifier_confidence >= CONFIDENCE_THRESHOLD_FOR_SKIP
                )
                should_skip = (
                    category_upper in AUTHORITATIVE_CATEGORIES or
                    classifier_is_authoritative or
                    prevent_router_override or
                    confidence_is_high
                )
                
                if should_skip:
                    skip_semantic_boost = True
                    prior_context['skip_semantic_boost'] = True
                    conf_str = f"{classifier_confidence:.2f}" if classifier_confidence is not None else "N/A"
                    logger.info(
                        f"[ToolSelector] Skipping SemanticBoost: LLM classifier is authoritative "
                        f"for category={classifier_category} (confidence={conf_str})"
                    )
            
            # Extract query text from problem for semantic matching (if not skipping)
            query_text = None
            
            if not skip_semantic_boost:
                # Source 1: request.context
                if hasattr(request, 'context') and isinstance(request.context, dict):
                    query_text = request.context.get('query')
                
                # Source 2: request.problem (string)
                if not query_text and hasattr(request, 'problem'):
                    if isinstance(request.problem, str):
                        query_text = request.problem
                    elif isinstance(request.problem, dict):
                        query_text = (
                            request.problem.get('text') or 
                            request.problem.get('query') or 
                            request.problem.get('content')
                        )
                
                # Source 3: request.query directly
                if not query_text and hasattr(request, 'query'):
                    query_text = request.query
                
                if query_text:
                    prior_context['query'] = str(query_text)
                    # Log only query length to avoid exposing sensitive user data
                    logger.info(f"[ToolSelector] Found query for semantic matching (length={len(str(query_text))} chars)")
                else:
                    logger.warning("[ToolSelector] NO QUERY TEXT found - semantic matching will use features only")
                    # Log only safe attributes (type names) to avoid exposing sensitive data
                    safe_attrs = ['problem', 'context', 'query', 'constraints', 'mode', 'available_tools']
                    available_attrs = [attr for attr in safe_attrs if hasattr(request, attr)]
                    logger.debug(f"[ToolSelector] Request has attributes: {available_attrs}")
            
            # DEBUG: Log what we're passing to compute_prior
            logger.info(f"[ToolSelector] Calling compute_prior with context keys: {list(prior_context.keys())}")
            
            prior_dist = self.memory_prior.compute_prior(
                features, safe_candidates, prior_context
            )
            
            # Apply learned weight adjustments from learning system
            if self.learning_system and hasattr(prior_dist, 'tool_probs') and isinstance(prior_dist.tool_probs, dict):
                for tool in prior_dist.tool_probs:
                    adjustment = self.learning_system.get_tool_weight_adjustment(tool)
                    if adjustment != 0:
                        prior_dist.tool_probs[tool] += adjustment
                        logger.info(f"[ToolSelector] Applied learned adjustment to '{tool}': {adjustment:+.3f}")
                # Ensure no negative probabilities and renormalize
                for tool in prior_dist.tool_probs:
                    if prior_dist.tool_probs[tool] < 0:
                        prior_dist.tool_probs[tool] = 0.0
                total = sum(prior_dist.tool_probs.values())
                if total > 0:
                    prior_dist.tool_probs = {k: v / total for k, v in prior_dist.tool_probs.items()}
                else:
                    # All weights were zero/negative, reset to uniform
                    n_tools = len(prior_dist.tool_probs)
                    if n_tools > 0:
                        uniform_prob = 1.0 / n_tools
                        prior_dist.tool_probs = {k: uniform_prob for k in prior_dist.tool_probs}
                # Update most likely tool
                if prior_dist.tool_probs:
                    prior_dist.most_likely_tool = max(prior_dist.tool_probs.items(), key=lambda x: x[1])[0]

            # Step 7: Generate candidate tools with utilities
            candidates = self._generate_candidates(
                request, features, safe_candidates, prior_dist
            )

            # Step 7.5: Apply post-semantic safety checks
            # This respects the semantic_boost_applied flag from prior computation
            if self.config.get("safety_enabled") and candidates:
                semantic_boost_applied = prior_dist.metadata.get('semantic_boost_applied', False)
                candidate_tools = [c['tool'] for c in candidates]
                
                # Build context for safety check with semantic boost flag
                safety_context_dict = {
                    'semantic_boost_applied': semantic_boost_applied,
                    'problem': request.problem,
                    'query': prior_context.get('query', ''),
                    'constraints': request.constraints,
                }
                
                # Apply safety checks that respect semantic selection
                final_tools = self.safety_governor.apply_safety_checks(
                    candidate_tools, safety_context_dict
                )
                
                # Filter candidates to only include tools that passed safety
                candidates = [c for c in candidates if c['tool'] in final_tools]
                
                logger.info(f"[ToolSelector] Tool selection complete: tools={final_tools}")

            # Step 8: Select execution strategy
            strategy = self._select_strategy(request, candidates)

            # Step 9: Execute with portfolio executor
            execution_result = self._execute_portfolio(request, candidates, strategy)

            # Step 10: Post-process and validate result
            final_result = self._postprocess_result(
                request, execution_result, start_time
            )

            # Step 11: Update learning components
            if self.config.get("learning_enabled"):
                self._update_learning(request, final_result)

            # Step 12: Cache result
            if self.config.get("cache_enabled"):
                self._cache_result(request, final_result)

            # Step 13: Update statistics
            self._update_statistics(final_result)

            return final_result
        except Exception as e:
            logger.error(f"Selection and execution failed: {e}")
            return self._create_failure_result()

    def _check_admission(
        self, request: SelectionRequest
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check admission control"""

        try:
            return self.admission_control.check_admission(
                problem=request.problem,
                constraints=request.constraints,
                priority=request.priority,
                callback=request.callback,
            )
        except Exception as e:
            logger.error(f"Admission check failed: {e}")
            return False, {"reason": f"error: {str(e)}"}

    def _check_cache(self, request: SelectionRequest) -> Optional[SelectionResult]:
        """Check if result is cached"""

        if not self.config.get("cache_enabled"):
            return None

        try:
            # Check selection cache
            if request.features is not None:
                cached = self.cache.get_cached_selection(
                    request.features, request.constraints
                )
                if cached:
                    tool = cached["tool"]

                    # Check result cache
                    cached_result = self.cache.get_cached_result(tool, request.problem)
                    if cached_result:
                        return SelectionResult(
                            selected_tool=tool,
                            execution_result=cached_result["result"],
                            confidence=cached.get("confidence", 0.5),
                            calibrated_confidence=cached.get("confidence", 0.5),
                            execution_time_ms=cached_result["execution_time"],
                            energy_used_mj=cached_result["energy"],
                            strategy_used=ExecutionStrategy.SINGLE,
                            all_results={tool: cached_result["result"]},
                            metadata={"cache_hit": True},
                        )
        except Exception as e:
            logger.error(f"Cache check failed: {e}")

        return None

    def _extract_features(self, request: SelectionRequest) -> np.ndarray:
        """Extract features from problem"""

        try:
            if request.features is not None:
                return request.features

            # Check feature cache
            cached_features = self.cache.get_cached_features(request.problem)
            if cached_features is not None:
                return cached_features

            # Extract features with appropriate tier
            time_budget = request.constraints.get("time_budget_ms", 5000)

            if request.mode == SelectionMode.FAST:
                features = self.feature_extractor.extract_tier1(request.problem)
            elif request.mode == SelectionMode.ACCURATE:
                features = self.feature_extractor.extract_tier3(request.problem)
            else:
                features = self.feature_extractor.extract_adaptive(
                    request.problem,
                    time_budget * 0.02,  # Use 2% of budget for extraction
                )

            # ================================================================
            # Note: Stricter multimodal detection
            # Only trigger multimodal when ACTUAL multimodal data is present,
            # not just keyword mentions like "image" or "picture" in text queries.
            # This prevents false positives like "2+2" triggering multimodal boost
            # because the context flag was set incorrectly.
            # ================================================================
            is_multimodal = False
            if isinstance(request.problem, dict):
                # Check for actual multimodal data (binary content, URLs, base64)
                multimodal_data_keys = ['image', 'images', 'audio', 'video', 'file', 'attachment']
                for key in multimodal_data_keys:
                    if key in request.problem:
                        value = request.problem[key]
                        # Only count as multimodal if there's actual data, not just a key
                        if value is not None and value != '' and value != []:
                            # Check if it looks like actual data (bytes, URL, base64, or non-empty list)
                            if isinstance(value, (bytes, bytearray)):
                                is_multimodal = True
                                break
                            elif isinstance(value, str) and len(value) > MULTIMODAL_MIN_URL_LENGTH:
                                # Likely a URL, file path, or base64 data (not just a filename mention)
                                if value.startswith(('http://', 'https://', 'data:', '/', 'file:')) or \
                                   len(value) > MULTIMODAL_MIN_BASE64_LENGTH:  # Base64 data is typically long
                                    is_multimodal = True
                                    break
                            elif isinstance(value, list) and len(value) > 0:
                                # List of images/files
                                is_multimodal = True
                                break
            # NOTE: We intentionally do NOT set is_multimodal based on text keywords
            # like "image", "picture", "photo" in string queries. A query about images
            # (e.g., "How do I process an image?") is NOT the same as a query WITH images.
            # Text-only queries should use text reasoning tools, not multimodal tools.
            
            if is_multimodal:
                request.context = request.context or {}
                request.context['is_multimodal'] = True
                logger.info("[ToolSelector] Multimodal content detected: actual binary/URL data present in request")

            # Cache features
            self.cache.cache_features(request.problem, features)

            return features
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            # Use deterministic zeros instead of random features.
            # Random features cause non-deterministic tool selection.
            return np.zeros(128)

    def _create_safety_context(self, request: SelectionRequest) -> SafetyContext:
        """Create safety context from request"""

        return SafetyContext(
            problem=request.problem,
            tool_name="",  # Will be filled per tool
            features=request.features,
            constraints=request.constraints,
            user_context=request.context,
            safety_level=request.safety_level,
        )

    def _safety_precheck(self, context: SafetyContext) -> List[str]:
        """Pre-check which tools are safe to use.
        
        This uses critical-only safety checks to allow semantic matching
        to consider all viable tools. Resource constraints are checked
        after semantic matching selects tools.
        """

        if not self.config.get("safety_enabled"):
            return self.tool_names

        try:
            safe_tools = []

            for tool_name in self.tool_names:
                context.tool_name = tool_name
                # Use critical-only check for initial filtering
                action, reason = self.safety_governor.check_critical_safety_only(context)

                if action.value in ["allow", "sanitize", "log_and_allow"]:
                    safe_tools.append(tool_name)

            return safe_tools
        except Exception as e:
            logger.error(f"Safety precheck failed: {e}")
            return self.tool_names

    def _check_voi(
        self, request: SelectionRequest, features: np.ndarray
    ) -> Tuple[bool, Optional[str]]:
        """Check value of information for deeper analysis"""

        if not self.config.get("enable_voi"):
            return False, None

        try:
            budget_remaining = {
                "time_ms": request.constraints.get("time_budget_ms", 5000),
                "energy_mj": request.constraints.get("energy_budget_mj", 1000),
            }

            return self.voi_gate.should_probe_deeper(features, None, budget_remaining)
        except Exception as e:
            logger.error(f"VOI check failed: {e}")
            return False, None

    def _refine_features(self, features: np.ndarray, voi_action: str) -> np.ndarray:
        """Refine features based on VOI recommendation"""

        try:
            if voi_action == "tier2_structural":
                return self.feature_extractor.extract_tier2(features)
            elif voi_action == "tier3_semantic":
                return self.feature_extractor.extract_tier3(features)
            elif voi_action == "tier4_multimodal":
                return self.feature_extractor.extract_tier4(features)
            else:
                return features
        except Exception as e:
            logger.error(f"Feature refinement failed: {e}")
            return features

    def _extract_query_text(self, problem: Any) -> str:
        """
        Extract query text from problem for classification.
        
        Args:
            problem: The problem object (can be string, dict, or object with attributes)
            
        Returns:
            Query text string extracted from problem
        """
        if isinstance(problem, str):
            return problem
        elif isinstance(problem, dict):
            # Try common keys
            for key in ['query', 'text', 'question', 'problem', 'input']:
                if key in problem:
                    return str(problem[key])
            return str(problem)
        elif hasattr(problem, 'query'):
            return str(problem.query)
        elif hasattr(problem, 'text'):
            return str(problem.text)
        else:
            return str(problem)[:1000]  # Truncate for safety
    
    def _get_llm_classification(
        self, 
        query_text: str, 
        safe_tools: List[str]
    ) -> Optional[List[str]]:
        """
        Get tool candidates from LLM Router.
        
        Args:
            query_text: The query to classify
            safe_tools: List of tools that passed safety checks
            
        Returns:
            List of candidate tool names, or None if classification failed/low confidence
        """
        if not LLM_CLASSIFICATION_ENABLED or not LLM_ROUTER_AVAILABLE:
            return None
        
        try:
            router = get_llm_router()
            decision = router.route(query_text)
            
            logger.debug(
                f"[ToolSelector] LLM Router: destination={decision.destination}, "
                f"engine={decision.engine}, confidence={decision.confidence:.2f}"
            )
            
            # Check confidence threshold
            if decision.confidence < LLM_CLASSIFICATION_CONFIDENCE_THRESHOLD:
                logger.debug(
                    f"[ToolSelector] LLM confidence {decision.confidence:.2f} "
                    f"below threshold {LLM_CLASSIFICATION_CONFIDENCE_THRESHOLD}, using fallback"
                )
                return None
            
            # Map engine to tool candidate
            candidates = []
            if decision.engine and decision.engine in safe_tools:
                candidates.append(decision.engine)
            
            if not candidates:
                logger.debug(
                    f"[ToolSelector] No LLM-suggested tools in safe_tools list, using fallback"
                )
                return None
            
            # Limit candidates
            candidates = candidates[:CANDIDATE_MAX_COUNT]
            
            logger.info(
                f"[ToolSelector] Using LLM Router: {candidates} "
                f"(destination={decision.destination}, confidence={decision.confidence:.2f})"
            )
            
            return candidates
            
        except Exception as e:
            logger.warning(f"[ToolSelector] LLM Router failed: {e}, using fallback")
            return None
    
    def _generate_candidates(
        self,
        request: SelectionRequest,
        features: np.ndarray,
        safe_tools: List[str],
        prior_dist: Any,
    ) -> List[Dict[str, Any]]:
        """Generate tool candidates filtered by semantic matching prior.
        
        CRITICAL Note: Different reasoning paradigms (causal, symbolic, 
        probabilistic, analogical, multimodal) are COMPLEMENTARY, not redundant.
        They produce different outputs BY DESIGN. We should run the best-matched 
        tool(s), not all 5.
        
        When semantic matching returns {causal: 0.70, symbolic: 0.08, ...}, 
        we only run the clearly winning tool, not all 5 tools.
        
        Integration with LLM Classification:
        1. Extract query text from problem
        2. Call LLM classifier for high-confidence tool suggestions
        3. If confidence >= 0.8, use LLM-suggested tools as primary candidates
        4. Fall back to SemanticToolMatcher + BayesianMemoryPrior if LLM fails/low confidence
        """
        candidates = []

        # ==============================================================================
        # PHASE 1: Try LLM classification first (PRIMARY PATH)
        # ==============================================================================
        try:
            # Extract query text from the problem
            query_text = self._extract_query_text(request.problem)
            
            # Get LLM classification candidates
            llm_candidates = self._get_llm_classification(query_text, safe_tools)
            
            if llm_candidates:
                # LLM classification succeeded with high confidence
                # Build candidate list using LLM-suggested tools
                # 
                # INDUSTRY-STANDARD FIX: Mark that LLM classification was authoritative
                # This allows downstream components to trust the LLM's classification
                # and skip redundant per-engine gate checks
                llm_authoritative = True  # LLM confidence >= 0.8
                
                for tool_name in llm_candidates:
                    cost_dist = self.cost_model.predict_cost(tool_name, features)
                    
                    time_budget = request.constraints.get("time_budget_ms", float("inf"))
                    if tool_name == "multimodal":
                        time_budget *= MULTIMODAL_TIME_BUDGET_MULTIPLIER
                    
                    if cost_dist["time"]["mean"] > time_budget:
                        logger.debug(f"Tool {tool_name} filtered: cost > budget")
                        continue
                    if cost_dist["energy"]["mean"] > request.constraints.get("energy_budget_mj", float("inf")):
                        continue
                    
                    # High quality estimate for LLM-selected tools
                    quality_estimate = 0.9  # LLM has high confidence
                    
                    candidates.append({
                        "tool": tool_name,
                        "utility": self.utility_model.compute_utility(
                            quality=quality_estimate,
                            time=cost_dist["time"]["mean"],
                            energy=cost_dist["energy"]["mean"],
                            risk=0.1,  # Low risk for LLM-selected tools
                            context={"mode": request.mode.value},
                        ),
                        "quality": quality_estimate,
                        "cost": cost_dist,
                        "prior": 0.9,  # High prior for LLM selection
                        "source": "llm_classification",
                        # INDUSTRY-STANDARD FIX: Add metadata for skip_gate_check
                        "skip_gate_check": True,  # LLM was authoritative
                        "llm_authoritative": True,
                        "llm_confidence": 0.9,  # High confidence (>= 0.8)
                    })
                
                if candidates:
                    logger.info(
                        f"[ToolSelector] Using {len(candidates)} LLM-classified candidates: "
                        f"{[c['tool'] for c in candidates]}"
                    )
                    candidates.sort(key=lambda x: x["utility"], reverse=True)
                    return candidates
                else:
                    logger.debug("[ToolSelector] LLM candidates filtered by budget, falling back")
        except Exception as e:
            logger.warning(f"[ToolSelector] LLM classification error: {e}, using fallback")

        # ==============================================================================
        # PHASE 2: Fallback to SemanticToolMatcher + BayesianMemoryPrior
        # ==============================================================================
        try:
            tool_priors = prior_dist.tool_probs if hasattr(prior_dist, 'tool_probs') and prior_dist.tool_probs else {}
            
            if not tool_priors:
                # Fallback: if no priors, just use first safe tool
                if safe_tools:
                    cost_dist = self.cost_model.predict_cost(safe_tools[0], features)
                    return [{"tool": safe_tools[0], "utility": 0.5, "quality": 0.5, 
                             "cost": cost_dist, "prior": 0.2}]
                return []
            
            # Sort tools by prior probability
            sorted_tools = sorted(tool_priors.items(), key=lambda x: x[1], reverse=True)
            
            # CRITICAL: If one tool dominates (2x the next), just use that one
            if len(sorted_tools) >= 2:
                top_tool, top_prior = sorted_tools[0]
                second_tool, second_prior = sorted_tools[1]
                
                if top_prior >= second_prior * CANDIDATE_DOMINANCE_RATIO:
                    # Clear winner - only run this tool
                    logger.info(f"[ToolSelector] Clear winner: {top_tool} ({top_prior:.3f}) >> {second_tool} ({second_prior:.3f})")
                    
                    if top_tool in safe_tools:
                        cost_dist = self.cost_model.predict_cost(top_tool, features)
                        return [{
                            "tool": top_tool,
                            "utility": 0.5 + top_prior,
                            "quality": 0.5 + top_prior,
                            "cost": cost_dist,
                            "prior": top_prior,
                        }]
            
            # No clear winner - take top N tools above threshold
            viable_tools = [
                (tool, prior) for tool, prior in sorted_tools
                if tool in safe_tools and prior >= CANDIDATE_PRIOR_THRESHOLD
            ][:CANDIDATE_MAX_COUNT]
            
            if not viable_tools:
                # Fallback to top tool even if below threshold
                for tool, prior in sorted_tools:
                    if tool in safe_tools:
                        viable_tools = [(tool, prior)]
                        break
            
            logger.info(f"[ToolSelector] Selected {len(viable_tools)} from {len(safe_tools)}: {[t[0] for t in viable_tools]}")
            
            # Build candidate list with cost checking
            for tool_name, prior in viable_tools:
                cost_dist = self.cost_model.predict_cost(tool_name, features)
                
                time_budget = request.constraints.get("time_budget_ms", float("inf"))
                if tool_name == "multimodal":
                    time_budget *= MULTIMODAL_TIME_BUDGET_MULTIPLIER
                
                if cost_dist["time"]["mean"] > time_budget:
                    logger.debug(f"Tool {tool_name} filtered: cost > budget")
                    continue
                if cost_dist["energy"]["mean"] > request.constraints.get("energy_budget_mj", float("inf")):
                    continue
                
                quality_estimate = 0.5 + prior
                
                # FIX: Reduce quality/utility for meta/world_model tools when domain-specific tools exist
                # This ensures domain-specific engines are preferred over meta-reasoning
                is_meta_tool = tool_name in ('world_model', 'philosophical')
                has_domain_tools = any(t in safe_tools for t in ('symbolic', 'mathematical', 'probabilistic', 'causal'))
                
                if is_meta_tool and has_domain_tools:
                    # Reduce quality estimate when domain tools are available
                    quality_estimate *= META_TOOL_QUALITY_PENALTY
                    logger.debug(
                        f"[ToolSelector] Meta tool penalty applied to {tool_name}: "
                        f"quality reduced by {(1-META_TOOL_QUALITY_PENALTY)*100:.0f}% when domain tools available"
                    )
                
                candidates.append({
                    "tool": tool_name,
                    "utility": self.utility_model.compute_utility(
                        quality=quality_estimate,
                        time=cost_dist["time"]["mean"],
                        energy=cost_dist["energy"]["mean"],
                        risk=max(0.0, 0.5 - prior),  # Ensure non-negative risk
                        context={"mode": request.mode.value},
                    ),
                    "quality": quality_estimate,
                    "cost": cost_dist,
                    "prior": prior,
                })
            
            # BUG #6 FIX: Apply router suggestion boost (as ONE input, not override)
            # This gives router suggestions a moderate boost, but SemanticBoost results
            # still have priority if they score higher
            if hasattr(request, 'context') and isinstance(request.context, dict):
                router_suggestions = request.context.get('router_suggestions', [])
                router_boost = request.context.get('router_suggestion_boost', 0.2)
                
                if router_suggestions:
                    for candidate in candidates:
                        if candidate['tool'] in router_suggestions:
                            original_utility = candidate['utility']
                            candidate['utility'] += router_boost
                            candidate['router_boosted'] = True
                            logger.debug(
                                f"[ToolSelector] BUG #6 FIX: Router boost applied to {candidate['tool']}: "
                                f"{original_utility:.3f} -> {candidate['utility']:.3f}"
                            )
            
            candidates.sort(key=lambda x: x["utility"], reverse=True)
            
        except Exception as e:
            logger.error(f"Candidate generation failed: {e}")

        return candidates

    def _select_strategy(
        self, request: SelectionRequest, candidates: List[Dict[str, Any]]
    ) -> ExecutionStrategy:
        """Select execution strategy - prefer SINGLE tool for different reasoning paradigms.
        
        Note: Different reasoning types (causal, symbolic, etc.) are 
        complementary, not redundant. Running multiple and checking "consensus" 
        is wrong - they SHOULD differ. Prefer SINGLE tool in most cases.
        """
        if not candidates:
            return ExecutionStrategy.SINGLE
        
        # With 1 candidate, always SINGLE
        if len(candidates) == 1:
            return ExecutionStrategy.SINGLE
        
        # With 2+ candidates, check if top one dominates
        if len(candidates) >= 2:
            top_prior = candidates[0].get("prior", 0)
            second_prior = candidates[1].get("prior", 0)
            
            if top_prior > second_prior * 1.5:
                logger.info(f"[ToolSelector] Using SINGLE: {candidates[0]['tool']} dominates")
                return ExecutionStrategy.SINGLE
        
        # Only use multi-tool strategies in specific modes
        if request.mode == SelectionMode.ACCURATE and len(candidates) >= 2:
            # Run top 2 and pick best result (not consensus!)
            return ExecutionStrategy.SPECULATIVE_PARALLEL
        
        if request.mode == SelectionMode.SAFE and len(candidates) >= 2:
            return ExecutionStrategy.SEQUENTIAL_REFINEMENT
        
        # Default: just run the best tool
        return ExecutionStrategy.SINGLE

    def _execute_with_selected_tools(
        self,
        request: SelectionRequest,
        candidates: List[Dict[str, Any]],
        features: np.ndarray,
        start_time: float,
    ) -> SelectionResult:
        """Execute reasoning with pre-selected tools (e.g., from keyword override).
        
        This method is called when tools are selected via keyword patterns
        or other override mechanisms, bypassing the normal candidate generation.
        It uses the existing execution pipeline to run the selected tools.
        
        Args:
            request: The selection request
            candidates: List of pre-selected tool candidates with utilities
            features: Extracted features for the request
            start_time: Start time for tracking execution duration
            
        Returns:
            SelectionResult with execution results
        """
        try:
            # Ensure features are set on request
            request.features = features
            
            # Select execution strategy based on candidates
            strategy = self._select_strategy(request, candidates)
            
            # Execute the portfolio with selected tools
            execution_result = self._execute_portfolio(request, candidates, strategy)
            
            # Post-process the results
            final_result = self._postprocess_result(request, execution_result, start_time)
            
            # Update learning system if enabled
            if self.config.get("learning_enabled"):
                self._update_learning(request, final_result)
            
            # Cache result if enabled
            if self.config.get("cache_enabled"):
                self._cache_result(request, final_result)
            
            # Update statistics
            self._update_statistics(final_result)
            
            logger.info(
                f"[ToolSelector] Executed with pre-selected tools: "
                f"{[c['tool'] for c in candidates]} (confidence={final_result.calibrated_confidence:.3f})"
            )
            
            return final_result
            
        except Exception as e:
            logger.error(f"[ToolSelector] Execution with selected tools failed: {e}", exc_info=True)
            return self._create_failure_result()

    def _execute_portfolio(
        self,
        request: SelectionRequest,
        candidates: List[Dict[str, Any]],
        strategy: ExecutionStrategy,
    ) -> Any:
        """Execute tools using portfolio executor.
        
        Note: Limit tools based on strategy to prevent excessive
        multi-tool execution even when candidates are filtered.
        """

        try:
            if not candidates:
                return None

            # CRITICAL FIX: Limit to appropriate number of tools based on strategy
            if strategy == ExecutionStrategy.SINGLE:
                tool_names = [candidates[0]["tool"]]
            elif strategy == ExecutionStrategy.COMMITTEE_CONSENSUS:
                tool_names = [c["tool"] for c in candidates[:3]]  # Max 3 for committee
            else:
                tool_names = [c["tool"] for c in candidates[:2]]  # Max 2 otherwise
            
            logger.info(f"[ToolSelector] Executing {len(tool_names)} tools with {strategy.value}")

            # Create monitor
            monitor = ExecutionMonitor(
                time_budget_ms=request.constraints.get("time_budget_ms", 5000),
                energy_budget_mj=request.constraints.get("energy_budget_mj", 1000),
                min_confidence=request.constraints.get("min_confidence", 0.5),
            )

            # Execute
            return self.portfolio_executor.execute(
                strategy=strategy,
                tool_names=tool_names,
                problem=request.problem,
                constraints=request.constraints,
                monitor=monitor,
            )
        except Exception as e:
            logger.error(f"Portfolio execution failed: {e}")
            return None

    def _postprocess_result(
        self, request: SelectionRequest, execution_result: Any, start_time: float
    ) -> SelectionResult:
        """Post-process and validate execution result"""

        try:
            if execution_result is None:
                return self._create_failure_result()

            # Extract primary tool and result
            primary_tool = (
                execution_result.tools_used[0]
                if execution_result.tools_used
                else "unknown"
            )
            primary_result = execution_result.primary_result

            # Calibrate confidence if enabled
            # Note: Properly extract confidence from engine result
            # The primary_result is often a Dict, not an object with attributes
            confidence = 0.5
            calibrated_confidence = 0.5

            if primary_result:
                # Try to extract confidence from the result
                # Note: Handle both dict and object forms
                if isinstance(primary_result, dict):
                    confidence = primary_result.get("confidence", 0.5)
                elif hasattr(primary_result, "confidence"):
                    confidence = primary_result.confidence
                
                # Note: If confidence is 0.0 or very low, don't override to 0.5
                # This respects the engine's assessment that it couldn't answer
                if confidence <= 0.1:
                    logger.warning(
                        f"[ToolSelector] Engine returned very low confidence ({confidence:.3f}) - "
                        f"respecting engine's assessment that it may not be applicable"
                    )

                if self.config.get("enable_calibration"):
                    calibrated_confidence = self.calibrator.calibrate_confidence(
                        primary_tool, confidence, request.features
                    )
                else:
                    calibrated_confidence = confidence

            # Safety post-check
            if self.config.get("safety_enabled"):
                is_safe, safety_reason = self.safety_governor.validate_output(
                    primary_tool, primary_result, self._create_safety_context(request)
                )

                if not is_safe:
                    logger.warning(f"Output safety violation: {safety_reason}")
                    # Could return safety-filtered result here

            # Check consensus if multiple results
            # FIX: Only check consensus for REDUNDANT tools (same paradigm), not
            # for COMPLEMENTARY tools (different paradigms) which SHOULD give different results.
            if len(execution_result.all_results) > 1:
                # Determine if tools are from the same paradigm or different paradigms
                tool_paradigms = {self._get_tool_paradigm(t) for t in execution_result.tools_used}
                
                if len(tool_paradigms) == 1:
                    # Same paradigm - consensus IS expected (redundant execution)
                    is_consistent, consensus_conf, details = (
                        self.safety_governor.check_consensus(execution_result.all_results)
                    )

                    if not is_consistent and consensus_conf < 0.5:
                        logger.warning(f"Low consensus among redundant tools: {details}")
                else:
                    # Different paradigms - consensus NOT expected (complementary reasoning)
                    # Each paradigm provides different insights, disagreement is normal
                    logger.debug(
                        f"[ToolSelector] Multi-paradigm execution ({tool_paradigms}) - "
                        f"diverse results expected, skipping consensus check"
                    )

            execution_time = (time.time() - start_time) * 1000
            
            # DIAGNOSTIC LOGGING: Log final tool selection
            logger.info(
                f"[ToolSelector] Selected tools: {execution_result.tools_used if execution_result else [primary_tool]}, "
                f"confidence={confidence:.2f}"
            )
            logger.info(
                f"[ToolSelector] Selection reason: primary_tool={primary_tool}, "
                f"execution_time={execution_time:.2f}ms, calibrated_confidence={calibrated_confidence:.2f}"
            )

            return SelectionResult(
                selected_tool=primary_tool,
                execution_result=primary_result,
                confidence=confidence,
                calibrated_confidence=calibrated_confidence,
                execution_time_ms=execution_time,
                energy_used_mj=execution_result.energy_used,
                strategy_used=execution_result.strategy,
                all_results=execution_result.all_results,
                metadata=execution_result.metadata,
            )
        except Exception as e:
            logger.error(f"Result post-processing failed: {e}")
            return self._create_failure_result()

    def _update_learning(self, request: SelectionRequest, result: SelectionResult):
        """
        Update learning components including mathematical accuracy feedback.
        
        Note: Now checks if result was verified before rewarding.
        Note: Now checks if result came from fallback.
        """

        try:
            # Note: Check if result was mathematically verified
            is_verified = False
            if result.metadata:
                math_verification = result.metadata.get("math_verification", {})
                is_verified = math_verification.get("status") == "verified"
            
            # Note: Check if result came from fallback
            is_fallback = False
            if result.metadata:
                is_fallback = result.metadata.get("used_fallback", False)
                # Also check execution result for fallback indicators
                if isinstance(result.execution_result, dict):
                    is_fallback = is_fallback or result.execution_result.get("is_fallback", False)
            
            # Log learning update with verification status
            if is_fallback:
                logger.info(
                    f"[ToolSelector] Learning update for FALLBACK result - reduced reward"
                )
            if not is_verified and result.confidence > 0.7:
                logger.info(
                    f"[ToolSelector] Learning update for UNVERIFIED high-confidence result - reduced reward"
                )
            
            # Update bandit with verification and fallback status
            self.bandit.update_from_execution(
                features=request.features,
                tool_name=result.selected_tool,
                quality=result.confidence,
                time_ms=result.execution_time_ms,
                energy_mj=result.energy_used_mj,
                constraints=request.constraints,
                is_verified=is_verified,
                is_fallback=is_fallback,
            )

            # Update memory prior
            # FIX #3: Changed > 0.5 to >= 0.5 so exactly 0.5 confidence doesn't fail
            self.memory_prior.update(
                features=request.features,
                tool_used=result.selected_tool,
                success=result.confidence >= 0.5,
                confidence=result.calibrated_confidence,
                execution_time=result.execution_time_ms,
                energy_used=result.energy_used_mj,
                context=request.context,
            )

            # Update calibration
            if self.config.get("enable_calibration"):
                self.calibrator.update_calibration(
                    result.selected_tool,
                    result.confidence,
                    result.confidence >= 0.5,  # FIX #3: Changed > to >= for threshold
                )

            # Mathematical verification for probabilistic/Bayesian results
            # This provides accuracy feedback to the learning system
            if self.config.get("enable_math_verification", True):
                self._verify_mathematical_result(request, result)

            # Check for distribution shift
            if self.config.get("enable_distribution_monitoring"):
                if self.distribution_monitor.detect_shift(request.features, result):
                    self._handle_distribution_shift()
        except Exception as e:
            logger.error(f"Learning update failed: {e}")

    def _verify_mathematical_result(
        self, request: SelectionRequest, result: SelectionResult
    ):
        """
        Verify mathematical accuracy of results and provide feedback to learning system.
        
        This method checks if the result contains mathematical/probabilistic content
        and verifies it using the MathematicalVerificationEngine. Errors are reported
        to the learning system to penalize tools that produce mathematical errors.
        
        Critical focus: Detecting specificity/sensitivity confusion in Bayesian reasoning.
        """
        if not self.math_verifier or not MATH_VERIFICATION_AVAILABLE:
            return
        
        tool_name = result.selected_tool
        exec_result = result.execution_result
        
        # Only verify probabilistic/Bayesian results
        if tool_name not in ("probabilistic", "symbolic", "causal"):
            return
        
        try:
            # Check if result contains Bayesian/probability content
            if not isinstance(exec_result, dict):
                return
            
            # Look for Bayesian problem indicators
            has_posterior = "posterior" in exec_result or "probability" in exec_result
            has_prior = "prior" in exec_result
            has_test_metrics = any(
                k in exec_result for k in ["sensitivity", "specificity", "likelihood"]
            )
            
            if not (has_posterior and (has_prior or has_test_metrics)):
                return
            
            # Extract Bayesian problem parameters
            posterior = exec_result.get("posterior") or exec_result.get("probability")
            if posterior is None:
                return
            
            prior = exec_result.get("prior", 0.5)
            sensitivity = exec_result.get("sensitivity")
            specificity = exec_result.get("specificity")
            likelihood = exec_result.get("likelihood")
            
            # Create Bayesian problem for verification
            problem = BayesianProblem(
                prior=float(prior),
                likelihood=float(likelihood) if likelihood else None,
                sensitivity=float(sensitivity) if sensitivity else None,
                specificity=float(specificity) if specificity else None,
            )
            
            # Verify the calculation
            verification_result = self.math_verifier.verify_bayesian_calculation(
                problem, float(posterior)
            )
            
            # Update metrics
            with self.stats_lock:
                metrics = self.math_accuracy_metrics[tool_name]
                metrics["verifications"] += 1
                
                if verification_result.status == MathVerificationStatus.VERIFIED:
                    metrics["verified_correct"] += 1
                    # Reward tool for correct mathematical result
                    if self.learning_system:
                        self._apply_math_reward(tool_name)
                    logger.info(
                        f"[MathVerify] Tool '{tool_name}' VERIFIED correct Bayesian result"
                    )
                elif verification_result.status == MathVerificationStatus.ERROR_DETECTED:
                    metrics["errors_detected"] += 1
                    for error in verification_result.errors:
                        metrics["error_types"][error.value] += 1
                    
                    # Penalize tool for mathematical error
                    if self.learning_system:
                        self._apply_math_penalty(
                            tool_name, 
                            verification_result.errors[0] if verification_result.errors else None
                        )
                    
                    logger.warning(
                        f"[MathVerify] Tool '{tool_name}' ERROR: {verification_result.explanation}"
                    )
                    
                    # Add correction info to result metadata
                    if hasattr(result, 'metadata') and result.metadata is not None:
                        result.metadata["math_verification"] = {
                            "status": "error_detected",
                            "errors": [e.value for e in verification_result.errors],
                            "corrections": verification_result.corrections,
                            "explanation": verification_result.explanation,
                        }
                    
                    # Note: Apply correction to execution result
                    # If verification detected an error and we have a correct value,
                    # update the result to use the corrected value instead of the wrong one.
                    # This ensures downstream consumers get the mathematically correct answer.
                    #
                    # THREAD-SAFETY FIX: Create a copy of the dictionary to avoid concurrent
                    # modification issues. The correction is stored in metadata as well.
                    if verification_result.corrections and "correct_posterior" in verification_result.corrections:
                        correct_value = verification_result.corrections["correct_posterior"]
                        
                        # Update the execution result with corrected value
                        if isinstance(exec_result, dict):
                            # Create a copy for thread-safe modification
                            corrected_result = dict(exec_result)
                            
                            # Preserve original for audit, add correction
                            corrected_result["original_posterior"] = posterior
                            corrected_result["corrected_posterior"] = correct_value
                            corrected_result["math_corrected"] = True
                            
                            # Replace the primary value with the corrected one
                            if "posterior" in corrected_result:
                                corrected_result["posterior"] = correct_value
                            elif "probability" in corrected_result:
                                corrected_result["probability"] = correct_value
                            
                            # Update the result object with the corrected data
                            result.execution_result = corrected_result
                            
                            logger.info(
                                f"[MathVerify] CORRECTED result: {posterior:.6f} -> {correct_value:.6f}"
                            )
                        
        except Exception as e:
            logger.debug(f"Mathematical verification skipped: {e}")

    def _apply_math_reward(self, tool_name: str):
        """Apply reward to tool for correct mathematical result."""
        if not self.learning_system:
            return
        
        try:
            if hasattr(self.learning_system, '_weight_lock'):
                with self.learning_system._weight_lock:
                    if tool_name not in self.learning_system.tool_weight_adjustments:
                        self.learning_system.tool_weight_adjustments[tool_name] = 0.0
                    
                    # Reward for mathematical correctness (0.015)
                    reward = 0.015
                    old_weight = self.learning_system.tool_weight_adjustments[tool_name]
                    self.learning_system.tool_weight_adjustments[tool_name] = min(
                        0.2,  # MAX_TOOL_WEIGHT
                        old_weight + reward
                    )
                    logger.info(
                        f"[MathVerify] Rewarded '{tool_name}': {old_weight:.4f} -> "
                        f"{self.learning_system.tool_weight_adjustments[tool_name]:.4f}"
                    )
        except Exception as e:
            logger.warning(f"Failed to apply math reward: {e}")

    def _apply_math_penalty(self, tool_name: str, error_type: Optional["MathErrorType"]):
        """Apply penalty to tool for mathematical error."""
        if not self.learning_system:
            return
        
        try:
            # Penalty varies by error severity
            penalty_map = {
                "specificity_confusion": -0.02,
                "base_rate_neglect": -0.015,
                "complement_error": -0.01,
                "arithmetic_error": -0.008,
            }
            
            error_name = error_type.value if error_type else "unknown"
            penalty = penalty_map.get(error_name, -0.01)
            
            if hasattr(self.learning_system, '_weight_lock'):
                with self.learning_system._weight_lock:
                    if tool_name not in self.learning_system.tool_weight_adjustments:
                        self.learning_system.tool_weight_adjustments[tool_name] = 0.0
                    
                    old_weight = self.learning_system.tool_weight_adjustments[tool_name]
                    self.learning_system.tool_weight_adjustments[tool_name] = max(
                        -0.1,  # MIN_TOOL_WEIGHT
                        old_weight + penalty
                    )
                    logger.warning(
                        f"[MathVerify] Penalized '{tool_name}' for {error_name}: "
                        f"{old_weight:.4f} -> {self.learning_system.tool_weight_adjustments[tool_name]:.4f}"
                    )
        except Exception as e:
            logger.warning(f"Failed to apply math penalty: {e}")

    def _cache_result(self, request: SelectionRequest, result: SelectionResult):
        """Cache selection and result"""

        try:
            # Cache selection decision
            self.cache.cache_selection(
                features=request.features,
                constraints=request.constraints,
                selection=result.selected_tool,
                confidence=result.calibrated_confidence,
            )

            # Cache execution result
            self.cache.cache_result(
                tool=result.selected_tool,
                problem=request.problem,
                result=result.execution_result,
                execution_time=result.execution_time_ms,
                energy=result.energy_used_mj,
            )
        except Exception as e:
            logger.error(f"Result caching failed: {e}")

    def _update_statistics(self, result: SelectionResult):
        """Update performance statistics and record implicit feedback"""

        try:
            with self.stats_lock:
                tool_stats = self.performance_metrics[result.selected_tool]
                tool_stats["count"] += 1

                # FIX #3: Changed > 0.5 to >= 0.5 so exactly 0.5 confidence counts as success
                if result.confidence >= 0.5:
                    tool_stats["successes"] += 1

                # Update running averages
                alpha = 0.1  # Exponential moving average
                tool_stats["avg_time"] = (1 - alpha) * tool_stats[
                    "avg_time"
                ] + alpha * result.execution_time_ms
                tool_stats["avg_energy"] = (1 - alpha) * tool_stats[
                    "avg_energy"
                ] + alpha * result.energy_used_mj
                tool_stats["avg_confidence"] = (1 - alpha) * tool_stats[
                    "avg_confidence"
                ] + alpha * result.calibrated_confidence

                # Add to history
                self.execution_history.append(
                    {
                        "timestamp": time.time(),
                        "tool": result.selected_tool,
                        "confidence": result.calibrated_confidence,
                        "time_ms": result.execution_time_ms,
                        "energy_mj": result.energy_used_mj,
                        "strategy": result.strategy_used.value,
                    }
                )
            
            # Record implicit feedback to outcome bridge for learning system
            # This enables CuriosityEngine to learn from tool selection outcomes
            self._record_implicit_feedback(result)
                
        except Exception as e:
            logger.error(f"Statistics update failed: {e}")

    def _record_implicit_feedback(self, result: SelectionResult):
        """
        Record implicit feedback to the outcome bridge for learning.
        
        This enables the CuriosityEngine and UnifiedLearningSystem to learn from
        tool selection outcomes. The feedback includes:
        - Response latency (fast = good, slow = needs improvement)
        - Confidence scores (high confidence = successful selection)
        - Tool used (enables tool selection pattern analysis)
        
        Implicit signals captured:
        1. Latency: < 5s = good, > 30s = needs improvement
        2. Confidence: > 0.7 = success, < 0.3 = failure
        3. Strategy: single tool = simple query, portfolio = complex query
        
        Note: Status should reflect whether the tool EXECUTED successfully,
        not whether different reasoning paradigms "agreed" (they SHOULD differ).
        """
        if not OUTCOME_BRIDGE_AVAILABLE or record_query_outcome is None:
            return
        
        try:
            # Generate a unique query ID for this outcome
            query_id = f"tool_sel_{uuid.uuid4().hex[:12]}"
            
            # Note: Determine status based on execution success, not consensus
            # A tool selection is successful if:
            # 1. A tool was selected
            # 2. The tool executed and produced a result
            # 3. Confidence is above minimum threshold (not necessarily high)
            
            # Check if we have a valid result at all
            has_result = result is not None and hasattr(result, 'selected_tool') and result.selected_tool
            
            # A minimum confidence is acceptable - different paradigms may have different scales
            min_acceptable_confidence = 0.3  # Lowered from SUCCESS_CONFIDENCE_THRESHOLD
            
            is_success = (
                has_result 
                and result.confidence >= min_acceptable_confidence
                and result.execution_time_ms < MAX_SUCCESS_TIME_MS
            )
            
            # Partial success: tool ran but took too long or had low confidence
            if has_result and not is_success:
                if result.execution_time_ms >= MAX_SUCCESS_TIME_MS:
                    status = "slow"  # Not an error, just slow
                elif result.confidence < min_acceptable_confidence:
                    status = "low_confidence"  # Tool ran, just low certainty
                else:
                    status = "partial"
            elif is_success:
                status = "success"
            else:
                status = "no_result"
            
            # Determine error type only for actual failures
            error_type = None
            if status in ("low_confidence", "slow"):
                error_type = status  # Use status as error type for tracking
            elif status == "no_result":
                error_type = "execution_failed"
            
            # Estimate complexity from the strategy used
            # Single tool = simpler query, portfolio = complex query
            complexity = 0.3  # Default
            if hasattr(result, 'strategy_used'):
                if result.strategy_used.value == "single":
                    complexity = 0.2
                elif result.strategy_used.value == "racing":
                    complexity = 0.5
                elif result.strategy_used.value == "parallel":
                    complexity = 0.6
                elif result.strategy_used.value == "sequential_fallback":
                    complexity = 0.7
            
            # Extract response text from execution result for quality assessment
            # The execution_result may be a dict with various keys depending on the engine
            response_text = None
            if result.execution_result and isinstance(result.execution_result, dict):
                # Try common response keys in order of preference
                for key in ("response", "answer", "result", "output", "text", "explanation"):
                    if key in result.execution_result:
                        val = result.execution_result[key]
                        if isinstance(val, str):
                            response_text = val
                            break
                        elif val is not None:
                            response_text = str(val)
                            break
                
                # Only build diagnostic response_text when there's an error to report
                # This enables quality assessment to detect parse errors and failures.
                # We DON'T build it for successful results (proven: True) because those
                # would be detected as raw data dumps. Successful results without
                # explanation text should be left as response_text=None ("unknown" quality).
                if response_text is None:
                    error_msg = result.execution_result.get("error", "")
                    proven = result.execution_result.get("proven")
                    
                    # Only create diagnostic string if:
                    # 1. There's an explicit error message, OR
                    # 2. proven is explicitly False (not just missing)
                    if error_msg:
                        response_text = f"Error: {error_msg}"
                    elif proven is False:  # Explicit False, not None
                        confidence_val = result.execution_result.get("confidence", 0)
                        method = result.execution_result.get("method", "unknown")
                        response_text = f"Failed to prove. confidence: {confidence_val}, method: {method}"
            
            # Record outcome to bridge with quality assessment data
            record_query_outcome(
                query_id=query_id,
                status=status,
                routing_time_ms=0.0,  # Not applicable for tool selection
                total_time_ms=result.execution_time_ms,
                complexity=complexity,
                query_type=f"reasoning_{result.selected_tool}",
                tasks=1,
                error_type=error_type,
                tools=[result.selected_tool] if result.selected_tool else [],
                response_text=response_text,
                confidence=result.confidence,
            )
            
            logger.debug(
                f"[ImplicitFeedback] Recorded outcome: tool={result.selected_tool}, "
                f"status={status}, confidence={result.confidence:.2f}, "
                f"time={result.execution_time_ms:.0f}ms, has_response={response_text is not None}"
            )
            
        except Exception as e:
            # Don't fail the main flow if feedback recording fails
            logger.debug(f"Implicit feedback recording failed (non-critical): {e}")

    def _handle_distribution_shift(self):
        """Handle detected distribution shift"""

        try:
            logger.warning("Distribution shift detected")

            # Increase exploration
            if hasattr(self.bandit, "increase_exploration"):
                self.bandit.increase_exploration()

            # Clear caches
            self.cache.feature_cache.l1.clear()
            self.cache.selection_cache.l1.clear()

            # Could trigger retraining here
        except Exception as e:
            logger.error(f"Distribution shift handling failed: {e}")

    def _create_rejection_result(self, reason: str) -> SelectionResult:
        """Create result for rejected request"""

        return SelectionResult(
            selected_tool="none",
            execution_result=None,
            confidence=0.0,
            calibrated_confidence=0.0,
            execution_time_ms=0.0,
            energy_used_mj=0.0,
            strategy_used=ExecutionStrategy.SINGLE,
            all_results={},
            metadata={"rejection_reason": reason},
        )

    def _create_safety_veto_result(self) -> SelectionResult:
        """Create result for safety veto"""

        return SelectionResult(
            selected_tool="none",
            execution_result=None,
            confidence=0.0,
            calibrated_confidence=0.0,
            execution_time_ms=0.0,
            energy_used_mj=0.0,
            strategy_used=ExecutionStrategy.SINGLE,
            all_results={},
            metadata={"safety_veto": True},
        )

    def _get_tool_paradigm(self, tool_name: str) -> str:
        """
        Map tool name to its reasoning paradigm.
        
        Tools within the same paradigm are expected to give similar results.
        Tools from different paradigms are EXPECTED to give different results
        (complementary reasoning).
        
        Paradigm categories:
        - logic: Symbolic, formal reasoning (proofs, theorems)
        - probability: Statistical, Bayesian reasoning
        - causality: Causal inference, interventions
        - analogy: Analogical reasoning, structure mapping
        - computation: Mathematical calculations
        - philosophical: Ethical, deontic reasoning
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Paradigm name (string)
        """
        paradigm_map = {
            'symbolic': 'logic',
            'probabilistic': 'probability',
            'bayesian': 'probability',
            'causal': 'causality',
            'analogical': 'analogy',
            'mathematical': 'computation',
            'philosophical': 'philosophical',
            'world_model': 'meta',
            'multimodal': 'multimodal',
        }
        return paradigm_map.get(tool_name.lower(), 'unknown')

    def _create_failure_result(self) -> SelectionResult:
        """Create result for execution failure"""

        return SelectionResult(
            selected_tool="none",
            execution_result=None,
            confidence=0.0,
            calibrated_confidence=0.0,
            execution_time_ms=0.0,
            energy_used_mj=0.0,
            strategy_used=ExecutionStrategy.SINGLE,
            all_results={},
            metadata={"execution_failed": True},
        )

    def record_selection_outcome(self, query: str, selected_tools: List[str], 
                                 success: bool, latency_ms: float):
        """Record tool selection outcome for learning
        
        This method allows the learning system to track tool selection outcomes
        and improve future selections based on historical performance.
        
        Args:
            query: The query text that triggered tool selection
            selected_tools: List of tool names that were selected
            success: Whether the tool selection was successful
            latency_ms: Time taken for tool selection in milliseconds
        """
        if self.learning_system is None:
            return
            
        try:
            from vulcan.learning import TaskInfo
            
            # Use hashlib for consistent, collision-resistant task IDs
            task_hash = hashlib.sha256(query.encode()).hexdigest()[:8]
            
            task_info = TaskInfo(
                task_id=f"tool_selection_{task_hash}",
                task_type="tool_selection",
                difficulty=0.5,
                samples_seen=1,
                performance=1.0 if success else 0.0,
                metadata={
                    'query': query[:200],  # Truncate long queries
                    'tools': selected_tools,
                    'latency_ms': latency_ms,
                }
            )
            
            if hasattr(self.learning_system, 'curriculum_learner') and self.learning_system.curriculum_learner:
                self.learning_system.curriculum_learner.record_task_outcome(
                    task_info, success
                )
        except ImportError:
            logger.debug("TaskInfo not available for outcome recording")
        except Exception as e:
            logger.debug(f"Failed to record selection outcome: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""

        try:
            with self.stats_lock:
                return {
                    "performance_metrics": dict(self.performance_metrics),
                    "cache_stats": self.cache.get_statistics(),
                    "safety_stats": self.safety_governor.get_statistics(),
                    "executor_stats": self.portfolio_executor.get_statistics(),
                    "bandit_stats": (
                        self.bandit.get_statistics()
                        if hasattr(self.bandit, "get_statistics")
                        else {}
                    ),
                    "voi_stats": self.voi_gate.get_statistics(),
                    "total_executions": len(self.execution_history),
                    "recent_executions": list(self.execution_history)[-10:],
                }
        except Exception as e:
            logger.error(f"Get statistics failed: {e}")
            return {}

    def save_state(self, path: str):
        """Save selector state to disk"""

        try:
            save_path = Path(path)
            save_path.mkdir(parents=True, exist_ok=True)

            # Save components
            self.memory_prior.save_state(save_path / "memory_prior")
            self.bandit.save_model(save_path / "bandit")
            self.cache.save_cache(save_path / "cache")
            self.cost_model.save_model(save_path / "cost_model")
            self.calibrator.save_calibration(save_path / "calibration")

            # Save statistics
            with open(save_path / "statistics.json", "w", encoding="utf-8") as f:
                json.dump(self.get_statistics(), f, indent=2, default=str)

            logger.info(f"Tool selector state saved to {save_path}")
        except Exception as e:
            logger.error(f"State save failed: {e}")

    def load_state(self, path: str):
        """Load selector state from disk"""

        try:
            load_path = Path(path)

            if not load_path.exists():
                logger.warning(f"No saved state found at {load_path}")
                return

            # Load components
            if (load_path / "memory_prior").exists():
                self.memory_prior.load_state(load_path / "memory_prior")

            if (load_path / "bandit").exists():
                self.bandit.load_model(load_path / "bandit")

            if (load_path / "cost_model").exists():
                self.cost_model.load_model(load_path / "cost_model")

            if (load_path / "calibration").exists():
                self.calibrator.load_calibration(load_path / "calibration")

            logger.info(f"Tool selector state loaded from {load_path}")
        except Exception as e:
            logger.error(f"State load failed: {e}")

    def shutdown(self, timeout: float = 5.0):
        """Graceful shutdown - CRITICAL: Fast shutdown with interruptible threads"""

        with self.shutdown_lock:
            if self.is_shutdown:
                return
            self.is_shutdown = True

        logger.info("Shutting down tool selector")

        try:
            # Signal all threads to stop immediately
            self._shutdown_event.set()

            # Save state
            self.save_state("./shutdown_state")

            # Shutdown components with timeout
            deadline = time.time() + timeout

            component_timeout = max(0.1, timeout / 4)

            if self.admission_control:
                self.admission_control.shutdown(timeout=component_timeout)

            if self.portfolio_executor:
                self.portfolio_executor.shutdown(timeout=component_timeout)

            if self.cache:
                self.cache.shutdown()

            if self.warm_pool:
                remaining = max(0.1, deadline - time.time())
                self.warm_pool.shutdown(timeout=min(component_timeout, remaining))

            # Shutdown executor - CRITICAL FIX: Remove timeout parameter for Python 3.8 compatibility
            self.executor.shutdown(wait=True)

            logger.info("Tool selector shutdown complete")
        except Exception as e:
            logger.error(f"Shutdown failed: {e}")


# Convenience function for creating selector
def create_tool_selector(config: Optional[Dict[str, Any]] = None) -> ToolSelector:
    """Create and configure tool selector"""
    return ToolSelector(config)
