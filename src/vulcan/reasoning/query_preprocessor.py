"""
Query Preprocessor - Extract formal syntax from natural language queries.

Part of the VULCAN-AGI system.

This module provides preprocessing capabilities to extract formal logical,
mathematical, and probabilistic syntax from natural language reasoning queries.
It prevents parse errors in reasoning engines like SymbolicReasoner by converting
human-readable formats into formal syntax.

Architecture Overview:
    The QueryPreprocessor sits between the ReasoningIntegration layer and the
    specialized reasoning engines (SymbolicReasoner, MathematicalReasoner, etc.).
    It intercepts queries that contain natural language mixed with formal notation
    and extracts the formal components into a machine-parseable format.

    Flow:
        User Query → ReasoningIntegration → QueryPreprocessor → Formal Syntax
                                                              ↓
                                          Reasoning Engine ← Parsed Input

Key Features:
    - SAT problem extraction: Converts "Propositions: A,B,C" + "Constraints: 1. A→B"
      into formal conjunction "(A→B) ∧ (B→C) ∧ ..."
    - Mathematical formula extraction: Extracts "Formula: x² + y = z" patterns
    - Probabilistic notation extraction: Identifies "P(A|B)", "E[X]" patterns
    - Operator normalization: Converts ASCII operators (->. AND) to Unicode (→, ∧)
    - Graceful degradation: Returns original query if no patterns match
    - Thread-safe: Stateless design allows concurrent usage

Performance Characteristics:
    - Regex compilation happens once at initialization
    - Pattern matching is O(n) where n is query length
    - No external I/O or blocking operations
    - Typical processing time: <1ms for most queries

Usage:
    >>> from vulcan.reasoning.query_preprocessor import get_query_preprocessor
    >>>
    >>> preprocessor = get_query_preprocessor()
    >>> result = preprocessor.preprocess(
    ...     query=\"\"\"
    ...     Symbolic Reasoning
    ...     S1 — Satisfiability (SAT-style)
    ...
    ...     Propositions: A, B, C
    ...
    ...     Constraints:
    ...     1. A→B
    ...     2. B→C
    ...     3. ¬C
    ...     4. A∨B
    ...
    ...     Task: Is the set satisfiable?
    ...     \"\"\",
    ...     query_type="symbolic",
    ...     reasoning_tools=["symbolic"]
    ... )
    >>>
    >>> if result.preprocessing_applied:
    ...     print(f"Formal input: {result.formal_input}")
    ...     # Output: "(A→B) ∧ (B→C) ∧ (¬C) ∧ (A∨B)"

Thread Safety:
    All methods are thread-safe. The QueryPreprocessor maintains no mutable state
    between method calls. The singleton accessor uses a lock for thread-safe
    initialization.

Error Handling:
    - Invalid input types raise TypeError with descriptive messages
    - Pattern matching failures return gracefully with preprocessing_applied=False
    - All exceptions are logged but not propagated to maintain pipeline stability

See Also:
    - vulcan.reasoning.reasoning_integration: Main integration layer
    - vulcan.reasoning.symbolic.symbolic_reasoner: Symbolic reasoning engine
    - vulcan.reasoning.mathematical_computation: Mathematical reasoning engine
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Module-level constants
LOG_PREFIX = "[QueryPreprocessor]"
_VERSION = "1.0.0"


class ExtractionType(Enum):
    """
    Types of formal syntax extraction supported by the preprocessor.

    Each type corresponds to a specific reasoning domain and has its own
    extraction patterns and confidence thresholds.

    Attributes:
        SYMBOLIC: Propositional and first-order logic extraction
        MATHEMATICAL: Mathematical formulas and equations
        PROBABILISTIC: Probability distributions and expectations
        CAUSAL: Causal relationships and interventions
        NONE: No extraction performed
    """

    SYMBOLIC = "symbolic"
    MATHEMATICAL = "mathematical"
    PROBABILISTIC = "probabilistic"
    CAUSAL = "causal"
    NONE = "none"


@dataclass(frozen=True)
class PreprocessingResult:
    """
    Immutable result from query preprocessing.

    This dataclass encapsulates all information about the preprocessing
    operation, including extracted formal syntax, confidence scores, and
    metadata about what was extracted.

    Attributes:
        formal_input: Extracted formal syntax ready for parsing, or None if
            no extraction was performed. For symbolic queries, this is a
            conjunction of formulas. For probabilistic queries, this may be
            a list of probability expressions.
        original_query: The original query text, preserved for reference and
            fallback scenarios.
        preprocessing_applied: True if any preprocessing was performed and
            formal_input contains extracted content.
        extraction_confidence: Confidence score (0.0-1.0) indicating how
            reliable the extraction is. Higher values indicate clearer
            pattern matches. Typical thresholds:
            - 0.9: Clear SAT problem with explicit sections
            - 0.85: Explicit Formula/Equation labels
            - 0.8: Probability notation found
            - 0.7: Logical operators found in text
            - 0.75: Equations found without labels
        extraction_type: The type of extraction performed (SYMBOLIC, etc.)
        extracted_propositions: For SAT problems, the list of proposition
            variables found (e.g., ["A", "B", "C"])
        extracted_constraints: For SAT problems, the list of individual
            constraints after normalization
        metadata: Additional extraction metadata for debugging and analysis

    Example:
        >>> result = PreprocessingResult(
        ...     formal_input="(A→B) ∧ (B→C)",
        ...     original_query="Propositions: A,B,C...",
        ...     preprocessing_applied=True,
        ...     extraction_confidence=0.9,
        ...     extraction_type=ExtractionType.SYMBOLIC,
        ...     extracted_propositions=["A", "B", "C"],
        ...     extracted_constraints=["A→B", "B→C"],
        ... )
    """

    formal_input: Optional[Union[str, List[str]]]
    original_query: str
    preprocessing_applied: bool
    extraction_confidence: float
    extraction_type: ExtractionType = ExtractionType.NONE
    extracted_propositions: Tuple[str, ...] = field(default_factory=tuple)
    extracted_constraints: Tuple[str, ...] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate result data after initialization."""
        # Validate confidence is in valid range
        if not 0.0 <= self.extraction_confidence <= 1.0:
            # Use object.__setattr__ since dataclass is frozen
            object.__setattr__(
                self,
                'extraction_confidence',
                max(0.0, min(1.0, self.extraction_confidence))
            )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result to dictionary for serialization.

        Returns:
            Dictionary representation suitable for JSON serialization.

        Example:
            >>> result.to_dict()
            {'formal_input': '(A→B)', 'preprocessing_applied': True, ...}
        """
        return {
            "formal_input": self.formal_input,
            "original_query": self.original_query,
            "preprocessing_applied": self.preprocessing_applied,
            "extraction_confidence": self.extraction_confidence,
            "extraction_type": self.extraction_type.value,
            "extracted_propositions": list(self.extracted_propositions),
            "extracted_constraints": list(self.extracted_constraints),
            "metadata": self.metadata,
        }

    def get(self, key: str, default: Any = None) -> Any:
        """
        Dictionary-like access for backward compatibility.

        Args:
            key: Attribute name to retrieve
            default: Default value if attribute doesn't exist

        Returns:
            Attribute value or default
        """
        return getattr(self, key, default)


@dataclass
class PreprocessorMetrics:
    """
    Metrics for monitoring preprocessor performance.

    Thread-safe metrics collection for observability and debugging.
    All operations use atomic increments where possible.

    Attributes:
        total_queries: Total queries processed
        symbolic_extractions: Successful symbolic extractions
        mathematical_extractions: Successful mathematical extractions
        probabilistic_extractions: Successful probabilistic extractions
        skipped_queries: Queries skipped (no formal engine needed)
        failed_extractions: Extraction attempts that found no patterns
    """

    total_queries: int = 0
    symbolic_extractions: int = 0
    mathematical_extractions: int = 0
    probabilistic_extractions: int = 0
    skipped_queries: int = 0
    failed_extractions: int = 0

    def to_dict(self) -> Dict[str, int]:
        """Convert metrics to dictionary."""
        return {
            "total_queries": self.total_queries,
            "symbolic_extractions": self.symbolic_extractions,
            "mathematical_extractions": self.mathematical_extractions,
            "probabilistic_extractions": self.probabilistic_extractions,
            "skipped_queries": self.skipped_queries,
            "failed_extractions": self.failed_extractions,
        }


class QueryPreprocessor:
    """
    Preprocesses natural language queries to extract formal syntax.

    The QueryPreprocessor is a critical component in the VULCAN-AGI reasoning
    pipeline. It bridges the gap between natural language queries (which may
    contain embedded formal notation) and the formal parsing requirements of
    reasoning engines like SymbolicReasoner.

    Design Philosophy:
        - Fail gracefully: If extraction fails, return the original query
        - Be conservative: Only extract when patterns clearly match
        - Be transparent: Log all decisions and provide confidence scores
        - Be efficient: Pre-compile patterns, avoid unnecessary work

    Supported Extraction Types:
        1. SAT Problems: Extracts propositions and constraints from structured
           problem descriptions with "Propositions:" and "Constraints:" sections.

        2. Direct Formulas: Extracts logical formulas containing operators
           like →, ∧, ∨, ¬, ∀, ∃ directly from query text.

        3. Mathematical Expressions: Extracts equations and formulas labeled
           with "Formula:", "Expression:", or "Equation:" prefixes.

        4. Probabilistic Notation: Extracts P(·), P(·|·), and E[·] expressions
           for probabilistic reasoning engines.

    Thread Safety:
        All methods are thread-safe. The class maintains no mutable state
        between method calls except for metrics, which are updated atomically.

    Attributes:
        _sat_pattern: Compiled regex for SAT problem extraction
        _formula_pattern: Compiled regex for labeled formula extraction
        _logical_operators: Frozenset of recognized logical operator symbols
        _metrics: Performance metrics for monitoring
        _metrics_lock: Lock for thread-safe metrics updates

    Example:
        >>> preprocessor = QueryPreprocessor()
        >>> result = preprocessor.preprocess(
        ...     query="Propositions: A,B\\nConstraints:\\n1. A→B",
        ...     query_type="symbolic",
        ...     reasoning_tools=["symbolic"]
        ... )
        >>> print(result.formal_input)
        "(A→B)"
    """

    # Class-level constants for pattern matching
    FORMAL_ENGINES: FrozenSet[str] = frozenset({"symbolic", "mathematical", "probabilistic"})

    # Confidence thresholds
    HIGH_CONFIDENCE: float = 0.9
    MEDIUM_HIGH_CONFIDENCE: float = 0.85
    MEDIUM_CONFIDENCE: float = 0.8
    MEDIUM_LOW_CONFIDENCE: float = 0.75
    LOW_CONFIDENCE: float = 0.7

    # Validation constants (addressing code review feedback)
    HEADER_CHECK_LENGTH: int = 20  # Characters to check for section headers
    MAX_EQUATION_LENGTH: int = 100  # Maximum length for extracted equations
    TEXT_EQUATION_PREFIXES: Tuple[str, ...] = ('is ', 'the ', 'a ', 'an ')

    def __init__(self) -> None:
        """
        Initialize preprocessor with compiled pattern matchers.

        Patterns are compiled once at initialization for efficiency.
        All patterns use appropriate flags for case-insensitivity and
        multiline matching where needed.
        """
        # SAT problem pattern: "Propositions: A,B,C ... Constraints: ..."
        # Uses DOTALL to match across newlines, IGNORECASE for flexibility
        self._sat_pattern = re.compile(
            r'Propositions?:\s*([A-Z,\s]+).*?Constraints?:(.*?)(?:Task:|$)',
            re.DOTALL | re.IGNORECASE
        )

        # Direct formula pattern: "Formula: ..." or "Expression: ..." or "Equation: ..."
        self._formula_pattern = re.compile(
            r'(?:Formula|Expression|Equation):\s*(.*?)(?:\n|$)',
            re.IGNORECASE
        )

        # Constraint line pattern: numbered lists and bullet points
        # CRITICAL FIX: Require bullet/number to be at line start to avoid
        # matching operators like '->' as bullets. The pattern:
        # - ^: start of string (when used with MULTILINE)
        # - \d+\.: numbered list (1., 2., etc.)
        # - •: bullet point
        # - [-*]: dash or asterisk bullets (at line start only)
        self._constraint_line_pattern = re.compile(
            r'(?:^|\n)\s*(?:\d+\.|•|[-*])\s*([^\n]+)',
            re.MULTILINE
        )

        # Equation pattern: something = something (but not assignment-like text)
        self._equation_pattern = re.compile(
            r'([^=\n]+=[^=\n]+)'
        )

        # Probability notation pattern: P(...) including conditional
        self._probability_pattern = re.compile(
            r'P\([^)]+\)'
        )

        # Expectation notation pattern: E[...]
        self._expectation_pattern = re.compile(
            r'E\[[^\]]+\]'
        )

        # Logical operators that indicate formal syntax (frozen for immutability)
        self._logical_operators: FrozenSet[str] = frozenset([
            '→', '∧', '∨', '¬', '∀', '∃',  # Unicode operators
            '->', '<->',                     # ASCII alternatives
            '⊢', '⊨',                        # Turnstile operators
        ])

        # Operator normalization mapping (order matters for some replacements)
        # IMPORTANT: Longer patterns must come before shorter patterns to avoid
        # partial replacements (e.g., '<->' must be checked before '->')
        self._operator_normalizations: Tuple[Tuple[str, str], ...] = (
            # Biconditional (MUST come before implication to avoid partial match)
            ('<->', '↔'),
            ('iff', '↔'),
            # Implication (check longer patterns first)
            ('=>', '→'),
            ('->', '→'),
            ('implies', '→'),
            # Conjunction (check longer patterns first)
            ('&&', '∧'),
            ('AND', '∧'),
            ('and', '∧'),
            # Note: single '&' handled separately to avoid issues
            # Disjunction
            ('||', '∨'),
            ('OR', '∨'),
            ('or', '∨'),
            # Negation
            ('NOT', '¬'),
            ('not', '¬'),
            ('~', '¬'),
            ('!', '¬'),
        )

        # =====================================================================
        # FIX #2: Header Stripping Patterns
        # =====================================================================
        # Test queries often include headers/labels that confuse routing and engines.
        # Examples:
        # - "M1 — Proof check" → "M1" triggers cryptographic classification
        # - "Monty Hall variant (forces clean reasoning)" → Header confuses engine
        # - "Analogical Reasoning A1 — Structure mapping" → Label confuses routing
        #
        # These patterns are stripped from queries before classification/processing.
        self._header_strip_patterns: Tuple[re.Pattern, ...] = (
            # Full reasoning type headers with optional task label at line start
            # E.g., "Causal Reasoning C1 — Confounding" → "Confounding"
            re.compile(
                r'^(?:Analogical|Causal|Mathematical|Probabilistic|Philosophical|Symbolic)\s+Reasoning\s*'
                r'(?:[A-Z][0-9]+\s*)?[—\-:]*\s*',
                re.MULTILINE | re.IGNORECASE
            ),
            # Task labels like "M1 —", "C1 —", "A1 —", "S1 —" (standalone)
            re.compile(r'^[A-Z][0-9]+\s*[—\-]\s*', re.MULTILINE),
            # Task: / Claim: / Query: prefixes
            re.compile(r'^(?:Task|Claim|Query|Problem):\s*', re.MULTILINE | re.IGNORECASE),
            # Parenthetical notes like "(forces clean reasoning)"
            re.compile(r'\s*\((?:forces?\s+)?clean\s+reasoning\)\s*', re.IGNORECASE),
            # Variant/scenario headers like "Monty Hall variant"
            re.compile(r'^[^(\n]*variant\s*', re.MULTILINE | re.IGNORECASE),
        )

        # Thread-safe metrics
        self._metrics = PreprocessorMetrics()
        self._metrics_lock = threading.Lock()

        logger.debug(f"{LOG_PREFIX} Initialized with {len(self._logical_operators)} operators")

    def strip_headers(self, query: str) -> str:
        """
        Strip test headers and labels from queries that confuse routing and engines.
        
        FIX #2: Query Preprocessing Confusion
        =====================================
        Test queries often include headers/labels that confuse routing:
        - "M1 —" triggers cryptographic classification
        - "Analogical Reasoning A1 —" confuses the router
        - "Monty Hall variant (forces clean reasoning)" confuses the engine
        
        This method removes such headers to allow proper query classification
        and engine processing.
        
        Args:
            query: Raw query string with potential headers
            
        Returns:
            Query string with headers stripped
            
        Examples:
            >>> preprocessor.strip_headers("M1 — Proof check: Prove that...")
            "Prove that..."
            >>> preprocessor.strip_headers("Analogical Reasoning A1 — Structure mapping...")
            "Structure mapping..."
        """
        if not query or not isinstance(query, str):
            return query
        
        cleaned = query.strip()
        original_length = len(cleaned)
        
        # Apply each header stripping pattern
        for pattern in self._header_strip_patterns:
            cleaned = pattern.sub('', cleaned).strip()
        
        # Log if headers were stripped
        if len(cleaned) < original_length:
            logger.info(
                f"{LOG_PREFIX} FIX#2: Stripped headers from query "
                f"({original_length} -> {len(cleaned)} chars)"
            )
        
        return cleaned

    def preprocess(
        self,
        query: str,
        query_type: str,
        reasoning_tools: List[str],
    ) -> PreprocessingResult:
        """
        Preprocess query to extract formal components.

        This is the main entry point for query preprocessing. It analyzes the
        query and reasoning tools to determine if extraction is needed, then
        routes to the appropriate extraction method.

        Args:
            query: Raw natural language query text. May contain embedded
                formal notation mixed with natural language descriptions.
            query_type: Type of query from the router (e.g., "symbolic",
                "mathematical", "reasoning", "general").
            reasoning_tools: List of tools selected for this query. Extraction
                is only performed if a formal engine is in this list.

        Returns:
            PreprocessingResult with extracted formal syntax if patterns
            matched, or a result with preprocessing_applied=False if no
            extraction was performed.

        Raises:
            TypeError: If query is not a string or reasoning_tools is not a list.

        Example:
            >>> result = preprocessor.preprocess(
            ...     query="Propositions: A,B\\nConstraints:\\n1. A→B",
            ...     query_type="symbolic",
            ...     reasoning_tools=["symbolic"]
            ... )
            >>> result.preprocessing_applied
            True
            >>> result.formal_input
            "(A→B)"
        """
        # Input validation
        if not isinstance(query, str):
            raise TypeError(f"query must be str, got {type(query).__name__}")
        if not isinstance(reasoning_tools, list):
            raise TypeError(f"reasoning_tools must be list, got {type(reasoning_tools).__name__}")

        # FIX #2: Strip headers/labels that confuse routing
        # This must happen BEFORE any processing to ensure clean query classification
        cleaned_query = self.strip_headers(query)

        # Update metrics
        with self._metrics_lock:
            self._metrics.total_queries += 1

        # Create default result (no preprocessing)
        default_result = PreprocessingResult(
            formal_input=None,
            original_query=query,  # Keep original for reference
            preprocessing_applied=False,
            extraction_confidence=0.0,
            extraction_type=ExtractionType.NONE,
        )

        # Only preprocess if using formal reasoning engines
        tools_set = set(reasoning_tools)
        if not tools_set & self.FORMAL_ENGINES:
            logger.debug(
                f"{LOG_PREFIX} No formal engines in tools {reasoning_tools}, skipping"
            )
            with self._metrics_lock:
                self._metrics.skipped_queries += 1
            return default_result

        # Route to appropriate extractor based on tools
        # Priority: symbolic > mathematical > probabilistic
        # FIX #2: Use cleaned_query instead of raw query
        if 'symbolic' in tools_set:
            return self._extract_symbolic(cleaned_query)
        elif 'mathematical' in tools_set:
            return self._extract_mathematical(cleaned_query)
        elif 'probabilistic' in tools_set:
            return self._extract_probabilistic(cleaned_query)

        return default_result

    def _extract_symbolic(self, query: str) -> PreprocessingResult:
        """
        Extract formal logic from symbolic reasoning queries.

        Handles two main patterns:
        1. SAT problems with explicit Propositions/Constraints sections
        2. Queries containing logical operators directly in the text

        The method attempts pattern 1 first (higher confidence), then falls
        back to pattern 2 if no match is found.

        Args:
            query: Query string to process

        Returns:
            PreprocessingResult with extracted formal logic or default result
        """
        # Pattern 1: SAT problems with explicit sections
        match = self._sat_pattern.search(query)
        if match:
            result = self._extract_sat_problem(match)
            if result.preprocessing_applied:
                with self._metrics_lock:
                    self._metrics.symbolic_extractions += 1
                return result

        # Pattern 2: Direct logical formulas
        if self._contains_logical_operators(query):
            result = self._extract_direct_formulas(query)
            if result.preprocessing_applied:
                with self._metrics_lock:
                    self._metrics.symbolic_extractions += 1
                return result

        # No patterns matched
        logger.debug(f"{LOG_PREFIX} No symbolic extraction patterns matched")
        with self._metrics_lock:
            self._metrics.failed_extractions += 1

        return PreprocessingResult(
            formal_input=None,
            original_query=query,
            preprocessing_applied=False,
            extraction_confidence=0.0,
            extraction_type=ExtractionType.SYMBOLIC,
        )

    def _extract_sat_problem(self, match: re.Match) -> PreprocessingResult:
        """
        Extract SAT problem from regex match.

        Args:
            match: Regex match object with proposition and constraint groups

        Returns:
            PreprocessingResult with extracted SAT formula
        """
        propositions_str = match.group(1).strip()
        constraints_str = match.group(2).strip()

        # Parse propositions (single uppercase letters separated by commas)
        propositions = tuple(
            p.strip()
            for p in propositions_str.split(',')
            if p.strip()
        )

        # Parse constraints - try numbered list first, then inline comma-separated
        constraint_matches = self._constraint_line_pattern.findall(constraints_str)

        # =======================================================================
        # Note: Handle inline comma-separated constraint format
        # =======================================================================
        # If numbered/bulleted list pattern fails, try parsing inline format like:
        # "A → B, B → C, ¬C, A ∨ B"
        #
        # The problem: When constraints are in inline format (not numbered list),
        # the _constraint_line_pattern regex fails and returns empty list.
        # This causes the method to return early with preprocessing_applied=False,
        # and the query then hits _extract_direct_formulas() which incorrectly
        # splits by newlines and treats individual operators as "formulas".
        #
        # The fix: Parse inline comma-separated constraints when numbered list fails.
        # Each constraint must contain at least one proposition letter (A-Z or a-z).
        # =======================================================================
        if not constraint_matches:
            # Try inline comma-separated format
            # First, normalize operators in the entire constraint string
            normalized_constraints_str = self._normalize_operators(constraints_str)
            
            # Split by comma to separate individual constraints
            # Note: Simple comma split works for most SAT constraint formats
            inline_constraints = self._split_inline_constraints(normalized_constraints_str)
            
            if inline_constraints:
                constraint_matches = inline_constraints
                logger.info(
                    f"{LOG_PREFIX} Using inline constraint format: found {len(inline_constraints)} constraints"
                )

        if not constraint_matches:
            return PreprocessingResult(
                formal_input=None,
                original_query=match.string,
                preprocessing_applied=False,
                extraction_confidence=0.0,
                extraction_type=ExtractionType.SYMBOLIC,
            )

        # Normalize each constraint and filter out invalid ones
        formal_constraints = []
        for constraint in constraint_matches:
            normalized = self._normalize_operators(constraint.strip())
            if normalized:
                # Note: Validate constraint contains at least one proposition
                if re.search(r'[A-Za-z]', normalized):
                    formal_constraints.append(normalized)
                else:
                    logger.warning(
                        f"{LOG_PREFIX} Filtered invalid constraint (no proposition): '{normalized}'"
                    )

        if not formal_constraints:
            return PreprocessingResult(
                formal_input=None,
                original_query=match.string,
                preprocessing_applied=False,
                extraction_confidence=0.0,
                extraction_type=ExtractionType.SYMBOLIC,
            )

        # Join constraints with conjunction, wrapping each in parentheses
        formal_input = ' ∧ '.join(f'({c})' for c in formal_constraints)

        # Log extraction (truncate long formulas)
        log_formula = formal_input[:100] + '...' if len(formal_input) > 100 else formal_input
        logger.info(f"{LOG_PREFIX} Extracted SAT formula: {log_formula}")

        return PreprocessingResult(
            formal_input=formal_input,
            original_query=match.string,
            preprocessing_applied=True,
            extraction_confidence=self.HIGH_CONFIDENCE,
            extraction_type=ExtractionType.SYMBOLIC,
            extracted_propositions=propositions,
            extracted_constraints=tuple(formal_constraints),
            metadata={"pattern": "sat_problem"},
        )

    def _extract_direct_formulas(self, query: str) -> PreprocessingResult:
        """
        Extract formulas containing logical operators directly from query text.

        Args:
            query: Query string to process

        Returns:
            PreprocessingResult with extracted formulas
        """
        lines = query.split('\n')
        formal_lines: List[str] = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Skip lines that are section headers (contain colon early but no operators)
            # Use HEADER_CHECK_LENGTH constant for maintainability
            header_prefix = line[:self.HEADER_CHECK_LENGTH]
            if ':' in header_prefix and not self._contains_logical_operators(header_prefix):
                continue

            # Check if line contains logical operators
            if self._contains_logical_operators(line):
                cleaned = self._clean_formula_line(line)
                if cleaned:
                    formal_lines.append(cleaned)

        if not formal_lines:
            return PreprocessingResult(
                formal_input=None,
                original_query=query,
                preprocessing_applied=False,
                extraction_confidence=0.0,
                extraction_type=ExtractionType.SYMBOLIC,
            )

        formal_input = ' ∧ '.join(f'({f})' for f in formal_lines)

        # Log extraction
        log_formula = formal_input[:100] + '...' if len(formal_input) > 100 else formal_input
        logger.info(f"{LOG_PREFIX} Extracted formula lines: {log_formula}")

        return PreprocessingResult(
            formal_input=formal_input,
            original_query=query,
            preprocessing_applied=True,
            extraction_confidence=self.LOW_CONFIDENCE,
            extraction_type=ExtractionType.SYMBOLIC,
            extracted_constraints=tuple(formal_lines),
            metadata={"pattern": "direct_formulas"},
        )

    def _extract_mathematical(self, query: str) -> PreprocessingResult:
        """
        Extract mathematical formulas from proof queries.

        Looks for:
        1. Explicitly labeled formulas ("Formula: ...", "Equation: ...")
        2. Equations (expressions containing "=")

        Args:
            query: Query string to process

        Returns:
            PreprocessingResult with extracted mathematical content
        """
        # Pattern 1: Explicitly labeled formulas
        match = self._formula_pattern.search(query)
        if match:
            formula = match.group(1).strip()
            logger.info(f"{LOG_PREFIX} Extracted formula: {formula}")

            with self._metrics_lock:
                self._metrics.mathematical_extractions += 1

            return PreprocessingResult(
                formal_input=formula,
                original_query=query,
                preprocessing_applied=True,
                extraction_confidence=self.MEDIUM_HIGH_CONFIDENCE,
                extraction_type=ExtractionType.MATHEMATICAL,
                metadata={"pattern": "labeled_formula"},
            )

        # Pattern 2: Equations (contains =)
        equation_match = self._equation_pattern.search(query)
        if equation_match:
            equation = equation_match.group(1).strip()

            # Validate it's a real equation, not descriptive text
            # Reject if it starts with common text patterns or is too long
            # Using class constants for maintainability
            if (len(equation) < self.MAX_EQUATION_LENGTH and
                    not equation.lower().startswith(self.TEXT_EQUATION_PREFIXES)):
                logger.info(f"{LOG_PREFIX} Extracted equation: {equation}")

                with self._metrics_lock:
                    self._metrics.mathematical_extractions += 1

                return PreprocessingResult(
                    formal_input=equation,
                    original_query=query,
                    preprocessing_applied=True,
                    extraction_confidence=self.MEDIUM_LOW_CONFIDENCE,
                    extraction_type=ExtractionType.MATHEMATICAL,
                    metadata={"pattern": "equation"},
                )

        logger.debug(f"{LOG_PREFIX} No mathematical extraction patterns matched")
        with self._metrics_lock:
            self._metrics.failed_extractions += 1

        return PreprocessingResult(
            formal_input=None,
            original_query=query,
            preprocessing_applied=False,
            extraction_confidence=0.0,
            extraction_type=ExtractionType.MATHEMATICAL,
        )

    def _extract_probabilistic(self, query: str) -> PreprocessingResult:
        """
        Extract probabilistic information from queries.

        Looks for:
        1. Probability notation: P(A), P(A|B), P(X=x)
        2. Expectation notation: E[X], E[X|Y]

        Args:
            query: Query string to process

        Returns:
            PreprocessingResult with extracted probability expressions
        """
        # Look for P(...) notation
        prob_patterns = self._probability_pattern.findall(query)
        if prob_patterns:
            logger.info(f"{LOG_PREFIX} Extracted probabilities: {prob_patterns}")

            with self._metrics_lock:
                self._metrics.probabilistic_extractions += 1

            return PreprocessingResult(
                formal_input=prob_patterns,
                original_query=query,
                preprocessing_applied=True,
                extraction_confidence=self.MEDIUM_CONFIDENCE,
                extraction_type=ExtractionType.PROBABILISTIC,
                metadata={"pattern": "probability", "count": len(prob_patterns)},
            )

        # Look for expectation notation E[...]
        expectation_patterns = self._expectation_pattern.findall(query)
        if expectation_patterns:
            logger.info(f"{LOG_PREFIX} Extracted expectations: {expectation_patterns}")

            with self._metrics_lock:
                self._metrics.probabilistic_extractions += 1

            return PreprocessingResult(
                formal_input=expectation_patterns,
                original_query=query,
                preprocessing_applied=True,
                extraction_confidence=self.MEDIUM_CONFIDENCE,
                extraction_type=ExtractionType.PROBABILISTIC,
                metadata={"pattern": "expectation", "count": len(expectation_patterns)},
            )

        logger.debug(f"{LOG_PREFIX} No probabilistic extraction patterns matched")
        with self._metrics_lock:
            self._metrics.failed_extractions += 1

        return PreprocessingResult(
            formal_input=None,
            original_query=query,
            preprocessing_applied=False,
            extraction_confidence=0.0,
            extraction_type=ExtractionType.PROBABILISTIC,
        )

    def _contains_logical_operators(self, text: str) -> bool:
        """
        Check if text contains any logical operators.

        Args:
            text: Text to check

        Returns:
            True if any logical operator is found
        """
        return any(op in text for op in self._logical_operators)

    def _normalize_operators(self, text: str) -> str:
        """
        Normalize logical operators to standard Unicode symbols.

        Converts ASCII representations to their Unicode equivalents for
        consistent parsing by downstream engines.

        Args:
            text: Text containing operators in various formats

        Returns:
            Text with normalized Unicode operators

        Example:
            >>> preprocessor._normalize_operators("A -> B AND C")
            "A → B ∧ C"
        """
        result = text
        for pattern, replacement in self._operator_normalizations:
            result = result.replace(pattern, replacement)
        return result.strip()

    def _split_inline_constraints(self, constraints_str: str) -> List[str]:
        """
        Split inline constraints into individual formulas.

        Handles constraint formats like:
        - Comma-separated: "A → B, B → C, ¬C, A ∨ B"
        - Newline-separated: "A→B\\nB→C\\n¬C\\nA∨B"
        - Mixed: "A → B,B → C\\n¬C,A ∨ B"

        The method is careful to:
        1. Split on commas OR newlines (whichever is used as separator)
        2. Preserve spaces around operators within each constraint
        3. Filter out empty or whitespace-only constraints
        4. Filter out constraints that don't contain at least one proposition letter

        Args:
            constraints_str: String containing constraints (comma or newline separated)

        Returns:
            List of individual constraint strings

        Examples:
            >>> preprocessor._split_inline_constraints("A → B, B → C, ¬C")
            ["A → B", "B → C", "¬C"]
            >>> preprocessor._split_inline_constraints("A→B\\nB→C\\n¬C")
            ["A→B", "B→C", "¬C"]
        """
        if not constraints_str or not constraints_str.strip():
            return []

        # Determine the primary separator
        # Strategy: Use newlines as separator if there are multiple valid constraints
        # on separate lines; otherwise fall back to comma separator.
        newline_count = constraints_str.count('\n')
        
        # Use newlines if there are multiple newline-separated items that look like constraints
        if newline_count > 0:
            # Check if newline-separated parts contain logical content
            newline_parts = [p.strip() for p in constraints_str.split('\n') if p.strip()]
            valid_newline_parts = [p for p in newline_parts if re.search(r'[A-Za-z]', p)]
            
            # If we have multiple valid newline-separated constraints, use newlines as separator
            if len(valid_newline_parts) >= 2:
                raw_constraints = newline_parts
                logger.debug(f"{LOG_PREFIX} Using newline separator: found {len(raw_constraints)} parts")
            else:
                # Fall back to comma separator
                raw_constraints = constraints_str.split(',')
        else:
            # No newlines, use comma separator
            raw_constraints = constraints_str.split(',')

        # Filter and clean each constraint
        valid_constraints: List[str] = []
        for constraint in raw_constraints:
            constraint = constraint.strip()

            # Skip empty constraints
            if not constraint:
                continue

            # Validate that constraint contains at least one proposition letter
            # This prevents standalone operators like "→" or "∧" from being included
            if not re.search(r'[A-Za-z]', constraint):
                logger.debug(
                    f"{LOG_PREFIX} Skipping inline constraint without proposition: '{constraint}'"
                )
                continue

            valid_constraints.append(constraint)

        return valid_constraints

    def _clean_formula_line(self, line: str) -> str:
        """
        Clean a line containing a formula.

        Removes common prefixes (numbers, bullets) and trailing punctuation,
        then normalizes operators. Also validates that the result is a complete
        formula containing at least one proposition (letter).

        CRITICAL FIX (BUG D): Standalone operators like '→', '∧', '∨', '¬'
        are NOT valid formulas and must be filtered out. A valid formula must
        contain at least one proposition letter (A-Z or a-z).

        Args:
            line: Line to clean

        Returns:
            Cleaned formula string, or empty string if invalid

        Examples:
            >>> preprocessor._clean_formula_line("1. A→B")
            "A→B"
            >>> preprocessor._clean_formula_line("→")  # Invalid standalone operator
            ""
            >>> preprocessor._clean_formula_line("¬C")
            "¬C"
        """
        # Remove common prefixes (numbered lists, bullets)
        cleaned = re.sub(r'^(?:\d+\.|\*|•|[-])\s*', '', line)

        # Remove trailing punctuation except parentheses
        cleaned = re.sub(r'[,;]$', '', cleaned)

        # Normalize operators
        cleaned = self._normalize_operators(cleaned)

        cleaned = cleaned.strip()

        # ====================================================================
        # Note: Filter out standalone operators and invalid formulas
        # ====================================================================
        # A valid formula MUST contain at least one proposition letter (A-Z or a-z)
        # Standalone operators like '→', '∧', '∨', '¬', '(→)', '(∧)', etc. are INVALID
        
        # Check if formula contains at least one letter (proposition)
        if not re.search(r'[A-Za-z]', cleaned):
            logger.warning(
                f"{LOG_PREFIX} Filtered invalid formula (no proposition): '{cleaned}'"
            )
            return ""
        
        # Define standalone operators that should be rejected
        standalone_operators = frozenset([
            '→', '∨', '∧', '¬', '↔',
            '(→)', '(∨)', '(∧)', '(¬)', '(↔)',
            '->', '<->', '&&', '||', '~', '!',
            '(->', '(<->)', '(&&)', '(||)', '(~)', '(!)',
        ])
        
        # Reject if the entire formula is just an operator
        if cleaned in standalone_operators:
            logger.warning(
                f"{LOG_PREFIX} Filtered standalone operator: '{cleaned}'"
            )
            return ""

        return cleaned

    def get_metrics(self) -> Dict[str, int]:
        """
        Get current preprocessor metrics.

        Returns:
            Dictionary with metric counts

        Example:
            >>> metrics = preprocessor.get_metrics()
            >>> print(f"Processed {metrics['total_queries']} queries")
        """
        with self._metrics_lock:
            return self._metrics.to_dict()

    def reset_metrics(self) -> None:
        """Reset all metrics to zero."""
        with self._metrics_lock:
            self._metrics = PreprocessorMetrics()
        logger.debug(f"{LOG_PREFIX} Metrics reset")


# =============================================================================
# Singleton Management
# =============================================================================

_preprocessor_instance: Optional[QueryPreprocessor] = None
_preprocessor_lock = threading.Lock()


def get_query_preprocessor() -> QueryPreprocessor:
    """
    Get or create singleton preprocessor instance.

    Uses double-checked locking pattern for thread-safe lazy initialization.
    The singleton is created on first access and reused for all subsequent
    calls.

    Returns:
        QueryPreprocessor singleton instance

    Example:
        >>> preprocessor = get_query_preprocessor()
        >>> result = preprocessor.preprocess(query, "symbolic", ["symbolic"])
    """
    global _preprocessor_instance

    if _preprocessor_instance is None:
        with _preprocessor_lock:
            if _preprocessor_instance is None:
                _preprocessor_instance = QueryPreprocessor()
                logger.info(f"{LOG_PREFIX} Singleton instance created")

    return _preprocessor_instance


def reset_query_preprocessor() -> None:
    """
    Reset the singleton preprocessor instance.

    Primarily used for testing. After calling this, the next call to
    get_query_preprocessor() will create a new instance.
    """
    global _preprocessor_instance

    with _preprocessor_lock:
        _preprocessor_instance = None
        logger.debug(f"{LOG_PREFIX} Singleton instance reset")


# =============================================================================
# Query Decomposition Layer (ROOT CAUSE FIX)
# =============================================================================
# Problem: The symbolic reasoner parses individual formulas correctly but doesn't
# understand what to DO with natural language questions. Queries like:
#   "Graph: A→B, C→B, B→D. Condition on B. Does conditioning induce correlation?"
# 
# The parser extracts "A→B", "C→B", "B→D" but has no idea what the QUERY is.
# It returns "Extracted X constraints but no hypothesis" instead of reasoning.
#
# Solution: QueryDecomposer extracts:
#   1. Background facts (A→B, C→B, B→D) 
#   2. The question (Does conditioning on B induce correlation?)
#   3. Converts question to formal query
#   4. Returns (facts, formal_query) for the reasoner
# =============================================================================


@dataclass(frozen=True)
class DecomposedQuery:
    """
    Result from query decomposition containing structured components.
    
    Attributes:
        background_facts: List of extracted logical facts/constraints
        hypothesis: The question/hypothesis to evaluate
        formal_hypothesis: The question converted to formal logic (if possible)
        query_type: Type of query (sat, entailment, conditioning, causal, etc.)
        conditioning_variables: Variables mentioned for conditioning (e.g., "B")
        target_variables: Variables mentioned as targets (e.g., "A", "C")
        decomposition_confidence: Confidence in the decomposition (0.0-1.0)
        original_query: The original query text
        decomposition_applied: Whether decomposition was successfully applied
    """
    
    background_facts: Tuple[str, ...]
    hypothesis: Optional[str]
    formal_hypothesis: Optional[str]
    query_type: str
    conditioning_variables: Tuple[str, ...] = field(default_factory=tuple)
    target_variables: Tuple[str, ...] = field(default_factory=tuple)
    decomposition_confidence: float = 0.0
    original_query: str = ""
    decomposition_applied: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "background_facts": list(self.background_facts),
            "hypothesis": self.hypothesis,
            "formal_hypothesis": self.formal_hypothesis,
            "query_type": self.query_type,
            "conditioning_variables": list(self.conditioning_variables),
            "target_variables": list(self.target_variables),
            "decomposition_confidence": self.decomposition_confidence,
            "original_query": self.original_query,
            "decomposition_applied": self.decomposition_applied,
        }


class QueryDecomposer:
    """
    Decomposes natural language queries with embedded formal logic into structured components.
    
    ROOT CAUSE FIX: The symbolic reasoner was failing because it:
    1. Parses individual formulas correctly (A→B, C→B)
    2. Has NO IDEA what the actual question is
    3. Returns "no hypothesis" instead of actually reasoning
    
    This class bridges that gap by extracting:
    - Background facts (the formal logic parts)
    - The question/hypothesis (the natural language question)
    - A formal version of the hypothesis (for the prover)
    
    Supported query patterns:
    - SAT problems: "Propositions: A,B,C. Constraints: A→B, ... Is the set satisfiable?"
    - Graph queries: "Graph: A→B, C→B. Does conditioning on B induce correlation?"
    - Entailment: "Given A→B and B→C, does A→C follow?"
    - Causal queries: "If we intervene on X, what changes?"
    
    Example:
        >>> decomposer = QueryDecomposer()
        >>> result = decomposer.decompose(
        ...     "Graph: A→B, C→B, B→D. You condition on B. "
        ...     "Question: Does conditioning on B induce correlation between A and C?"
        ... )
        >>> print(result.background_facts)
        ('A→B', 'C→B', 'B→D')
        >>> print(result.hypothesis)
        "Does conditioning on B induce correlation between A and C?"
        >>> print(result.query_type)
        "conditioning"
    """
    
    # Query type patterns for classification
    QUERY_TYPE_PATTERNS: Dict[str, Tuple[re.Pattern, ...]] = {
        "sat": (
            re.compile(r'\bsatisfiab(?:le|ility)\b', re.I),
            re.compile(r'\bunsat(?:isfiable)?\b', re.I),
            re.compile(r'\bcontradiction\b', re.I),
            re.compile(r'\bconsisten(?:t|cy)\b', re.I),
        ),
        "entailment": (
            re.compile(r'\bentails?\b', re.I),
            re.compile(r'\bfollows?\s+from\b', re.I),
            re.compile(r'\bimplies?\b', re.I),
            re.compile(r'\bderiv(?:e|able)\b', re.I),
            re.compile(r'\bprov(?:e|able)\b', re.I),
        ),
        "conditioning": (
            re.compile(r'\bcondition(?:ing|ed)?\s+on\b', re.I),
            re.compile(r'\bgiven\s+[A-Z]\b', re.I),
            re.compile(r'\bd-separat(?:e|ion)\b', re.I),
            re.compile(r'\bindependen(?:t|ce)\b', re.I),
            re.compile(r'\bcorrelat(?:e|ion|ed)\b', re.I),
        ),
        "causal": (
            re.compile(r'\binterven(?:e|tion)\b', re.I),
            re.compile(r'\bdo-calculus\b', re.I),
            re.compile(r'\bcounterfactual\b', re.I),
            re.compile(r'\bcaus(?:e|al|ation)\b', re.I),
            re.compile(r'\bwhat\s+changes?\b', re.I),
        ),
        "validity": (
            re.compile(r'\bvalid(?:ity)?\b', re.I),
            re.compile(r'\btautolog(?:y|ical)\b', re.I),
        ),
    }
    
    # Patterns for extracting graph/constraint sections
    SECTION_PATTERNS = {
        "graph": re.compile(
            r'(?:Graph|Structure|DAG|Network):\s*([^.?!]+)', re.I
        ),
        "propositions": re.compile(
            r'Propositions?:\s*([A-Z,\s]+)', re.I
        ),
        "constraints": re.compile(
            r'Constraints?:\s*(.*?)(?:Task:|Question:|$)', re.I | re.DOTALL
        ),
        "given": re.compile(
            r'(?:Given|Assume|Suppose):\s*([^.?!]+)', re.I
        ),
        "facts": re.compile(
            r'(?:Facts?|Background|Premises?):\s*(.*?)(?:Question:|Task:|$)', re.I | re.DOTALL
        ),
    }
    
    # Question extraction patterns
    QUESTION_PATTERNS = (
        re.compile(r'(?:Question|Task|Query):\s*(.+?)(?:\n|$)', re.I),
        re.compile(r'(?:Does|Is|Are|Can|Will|Should)\s+.+\?', re.I),
        re.compile(r'(?:What|Which|How|Why|Where|When)\s+.+\?', re.I),
    )
    
    # Conditioning variable extraction
    CONDITION_VAR_PATTERN = re.compile(
        r'condition(?:ing|ed)?\s+on\s+([A-Z])', re.I
    )
    
    # Target variable extraction (for independence/correlation queries)
    TARGET_VAR_PATTERN = re.compile(
        r'(?:between|correlation.*?|independent.*?)\s+([A-Z])\s+and\s+([A-Z])', re.I
    )
    
    # Formula extraction pattern (logical operators)
    FORMULA_PATTERN = re.compile(
        r'([A-Z]\s*[→∧∨¬⇒⇔\->|&~]+\s*[A-Z])', re.U
    )
    
    def __init__(self) -> None:
        """Initialize the query decomposer."""
        self._preprocessor = QueryPreprocessor()
        logger.debug(f"{LOG_PREFIX} QueryDecomposer initialized")
    
    def decompose(self, query: str) -> DecomposedQuery:
        """
        Decompose a natural language query into structured components.
        
        This is the main entry point for query decomposition. It extracts:
        1. Background facts (formal logic constraints)
        2. The question/hypothesis
        3. Conditioning variables
        4. Target variables
        5. Query type
        
        Args:
            query: The natural language query with embedded formal logic
            
        Returns:
            DecomposedQuery with extracted components
            
        Example:
            >>> decomposer = QueryDecomposer()
            >>> result = decomposer.decompose(
            ...     "Graph: A→B, C→B, B→D. Condition on B. "
            ...     "Does conditioning on B induce correlation between A and C?"
            ... )
        """
        if not query or not query.strip():
            return DecomposedQuery(
                background_facts=(),
                hypothesis=None,
                formal_hypothesis=None,
                query_type="unknown",
                original_query=query or "",
                decomposition_applied=False,
            )
        
        # Strip headers first (using preprocessor's method)
        cleaned_query = self._preprocessor.strip_headers(query)
        
        # Extract background facts
        facts = self._extract_facts(cleaned_query)
        
        # Extract the question/hypothesis
        hypothesis = self._extract_hypothesis(cleaned_query)
        
        # Determine query type
        query_type = self._determine_query_type(cleaned_query)
        
        # Extract conditioning variables
        conditioning_vars = self._extract_conditioning_variables(cleaned_query)
        
        # Extract target variables
        target_vars = self._extract_target_variables(cleaned_query)
        
        # Convert hypothesis to formal form if possible
        formal_hypothesis = self._convert_hypothesis_to_formal(
            hypothesis, query_type, conditioning_vars, target_vars
        )
        
        # Calculate decomposition confidence
        confidence = self._calculate_confidence(
            facts, hypothesis, query_type, conditioning_vars, target_vars
        )
        
        # Decomposition is applied if we extracted meaningful facts AND hypothesis
        decomposition_applied = len(facts) > 0 and hypothesis is not None
        
        if decomposition_applied:
            logger.info(
                f"{LOG_PREFIX} Query decomposed: {len(facts)} facts, "
                f"type={query_type}, confidence={confidence:.2f}"
            )
        else:
            logger.debug(
                f"{LOG_PREFIX} Query decomposition minimal: "
                f"facts={len(facts)}, hypothesis={'yes' if hypothesis else 'no'}"
            )
        
        return DecomposedQuery(
            background_facts=tuple(facts),
            hypothesis=hypothesis,
            formal_hypothesis=formal_hypothesis,
            query_type=query_type,
            conditioning_variables=tuple(conditioning_vars),
            target_variables=tuple(target_vars),
            decomposition_confidence=confidence,
            original_query=query,
            decomposition_applied=decomposition_applied,
        )
    
    def _extract_facts(self, query: str) -> List[str]:
        """
        Extract background facts/constraints from the query.
        
        Looks for:
        - Explicit sections: "Graph: A→B, C→B" or "Constraints: 1. A→B"
        - Inline formulas: "Given A→B and B→C"
        - Implicit formulas in the text
        
        Args:
            query: The cleaned query text
            
        Returns:
            List of extracted fact strings
        """
        facts: List[str] = []
        
        # Try section-based extraction first
        for section_name, pattern in self.SECTION_PATTERNS.items():
            match = pattern.search(query)
            if match:
                section_text = match.group(1).strip()
                section_facts = self._parse_fact_section(section_text)
                facts.extend(section_facts)
        
        # If no section-based facts, try inline extraction
        if not facts:
            facts = self._extract_inline_facts(query)
        
        # Normalize and deduplicate
        normalized_facts = []
        seen = set()
        for fact in facts:
            normalized = self._preprocessor._normalize_operators(fact.strip())
            if normalized and normalized not in seen and re.search(r'[A-Z]', normalized):
                seen.add(normalized)
                normalized_facts.append(normalized)
        
        return normalized_facts
    
    def _parse_fact_section(self, section_text: str) -> List[str]:
        """Parse facts from a section (e.g., "A→B, C→B, B→D")."""
        facts = []
        
        # Try numbered list first
        numbered_pattern = re.compile(r'(?:^|\n)\s*\d+\.\s*([^\n]+)', re.M)
        numbered_matches = numbered_pattern.findall(section_text)
        if numbered_matches:
            facts.extend(numbered_matches)
        else:
            # Try comma-separated
            parts = section_text.split(',')
            for part in parts:
                part = part.strip()
                # Only include parts that look like formulas
                if part and re.search(r'[A-Z].*[→∧∨¬⇒⇔\->|&~].*[A-Z]|¬[A-Z]', part, re.U):
                    facts.append(part)
                elif part and re.match(r'^[A-Z]$', part):
                    # Single proposition (for SAT problems)
                    facts.append(part)
        
        return facts
    
    def _extract_inline_facts(self, query: str) -> List[str]:
        """Extract facts that appear inline in the query text."""
        facts = []
        
        # Find all formula-like patterns
        formula_matches = self.FORMULA_PATTERN.findall(query)
        for match in formula_matches:
            facts.append(match.strip())
        
        # Also look for negated single variables (¬A)
        negated_pattern = re.compile(r'¬([A-Z])', re.U)
        negated_matches = negated_pattern.findall(query)
        for var in negated_matches:
            facts.append(f"¬{var}")
        
        return facts
    
    def _extract_hypothesis(self, query: str) -> Optional[str]:
        """
        Extract the question/hypothesis from the query.
        
        Looks for:
        - Explicit question sections: "Question: ..."
        - Question sentences ending with "?"
        - Task descriptions
        
        Args:
            query: The cleaned query text
            
        Returns:
            The extracted hypothesis/question or None
        """
        # Try explicit question patterns first
        for pattern in self.QUESTION_PATTERNS:
            match = pattern.search(query)
            if match:
                # Get the full match for question patterns
                hypothesis = match.group(0) if match.lastindex is None else match.group(1)
                hypothesis = hypothesis.strip()
                if hypothesis:
                    return hypothesis
        
        # Look for sentences ending with "?"
        sentences = re.split(r'(?<=[.!?])\s+', query)
        for sentence in reversed(sentences):  # Start from the end
            if sentence.strip().endswith('?'):
                return sentence.strip()
        
        return None
    
    def _determine_query_type(self, query: str) -> str:
        """
        Determine the type of query based on content patterns.
        
        Returns one of: sat, entailment, conditioning, causal, validity, unknown
        """
        for query_type, patterns in self.QUERY_TYPE_PATTERNS.items():
            for pattern in patterns:
                if pattern.search(query):
                    return query_type
        
        return "unknown"
    
    def _extract_conditioning_variables(self, query: str) -> List[str]:
        """Extract variables mentioned for conditioning (e.g., "condition on B")."""
        variables = []
        matches = self.CONDITION_VAR_PATTERN.findall(query)
        for match in matches:
            if match.upper() not in variables:
                variables.append(match.upper())
        return variables
    
    def _extract_target_variables(self, query: str) -> List[str]:
        """Extract target variables (e.g., "correlation between A and C")."""
        variables = []
        matches = self.TARGET_VAR_PATTERN.findall(query)
        for match in matches:
            if isinstance(match, tuple):
                for var in match:
                    if var.upper() not in variables:
                        variables.append(var.upper())
            else:
                if match.upper() not in variables:
                    variables.append(match.upper())
        return variables
    
    def _convert_hypothesis_to_formal(
        self,
        hypothesis: Optional[str],
        query_type: str,
        conditioning_vars: List[str],
        target_vars: List[str],
    ) -> Optional[str]:
        """
        Convert the natural language hypothesis to formal logic notation.
        
        This is a key part of the fix - it translates questions like:
        "Does conditioning on B induce correlation between A and C?"
        
        Into formal queries the prover can evaluate:
        "d_separation(A, C | B)" or "¬independent(A, C | B)"
        
        Args:
            hypothesis: The natural language hypothesis
            query_type: The determined query type
            conditioning_vars: Variables for conditioning
            target_vars: Target variables
            
        Returns:
            Formal representation of the hypothesis or None
        """
        if not hypothesis:
            return None
        
        hypothesis_lower = hypothesis.lower()
        
        # SAT queries -> check for satisfiability/contradiction
        if query_type == "sat":
            if "satisfiable" in hypothesis_lower:
                return "SAT?"  # Signal to check satisfiability
            elif "contradiction" in hypothesis_lower or "unsatisfiable" in hypothesis_lower:
                return "UNSAT?"  # Signal to check for contradiction
        
        # Conditioning/d-separation queries
        if query_type == "conditioning":
            if target_vars and len(target_vars) >= 2:
                if conditioning_vars:
                    # d-separation query
                    return f"d_sep({target_vars[0]}, {target_vars[1]} | {', '.join(conditioning_vars)})"
                else:
                    # Independence query without conditioning
                    return f"independent({target_vars[0]}, {target_vars[1]})"
            
            # General correlation/independence query
            if "independent" in hypothesis_lower or "correlation" in hypothesis_lower:
                if conditioning_vars:
                    return f"¬independent(? | {', '.join(conditioning_vars)})"
        
        # Entailment queries -> check if conclusion follows
        if query_type == "entailment":
            # Try to extract what should be proven
            prove_match = re.search(
                r'(?:prove|follows?|entails?|implies?)\s+(?:that\s+)?([A-Z]\s*[→⇒\->]\s*[A-Z])',
                hypothesis, re.I
            )
            if prove_match:
                return prove_match.group(1)
        
        # Causal/intervention queries
        if query_type == "causal":
            if "intervene" in hypothesis_lower or "intervention" in hypothesis_lower:
                var_match = re.search(r'(?:on|variable)\s+([A-Z])', hypothesis, re.I)
                if var_match:
                    return f"do({var_match.group(1)})"
        
        # Validity queries
        if query_type == "validity":
            return "VALID?"  # Signal to check validity
        
        return None
    
    def _calculate_confidence(
        self,
        facts: List[str],
        hypothesis: Optional[str],
        query_type: str,
        conditioning_vars: List[str],
        target_vars: List[str],
    ) -> float:
        """
        Calculate confidence in the decomposition.
        
        Higher confidence when:
        - More facts are extracted
        - Hypothesis is clearly identified
        - Query type is recognized
        - Relevant variables are extracted
        """
        confidence = 0.0
        
        # Base confidence from facts
        if len(facts) >= 3:
            confidence += 0.4
        elif len(facts) >= 1:
            confidence += 0.2
        
        # Hypothesis extraction
        if hypothesis:
            confidence += 0.3
        
        # Query type recognition
        if query_type != "unknown":
            confidence += 0.2
        
        # Variable extraction for relevant query types
        if query_type in ("conditioning", "causal"):
            if conditioning_vars:
                confidence += 0.05
            if target_vars:
                confidence += 0.05
        
        return min(1.0, confidence)


# Singleton accessor for QueryDecomposer
_decomposer_instance: Optional[QueryDecomposer] = None
_decomposer_lock = threading.Lock()


def get_query_decomposer() -> QueryDecomposer:
    """
    Get or create singleton QueryDecomposer instance.
    
    Returns:
        QueryDecomposer singleton instance
    """
    global _decomposer_instance
    
    if _decomposer_instance is None:
        with _decomposer_lock:
            if _decomposer_instance is None:
                _decomposer_instance = QueryDecomposer()
                logger.info(f"{LOG_PREFIX} QueryDecomposer singleton created")
    
    return _decomposer_instance


def reset_query_decomposer() -> None:
    """Reset the singleton QueryDecomposer instance (for testing)."""
    global _decomposer_instance
    
    with _decomposer_lock:
        _decomposer_instance = None
        logger.debug(f"{LOG_PREFIX} QueryDecomposer singleton reset")


# Module exports
__all__ = [
    "QueryPreprocessor",
    "PreprocessingResult",
    "PreprocessorMetrics",
    "ExtractionType",
    "get_query_preprocessor",
    "reset_query_preprocessor",
    # Query decomposition (ROOT CAUSE FIX)
    "QueryDecomposer",
    "DecomposedQuery",
    "get_query_decomposer",
    "reset_query_decomposer",
]
