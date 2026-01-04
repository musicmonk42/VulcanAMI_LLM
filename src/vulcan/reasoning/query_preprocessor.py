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
        self._constraint_line_pattern = re.compile(
            r'(?:\d+\.|•|[-*])\s*([^\n]+)'
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

        # Thread-safe metrics
        self._metrics = PreprocessorMetrics()
        self._metrics_lock = threading.Lock()

        logger.debug(f"{LOG_PREFIX} Initialized with {len(self._logical_operators)} operators")

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

        # Update metrics
        with self._metrics_lock:
            self._metrics.total_queries += 1

        # Create default result (no preprocessing)
        default_result = PreprocessingResult(
            formal_input=None,
            original_query=query,
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
        if 'symbolic' in tools_set:
            return self._extract_symbolic(query)
        elif 'mathematical' in tools_set:
            return self._extract_mathematical(query)
        elif 'probabilistic' in tools_set:
            return self._extract_probabilistic(query)

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

        # Parse constraints (numbered list or bullet points)
        constraint_matches = self._constraint_line_pattern.findall(constraints_str)

        if not constraint_matches:
            return PreprocessingResult(
                formal_input=None,
                original_query=match.string,
                preprocessing_applied=False,
                extraction_confidence=0.0,
                extraction_type=ExtractionType.SYMBOLIC,
            )

        # Normalize each constraint
        formal_constraints = []
        for constraint in constraint_matches:
            normalized = self._normalize_operators(constraint.strip())
            if normalized:
                formal_constraints.append(normalized)

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

    def _clean_formula_line(self, line: str) -> str:
        """
        Clean a line containing a formula.

        Removes common prefixes (numbers, bullets) and trailing punctuation,
        then normalizes operators.

        Args:
            line: Line to clean

        Returns:
            Cleaned formula string
        """
        # Remove common prefixes (numbered lists, bullets)
        cleaned = re.sub(r'^(?:\d+\.|\*|•|[-])\s*', '', line)

        # Remove trailing punctuation except parentheses
        cleaned = re.sub(r'[,;]$', '', cleaned)

        # Normalize operators
        cleaned = self._normalize_operators(cleaned)

        return cleaned.strip()

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


# Module exports
__all__ = [
    "QueryPreprocessor",
    "PreprocessingResult",
    "PreprocessorMetrics",
    "ExtractionType",
    "get_query_preprocessor",
    "reset_query_preprocessor",
]
