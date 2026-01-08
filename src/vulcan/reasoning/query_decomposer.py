"""
Query Decomposer - Extract structured components from natural language queries.

Part of the VULCAN-AGI system.

This module provides query decomposition capabilities for extracting:
- Background facts (formal logic constraints)
- The question/hypothesis to evaluate
- Query type classification
- Conditioning and target variables

ROOT CAUSE FIX: The symbolic reasoner was failing because it:
1. Parses individual formulas correctly (A→B, C→B)
2. Has NO IDEA what the actual question is
3. Returns "no hypothesis" instead of actually reasoning

This module bridges that gap by extracting structured components.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

LOG_PREFIX = "[QueryDecomposer]"


# Operator normalization mapping
# Converts ASCII operators to Unicode for consistent parsing
# IMPORTANT: Longer patterns must come before shorter ones
OPERATOR_NORMALIZATIONS: Tuple[Tuple[str, str], ...] = (
    # Biconditional (MUST come before implication)
    ('<->', '↔'),
    ('iff', '↔'),
    # Implication
    ('=>', '→'),
    ('->', '→'),
    ('implies', '→'),
    # Conjunction
    ('&&', '∧'),
    ('AND', '∧'),
    ('and', '∧'),
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

# Header stripping patterns
# Test queries often include headers that confuse routing
HEADER_STRIP_PATTERNS: Tuple[re.Pattern, ...] = (
    # Full reasoning type headers
    re.compile(
        r'^(?:Analogical|Causal|Mathematical|Probabilistic|Philosophical|Symbolic)\s+Reasoning\s*'
        r'(?:[A-Z][0-9]+\s*)?[—\-:]*\s*',
        re.MULTILINE | re.IGNORECASE
    ),
    # Task labels like "M1 —", "C1 —"
    re.compile(r'^[A-Z][0-9]+\s*[—\-]\s*', re.MULTILINE),
    # Task: / Claim: / Query: prefixes
    re.compile(r'^(?:Task|Claim|Query|Problem):\s*', re.MULTILINE | re.IGNORECASE),
    # Parenthetical notes like "(forces clean reasoning)"
    re.compile(r'\s*\((?:forces?\s+)?clean\s+reasoning\)\s*', re.IGNORECASE),
)


@dataclass
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
    
    # Variable extraction patterns
    CONDITION_VAR_PATTERN = re.compile(
        r'condition(?:ing|ed)?\s+on\s+([A-Z])', re.I
    )
    TARGET_VAR_PATTERN = re.compile(
        r'(?:between|correlation.*?|independent.*?)\s+([A-Z])\s+and\s+([A-Z])', re.I
    )
    
    # Formula extraction patterns
    FORMULA_PATTERN = re.compile(
        r'([A-Z]\s*[→∧∨¬⇒⇔\->|&~]+\s*[A-Z])', re.U
    )
    EXTENDED_FORMULA_PATTERN = re.compile(
        r'[A-Z].*[→∧∨¬⇒⇔\->|&~].*[A-Z]|¬[A-Z]', re.U
    )
    
    def __init__(self) -> None:
        """Initialize the query decomposer."""
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
        
        # Strip headers first
        cleaned_query = self._strip_headers(query)
        
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
    
    def _strip_headers(self, query: str) -> str:
        """Strip test headers and labels from queries."""
        if not query:
            return query
        
        cleaned = query.strip()
        for pattern in HEADER_STRIP_PATTERNS:
            cleaned = pattern.sub('', cleaned).strip()
        
        return cleaned
    
    def _normalize_operators(self, text: str) -> str:
        """Normalize logical operators to Unicode symbols."""
        result = text
        for pattern, replacement in OPERATOR_NORMALIZATIONS:
            result = result.replace(pattern, replacement)
        return result.strip()
    
    def _extract_facts(self, query: str) -> List[str]:
        """Extract background facts/constraints from the query."""
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
            normalized = self._normalize_operators(fact.strip())
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
                if part and self.EXTENDED_FORMULA_PATTERN.search(part):
                    facts.append(part)
                elif part and re.match(r'^[A-Z]$', part):
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
        """Extract the question/hypothesis from the query."""
        # Try explicit question patterns first
        for pattern in self.QUESTION_PATTERNS:
            match = pattern.search(query)
            if match:
                hypothesis = match.group(0) if match.lastindex is None else match.group(1)
                hypothesis = hypothesis.strip()
                if hypothesis:
                    return hypothesis
        
        # Look for sentences ending with "?"
        sentences = re.split(r'(?<=[.!?])\s+', query)
        for sentence in reversed(sentences):
            if sentence.strip().endswith('?'):
                return sentence.strip()
        
        return None
    
    def _determine_query_type(self, query: str) -> str:
        """Determine the type of query based on content patterns."""
        for query_type, patterns in self.QUERY_TYPE_PATTERNS.items():
            for pattern in patterns:
                if pattern.search(query):
                    return query_type
        
        return "unknown"
    
    def _extract_conditioning_variables(self, query: str) -> List[str]:
        """Extract variables mentioned for conditioning."""
        variables = []
        matches = self.CONDITION_VAR_PATTERN.findall(query)
        for match in matches:
            if match.upper() not in variables:
                variables.append(match.upper())
        return variables
    
    def _extract_target_variables(self, query: str) -> List[str]:
        """Extract target variables."""
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
        """Convert the natural language hypothesis to formal logic notation."""
        if not hypothesis:
            return None
        
        hypothesis_lower = hypothesis.lower()
        
        # SAT queries
        if query_type == "sat":
            if "satisfiable" in hypothesis_lower:
                return "SAT?"
            elif "contradiction" in hypothesis_lower or "unsatisfiable" in hypothesis_lower:
                return "UNSAT?"
        
        # Conditioning/d-separation queries
        if query_type == "conditioning":
            if target_vars and len(target_vars) >= 2:
                if conditioning_vars:
                    return f"d_sep({target_vars[0]}, {target_vars[1]} | {', '.join(conditioning_vars)})"
                else:
                    return f"independent({target_vars[0]}, {target_vars[1]})"
            
            if "independent" in hypothesis_lower or "correlation" in hypothesis_lower:
                if conditioning_vars:
                    return f"¬independent(? | {', '.join(conditioning_vars)})"
        
        # Entailment queries
        if query_type == "entailment":
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
            return "VALID?"
        
        return None
    
    def _calculate_confidence(
        self,
        facts: List[str],
        hypothesis: Optional[str],
        query_type: str,
        conditioning_vars: List[str],
        target_vars: List[str],
    ) -> float:
        """Calculate confidence in the decomposition."""
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
        
        # Variable extraction
        if query_type in ("conditioning", "causal"):
            if conditioning_vars:
                confidence += 0.05
            if target_vars:
                confidence += 0.05
        
        return min(1.0, confidence)


# Singleton accessor
_decomposer_instance: Optional[QueryDecomposer] = None
_decomposer_lock = threading.Lock()


def get_query_decomposer() -> QueryDecomposer:
    """Get or create singleton QueryDecomposer instance."""
    global _decomposer_instance
    
    if _decomposer_instance is None:
        with _decomposer_lock:
            if _decomposer_instance is None:
                _decomposer_instance = QueryDecomposer()
                logger.info(f"{LOG_PREFIX} QueryDecomposer singleton created")
    
    return _decomposer_instance
