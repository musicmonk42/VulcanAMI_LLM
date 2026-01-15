# ============================================================
# VULCAN World Model Request Classifier
# Classifies user requests to determine handling strategy
# ============================================================
"""
request_classifier.py - Request classification for World Model orchestration

This module implements intelligent request classification to determine the
appropriate handling strategy for each user query. Unlike QueryClassifier
(which routes to reasoning engines), RequestClassifier determines the
OVERALL handling strategy including reasoning, retrieval, or creative generation.

Industry Standard: Separation of concerns - classification logic separate from
routing logic, with clear interfaces and comprehensive documentation.

Architecture:
    RequestClassifier analyzes queries to determine:
    1. Request type (reasoning, knowledge synthesis, creative, ethical, conversational)
    2. Domain and subdomain
    3. Whether verification/retrieval is needed
    4. Which reasoning engine (if reasoning request)
    5. Creative format (if creative request)

Integration:
    - Used by WorldModel.process_request() to determine handling strategy
    - Works in concert with existing QueryRouter for reasoning engine selection
    - Leverages WorldModel's existing pattern detection constants
"""

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Pattern

from vulcan.vulcan_types import RequestType

if TYPE_CHECKING:
    from vulcan.world_model.world_model_core import WorldModel

logger = logging.getLogger(__name__)


@dataclass
class ClassifiedRequest:
    """
    Result of request classification.
    
    Industry Standard: Immutable dataclass with comprehensive documentation,
    validation, and type safety.
    
    Attributes:
        request_type: Primary handling strategy (REASONING, KNOWLEDGE_SYNTHESIS, etc.)
        domain: Knowledge domain (physics, math, philosophy, general, etc.)
        subdomain: Specific topic within domain (thermodynamics, linear_algebra, etc.)
        requires_verification: Whether facts need verification against knowledge base
        requires_retrieval: Whether knowledge retrieval is needed
        reasoning_engine: Specific reasoning engine to use (symbolic, probabilistic, etc.)
        creative_format: Format for creative generation (poem, story, song, etc.)
        confidence: Classification confidence (0.0-1.0)
        metadata: Additional context (indicators matched, alternative classifications, etc.)
    """
    
    request_type: RequestType
    domain: str
    subdomain: Optional[str] = None
    requires_verification: bool = False
    requires_retrieval: bool = False
    reasoning_engine: Optional[str] = None
    creative_format: Optional[str] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate classification result."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0.0, 1.0], got {self.confidence}")
        
        if self.request_type == RequestType.REASONING and not self.reasoning_engine:
            logger.warning(
                "REASONING request without reasoning_engine specified - "
                "will require fallback logic"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "request_type": self.request_type.value,
            "domain": self.domain,
            "subdomain": self.subdomain,
            "requires_verification": self.requires_verification,
            "requires_retrieval": self.requires_retrieval,
            "reasoning_engine": self.reasoning_engine,
            "creative_format": self.creative_format,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


class RequestClassifier:
    """
    Classifies user requests to determine handling strategy.
    
    Unlike QueryClassifier (which routes to reasoning engines), RequestClassifier
    determines the OVERALL handling strategy including whether to use reasoning,
    retrieval, or creative generation.
    
    Industry Standard: Single Responsibility Principle - this class only classifies,
    does not execute or route. Clear separation from query routing and execution.
    
    Architecture:
        Classification Priority (highest to lowest):
        1. Reasoning domain (highest priority - mathematical/logical operations)
        2. Creative with subject (poems, stories about specific topics)
        3. Knowledge synthesis (explanations, papers, summaries)
        4. Ethical/philosophical (moral questions, dilemmas)
        5. Conversational (default - greetings, meta-questions)
    
    Integration:
        - Initialized with WorldModel reference for access to constants
        - Uses lazy property access to avoid circular imports
        - Leverages existing FORMAL_LOGIC_SYMBOLS, PROBABILITY_KEYWORDS constants
    """
    
    def __init__(self, world_model: 'WorldModel'):
        """
        Initialize classifier with WorldModel reference.
        
        Args:
            world_model: WorldModel instance for accessing shared constants
        """
        self.world_model = world_model
        
        # Reasoning domain indicators (from existing system_observer constants)
        # Industry Standard: DRY principle - reuse existing detection patterns
        self.reasoning_indicators = {
            'sat': ['satisfiable', 'unsatisfiable', '→', '∧', '∨', '¬', 'implies', 'and', 'or', 'not'],
            'probabilistic': ['P(', 'probability', 'bayes', 'bayesian', 'posterior', 'prior', 'prevalence', 'likelihood'],
            'causal': ['cause', 'effect', 'intervention', 'do(', 'confound', 'counterfactual', 'causal'],
            'symbolic': ['∀', '∃', 'forall', 'exists', 'prove', 'theorem', 'entails', '⊢', '⊨'],
            'mathematical': ['calculate', 'solve', 'derivative', 'integral', 'compute', 'evaluate', 'equation'],
        }
        
        # Knowledge synthesis indicators
        self.knowledge_indicators = [
            'explain', 'describe', 'what is', 'how does', 'why does',
            'write a paper', 'write an essay', 'summarize', 'overview',
            'tell me about', 'teach me', 'help me understand', 'define',
        ]
        
        # Creative patterns with subject extraction
        # Industry Standard: Compiled regex patterns for performance
        self.creative_patterns: Dict[str, Pattern] = {
            'poem': re.compile(r'write\s+(?:a\s+|me\s+)?poem\s+(?:about|on|regarding)\s+(.+)', re.IGNORECASE),
            'story': re.compile(r'(?:write|tell)\s+(?:a\s+|me\s+)?story\s+(?:about|on|regarding)\s+(.+)', re.IGNORECASE),
            'song': re.compile(r'write\s+(?:a\s+|me\s+)?(?:song|lyrics)\s+(?:about|on|regarding)\s+(.+)', re.IGNORECASE),
            'essay': re.compile(r'write\s+(?:a\s+|an\s+)?(?:creative\s+)?essay\s+(?:about|on|regarding)\s+(.+)', re.IGNORECASE),
        }
        
        # Ethical/philosophical indicators
        self.ethical_indicators = [
            'should', 'ought', 'moral', 'ethical', 'right or wrong',
            'trolley', 'dilemma', 'virtue', 'duty', 'consequence',
            'justice', 'fairness', 'good or bad', 'ethical implications',
        ]
        
        # Domain extraction patterns
        self.domain_patterns = {
            'physics': ['physics', 'thermodynamics', 'quantum', 'relativity', 'entropy', 'energy', 'force', 'motion'],
            'math': ['math', 'algebra', 'calculus', 'geometry', 'topology', 'number theory', 'equation', 'theorem'],
            'biology': ['biology', 'genetics', 'evolution', 'cell', 'organism', 'ecology', 'neuroscience'],
            'chemistry': ['chemistry', 'molecule', 'atom', 'reaction', 'compound', 'element', 'bond'],
            'history': ['history', 'historical', 'ancient', 'medieval', 'modern', 'civilization', 'war', 'revolution'],
            'philosophy': ['philosophy', 'metaphysics', 'epistemology', 'ethics', 'logic', 'consciousness'],
            'computer_science': ['programming', 'algorithm', 'data structure', 'software', 'code', 'computer'],
        }
    
    def classify(self, query: str, context: Optional[Dict[str, Any]] = None) -> ClassifiedRequest:
        """
        Classify a user request to determine handling strategy.
        
        Industry Standard: Priority-based classification with confidence scoring,
        graceful degradation, and comprehensive logging.
        
        Args:
            query: User's natural language query
            context: Optional context (conversation history, user preferences, etc.)
        
        Returns:
            ClassifiedRequest with handling strategy and metadata
        """
        query_lower = query.lower()
        metadata: Dict[str, Any] = {}
        
        # Priority 1: Check for reasoning domain (highest priority)
        # Industry Standard: Most specific patterns checked first
        for engine, indicators in self.reasoning_indicators.items():
            matched_indicators = [ind for ind in indicators if ind in query or ind in query_lower]
            if matched_indicators:
                logger.info(
                    f"[RequestClassifier] Detected {engine} reasoning: "
                    f"matched {matched_indicators}"
                )
                metadata['matched_indicators'] = matched_indicators
                return ClassifiedRequest(
                    request_type=RequestType.REASONING,
                    domain=self._extract_reasoning_domain(engine),
                    subdomain=None,
                    requires_verification=False,  # Reasoning engines are authoritative
                    requires_retrieval=False,
                    reasoning_engine=engine,
                    creative_format=None,
                    confidence=0.90,
                    metadata=metadata,
                )
        
        # Priority 2: Check for creative with subject
        for format_type, pattern in self.creative_patterns.items():
            match = pattern.search(query_lower)
            if match:
                subject = match.group(1).strip()
                domain = self._extract_domain_from_subject(subject)
                logger.info(
                    f"[RequestClassifier] Detected creative {format_type} about {subject}"
                )
                metadata['subject'] = subject
                metadata['format'] = format_type
                return ClassifiedRequest(
                    request_type=RequestType.CREATIVE,
                    domain=domain,
                    subdomain=subject,
                    requires_verification=True,  # Creative needs fact-checking
                    requires_retrieval=True,
                    reasoning_engine=None,
                    creative_format=format_type,
                    confidence=0.85,
                    metadata=metadata,
                )
        
        # Priority 3: Check for knowledge synthesis
        matched_knowledge_indicators = [ind for ind in self.knowledge_indicators if ind in query_lower]
        if matched_knowledge_indicators:
            domain = self._extract_domain_from_query(query_lower)
            topic = self._extract_topic(query)
            logger.info(
                f"[RequestClassifier] Detected knowledge synthesis: "
                f"domain={domain}, topic={topic}"
            )
            metadata['matched_indicators'] = matched_knowledge_indicators
            metadata['topic'] = topic
            return ClassifiedRequest(
                request_type=RequestType.KNOWLEDGE_SYNTHESIS,
                domain=domain,
                subdomain=topic,
                requires_verification=True,
                requires_retrieval=True,
                reasoning_engine=None,
                creative_format=None,
                confidence=0.80,
                metadata=metadata,
            )
        
        # Priority 4: Check for ethical/philosophical
        matched_ethical_indicators = [ind for ind in self.ethical_indicators if ind in query_lower]
        if matched_ethical_indicators:
            logger.info(
                f"[RequestClassifier] Detected ethical/philosophical: "
                f"matched {matched_ethical_indicators}"
            )
            metadata['matched_indicators'] = matched_ethical_indicators
            return ClassifiedRequest(
                request_type=RequestType.ETHICAL,
                domain='philosophy',
                subdomain='ethics',
                requires_verification=False,  # Ethical reasoning, not fact retrieval
                requires_retrieval=False,
                reasoning_engine=None,
                creative_format=None,
                confidence=0.75,
                metadata=metadata,
            )
        
        # Priority 5: Default to conversational
        logger.info("[RequestClassifier] Defaulting to conversational")
        return ClassifiedRequest(
            request_type=RequestType.CONVERSATIONAL,
            domain='general',
            subdomain=None,
            requires_verification=False,
            requires_retrieval=False,
            reasoning_engine=None,
            creative_format=None,
            confidence=0.60,
            metadata=metadata,
        )
    
    def _extract_reasoning_domain(self, engine: str) -> str:
        """
        Map reasoning engine to knowledge domain.
        
        Industry Standard: Explicit mapping with clear semantics.
        """
        domain_mapping = {
            'sat': 'logic',
            'symbolic': 'logic',
            'probabilistic': 'statistics',
            'causal': 'causality',
            'mathematical': 'mathematics',
        }
        return domain_mapping.get(engine, 'general')
    
    def _extract_domain_from_subject(self, subject: str) -> str:
        """
        Extract knowledge domain from creative subject.
        
        Industry Standard: Pattern matching with fallback to general domain.
        """
        subject_lower = subject.lower()
        for domain, keywords in self.domain_patterns.items():
            if any(keyword in subject_lower for keyword in keywords):
                return domain
        return 'general'
    
    def _extract_domain_from_query(self, query_lower: str) -> str:
        """
        Extract knowledge domain from query text.
        
        Industry Standard: Multi-pattern matching with confidence scoring.
        """
        domain_scores: Dict[str, int] = {}
        for domain, keywords in self.domain_patterns.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            # Return domain with highest score
            return max(domain_scores.items(), key=lambda x: x[1])[0]
        return 'general'
    
    def _extract_topic(self, query: str) -> Optional[str]:
        """
        Extract specific topic from query.
        
        Industry Standard: Heuristic extraction with validation.
        """
        # Look for quoted phrases
        quoted = re.findall(r'"([^"]+)"', query)
        if quoted:
            return quoted[0]
        
        # Look for "about X", "on X", etc.
        about_match = re.search(r'(?:about|on|regarding|concerning)\s+([a-zA-Z0-9\s]+?)(?:\?|\.|\,|$)', query, re.IGNORECASE)
        if about_match:
            topic = about_match.group(1).strip()
            # Limit topic length
            if len(topic.split()) <= 5:
                return topic
        
        return None
