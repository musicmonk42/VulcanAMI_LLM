# ============================================================
# VULCAN World Model LLM Guidance Builder
# Builds structured guidance for LLM generation
# ============================================================
"""
llm_guidance.py - LLM guidance construction for World Model orchestration

This module builds structured guidance for LLM generation across different request
types (reasoning, knowledge synthesis, creative, ethical, conversational). It combines
verified content, constraints, permissions, and formatting into clear instructions.

Industry Standard: Separation of concerns - guidance building separate from content
generation, with clear interfaces, comprehensive documentation, and type safety.

Architecture:
    LLMGuidanceBuilder constructs guidance for:
    1. Reasoning results → Natural language explanation
    2. Verified knowledge → Structured synthesis (papers, explanations)
    3. Creative guidance → Creative generation with constraints
    4. Ethical analysis → Philosophical reasoning output
    5. Conversational → Direct conversational response
    
Integration:
    - Used by WorldModel handlers to prepare LLM generation
    - Integrates with reasoning engines, knowledge handlers, creative handlers
    - Provides consistent interface across all request types
"""

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from vulcan.reasoning.reasoning_types import ReasoningResult
    from vulcan.world_model.knowledge_handler import VerifiedKnowledge
    from vulcan.world_model.creative_handler import CreativeGuidance

logger = logging.getLogger(__name__)


@dataclass
class LLMGuidance:
    """
    Complete guidance for LLM generation.
    
    Industry Standard: Comprehensive data structure with all information
    needed for high-quality LLM generation, with validation and clear documentation.
    
    Attributes:
        task: Description of generation task
        verified_content: Verified facts, knowledge, or reasoning results
        structure: Required output structure (sections, format, etc.)
        constraints: What the LLM must/must not do
        permissions: What the LLM is allowed to do
        tone: Desired tone (formal, conversational, educational, etc.)
        format: Output format (markdown, plain text, structured, etc.)
        max_length: Maximum output length (words or tokens)
        metadata: Additional context (confidence scores, sources, etc.)
    """
    
    task: str
    verified_content: Dict[str, Any]
    structure: Dict[str, Any]
    constraints: List[str]
    permissions: List[str]
    tone: str
    format: str
    max_length: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate LLM guidance."""
        if not self.task or not isinstance(self.task, str):
            raise ValueError("task must be a non-empty string")
        
        if not isinstance(self.verified_content, dict):
            raise TypeError("verified_content must be a dictionary")
        
        if not isinstance(self.structure, dict):
            raise TypeError("structure must be a dictionary")
        
        if not isinstance(self.constraints, list):
            raise TypeError("constraints must be a list")
        
        if not isinstance(self.permissions, list):
            raise TypeError("permissions must be a list")
        
        if not self.tone or not isinstance(self.tone, str):
            raise ValueError("tone must be a non-empty string")
        
        if not self.format or not isinstance(self.format, str):
            raise ValueError("format must be a non-empty string")
        
        if self.max_length is not None and self.max_length <= 0:
            raise ValueError(f"max_length must be positive, got {self.max_length}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'task': self.task,
            'verified_content': self.verified_content,
            'structure': self.structure,
            'constraints': self.constraints,
            'permissions': self.permissions,
            'tone': self.tone,
            'format': self.format,
            'max_length': self.max_length,
            'metadata': self.metadata,
        }


class LLMGuidanceBuilder:
    """
    Builds structured guidance for LLM generation.
    
    Industry Standard: Builder pattern with clear separation of concerns,
    type safety, comprehensive documentation, and consistent interface.
    
    Architecture:
        Each request type has a dedicated build method:
        - build_for_reasoning: Reasoning result → Natural language
        - build_for_knowledge_synthesis: Knowledge → Structured content
        - build_for_creative: Creative guidance → Creative content
        - build_for_ethical: Ethical analysis → Philosophical reasoning
        - build_for_conversational: Query → Conversational response
    
    Integration:
        - Used by WorldModel for all request types
        - Provides consistent guidance structure
        - Combines content, constraints, and formatting
    """
    
    # Industry Standard: Class-level constants for universal constraints and permissions
    UNIVERSAL_CONSTRAINTS = [
        "Do NOT generate facts without verification",  # Match test expectation
        "Do not fabricate information or sources",
        "Do not contradict verified facts",
        "Do not make unsupported claims without evidence",
        "Do not include harmful, biased, or inappropriate content",
        "Acknowledge uncertainty when appropriate",
        "Cite sources when available",
    ]
    
    UNIVERSAL_PERMISSIONS = [
        "Use natural language and clear explanations",
        "Organize information logically",
        "Use examples and analogies when helpful",
        "Provide context and background information",
        "Ask clarifying questions if needed",
        "Adapt word choice for clarity and precision",  # Match test expectation
    ]
    
    def build_for_reasoning(
        self,
        reasoning_result: Dict[str, Any],
        original_query: str,
    ) -> LLMGuidance:
        """
        Build guidance for explaining reasoning results.
        
        Industry Standard: Transform formal reasoning output into
        natural language explanation with clear structure.
        
        Args:
            reasoning_result: Dictionary with reasoning output (conclusion, confidence, proof, etc.)
            original_query: Original user query for context
        
        Returns:
            LLMGuidance for natural language explanation
        """
        logger.info(
            f"[LLMGuidanceBuilder] Building reasoning guidance: "
            f"confidence={reasoning_result.get('confidence', 0.0)}"
        )
        
        # Extract reasoning information - handle dict input
        verified_content = {
            'conclusion': reasoning_result.get('conclusion'),
            'confidence': reasoning_result.get('confidence'),
            'reasoning_type': reasoning_result.get('reasoning_type'),
            'explanation': reasoning_result.get('explanation'),
            'evidence': reasoning_result.get('evidence'),
            'uncertainty': reasoning_result.get('uncertainty'),
            'proof': reasoning_result.get('proof'),
            'status': reasoning_result.get('status'),
        }
        
        # Build structure for explanation
        structure = {
            'sections': [
                'answer',
                'reasoning_steps',
                'confidence_assessment',
                'caveats',
            ],
            'format': 'clear and structured',
        }
        
        # Reasoning-specific constraints
        constraints = self.UNIVERSAL_CONSTRAINTS.copy()
        constraints.extend([
            "Explain reasoning steps clearly",
            "Indicate confidence level and uncertainty",
            "Present conclusion prominently",
            "Base explanation on provided reasoning",
        ])
        
        # Reasoning-specific permissions
        permissions = self.UNIVERSAL_PERMISSIONS.copy()
        permissions.extend([
            "Use mathematical notation if helpful",
            "Provide step-by-step derivation",
            "Explain logical connections",
        ])
        
        return LLMGuidance(
            task=f"Explain the reasoning result for: {original_query}",
            verified_content=verified_content,
            structure=structure,
            constraints=constraints,
            permissions=permissions,
            tone='clear and educational',
            format='markdown',
            max_length=500,
            metadata={
                'reasoning_type': reasoning_result.get('reasoning_type'),
                'confidence': reasoning_result.get('confidence', 0.0),
            },
        )
    
    def build_for_knowledge_synthesis(
        self,
        verified_knowledge: 'VerifiedKnowledge',
        topic: str,
        requested_format: str = 'explanation',
    ) -> LLMGuidance:
        """
        Build guidance for knowledge synthesis.
        
        Industry Standard: Transform verified knowledge into requested format
        (paper, explanation, summary) with proper structure and citations.
        
        Args:
            verified_knowledge: Verified knowledge from KnowledgeHandler
            topic: Topic for synthesis
            requested_format: Desired output format (paper, explanation, summary)
        
        Returns:
            LLMGuidance for knowledge synthesis
        """
        logger.info(
            f"[LLMGuidanceBuilder] Building knowledge synthesis guidance: "
            f"topic={topic}, format={requested_format}, "
            f"{len(verified_knowledge.verified_facts)} verified facts"
        )
        
        # Extract verified knowledge
        verified_content = {
            'facts': verified_knowledge.verified_facts,  # Use 'facts' key for test compatibility
            'verified_facts': verified_knowledge.verified_facts,
            'unverified_facts': verified_knowledge.unverified_facts,
            'conflicts': verified_knowledge.conflicts,
            'verification_confidence': verified_knowledge.confidence,
            'topic': topic,
        }
        
        # Get structure for requested format
        structure = self._get_synthesis_structure(requested_format)
        
        # Knowledge synthesis constraints
        constraints = self.UNIVERSAL_CONSTRAINTS.copy()
        constraints.extend([
            "Only use verified facts from provided knowledge",
            "Clearly indicate unverified information if included",
            "Flag any conflicts or contradictions",
            "Organize information logically by topic",
        ])
        
        # Knowledge synthesis permissions
        permissions = self.UNIVERSAL_PERMISSIONS.copy()
        permissions.extend([
            "Connect related facts and concepts",
            "Provide transitions between ideas",
            "Use headings and subheadings",
            "Include relevant details and context",
        ])
        
        # Determine tone and length for format
        tone = self._get_tone_for_format(requested_format)
        max_length = self._get_length_for_format(requested_format)
        
        return LLMGuidance(
            task=f"Synthesize knowledge about {topic} in {requested_format} format",
            verified_content=verified_content,
            structure=structure,
            constraints=constraints,
            permissions=permissions,
            tone=tone,
            format=requested_format,  # Use requested_format for test compatibility
            max_length=max_length,
            metadata={
                'requested_format': requested_format,
                'topic': topic,
                'verification_confidence': verified_knowledge.confidence,
                'num_facts': len(verified_knowledge.verified_facts),
            },
        )
    
    def build_for_creative(
        self,
        creative_guidance: 'CreativeGuidance',
        original_query: str,
    ) -> LLMGuidance:
        """
        Build guidance for creative generation.
        
        Industry Standard: Combine creative freedom with factual accuracy,
        using constraints to ensure knowledge grounding.
        
        Args:
            creative_guidance: Creative guidance from CreativeHandler
            original_query: Original user query
        
        Returns:
            LLMGuidance for creative generation
        """
        logger.info(
            f"[LLMGuidanceBuilder] Building creative guidance: "
            f"format={creative_guidance.format}, "
            f"domain={creative_guidance.subject_knowledge.domain}"
        )
        
        # Extract creative information
        verified_content = {
            'subject_facts': creative_guidance.subject_knowledge.facts,
            'subject_definitions': creative_guidance.subject_knowledge.definitions,
            'subject_equations': creative_guidance.subject_knowledge.equations,
            'sources': creative_guidance.subject_knowledge.sources,
            'must_be_accurate': creative_guidance.constraints.must_be_accurate,
            'must_avoid': creative_guidance.constraints.must_avoid,
        }
        
        # Creative structure
        structure = creative_guidance.structure.copy()
        structure['style_guidance'] = creative_guidance.constraints.style_guidance
        
        # Creative constraints
        constraints = self.UNIVERSAL_CONSTRAINTS.copy()
        constraints.extend([
            f"Maintain {creative_guidance.constraints.tone} tone",
            "Ensure factual accuracy for: " + ", ".join(
                creative_guidance.constraints.must_be_accurate[:3]  # First 3 for brevity
            ),
            "Avoid misconceptions: " + ", ".join(
                creative_guidance.constraints.must_avoid[:3]  # First 3 for brevity
            ),
            f"Follow {creative_guidance.format} structure",
        ])
        
        # Creative permissions
        permissions = self.UNIVERSAL_PERMISSIONS.copy()
        permissions.extend([
            "Use creative language and imagery",
            "Develop narrative or poetic elements",
            "Be artistic with: " + ", ".join(
                creative_guidance.constraints.can_be_artistic[:3]
            ),
            "Express emotions and themes",
        ])
        
        return LLMGuidance(
            task=f"Create a {creative_guidance.format} about: {original_query}",
            verified_content=verified_content,
            structure=structure,
            constraints=constraints,
            permissions=permissions,
            tone=creative_guidance.constraints.tone,
            format='plain_text',
            max_length=1000,
            metadata={
                'creative_format': creative_guidance.format,
                'subject': creative_guidance.subject_knowledge.topic,
                'domain': creative_guidance.subject_knowledge.domain,
                'knowledge_confidence': creative_guidance.subject_knowledge.confidence,
            },
        )
    
    def build_for_ethical(
        self,
        ethical_analysis: Dict[str, Any],
        original_query: str,
    ) -> LLMGuidance:
        """
        Build guidance for ethical/philosophical reasoning.
        
        Industry Standard: Structure philosophical reasoning with multiple
        perspectives, acknowledge complexity and nuance.
        
        Args:
            ethical_analysis: Ethical analysis from meta-reasoning
            original_query: Original ethical query
        
        Returns:
            LLMGuidance for ethical reasoning explanation
        """
        logger.info(
            f"[LLMGuidanceBuilder] Building ethical guidance for: {original_query}"
        )
        
        # Extract ethical analysis
        verified_content = {
            'perspectives': ethical_analysis.get('perspectives', []),
            'principles': ethical_analysis.get('principles', []),
            'considerations': ethical_analysis.get('considerations', []),
            'conflicts': ethical_analysis.get('conflicts', []),
        }
        
        # Ethical structure
        structure = {
            'sections': [
                'question_framing',
                'multiple_perspectives',
                'ethical_principles',
                'considerations',
                'conclusion',
            ],
            'format': 'balanced and nuanced',
        }
        
        # Ethical constraints
        constraints = self.UNIVERSAL_CONSTRAINTS.copy()
        constraints.extend([
            "Present multiple ethical perspectives fairly",
            "Acknowledge complexity and uncertainty",
            "Avoid imposing single moral view",
            "Consider consequences and principles",
            "Respect diverse value systems",
        ])
        
        # Ethical permissions
        permissions = self.UNIVERSAL_PERMISSIONS.copy()
        permissions.extend([
            "Explore thought experiments",
            "Discuss philosophical frameworks",
            "Compare different ethical theories",
            "Raise important considerations",
        ])
        
        return LLMGuidance(
            task=f"Provide ethical reasoning for: {original_query}",
            verified_content=verified_content,
            structure=structure,
            constraints=constraints,
            permissions=permissions,
            tone='thoughtful and balanced',
            format='markdown',
            max_length=800,
            metadata={
                'ethical_query': original_query,
                'num_perspectives': len(verified_content.get('perspectives', [])),
            },
        )
    
    def build_for_conversational(
        self,
        query: str,
    ) -> LLMGuidance:
        """
        Build guidance for conversational response.
        
        Industry Standard: Simple, direct conversational guidance for
        greetings, meta-questions, and general queries.
        
        Args:
            query: User's conversational query
        
        Returns:
            LLMGuidance for conversational response
        """
        logger.info(f"[LLMGuidanceBuilder] Building conversational guidance")
        
        # Minimal verified content for conversational
        verified_content = {
            'query': query,
            'response_type': 'conversational',
        }
        
        # Simple conversational structure
        structure = {
            'format': 'direct and friendly',
            'length': 'concise',
        }
        
        # Conversational constraints
        constraints = self.UNIVERSAL_CONSTRAINTS.copy()
        constraints.extend([
            "Be helpful and friendly",
            "Keep response concise and relevant",
            "Answer directly if possible",
        ])
        
        # Conversational permissions
        permissions = self.UNIVERSAL_PERMISSIONS.copy()
        permissions.extend([
            "Use conversational language",
            "Be warm and engaging",
            "Ask follow-up questions if helpful",
        ])
        
        return LLMGuidance(
            task=f"Respond conversationally to: {query}",
            verified_content=verified_content,
            structure=structure,
            constraints=constraints,
            permissions=permissions,
            tone='friendly and helpful',
            format='conversation',  # Use 'conversation' for test compatibility
            max_length=200,
            metadata={
                'query_type': 'conversational',
            },
        )
    
    def _get_synthesis_structure(self, requested_format: str) -> Dict[str, Any]:
        """
        Get structure for knowledge synthesis format.
        
        Industry Standard: Format-specific structure templates with
        clear expectations for each format type.
        """
        structures = {
            'paper': {
                'sections': [
                    'abstract',
                    'introduction',
                    'background',
                    'main_content',
                    'conclusion',
                    'references',
                ],
                'format': 'academic',
            },
            'explanation': {
                'sections': [
                    'overview',
                    'key_concepts',
                    'detailed_explanation',
                    'examples',
                    'summary',
                ],
                'format': 'educational',
            },
            'summary': {
                'sections': [
                    'key_points',
                    'supporting_details',
                    'conclusion',
                ],
                'format': 'concise',
            },
            'essay': {
                'sections': [
                    'introduction',
                    'body_paragraphs',
                    'conclusion',
                ],
                'format': 'structured',
            },
        }
        
        return structures.get(
            requested_format,
            {'sections': ['content'], 'format': 'flexible'}
        )
    
    def _get_tone_for_format(self, requested_format: str) -> str:
        """
        Get appropriate tone for knowledge format.
        
        Industry Standard: Format-specific tone matching with clear defaults.
        """
        tone_map = {
            'paper': 'formal and academic',
            'explanation': 'clear and educational',
            'summary': 'concise and informative',
            'essay': 'structured and analytical',
        }
        
        return tone_map.get(requested_format, 'informative')
    
    def _get_length_for_format(self, requested_format: str) -> int:
        """
        Get appropriate length for knowledge format.
        
        Industry Standard: Format-specific length guidelines with reasonable defaults.
        """
        length_map = {
            'paper': 2000,      # Academic paper
            'explanation': 800,  # Detailed explanation
            'summary': 300,      # Concise summary
            'essay': 1200,       # Standard essay
        }
        
        return length_map.get(requested_format, 500)
