# ============================================================
# VULCAN World Model Creative Handler
# Handles creative content generation with knowledge grounding
# ============================================================
"""
creative_handler.py - Creative content generation with fact verification

This module implements creative content generation (poems, stories, songs) that is
grounded in verified knowledge. It ensures creative outputs are factually accurate
while maintaining artistic freedom.

Industry Standard: Separation of concerns - creative logic separate from knowledge
retrieval and verification, with clear interfaces and comprehensive documentation.

Architecture:
    CreativeHandler coordinates:
    1. Knowledge retrieval for subject matter
    2. Constraint building (accuracy requirements, misconceptions to avoid)
    3. Output verification (fact-checking creative content)
    
Integration:
    - Used by WorldModel for CREATIVE request handling
    - Leverages KnowledgeHandler for knowledge retrieval
    - Provides verification methods for creative outputs
"""

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from vulcan.world_model.world_model_core import WorldModel
    from vulcan.world_model.knowledge_handler import KnowledgeHandler

from vulcan.world_model.knowledge_handler import RetrievedKnowledge

logger = logging.getLogger(__name__)


@dataclass
class CreativeConstraints:
    """
    Constraints for creative generation to ensure accuracy.
    
    Industry Standard: Explicit constraint modeling with validation,
    clear documentation, and type safety.
    
    Attributes:
        must_be_accurate: Topics/facts that must be accurate (e.g., historical dates)
        can_be_artistic: Topics/elements where artistic freedom is allowed
        must_avoid: Common misconceptions or errors to avoid
        tone: Desired tone (serious, playful, educational, etc.)
        style_guidance: Specific style guidance (rhyme scheme, narrative structure, etc.)
    """
    
    must_be_accurate: List[str]
    can_be_artistic: List[str]
    must_avoid: List[str]
    tone: str
    style_guidance: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate creative constraints."""
        if not isinstance(self.must_be_accurate, list):
            raise TypeError("must_be_accurate must be a list")
        
        if not isinstance(self.can_be_artistic, list):
            raise TypeError("can_be_artistic must be a list")
        
        if not isinstance(self.must_avoid, list):
            raise TypeError("must_avoid must be a list")
        
        if not self.tone or not isinstance(self.tone, str):
            raise ValueError("tone must be a non-empty string")


@dataclass
class CreativeGuidance:
    """
    Complete guidance for creative generation.
    
    Industry Standard: Comprehensive data structure combining all inputs
    needed for creative generation with clear documentation.
    
    Attributes:
        format: Creative format (poem, story, song, essay)
        subject_knowledge: Retrieved knowledge about the subject
        constraints: Accuracy and style constraints
        structure: Structural requirements (stanzas, chapters, etc.)
        examples: Optional example outputs for style guidance
    """
    
    format: str
    subject_knowledge: RetrievedKnowledge
    constraints: CreativeConstraints
    structure: Dict[str, Any]
    examples: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate creative guidance."""
        if not self.format or not isinstance(self.format, str):
            raise ValueError("format must be a non-empty string")
        
        if not isinstance(self.subject_knowledge, RetrievedKnowledge):
            raise TypeError("subject_knowledge must be RetrievedKnowledge instance")
        
        if not isinstance(self.constraints, CreativeConstraints):
            raise TypeError("constraints must be CreativeConstraints instance")
        
        if not isinstance(self.structure, dict):
            raise TypeError("structure must be a dictionary")


class CreativeHandler:
    """
    Handles creative content generation with knowledge grounding.
    
    Industry Standard: Single Responsibility Principle - this class handles
    creative generation setup and verification, delegating knowledge retrieval
    to KnowledgeHandler.
    
    Architecture:
        Preparation → Generation (external) → Verification
        - Retrieve subject knowledge
        - Build accuracy constraints
        - Verify creative output against knowledge
        - Check for misconceptions and contradictions
    
    Integration:
        - Initialized with WorldModel and KnowledgeHandler
        - Used by WorldModel for CREATIVE request handling
        - Provides guidance for LLM generation
        - Verifies generated content
    """
    
    # Industry Standard: Class-level constants for domain-specific accuracy requirements
    ACCURACY_REQUIREMENTS = {
        'physics': [
            'laws of physics',
            'physical constants',
            'scientific principles',
            'measurements and units',
        ],
        'math': [
            'mathematical operations',
            'theorems and proofs',
            'numerical values',
            'equations and formulas',
        ],
        'history': [
            'dates and timelines',
            'historical figures',
            'major events',
            'geographical locations',
        ],
        'biology': [
            'anatomical facts',
            'biological processes',
            'species information',
            'scientific terminology',
        ],
        'chemistry': [
            'chemical formulas',
            'element properties',
            'reaction mechanisms',
            'safety information',
        ],
        'general': [
            'factual claims',
            'scientific consensus',
            'established facts',
        ],
    }
    
    # Industry Standard: Common misconceptions by domain to actively avoid
    COMMON_MISCONCEPTIONS = {
        'physics': [
            'heavier objects fall faster',
            'centrifugal force is real',
            'entropy means disorder',
            'glass is a liquid',
        ],
        'biology': [
            'evolution is just a theory',
            'humans only use 10% of their brains',
            'goldfish have 3-second memory',
            'lightning never strikes twice',
        ],
        'history': [
            'Vikings wore horned helmets',
            'Columbus proved Earth is round',
            'medieval people thought Earth was flat',
        ],
        'math': [
            '0.999... is not equal to 1',
            'division by zero is infinity',
            'probability can exceed 1',
        ],
        'general': [
            'correlation implies causation',
            'natural means safe',
            'theory means guess',
        ],
    }
    
    def __init__(
        self,
        world_model: 'WorldModel',
        knowledge_handler: 'KnowledgeHandler',
    ):
        """
        Initialize creative handler.
        
        Industry Standard: Dependency injection for testability and flexibility.
        
        Args:
            world_model: WorldModel instance for accessing shared resources
            knowledge_handler: KnowledgeHandler for knowledge retrieval
        """
        self.world_model = world_model
        self.knowledge_handler = knowledge_handler
    
    def prepare_creative_guidance(
        self,
        creative_format: str,
        subject: str,
        domain: str,
        original_query: str,
    ) -> CreativeGuidance:
        """
        Prepare complete guidance for creative generation.
        
        Industry Standard: Multi-step preparation with comprehensive logging,
        error handling, and graceful degradation.
        
        Args:
            creative_format: Format (poem, story, song, essay)
            subject: Subject matter for creative content
            domain: Knowledge domain
            original_query: Original user query for context
        
        Returns:
            CreativeGuidance with knowledge, constraints, and structure
        """
        logger.info(
            f"[CreativeHandler] Preparing creative guidance: "
            f"format={creative_format}, subject={subject}, domain={domain}"
        )
        
        # Step 1: Retrieve knowledge about subject
        try:
            subject_knowledge = self.knowledge_handler.retrieve_knowledge(
                domain=domain,
                topic=subject,
                query=original_query,
                max_results=15,
            )
            logger.info(
                f"[CreativeHandler] Retrieved {len(subject_knowledge.facts)} facts, "
                f"{len(subject_knowledge.equations)} equations"
            )
        except Exception as e:
            logger.error(f"[CreativeHandler] Knowledge retrieval failed: {e}")
            # Industry Standard: Graceful degradation with empty knowledge
            subject_knowledge = RetrievedKnowledge(
                facts=[],
                equations=[],
                definitions=[],
                sources=[],
                confidence=0.0,
                domain=domain,
                topic=subject,
            )
        
        # Step 2: Build constraints
        constraints = self._build_constraints(
            domain=domain,
            subject=subject,
            creative_format=creative_format,
        )
        
        # Step 3: Get format structure
        structure = self._get_format_structure(creative_format)
        
        # Step 4: Determine tone
        tone = self._determine_tone(creative_format, original_query)
        constraints.tone = tone
        
        logger.info(
            f"[CreativeHandler] Creative guidance prepared: "
            f"{len(constraints.must_be_accurate)} accuracy requirements, "
            f"{len(constraints.must_avoid)} misconceptions to avoid, tone={tone}"
        )
        
        return CreativeGuidance(
            format=creative_format,
            subject_knowledge=subject_knowledge,
            constraints=constraints,
            structure=structure,
            examples=[],
        )
    
    def verify_creative_output(
        self,
        output: str,
        guidance: CreativeGuidance,
    ) -> Dict[str, Any]:
        """
        Verify creative output for factual accuracy.
        
        Industry Standard: Multi-level verification with detailed feedback,
        identifying specific issues and providing actionable recommendations.
        
        Args:
            output: Generated creative content
            guidance: Original creative guidance with knowledge base
        
        Returns:
            Verification result with issues, conflicts, and confidence
        """
        logger.info("[CreativeHandler] Verifying creative output")
        
        issues = []
        conflicts = []
        warnings = []
        
        # Check 1: Look for common misconceptions
        for misconception in guidance.constraints.must_avoid:
            if self._contains_misconception(output, misconception):
                issues.append(
                    f"Contains misconception: {misconception}"
                )
                logger.warning(
                    f"[CreativeHandler] Detected misconception: {misconception}"
                )
        
        # Check 2: Verify facts against retrieved knowledge
        for fact in guidance.subject_knowledge.facts:
            if self._contradicts_fact(output, fact):
                conflicts.append(
                    f"Contradicts known fact: {fact}"
                )
                logger.warning(
                    f"[CreativeHandler] Detected contradiction: {fact}"
                )
        
        # Check 3: Check accuracy requirements are met
        accuracy_reqs = guidance.constraints.must_be_accurate
        for req in accuracy_reqs:
            # Heuristic: check if requirement topic is mentioned
            if req.lower() in output.lower():
                # Mentioned - need to verify accuracy
                # For now, flag as warning (could be enhanced with specific checks)
                warnings.append(
                    f"Verify accuracy of: {req}"
                )
        
        # Calculate verification confidence
        total_checks = (
            len(guidance.constraints.must_avoid) +
            len(guidance.subject_knowledge.facts) +
            len(accuracy_reqs)
        )
        
        if total_checks == 0:
            verification_confidence = 0.5  # No knowledge to verify against
        else:
            issues_found = len(issues) + len(conflicts)
            verification_confidence = max(0.0, 1.0 - (issues_found / total_checks))
        
        is_verified = len(issues) == 0 and len(conflicts) == 0
        
        logger.info(
            f"[CreativeHandler] Verification complete: "
            f"verified={is_verified}, {len(issues)} issues, "
            f"{len(conflicts)} conflicts, {len(warnings)} warnings, "
            f"confidence={verification_confidence:.2f}"
        )
        
        return {
            'passed': is_verified,  # Match key expected by calling code
            'is_verified': is_verified,  # Keep for backward compatibility
            'issues': issues,
            'conflicts': conflicts,
            'warnings': warnings,
            'verification_confidence': verification_confidence,
            'metadata': {
                'total_checks': total_checks,
                'misconception_checks': len(guidance.constraints.must_avoid),
                'fact_checks': len(guidance.subject_knowledge.facts),
            },
        }
    
    def _build_constraints(
        self,
        domain: str,
        subject: str,
        creative_format: str,
    ) -> CreativeConstraints:
        """
        Build accuracy and style constraints.
        
        Industry Standard: Domain-specific constraint generation with
        clear separation of accuracy requirements and artistic freedom.
        """
        # Get domain-specific accuracy requirements
        must_be_accurate = self.ACCURACY_REQUIREMENTS.get(
            domain,
            self.ACCURACY_REQUIREMENTS['general']
        ).copy()
        
        # Get domain-specific misconceptions to avoid
        must_avoid = self.COMMON_MISCONCEPTIONS.get(
            domain,
            self.COMMON_MISCONCEPTIONS['general']
        ).copy()
        
        # Define what can be artistic
        can_be_artistic = [
            'narrative structure',
            'character development',
            'dialogue and language',
            'metaphors and imagery',
            'emotional tone',
            'pacing and rhythm',
        ]
        
        # Format-specific style guidance
        style_guidance = {}
        if creative_format == 'poem':
            style_guidance['rhyme_scheme'] = 'optional'
            style_guidance['meter'] = 'flexible'
            style_guidance['length'] = '3-5 stanzas'
        elif creative_format == 'story':
            style_guidance['narrative_arc'] = 'beginning, middle, end'
            style_guidance['length'] = '500-1000 words'
        elif creative_format == 'song':
            style_guidance['structure'] = 'verse-chorus-verse'
            style_guidance['length'] = '2-3 verses, 1-2 choruses'
        elif creative_format == 'essay':
            style_guidance['structure'] = 'intro, body, conclusion'
            style_guidance['length'] = '800-1500 words'
        
        return CreativeConstraints(
            must_be_accurate=must_be_accurate,
            can_be_artistic=can_be_artistic,
            must_avoid=must_avoid,
            tone='neutral',  # Will be updated by _determine_tone
            style_guidance=style_guidance,
        )
    
    def _get_format_structure(self, creative_format: str) -> Dict[str, Any]:
        """
        Get structural requirements for creative format.
        
        Industry Standard: Explicit structure definition with clear expectations.
        """
        structures = {
            'poem': {
                'sections': ['stanza_1', 'stanza_2', 'stanza_3'],
                'lines_per_section': '4-6',
                'optional_elements': ['title', 'epigraph'],
            },
            'story': {
                'sections': ['opening', 'rising_action', 'climax', 'resolution'],
                'paragraphs': '8-15',
                'optional_elements': ['title', 'dialogue'],
            },
            'song': {
                'sections': ['verse_1', 'chorus', 'verse_2', 'chorus', 'bridge'],
                'lines_per_verse': '4-8',
                'lines_per_chorus': '4',
            },
            'essay': {
                'sections': ['introduction', 'body_paragraphs', 'conclusion'],
                'paragraphs': '5-8',
                'optional_elements': ['thesis_statement', 'topic_sentences'],
            },
        }
        
        return structures.get(creative_format, {'sections': [], 'flexible': True})
    
    def _determine_tone(self, creative_format: str, query: str) -> str:
        """
        Determine appropriate tone from format and query.
        
        Industry Standard: Heuristic tone detection with clear defaults.
        """
        query_lower = query.lower()
        
        # Check for explicit tone indicators
        if any(word in query_lower for word in ['serious', 'solemn', 'grave']):
            return 'serious'
        elif any(word in query_lower for word in ['funny', 'humorous', 'playful', 'silly']):
            return 'playful'
        elif any(word in query_lower for word in ['educational', 'informative', 'teach']):
            return 'educational'
        elif any(word in query_lower for word in ['inspiring', 'uplifting', 'motivational']):
            return 'inspiring'
        elif any(word in query_lower for word in ['dark', 'sad', 'melancholy']):
            return 'melancholic'
        
        # Format defaults
        format_defaults = {
            'poem': 'expressive',
            'story': 'engaging',
            'song': 'rhythmic',
            'essay': 'informative',
        }
        
        return format_defaults.get(creative_format, 'neutral')
    
    def _contains_misconception(self, output: str, misconception: str) -> bool:
        """
        Check if output contains a misconception.
        
        Industry Standard: Pattern-based detection with keyword matching.
        This is a heuristic - could be enhanced with semantic similarity.
        """
        output_lower = output.lower()
        misconception_lower = misconception.lower()
        
        # Extract key phrases from misconception
        keywords = misconception_lower.split()
        
        # Check if most keywords appear in output
        matches = sum(1 for kw in keywords if kw in output_lower)
        
        # Heuristic: if 60%+ of keywords match, likely contains misconception
        return matches >= len(keywords) * 0.6
    
    def _contradicts_fact(self, output: str, fact: str) -> bool:
        """
        Check if output contradicts a known fact.
        
        Industry Standard: Negation detection with keyword matching.
        This is a heuristic - could be enhanced with semantic contradiction detection.
        """
        output_lower = output.lower()
        fact_lower = fact.lower()
        
        # Extract keywords from fact
        fact_keywords = set(fact_lower.split()) - {
            'the', 'a', 'is', 'are', 'was', 'were', 'of', 'to', 'in', 'on', 'at'
        }
        
        # Check if fact keywords appear with negation
        negation_patterns = [
            'not', 'never', 'no', 'cannot', 'can\'t', "doesn't", "isn't",
            "aren't", "wasn't", "weren't", 'false', 'incorrect', 'wrong'
        ]
        
        # Look for negation near fact keywords
        for keyword in fact_keywords:
            if keyword in output_lower:
                # Find keyword position
                keyword_pos = output_lower.find(keyword)
                # Check surrounding context (50 chars before and after)
                context_start = max(0, keyword_pos - 50)
                context_end = min(len(output_lower), keyword_pos + len(keyword) + 50)
                context = output_lower[context_start:context_end]
                
                # Check for negation in context
                if any(neg in context for neg in negation_patterns):
                    return True
        
        return False
