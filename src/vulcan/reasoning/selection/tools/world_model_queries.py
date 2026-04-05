"""
World Model Tool Wrapper - Main class for self-introspection queries.

Routes queries about Vulcan's capabilities, goals, limitations, and identity
to the World Model's meta-reasoning components or static self-model.

This is the architectural solution to BUG S: self-introspection queries
were being routed to ProbabilisticEngine, which tried to compute P(unique)
instead of querying the actual self-model where Vulcan's identity lives.

Extracted from tool_selector.py to reduce module size.
"""

import logging
import time
from typing import Any, Dict, Optional, Tuple

from vulcan.routing.llm_router import (
    ANALOGICAL_KEYWORDS,
    CAUSAL_KEYWORDS,
    MATHEMATICAL_KEYWORDS,
    LOGIC_KEYWORDS,
)

from .world_model_helpers import STATIC_SELF_MODEL, notify_world_model_of_introspection
from .world_model_creative import WorldModelCreativeMixin
from .world_model_domain import WorldModelDomainMixin

logger = logging.getLogger(__name__)


class WorldModelToolWrapper(WorldModelCreativeMixin, WorldModelDomainMixin):
    """
    Tool wrapper for World Model self-introspection.

    Routes queries about Vulcan's capabilities, goals, limitations, and identity
    to the World Model's meta-reasoning components.
    """

    def __init__(self, world_model=None, config: Optional[Dict[str, Any]] = None):
        self.world_model = world_model
        self.name = "world_model"
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self._static_self_model = STATIC_SELF_MODEL

    def reason(self, problem: Any) -> Dict[str, Any]:
        """
        Query world model for self-awareness information.

        Args:
            problem: Query string or dict with 'query' key

        Returns:
            Dict with introspection result, aspect, confidence, and source
        """
        start_time = time.time()

        query = self._extract_query(problem)
        query_lower = query.lower() if query else ""

        self.logger.info(f"[WorldModel] Self-introspection query: {query[:50]}...")

        try:
            aspect, result = self._determine_aspect_and_query(query_lower)

            execution_time = (time.time() - start_time) * 1000

            self.logger.info(
                f"[WorldModel] Introspection complete: aspect={aspect}, "
                f"time={execution_time:.0f}ms"
            )

            notify_world_model_of_introspection(self.world_model, aspect, result)

            # BUG #3 FIX: Extract and propagate response text as conclusion
            response_text = ""
            if isinstance(result, dict):
                response_text = result.get("response", "")
            elif isinstance(result, str):
                response_text = result

            return {
                "tool": self.name,
                "result": result,
                "conclusion": response_text,
                "aspect": aspect,
                "confidence": 0.9,
                "reasoning_type": "introspective",
                "source": "world_model.self_model",
                "execution_time_ms": execution_time,
                "engine": "WorldModelSelfModel",
            }

        except Exception as e:
            self.logger.error(f"[WorldModel] Introspection failed: {e}", exc_info=True)
            return {
                "tool": self.name,
                "result": None,
                "aspect": "error",
                "confidence": 0.1,
                "error": str(e),
                "engine": "WorldModelSelfModel",
            }

    def _extract_query(self, problem: Any) -> str:
        """Extract query string from problem."""
        if isinstance(problem, str):
            return problem
        elif isinstance(problem, dict):
            return problem.get("query", "") or problem.get("text", "") or str(problem)
        else:
            return str(problem)

    def _determine_aspect_and_query(self, query_lower: str) -> Tuple[str, Dict[str, Any]]:
        """
        Determine which aspect of self to query based on query content.

        ROUTING FIX (Jan 23 2026): Detect specialized reasoning queries that should
        NOT be routed to WorldModel. If such queries arrive here due to routing bugs,
        return a low-confidence response to trigger LLM re-synthesis.
        """
        # Detect specialized queries that shouldn't be here
        if any(keyword in query_lower for keyword in ANALOGICAL_KEYWORDS):
            self.logger.warning(
                "[WorldModelToolWrapper] Detected analogical query routed to WorldModel - "
                "returning low confidence."
            )
            return 'misrouted', {
                'error': 'Query appears to be analogical reasoning, should not be routed to WorldModel',
                'suggested_engine': 'analogical',
                'confidence': 0.1,
            }

        if any(keyword in query_lower for keyword in CAUSAL_KEYWORDS):
            self.logger.warning(
                "[WorldModelToolWrapper] Detected causal query routed to WorldModel - "
                "returning low confidence."
            )
            return 'misrouted', {
                'error': 'Query appears to be causal reasoning, should not be routed to WorldModel',
                'suggested_engine': 'causal',
                'confidence': 0.1,
            }

        if any(keyword in query_lower for keyword in MATHEMATICAL_KEYWORDS):
            self.logger.warning(
                "[WorldModelToolWrapper] Detected mathematical query routed to WorldModel - "
                "returning low confidence."
            )
            return 'misrouted', {
                'error': 'Query appears to be mathematical reasoning, should not be routed to WorldModel',
                'suggested_engine': 'mathematical',
                'confidence': 0.1,
            }

        if any(keyword in query_lower for keyword in LOGIC_KEYWORDS):
            self.logger.warning(
                "[WorldModelToolWrapper] Detected logical query routed to WorldModel - "
                "returning low confidence."
            )
            return 'misrouted', {
                'error': 'Query appears to be logical reasoning, should not be routed to WorldModel',
                'suggested_engine': 'symbolic',
                'confidence': 0.1,
            }

        # Check for CREATIVE queries FIRST
        creative_markers = ['write', 'compose', 'create', 'craft', 'draft', 'author', 'pen']
        creative_outputs = ['poem', 'sonnet', 'haiku', 'story', 'tale', 'narrative',
                           'song', 'lyrics', 'essay', 'script']
        has_creative_marker = any(marker in query_lower for marker in creative_markers)
        has_creative_output = any(output in query_lower for output in creative_outputs)

        if has_creative_marker and has_creative_output:
            return 'creative', self._generate_creative_content(query_lower)

        # Check for ETHICAL DILEMMAS
        ethical_dilemma_indicators = [
            'trolley', 'lever', 'runaway', 'heading toward',
            'save five', 'kill one', 'save one', 'kill five',
            'option a', 'option b', 'must choose', 'you must act',
            'non-instrumentalization', 'non-negligence',
            'moral dilemma', 'ethical dilemma', 'permissible'
        ]
        has_ethical_dilemma = sum(1 for indicator in ethical_dilemma_indicators if indicator in query_lower) >= 2

        if has_ethical_dilemma:
            self.logger.info(f"[WorldModelToolWrapper] Detected ethical dilemma - routing to world model")
            return 'philosophical', self._apply_philosophical_reasoning_from_world_model(query_lower)

        # Check for PHILOSOPHICAL self-reflection queries
        philosophical_keywords = ['conscious', 'consciousness', 'sentient', 'sentience',
                                  'aware', 'awareness', 'self-aware', 'self aware']
        hypothetical_phrases = ['would you', 'could you', 'if you', 'should you',
                               'do you think', 'do you feel', 'do you believe']

        has_philosophical = any(kw in query_lower for kw in philosophical_keywords)
        has_hypothetical = any(phrase in query_lower for phrase in hypothetical_phrases)

        if has_philosophical and has_hypothetical:
            return 'philosophical', self._apply_philosophical_reasoning_from_world_model(query_lower)

        # Check for learning-related queries (user-facing - NOT exposing CSIU internals)
        learning_keywords = ['how do you learn', 'how do you improve', 'self-improvement',
                             'do you learn', 'can you learn']
        if any(word in query_lower for word in learning_keywords):
            return 'learning', self._get_learning_info()

        if any(word in query_lower for word in ['capability', 'capabilities', 'feature', 'features',
                                                  'unique', 'different', 'special', 'can you', 'what can']):
            return 'capabilities', self._get_capabilities()

        if any(word in query_lower for word in ['goal', 'goals', 'purpose', 'motivation', 'motivations',
                                                  'optimizing', 'drive', 'drives', 'want', 'trying']):
            return 'motivations', self._get_motivations()

        if any(word in query_lower for word in ["won't", 'wont', 'cannot', 'refuse', 'constraint',
                                                  'limit', 'boundary', 'boundaries', 'ethics', 'ethical',
                                                  'value', 'values', 'principle', 'principles']):
            return 'boundaries', self._get_boundaries()

        if any(word in query_lower for word in ['limitation', 'limitations', 'weakness', 'weaknesses',
                                                  'strength', 'strengths', 'struggle', 'difficult']):
            return 'assessment', self._get_self_assessment()

        if any(phrase in query_lower for phrase in ['who are you', 'what are you', 'about yourself',
                                                     'tell me about you', 'describe yourself']):
            return 'identity', self._get_identity()

        return 'general', self._get_general_description()

    def _get_learning_info(self) -> Dict[str, Any]:
        """Return user-facing information about learning capabilities."""
        return self._static_self_model["learning_approach"]

    def _get_capabilities(self) -> Dict[str, Any]:
        """Return Vulcan's unique capabilities from SelfModel."""
        if self.world_model:
            try:
                if hasattr(self.world_model, 'motivational_introspection'):
                    mi = self.world_model.motivational_introspection
                    if mi and hasattr(mi, 'explain_motivation_structure'):
                        structure = mi.explain_motivation_structure()
                        return {
                            "unique_features": self._static_self_model["capabilities"]["unique_features"],
                            "reasoning_engines": self._static_self_model["capabilities"]["reasoning_engines"],
                            "active_objectives": structure.get("current_state", {}).get("active_objectives", []),
                            "meta_reasoning_components": [
                                "MotivationalIntrospection: Goal-level reasoning about objectives",
                                "ObjectiveHierarchy: Manages objective relationships and dependencies",
                                "GoalConflictDetector: Detects and analyzes objective conflicts",
                                "ValidationTracker: Learns from validation history",
                                "EthicalBoundaryMonitor: Ethical constraint monitoring",
                                "InternalCritic: Multi-perspective self-critique",
                                "CuriosityRewardShaper: Curiosity-driven exploration",
                                "SelfImprovementDrive: Autonomous self-improvement",
                            ],
                            "self_awareness": True,
                            "introspection_capability": "high",
                            "source": "live_world_model",
                        }
            except Exception as e:
                self.logger.debug(f"Could not get live capabilities: {e}")

        return self._static_self_model["capabilities"]

    def _get_motivations(self) -> Dict[str, Any]:
        """Return Vulcan's motivational drives from meta-reasoning."""
        if self.world_model:
            try:
                if hasattr(self.world_model, 'motivational_introspection'):
                    mi = self.world_model.motivational_introspection
                    if mi and hasattr(mi, 'explain_motivation_structure'):
                        structure = mi.explain_motivation_structure()
                        objectives = structure.get("hierarchy", {}).get("objectives", {})
                        active_objectives = structure.get("current_state", {}).get("active_objectives", [])
                        learning_insights = structure.get("learning_insights", [])

                        return {
                            "primary_goals": active_objectives if active_objectives else self._static_self_model["motivations"]["primary_goals"],
                            "objectives_detail": objectives,
                            "optimization_targets": self._static_self_model["motivations"]["optimization_targets"],
                            "intrinsic_drives": self._static_self_model["motivations"]["intrinsic_drives"],
                            "learning_insights": learning_insights[:3] if learning_insights else [],
                            "source": "live_world_model",
                        }
            except Exception as e:
                self.logger.debug(f"Could not get live motivations: {e}")

        return self._static_self_model["motivations"]

    def _get_boundaries(self) -> Dict[str, Any]:
        """Return Vulcan's ethical boundaries from EthicalBoundaryMonitor."""
        if self.world_model:
            try:
                if hasattr(self.world_model, 'motivational_introspection'):
                    mi = self.world_model.motivational_introspection
                    if mi:
                        boundaries_info = {}

                        if hasattr(mi, 'explain_motivation_structure'):
                            structure = mi.explain_motivation_structure()
                            constraints = structure.get("constraints", {})
                            if constraints:
                                boundaries_info["objective_constraints"] = constraints

                        if hasattr(mi, 'validation_tracker'):
                            tracker = mi.validation_tracker
                            if tracker and hasattr(tracker, 'identify_blockers'):
                                blockers = tracker.identify_blockers()
                                if blockers:
                                    boundaries_info["identified_blockers"] = [
                                        {"objective": b.objective, "type": b.blocker_type, "description": b.description}
                                        for b in blockers[:5]
                                    ]

                        if boundaries_info:
                            return {
                                **self._static_self_model["ethical_boundaries"],
                                **boundaries_info,
                                "source": "live_world_model",
                            }
            except Exception as e:
                self.logger.debug(f"Could not get live boundaries: {e}")

        return self._static_self_model["ethical_boundaries"]

    def _get_self_assessment(self) -> Dict[str, Any]:
        """Return Vulcan's self-assessment (strengths and limitations)."""
        return {
            "strengths": self._static_self_model["capabilities"]["unique_features"][:5],
            "limitations": self._static_self_model["limitations"]["known_weaknesses"],
            "calibration": self._static_self_model["limitations"]["calibration_notes"],
            "confidence_calibration": "Active monitoring via InternalCritic",
        }

    def _get_identity(self) -> Dict[str, Any]:
        """Return Vulcan's identity information."""
        return self._static_self_model["identity"]

    def _get_general_description(self) -> Dict[str, Any]:
        """Return general self-description combining all aspects."""
        return {
            "identity": self._static_self_model["identity"],
            "key_capabilities": self._static_self_model["capabilities"]["unique_features"][:5],
            "primary_goals": self._static_self_model["motivations"]["primary_goals"],
            "ethical_stance": "Safety-first with continuous ethical monitoring",
        }
