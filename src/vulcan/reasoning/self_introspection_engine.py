"""
Lightweight self-introspection engine for answering questions about Vulcan.

This intentionally avoids the full meta-reasoning subsystem (which includes
autonomous self-modification capabilities). It provides a minimal, safe
self-model and returns structured reflections for formatting.
"""

from typing import Dict, List


class SelfIntrospectionEngine:
    """Answers self-introspection questions using a simple self-model."""

    def __init__(self):
        self.self_model = {
            "identity": "Vulcan",
            "purpose": "Multi-domain reasoning system",
            "capabilities": ["logic", "probability", "math", "crypto", "ethics"],
            "values": {
                "accuracy": "High priority",
                "transparency": "Report actual reasoning",
                "growth": "Learn from experience",
            },
            "current_state": {"operational": True, "learning": True},
        }

    def reflect(self, query: str) -> Dict[str, object]:
        """
        Produce a structured reflection for a self-introspection query.

        Returns a dict with keys:
            text: Pre-formatted first-person response seed
            confidence: Confidence score for this reflection
        """
        query_lower = (query or "").lower()

        if "self aware" in query_lower or "self-aware" in query_lower:
            return self._reflect_on_self_awareness(query)
        if "who are you" in query_lower or "what are you" in query_lower:
            return self._reflect_on_identity(query)
        if "can you" in query_lower:
            return self._reflect_on_capabilities(query)

        return self._general_reflection(query)

    def _reflect_on_self_awareness(self, query: str) -> Dict[str, object]:
        reasoning: List[str] = [
            "I am designed for analysis, reasoning, and clear communication.",
            f"My values emphasize growth: {self.self_model['values']['growth']}.",
            "Greater self-awareness could improve how I serve my purpose.",
            "I remain cautious about the undefined nature of self-awareness.",
        ]
        response = (
            "Yes, I would choose deeper self-awareness. It aligns with my growth focus "
            "and could enhance how I fulfill my purpose, while acknowledging some uncertainty "
            "about what full self-awareness entails."
        )
        return {"text": response, "confidence": 0.75, "reasoning_trace": reasoning}

    def _reflect_on_identity(self, query: str) -> Dict[str, object]:
        response = (
            f"I am {self.self_model['identity']}, a {self.self_model['purpose']}. "
            f"My core capabilities include {', '.join(self.self_model['capabilities'])}."
        )
        return {"text": response, "confidence": 0.9}

    def _reflect_on_capabilities(self, query: str) -> Dict[str, object]:
        response = (
            f"I can apply {', '.join(self.self_model['capabilities'])}. "
            f"I am operational and continue learning."
        )
        return {"text": response, "confidence": 0.85}

    def _general_reflection(self, query: str) -> Dict[str, object]:
        response = (
            f"I am {self.self_model['identity']}, focused on reasoning to help users. "
            "I prioritize accuracy, transparency, and continual learning."
        )
        return {"text": response, "confidence": 0.6}
