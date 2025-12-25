# ============================================================
# VULCAN-AGI Distillation Models Module
# Data models for knowledge distillation
# ============================================================

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Module metadata
__version__ = "1.0.0"
__author__ = "VULCAN-AGI Team"

logger = logging.getLogger(__name__)


@dataclass
class DistillationExample:
    """
    Structured training example with full provenance tracking.
    
    Follows the recommended format:
    - instruction: sanitized prompt
    - context: routing outputs / tools / memory snippets
    - teacher_answer: OpenAI response
    - labels: domain, difficulty, success/failure signals
    
    Attributes:
        instruction: Sanitized prompt (PII redacted)
        teacher_answer: OpenAI response
        context: Routing metadata, tools used
        labels: Domain, difficulty, validation results
        prompt_hash: SHA256 of original prompt
        response_hash: SHA256 of response
        teacher_model: Model that generated the response (e.g., "gpt-3.5-turbo")
        timestamp: Unix timestamp of capture
        quality_score: Quality score from validator
        validation_passed: Whether example passed validation
        rejection_reasons: List of rejection reasons if any
        session_opted_in: Whether session opted into training
        retention_expires: Unix timestamp for data expiry
    """
    instruction: str  # Sanitized prompt (PII redacted)
    teacher_answer: str  # OpenAI response
    context: Dict[str, Any] = field(default_factory=dict)  # Routing metadata, tools used
    labels: Dict[str, Any] = field(default_factory=dict)  # Domain, difficulty, validation results
    
    # Provenance tracking
    prompt_hash: str = ""  # SHA256 of original prompt
    response_hash: str = ""  # SHA256 of response
    teacher_model: str = "gpt-3.5-turbo"  # e.g., "gpt-3.5-turbo"
    timestamp: float = 0.0
    
    # Quality metrics
    quality_score: float = 0.0
    validation_passed: bool = False
    rejection_reasons: List[str] = field(default_factory=list)
    
    # Governance
    session_opted_in: bool = False
    retention_expires: Optional[float] = None  # Unix timestamp for data expiry
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "instruction": self.instruction,
            "teacher_answer": self.teacher_answer,
            "context": self.context,
            "labels": self.labels,
            "prompt_hash": self.prompt_hash,
            "response_hash": self.response_hash,
            "teacher_model": self.teacher_model,
            "timestamp": self.timestamp,
            "quality_score": self.quality_score,
            "validation_passed": self.validation_passed,
            "rejection_reasons": self.rejection_reasons,
            "session_opted_in": self.session_opted_in,
            "retention_expires": self.retention_expires,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DistillationExample":
        """Create from dictionary."""
        return cls(
            instruction=data.get("instruction", ""),
            teacher_answer=data.get("teacher_answer", ""),
            context=data.get("context", {}),
            labels=data.get("labels", {}),
            prompt_hash=data.get("prompt_hash", ""),
            response_hash=data.get("response_hash", ""),
            teacher_model=data.get("teacher_model", "gpt-3.5-turbo"),
            timestamp=data.get("timestamp", 0.0),
            quality_score=data.get("quality_score", 0.0),
            validation_passed=data.get("validation_passed", False),
            rejection_reasons=data.get("rejection_reasons", []),
            session_opted_in=data.get("session_opted_in", False),
            retention_expires=data.get("retention_expires"),
        )


__all__ = ["DistillationExample"]
