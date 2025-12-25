# ============================================================
# VULCAN-AGI OpenAI Knowledge Distiller Module
# Capture layer for OpenAI knowledge distillation
# ============================================================
#
# ARCHITECTURAL NOTE:
# This component is responsible ONLY for capturing and storing external
# experience artifacts from OpenAI. It does NOT perform training directly.
#
# Training is delegated to Vulcan's existing systems:
#     - GovernedTrainer: Consensus-based weight updates
#     - SelfImprovingTraining: Meta-learning orchestrator
#     - train_llm_with_self_improvement.py: Full training loop
#
# The capture → train flow is:
#
#     main.py (inference)
#         └─ OpenAI response
#         └─ capture_response() → Distillation Store (JSONL)
#                                     ↓
#     [Async/Batched - via GovernedTrainer]
#         └─ GovernedTrainer reads from Distillation Store
#         └─ Proposes weight updates
#         └─ ConsensusEngine approves/rejects
#         └─ SelfImprovingTraining evaluates
#         └─ Promotion or rollback
#
# VERSION HISTORY:
#     1.0.0 - Extracted from main.py for modular architecture
# ============================================================

import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from vulcan.distillation.pii_redactor import PIIRedactor
from vulcan.distillation.governance_checker import GovernanceSensitivityChecker
from vulcan.distillation.quality_validator import ExampleQualityValidator
from vulcan.distillation.storage import DistillationStorageBackend

# Module metadata
__version__ = "1.0.0"
__author__ = "VULCAN-AGI Team"

logger = logging.getLogger(__name__)


class OpenAIKnowledgeDistiller:
    """
    Capture layer for OpenAI knowledge distillation.
    
    ARCHITECTURAL NOTE:
    This component is responsible ONLY for capturing and storing external
    experience artifacts from OpenAI. It does NOT perform training directly.
    
    Training is delegated to Vulcan's existing systems:
    - GovernedTrainer: Consensus-based weight updates
    - SelfImprovingTraining: Meta-learning orchestrator
    - train_llm_with_self_improvement.py: Full training loop
    
    Implements comprehensive capture safeguards:
    
    A) Capture Layer (Privacy & Consent)
       - Policy gate: only capture when training_opt_in=true (per-session)
       - PII redaction before storage
       - Secrets hard rejection
       - Governance sensitivity check
       - Full provenance tracking
    
    B) Quality Filtering (Garbage-in Prevention)
       - Non-trivial length and low boilerplate score
       - No refusal/safety boilerplate
       - Diversity sampling (no duplicate Q&As)
       - Domain-specific validators
    
    C) Storage Format
       - JSONL format for training system consumption
       - Optional encryption at rest
       - Provenance hashes for integrity
       - Retention limits with automatic cleanup
    
    DOES NOT:
    - Perform weight updates (GovernedTrainer does this)
    - Run evaluation (SelfImprovingTraining does this)
    - Promote/rollback weights (ConsensusEngine governs this)
    """
    
    # Configuration defaults
    DEFAULT_MAX_BUFFER_SIZE = 1000
    DEFAULT_RETENTION_DAYS = 30
    
    def __init__(
        self,
        local_llm: Optional[Any] = None,
        storage_path: str = "data/distillation_examples.json",
        max_buffer_size: int = DEFAULT_MAX_BUFFER_SIZE,
        retention_days: int = DEFAULT_RETENTION_DAYS,
        require_opt_in: bool = True,  # Privacy-first default
        enable_pii_redaction: bool = True,
        enable_governance_check: bool = True,
    ):
        """
        Initialize the capture layer for knowledge distillation.
        
        NOTE: This component only captures artifacts. Training is handled
        by Vulcan's GovernedTrainer and SelfImprovingTraining systems.
        
        Args:
            local_llm: Reference to Vulcan's local LLM (optional, for future use)
            storage_path: Path to store training examples persistently
            max_buffer_size: Maximum buffer size before flush to disk
            retention_days: Days to retain training examples before expiry
            require_opt_in: Require explicit opt-in for capture (default: True)
            enable_pii_redaction: Enable PII redaction (default: True)
            enable_governance_check: Check governance/CSIU before capture
        """
        self.local_llm = local_llm
        self.storage_path = Path(storage_path)
        self.max_buffer_size = max_buffer_size
        self.retention_days = retention_days
        self.require_opt_in = require_opt_in
        self.enable_pii_redaction = enable_pii_redaction
        self.enable_governance_check = enable_governance_check
        
        self.logger = logging.getLogger("OpenAIKnowledgeDistiller")
        
        # Initialize capture components
        self.pii_redactor = PIIRedactor()
        self.quality_validator = ExampleQualityValidator()
        self.governance_checker = GovernanceSensitivityChecker()
        
        # Initialize storage backend (JSONL with optional encryption)
        # Training systems (GovernedTrainer, SelfImprovingTraining) read from here
        storage_dir = str(self.storage_path.parent / "distillation")
        self.storage_backend = DistillationStorageBackend(
            storage_path=storage_dir,
            use_encryption=os.getenv("DISTILLATION_ENCRYPT", "false").lower() == "true",
            encryption_key=os.getenv("DISTILLATION_ENCRYPTION_KEY"),
        )
        
        # Thread-safe buffer for capture (flushed to storage periodically)
        self._buffer_lock = threading.Lock()
        self._capture_buffer: List[Dict[str, Any]] = []
        
        # Capture statistics (training stats live in GovernedTrainer)
        self.stats = {
            "examples_captured": 0,
            "examples_rejected": 0,
            "rejection_reasons": {},
            "pii_redactions": 0,
            "secrets_detected": 0,
            "governance_sensitive_rejections": 0,
            "average_quality_score": 0.0,
            "opt_in_required_skips": 0,
            "buffer_flushes": 0,
        }
        
        # Audit log for governance
        self._audit_log: List[Dict[str, Any]] = []
        self._max_audit_entries = 10000
        
        # Ensure storage directory exists
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing state if available
        self._load_state()
        
        self.logger.info(
            f"OpenAI Knowledge Distiller (Capture Layer) initialized. "
            f"Opt-in required: {require_opt_in}, PII redaction: {enable_pii_redaction}, "
            f"Governance check: {enable_governance_check}. "
            f"Training delegated to GovernedTrainer/SelfImprovingTraining."
        )
    
    def _load_state(self):
        """Load existing capture state from storage."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r") as f:
                    data = json.load(f)
                    self._capture_buffer = data.get("capture_buffer", [])
                    self.stats = {**self.stats, **data.get("stats", {})}
                    
                    self.logger.info(
                        f"Loaded {len(self._capture_buffer)} pending capture examples"
                    )
                    
                    # Clean expired examples
                    self._clean_expired_examples()
                    
            except Exception as e:
                self.logger.warning(f"Failed to load existing state: {e}")
    
    def _save_state(self):
        """Persist capture state to storage."""
        try:
            with open(self.storage_path, "w") as f:
                json.dump({
                    "capture_buffer": self._capture_buffer,
                    "stats": self.stats,
                    "last_save": time.time(),
                }, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
    
    def _clean_expired_examples(self):
        """Remove examples past retention period."""
        if self.retention_days <= 0:
            return
            
        expiry_threshold = time.time() - (self.retention_days * 86400)
        
        with self._buffer_lock:
            original_count = len(self._capture_buffer)
            self._capture_buffer = [
                ex for ex in self._capture_buffer
                if ex.get("timestamp", 0) > expiry_threshold
            ]
            removed = original_count - len(self._capture_buffer)
            
            if removed > 0:
                self.logger.info(f"Cleaned {removed} expired capture examples")
    
    def _log_audit(self, action: str, details: Dict[str, Any]):
        """Log action for governance audit trail."""
        entry = {
            "timestamp": time.time(),
            "action": action,
            "details": details,
        }
        
        self._audit_log.append(entry)
        
        # Limit audit log size
        if len(self._audit_log) > self._max_audit_entries:
            self._audit_log = self._audit_log[-self._max_audit_entries:]
    
    def _flush_to_storage(self):
        """
        Flush capture buffer to JSONL storage for training system consumption.
        
        The stored examples are consumed by:
        - GovernedTrainer: For consensus-based weight updates
        - SelfImprovingTraining: For meta-learning orchestration
        """
        with self._buffer_lock:
            if not self._capture_buffer:
                return 0
            
            examples_to_flush = self._capture_buffer.copy()
            self._capture_buffer.clear()
        
        flushed = 0
        for example in examples_to_flush:
            if self.storage_backend.append_example(example):
                flushed += 1
        
        self.stats["buffer_flushes"] += 1
        self._save_state()
        
        self.logger.info(
            f"Flushed {flushed} examples to distillation store "
            f"(available for GovernedTrainer)"
        )
        
        return flushed
    
    def capture_response(
        self,
        prompt: str,
        openai_response: str,
        local_response: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        session_opted_in: bool = False,
        teacher_model: str = "gpt-3.5-turbo",
    ) -> bool:
        """
        Capture an OpenAI response as a potential training example.
        
        Implements the capture layer with full safeguards:
        - Opt-in policy gate (per-session, NOT global)
        - Secrets/credentials hard rejection
        - Governance sensitivity check
        - PII redaction before storage
        - Quality validation with hard reject thresholds
        - Dedupe and diversity sampling
        - Full provenance tracking
        
        Args:
            prompt: The input prompt
            openai_response: The response from OpenAI
            local_response: Optional response from local LLM for comparison
            metadata: Additional metadata (routing, tools, context)
            session_opted_in: Whether session has opted into training capture
            teacher_model: The OpenAI model that generated the response
            
        Returns:
            True if the example was captured, False if rejected
        """
        metadata = metadata or {}
        
        # ================================================================
        # GATE 1: Per-session opt-in requirement (NOT a global flag)
        # ================================================================
        if self.require_opt_in and not session_opted_in:
            self.stats["opt_in_required_skips"] += 1
            self.logger.debug("Capture skipped: session not opted in")
            return False
        
        # ================================================================
        # GATE 2: Secrets/credentials HARD REJECTION (never capture)
        # ================================================================
        if self.pii_redactor.contains_secrets(prompt) or self.pii_redactor.contains_secrets(openai_response):
            self.stats["secrets_detected"] += 1
            self.stats["examples_rejected"] += 1
            self._log_audit("capture_rejected", {
                "reason": "contains_secrets",
                "prompt_preview": prompt[:50] + "...",
            })
            self.logger.warning("Capture rejected: contains secrets/credentials")
            return False
        
        # ================================================================
        # GATE 3: Governance sensitivity check
        # ================================================================
        if self.enable_governance_check:
            is_sensitive, category, reasons = self.governance_checker.check_sensitivity(
                prompt, openai_response, metadata
            )
            if is_sensitive:
                self.stats["governance_sensitive_rejections"] += 1
                self.stats["examples_rejected"] += 1
                self._log_audit("capture_rejected", {
                    "reason": "governance_sensitive",
                    "category": category,
                    "details": reasons,
                })
                self.logger.debug(f"Capture rejected: governance sensitive ({category})")
                return False
        
        # ================================================================
        # STEP 4: PII Redaction (scrub before storage)
        # ================================================================
        redacted_prompt = prompt
        redacted_response = openai_response
        pii_stats = {}
        
        if self.enable_pii_redaction:
            redacted_prompt, prompt_pii = self.pii_redactor.redact(prompt)
            redacted_response, response_pii = self.pii_redactor.redact(openai_response)
            pii_stats = {**prompt_pii, **response_pii}
            
            if pii_stats:
                self.stats["pii_redactions"] += sum(pii_stats.values())
                self.logger.debug(f"PII redacted: {pii_stats}")
        
        # ================================================================
        # GATE 5: Quality validation with hard reject thresholds
        # ================================================================
        passed, quality_score, rejection_reasons = self.quality_validator.validate(
            redacted_prompt, redacted_response, local_response
        )
        
        if not passed:
            self.stats["examples_rejected"] += 1
            for reason in rejection_reasons:
                reason_key = reason.split(":")[0]  # Remove numeric suffix
                self.stats["rejection_reasons"][reason_key] = (
                    self.stats["rejection_reasons"].get(reason_key, 0) + 1
                )
            self._log_audit("capture_rejected", {
                "reason": "quality_validation_failed",
                "quality_score": quality_score,
                "rejection_reasons": rejection_reasons,
            })
            self.logger.debug(f"Example rejected: {rejection_reasons}")
            return False
        
        # ================================================================
        # CREATE: Structured example with full provenance
        # ================================================================
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        response_hash = hashlib.sha256(openai_response.encode()).hexdigest()
        
        example = {
            # Core content (redacted) - JSONL compatible schema
            "instruction": redacted_prompt,
            "teacher_answer": redacted_response,
            "context": {
                "routing_metadata": metadata.get("routing", {}),
                "tools_used": metadata.get("tools", []),
                "systems_used": metadata.get("systems_used", []),
            },
            "labels": {
                "domain": self._detect_domain(redacted_prompt),
                "quality_score": quality_score,
                "validation_passed": True,
            },
            
            # Provenance (hashes for deduplication and integrity)
            "prompt_hash": prompt_hash,
            "response_hash": response_hash,
            "teacher_model": teacher_model,
            "timestamp": time.time(),
            
            # Governance
            "session_opted_in": session_opted_in,
            "retention_expires": time.time() + (self.retention_days * 86400),
            "pii_redacted": bool(pii_stats),
        }
        
        # Store example in capture buffer
        with self._buffer_lock:
            self._capture_buffer.append(example)
            self.stats["examples_captured"] += 1
            
            # Update running average quality
            total = self.stats["examples_captured"]
            avg = self.stats["average_quality_score"]
            self.stats["average_quality_score"] = (
                (avg * (total - 1) + quality_score) / total
            )
        
        # Log audit entry
        self._log_audit("capture", {
            "prompt_hash": prompt_hash[:16],
            "response_hash": response_hash[:16],
            "quality_score": quality_score,
            "pii_redacted": bool(pii_stats),
        })
        
        self.logger.debug(
            f"Captured example (quality: {quality_score:.2f}, "
            f"buffer: {len(self._capture_buffer)})"
        )
        
        # Flush to storage when buffer is full
        # (Training systems will consume from storage asynchronously)
        if len(self._capture_buffer) >= self.max_buffer_size:
            self._flush_to_storage()
        
        return True
    
    def _detect_domain(self, prompt: str) -> str:
        """Detect the domain of a prompt for labeling."""
        prompt_lower = prompt.lower()
        
        if any(kw in prompt_lower for kw in ["code", "function", "python", "javascript", "program"]):
            return "code"
        elif any(kw in prompt_lower for kw in ["calculate", "math", "equation", "number"]):
            return "math"
        elif any(kw in prompt_lower for kw in ["explain", "what is", "how does", "why"]):
            return "explanation"
        elif any(kw in prompt_lower for kw in ["write", "create", "compose", "draft"]):
            return "creative"
        else:
            return "general"
    
    def flush(self) -> int:
        """
        Manually flush capture buffer to storage.
        
        Called when you want to make captured examples immediately
        available to the training systems (GovernedTrainer, SelfImprovingTraining).
        
        Returns:
            Number of examples flushed
        """
        return self._flush_to_storage()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the capture layer."""
        return {
            "enabled": True,
            "config": {
                "require_opt_in": self.require_opt_in,
                "pii_redaction": self.enable_pii_redaction,
                "governance_check": self.enable_governance_check,
                "retention_days": self.retention_days,
                "max_buffer_size": self.max_buffer_size,
            },
            "state": {
                "buffer_size": len(self._capture_buffer),
                "storage_stats": self.storage_backend.get_stats(),
            },
            "stats": self.stats,
            "note": "Training delegated to GovernedTrainer/SelfImprovingTraining",
        }
    
    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent audit log entries."""
        return self._audit_log[-limit:]
    
    def clear_buffer(self) -> int:
        """Clear capture buffer without flushing to storage."""
        with self._buffer_lock:
            count = len(self._capture_buffer)
            self._capture_buffer.clear()
            self._save_state()
            self._log_audit("clear_buffer", {"examples_cleared": count})
            return count
    
    def set_opt_in(self, session_id: str, opted_in: bool):
        """Set opt-in status for a session (for external tracking)."""
        self._log_audit("opt_in_change", {
            "session_id": session_id[:16] if session_id else "unknown",
            "opted_in": opted_in,
        })


__all__ = ["OpenAIKnowledgeDistiller"]
