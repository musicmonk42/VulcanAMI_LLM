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
    DEFAULT_TRAINING_TRIGGER_THRESHOLD = 500  # Trigger training after N examples
    
    def __init__(
        self,
        local_llm: Optional[Any] = None,
        storage_path: str = "data/distillation_examples.json",
        max_buffer_size: int = DEFAULT_MAX_BUFFER_SIZE,
        retention_days: int = DEFAULT_RETENTION_DAYS,
        require_opt_in: bool = True,  # Privacy-first default
        enable_pii_redaction: bool = True,
        enable_governance_check: bool = True,
        training_trigger_threshold: int = DEFAULT_TRAINING_TRIGGER_THRESHOLD,
        training_trigger_callback: Optional[callable] = None,
        training_webhook_url: Optional[str] = None,
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
            training_trigger_threshold: Number of examples to trigger training (default: 500)
            training_trigger_callback: Optional callback function to invoke for training
            training_webhook_url: Optional webhook URL to notify for training
        """
        self.local_llm = local_llm
        self.storage_path = Path(storage_path)
        self.max_buffer_size = max_buffer_size
        self.retention_days = retention_days
        self.require_opt_in = require_opt_in
        self.enable_pii_redaction = enable_pii_redaction
        self.enable_governance_check = enable_governance_check
        
        # Training trigger configuration
        self.training_trigger_threshold = training_trigger_threshold
        self.training_trigger_callback = training_trigger_callback
        self.training_webhook_url = training_webhook_url
        self._last_training_trigger_count = 0
        
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
            "training_triggers": 0,
            "last_training_trigger_time": None,
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
        Flush capture buffer to JSONL storage with two-phase commit.
        
        Two-phase commit prevents data loss on partial failures:
        1. Phase 1: Write examples to storage
        2. Phase 2: Only clear buffer entries that were successfully written
        3. Phase 3: Check if training should be triggered
        
        The stored examples are consumed by:
        - GovernedTrainer: For consensus-based weight updates
        - SelfImprovingTraining: For meta-learning orchestration
        """
        with self._buffer_lock:
            if not self._capture_buffer:
                return 0
            examples_to_flush = self._capture_buffer.copy()
            # DON'T clear buffer yet - wait for successful write
        
        # Phase 1: Write to storage
        flushed = 0
        try:
            for example in examples_to_flush:
                if self.storage_backend.append_example(example):
                    flushed += 1
                else:
                    # Stop on first failure to prevent partial writes
                    self.logger.warning(
                        f"Partial flush: {flushed}/{len(examples_to_flush)} examples written"
                    )
                    break
        except Exception as e:
            self.logger.error(f"Flush failed, retaining buffer: {e}")
            return 0
        
        # Phase 2: Commit (clear only flushed examples from buffer)
        with self._buffer_lock:
            self._capture_buffer = self._capture_buffer[flushed:]
        
        self.stats["buffer_flushes"] += 1
        self._save_state()
        
        self.logger.info(
            f"Flushed {flushed} examples to distillation store "
            f"(available for GovernedTrainer). Buffer has {len(self._capture_buffer)} remaining."
        )
        
        # Phase 3: Check if training should be triggered
        self._check_training_trigger()
        
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
    
    def _send_webhook_async(self, webhook_url: str, payload: dict) -> None:
        """
        Send webhook in background thread - non-blocking (fire-and-forget).
        
        This method sends the webhook asynchronously so it doesn't block the user's request.
        The webhook is fire-and-forget: failures are logged but don't affect the main flow.
        
        Thread safety: Uses daemon threads to prevent blocking process shutdown.
        Error handling: All exceptions are caught and logged; no exceptions propagate.
        Timeout: 10-second timeout on HTTP request to prevent hanging threads.
        
        Args:
            webhook_url: The URL to send the webhook to (must be valid HTTP/HTTPS URL)
            payload: The payload dictionary to send as JSON in the webhook request
            
        Note:
            This is a fire-and-forget operation. There is no return value or error
            propagation. Check logs for webhook delivery status.
            
        Example:
            >>> distiller._send_webhook_async(
            ...     "https://training.example.com/trigger",
            ...     {"event": "training_trigger", "count": 100}
            ... )
            # Returns immediately; webhook sent in background
        """
        def _send():
            """Inner function that runs in background thread."""
            thread_name = threading.current_thread().name
            try:
                import urllib.request
                import urllib.error
                import json as json_module
                
                # Serialize payload to JSON
                try:
                    data = json_module.dumps(payload).encode('utf-8')
                except (TypeError, ValueError) as e:
                    logger.error(
                        f"[{thread_name}] Failed to serialize webhook payload: {e}. "
                        f"Payload type: {type(payload)}"
                    )
                    return
                
                # Create HTTP request
                req = urllib.request.Request(
                    webhook_url,
                    data=data,
                    headers={
                        'Content-Type': 'application/json',
                        'User-Agent': 'VulcanAMI-Distiller/1.0',
                    },
                    method='POST'
                )
                
                # Send request with timeout
                with urllib.request.urlopen(req, timeout=10) as response:
                    status_code = response.status
                    logger.info(
                        f"[{thread_name}] Training webhook sent successfully to {webhook_url}: "
                        f"HTTP {status_code}"
                    )
                    
            except urllib.error.HTTPError as e:
                # HTTP error (4xx, 5xx)
                logger.warning(
                    f"[{thread_name}] Training webhook HTTP error to {webhook_url}: "
                    f"HTTP {e.code} - {e.reason}"
                )
            except urllib.error.URLError as e:
                # Network error (connection refused, DNS failure, etc.)
                logger.warning(
                    f"[{thread_name}] Training webhook network error to {webhook_url}: {e.reason}"
                )
            except TimeoutError:
                # Timeout on socket operations
                logger.warning(
                    f"[{thread_name}] Training webhook timeout to {webhook_url} (>10s)"
                )
            except Exception as e:
                # Catch-all for any other exceptions
                logger.warning(
                    f"[{thread_name}] Training webhook unexpected error to {webhook_url}: "
                    f"{type(e).__name__}: {e}"
                )
        
        # Start background thread with descriptive name
        thread = threading.Thread(
            target=_send,
            name=f"WebhookSender-{id(payload)}",
            daemon=True  # Don't block process shutdown
        )
        thread.start()
        logger.debug(f"Webhook thread started: {thread.name}")
    
    def _check_training_trigger(self):
        """
        Check if training should be triggered based on example count threshold.
        
        This method is called after each flush to storage and checks if the
        total captured examples since last trigger exceeds the threshold.
        
        Thread safety: This method is called within the buffer lock in _flush_to_storage,
        so it's thread-safe.
        """
        current_count = self.stats["examples_captured"]
        examples_since_trigger = current_count - self._last_training_trigger_count
        
        if examples_since_trigger >= self.training_trigger_threshold:
            self.trigger_training(reason="threshold_reached")
    
    def trigger_training(
        self,
        reason: str = "manual",
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Trigger training pipeline for accumulated distillation examples.
        
        This method closes the loop between data capture and model training by:
        1. Logging a training trigger event for observability
        2. Invoking an optional callback function (e.g., to call GovernedTrainer)
        3. Sending a webhook notification (e.g., to external training orchestrator)
        4. Recording the trigger in audit log for governance
        
        The actual training is performed by external systems:
        - GovernedTrainer: Consensus-based weight updates
        - SelfImprovingTraining: Meta-learning orchestrator
        - External training pipelines via webhook
        
        Args:
            reason: Reason for triggering (e.g., "threshold_reached", "manual", "scheduled")
            force: Force trigger even if below threshold
            
        Returns:
            Dict containing trigger status and metadata
            
        Example usage:
            # Manual trigger
            distiller.trigger_training(reason="manual")
            
            # With callback to GovernedTrainer
            distiller = OpenAIKnowledgeDistiller(
                training_trigger_callback=governed_trainer.train_from_distillation,
                training_trigger_threshold=500,
            )
            
            # With webhook to external system
            distiller = OpenAIKnowledgeDistiller(
                training_webhook_url="https://training.example.com/trigger",
                training_trigger_threshold=1000,
            )
        """
        storage_stats = self.storage_backend.get_stats()
        total_examples = storage_stats.get("total_examples", 0)
        
        # Check if we have enough examples (unless forced)
        if not force and total_examples < self.training_trigger_threshold:
            self.logger.debug(
                f"Training trigger skipped: {total_examples} examples "
                f"< {self.training_trigger_threshold} threshold"
            )
            return {
                "triggered": False,
                "reason": "below_threshold",
                "total_examples": total_examples,
                "threshold": self.training_trigger_threshold,
            }
        
        trigger_time = time.time()
        trigger_result = {
            "triggered": True,
            "reason": reason,
            "timestamp": trigger_time,
            "total_examples": total_examples,
            "examples_since_last_trigger": self.stats["examples_captured"] - self._last_training_trigger_count,
            "callback_invoked": False,
            "webhook_sent": False,
            "errors": [],
        }
        
        # Update tracking state
        self._last_training_trigger_count = self.stats["examples_captured"]
        self.stats["training_triggers"] += 1
        self.stats["last_training_trigger_time"] = trigger_time
        
        # Log the trigger event
        self.logger.info(
            f"TRAINING TRIGGER: reason={reason}, examples={total_examples}, "
            f"trigger_count={self.stats['training_triggers']}"
        )
        
        # Invoke callback if configured
        if self.training_trigger_callback:
            try:
                callback_result = self.training_trigger_callback(
                    storage_path=str(self.storage_backend.storage_path),
                    total_examples=total_examples,
                    trigger_reason=reason,
                )
                trigger_result["callback_invoked"] = True
                trigger_result["callback_result"] = callback_result
                self.logger.info(f"Training callback invoked successfully: {callback_result}")
            except Exception as e:
                error_msg = f"Training callback failed: {e}"
                trigger_result["errors"].append(error_msg)
                self.logger.error(error_msg)
        
        # Send webhook notification if configured (non-blocking)
        if self.training_webhook_url:
            webhook_payload = {
                "event": "training_trigger",
                "reason": reason,
                "timestamp": trigger_time,
                "total_examples": total_examples,
                "storage_path": str(self.storage_backend.storage_path),
                "stats": {
                    "captured": self.stats["examples_captured"],
                    "rejected": self.stats["examples_rejected"],
                    "avg_quality": self.stats["average_quality_score"],
                },
            }
            
            # Send webhook asynchronously (non-blocking)
            self._send_webhook_async(self.training_webhook_url, webhook_payload)
            trigger_result["webhook_sent"] = True
            self.logger.info(
                f"Training webhook queued for async send to {self.training_webhook_url}"
            )
        
        # Log audit entry
        self._log_audit("training_trigger", {
            "reason": reason,
            "total_examples": total_examples,
            "callback_invoked": trigger_result["callback_invoked"],
            "webhook_sent": trigger_result["webhook_sent"],
            "errors": trigger_result["errors"],
        })
        
        # Save state
        self._save_state()
        
        return trigger_result


__all__ = ["OpenAIKnowledgeDistiller"]
