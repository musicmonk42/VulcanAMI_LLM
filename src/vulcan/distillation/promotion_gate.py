# ============================================================
# VULCAN-AGI Promotion Gate Module
# Explicit promotion gate for trained weights
# ============================================================
#
# Requires:
#     - Evaluation score >= threshold
#     - Regression suite pass
#     - Provenance record created
#
# Only promotes weights after ALL requirements are met.
#
# VERSION HISTORY:
#     1.0.0 - Extracted from main.py for modular architecture
# ============================================================

import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from vulcan.distillation.storage import DistillationStorageBackend

# Module metadata
__version__ = "1.0.0"
__author__ = "VULCAN-AGI Team"

logger = logging.getLogger(__name__)


class PromotionGate:
    """
    Explicit promotion gate for trained weights.
    
    Requires:
    - Evaluation score >= threshold
    - Regression suite pass
    - Provenance record created
    
    Only promotes weights after ALL requirements are met.
    """
    
    # Promotion requirements
    MIN_EVAL_SCORE = 0.7
    MAX_REGRESSION_COUNT = 0
    
    def __init__(
        self,
        storage_backend: Optional["DistillationStorageBackend"] = None,
        min_eval_score: float = MIN_EVAL_SCORE,
        allow_regressions: int = MAX_REGRESSION_COUNT,
    ):
        """
        Initialize promotion gate.
        
        Args:
            storage_backend: Storage for provenance records
            min_eval_score: Minimum evaluation score for promotion
            allow_regressions: Maximum allowed regressions (0 = none)
        """
        self.storage = storage_backend
        self.min_eval_score = min_eval_score
        self.allow_regressions = allow_regressions
        self.logger = logging.getLogger("PromotionGate")
        
        # Promotion history
        self.promotions: List[Dict[str, Any]] = []
        self.rejections: List[Dict[str, Any]] = []
    
    def evaluate_for_promotion(
        self,
        eval_results: Dict[str, Any],
        training_metadata: Dict[str, Any],
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Evaluate if weights should be promoted.
        
        Args:
            eval_results: Evaluation results dictionary
            training_metadata: Training metadata dictionary
            
        Returns:
            Tuple of (approved, decision_details)
        """
        decision = {
            "timestamp": time.time(),
            "approved": False,
            "requirements": {},
            "reasons": [],
        }
        
        # Requirement 1: Evaluation score
        eval_score = eval_results.get("average_score", 0.0)
        eval_passed = eval_score >= self.min_eval_score
        decision["requirements"]["eval_score"] = {
            "required": self.min_eval_score,
            "actual": eval_score,
            "passed": eval_passed,
        }
        if not eval_passed:
            decision["reasons"].append(
                f"eval_score_below_threshold:{eval_score:.3f}<{self.min_eval_score}"
            )
        
        # Requirement 2: Regression check
        regressions = eval_results.get("regressions", [])
        regression_count = len(regressions)
        regression_passed = regression_count <= self.allow_regressions
        decision["requirements"]["regression_check"] = {
            "max_allowed": self.allow_regressions,
            "actual": regression_count,
            "passed": regression_passed,
            "details": regressions,
        }
        if not regression_passed:
            decision["reasons"].append(
                f"regression_count_exceeded:{regression_count}>{self.allow_regressions}"
            )
        
        # Requirement 3: Training metadata completeness
        required_fields = ["examples_count", "loss", "training_id"]
        metadata_complete = all(
            field in training_metadata for field in required_fields
        )
        decision["requirements"]["metadata_complete"] = {
            "required_fields": required_fields,
            "passed": metadata_complete,
        }
        if not metadata_complete:
            missing = [f for f in required_fields if f not in training_metadata]
            decision["reasons"].append(f"missing_metadata:{missing}")
        
        # Final decision
        decision["approved"] = eval_passed and regression_passed and metadata_complete
        
        # Record decision
        if decision["approved"]:
            self.promotions.append(decision)
        else:
            self.rejections.append(decision)
        
        return decision["approved"], decision
    
    def create_provenance_record(
        self,
        training_metadata: Dict[str, Any],
        eval_results: Dict[str, Any],
        promotion_decision: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Create a signed provenance record for the promotion.
        
        This record is immutable and provides audit trail.
        
        Args:
            training_metadata: Training metadata dictionary
            eval_results: Evaluation results dictionary
            promotion_decision: Promotion decision dictionary
            
        Returns:
            Provenance record dictionary
        """
        record = {
            "record_type": "weight_promotion",
            "record_id": hashlib.sha256(
                f"{time.time()}{training_metadata.get('training_id', '')}".encode()
            ).hexdigest()[:16],
            "created_at": time.time(),
            
            # Training details
            "training": {
                "id": training_metadata.get("training_id"),
                "examples_count": training_metadata.get("examples_count"),
                "loss": training_metadata.get("loss"),
                "timestamp": training_metadata.get("timestamp"),
            },
            
            # Evaluation details
            "evaluation": {
                "score": eval_results.get("average_score"),
                "domains_tested": list(eval_results.get("scores", {}).keys()),
                "regressions": eval_results.get("regressions", []),
                "improvements": eval_results.get("improvements", []),
            },
            
            # Promotion decision
            "decision": {
                "approved": promotion_decision.get("approved"),
                "requirements_met": promotion_decision.get("requirements"),
                "rejection_reasons": promotion_decision.get("reasons", []),
            },
            
            # Provenance hash (for integrity verification)
            "hash": None,  # Computed below
        }
        
        # Compute record hash (excluding hash field itself)
        record_str = json.dumps(record, sort_keys=True, separators=(',', ':'))
        record["hash"] = hashlib.sha256(record_str.encode()).hexdigest()
        
        # Store provenance record
        if self.storage:
            self.storage.append_provenance(record)
        
        self.logger.info(
            f"Provenance record created: {record['record_id']} "
            f"(approved={record['decision']['approved']})"
        )
        
        return record
    
    def get_promotion_history(self, limit: int = 100) -> Dict[str, Any]:
        """
        Get recent promotion history.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            Dictionary with promotion history
        """
        return {
            "total_promotions": len(self.promotions),
            "total_rejections": len(self.rejections),
            "recent_promotions": self.promotions[-limit:],
            "recent_rejections": self.rejections[-limit:],
        }


__all__ = ["PromotionGate"]
