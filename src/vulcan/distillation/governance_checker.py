# ============================================================
# VULCAN-AGI Governance Sensitivity Checker Module
# Checks content against governance rules for sensitivity marking
# ============================================================
#
# Integrates with CSIU/governance system to identify content that
# should NOT be captured for training regardless of opt-in status.
#
# VERSION HISTORY:
#     1.0.0 - Extracted from main.py for modular architecture
# ============================================================

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

# Module metadata
__version__ = "1.0.0"
__author__ = "VULCAN-AGI Team"

logger = logging.getLogger(__name__)


class GovernanceSensitivityChecker:
    """
    Checks content against governance rules for sensitivity marking.
    
    Integrates with CSIU/governance system to identify content that
    should NOT be captured for training regardless of opt-in status.
    """
    
    # Hard-reject categories (never capture)
    SENSITIVE_CATEGORIES = {
        "auth_credentials": [
            r"\b(login|signin|sign[\s-]?in)\b.*\b(password|passwd|pwd)\b",
            r"\b(authenticate|authorization)\b",
            r"\bbearer\s+\w+",
            r"\bbasic\s+[A-Za-z0-9+/=]+",
        ],
        "payment_info": [
            r"\b(credit|debit)\s*card\b",
            r"\bcvv\b|\bcvc\b|\bsecurity\s*code\b",
            r"\bpayment\s*(method|info|details)\b",
            r"\bbank\s*(account|routing)\b",
            r"\biban\b|\bswift\b|\baba\b",
        ],
        "medical_phi": [
            r"\b(diagnosis|prescription|medication)\b",
            r"\bmedical\s*(record|history|condition)\b",
            r"\bpatient\s*(id|name|info)\b",
            r"\bhipaa\b",
        ],
        "legal_privileged": [
            r"\battorney[\s-]?client\b",
            r"\blegal\s*advice\b",
            r"\bconfidential\s*(legal|settlement)\b",
        ],
    }
    
    # Governance markers that indicate "do not capture"
    DO_NOT_CAPTURE_MARKERS = [
        "[CONFIDENTIAL]",
        "[DO NOT LOG]",
        "[SENSITIVE]",
        "[PRIVATE]",
        "[NO_TRAINING]",
        "[GOVERNANCE_RESTRICTED]",
    ]
    
    def __init__(self):
        """Initialize the governance sensitivity checker."""
        self.compiled_patterns = {}
        for category, patterns in self.SENSITIVE_CATEGORIES.items():
            self.compiled_patterns[category] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
        self.rejections_by_category: Dict[str, int] = {}
    
    def check_sensitivity(
        self,
        prompt: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str, List[str]]:
        """
        Check if content is marked sensitive by governance rules.
        
        Args:
            prompt: The user prompt
            response: The model response
            metadata: Optional metadata dictionary
            
        Returns:
            Tuple of (is_sensitive, category, reasons)
        """
        combined_text = f"{prompt} {response}".lower()
        reasons = []
        
        # Check for explicit governance markers
        for marker in self.DO_NOT_CAPTURE_MARKERS:
            if marker.lower() in combined_text:
                reasons.append(f"governance_marker:{marker}")
                return True, "governance_marked", reasons
        
        # Check metadata for governance flags
        if metadata:
            if metadata.get("governance_restricted"):
                reasons.append("metadata:governance_restricted")
                return True, "governance_flag", reasons
            if metadata.get("do_not_capture"):
                reasons.append("metadata:do_not_capture")
                return True, "explicit_flag", reasons
            if metadata.get("sensitivity_level", "").lower() in ["high", "critical"]:
                reasons.append(f"sensitivity_level:{metadata.get('sensitivity_level')}")
                return True, "sensitivity_level", reasons
        
        # Check against sensitive categories
        for category, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(combined_text):
                    reasons.append(f"category:{category}")
                    self.rejections_by_category[category] = (
                        self.rejections_by_category.get(category, 0) + 1
                    )
                    return True, category, reasons
        
        return False, "", []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rejection statistics by category."""
        return {
            "rejections_by_category": dict(self.rejections_by_category),
            "total_rejections": sum(self.rejections_by_category.values()),
        }
    
    def reset_stats(self):
        """Reset rejection statistics."""
        self.rejections_by_category.clear()


__all__ = ["GovernanceSensitivityChecker"]
