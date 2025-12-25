# ============================================================
# VULCAN-AGI PII Redactor Module
# Redacts Personally Identifiable Information AND Secrets from text
# ============================================================
#
# Implements privacy and security compliance by detecting and masking:
#     - Email addresses
#     - Phone numbers
#     - Credit card numbers
#     - SSN patterns
#     - IP addresses
#     - Names (basic detection)
#     - API keys and tokens (OpenAI, AWS, GitHub, etc.)
#     - Passwords and credentials
#     - Bearer tokens and JWTs
#     - Connection strings
#
# VERSION HISTORY:
#     1.0.0 - Extracted from main.py for modular architecture
# ============================================================

import logging
import re
from typing import Dict, Tuple

# Module metadata
__version__ = "1.0.0"
__author__ = "VULCAN-AGI Team"

logger = logging.getLogger(__name__)


class PIIRedactor:
    """
    Redacts Personally Identifiable Information AND Secrets from text.
    
    Implements privacy and security compliance by detecting and masking:
    - Email addresses
    - Phone numbers
    - Credit card numbers
    - SSN patterns
    - IP addresses
    - Names (basic detection)
    - API keys and tokens (OpenAI, AWS, GitHub, etc.)
    - Passwords and credentials
    - Bearer tokens and JWTs
    - Connection strings
    """
    
    # Regex patterns for PII
    PII_PATTERNS = {
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "phone": r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
        "ssn": r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b',
        "credit_card": r'\b(?:\d{4}[-.\s]?){3}\d{4}\b',
        "ip_address": r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
    }
    
    # Regex patterns for secrets/credentials (CRITICAL - must never be stored)
    SECRET_PATTERNS = {
        "openai_key": r'\bsk-[A-Za-z0-9]{20,}\b',
        "aws_access_key": r'\b(AKIA|ABIA|ACCA|ASIA)[0-9A-Z]{16}\b',
        "aws_secret_key": r'\b[A-Za-z0-9/+=]{40}\b',
        "github_token": r'\b(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}\b',
        "generic_api_key": r'\b(api[_-]?key|apikey|access[_-]?token)["\s:=]+["\']?[A-Za-z0-9_\-]{20,}["\']?\b',
        "bearer_token": r'\b[Bb]earer\s+[A-Za-z0-9_\-\.]+\b',
        "jwt_token": r'\beyJ[A-Za-z0-9_-]*\.eyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_\-]+\b',
        "password_field": r'\b(password|passwd|pwd)["\s:=]+["\']?[^\s"\']{4,}["\']?\b',
        "connection_string": r'\b(mongodb|mysql|postgres|redis)://[^\s]+\b',
        "private_key": r'-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----',
    }
    
    # Common name patterns (very basic - production would use NER)
    NAME_MARKERS = ["my name is", "i am", "i'm", "call me", "this is"]
    
    def __init__(self):
        """Initialize the PII redactor with compiled patterns."""
        self.pii_patterns = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self.PII_PATTERNS.items()
        }
        self.secret_patterns = {
            name: re.compile(pattern, re.IGNORECASE if "password" in name else 0)
            for name, pattern in self.SECRET_PATTERNS.items()
        }
        self.redaction_count = 0
        self.secrets_detected = 0
    
    def redact(self, text: str) -> Tuple[str, Dict[str, int]]:
        """
        Redact PII and secrets from text.
        
        Args:
            text: The text to redact
            
        Returns:
            Tuple of (redacted_text, redaction_stats)
        """
        redacted = text
        stats = {}
        
        # CRITICAL: Redact secrets FIRST (highest priority)
        for name, pattern in self.secret_patterns.items():
            matches = pattern.findall(redacted)
            if matches:
                stats[f"SECRET_{name}"] = len(matches)
                redacted = pattern.sub(f"[REDACTED_SECRET_{name.upper()}]", redacted)
                self.secrets_detected += len(matches)
        
        # Then redact PII
        for name, pattern in self.pii_patterns.items():
            matches = pattern.findall(redacted)
            if matches:
                stats[name] = len(matches)
                redacted = pattern.sub(f"[REDACTED_{name.upper()}]", redacted)
                self.redaction_count += len(matches)
        
        # Basic name detection (after markers)
        for marker in self.NAME_MARKERS:
            if marker in redacted.lower():
                pattern = re.compile(
                    f"({re.escape(marker)})\\s+(\\w+)",
                    re.IGNORECASE
                )
                if pattern.search(redacted):
                    stats["potential_name"] = stats.get("potential_name", 0) + 1
                    redacted = pattern.sub(r"\1 [REDACTED_NAME]", redacted)
        
        return redacted, stats
    
    def contains_secrets(self, text: str) -> bool:
        """
        Check if text contains any secrets (for hard rejection).
        
        Args:
            text: The text to check
            
        Returns:
            True if text contains secrets, False otherwise
        """
        for pattern in self.secret_patterns.values():
            if pattern.search(text):
                return True
        return False
    
    def get_stats(self) -> Dict[str, int]:
        """Get redaction statistics."""
        return {
            "total_pii_redactions": self.redaction_count,
            "secrets_detected": self.secrets_detected,
        }
    
    def reset_stats(self):
        """Reset redaction statistics."""
        self.redaction_count = 0
        self.secrets_detected = 0


__all__ = ["PIIRedactor"]
