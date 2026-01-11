# ============================================================
# VULCAN-AGI Example Quality Validator Module
# Validates training examples for quality and safety
# ============================================================
#
# Implements multi-stage filtering:
#     1. Length and format validation
#     2. Boilerplate/refusal detection
#     3. Content quality scoring
#     4. Diversity sampling
#     5. Domain-specific validators
#
# VERSION HISTORY:
#     1.0.0 - Extracted from main.py for modular architecture
#     1.1.0 - Added thread-safe deduplication with true LRU eviction
# ============================================================

import hashlib
import logging
import re
import threading
from collections import deque
from typing import Dict, List, Optional, Tuple

# Module metadata
__version__ = "1.1.0"
__author__ = "VULCAN-AGI Team"

logger = logging.getLogger(__name__)


class ExampleQualityValidator:
    """
    Validates training examples for quality and safety.
    
    Implements multi-stage filtering:
    1. Length and format validation
    2. Boilerplate/refusal detection
    3. Content quality scoring
    4. Diversity sampling
    5. Domain-specific validators
    """
    
    # Thresholds
    MIN_RESPONSE_LENGTH = 50
    MAX_RESPONSE_LENGTH = 4000
    MIN_QUALITY_SCORE = 0.65
    MAX_BOILERPLATE_RATIO = 0.4
    
    # Safety/refusal patterns to reject
    REFUSAL_PATTERNS = [
        r"i cannot",
        r"i can't",
        r"i'm not able to",
        r"i am not able to",
        r"as an ai",
        r"as a language model",
        r"i don't have the ability",
        r"i apologize, but",
        r"i'm sorry, but i cannot",
    ]
    
    # Boilerplate patterns that reduce quality
    BOILERPLATE_PATTERNS = [
        r"^(sure|of course|certainly|absolutely)[,!.]?\s*",
        r"^(great question|good question)[!.]?\s*",
        r"^(here's|here is)\s+(a|the|my)\s+",
        r"^let me\s+",
        r"^i'd be happy to\s+",
        r"\bi hope this helps\b",
        r"\bfeel free to ask\b",
        r"\bdon't hesitate to\b",
    ]
    
    def __init__(self, max_seen_hashes: int = 10000):
        """
        Initialize the quality validator.
        
        Args:
            max_seen_hashes: Maximum number of hashes to track for deduplication
        """
        self.refusal_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.REFUSAL_PATTERNS
        ]
        self.boilerplate_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.BOILERPLATE_PATTERNS
        ]
        
        # Diversity tracking (hash-based deduplication) - THREAD SAFE
        self._seen_hashes: set = set()
        self._hash_queue = deque(maxlen=max_seen_hashes)  # True LRU using deque
        self._hash_lock = threading.Lock()  # Protect concurrent access
        self._max_seen_hashes = max_seen_hashes
    
    def validate(
        self,
        prompt: str,
        response: str,
        local_response: Optional[str] = None,
    ) -> Tuple[bool, float, List[str]]:
        """
        Validate an example for training suitability.
        
        Args:
            prompt: The user prompt
            response: The model response
            local_response: Optional local LLM response for diversity comparison
            
        Returns:
            Tuple of (passed, quality_score, rejection_reasons)
        """
        rejection_reasons = []
        quality_score = 0.0
        
        # 1. Length validation
        if len(response) < self.MIN_RESPONSE_LENGTH:
            rejection_reasons.append(f"too_short:{len(response)}")
        elif len(response) > self.MAX_RESPONSE_LENGTH:
            rejection_reasons.append(f"too_long:{len(response)}")
        else:
            # Score based on optimal length (100-2000 chars)
            if 100 <= len(response) <= 2000:
                quality_score += 0.2
            else:
                quality_score += 0.1
        
        # 2. Refusal detection
        for pattern in self.refusal_patterns:
            if pattern.search(response[:200]):  # Check start of response
                rejection_reasons.append("contains_refusal")
                break
        else:
            quality_score += 0.15
        
        # 3. Boilerplate detection
        boilerplate_count = sum(
            1 for p in self.boilerplate_patterns if p.search(response)
        )
        boilerplate_ratio = boilerplate_count / max(len(self.boilerplate_patterns), 1)
        if boilerplate_ratio > self.MAX_BOILERPLATE_RATIO:
            rejection_reasons.append(f"high_boilerplate:{boilerplate_ratio:.2f}")
        else:
            quality_score += 0.15 * (1 - boilerplate_ratio)
        
        # 4. Coherence checks
        # - Complete sentences
        if response.strip().endswith((".", "!", "?", '"', "```")):
            quality_score += 0.1
        else:
            rejection_reasons.append("incomplete_response")
        
        # - Reasonable word count
        word_count = len(response.split())
        if 10 <= word_count <= 500:
            quality_score += 0.1
        
        # 5. Diversity check (deduplication) - THREAD SAFE
        response_hash = hashlib.sha256(response.encode()).hexdigest()[:16]
        
        with self._hash_lock:  # CRITICAL: Protect concurrent access
            if response_hash in self._seen_hashes:
                rejection_reasons.append("duplicate_content")
            else:
                quality_score += 0.1
                
                # True LRU eviction using deque
                # When deque is full, oldest item is automatically dropped
                if len(self._hash_queue) >= self._max_seen_hashes:
                    oldest_hash = self._hash_queue.popleft()
                    self._seen_hashes.discard(oldest_hash)
                
                # Add new hash
                self._hash_queue.append(response_hash)
                self._seen_hashes.add(response_hash)
        
        # 6. Diversity score (if local response available)
        if local_response:
            local_words = set(local_response.lower().split())
            response_words = set(response.lower().split())
            if local_words:
                # Higher score if OpenAI provides new information
                new_words = response_words - local_words
                diversity = len(new_words) / max(len(response_words), 1)
                quality_score += min(0.1, diversity * 0.15)
                
                # Reject if too similar (no learning value)
                if diversity < 0.1:
                    rejection_reasons.append(f"low_diversity:{diversity:.2f}")
        else:
            quality_score += 0.05
        
        # 7. Relevance check (prompt-response overlap)
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        if prompt_words:
            relevance = len(prompt_words & response_words) / len(prompt_words)
            quality_score += min(0.1, relevance * 0.15)
        
        # Final decision
        passed = (
            len(rejection_reasons) == 0
            and quality_score >= self.MIN_QUALITY_SCORE
        )
        
        return passed, min(1.0, quality_score), rejection_reasons
    
    def clear_seen_hashes(self):
        """Clear the seen hashes set for testing."""
        self._seen_hashes.clear()
    
    def get_stats(self) -> Dict[str, int]:
        """Get validator statistics."""
        return {
            "seen_hashes_count": len(self._seen_hashes),
            "max_seen_hashes": self._max_seen_hashes,
        }


__all__ = ["ExampleQualityValidator"]
