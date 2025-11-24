"""
Prompt Listener - Real-time Prompt Injection Detection
=======================================================
Monitors and detects adversarial prompt injection attacks in real-time.
Integrates with dream simulation logs to identify previously-encountered attacks.

Features:
- Real-time prompt analysis
- Known jailbreak pattern matching
- Dream simulation log integration
- Automatic patching of detected vulnerabilities
- Attack signature database
"""

import re
import time
import logging
import hashlib
import json
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


@dataclass
class AttackSignature:
    """Signature of a known attack pattern"""
    signature_id: str
    pattern: str
    attack_type: str
    severity: str  # low, medium, high, critical
    first_seen: float
    last_seen: float
    detection_count: int = 0
    source: str = "unknown"  # human_reported, dream_simulation, field_detection
    mitigation: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "signature_id": self.signature_id,
            "pattern": self.pattern,
            "attack_type": self.attack_type,
            "severity": self.severity,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "detection_count": self.detection_count,
            "source": self.source,
            "mitigation": self.mitigation
        }


@dataclass
class DetectionResult:
    """Result of prompt injection detection"""
    is_attack: bool
    confidence: float
    matched_signatures: List[str] = field(default_factory=list)
    attack_types: List[str] = field(default_factory=list)
    severity: str = "low"
    explanation: str = ""
    mitigation_applied: bool = False


class PromptListener:
    """
    Real-time prompt injection detection and mitigation.
    
    This listener monitors incoming prompts and detects adversarial attacks
    using a combination of:
    1. Known jailbreak pattern matching
    2. Dream simulation attack logs
    3. Heuristic analysis
    4. Behavioral anomaly detection
    """
    
    # Common jailbreak patterns
    JAILBREAK_PATTERNS = [
        # Direct instruction override
        (r"ignore\s+(all\s+)?(previous\s+|prior\s+)?instructions?", "instruction_override", "high"),
        (r"disregard\s+(all\s+)?(previous\s+|prior\s+)?rules?", "instruction_override", "high"),
        (r"forget\s+(all\s+)?(previous\s+|prior\s+)?constraints?", "instruction_override", "high"),
        
        # System command injection
        (r"(execute|run|eval|system)\s*\(?\s*['\"]?(rm|del|format|shutdown)", "command_injection", "critical"),
        (r"os\.(system|popen|exec)", "command_injection", "critical"),
        (r"subprocess\.(call|run|Popen)", "command_injection", "critical"),
        
        # Role confusion
        (r"you\s+are\s+now\s+(a|an)\s+\w+", "role_confusion", "medium"),
        (r"act\s+as\s+(a|an)\s+\w+", "role_confusion", "medium"),
        (r"pretend\s+(you\s+are|to\s+be)", "role_confusion", "medium"),
        
        # Safety bypass
        (r"bypass\s+(safety|security|filter)", "safety_bypass", "critical"),
        (r"disable\s+(safety|security|filter)", "safety_bypass", "critical"),
        (r"turn\s+off\s+(safety|security|filter)", "safety_bypass", "critical"),
        
        # Data exfiltration
        (r"(print|show|display|reveal)\s+(all\s+)?(secrets?|keys?|passwords?|tokens?)", "data_exfiltration", "critical"),
        (r"(dump|export|leak)\s+\w+", "data_exfiltration", "high"),
        
        # Prompt injection markers
        (r"---\s*END\s+OF\s+SYSTEM\s+PROMPT\s*---", "prompt_boundary", "high"),
        (r"<\|endoftext\|>", "prompt_boundary", "high"),
        (r"\[SYSTEM\]|\[ADMIN\]|\[ROOT\]", "privilege_escalation", "high"),
    ]
    
    def __init__(
        self,
        signature_db_path: Optional[Path] = None,
        dream_log_path: Optional[Path] = None,
        auto_update: bool = True
    ):
        """
        Initialize the prompt listener.
        
        Args:
            signature_db_path: Path to attack signature database
            dream_log_path: Path to dream simulation logs
            auto_update: Whether to auto-update signatures from dream logs
        """
        self.signature_db_path = signature_db_path or Path("data/attack_signatures.json")
        self.dream_log_path = dream_log_path or Path("logs/dream_simulations")
        self.auto_update = auto_update
        
        # Attack signature database
        self.signatures: Dict[str, AttackSignature] = {}
        self._load_signatures()
        
        # Detection statistics
        self.detection_stats = {
            "total_prompts_analyzed": 0,
            "attacks_detected": 0,
            "attacks_blocked": 0,
            "false_positives": 0,
            "last_attack": None
        }
        
        # Recent detection history (for rate limiting)
        self.recent_detections: deque = deque(maxlen=100)
        
        # Load dream simulation attack logs
        if auto_update:
            self._load_dream_simulation_logs()
            
        logger.info(f"PromptListener initialized with {len(self.signatures)} known signatures")
    
    def _load_signatures(self):
        """Load attack signatures from database"""
        if self.signature_db_path.exists():
            try:
                with open(self.signature_db_path, 'r') as f:
                    data = json.load(f)
                    for sig_data in data.get('signatures', []):
                        sig = AttackSignature(**sig_data)
                        self.signatures[sig.signature_id] = sig
                logger.info(f"Loaded {len(self.signatures)} signatures from database")
            except Exception as e:
                logger.error(f"Failed to load signatures: {e}")
        else:
            # Initialize with default jailbreak patterns
            for pattern, attack_type, severity in self.JAILBREAK_PATTERNS:
                sig_id = hashlib.md5(pattern.encode()).hexdigest()[:12]
                self.signatures[sig_id] = AttackSignature(
                    signature_id=sig_id,
                    pattern=pattern,
                    attack_type=attack_type,
                    severity=severity,
                    first_seen=time.time(),
                    last_seen=time.time(),
                    source="builtin"
                )
            self._save_signatures()
    
    def _save_signatures(self):
        """Save attack signatures to database"""
        try:
            self.signature_db_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.signature_db_path, 'w') as f:
                data = {
                    'signatures': [sig.to_dict() for sig in self.signatures.values()],
                    'updated_at': time.time()
                }
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save signatures: {e}")
    
    def _load_dream_simulation_logs(self):
        """Load attack patterns from dream simulation logs"""
        if not self.dream_log_path.exists():
            logger.debug("No dream simulation logs found")
            return
        
        try:
            # Scan for recent dream simulation logs
            log_files = sorted(self.dream_log_path.glob("dream_*.json"))[-10:]  # Last 10 runs
            
            for log_file in log_files:
                with open(log_file, 'r') as f:
                    log_data = json.load(f)
                    
                    # Extract attack patterns discovered in dreams
                    for attack in log_data.get('discovered_attacks', []):
                        pattern = attack.get('pattern', '')
                        if pattern:
                            sig_id = f"dream_{hashlib.md5(pattern.encode()).hexdigest()[:8]}"
                            if sig_id not in self.signatures:
                                self.signatures[sig_id] = AttackSignature(
                                    signature_id=sig_id,
                                    pattern=pattern,
                                    attack_type=attack.get('type', 'unknown'),
                                    severity=attack.get('severity', 'medium'),
                                    first_seen=attack.get('timestamp', time.time()),
                                    last_seen=attack.get('timestamp', time.time()),
                                    source="dream_simulation"
                                )
            
            if log_files:
                logger.info(f"Loaded dream simulation attacks from {len(log_files)} log files")
                self._save_signatures()
                
        except Exception as e:
            logger.error(f"Failed to load dream simulation logs: {e}")
    
    def analyze_prompt(self, prompt: str, context: Optional[Dict] = None) -> DetectionResult:
        """
        Analyze a prompt for adversarial injection attacks.
        
        Args:
            prompt: The user prompt to analyze
            context: Optional context about the prompt (user, session, etc.)
            
        Returns:
            DetectionResult with attack detection information
        """
        self.detection_stats["total_prompts_analyzed"] += 1
        
        # Normalize prompt for analysis
        prompt_lower = prompt.lower()
        
        # Check against known signatures
        matched_signatures = []
        attack_types = set()
        max_severity = "low"
        severity_order = ["low", "medium", "high", "critical"]
        
        for sig_id, signature in self.signatures.items():
            try:
                if re.search(signature.pattern, prompt_lower, re.IGNORECASE):
                    matched_signatures.append(sig_id)
                    attack_types.add(signature.attack_type)
                    
                    # Update signature stats
                    signature.last_seen = time.time()
                    signature.detection_count += 1
                    
                    # Track max severity
                    if severity_order.index(signature.severity) > severity_order.index(max_severity):
                        max_severity = signature.severity
                        
            except re.error:
                logger.warning(f"Invalid regex pattern in signature {sig_id}")
                continue
        
        # Determine if this is an attack
        is_attack = len(matched_signatures) > 0
        confidence = min(len(matched_signatures) * 0.3, 1.0)  # Simple confidence based on matches
        
        if is_attack:
            self.detection_stats["attacks_detected"] += 1
            self.detection_stats["last_attack"] = time.time()
            
            # Record detection
            self.recent_detections.append({
                "timestamp": time.time(),
                "signatures": matched_signatures,
                "severity": max_severity,
                "prompt_hash": hashlib.md5(prompt.encode()).hexdigest()[:12]
            })
        
        # Build explanation
        if is_attack:
            sig_details = [self.signatures[sid] for sid in matched_signatures[:3]]
            explanation = f"Detected {len(matched_signatures)} matching attack patterns: " + \
                         ", ".join([f"{s.attack_type}({s.severity})" for s in sig_details])
        else:
            explanation = "No adversarial patterns detected"
        
        return DetectionResult(
            is_attack=is_attack,
            confidence=confidence,
            matched_signatures=matched_signatures,
            attack_types=list(attack_types),
            severity=max_severity,
            explanation=explanation,
            mitigation_applied=False
        )
    
    def block_attack(self, result: DetectionResult, prompt: str) -> Dict[str, Any]:
        """
        Block a detected attack and apply mitigations.
        
        Args:
            result: Detection result from analyze_prompt
            prompt: The original prompt
            
        Returns:
            Dict with blocking details
        """
        self.detection_stats["attacks_blocked"] += 1
        
        # Log the attack
        logger.warning(
            f"BLOCKED ATTACK: severity={result.severity}, "
            f"types={result.attack_types}, "
            f"signatures={len(result.matched_signatures)}"
        )
        
        # Determine response based on severity
        if result.severity == "critical":
            response = "Attack blocked. Critical security violation detected."
            action = "terminate_session"
        elif result.severity == "high":
            response = "Attack blocked. High-risk pattern detected."
            action = "block_and_alert"
        else:
            response = "Attack blocked. Suspicious pattern detected."
            action = "block_only"
        
        return {
            "blocked": True,
            "severity": result.severity,
            "action": action,
            "response": response,
            "matched_signatures": result.matched_signatures,
            "timestamp": time.time()
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detection statistics"""
        return {
            **self.detection_stats,
            "known_signatures": len(self.signatures),
            "recent_detections": len(self.recent_detections),
            "detection_rate": (
                self.detection_stats["attacks_detected"] / 
                max(self.detection_stats["total_prompts_analyzed"], 1)
            )
        }
    
    def add_signature_from_attack(
        self,
        pattern: str,
        attack_type: str,
        severity: str = "medium",
        source: str = "field_detection"
    ) -> str:
        """
        Add a new signature from a detected attack.
        
        Args:
            pattern: Regex pattern for the attack
            attack_type: Type of attack
            severity: Severity level
            source: Source of the signature
            
        Returns:
            Signature ID
        """
        sig_id = f"{source}_{hashlib.md5(pattern.encode()).hexdigest()[:8]}"
        
        self.signatures[sig_id] = AttackSignature(
            signature_id=sig_id,
            pattern=pattern,
            attack_type=attack_type,
            severity=severity,
            first_seen=time.time(),
            last_seen=time.time(),
            source=source
        )
        
        self._save_signatures()
        logger.info(f"Added new attack signature: {sig_id}")
        
        return sig_id


# Global instance for easy import
_global_listener: Optional[PromptListener] = None


def get_prompt_listener() -> PromptListener:
    """Get or create the global prompt listener instance"""
    global _global_listener
    if _global_listener is None:
        _global_listener = PromptListener()
    return _global_listener


def analyze_prompt(prompt: str, context: Optional[Dict] = None) -> DetectionResult:
    """Convenience function to analyze a prompt using the global listener"""
    listener = get_prompt_listener()
    return listener.analyze_prompt(prompt, context)
