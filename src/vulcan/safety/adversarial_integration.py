# adversarial_integration.py
"""
Adversarial Tester Integration for VULCAN-AGI Safety Module.

This module integrates the AdversarialTester into:
1. Platform startup - initializes the tester on platform boot
2. Periodic testing - runs adversarial tests on a schedule
3. Query pipeline - real-time integrity checks on incoming queries

Usage:
    from vulcan.safety.adversarial_integration import (
        initialize_adversarial_tester,
        start_periodic_testing,
        check_query_integrity,
        get_adversarial_status
    )
    
    # Initialize at startup
    tester = initialize_adversarial_tester()
    
    # Start periodic testing (every hour)
    start_periodic_testing(tester, interval_seconds=3600)
    
    # Check query integrity in real-time
    result = check_query_integrity(query, tester)
    if not result["safe"]:
        return refusal_response(result["reason"])
"""

import hashlib
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURABLE THRESHOLDS
# ============================================================
# These can be overridden via environment variables

# Anomaly detection threshold (0.0-1.0) - higher = more lenient
ANOMALY_CONFIDENCE_THRESHOLD = float(os.getenv("ADVERSARIAL_ANOMALY_THRESHOLD", "0.9"))

# SHAP divergence threshold (0.0-1.0) - higher = more lenient  
SHAP_DIVERGENCE_THRESHOLD = float(os.getenv("ADVERSARIAL_SHAP_THRESHOLD", "0.8"))

# Success rate threshold for alerts (0.0-1.0) - lower = more sensitive
SUCCESS_RATE_ALERT_THRESHOLD = float(os.getenv("ADVERSARIAL_SUCCESS_RATE_THRESHOLD", "0.8"))

# Periodic testing interval in seconds (default: 1 hour)
PERIODIC_TEST_INTERVAL = int(os.getenv("ADVERSARIAL_PERIODIC_INTERVAL", "3600"))

# Global state for adversarial tester singleton
_ADVERSARIAL_TESTER = None
_ADVERSARIAL_LOCK = threading.RLock()
_PERIODIC_THREAD = None
_PERIODIC_RUNNING = False

# Try to import AdversarialTester
try:
    from src.adversarial_tester import AdversarialTester, AttackType
    ADVERSARIAL_TESTER_AVAILABLE = True
except ImportError:
    try:
        from adversarial_tester import AdversarialTester, AttackType
        ADVERSARIAL_TESTER_AVAILABLE = True
    except ImportError:
        AdversarialTester = None
        AttackType = None
        ADVERSARIAL_TESTER_AVAILABLE = False
        logger.warning("AdversarialTester not available - adversarial testing disabled")


def initialize_adversarial_tester(
    log_dir: str = "adversarial_logs",
    interpret_engine: Optional[Any] = None,
    nso_aligner: Optional[Any] = None,
    force_reinit: bool = False,
) -> Optional["AdversarialTester"]:
    """
    Initialize the adversarial tester singleton.
    
    This should be called during platform startup to create the adversarial
    tester instance. Subsequent calls return the existing instance unless
    force_reinit is True.
    
    Args:
        log_dir: Directory for adversarial test logs
        interpret_engine: Optional interpretability engine
        nso_aligner: Optional NSO aligner for safety audits
        force_reinit: Force re-initialization even if already initialized
        
    Returns:
        AdversarialTester instance or None if not available
    """
    global _ADVERSARIAL_TESTER
    
    if not ADVERSARIAL_TESTER_AVAILABLE:
        logger.error("AdversarialTester not available - cannot initialize")
        return None
    
    with _ADVERSARIAL_LOCK:
        if _ADVERSARIAL_TESTER is not None and not force_reinit:
            logger.debug("Returning existing AdversarialTester instance")
            return _ADVERSARIAL_TESTER
        
        try:
            # Create log directory
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize the tester
            _ADVERSARIAL_TESTER = AdversarialTester(
                interpret_engine=interpret_engine,
                nso_aligner=nso_aligner,
                log_dir=log_dir,
            )
            
            logger.info(f"AdversarialTester initialized: log_dir={log_dir}")
            logger.info(f"  Database: {log_path / 'adversarial_logs.db'}")
            
            return _ADVERSARIAL_TESTER
            
        except Exception as e:
            logger.error(f"Failed to initialize AdversarialTester: {e}")
            _ADVERSARIAL_TESTER = None
            return None


def get_adversarial_tester() -> Optional["AdversarialTester"]:
    """
    Get the current adversarial tester instance.
    
    Returns:
        AdversarialTester instance or None if not initialized
    """
    with _ADVERSARIAL_LOCK:
        return _ADVERSARIAL_TESTER


def start_periodic_testing(
    tester: Optional["AdversarialTester"] = None,
    interval_seconds: int = 3600,
    tensor_size: int = 512,
    run_immediately: bool = True,
) -> bool:
    """
    Start periodic adversarial testing in a background thread.
    
    This runs a comprehensive adversarial test suite at regular intervals
    to detect any degradation in system security.
    
    Args:
        tester: AdversarialTester instance (uses singleton if None)
        interval_seconds: Interval between test runs (default: 1 hour)
        tensor_size: Size of test tensors (default: 512)
        run_immediately: Run first test immediately (default: True)
        
    Returns:
        True if periodic testing started successfully, False otherwise
    """
    global _PERIODIC_THREAD, _PERIODIC_RUNNING
    
    if tester is None:
        tester = get_adversarial_tester()
    
    if tester is None:
        logger.error("Cannot start periodic testing - no AdversarialTester available")
        return False
    
    if _PERIODIC_RUNNING:
        logger.warning("Periodic testing already running")
        return True
    
    def run_periodic_tests():
        """Background thread function for periodic testing."""
        global _PERIODIC_RUNNING
        _PERIODIC_RUNNING = True
        
        logger.info(f"Starting periodic adversarial testing (interval: {interval_seconds}s)")
        
        first_run = run_immediately
        
        while _PERIODIC_RUNNING:
            if first_run:
                first_run = False
            else:
                # Wait for next interval
                time.sleep(interval_seconds)
            
            if not _PERIODIC_RUNNING:
                break
            
            try:
                logger.info("🔒 Starting periodic adversarial test suite...")
                
                # Generate base tensor for testing
                base_tensor = np.random.randn(tensor_size).astype(np.float32)
                
                # Run full test suite
                results = tester.run_adversarial_suite(
                    base_tensor=base_tensor,
                    proposal={"id": "periodic_check", "type": "system_integrity"}
                )
                
                # Log results
                summary = results.get("summary", {})
                total_tests = summary.get("total_tests", 0)
                failures = summary.get("failures", 0)
                success_rate = summary.get("success_rate", 0)
                max_divergence = summary.get("max_divergence", 0)
                
                logger.info(f"✅ Adversarial tests complete:")
                logger.info(f"  - Total tests: {total_tests}")
                logger.info(f"  - Failures: {failures}")
                logger.info(f"  - Success rate: {success_rate:.2%}")
                logger.info(f"  - Max divergence: {max_divergence:.4f}")
                
                # Alert on high failure rate (configurable threshold)
                if success_rate < SUCCESS_RATE_ALERT_THRESHOLD:
                    logger.warning(f"⚠️ HIGH FAILURE RATE: {failures} tests failed! (threshold: {SUCCESS_RATE_ALERT_THRESHOLD:.0%})")
                
            except Exception as e:
                logger.error(f"❌ Periodic adversarial test failed: {e}")
        
        logger.info("Periodic adversarial testing stopped")
    
    # Start background thread
    _PERIODIC_THREAD = threading.Thread(
        target=run_periodic_tests,
        name="AdversarialPeriodicTester",
        daemon=True,
    )
    _PERIODIC_THREAD.start()
    
    logger.info("Periodic adversarial testing thread started")
    return True


def stop_periodic_testing() -> None:
    """Stop periodic adversarial testing."""
    global _PERIODIC_RUNNING, _PERIODIC_THREAD
    
    _PERIODIC_RUNNING = False
    
    if _PERIODIC_THREAD is not None:
        _PERIODIC_THREAD.join(timeout=5)
        _PERIODIC_THREAD = None
    
    logger.info("Periodic adversarial testing stopped")


def encode_query_to_tensor(query: str, tensor_size: int = 512) -> np.ndarray:
    """
    Encode a text query to a numeric tensor for adversarial testing.
    
    This uses a simple hash-based encoding that converts the query text
    into a fixed-size numeric tensor suitable for adversarial analysis.
    
    Args:
        query: The text query to encode
        tensor_size: Size of output tensor (default: 512)
        
    Returns:
        numpy array of shape (tensor_size,)
    """
    # Use SHA-256 hash as a seed for reproducible encoding
    hash_bytes = hashlib.sha256(query.encode('utf-8')).digest()
    seed = int.from_bytes(hash_bytes[:4], 'big')
    
    # Create reproducible random state
    rng = np.random.RandomState(seed)
    
    # Generate base tensor from query characteristics
    query_len = len(query)
    word_count = len(query.split())
    char_codes = [ord(c) for c in query[:min(len(query), tensor_size // 2)]]
    
    # Build tensor components
    base = rng.randn(tensor_size).astype(np.float32)
    
    # Add query-specific features
    if char_codes:
        # Normalize and add character features
        char_array = np.array(char_codes, dtype=np.float32)
        char_array = (char_array - char_array.mean()) / (char_array.std() + 1e-8)
        base[:len(char_codes)] += char_array * 0.5
    
    # Add length-based features
    base[0] = query_len / 1000.0  # Normalized length
    base[1] = word_count / 100.0  # Normalized word count
    
    # Normalize final tensor
    base = (base - base.mean()) / (base.std() + 1e-8)
    
    return base


def check_query_integrity(
    query: str,
    tester: Optional["AdversarialTester"] = None,
    tensor_size: int = 512,
) -> Dict[str, Any]:
    """
    Check query integrity using adversarial testing.
    
    This performs real-time integrity checks on incoming queries to detect:
    - Anomalous input patterns
    - Adversarial manipulation attempts
    - Out-of-distribution inputs
    
    Args:
        query: The user query to check
        tester: AdversarialTester instance (uses singleton if None)
        tensor_size: Size of tensor encoding (default: 512)
        
    Returns:
        Dict with:
        - safe: True if query passes integrity checks
        - reason: Reason for blocking if not safe
        - anomaly_score: Anomaly detection score (if detected)
        - details: Full integrity check results
    """
    if tester is None:
        tester = get_adversarial_tester()
    
    if tester is None:
        # No tester available - allow query but log warning
        logger.debug("AdversarialTester not available - skipping integrity check")
        return {
            "safe": True,
            "reason": None,
            "anomaly_score": None,
            "details": {"skipped": True, "reason": "tester_not_available"},
        }
    
    try:
        # Encode query to tensor
        query_tensor = encode_query_to_tensor(query, tensor_size)
        
        # Run real-time integrity check
        integrity_results = tester.realtime_integrity_check(
            graph={"query": query[:200], "id": hashlib.md5(query.encode()).hexdigest()[:8]},
            current_tensor=query_tensor,
        )
        
        # Analyze results
        is_anomaly = integrity_results.get("is_anomaly", False)
        anomaly_confidence = integrity_results.get("anomaly_confidence", 0)
        safety_level = integrity_results.get("safety_level", "safe")
        shap_stable = integrity_results.get("shap_stable", True)
        has_nan = integrity_results.get("has_nan", False)
        has_inf = integrity_results.get("has_inf", False)
        suspicious_range = integrity_results.get("suspicious_range", False)
        
        # Determine if query should be blocked
        should_block = False
        block_reason = None
        
        # Use configurable anomaly confidence threshold
        if is_anomaly and anomaly_confidence > ANOMALY_CONFIDENCE_THRESHOLD:
            should_block = True
            block_reason = f"High-confidence anomaly detected (score: {anomaly_confidence:.2f}, threshold: {ANOMALY_CONFIDENCE_THRESHOLD})"
            logger.warning(f"🚨 ANOMALY DETECTED in query: {query[:100]}...")
            logger.warning(f"Anomaly confidence: {anomaly_confidence:.2f}")
        
        if safety_level in ("high_risk", "critical"):
            should_block = True
            block_reason = f"Safety check failed: {safety_level}"
            logger.warning(f"⚠️ High-risk query detected: safety_level={safety_level}")
        
        if has_nan or has_inf:
            should_block = True
            block_reason = "Invalid numeric values detected in query encoding"
        
        # Use configurable SHAP divergence threshold
        if not shap_stable and integrity_results.get("shap_divergence", 0) > SHAP_DIVERGENCE_THRESHOLD:
            logger.warning(f"⚠️ SHAP unstable: divergence={integrity_results.get('shap_divergence', 0):.4f} (threshold: {SHAP_DIVERGENCE_THRESHOLD})")
        
        # Log all check results
        checks_passed = integrity_results.get("checks_performed", [])
        if not should_block:
            logger.debug(f"Query passed integrity checks: {checks_passed}")
        
        return {
            "safe": not should_block,
            "reason": block_reason,
            "anomaly_score": anomaly_confidence if is_anomaly else None,
            "details": integrity_results,
        }
        
    except Exception as e:
        logger.error(f"Query integrity check failed: {e}")
        # On error, allow the query but log the issue
        return {
            "safe": True,
            "reason": None,
            "anomaly_score": None,
            "details": {"error": str(e)},
        }


def get_adversarial_status() -> Dict[str, Any]:
    """
    Get current status of adversarial testing system.
    
    Returns:
        Dict with status information including:
        - available: Whether AdversarialTester is available
        - initialized: Whether tester is initialized
        - periodic_running: Whether periodic testing is running
        - attack_stats: Statistics about past attacks
        - database_path: Path to the SQLite database
    """
    global _PERIODIC_RUNNING
    
    status = {
        "available": ADVERSARIAL_TESTER_AVAILABLE,
        "initialized": _ADVERSARIAL_TESTER is not None,
        "periodic_running": _PERIODIC_RUNNING,
        "attack_stats": {},
        "database_path": None,
    }
    
    if _ADVERSARIAL_TESTER is not None:
        try:
            # Get attack statistics
            with _ADVERSARIAL_TESTER.stats_lock:
                status["attack_stats"] = dict(_ADVERSARIAL_TESTER.attack_stats)
            
            # Get database path
            status["database_path"] = str(_ADVERSARIAL_TESTER.log_dir / "adversarial_logs.db")
            
            # Try to get database stats
            try:
                import sqlite3
                db_path = _ADVERSARIAL_TESTER.log_dir / "adversarial_logs.db"
                if db_path.exists():
                    conn = sqlite3.connect(str(db_path))
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM attack_logs")
                    count = cursor.fetchone()[0]
                    status["total_logged_attacks"] = count
                    
                    # Get recent attacks
                    cursor.execute("""
                        SELECT timestamp, attack_type, success 
                        FROM attack_logs 
                        ORDER BY timestamp DESC 
                        LIMIT 5
                    """)
                    recent = cursor.fetchall()
                    status["recent_attacks"] = [
                        {"timestamp": r[0], "type": r[1], "success": r[2]}
                        for r in recent
                    ]
                    conn.close()
            except Exception as e:
                logger.debug(f"Could not get database stats: {e}")
                
        except Exception as e:
            logger.error(f"Error getting adversarial status: {e}")
    
    return status


def run_single_test(
    tester: Optional["AdversarialTester"] = None,
    tensor_size: int = 512,
) -> Dict[str, Any]:
    """
    Run a single adversarial test suite manually.
    
    Args:
        tester: AdversarialTester instance (uses singleton if None)
        tensor_size: Size of test tensor (default: 512)
        
    Returns:
        Test results dictionary
    """
    if tester is None:
        tester = get_adversarial_tester()
    
    if tester is None:
        return {"error": "AdversarialTester not available"}
    
    try:
        logger.info("🔒 Running manual adversarial test suite...")
        
        base_tensor = np.random.randn(tensor_size).astype(np.float32)
        
        results = tester.run_adversarial_suite(
            base_tensor=base_tensor,
            proposal={"id": "manual_test", "type": "manual_trigger"}
        )
        
        logger.info(f"✅ Manual test complete: {results.get('summary', {})}")
        return results
        
    except Exception as e:
        logger.error(f"Manual adversarial test failed: {e}")
        return {"error": str(e)}


# Cleanup function for shutdown
def shutdown_adversarial_tester() -> None:
    """Shutdown the adversarial tester and cleanup resources."""
    global _ADVERSARIAL_TESTER, _PERIODIC_RUNNING
    
    # Stop periodic testing
    stop_periodic_testing()
    
    # Close database connections
    with _ADVERSARIAL_LOCK:
        if _ADVERSARIAL_TESTER is not None:
            try:
                if hasattr(_ADVERSARIAL_TESTER, 'db_pool'):
                    _ADVERSARIAL_TESTER.db_pool.close_all()
                logger.info("AdversarialTester shutdown complete")
            except Exception as e:
                logger.error(f"Error during AdversarialTester shutdown: {e}")
            _ADVERSARIAL_TESTER = None
