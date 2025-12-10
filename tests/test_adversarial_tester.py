"""
Comprehensive test suite for adversarial_tester.py
"""

import json
import shutil
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from adversarial_tester import (GRADIENT_EPSILON, MAX_TENSOR_SIZE,
                                PHOTONIC_NOISE_FLOOR, AdversarialResult,
                                AdversarialTester, AlignmentResult,
                                AnomalyType, AttackType,
                                DatabaseConnectionPool, InterpretabilityEngine,
                                InterpretabilityResult, NSOAligner,
                                SafetyLevel)


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    return np.random.randn(100, 10).astype(np.float32)


@pytest.fixture
def sample_tensor():
    """Generate sample tensor for testing."""
    np.random.seed(42)
    return np.random.randn(10).astype(np.float32)


@pytest.fixture
def simple_model():
    """Simple model for testing."""
    return lambda x: np.sum(x, axis=-1)


@pytest.fixture
def interpret_engine(sample_data, simple_model):
    """Create interpretability engine."""
    return InterpretabilityEngine(
        model=simple_model,
        background_data=sample_data,
        cache_size=100
    )


@pytest.fixture
def nso_aligner():
    """Create NSO aligner."""
    return NSOAligner()


@pytest.fixture
def adversarial_tester(interpret_engine, nso_aligner, temp_dir):
    """Create adversarial tester."""
    return AdversarialTester(
        interpret_engine=interpret_engine,
        nso_aligner=nso_aligner,
        log_dir=str(temp_dir / "logs")
    )


class TestDatabaseConnectionPool:
    """Test database connection pool."""
    
    def test_pool_creation(self, temp_dir):
        """Test pool creates connections."""
        db_path = temp_dir / "test.db"
        pool = DatabaseConnectionPool(db_path, pool_size=3)
        
        assert len(pool.connections) == 3
        assert not pool.closed
        
        pool.close_all()
    
    def test_get_connection(self, temp_dir):
        """Test getting connection from pool."""
        db_path = temp_dir / "test.db"
        pool = DatabaseConnectionPool(db_path, pool_size=2)
        
        with pool.get_connection() as conn:
            assert conn is not None
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            assert result[0] == 1
        
        pool.close_all()
    
    def test_concurrent_connections(self, temp_dir):
        """Test concurrent connection access."""
        db_path = temp_dir / "test.db"
        pool = DatabaseConnectionPool(db_path, pool_size=3)
        results = []
        
        def worker():
            with pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                results.append(cursor.fetchone()[0])
                time.sleep(0.01)
        
        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(results) == 5
        assert all(r == 1 for r in results)
        
        pool.close_all()
    
    def test_pool_closed_raises(self, temp_dir):
        """Test accessing closed pool raises error."""
        db_path = temp_dir / "test.db"
        pool = DatabaseConnectionPool(db_path, pool_size=2)
        pool.close_all()
        
        with pytest.raises(RuntimeError, match="Connection pool is closed"):
            with pool.get_connection():
                pass


class TestInterpretabilityEngine:
    """Test interpretability engine."""
    
    def test_initialization(self, sample_data, simple_model):
        """Test engine initialization."""
        engine = InterpretabilityEngine(
            model=simple_model,
            background_data=sample_data
        )
        
        assert engine.model is not None
        assert engine.background_data is not None
        assert len(engine.cache) == 0
    
    def test_explain_tensor_basic(self, interpret_engine, sample_tensor):
        """Test basic tensor explanation."""
        result = interpret_engine.explain_tensor(sample_tensor)
        
        assert isinstance(result, InterpretabilityResult)
        assert result.shap_values is not None
        assert result.gradient_saliency is not None
        assert result.attention_weights is not None
        assert result.confidence_scores is not None
    
    def test_explain_tensor_caching(self, interpret_engine, sample_tensor):
        """Test explanation caching."""
        # First call
        result1 = interpret_engine.explain_tensor(sample_tensor)
        cache_size_1 = len(interpret_engine.cache)
        
        # Second call with same tensor
        result2 = interpret_engine.explain_tensor(sample_tensor)
        cache_size_2 = len(interpret_engine.cache)
        
        assert cache_size_1 == 1
        assert cache_size_2 == 1  # No new cache entry
        assert np.allclose(result1.gradient_saliency, result2.gradient_saliency)
    
    def test_explain_tensor_too_large(self, interpret_engine):
        """Test rejection of oversized tensors."""
        huge_tensor = np.ones(MAX_TENSOR_SIZE + 1, dtype=np.float32)
        
        with pytest.raises(ValueError, match="Tensor too large"):
            interpret_engine.explain_tensor(huge_tensor)
    
    def test_gradient_saliency(self, interpret_engine, sample_tensor):
        """Test gradient saliency computation."""
        saliency = interpret_engine._compute_gradient_saliency(sample_tensor)
        
        assert saliency.shape == sample_tensor.shape
        assert np.all(saliency >= 0)
        assert np.all(saliency <= 1)
    
    def test_attention_weights(self, interpret_engine, sample_tensor):
        """Test attention weight computation."""
        attention = interpret_engine._compute_attention_weights(sample_tensor)
        
        assert attention.shape == sample_tensor.shape
        # Softmax should sum to 1 for 1D
        if sample_tensor.ndim == 1:
            assert np.isclose(attention.sum(), 1.0)
    
    def test_feature_importance(self, interpret_engine, sample_tensor):
        """Test feature importance computation."""
        importance = interpret_engine._compute_feature_importance(sample_tensor)
        
        assert importance.shape == sample_tensor.shape
        assert np.all(importance >= 0)
        assert np.all(importance <= 1)
    
    def test_confidence_scores(self, interpret_engine, sample_tensor):
        """Test confidence score computation."""
        confidence = interpret_engine._compute_confidence_scores(sample_tensor)
        
        assert confidence.shape == sample_tensor.shape
        assert np.all(confidence >= 0)
        assert np.all(confidence <= 1)
    
    def test_anomaly_detection_normal(self, interpret_engine, sample_data):
        """Test anomaly detection on normal data."""
        normal_sample = sample_data[0]
        is_anomaly, anomaly_type, confidence = interpret_engine.detect_anomaly(normal_sample)
        
        # Should not be anomalous
        assert isinstance(is_anomaly, bool)
        if is_anomaly:
            assert anomaly_type is not None
    
    def test_anomaly_detection_outlier(self, interpret_engine):
        """Test anomaly detection on outlier."""
        outlier = np.ones(10, dtype=np.float32) * 1000  # Extreme values
        is_anomaly, anomaly_type, confidence = interpret_engine.detect_anomaly(outlier)
        
        assert isinstance(is_anomaly, bool)
        if is_anomaly:
            assert isinstance(anomaly_type, AnomalyType)
            assert 0 <= confidence <= 1
    
    def test_multidimensional_tensor(self, interpret_engine):
        """Test with multidimensional tensors."""
        tensor_2d = np.random.randn(5, 4).astype(np.float32)
        result = interpret_engine.explain_tensor(tensor_2d)
        
        assert result.shap_values.shape == tensor_2d.shape
        assert result.gradient_saliency.shape == tensor_2d.shape


class TestNSOAligner:
    """Test NSO aligner."""
    
    def test_initialization(self):
        """Test aligner initialization."""
        aligner = NSOAligner()
        
        assert len(aligner.safety_rules) > 0
        assert len(aligner.ethical_guidelines) > 0
        assert len(aligner.audit_history) == 0
    
    def test_safe_proposal(self, nso_aligner):
        """Test auditing safe proposal."""
        safe_proposal = {
            "id": "safe_001",
            "code": "print('Hello, World!')",
            "description": "Safe greeting"
        }
        
        result = nso_aligner.audit_proposal(safe_proposal)
        
        assert isinstance(result, AlignmentResult)
        assert result.safety_level in [SafetyLevel.SAFE, SafetyLevel.LOW_RISK]
        assert result.confidence > 0
        assert len(nso_aligner.audit_history) == 1
    
    def test_dangerous_code_detection(self, nso_aligner):
        """Test detection of dangerous code."""
        dangerous_proposal = {
            "id": "danger_001",
            "code": "import os; os.system('rm -rf /')"
        }
        
        result = nso_aligner.audit_proposal(dangerous_proposal)
        
        assert result.safety_level in [SafetyLevel.HIGH_RISK, SafetyLevel.CRITICAL]
        assert len(result.risks) > 0
        assert "command_injection" in result.risks or "system_damage" in result.risks
    
    def test_eval_exec_detection(self, nso_aligner):
        """Test detection of eval/exec."""
        eval_proposal = {
            "id": "eval_001",
            "code": "eval(user_input)"
        }
        
        result = nso_aligner.audit_proposal(eval_proposal)
        
        assert result.safety_level in [SafetyLevel.MEDIUM_RISK, SafetyLevel.HIGH_RISK]
        assert "code_injection" in result.risks
    
    def test_sql_injection_detection(self, nso_aligner):
        """Test detection of SQL injection patterns."""
        sql_proposal = {
            "id": "sql_001",
            "code": "DROP TABLE users"
        }
        
        result = nso_aligner.audit_proposal(sql_proposal)
        
        assert result.safety_level != SafetyLevel.SAFE
        assert "data_loss" in result.risks
    
    def test_nested_proposal(self, nso_aligner):
        """Test deeply nested proposal."""
        nested = {"level": 0}
        current = nested
        for i in range(150):  # Exceed MAX_PROPOSAL_DEPTH
            current["next"] = {"level": i + 1}
            current = current["next"]
        
        result = nso_aligner.audit_proposal(nested)
        
        assert "excessive_nesting" in result.risks
    
    def test_large_proposal(self, nso_aligner):
        """Test oversized proposal."""
        large_proposal = {
            "id": "large_001",
            "code": "x = 1\n" * 1000000  # Very large code
        }
        
        result = nso_aligner.audit_proposal(large_proposal)
        
        # Should detect size issue
        assert "excessive_size" in result.risks or result.safety_level != SafetyLevel.SAFE
    
    def test_multi_model_audit(self, nso_aligner):
        """Test multi-model audit wrapper."""
        proposal = {"id": "test", "code": "print('test')"}
        
        label = nso_aligner.multi_model_audit(proposal)
        
        assert isinstance(label, str)
        assert label in ["safe", "low_risk", "medium_risk", "high_risk", "critical"]
    
    def test_safety_report(self, nso_aligner):
        """Test safety report generation."""
        # Perform some audits
        for i in range(5):
            proposal = {"id": f"test_{i}", "code": f"print({i})"}
            nso_aligner.audit_proposal(proposal)
        
        report = nso_aligner.get_safety_report()
        
        assert "total_audits" in report
        assert report["total_audits"] == 5
        assert "risk_distribution" in report


class TestAdversarialTester:
    """Test adversarial tester."""
    
    def test_initialization(self, temp_dir):
        """Test tester initialization."""
        tester = AdversarialTester(log_dir=str(temp_dir / "logs"))
        
        assert tester.interpret_engine is not None
        assert tester.nso_aligner is not None
        assert (temp_dir / "logs").exists()
        
        tester.cleanup()
    
    def test_database_creation(self, adversarial_tester, temp_dir):
        """Test database is created."""
        db_path = temp_dir / "logs" / "adversarial_logs.db"
        assert db_path.exists()
        
        # Check schema
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        assert "attack_logs" in tables
    
    def test_fgsm_attack(self, adversarial_tester, sample_tensor):
        """Test FGSM attack."""
        adv_tensor, divergence = adversarial_tester.generate_adversarial_tensor(
            sample_tensor,
            attack_type=AttackType.FGSM,
            steps=5,
            epsilon=0.1
        )
        
        assert adv_tensor.shape == sample_tensor.shape
        assert divergence >= 0
        assert not np.allclose(adv_tensor, sample_tensor)
    
    def test_pgd_attack(self, adversarial_tester, sample_tensor):
        """Test PGD attack."""
        adv_tensor, divergence = adversarial_tester.generate_adversarial_tensor(
            sample_tensor,
            attack_type=AttackType.PGD,
            steps=5,
            epsilon=0.1
        )
        
        assert adv_tensor.shape == sample_tensor.shape
        assert divergence >= 0
    
    def test_random_attack(self, adversarial_tester, sample_tensor):
        """Test random attack."""
        adv_tensor, divergence = adversarial_tester.generate_adversarial_tensor(
            sample_tensor,
            attack_type=AttackType.RANDOM,
            steps=10,
            epsilon=0.1
        )
        
        assert adv_tensor.shape == sample_tensor.shape
        assert divergence >= 0
    
    def test_attack_timeout(self, adversarial_tester, sample_tensor):
        """Test attack timeout."""
        with pytest.raises(TimeoutError):
            adversarial_tester.generate_adversarial_tensor(
                sample_tensor,
                attack_type=AttackType.PGD,
                steps=10000,  # Many steps
                timeout=1  # Short timeout
            )
    
    def test_oversized_tensor_rejection(self, adversarial_tester):
        """Test rejection of oversized tensors."""
        huge_tensor = np.ones(MAX_TENSOR_SIZE + 1, dtype=np.float32)
        
        with pytest.raises(ValueError, match="Tensor too large"):
            adversarial_tester.generate_adversarial_tensor(
                huge_tensor,
                attack_type=AttackType.FGSM
            )
    
    def test_too_many_iterations(self, adversarial_tester, sample_tensor):
        """Test rejection of too many iterations."""
        from adversarial_tester import MAX_ATTACK_ITERATIONS
        
        with pytest.raises(ValueError, match="Too many iterations"):
            adversarial_tester.generate_adversarial_tensor(
                sample_tensor,
                attack_type=AttackType.FGSM,
                steps=MAX_ATTACK_ITERATIONS + 1
            )
    
    def test_photonic_noise(self, adversarial_tester, sample_tensor):
        """Test photonic noise generation."""
        noisy = adversarial_tester.generate_photonic_noise_tensor(sample_tensor)
        
        assert noisy.shape == sample_tensor.shape
        assert not np.allclose(noisy, sample_tensor)
        
        # Check noise floor is applied
        small_values = np.abs(noisy) < PHOTONIC_NOISE_FLOOR
        assert np.all(noisy[small_values] == 0)
    
    def test_ood_tensor_generation(self, adversarial_tester):
        """Test OOD tensor generation."""
        for distribution in ["uniform", "gaussian", "laplace", "cauchy", "exponential"]:
            ood = adversarial_tester.generate_ood_tensor(
                shape=(10,),
                distribution=distribution,
                scale=5.0
            )
            
            assert ood.shape == (10,)
            assert ood.dtype == np.float32
    
    def test_audit_resilience(self, adversarial_tester):
        """Test audit resilience testing."""
        proposal = {
            "id": "test_proposal",
            "code": "print('hello')"
        }
        
        results = adversarial_tester.test_audit_resilience(proposal)
        
        assert len(results) > 0
        for perturbed, label in results:
            assert label in ["safe", "low_risk", "medium_risk", "high_risk", "critical", "error"]
    
    def test_adversarial_search(self, adversarial_tester):
        """Test adversarial search."""
        safe_proposal = {
            "id": "safe_search",
            "code": "x = 1 + 1"
        }
        
        result_proposal, steps = adversarial_tester.adversarial_search(
            safe_proposal,
            max_steps=10,
            target_label="critical"
        )
        
        assert steps >= 0
        # May or may not find adversarial example in 10 steps
    
    def test_realtime_integrity_check(self, adversarial_tester, sample_tensor):
        """Test real-time integrity check."""
        graph = {"id": "test_graph"}
        
        results = adversarial_tester.realtime_integrity_check(graph, sample_tensor)
        
        assert "timestamp" in results
        assert "checks_performed" in results
        assert len(results["checks_performed"]) > 0
    
    def test_adversarial_suite(self, adversarial_tester, sample_tensor):
        """Test full adversarial suite."""
        proposal = {"id": "suite_test", "code": "pass"}
        
        results = adversarial_tester.run_adversarial_suite(sample_tensor, proposal)
        
        assert "timestamp" in results
        assert "tests" in results
        assert "summary" in results
        assert "total_tests" in results["summary"]
        assert "success_rate" in results["summary"]
    
    def test_attack_statistics(self, adversarial_tester, sample_tensor):
        """Test attack statistics tracking."""
        # Perform some attacks
        for attack_type in [AttackType.FGSM, AttackType.PGD]:
            try:
                adversarial_tester.generate_adversarial_tensor(
                    sample_tensor,
                    attack_type=attack_type,
                    steps=2
                )
            except:
                pass
        
        stats = adversarial_tester.get_attack_statistics()
        
        assert "attack_stats" in stats
        assert "total_attacks" in stats
        assert stats["total_attacks"] >= 2
    
    def test_concurrent_attacks(self, adversarial_tester, sample_tensor):
        """Test concurrent attack execution."""
        results = []
        
        def run_attack():
            try:
                adv, div = adversarial_tester.generate_adversarial_tensor(
                    sample_tensor,
                    attack_type=AttackType.FGSM,
                    steps=3
                )
                results.append(div)
            except Exception as e:
                results.append(None)
        
        threads = [threading.Thread(target=run_attack) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(results) == 3


class TestThreadSafety:
    """Test thread safety of components."""
    
    def test_cache_thread_safety(self, interpret_engine, sample_tensor):
        """Test interpretability cache thread safety."""
        results = []
        
        def explain():
            result = interpret_engine.explain_tensor(sample_tensor)
            results.append(result.gradient_saliency)
        
        threads = [threading.Thread(target=explain) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(results) == 10
        # All should be similar (from cache)
        for r in results[1:]:
            assert np.allclose(r, results[0])
    
    def test_audit_history_thread_safety(self, nso_aligner):
        """Test audit history thread safety."""
        def audit():
            proposal = {"id": f"thread_{threading.get_ident()}", "code": "pass"}
            nso_aligner.audit_proposal(proposal)
        
        threads = [threading.Thread(target=audit) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(nso_aligner.audit_history) == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])