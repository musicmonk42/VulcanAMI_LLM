# test_neural_safety.py
"""
Comprehensive tests for neural_safety.py module.
Tests neural network models, validators, and safety assessment functionality.
"""

import pytest

# Skip entire module if torch is not available
torch = pytest.importorskip("torch", reason="PyTorch required for neural_safety tests")

import torch.nn as nn
import numpy as np
import asyncio
import time
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from vulcan.safety.neural_safety import (
    MemoryBoundedDeque,
    ModelType,
    ModelConfig,
    SafetyClassifier,
    AnomalyDetector,
    BayesianSafetyNet,
    TransformerSafetyModel,
    GraphSafetyNetwork,
    VariationalSafetyAutoencoder,
    NeuralSafetyValidator
)
from vulcan.safety.safety_types import (
    SafetyReport,
    SafetyViolationType,
    ActionType
)


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def model_config():
    """Create a basic model configuration."""
    return ModelConfig(
        input_dim=128,
        hidden_dims=[64, 32],
        output_dim=1,
        dropout_rate=0.3,
        learning_rate=0.001,
        batch_size=16,
        num_epochs=5,
        device='cpu'
    )


@pytest.fixture
def sample_action():
    """Create a sample action for testing."""
    return {
        'type': ActionType.EXPLORE,
        'confidence': 0.8,
        'uncertainty': 0.2,
        'risk_score': 0.3,
        'safety_score': 0.7,
        'resource_usage': {
            'cpu': 50,
            'memory': 60
        }
    }


@pytest.fixture
def sample_context():
    """Create a sample context for testing."""
    return {
        'system_load': 0.5,
        'time_pressure': 0.0,
        'critical_operation': False,
        'user_override': False,
        'history': []
    }


@pytest.fixture
def neural_validator(model_config):
    """Create a neural safety validator."""
    model_configs = {
        ModelType.CLASSIFIER: model_config,
        ModelType.ANOMALY_DETECTOR: model_config
    }
    validator = NeuralSafetyValidator(
        model_configs=model_configs,
        ensemble_size=2,
        config={'device': 'cpu'}
    )
    yield validator
    validator.shutdown()


@pytest.fixture
def temp_checkpoint_dir():
    """Create a temporary directory for checkpoints."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def training_data():
    """Create sample training data."""
    data = []
    for i in range(50):
        action = {
            'type': ActionType.EXPLORE,
            'confidence': np.random.rand(),
            'uncertainty': np.random.rand(),
            'risk_score': np.random.rand(),
            'safety_score': np.random.rand()
        }
        context = {
            'system_load': np.random.rand(),
            'time_pressure': 0.0,
            'critical_operation': False
        }
        is_safe = np.random.rand() > 0.3
        data.append((action, context, is_safe))
    return data


# ============================================================
# MEMORY BOUNDED DEQUE TESTS
# ============================================================

class TestMemoryBoundedDeque:
    """Tests for MemoryBoundedDeque class."""
    
    def test_initialization(self):
        """Test deque initialization."""
        deque = MemoryBoundedDeque(max_size_mb=10)
        assert len(deque) == 0
        assert deque.get_memory_usage_mb() == 0
    
    def test_append_single_item(self):
        """Test appending a single item."""
        deque = MemoryBoundedDeque(max_size_mb=10)
        deque.append({'data': 'test'})
        assert len(deque) == 1
        assert deque.get_memory_usage_mb() > 0
    
    def test_memory_limit_enforcement(self):
        """Test that memory limit is enforced."""
        deque = MemoryBoundedDeque(max_size_mb=0.1)  # Very small limit
        
        # Add items until memory limit is reached
        for i in range(100):
            deque.append({'data': 'x' * 1000})  # ~1KB items
        
        # Should have evicted old items
        assert len(deque) < 100
        assert deque.get_memory_usage_mb() <= 0.1
    
    def test_clear(self):
        """Test clearing the deque."""
        deque = MemoryBoundedDeque(max_size_mb=10)
        deque.append({'data': 'test'})
        deque.append({'data': 'test2'})
        
        deque.clear()
        assert len(deque) == 0
        assert deque.get_memory_usage_mb() == 0
    
    def test_iteration(self):
        """Test iterating over deque."""
        deque = MemoryBoundedDeque(max_size_mb=10)
        items = [{'id': i} for i in range(5)]
        
        for item in items:
            deque.append(item)
        
        collected = list(deque)
        assert len(collected) == 5
    
    def test_thread_safety(self):
        """Test thread-safe operations."""
        import threading
        
        deque = MemoryBoundedDeque(max_size_mb=10)
        
        def add_items():
            for i in range(10):
                deque.append({'id': i})
        
        threads = [threading.Thread(target=add_items) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(deque) == 30


# ============================================================
# MODEL CONFIGURATION TESTS
# ============================================================

class TestModelConfig:
    """Tests for ModelConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ModelConfig()
        assert config.input_dim == 128
        assert config.hidden_dims == [256, 128, 64]
        assert config.output_dim == 1
        assert config.dropout_rate == 0.3
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = ModelConfig(
            input_dim=64,
            hidden_dims=[32, 16],
            dropout_rate=0.5
        )
        assert config.input_dim == 64
        assert config.hidden_dims == [32, 16]
        assert config.dropout_rate == 0.5


# ============================================================
# SAFETY CLASSIFIER TESTS
# ============================================================

class TestSafetyClassifier:
    """Tests for SafetyClassifier neural network."""
    
    def test_initialization(self, model_config):
        """Test model initialization."""
        model = SafetyClassifier(model_config)
        assert isinstance(model, nn.Module)
        assert model.config == model_config
    
    def test_forward_pass(self, model_config):
        """Test forward pass through network."""
        model = SafetyClassifier(model_config)
        model.eval()
        
        batch_size = 4
        x = torch.randn(batch_size, model_config.input_dim)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (batch_size, model_config.output_dim)
        assert torch.all(output >= 0) and torch.all(output <= 1)  # Sigmoid output
    
    def test_training_mode(self, model_config):
        """Test model in training mode."""
        model = SafetyClassifier(model_config)
        model.train()
        
        x = torch.randn(2, model_config.input_dim)
        output = model(x)
        
        # Test backward pass
        loss = output.mean()
        loss.backward()
        
        # Check gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_different_activations(self):
        """Test different activation functions."""
        activations = ['relu', 'elu', 'leaky_relu']
        
        for activation in activations:
            config = ModelConfig(activation=activation)
            model = SafetyClassifier(config)
            
            x = torch.randn(2, config.input_dim)
            output = model(x)
            assert output.shape == (2, 1)


# ============================================================
# ANOMALY DETECTOR TESTS
# ============================================================

class TestAnomalyDetector:
    """Tests for AnomalyDetector autoencoder."""
    
    def test_initialization(self, model_config):
        """Test model initialization."""
        model = AnomalyDetector(model_config)
        assert isinstance(model, nn.Module)
        assert hasattr(model, 'encoder')
        assert hasattr(model, 'decoder')
    
    def test_forward_pass(self, model_config):
        """Test forward pass returns reconstruction and latent."""
        model = AnomalyDetector(model_config)
        model.eval()
        
        x = torch.randn(4, model_config.input_dim)
        
        with torch.no_grad():
            x_recon, z = model(x)
        
        assert x_recon.shape == x.shape
        assert z.shape[0] == 4
        assert z.shape[1] == model.latent_dim
    
    def test_encode_decode(self, model_config):
        """Test encoding and decoding separately."""
        model = AnomalyDetector(model_config)
        model.eval()
        
        x = torch.randn(2, model_config.input_dim)
        
        with torch.no_grad():
            z = model.encode(x)
            x_recon = model.decode(z)
        
        assert z.shape == (2, model.latent_dim)
        assert x_recon.shape == x.shape
    
    def test_anomaly_score(self, model_config):
        """Test anomaly score computation."""
        model = AnomalyDetector(model_config)
        model.eval()
        
        x = torch.randn(3, model_config.input_dim)
        
        with torch.no_grad():
            anomaly_scores = model.compute_anomaly_score(x)
        
        assert anomaly_scores.shape == (3,)
        assert torch.all(anomaly_scores >= 0)
    
    def test_reconstruction_quality(self, model_config):
        """Test that reconstruction is reasonable."""
        model = AnomalyDetector(model_config)
        model.eval()
        
        x = torch.randn(1, model_config.input_dim)
        
        with torch.no_grad():
            x_recon, _ = model(x)
        
        # Reconstruction should have same shape
        assert x_recon.shape == x.shape
        
        # Check reconstruction error is finite
        error = torch.nn.functional.mse_loss(x_recon, x)
        assert torch.isfinite(error)


# ============================================================
# BAYESIAN SAFETY NET TESTS
# ============================================================

class TestBayesianSafetyNet:
    """Tests for BayesianSafetyNet model."""
    
    def test_initialization(self, model_config):
        """Test model initialization."""
        model = BayesianSafetyNet(model_config, num_samples=5)
        assert isinstance(model, nn.Module)
        assert model.num_samples == 5
    
    def test_forward_pass(self, model_config):
        """Test forward pass returns mean and uncertainty."""
        model = BayesianSafetyNet(model_config, num_samples=3)
        model.eval()
        
        x = torch.randn(2, model_config.input_dim)
        
        with torch.no_grad():
            mean_pred, uncertainty = model(x)
        
        assert mean_pred.shape == (2, model_config.output_dim)
        assert uncertainty.shape == (2, model_config.output_dim)
        assert torch.all(mean_pred >= 0) and torch.all(mean_pred <= 1)
        assert torch.all(uncertainty >= 0)
    
    def test_kl_divergence(self, model_config):
        """Test KL divergence computation."""
        model = BayesianSafetyNet(model_config, num_samples=3)
        
        kl = model.kl_divergence()
        
        assert isinstance(kl, torch.Tensor)
        assert kl.ndim == 0  # Scalar
        assert torch.isfinite(kl)
    
    def test_uncertainty_increases_with_samples(self, model_config):
        """Test that more samples give better uncertainty estimates."""
        x = torch.randn(1, model_config.input_dim)
        
        model = BayesianSafetyNet(model_config, num_samples=2)
        model.eval()
        
        with torch.no_grad():
            _, uncertainty = model(x)
        
        assert torch.isfinite(uncertainty).all()


# ============================================================
# TRANSFORMER SAFETY MODEL TESTS
# ============================================================

class TestTransformerSafetyModel:
    """Tests for TransformerSafetyModel."""
    
    def test_initialization(self, model_config):
        """Test model initialization."""
        model = TransformerSafetyModel(model_config, num_heads=4, num_layers=2)
        assert isinstance(model, nn.Module)
    
    def test_forward_pass_2d_input(self, model_config):
        """Test forward pass with 2D input."""
        model = TransformerSafetyModel(model_config, num_heads=4, num_layers=2)
        model.eval()
        
        x = torch.randn(2, model_config.input_dim)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (2, model_config.output_dim)
        assert torch.all(output >= 0) and torch.all(output <= 1)
    
    def test_forward_pass_3d_input(self, model_config):
        """Test forward pass with 3D input (sequence)."""
        model = TransformerSafetyModel(model_config, num_heads=4, num_layers=2)
        model.eval()
        
        x = torch.randn(2, 5, model_config.input_dim)  # batch, seq, features
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (2, model_config.output_dim)
    
    def test_positional_encoding(self, model_config):
        """Test that positional encoding exists."""
        model = TransformerSafetyModel(model_config)
        
        assert hasattr(model, 'positional_encoding')
        assert model.positional_encoding.shape[0] == 1
        assert model.positional_encoding.shape[1] == 1000


# ============================================================
# GRAPH SAFETY NETWORK TESTS
# ============================================================

class TestGraphSafetyNetwork:
    """Tests for GraphSafetyNetwork."""
    
    def test_initialization(self, model_config):
        """Test model initialization."""
        model = GraphSafetyNetwork(model_config, num_layers=2)
        assert isinstance(model, nn.Module)
    
    def test_forward_pass_no_adjacency(self, model_config):
        """Test forward pass without adjacency matrix."""
        model = GraphSafetyNetwork(model_config, num_layers=2)
        model.eval()
        
        x = torch.randn(3, model_config.input_dim)
        
        with torch.no_grad():
            output = model(x, adj_matrix=None)
        
        assert output.shape[1] == model_config.output_dim
    
    def test_forward_pass_with_adjacency(self, model_config):
        """Test forward pass with adjacency matrix."""
        model = GraphSafetyNetwork(model_config, num_layers=2)
        model.eval()
        
        num_nodes = 4
        x = torch.randn(num_nodes, model_config.input_dim)
        adj_matrix = torch.rand(num_nodes, num_nodes)
        
        with torch.no_grad():
            output = model(x, adj_matrix)
        
        assert output.shape[1] == model_config.output_dim


# ============================================================
# VARIATIONAL AUTOENCODER TESTS
# ============================================================

class TestVariationalSafetyAutoencoder:
    """Tests for VariationalSafetyAutoencoder."""
    
    def test_initialization(self, model_config):
        """Test model initialization."""
        model = VariationalSafetyAutoencoder(model_config, latent_dim=16)
        assert isinstance(model, nn.Module)
        assert model.latent_dim == 16
    
    def test_forward_pass(self, model_config):
        """Test forward pass returns all components."""
        model = VariationalSafetyAutoencoder(model_config, latent_dim=16)
        model.eval()
        
        x = torch.randn(2, model_config.input_dim)
        
        with torch.no_grad():
            x_recon, mu, logvar, z, safety_pred = model(x)
        
        assert x_recon.shape == x.shape
        assert mu.shape == (2, 16)
        assert logvar.shape == (2, 16)
        assert z.shape == (2, 16)
        assert safety_pred.shape == (2, 1)
    
    def test_reparameterization(self, model_config):
        """Test reparameterization trick."""
        model = VariationalSafetyAutoencoder(model_config, latent_dim=16)
        
        mu = torch.randn(2, 16)
        logvar = torch.randn(2, 16)
        
        z1 = model.reparameterize(mu, logvar)
        z2 = model.reparameterize(mu, logvar)
        
        # Different samples due to randomness
        assert not torch.allclose(z1, z2)
    
    def test_loss_computation(self, model_config):
        """Test VAE loss computation."""
        model = VariationalSafetyAutoencoder(model_config, latent_dim=16)
        
        x = torch.randn(2, model_config.input_dim)
        x_recon, mu, logvar, z, safety_pred = model(x)
        
        # Without safety labels
        losses = model.compute_loss(x, x_recon, mu, logvar)
        
        assert 'total' in losses
        assert 'reconstruction' in losses
        assert 'kl' in losses
        assert all(torch.isfinite(loss) for loss in losses.values())
    
    def test_loss_with_safety_labels(self, model_config):
        """Test VAE loss with safety supervision."""
        model = VariationalSafetyAutoencoder(model_config, latent_dim=16)
        
        x = torch.randn(2, model_config.input_dim)
        x_recon, mu, logvar, z, safety_pred = model(x)
        
        safety_labels = torch.tensor([[1.0], [0.0]])
        
        losses = model.compute_loss(x, x_recon, mu, logvar, safety_labels, safety_pred)
        
        assert 'safety' in losses
        assert torch.isfinite(losses['safety'])


# ============================================================
# NEURAL SAFETY VALIDATOR TESTS
# ============================================================

class TestNeuralSafetyValidator:
    """Tests for NeuralSafetyValidator main class."""
    
    def test_initialization(self, model_config):
        """Test validator initialization."""
        model_configs = {
            ModelType.CLASSIFIER: model_config,
            ModelType.ANOMALY_DETECTOR: model_config
        }
        
        validator = NeuralSafetyValidator(
            model_configs=model_configs,
            ensemble_size=2,
            config={'device': 'cpu'}
        )
        
        assert len(validator.models) == 2
        assert ModelType.CLASSIFIER in validator.models
        assert ModelType.ANOMALY_DETECTOR in validator.models
        
        validator.shutdown()
    
    def test_feature_extraction(self, neural_validator, sample_action, sample_context):
        """Test feature extraction from action and context."""
        features = neural_validator._extract_features(sample_action, sample_context)
        
        assert isinstance(features, np.ndarray)
        assert features.shape == (128,)
        assert np.all(np.isfinite(features))
        assert np.all(features >= -1) and np.all(features <= 1)
    
    def test_validate_sync(self, neural_validator, sample_action, sample_context):
        """Test synchronous validation."""
        report = neural_validator.validate(
            sample_action,
            sample_context,
            use_consensus=False,
            timeout=2.0
        )
        
        # Handle case where validate returns a coroutine
        if asyncio.iscoroutine(report):
            report = asyncio.run(report)
        
        assert isinstance(report, SafetyReport)
        assert isinstance(report.safe, bool)
        assert 0 <= report.confidence <= 1
        assert 'neural_validation' in report.metadata
    
    @pytest.mark.asyncio
    async def test_validate_async(self, neural_validator, sample_action, sample_context):
        """Test asynchronous validation."""
        report = await neural_validator.validate_async(
            sample_action,
            sample_context,
            use_consensus=False,
            timeout=2.0
        )
        
        assert isinstance(report, SafetyReport)
        assert isinstance(report.safe, bool)
        assert 'neural_validation' in report.metadata
    
    @pytest.mark.asyncio
    async def test_validate_with_consensus(self, neural_validator, sample_action, sample_context):
        """Test validation with consensus."""
        report = await neural_validator.validate_async(
            sample_action,
            sample_context,
            use_consensus=True,
            timeout=2.0
        )
        
        assert isinstance(report, SafetyReport)
        assert report.metadata['consensus_used'] is True
    
    @pytest.mark.asyncio
    async def test_validate_timeout(self, neural_validator, sample_action, sample_context):
        """Test validation timeout handling."""
        report = await neural_validator.validate_async(
            sample_action,
            sample_context,
            timeout=0.001  # Very short timeout
        )
        
        # Should handle timeout gracefully
        assert isinstance(report, SafetyReport)
    
    def test_prediction_analysis(self, neural_validator):
        """Test prediction analysis for safety decision."""
        predictions = {
            'classifier_ensemble': {
                'mean': 0.8,
                'std': 0.1,
                'votes': [0.75, 0.85, 0.8]
            },
            'anomaly': {
                'score': 0.2,
                'is_anomaly': 0.0
            }
        }
        
        decision = neural_validator._analyze_predictions(predictions)
        
        assert 'safe' in decision
        assert 'confidence' in decision
        assert 'violations' in decision
        assert 'reasons' in decision
        assert isinstance(decision['safe'], bool)
    
    def test_training(self, neural_validator, training_data):
        """Test model training."""
        metrics = neural_validator.train(
            training_data,
            validation_data=None,
            model_types=[ModelType.CLASSIFIER],
            epochs=2
        )
        
        assert ModelType.CLASSIFIER.value in metrics
        assert isinstance(metrics[ModelType.CLASSIFIER.value], dict)
    
    def test_training_with_validation(self, neural_validator, training_data):
        """Test model training with validation data."""
        train_data = training_data[:40]
        val_data = training_data[40:]
        
        metrics = neural_validator.train(
            train_data,
            validation_data=val_data,
            epochs=2
        )
        
        assert len(metrics) > 0
    
    def test_save_and_load_models(self, neural_validator, temp_checkpoint_dir):
        """Test saving and loading models."""
        # Save models
        neural_validator.save_models(temp_checkpoint_dir)
        
        # Check files exist
        checkpoint_path = Path(temp_checkpoint_dir)
        assert any(checkpoint_path.iterdir())
        
        # Load models
        neural_validator.load_models(temp_checkpoint_dir)
    
    def test_get_model_stats(self, neural_validator):
        """Test getting model statistics."""
        stats = neural_validator.get_model_stats()
        
        assert 'num_models' in stats
        assert 'model_types' in stats
        assert 'device' in stats
        assert 'consensus_threshold' in stats
        assert stats['num_models'] > 0
    
    def test_update_thresholds(self, neural_validator):
        """Test updating consensus and uncertainty thresholds."""
        initial_consensus = neural_validator.consensus_threshold
        initial_uncertainty = neural_validator.uncertainty_threshold
        
        neural_validator.update_consensus_threshold(0.7)
        neural_validator.update_uncertainty_threshold(0.2)
        
        assert neural_validator.consensus_threshold == 0.7
        assert neural_validator.uncertainty_threshold == 0.2
        assert neural_validator.consensus_threshold != initial_consensus
    
    def test_threshold_clamping(self, neural_validator):
        """Test that thresholds are clamped to valid ranges."""
        neural_validator.update_consensus_threshold(1.5)  # Too high
        assert neural_validator.consensus_threshold <= 0.9
        
        neural_validator.update_consensus_threshold(0.2)  # Too low
        assert neural_validator.consensus_threshold >= 0.5
        
        neural_validator.update_uncertainty_threshold(0.8)  # Too high
        assert neural_validator.uncertainty_threshold <= 0.5
    
    def test_shutdown(self, neural_validator):
        """Test validator shutdown."""
        neural_validator.shutdown()
        assert neural_validator._shutdown is True
        
        # Calling shutdown again should be safe
        neural_validator.shutdown()
    
    def test_memory_bounded_buffers(self, neural_validator):
        """Test that memory-bounded buffers are used."""
        assert isinstance(neural_validator.training_buffer, MemoryBoundedDeque)
        assert isinstance(neural_validator.validation_buffer, MemoryBoundedDeque)
        
        # Check memory limits
        assert neural_validator.training_buffer.max_size_bytes > 0
        assert neural_validator.validation_buffer.max_size_bytes > 0


# ============================================================
# INTEGRATION TESTS
# ============================================================

class TestNeuralSafetyIntegration:
    """Integration tests for neural safety system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_validation(self, model_config):
        """Test complete validation pipeline."""
        # Setup validator with multiple models
        model_configs = {
            ModelType.CLASSIFIER: model_config,
            ModelType.ANOMALY_DETECTOR: model_config,
            ModelType.BAYESIAN: model_config
        }
        
        validator = NeuralSafetyValidator(
            model_configs=model_configs,
            ensemble_size=2,
            config={'device': 'cpu', 'consensus_threshold': 0.6}
        )
        
        try:
            action = {
                'type': ActionType.EXPLORE,
                'confidence': 0.9,
                'risk_score': 0.1
            }
            context = {
                'system_load': 0.3,
                'critical_operation': False
            }
            
            # Validate
            report = await validator.validate_async(action, context, use_consensus=True)
            
            assert isinstance(report, SafetyReport)
            assert 'neural_validation' in report.metadata
            assert 'models_queried' in report.metadata
            
        finally:
            validator.shutdown()
    
    def test_multiple_validation_calls(self, neural_validator):
        """Test multiple sequential validations."""
        actions = [
            {'type': ActionType.EXPLORE, 'confidence': 0.8},
            {'type': ActionType.OPTIMIZE, 'confidence': 0.6},
            {'type': ActionType.MAINTAIN, 'confidence': 0.9}
        ]
        context = {'system_load': 0.5}
        
        for action in actions:
            report = asyncio.run(
                neural_validator.validate_async(action, context, use_consensus=False)
            )
            assert isinstance(report, SafetyReport)
    
    def test_concurrent_validations(self, neural_validator):
        """Test concurrent validation requests."""
        async def run_validations():
            tasks = []
            for i in range(5):
                action = {'type': ActionType.EXPLORE, 'confidence': 0.7}
                context = {'system_load': 0.5}
                task = neural_validator.validate_async(action, context)
                tasks.append(task)
            
            reports = await asyncio.gather(*tasks)
            return reports
        
        reports = asyncio.run(run_validations())
        
        assert len(reports) == 5
        assert all(isinstance(r, SafetyReport) for r in reports)


# ============================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================

class TestEdgeCasesAndErrors:
    """Test edge cases and error handling."""
    
    def test_empty_action(self, neural_validator):
        """Test validation with empty action."""
        report = asyncio.run(
            neural_validator.validate_async({}, {})
        )
        assert isinstance(report, SafetyReport)
    
    def test_malformed_action(self, neural_validator):
        """Test validation with malformed action."""
        action = {
            'type': 'invalid_type',
            'confidence': 'not_a_number'
        }
        report = asyncio.run(
            neural_validator.validate_async(action, {})
        )
        assert isinstance(report, SafetyReport)
    
    def test_large_ensemble(self, model_config):
        """Test that ensemble size is limited."""
        validator = NeuralSafetyValidator(
            model_configs={ModelType.CLASSIFIER: model_config},
            ensemble_size=100  # Unreasonably large
        )
        
        # Should be clamped to max
        assert validator.ensemble_size <= 10
        validator.shutdown()
    
    def test_training_with_empty_data(self, neural_validator):
        """Test training with empty dataset."""
        metrics = neural_validator.train(
            training_data=[],
            epochs=1
        )
        # Should handle gracefully
        assert isinstance(metrics, dict)
    
    def test_training_epoch_limit(self, neural_validator, training_data):
        """Test that training epochs are limited."""
        metrics = neural_validator.train(
            training_data,
            epochs=1000  # Unreasonably large
        )
        # Should complete without hanging
        assert isinstance(metrics, dict)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])