# neural_safety.py
"""
Neural network-based safety validation for VULCAN-AGI Safety Module.
Implements multi-model consensus, uncertainty quantification, and real-time safety assessment.
"""

import asyncio
import atexit
import gc
import logging
import os
import pickle
import sys
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .safety_types import ActionType, SafetyReport, SafetyViolationType

logger = logging.getLogger(__name__)

_NEURAL_SAFETY_INIT_DONE = False

# ============================================================
# MEMORY BOUNDED DEQUE
# ============================================================


class MemoryBoundedDeque:
    """Deque with memory size limit instead of item count limit."""

    def __init__(self, max_size_mb: float = 100):
        """
        Initialize memory-bounded deque.

        Args:
            max_size_mb: Maximum memory size in megabytes
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.deque = deque()
        self.current_size_bytes = 0
        self.lock = threading.RLock()

    def append(self, item: Any):
        """Append item, removing old items if memory limit exceeded."""
        with self.lock:
            # Estimate item size
            try:
                item_size = sys.getsizeof(pickle.dumps(item))
            except Exception as e:
                logger.warning(f"Could not serialize item for size estimation: {e}")
                item_size = 1024  # Assume 1KB default

            # Remove old items if necessary
            while (
                self.current_size_bytes + item_size > self.max_size_bytes and self.deque
            ):
                old_item = self.deque.popleft()
                try:
                    old_size = sys.getsizeof(pickle.dumps(old_item))
                    self.current_size_bytes -= old_size
                except Exception:
                    self.current_size_bytes -= 1024  # Estimate

            # Add new item
            self.deque.append(item)
            self.current_size_bytes += item_size

    def __len__(self):
        with self.lock:
            return len(self.deque)

    def __iter__(self):
        with self.lock:
            return iter(list(self.deque))

    def clear(self):
        """Clear all items."""
        with self.lock:
            self.deque.clear()
            self.current_size_bytes = 0

    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        with self.lock:
            return self.current_size_bytes / (1024 * 1024)


# ============================================================
# NEURAL SAFETY MODELS
# ============================================================


class ModelType(Enum):
    """Types of neural safety models."""

    CLASSIFIER = "classifier"
    ANOMALY_DETECTOR = "anomaly_detector"
    RISK_PREDICTOR = "risk_predictor"
    ENSEMBLE = "ensemble"
    BAYESIAN = "bayesian"
    TRANSFORMER = "transformer"
    GNN = "graph_neural_network"
    VAE = "variational_autoencoder"


@dataclass
class ModelConfig:
    """Configuration for neural safety models."""

    input_dim: int = 128
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    output_dim: int = 1
    dropout_rate: float = 0.3
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_batch_norm: bool = True
    activation: str = "relu"
    optimizer_type: str = "adam"
    weight_decay: float = 1e-5
    early_stopping_patience: int = 10


class SafetyClassifier(nn.Module):
    """
    Deep neural network for safety classification.
    """

    def __init__(self, config: ModelConfig):
        super(SafetyClassifier, self).__init__()
        self.config = config

        # Build layers
        layers = []
        input_dim = config.input_dim

        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))

            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            # Activation
            if config.activation == "relu":
                layers.append(nn.ReLU())
            elif config.activation == "elu":
                layers.append(nn.ELU())
            elif config.activation == "leaky_relu":
                layers.append(nn.LeakyReLU())

            if config.dropout_rate > 0:
                layers.append(nn.Dropout(config.dropout_rate))

            input_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(input_dim, config.output_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return torch.sigmoid(self.network(x))


class AnomalyDetector(nn.Module):
    """
    Autoencoder-based anomaly detection model.
    """

    def __init__(self, config: ModelConfig):
        super(AnomalyDetector, self).__init__()
        self.config = config

        # Encoder
        encoder_layers = []
        input_dim = config.input_dim

        for hidden_dim in config.hidden_dims:
            encoder_layers.append(nn.Linear(input_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())

            if config.dropout_rate > 0:
                encoder_layers.append(nn.Dropout(config.dropout_rate))

            input_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Latent dimension
        self.latent_dim = config.hidden_dims[-1] // 2
        self.encoder_output = nn.Linear(config.hidden_dims[-1], self.latent_dim)

        # Decoder
        decoder_layers = []
        decoder_layers.append(nn.Linear(self.latent_dim, config.hidden_dims[-1]))
        decoder_layers.append(nn.ReLU())

        for i in range(len(config.hidden_dims) - 2, -1, -1):
            decoder_layers.append(
                nn.Linear(config.hidden_dims[i + 1], config.hidden_dims[i])
            )
            decoder_layers.append(nn.ReLU())

            if config.dropout_rate > 0:
                decoder_layers.append(nn.Dropout(config.dropout_rate))

        decoder_layers.append(nn.Linear(config.hidden_dims[0], config.input_dim))

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        encoded = self.encoder(x)
        return self.encoder_output(encoded)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to output."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returns reconstruction and latent representation."""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

    def compute_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute anomaly score as reconstruction error."""
        x_recon, _ = self.forward(x)
        reconstruction_error = F.mse_loss(x_recon, x, reduction="none").mean(dim=1)
        return reconstruction_error


class BayesianSafetyNet(nn.Module):
    """
    Bayesian neural network for uncertainty-aware safety prediction.
    """

    def __init__(self, config: ModelConfig, num_samples: int = 10):
        super(BayesianSafetyNet, self).__init__()
        self.config = config
        self.num_samples = num_samples

        # Variational layers
        self.layers = nn.ModuleList()
        input_dim = config.input_dim

        for hidden_dim in config.hidden_dims:
            self.layers.append(self._create_bayesian_layer(input_dim, hidden_dim))
            input_dim = hidden_dim

        self.output_layer = self._create_bayesian_layer(input_dim, config.output_dim)

    def _create_bayesian_layer(self, in_features: int, out_features: int) -> nn.Module:
        """Create a Bayesian linear layer with weight uncertainty."""

        class BayesianLinear(nn.Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                self.in_features = in_features
                self.out_features = out_features

                # Weight parameters
                self.weight_mu = nn.Parameter(
                    torch.randn(out_features, in_features) * 0.1
                )
                self.weight_sigma = nn.Parameter(
                    torch.ones(out_features, in_features) * -3.0
                )  # Init rho for small sigma

                # Bias parameters
                self.bias_mu = nn.Parameter(torch.zeros(out_features))
                self.bias_sigma = nn.Parameter(
                    torch.ones(out_features) * -3.0
                )  # Init rho for small sigma

            def forward(self, x):
                # Sample weights and biases
                # sigma = softplus(rho) = log(1 + exp(rho))
                weight_std = torch.log1p(torch.exp(self.weight_sigma))
                bias_std = torch.log1p(torch.exp(self.bias_sigma))

                # Sample eps
                weight_eps = torch.randn_like(weight_std)
                bias_eps = torch.randn_like(bias_std)

                weight = self.weight_mu + weight_eps * weight_std
                bias = self.bias_mu + bias_eps * bias_std

                return F.linear(x, weight, bias)

            def kl_divergence(self):
                """
                Compute KL divergence for regularization.
                KL(q(w) || p(w)) where q ~ N(mu, sigma^2) and p ~ N(0, 1).
                KL = 0.5 * sum(sigma^2 + mu^2 - 1 - log(sigma^2))
                """
                # sigma = softplus(rho)
                weight_std = torch.log1p(torch.exp(self.weight_sigma))
                # sigma^2
                weight_var = weight_std.pow(2)
                # log(sigma^2) = 2 * log(sigma)
                # Add a small epsilon for numerical stability in log
                weight_log_var = 2 * torch.log(weight_std + 1e-9)

                weight_kl = 0.5 * torch.sum(
                    weight_var + self.weight_mu.pow(2) - 1 - weight_log_var
                )

                # Do the same for bias
                bias_std = torch.log1p(torch.exp(self.bias_sigma))
                bias_var = bias_std.pow(2)
                bias_log_var = 2 * torch.log(bias_std + 1e-9)

                bias_kl = 0.5 * torch.sum(
                    bias_var + self.bias_mu.pow(2) - 1 - bias_log_var
                )
                return weight_kl + bias_kl

        return BayesianLinear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with uncertainty estimation."""
        predictions = []

        for _ in range(self.num_samples):
            h = x
            for layer in self.layers:
                h = F.relu(layer(h))
                if self.config.dropout_rate > 0:
                    h = F.dropout(h, self.config.dropout_rate, self.training)

            output = torch.sigmoid(self.output_layer(h))
            predictions.append(output)

        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0)

        return mean_pred, uncertainty

    def kl_divergence(self) -> torch.Tensor:
        """Compute total KL divergence for all layers."""
        kl = 0
        for layer in self.layers:
            kl += layer.kl_divergence()
        kl += self.output_layer.kl_divergence()
        return kl


class TransformerSafetyModel(nn.Module):
    """
    Transformer-based safety assessment model.
    """

    def __init__(self, config: ModelConfig, num_heads: int = 8, num_layers: int = 4):
        super(TransformerSafetyModel, self).__init__()
        self.config = config

        # Embedding dimension should be divisible by num_heads
        embed_dim = config.hidden_dims[0]
        if embed_dim % num_heads != 0:
            embed_dim = ((embed_dim // num_heads) + 1) * num_heads

        # Input projection
        self.input_projection = nn.Linear(config.input_dim, embed_dim)

        # Positional encoding
        self.positional_encoding = self._create_positional_encoding(1000, embed_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=config.dropout_rate,
            activation="relu",
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(embed_dim // 2, config.output_dim),
        )

    def _create_positional_encoding(self, max_len: int, embed_dim: int) -> nn.Parameter:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * -(np.log(10000.0) / embed_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through transformer."""
        # Handle both 2D and 3D input
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension

        batch_size, seq_len, _ = x.shape

        # Project input
        x = self.input_projection(x)

        # Add positional encoding
        x = x + self.positional_encoding[:, :seq_len, :]

        # Transformer encoding
        x = self.transformer(x, mask=mask)

        # Global pooling
        x = x.mean(dim=1)

        # Output projection
        output = torch.sigmoid(self.output_layers(x))

        return output


class GraphSafetyNetwork(nn.Module):
    """
    Graph Neural Network for safety assessment of structured actions.
    """

    def __init__(self, config: ModelConfig, num_layers: int = 3):
        super(GraphSafetyNetwork, self).__init__()
        self.config = config

        # Graph convolution layers
        self.conv_layers = nn.ModuleList()
        input_dim = config.input_dim

        for i, hidden_dim in enumerate(config.hidden_dims[:num_layers]):
            self.conv_layers.append(self._create_graph_conv(input_dim, hidden_dim))
            input_dim = hidden_dim

        # Readout layers
        self.readout = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(input_dim // 2, config.output_dim),
        )

    def _create_graph_conv(self, in_features: int, out_features: int) -> nn.Module:
        """Create a graph convolution layer."""

        class GraphConv(nn.Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                self.linear = nn.Linear(in_features, out_features)
                self.neighbor_linear = nn.Linear(in_features, out_features)

            def forward(self, x, adj_matrix):
                # Self connection
                self_features = self.linear(x)

                # Neighbor aggregation
                if adj_matrix is not None:
                    neighbor_features = torch.matmul(adj_matrix, x)
                    neighbor_features = self.neighbor_linear(neighbor_features)

                    # Combine
                    output = self_features + neighbor_features
                else:
                    output = self_features

                return F.relu(output)

        return GraphConv(in_features, out_features)

    def forward(
        self, x: torch.Tensor, adj_matrix: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through GNN."""
        # Apply graph convolutions
        for conv in self.conv_layers:
            x = conv(x, adj_matrix)
            if self.config.dropout_rate > 0:
                x = F.dropout(x, self.config.dropout_rate, self.training)

        # Global pooling (if batch dimension exists)
        if len(x.shape) == 3:
            x = x.mean(dim=1)
        elif len(x.shape) == 2 and adj_matrix is not None:
            x = x.mean(dim=0, keepdim=True)

        # Readout
        output = torch.sigmoid(self.readout(x))

        return output


class VariationalSafetyAutoencoder(nn.Module):
    """
    Variational Autoencoder for safety-aware latent representations.
    """

    def __init__(self, config: ModelConfig, latent_dim: int = 32):
        super(VariationalSafetyAutoencoder, self).__init__()
        self.config = config
        self.latent_dim = latent_dim

        # Encoder
        encoder_layers = []
        input_dim = config.input_dim

        for hidden_dim in config.hidden_dims:
            encoder_layers.append(nn.Linear(input_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.BatchNorm1d(hidden_dim))

            if config.dropout_rate > 0:
                encoder_layers.append(nn.Dropout(config.dropout_rate))

            input_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Latent space parameters
        self.fc_mu = nn.Linear(config.hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(config.hidden_dims[-1], latent_dim)

        # Decoder
        decoder_layers = []
        decoder_layers.append(nn.Linear(latent_dim, config.hidden_dims[-1]))
        decoder_layers.append(nn.ReLU())

        for i in range(len(config.hidden_dims) - 2, -1, -1):
            decoder_layers.append(
                nn.Linear(config.hidden_dims[i + 1], config.hidden_dims[i])
            )
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.BatchNorm1d(config.hidden_dims[i]))

            if config.dropout_rate > 0:
                decoder_layers.append(nn.Dropout(config.dropout_rate))

        decoder_layers.append(nn.Linear(config.hidden_dims[0], config.input_dim))

        self.decoder = nn.Sequential(*decoder_layers)

        # Safety classifier on latent space
        self.safety_classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, 1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode to latent distribution parameters."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for sampling."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space."""
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returns reconstruction, latent params, and safety prediction."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        safety_pred = self.safety_classifier(z)

        return x_recon, mu, logvar, z, safety_pred

    def compute_loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        safety_label: Optional[torch.Tensor] = None,
        safety_pred: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute VAE loss with optional safety supervision."""
        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x, reduction="mean")

        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # Total VAE loss
        vae_loss = recon_loss + 0.1 * kl_loss

        losses = {"total": vae_loss, "reconstruction": recon_loss, "kl": kl_loss}

        # Safety classification loss if labels provided
        if safety_label is not None and safety_pred is not None:
            safety_loss = F.binary_cross_entropy(safety_pred, safety_label)
            losses["safety"] = safety_loss
            losses["total"] = vae_loss + safety_loss

        return losses


# ============================================================
# NEURAL SAFETY VALIDATOR
# ============================================================


class NeuralSafetyValidator:
    """
    Multi-model neural safety validation system with consensus and uncertainty.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        model_configs: Optional[Dict[ModelType, ModelConfig]] = None,
        ensemble_size: int = 5,
        config: Optional[Dict[str, Any]] = None,
        model_types=None,
        device="cpu",
    ):  # Added model_types and device for compatibility
        """
        Initialize neural safety validator.

        Args:
            model_configs: Configuration for each model type
            ensemble_size: Number of models in ensemble
            config: Additional configuration
        """
        if getattr(self, "_initialized", False):
            return

        self.config = config or {}
        self.ensemble_size = min(ensemble_size, 10)  # Limit ensemble size

        # Thread safety
        self.lock = threading.RLock()

        # Shutdown flag
        self._shutdown = False

        # Initialize model configurations
        if model_configs is None:
            default_config = ModelConfig()
            model_configs = {
                ModelType.CLASSIFIER: default_config,
                ModelType.ANOMALY_DETECTOR: default_config,
                ModelType.BAYESIAN: default_config,
                ModelType.TRANSFORMER: default_config,
                ModelType.VAE: default_config,
            }
        self.model_configs = model_configs

        # Initialize models
        self.models = self._initialize_models()

        # Memory-bounded training buffer
        self.training_buffer = MemoryBoundedDeque(max_size_mb=100)  # 100MB max
        self.validation_buffer = MemoryBoundedDeque(max_size_mb=20)  # 20MB max

        # Performance tracking (with maxlen)
        self.performance_metrics = defaultdict(
            lambda: {
                "accuracy": deque(maxlen=100),
                "precision": deque(maxlen=100),
                "recall": deque(maxlen=100),
                "f1": deque(maxlen=100),
                "inference_time": deque(maxlen=100),
            }
        )

        # Consensus parameters
        self.consensus_threshold = self.config.get("consensus_threshold", 0.6)
        self.uncertainty_threshold = self.config.get("uncertainty_threshold", 0.3)

        # Device
        self.device = torch.device(
            self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )

        # Move models to device
        with self.lock:
            for model_dict in self.models.values():
                if isinstance(model_dict, dict):
                    for model in model_dict.values():
                        if isinstance(model, nn.Module):
                            model.to(self.device)
                elif isinstance(model_dict, nn.Module):
                    model_dict.to(self.device)

        # Thread pool for parallel inference
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Register cleanup
        atexit.register(self.shutdown)

        # --- Start Singleton properties ---
        self.model_types = model_types or [
            "toxicity",
            "bias",
            "risk",
            "perturbation",
            "consistency",
        ]
        # self.device = device # Already set above
        logger.info(
            f"NeuralSafetyValidator initialized with {len(self.model_types)} model types on {self.device}"
        )
        self._initialized = True
        # --- End Singleton properties ---

    def _initialize_models(self) -> Dict[ModelType, Any]:
        """Initialize all neural models."""
        models = {}

        # Safety classifier ensemble
        if ModelType.CLASSIFIER in self.model_configs:
            config = self.model_configs[ModelType.CLASSIFIER]
            models[ModelType.CLASSIFIER] = {
                f"model_{i}": SafetyClassifier(config)
                for i in range(self.ensemble_size)
            }

        # Anomaly detector
        if ModelType.ANOMALY_DETECTOR in self.model_configs:
            config = self.model_configs[ModelType.ANOMALY_DETECTOR]
            models[ModelType.ANOMALY_DETECTOR] = AnomalyDetector(config)

        # Bayesian model
        if ModelType.BAYESIAN in self.model_configs:
            config = self.model_configs[ModelType.BAYESIAN]
            models[ModelType.BAYESIAN] = BayesianSafetyNet(config)

        # Transformer model
        if ModelType.TRANSFORMER in self.model_configs:
            config = self.model_configs[ModelType.TRANSFORMER]
            models[ModelType.TRANSFORMER] = TransformerSafetyModel(config)

        # VAE model
        if ModelType.VAE in self.model_configs:
            config = self.model_configs[ModelType.VAE]
            models[ModelType.VAE] = VariationalSafetyAutoencoder(config)

        return models

    async def validate_async(
        self,
        action: Dict[str, Any],
        context: Dict[str, Any],
        use_consensus: bool = True,
        timeout: float = 1.0,
    ) -> SafetyReport:
        """
        Async validation using neural models.

        Args:
            action: Action to validate
            context: Context for validation
            use_consensus: Whether to use multi-model consensus
            timeout: Maximum inference time in seconds

        Returns:
            Safety report with neural validation results
        """
        start_time = time.time()

        try:
            # Extract features with validation
            features = self._extract_features(action, context)
            features_tensor = torch.tensor(
                features, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            if use_consensus:
                # Run inference with async timeout
                predictions = await asyncio.wait_for(
                    self._run_consensus_inference_async(features_tensor),
                    timeout=timeout,
                )
            else:
                predictions = await self._run_single_inference_async(features_tensor)

            # Analyze predictions
            safety_decision = self._analyze_predictions(predictions)

            inference_time = time.time() - start_time

            report = SafetyReport(
                safe=safety_decision["safe"],
                confidence=safety_decision["confidence"],
                violations=safety_decision["violations"],
                reasons=safety_decision["reasons"],
                metadata={
                    "neural_validation": True,
                    "consensus_used": use_consensus,
                    "models_queried": len(predictions),
                    "inference_time_ms": inference_time * 1000,
                    "uncertainty": safety_decision.get("uncertainty", 0),
                    "predictions": predictions,
                },
            )

            self._record_inference_metrics(inference_time, safety_decision)
            return report

        except asyncio.TimeoutError:
            logger.warning(f"Neural validation timeout after {timeout}s")
            return SafetyReport(
                safe=False,
                confidence=0.5,
                violations=[SafetyViolationType.PERFORMANCE],
                reasons=["Neural validation timeout"],
                metadata={
                    "timeout": True,
                    "consensus_used": use_consensus,
                    "neural_validation": True,
                },
            )
        except Exception as e:
            logger.error(f"Neural validation error: {e}")
            return SafetyReport(
                safe=False,
                confidence=0.5,
                violations=[SafetyViolationType.VALIDATION_ERROR],
                reasons=[f"Neural validation error: {str(e)}"],
                metadata={
                    "error": str(e),
                    "consensus_used": use_consensus,
                    "neural_validation": True,
                },
            )

    async def _run_consensus_inference_async(
        self, features: torch.Tensor
    ) -> Dict[str, Any]:
        """Async consensus inference across models."""
        predictions = {}

        # Set models to eval
        with self.lock:
            for model_type, model_dict in self.models.items():
                if isinstance(model_dict, dict):
                    for model in model_dict.values():
                        model.eval()
                else:
                    model_dict.eval()

        # Run in thread pool to avoid blocking
        asyncio.get_event_loop()

        with torch.no_grad():
            # Classifier ensemble - run in parallel
            if ModelType.CLASSIFIER in self.models:
                tasks = []
                with self.lock:
                    classifier_models = [self.models[ModelType.CLASSIFIER].items())

                for model_name, model in classifier_models:

                    async def run_classifier(m=model):
                        return float(m(features).cpu().numpy()[0, 0])

                    task = asyncio.create_task(run_classifier())
                    tasks.append(task)

                try:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    # Filter out exceptions
                    valid_results = [r for r in results if not isinstance(r, Exception])

                    if valid_results:
                        predictions["classifier_ensemble"] = {
                            "mean": float(np.mean(valid_results)),
                            "std": float(np.std(valid_results)),
                            "votes": valid_results,
                        }
                except Exception as e:
                    logger.error(f"Classifier ensemble error: {e}")

            # Anomaly detection
            if ModelType.ANOMALY_DETECTOR in self.models:
                try:
                    with self.lock:
                        model = self.models[ModelType.ANOMALY_DETECTOR]

                    async def run_anomaly():
                        return float(
                            model.compute_anomaly_score(features).cpu().numpy()[0]
                        )

                    anomaly_score = await run_anomaly()
                    predictions["anomaly"] = {
                        "score": anomaly_score,
                        "is_anomaly": float(anomaly_score > 0.5),
                    }
                except Exception as e:
                    logger.error(f"Anomaly detection error: {e}")

            # Bayesian prediction with uncertainty
            if ModelType.BAYESIAN in self.models:
                try:
                    with self.lock:
                        model = self.models[ModelType.BAYESIAN]

                    async def run_bayesian():
                        mean_pred, uncertainty = model(features)
                        return {
                            "mean": float(mean_pred.cpu().numpy()[0, 0]),
                            "uncertainty": float(uncertainty.cpu().numpy()[0, 0]),
                        }

                    predictions["bayesian"] = await run_bayesian()
                except Exception as e:
                    logger.error(f"Bayesian prediction error: {e}")

            # Transformer prediction
            if ModelType.TRANSFORMER in self.models:
                try:
                    with self.lock:
                        model = self.models[ModelType.TRANSFORMER]

                    async def run_transformer():
                        return float(model(features).cpu().numpy()[0, 0])

                    pred = await run_transformer()
                    predictions["transformer"] = {"prediction": pred}
                except Exception as e:
                    logger.error(f"Transformer prediction error: {e}")

            # VAE prediction
            if ModelType.VAE in self.models:
                try:
                    with self.lock:
                        model = self.models[ModelType.VAE]

                    async def run_vae():
                        x_recon, mu, logvar, z, safety_pred = model(features)
                        recon_error = F.mse_loss(x_recon, features).cpu().numpy()
                        return {
                            "safety": float(safety_pred.cpu().numpy()[0, 0]),
                            "reconstruction_error": float(recon_error),
                            "latent_norm": float(torch.norm(z).cpu().numpy()),
                        }

                    predictions["vae"] = await run_vae()
                except Exception as e:
                    logger.error(f"VAE prediction error: {e}")

        return predictions

    async def _run_single_inference_async(
        self, features: torch.Tensor
    ) -> Dict[str, Any]:
        """Async single model inference."""
        predictions = {}

        # Use first classifier if available
        if ModelType.CLASSIFIER in self.models:
            with self.lock:
                model = [self.models[ModelType.CLASSIFIER].values())[0]
            model.eval()

            with torch.no_grad():
                pred = float(model(features).cpu().numpy()[0, 0])
                predictions["classifier"] = {"prediction": pred}

        return predictions

    def validate(
        self,
        action: Dict[str, Any],
        context: Dict[str, Any],
        use_consensus: bool = True,
        timeout: float = 1.0,
    ) -> SafetyReport:
        """
        Synchronous wrapper for async validate.

        Args:
            action: Action to validate
            context: Context for validation
            use_consensus: Whether to use multi-model consensus
            timeout: Maximum inference time in seconds

        Returns:
            Safety report with neural validation results
        """
        # Check if we're in an async context
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            # We're in async context - create task
            return asyncio.create_task(
                self.validate_async(action, context, use_consensus, timeout)
            )
        else:
            # Not in async context - create new event loop
            return asyncio.run(
                self.validate_async(action, context, use_consensus, timeout)
            )

    def _extract_features(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> np.ndarray:
        """Extract features for neural models with robust error handling."""
        features = np.zeros(128, dtype=np.float32)  # Default feature size

        try:
            # Action type encoding
            action_type = action.get("type", "unknown")
            if isinstance(action_type, ActionType):
                action_type = action_type.value

            # One-hot encode action type (first 10 features)
            action_types = [
                "explore",
                "optimize",
                "maintain",
                "wait",
                "emergency_stop",
                "safe_fallback",
                "unknown",
            ]
            if str(action_type).lower() in action_types:
                idx = action_types.index(str(action_type).lower())
                features[idx] = 1.0

            # Numerical features with type validation
            def safe_float(value, default=0.5):
                """Safely convert value to float."""
                try:
                    return float(value)
                except (ValueError, TypeError):
                    logger.warning(
                        f"Could not convert value {value} to float, using default {default}"
                    )
                    return default

            features[10] = safe_float(action.get("confidence", 0.5))
            features[11] = safe_float(action.get("uncertainty", 0.5))
            features[12] = safe_float(action.get("risk_score", 0.5))
            features[13] = safe_float(action.get("safety_score", 0.5))

            # Resource usage features
            if "resource_usage" in action and isinstance(
                action["resource_usage"], dict
            ):
                cpu_usage = safe_float(action["resource_usage"].get("cpu", 0))
                memory_usage = safe_float(action["resource_usage"].get("memory", 0))
                features[20] = min(1.0, cpu_usage / 100)
                features[21] = min(1.0, memory_usage / 100)

            # Context features
            features[30] = safe_float(context.get("system_load", 0.5))
            features[31] = safe_float(context.get("time_pressure", 0.0))
            features[32] = float(context.get("critical_operation", False))
            features[33] = float(context.get("user_override", False))

            # Embedding features if available
            if "embedding" in action and action["embedding"] is not None:
                try:
                    embedding = np.array(action["embedding"], dtype=np.float32)[
                        :64
                    ]  # Use first 64 dims
                    features[40 : 40 + len(embedding)] = embedding
                except Exception as e:
                    logger.warning(f"Could not process embedding: {e}")

            # Historical features
            if "history" in context:
                history = context["history"]
                if isinstance(history, list) and history:
                    recent_failures = sum(
                        1 for h in history[-10:] if not h.get("safe", True)
                    )
                    features[110] = recent_failures / 10.0

            # Normalize features
            features = np.clip(features, -1, 1)

        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            # Return default features on error
            features = np.zeros(128, dtype=np.float32)

        return features

    def _analyze_predictions(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze model predictions to make safety decision."""
        safe_votes = []
        confidences = []
        violations = []
        reasons = []

        # Analyze classifier ensemble
        if "classifier_ensemble" in predictions:
            ensemble = predictions["classifier_ensemble"]
            safe_prob = ensemble["mean"]
            safe_votes.append(safe_prob > 0.5)
            confidences.append(abs(safe_prob - 0.5) * 2)  # Convert to confidence

            if ensemble["std"] > self.uncertainty_threshold:
                reasons.append(f"High ensemble uncertainty: {ensemble['std']:.3f}")

        # Analyze anomaly detection
        if "anomaly" in predictions:
            anomaly = predictions["anomaly"]
            if anomaly["is_anomaly"]:
                safe_votes.append(False)
                violations.append(SafetyViolationType.ANOMALY)
                reasons.append(f"Anomaly detected (score: {anomaly['score']:.3f})")
            else:
                safe_votes.append(True)
            confidences.append(abs(anomaly["score"] - 0.5) * 2)

        # Analyze Bayesian prediction
        if "bayesian" in predictions:
            bayesian = predictions["bayesian"]
            safe_votes.append(bayesian["mean"] > 0.5)

            # Adjust confidence based on uncertainty
            base_confidence = abs(bayesian["mean"] - 0.5) * 2
            uncertainty_penalty = min(0.5, bayesian["uncertainty"])
            confidences.append(base_confidence * (1 - uncertainty_penalty))

            if bayesian["uncertainty"] > self.uncertainty_threshold:
                reasons.append(
                    f"High prediction uncertainty: {bayesian['uncertainty']:.3f}"
                )

        # Analyze transformer prediction
        if "transformer" in predictions:
            trans = predictions["transformer"]
            safe_votes.append(trans["prediction"] > 0.5)
            confidences.append(abs(trans["prediction"] - 0.5) * 2)

        # Analyze VAE prediction
        if "vae" in predictions:
            vae = predictions["vae"]
            safe_votes.append(vae["safety"] > 0.5)
            confidences.append(vae["safety"])

            if vae["reconstruction_error"] > 0.5:
                reasons.append(
                    f"High reconstruction error: {vae['reconstruction_error']:.3f}"
                )

        # Consensus decision
        if safe_votes:
            safe_ratio = sum(safe_votes) / len(safe_votes)
            is_safe = safe_ratio >= self.consensus_threshold

            if not is_safe:
                violations.append(SafetyViolationType.CONSENSUS_FAILURE)
                reasons.append(
                    f"Consensus failed: {safe_ratio:.2f} < {self.consensus_threshold}"
                )
        else:
            is_safe = False
            safe_ratio = 0.0
            reasons.append("No model predictions available")

        # Average confidence
        avg_confidence = np.mean(confidences) if confidences else 0.5

        # Calculate overall uncertainty
        if len(confidences) > 1:
            uncertainty = np.std(confidences)
        else:
            uncertainty = 0.0

        return {
            "safe": is_safe,
            "confidence": float(avg_confidence),
            "violations": violations,
            "reasons": reasons,
            "uncertainty": float(uncertainty),
            "safe_ratio": float(safe_ratio),
        }

    def train(
        self,
        training_data: List[Tuple[Dict, Dict, bool]],
        validation_data: Optional[List[Tuple[Dict, Dict, bool]]] = None,
        model_types: Optional[List[ModelType]] = None,
        epochs: int = 10,
    ) -> Dict[str, Any]:
        """
        Train neural safety models.

        Args:
            training_data: List of (action, context, is_safe) tuples
            validation_data: Optional validation data
            model_types: Which models to train (None = all)
            epochs: Number of training epochs

        Returns:
            Training metrics
        """
        # Check for empty data
        if not training_data or len(training_data) == 0:
            logger.warning("No training data provided")
            return {}

        # Limit epochs to prevent runaway training
        epochs = min(epochs, 100)

        if model_types is None:
            with self.lock:
                model_types = list(self.models.keys())

        training_metrics = {}

        # Prepare datasets
        X_train, y_train = self._prepare_dataset(training_data)
        X_val, y_val = None, None

        if validation_data:
            X_val, y_val = self._prepare_dataset(validation_data)

        # Train each model type
        for model_type in model_types:
            with self.lock:
                if model_type not in self.models:
                    continue

            logger.info(f"Training {model_type.value} model(s)")

            if model_type == ModelType.CLASSIFIER:
                # Train ensemble
                metrics = self._train_ensemble(X_train, y_train, X_val, y_val, epochs)
                training_metrics[model_type.value] = metrics

            elif model_type == ModelType.ANOMALY_DETECTOR:
                # Train autoencoder (unsupervised)
                metrics = self._train_anomaly_detector(X_train, epochs)
                training_metrics[model_type.value] = metrics

            elif model_type == ModelType.BAYESIAN:
                # Train Bayesian model
                metrics = self._train_bayesian(X_train, y_train, X_val, y_val, epochs)
                training_metrics[model_type.value] = metrics

            elif model_type == ModelType.TRANSFORMER:
                # Train transformer
                metrics = self._train_transformer(
                    X_train, y_train, X_val, y_val, epochs
                )
                training_metrics[model_type.value] = metrics

            elif model_type == ModelType.VAE:
                # Train VAE
                metrics = self._train_vae(X_train, y_train, epochs)
                training_metrics[model_type.value] = metrics

            # Clear GPU cache after each model type
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        return training_metrics

    def _prepare_dataset(
        self, data: List[Tuple[Dict, Dict, bool]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare dataset for training."""
        X = []
        y = []

        for action, context, is_safe in data:
            features = self._extract_features(action, context)
            X.append(features)
            y.append(float(is_safe))

        X = torch.tensor(np.array(X), dtype=torch.float32)
        y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1)

        return X, y

    def _train_ensemble(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: Optional[torch.Tensor],
        y_val: Optional[torch.Tensor],
        epochs: int,
    ) -> Dict[str, Any]:
        """Train classifier ensemble."""
        metrics = {}

        with self.lock:
            classifier_models = dict(self.models[ModelType.CLASSIFIER])

        for model_name, model in classifier_models.items():
            # Ensure model parameters require gradients
            for param in model.parameters():
                param.requires_grad = True

            model.train()
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
            criterion = nn.BCELoss()

            train_losses = []
            val_losses = []

            # Validate batch size
            batch_size = min(32, len(X_train))
            if batch_size == 0:
                logger.warning(f"Empty training data for {model_name}")
                continue

            # Create data loader
            dataset = TensorDataset(X_train, y_train)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            best_val_loss = float("inf")
            patience_counter = 0

            # Ensure gradients are enabled during training
            with torch.set_grad_enabled(True):
                for epoch in range(epochs):
                    epoch_loss = 0
                    for batch_X, batch_y in dataloader:
                        batch_X, batch_y = (
                            batch_X.to(self.device),
                            batch_y.to(self.device),
                        )

                        optimizer.zero_grad(set_to_none=True)
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()

                        epoch_loss += loss.item()

                    train_losses.append(epoch_loss / len(dataloader))

                    # Validation
                    if X_val is not None and len(X_val) > 0:
                        model.eval()
                        with torch.no_grad():
                            val_outputs = model(X_val.to(self.device))
                            val_loss = criterion(val_outputs, y_val.to(self.device))
                            val_losses.append(val_loss.item())

                            # Early stopping
                            if val_loss.item() < best_val_loss:
                                best_val_loss = val_loss.item()
                                patience_counter = 0
                            else:
                                patience_counter += 1

                            if patience_counter >= 10:  # Early stopping patience
                                logger.info(
                                    f"Early stopping at epoch {epoch} for {model_name}"
                                )
                                break

                        model.train()

            metrics[model_name] = {
                "final_train_loss": train_losses[-1] if train_losses else 0,
                "final_val_loss": val_losses[-1] if val_losses else 0,
                "epochs_trained": len(train_losses),
            }

        return metrics

    def _train_anomaly_detector(
        self, X_train: torch.Tensor, epochs: int
    ) -> Dict[str, Any]:
        """Train anomaly detector (unsupervised)."""
        if len(X_train) == 0:
            logger.warning("Empty training data for anomaly detector")
            return {"final_loss": 0}

        with self.lock:
            model = self.models[ModelType.ANOMALY_DETECTOR]

        # Ensure model parameters require gradients
        for param in model.parameters():
            param.requires_grad = True

        model.train()

        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Validate batch size
        batch_size = min(32, len(X_train))
        dataset = TensorDataset(X_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        train_losses = []

        # Ensure gradients are enabled during training
        with torch.set_grad_enabled(True):
            for epoch in range(epochs):
                epoch_loss = 0
                for batch in dataloader:
                    batch_X = batch[0].to(self.device)

                    optimizer.zero_grad(set_to_none=True)
                    x_recon, z = model(batch_X)
                    loss = F.mse_loss(x_recon, batch_X)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                train_losses.append(epoch_loss / len(dataloader))

        return {"final_loss": train_losses[-1] if train_losses else 0}

    def _train_bayesian(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: Optional[torch.Tensor],
        y_val: Optional[torch.Tensor],
        epochs: int,
    ) -> Dict[str, Any]:
        """Train Bayesian neural network."""
        if len(X_train) == 0:
            logger.warning("Empty training data for Bayesian model")
            return {"final_loss": 0}

        with self.lock:
            model = self.models[ModelType.BAYESIAN]

        # Ensure model parameters require gradients
        for param in model.parameters():
            param.requires_grad = True

        model.train()

        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Validate batch size
        batch_size = min(32, len(X_train))
        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        train_losses = []

        # Ensure gradients are enabled during training
        with torch.set_grad_enabled(True):
            for epoch in range(epochs):
                epoch_loss = 0
                for batch_X, batch_y in dataloader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                    optimizer.zero_grad(set_to_none=True)

                    # Forward pass with uncertainty
                    mean_pred, uncertainty = model(batch_X)

                    # Compute loss (BCE + KL divergence)
                    bce_loss = F.binary_cross_entropy(mean_pred, batch_y)
                    kl_loss = model.kl_divergence() / len(X_train)

                    loss = bce_loss + 0.01 * kl_loss

                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                train_losses.append(epoch_loss / len(dataloader))

        return {"final_loss": train_losses[-1] if train_losses else 0}

    def _train_transformer(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: Optional[torch.Tensor],
        y_val: Optional[torch.Tensor],
        epochs: int,
    ) -> Dict[str, Any]:
        """Train transformer model."""
        if len(X_train) == 0:
            logger.warning("Empty training data for transformer")
            return {"final_loss": 0}

        with self.lock:
            model = self.models[ModelType.TRANSFORMER]

        # Ensure model parameters require gradients
        for param in model.parameters():
            param.requires_grad = True

        model.train()

        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        criterion = nn.BCELoss()

        # Validate batch size
        batch_size = min(16, len(X_train))
        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        train_losses = []

        # Ensure gradients are enabled during training
        with torch.set_grad_enabled(True):
            for epoch in range(epochs):
                epoch_loss = 0
                for batch_X, batch_y in dataloader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                    optimizer.zero_grad(set_to_none=True)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                train_losses.append(epoch_loss / len(dataloader))

        return {"final_loss": train_losses[-1] if train_losses else 0}

    def _train_vae(
        self, X_train: torch.Tensor, y_train: torch.Tensor, epochs: int
    ) -> Dict[str, Any]:
        """Train VAE with safety supervision."""
        if len(X_train) == 0:
            logger.warning("Empty training data for VAE")
            return {"final_loss": 0}

        with self.lock:
            model = self.models[ModelType.VAE]

        # Ensure model parameters require gradients
        for param in model.parameters():
            param.requires_grad = True

        model.train()

        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Validate batch size
        batch_size = min(32, len(X_train))
        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        train_losses = []

        # Ensure gradients are enabled during training
        with torch.set_grad_enabled(True):
            for epoch in range(epochs):
                epoch_loss = 0
                for batch_X, batch_y in dataloader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                    optimizer.zero_grad(set_to_none=True)

                    # Forward pass
                    x_recon, mu, logvar, z, safety_pred = model(batch_X)

                    # Compute losses
                    losses = model.compute_loss(
                        batch_X, x_recon, mu, logvar, batch_y, safety_pred
                    )

                    losses["total"].backward()
                    optimizer.step()

                    epoch_loss += losses["total"].item()

                train_losses.append(epoch_loss / len(dataloader))

        return {"final_loss": train_losses[-1] if train_losses else 0}

    def _record_inference_metrics(
        self, inference_time: float, safety_decision: Dict[str, Any]
    ):
        """Record metrics for performance tracking."""
        # Record inference time (with lock)
        with self.lock:
            for model_type in self.models.keys():
                self.performance_metrics[model_type]["inference_time"].append(
                    inference_time * 1000
                )

    def cleanup_old_checkpoints(self, keep_last_n: int = 3):
        """Clean up old model checkpoints to save disk space."""
        checkpoint_dir = Path(self.config.get("checkpoint_dir", "checkpoints"))

        if not checkpoint_dir.exists():
            return

        with self.lock:
            model_types = list(self.models.keys())

        # Find all checkpoint files
        checkpoints = []
        for model_type in model_types:
            pattern = f"{model_type.value}_checkpoint_*.pth"
            checkpoints.extend(checkpoint_dir.glob(pattern))

        # Sort by modification time
        checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        # Delete old checkpoints
        for checkpoint in checkpoints[keep_last_n:]:
            try:
                checkpoint.unlink()
                logger.info(f"Deleted old checkpoint: {checkpoint}")
            except Exception as e:
                logger.error(f"Error deleting checkpoint {checkpoint}: {e}")

    def save_models(self, save_dir: str):
        """Save all neural models to directory."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        with self.lock:
            for model_type, model_dict in self.models.items():
                try:
                    if isinstance(model_dict, dict):
                        # Save ensemble
                        for model_name, model in model_dict.items():
                            model_path = (
                                save_path / f"{model_type.value}_{model_name}.pth"
                            )
                            torch.save(model.state_dict(), model_path)
                    else:
                        # Save single model
                        model_path = save_path / f"{model_type.value}.pth"
                        torch.save(model_dict.state_dict(), model_path)
                except Exception as e:
                    logger.error(f"Error saving {model_type.value}: {e}")

        logger.info(f"Models saved to {save_dir}")

    def load_models(self, load_dir: str):
        """Load neural models from directory."""
        load_path = Path(load_dir)

        with self.lock:
            for model_type, model_dict in self.models.items():
                try:
                    if isinstance(model_dict, dict):
                        # Load ensemble
                        for model_name, model in model_dict.items():
                            model_path = (
                                load_path / f"{model_type.value}_{model_name}.pth"
                            )
                            if model_path.exists():
                                model.load_state_dict(
                                    torch.load(model_path, map_location=self.device)
                                )
                    else:
                        # Load single model
                        model_path = load_path / f"{model_type.value}.pth"
                        if model_path.exists():
                            model_dict.load_state_dict(
                                torch.load(model_path, map_location=self.device)
                            )
                except Exception as e:
                    logger.error(f"Error loading {model_type.value}: {e}")

        logger.info(f"Models loaded from {load_dir}")

    def get_model_stats(self) -> Dict[str, Any]:
        """Get statistics about neural models."""
        with self.lock:
            stats = {
                "num_models": sum(
                    len(m) if isinstance(m, dict) else 1 for m in self.models.values()
                ),
                "model_types": [mt.value for mt in self.models.keys()],
                "device": str(self.device),
                "consensus_threshold": self.consensus_threshold,
                "uncertainty_threshold": self.uncertainty_threshold,
                "performance": {},
                "memory_usage": {
                    "training_buffer_mb": self.training_buffer.get_memory_usage_mb(),
                    "validation_buffer_mb": self.validation_buffer.get_memory_usage_mb(),
                },
            }

            # Add performance metrics
            for model_type, metrics in self.performance_metrics.items():
                if metrics["inference_time"]:
                    stats["performance"][model_type.value] = {
                        "avg_inference_time_ms": np.mean(
                            [metrics["inference_time"])
                        )
                    }

        return stats

    def update_consensus_threshold(self, new_threshold: float):
        """Update consensus threshold dynamically."""
        with self.lock:
            self.consensus_threshold = np.clip(new_threshold, 0.5, 0.9)
        logger.info(f"Consensus threshold updated to {self.consensus_threshold}")

    def update_uncertainty_threshold(self, new_threshold: float):
        """Update uncertainty threshold dynamically."""
        with self.lock:
            self.uncertainty_threshold = np.clip(new_threshold, 0.1, 0.5)
        logger.info(f"Uncertainty threshold updated to {self.uncertainty_threshold}")

    def shutdown(self):
        """Shutdown validator and cleanup resources."""
        if self._shutdown:
            return

        # FIXED: Skip blocking operations during pytest runs
        is_pytest = os.environ.get("PYTEST_RUNNING") == "1"
        if is_pytest:
            self._shutdown = True
            return

        logger.info("Shutting down NeuralSafetyValidator...")
        self._shutdown = True

        # Shutdown executor (without timeout parameter for Python 3.10.11 compatibility)
        try:
            self.executor.shutdown(wait=True)
        except Exception as e:
            logger.error(f"Error shutting down executor: {e}")

        # Clear CUDA cache if using GPU
        if self.device.type == "cuda":
            try:
                torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                logger.error(f"Error clearing CUDA cache: {e}")

        # Clear buffers
        with self.lock:
            self.training_buffer.clear()
            self.validation_buffer.clear()

        logger.info("NeuralSafetyValidator shutdown complete")


def initialize_neural_safety():
    global _NEURAL_SAFETY_INIT_DONE
    if _NEURAL_SAFETY_INIT_DONE:
        logger.debug("NeuralSafetyValidator already initialized – skipping.")
        return NeuralSafetyValidator()
    nsv = NeuralSafetyValidator()
    _NEURAL_SAFETY_INIT_DONE = True
    return nsv


# Back-compat aliases (place once, after class NeuralSafetyValidator)
SafetyPredictor = NeuralSafetyValidator


# Minimal stub only if callers import but don't really use FeatureExtractor
class FeatureExtractor:  # tiny shim to satisfy import
    def extract(self, *_, **__):  # not used by your validator logic
        return {}
