"""
Meta-learning implementation with task detection
"""

import logging
import pickle
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Safe import of ModalityType
try:
    from ..config import ModalityType

    MODALITY_TYPE_AVAILABLE = True
except (ImportError, AttributeError):
    ModalityType = None
    MODALITY_TYPE_AVAILABLE = False

from ..config import EMBEDDING_DIM, HIDDEN_DIM
from ..security_fixes import safe_pickle_load
from .learning_types import LearningConfig
from .parameter_history import ParameterHistoryManager

logger = logging.getLogger(__name__)

# ============================================================
# META-LEARNING TYPES
# ============================================================


class MetaLearningAlgorithm(Enum):
    """Supported meta-learning algorithms"""

    MAML = "maml"  # Model-Agnostic Meta-Learning
    FOMAML = "fomaml"  # First-Order MAML
    REPTILE = "reptile"  # Reptile algorithm
    PROTO = "proto"  # Prototypical Networks
    ANIL = "anil"  # Almost No Inner Loop


@dataclass
class TaskStatistics:
    """Statistics for a specific task"""

    task_id: str
    num_samples: int = 0
    avg_loss: float = 0.0
    best_loss: float = float("inf")
    adaptation_steps: List[float] = None
    last_seen: float = 0.0
    difficulty_score: float = 0.5

    def __post_init__(self):
        if self.adaptation_steps is None:
            self.adaptation_steps = []


# ============================================================
# ENHANCED TASK DETECTION
# ============================================================


class TaskDetector:
    """Detect and track learning tasks with persistence and clustering."""

    def __init__(self, threshold: float = 0.3, save_path: str = "task_signatures"):
        self.threshold = threshold
        self.task_signatures = {}
        self.task_statistics = {}
        self.current_task = None
        self.task_history = deque(maxlen=100)
        self.transition_matrix = defaultdict(lambda: defaultdict(float))

        # Task clustering
        self.task_clusters = {}
        self.cluster_centers = []

        # Persistence
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self._load_signatures()

        # Task embedding learning
        self.task_encoder = nn.Sequential(
            nn.Linear(self._get_signature_dim(), HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, EMBEDDING_DIM // 2),
            nn.Tanh(),
        )
        self.task_encoder_optimizer = optim.Adam(
            self.task_encoder.parameters(), lr=0.001
        )

        # FIXED: Get device from task encoder
        self.device = next(self.task_encoder.parameters()).device

        # Lock for thread safety
        self._lock = threading.RLock()

    def detect_task(self, experience: Dict[str, Any]) -> str:
        """Detect task from experience using signatures."""
        with self._lock:
            signature = self._compute_signature(experience)

            # Compute embedding with device management
            signature_tensor = torch.tensor(
                signature, dtype=torch.float32, device=self.device
            )
            with torch.no_grad():
                self.task_encoder(signature_tensor)

            # Compare with known tasks
            best_match = None
            best_similarity = 0
            similarities = {}

            for task_id, task_sig in self.task_signatures.items():
                similarity = self._similarity(signature, task_sig)
                similarities[task_id] = similarity
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = task_id

            # Check if it's a known task or new one
            if best_similarity > self.threshold:
                task_id = best_match
                # Update signature with exponential moving average
                alpha = 0.1
                self.task_signatures[task_id] = (
                    alpha * signature + (1 - alpha) * self.task_signatures[task_id]
                )
            else:
                # New task detected
                task_id = f"task_{len(self.task_signatures)}_{int(time.time())}"
                self.task_signatures[task_id] = signature
                self.task_statistics[task_id] = TaskStatistics(task_id=task_id)
                logger.info(f"New task detected: {task_id}")

                # Assign to cluster
                self._assign_to_cluster(task_id, signature)

            # Update statistics
            if task_id not in self.task_statistics:
                self.task_statistics[task_id] = TaskStatistics(task_id=task_id)

            stats = self.task_statistics[task_id]
            stats.num_samples += 1
            stats.last_seen = time.time()

            # Update task transition statistics
            if self.current_task and self.current_task != task_id:
                self.transition_matrix[self.current_task][task_id] += 1

            self.current_task = task_id
            self.task_history.append((task_id, time.time()))

            # Save periodically
            if stats.num_samples % 100 == 0:
                self._save_signatures()

            return task_id

    def _get_signature_dim(self) -> int:
        """Get dimension of task signature"""
        # 5 statistical features + 5 modality features + 1 reward + 1 complexity
        return 12

    def _compute_signature(self, experience: Dict[str, Any]) -> np.ndarray:
        """Compute task signature from experience."""
        features = []

        # Statistical features from embedding
        if "embedding" in experience and experience["embedding"] is not None:
            emb = experience["embedding"]
            if isinstance(emb, torch.Tensor):
                emb = emb.detach().cpu().numpy()
            elif not isinstance(emb, np.ndarray):
                emb = np.array(emb)

            # Handle empty arrays
            if emb.size > 0:
                features.extend(
                    [
                        np.mean(emb),
                        np.std(emb),
                        np.min(emb),
                        np.max(emb),
                        np.median(emb),
                    ]
                )
            else:
                features.extend([0, 0, 0, 0, 0])
        else:
            features.extend([0, 0, 0, 0, 0])

        # Handle modality encoding safely
        modality = experience.get("modality", "UNKNOWN")
        modality_encoding = [0] * 5

        # First try to handle as enum if available
        if MODALITY_TYPE_AVAILABLE and ModalityType is not None:
            try:
                # Check if modality is an instance of the enum
                # Use hasattr to check if it's an enum-like object
                if hasattr(modality, "value") and hasattr(modality, "name"):
                    # It looks like an enum member
                    try:
                        modality_list = list(ModalityType)
                        idx = modality_list.index(modality)
                        if idx < 5:
                            modality_encoding[idx] = 1
                    except (ValueError, TypeError):
                        # Not in the list or other error
                        pass
            except (TypeError, AttributeError):
                # Not an enum or other error
                pass

        # Also handle string representation
        if isinstance(modality, str):
            # Map common modality strings to indices
            modality_map = {
                "VISUAL": 0,
                "AUDIO": 1,
                "TEXT": 2,
                "TACTILE": 3,
                "UNKNOWN": 4,
                "SMELL": 4,
                "TASTE": 4,
                "PROPRIOCEPTIVE": 4,
                "VESTIBULAR": 4,
                "TEMPERATURE": 4,
            }
            idx = modality_map.get(modality.upper(), 4)
            if idx < 5:
                modality_encoding[min(idx, 4)] = 1

        features.extend(modality_encoding)

        # Reward characteristics
        reward = experience.get("reward", 0)
        if reward is None:
            reward = 0
        try:
            features.append(float(reward))
        except (TypeError, ValueError):
            features.append(0.0)

        # Complexity
        complexity = 0.5
        if "metadata" in experience and isinstance(experience["metadata"], dict):
            complexity = experience["metadata"].get("complexity", 0.5)
            if complexity is None:
                complexity = 0.5
        try:
            features.append(float(complexity))
        except (TypeError, ValueError):
            features.append(0.5)

        return np.array(features, dtype=np.float32)

    def _similarity(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        """Compute similarity between task signatures."""
        # Cosine similarity with safety checks
        norm1 = np.linalg.norm(sig1)
        norm2 = np.linalg.norm(sig2)

        # Add epsilon to prevent division by zero
        epsilon = 1e-10
        if norm1 < epsilon or norm2 < epsilon:
            return 0.0

        dot_product = np.dot(sig1, sig2)
        similarity = dot_product / (norm1 * norm2)

        # Clip to valid range
        return float(np.clip(similarity, -1.0, 1.0))

    def _assign_to_cluster(self, task_id: str, signature: np.ndarray):
        """Assign task to a cluster"""
        if not self.cluster_centers:
            # First cluster
            self.cluster_centers.append(signature)
            self.task_clusters[0] = [task_id]
        else:
            # Find nearest cluster
            min_dist = float("inf")
            best_cluster = 0

            for i, center in enumerate(self.cluster_centers):
                dist = np.linalg.norm(signature - center)
                if dist < min_dist:
                    min_dist = dist
                    best_cluster = i

            # Check if should create new cluster
            if min_dist > 0.5:  # Threshold for new cluster
                self.cluster_centers.append(signature)
                self.task_clusters[len(self.cluster_centers) - 1] = [task_id]
            else:
                if best_cluster not in self.task_clusters:
                    self.task_clusters[best_cluster] = []
                self.task_clusters[best_cluster].append(task_id)

                # Update cluster center
                alpha = 0.1
                self.cluster_centers[best_cluster] = (
                    alpha * signature + (1 - alpha) * self.cluster_centers[best_cluster]
                )

    def get_related_tasks(self, task_id: str, k: int = 5) -> List[str]:
        """Get k most related tasks"""
        if task_id not in self.task_signatures:
            return []

        signature = self.task_signatures[task_id]
        similarities = []

        for other_id, other_sig in self.task_signatures.items():
            if other_id != task_id:
                sim = self._similarity(signature, other_sig)
                similarities.append((other_id, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return [tid for tid, _ in similarities[:k]]

    def predict_next_task(self) -> Optional[str]:
        """Predict next task based on transition history."""
        if not self.current_task:
            return None

        transitions = self.transition_matrix.get(self.current_task, {})
        if not transitions:
            return None

        # Return most likely next task
        return max(transitions.items(), key=lambda x: x[1])[0]

    def get_task_difficulty(self, task_id: str) -> float:
        """Estimate task difficulty based on statistics"""
        if task_id not in self.task_statistics:
            return 0.5

        stats = self.task_statistics[task_id]

        # Factors for difficulty
        # Higher loss = more difficult
        loss_factor = min(1.0, stats.avg_loss)

        # More adaptation steps needed = more difficult
        adapt_factor = 0.0
        if stats.adaptation_steps:
            avg_steps = np.mean(stats.adaptation_steps)
            adapt_factor = min(1.0, avg_steps / 10.0)

        # Combine factors
        difficulty = 0.6 * loss_factor + 0.4 * adapt_factor

        stats.difficulty_score = difficulty
        return difficulty

    def _save_signatures(self):
        """Save task signatures to disk"""
        data = {
            "signatures": self.task_signatures,
            "statistics": {k: asdict(v) for k, v in self.task_statistics.items()},
            "transitions": dict(self.transition_matrix),
            "clusters": self.task_clusters,
            "cluster_centers": self.cluster_centers,
        }

        save_file = self.save_path / "task_data.pkl"
        with open(save_file, "wb") as f:
            pickle.dump(data, f)

    def _load_signatures(self):
        """Load task signatures from disk"""
        save_file = self.save_path / "task_data.pkl"
        if save_file.exists():
            try:
                with open(save_file, "rb") as f:
                    data = safe_pickle_load(f)

                self.task_signatures = data.get("signatures", {})

                # Reconstruct statistics
                for task_id, stats_dict in data.get("statistics", {}).items():
                    self.task_statistics[task_id] = TaskStatistics(**stats_dict)

                self.transition_matrix = defaultdict(
                    lambda: defaultdict(float), data.get("transitions", {})
                )
                self.task_clusters = data.get("clusters", {})
                self.cluster_centers = data.get("cluster_centers", [])

                logger.info(f"Loaded {len(self.task_signatures)} task signatures")
            except Exception as e:
                logger.error(f"Failed to load task signatures: {e}")


# ============================================================
# ENHANCED META-LEARNER
# ============================================================


class MetaLearner:
    """Enhanced Model-Agnostic Meta-Learning with multiple algorithms."""

    def __init__(
        self,
        base_model: nn.Module,
        config: LearningConfig = None,
        algorithm: MetaLearningAlgorithm = MetaLearningAlgorithm.MAML,
    ):
        self.config = config or LearningConfig()
        self.algorithm = algorithm
        self.base_model = base_model

        # FIXED: Get device from base model
        self.device = (
            next(base_model.parameters()).device
            if hasattr(base_model, "parameters")
            else torch.device("cpu")
        )

        # FIX: Get embedding dimension from base model, fallback to global constant
        self.embedding_dim = getattr(base_model, "embedding_dim", EMBEDDING_DIM)

        # Different optimizers for different algorithms
        if algorithm == MetaLearningAlgorithm.REPTILE:
            self.meta_optimizer = optim.SGD(
                base_model.parameters(), lr=self.config.meta_lr
            )
        else:
            self.meta_optimizer = optim.Adam(
                base_model.parameters(), lr=self.config.meta_lr
            )

        self.task_losses = deque(maxlen=100)
        self.adaptation_history = []

        # Online meta-learning components
        self.online_buffer = deque(maxlen=1000)
        self.task_embeddings = {}  # task_id -> embedding

        # Task-specific learning rates
        self.task_learning_rates = {}

        # Gradient statistics for analysis
        self.gradient_stats = {
            "mean_norm": deque(maxlen=100),
            "max_norm": deque(maxlen=100),
            "gradient_similarity": deque(maxlen=100),
        }

        # Parameter history for meta-learning
        self.param_history = ParameterHistoryManager(
            base_path="meta_params", config=config
        )

        # Prototypical network components (if using PROTO)
        if algorithm == MetaLearningAlgorithm.PROTO:
            self.prototype_layer = nn.Linear(self.embedding_dim, self.embedding_dim).to(
                self.device
            )

        # Lock for thread safety
        self._lock = threading.RLock()
        self._shutdown = threading.Event()

    def adapt(
        self,
        support_set: Dict[str, torch.Tensor],
        num_steps: Optional[int] = None,
        task_id: Optional[str] = None,
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """Fast adaptation on support set with tracking and analysis."""
        num_steps = num_steps or self.config.adaptation_steps

        # Get task-specific learning rate
        with self._lock:
            if task_id and task_id in self.task_learning_rates:
                inner_lr = self.task_learning_rates[task_id]
            else:
                inner_lr = self.config.inner_lr

        # Algorithm-specific adaptation
        if self.algorithm == MetaLearningAlgorithm.FOMAML:
            adapted_model, stats = self._adapt_fomaml(support_set, num_steps, inner_lr)
        elif self.algorithm == MetaLearningAlgorithm.REPTILE:
            adapted_model, stats = self._adapt_reptile(support_set, num_steps, inner_lr)
        elif self.algorithm == MetaLearningAlgorithm.ANIL:
            adapted_model, stats = self._adapt_anil(support_set, num_steps, inner_lr)
        else:  # MAML
            adapted_model, stats = self._adapt_maml(support_set, num_steps, inner_lr)

        # Store task embedding if task_id provided
        if task_id:
            with torch.no_grad():
                # Use final hidden state as task embedding
                x = support_set.get("x")
                if x is not None and len(x) > 0:
                    x = x.to(self.device)
                    embedding = adapted_model(x[:1]).detach().mean(dim=0)
                    with self._lock:
                        self.task_embeddings[task_id] = embedding

        return adapted_model, stats

    def _adapt_maml(
        self, support_set: Dict[str, torch.Tensor], num_steps: int, inner_lr: float
    ) -> Tuple[nn.Module, Dict]:
        """FIXED: Standard MAML adaptation with proper gradient handling"""
        # Clone model for task-specific adaptation
        adapted_model = self._clone_model()
        task_optimizer = optim.SGD(adapted_model.parameters(), lr=inner_lr)

        # Track adaptation trajectory
        adaptation_trajectory = []

        for step in range(num_steps):
            loss = self._compute_loss(adapted_model, support_set)

            task_optimizer.zero_grad()
            loss.backward(create_graph=True, retain_graph=True)  # Second-order

            # FIXED: Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(adapted_model.parameters(), 1.0)

            task_optimizer.step()

            grad_norm = self._compute_grad_norm(adapted_model)
            adaptation_trajectory.append(
                {"step": step, "loss": loss.item(), "grad_norm": grad_norm}
            )

            # FIXED: Detach intermediate states to prevent memory buildup
            if step < num_steps - 1:
                for param in adapted_model.parameters():
                    if param.grad is not None:
                        param.grad.detach_()

        stats = {
            "trajectory": adaptation_trajectory,
            "final_loss": (
                adaptation_trajectory[-1]["loss"] if adaptation_trajectory else 0
            ),
            "num_steps": num_steps,
        }

        return adapted_model, stats

    def _adapt_fomaml(
        self, support_set: Dict[str, torch.Tensor], num_steps: int, inner_lr: float
    ) -> Tuple[nn.Module, Dict]:
        """FIXED: First-order MAML with gradient detachment"""
        adapted_model = self._clone_model()
        task_optimizer = optim.SGD(adapted_model.parameters(), lr=inner_lr)

        adaptation_trajectory = []

        for step in range(num_steps):
            loss = self._compute_loss(adapted_model, support_set)

            task_optimizer.zero_grad()
            loss.backward()  # First-order only

            # FIXED: Clip gradients
            torch.nn.utils.clip_grad_norm_(adapted_model.parameters(), 1.0)

            task_optimizer.step()

            adaptation_trajectory.append(
                {
                    "step": step,
                    "loss": loss.item(),
                    "grad_norm": self._compute_grad_norm(adapted_model),
                }
            )

            # FIXED: Detach to prevent accumulation
            loss = loss.detach()

        stats = {
            "trajectory": adaptation_trajectory,
            "final_loss": (
                adaptation_trajectory[-1]["loss"] if adaptation_trajectory else 0
            ),
            "num_steps": num_steps,
            "algorithm": "fomaml",
        }

        return adapted_model, stats

    def _adapt_reptile(
        self, support_set: Dict[str, torch.Tensor], num_steps: int, inner_lr: float
    ) -> Tuple[nn.Module, Dict]:
        """FIXED: Reptile algorithm with proper detachment"""
        adapted_model = self._clone_model()
        task_optimizer = optim.SGD(adapted_model.parameters(), lr=inner_lr)

        adaptation_trajectory = []

        for step in range(num_steps):
            loss = self._compute_loss(adapted_model, support_set)

            task_optimizer.zero_grad()
            loss.backward()

            # FIXED: Clip gradients
            torch.nn.utils.clip_grad_norm_(adapted_model.parameters(), 1.0)

            task_optimizer.step()

            adaptation_trajectory.append({"step": step, "loss": loss.item()})

            # FIXED: Detach loss
            loss = loss.detach()

        stats = {
            "trajectory": adaptation_trajectory,
            "final_loss": (
                adaptation_trajectory[-1]["loss"] if adaptation_trajectory else 0
            ),
            "num_steps": num_steps,
            "algorithm": "reptile",
        }

        return adapted_model, stats

    def _adapt_anil(
        self, support_set: Dict[str, torch.Tensor], num_steps: int, inner_lr: float
    ) -> Tuple[nn.Module, Dict]:
        """FIXED: Almost No Inner Loop with gradient handling"""
        adapted_model = self._clone_model()

        # Only optimize final layer
        final_layer_params = []
        for name, param in adapted_model.named_parameters():
            if "final" in name or "head" in name or "classifier" in name:
                final_layer_params.append(param)

        if not final_layer_params:
            # Fallback to last layer
            all_params = list(adapted_model.parameters())
            if all_params:
                final_layer_params = [all_params[-1]]

        if not final_layer_params:
            # No parameters to optimize, return model as-is
            return adapted_model, {
                "trajectory": [],
                "final_loss": 0,
                "num_steps": 0,
                "algorithm": "anil",
            }

        task_optimizer = optim.SGD(final_layer_params, lr=inner_lr)

        adaptation_trajectory = []

        for step in range(num_steps):
            loss = self._compute_loss(adapted_model, support_set)

            task_optimizer.zero_grad()
            loss.backward()

            # FIXED: Clip gradients
            torch.nn.utils.clip_grad_norm_(final_layer_params, 1.0)

            task_optimizer.step()

            adaptation_trajectory.append({"step": step, "loss": loss.item()})

            # FIXED: Detach loss
            loss = loss.detach()

        stats = {
            "trajectory": adaptation_trajectory,
            "final_loss": (
                adaptation_trajectory[-1]["loss"] if adaptation_trajectory else 0
            ),
            "num_steps": num_steps,
            "algorithm": "anil",
        }

        return adapted_model, stats

    def meta_update(self, tasks: List[Dict[str, Any]]):
        """Meta-learning update across tasks with trajectory tracking."""
        with self._lock:
            if self.algorithm == MetaLearningAlgorithm.REPTILE:
                self._meta_update_reptile(tasks)
            else:
                self._meta_update_maml(tasks)

    def _meta_update_maml(self, tasks: List[Dict[str, Any]]):
        """FIXED: MAML meta-update with proper gradient accumulation

        For MAML, we need to compute meta-gradients that account for the adaptation.
        This uses a first-order approximation (FOMAML) by default, which is more
        stable and doesn't require second-order derivatives.

        The algorithm:
        1. For each task: adapt base model, compute query loss, accumulate gradients
        2. Apply meta-optimizer step with accumulated gradients
        3. This effectively updates θ based on performance after adaptation
        """
        # Start trajectory recording
        trajectory_id = self.param_history.start_trajectory(
            task_id=f"meta_batch_{time.time()}", agent_id="meta_learner"
        )

        # Zero gradients
        self.meta_optimizer.zero_grad()

        meta_loss_sum = 0
        task_gradients = []
        gradient_norms = []

        # Store original base model parameters
        original_state = {
            name: param.data.clone()
            for name, param in self.base_model.named_parameters()
        }

        for task in tasks:
            task_id = task.get("task_id")
            support_set = task["support"]
            query_set = task["query"]

            # Get inner learning rate
            inner_lr = self.task_learning_rates.get(task_id, self.config.inner_lr)

            # Perform adaptation on support set
            # Use a simple optimizer for inner loop
            inner_optimizer = optim.SGD(self.base_model.parameters(), lr=inner_lr)

            for step in range(self.config.adaptation_steps):
                inner_optimizer.zero_grad()
                support_loss = self._compute_loss(self.base_model, support_set)
                support_loss.backward()
                inner_optimizer.step()

            # Zero gradients from adaptation before computing meta-gradients
            self.base_model.zero_grad()

            # Evaluate on query set with adapted model
            query_loss = self._compute_loss(self.base_model, query_set)

            # Compute gradients from query loss
            # These gradients represent how to update the base model
            # to improve post-adaptation performance
            query_loss.backward()

            meta_loss_sum += query_loss.item()
            self.task_losses.append(query_loss.item())

            # Collect gradient statistics
            task_grad = []
            for param in self.base_model.parameters():
                if param.grad is not None:
                    task_grad.append(param.grad.clone().detach())
                    gradient_norms.append(param.grad.norm().item())
            task_gradients.append(task_grad)

            # Restore base model to original state for next task
            # This ensures each task starts from the same base model
            with torch.no_grad():
                for name, param in self.base_model.named_parameters():
                    param.data = original_state[name]

            # Update task-specific learning rate
            if task_id:
                adapt_stats = {"trajectory": [{"loss": query_loss.item()}]}
                self._update_task_learning_rate(task_id, adapt_stats)

        # Analyze gradients
        self._analyze_gradients(task_gradients, gradient_norms)

        # Apply meta-update with accumulated gradients from all tasks
        # The gradients have been accumulated across all tasks
        torch.nn.utils.clip_grad_norm_(self.base_model.parameters(), 1.0)
        self.meta_optimizer.step()

        # Calculate average loss
        avg_meta_loss = meta_loss_sum / len(tasks)

        # Save checkpoint periodically
        if (
            self.config.checkpoint_frequency > 0
            and len(self.adaptation_history) % self.config.checkpoint_frequency == 0
        ):
            checkpoint_path = self.param_history.save_checkpoint(
                self.base_model,
                metadata={
                    "meta_loss": avg_meta_loss,
                    "num_tasks": len(tasks),
                    "trajectory_id": trajectory_id,
                    "algorithm": self.algorithm.value,
                },
            )
            logger.info(f"Saved meta-learning checkpoint: {checkpoint_path}")

        # End trajectory
        self.param_history.end_trajectory()

        # Record meta-learning step
        self.adaptation_history.append(
            {
                "timestamp": time.time(),
                "num_tasks": len(tasks),
                "avg_loss": avg_meta_loss,
                "trajectory_id": trajectory_id,
                "gradient_stats": {k: list(v) for k, v in self.gradient_stats.items()},
            }
        )

    def _meta_update_reptile(self, tasks: List[Dict[str, Any]]):
        """FIXED: Reptile meta-update with proper parameter handling"""
        # Store original parameters
        [p.clone().detach() for p in self.base_model.parameters()]

        trajectory_id = self.param_history.start_trajectory(
            task_id=f"reptile_batch_{time.time()}", agent_id="meta_learner"
        )

        avg_loss = 0

        for task in tasks:
            # Adapt to task
            adapted_model, adapt_stats = self.adapt(
                task["support"], task_id=task.get("task_id")
            )

            # Evaluate
            query_loss = self._compute_loss(adapted_model, task["query"])
            avg_loss += query_loss.item()

            # Interpolate parameters
            epsilon = self.config.meta_lr
            for orig_param, adapted_param in zip(
                self.base_model.parameters(), adapted_model.parameters()
            ):
                orig_param.data = orig_param.data + epsilon * (
                    adapted_param.data.detach() - orig_param.data
                )

        avg_loss /= len(tasks)

        # Record
        self.adaptation_history.append(
            {
                "timestamp": time.time(),
                "num_tasks": len(tasks),
                "avg_loss": avg_loss,
                "trajectory_id": trajectory_id,
                "algorithm": "reptile",
            }
        )

        self.param_history.end_trajectory()

    def _analyze_gradients(
        self, task_gradients: List[List[torch.Tensor]], gradient_norms: List[float]
    ):
        """Analyze gradient statistics for monitoring"""
        if not gradient_norms:
            return

        # Gradient norm statistics
        self.gradient_stats["mean_norm"].append(np.mean(gradient_norms))
        self.gradient_stats["max_norm"].append(np.max(gradient_norms))

        # Gradient similarity between tasks
        if len(task_gradients) > 1:
            similarities = []
            for i in range(len(task_gradients)):
                for j in range(i + 1, len(task_gradients)):
                    sim = self._gradient_similarity(
                        task_gradients[i], task_gradients[j]
                    )
                    similarities.append(sim)

            if similarities:
                self.gradient_stats["gradient_similarity"].append(np.mean(similarities))

    def _gradient_similarity(
        self, grads1: List[torch.Tensor], grads2: List[torch.Tensor]
    ) -> float:
        """Compute cosine similarity between gradient vectors"""
        if not grads1 or not grads2 or len(grads1) != len(grads2):
            return 0.0

        # Flatten and concatenate gradients
        flat1 = torch.cat([g.flatten() for g in grads1])
        flat2 = torch.cat([g.flatten() for g in grads2])

        # Cosine similarity
        sim = F.cosine_similarity(flat1.unsqueeze(0), flat2.unsqueeze(0))
        return sim.item()

    def _update_task_learning_rate(self, task_id: str, adapt_stats: Dict):
        """Update task-specific learning rate based on adaptation performance"""
        if not task_id:
            return

        trajectory = adapt_stats.get("trajectory", [])
        if len(trajectory) < 2:
            return

        # Check if loss is decreasing
        losses = [step["loss"] for step in trajectory]
        loss_decrease = losses[0] - losses[-1]

        # Get current learning rate
        with self._lock:
            current_lr = self.task_learning_rates.get(task_id, self.config.inner_lr)

            # Adjust learning rate
            if loss_decrease < 0.01:  # Loss not decreasing enough
                # Increase learning rate
                new_lr = min(current_lr * 1.2, 0.1)
            elif loss_decrease > 1.0:  # Loss decreasing too fast (might be unstable)
                # Decrease learning rate
                new_lr = max(current_lr * 0.8, 1e-5)
            else:
                new_lr = current_lr

            self.task_learning_rates[task_id] = new_lr

    def online_meta_update(self, experience: Dict[str, Any]):
        """Online meta-learning from streaming experiences"""
        self.online_buffer.append(experience)

        # Perform meta-update when buffer is full
        if len(self.online_buffer) >= self.config.meta_batch_size * 2:
            # Create tasks from buffer
            tasks = self._create_tasks_from_buffer()

            if tasks:
                self.meta_update(tasks)

    def _create_tasks_from_buffer(self) -> List[Dict[str, Any]]:
        """Create meta-learning tasks from online buffer with validation split"""
        if len(self.online_buffer) < self.config.meta_batch_size * 3:
            return []

        tasks = []
        buffer_list = list(self.online_buffer)

        for _ in range(self.config.meta_batch_size):
            # Random split for support, query, and validation
            indices = np.random.permutation(len(buffer_list))

            # 50% support, 30% query, 20% validation
            support_end = len(indices) // 2
            query_end = support_end + (len(indices) * 3) // 10

            support_indices = indices[:support_end]
            query_indices = indices[support_end:query_end]
            val_indices = indices[query_end:]

            # Create sets
            support = self._create_batch_from_indices(buffer_list, support_indices)
            query = self._create_batch_from_indices(buffer_list, query_indices)
            validation = self._create_batch_from_indices(buffer_list, val_indices)

            task = {
                "support": support,
                "query": query,
                "validation": validation,
                "task_id": f"buffer_task_{time.time()}",
            }

            tasks.append(task)

        return tasks

    def _create_batch_from_indices(
        self, buffer: List, indices: np.ndarray
    ) -> Dict[str, torch.Tensor]:
        """FIXED: Create batch from buffer indices with robust error handling"""
        # Handle empty indices
        if len(indices) == 0:
            logger.warning("Empty indices in batch creation, returning dummy batch")
            return {
                "x": torch.randn(1, self.embedding_dim, device=self.device),
                "y": torch.zeros(1, device=self.device),
            }

        batch_x = []
        batch_y = []

        for idx in indices:
            try:
                exp = buffer[idx]
                if "embedding" in exp and exp["embedding"] is not None:
                    x = exp["embedding"]

                    # Convert to tensor if needed
                    if isinstance(x, np.ndarray):
                        x = torch.tensor(x, dtype=torch.float32)
                    elif not isinstance(x, torch.Tensor):
                        x = torch.tensor(x, dtype=torch.float32)

                    # FIXED: Ensure consistent dimensions
                    if x.dim() == 0:
                        x = x.unsqueeze(0)
                    elif x.dim() > 1:
                        x = x.flatten()

                    # FIXED: Pad/truncate to embedding_dim
                    if x.shape[0] < self.embedding_dim:
                        padding = torch.zeros(self.embedding_dim - x.shape[0])
                        x = torch.cat([x, padding])
                    elif x.shape[0] > self.embedding_dim:
                        x = x[: self.embedding_dim]

                    # FIXED: Move to correct device
                    x = x.to(self.device)
                    batch_x.append(x)

                    # Use reward as target if available
                    y = exp.get("reward", 0.0)
                    if not isinstance(y, torch.Tensor):
                        y = torch.tensor(y, dtype=torch.float32)
                    if y.dim() == 0:
                        y = y.unsqueeze(0)
                    y = y.to(self.device)
                    batch_y.append(y)
            except Exception as e:
                logger.warning(f"Failed to process buffer item at index {idx}: {e}")
                continue

        # FIXED: Handle case where all items failed to process
        if not batch_x:
            logger.warning("All buffer items failed to process, returning dummy batch")
            return {
                "x": torch.randn(1, self.embedding_dim, device=self.device),
                "y": torch.zeros(1, device=self.device),
            }

        return {
            "x": torch.stack(batch_x),
            "y": (
                torch.stack(batch_y).squeeze(-1)
                if batch_y
                else torch.zeros(len(batch_x), device=self.device)
            ),
        }

    def _clone_model(self) -> nn.Module:
        """Create a functional copy of the model.

        Instead of using deepcopy (which fails on models with threading locks),
        we create a new instance of the model and copy its state_dict.
        This avoids the pickling issues with threading locks.
        """
        model_class = type(self.base_model)

        # Try to create a new instance with known constructor signatures
        try:
            # For EnhancedContinualLearner which has this signature
            # Note: We use attribute detection instead of isinstance to avoid circular imports
            # (continual_learning.py imports MetaLearner from this module)
            if hasattr(self.base_model, "embedding_dim") and hasattr(
                self.base_model, "config"
            ):
                cloned = model_class(
                    embedding_dim=self.base_model.embedding_dim,
                    config=self.base_model.config,
                    use_hierarchical=getattr(
                        self.base_model, "use_hierarchical", False
                    ),
                    use_progressive=getattr(self.base_model, "use_progressive", False),
                )
            else:
                # For simple models, try no-arg constructor
                cloned = model_class()

            # Load the state dict to copy all parameters and buffers
            cloned.load_state_dict(self.base_model.state_dict())
            # Ensure cloned model is on same device
            cloned = cloned.to(self.device)
            return cloned

        except Exception as e:
            # If we can't instantiate a new model, log error and raise
            logger.error(
                f"Failed to clone model of type {model_class.__name__}: {e}. "
                f"Model cloning requires either a no-arg constructor or "
                f"embedding_dim, config, use_hierarchical, use_progressive attributes."
            )
            raise RuntimeError(
                f"Cannot clone model of type {model_class.__name__}. "
                f"Consider using a simpler model or adding model construction logic."
            ) from e

    def _compute_loss(
        self, model: nn.Module, data: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """FIXED: Compute task-specific loss with proper dimension handling for all cases"""
        x = data.get("x")
        y = data.get("y")

        if x is None or y is None:
            # Self-supervised loss - handle dimension mismatch properly
            x = data.get(
                "input", torch.randn(1, self.embedding_dim, device=self.device)
            )
            x = x.to(self.device)
            pred = model(x)

            # For self-supervised learning, when dimensions don't match,
            # use L2 regularization of the output as a learning signal
            if pred.shape[-1] != x.shape[-1]:
                # Use L2 norm of prediction as regularization loss
                # This provides a meaningful gradient while avoiding dimension mismatch
                return torch.mean(pred**2)
            else:
                # If dimensions match, use reconstruction loss
                return F.mse_loss(pred, x)

        # FIXED: Ensure tensors on correct device
        x = x.to(self.device)
        y = y.to(self.device)

        pred = model(x)

        # Determine loss type based on target dtype and shape
        if y.dtype == torch.long:
            # Classification - ensure y is 1D for cross_entropy
            if y.dim() > 1:
                y = y.squeeze(-1)
            return F.cross_entropy(pred, y)
        else:
            # Regression - handle dimension mismatch
            if pred.shape != y.shape:
                if len(pred.shape) == 2 and len(y.shape) == 1:
                    y = y.unsqueeze(1)
                elif len(pred.shape) == 1 and len(y.shape) == 2:
                    pred = pred.unsqueeze(1)
            return F.mse_loss(pred, y)

    def _compute_grad_norm(self, model: nn.Module) -> float:
        """Compute gradient norm for monitoring"""
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_norm += param.grad.norm().item() ** 2
        return total_norm**0.5

    def get_statistics(self) -> Dict[str, Any]:
        """Get meta-learning statistics"""
        with self._lock:
            stats = {
                "algorithm": self.algorithm.value,
                "num_adaptations": len(self.adaptation_history),
                "avg_task_loss": (
                    np.mean(list(self.task_losses)) if self.task_losses else 0
                ),
                "online_buffer_size": len(self.online_buffer),
                "num_task_embeddings": len(self.task_embeddings),
                "task_learning_rates": self.task_learning_rates.copy(),
                "gradient_stats": {
                    k: list(v)[-10:] for k, v in self.gradient_stats.items()
                },
                "device": str(self.device),
            }

            return stats

    def shutdown(self):
        """Clean shutdown"""
        logger.info("Shutting down MetaLearner...")
        self._shutdown.set()

        # Save final checkpoint
        if hasattr(self, "param_history"):
            try:
                self.param_history.save_checkpoint(
                    self.base_model,
                    metadata={"final": True, "algorithm": self.algorithm.value},
                )
                self.param_history.shutdown()
            except Exception as e:
                logger.error(f"Failed to save final checkpoint: {e}")

        logger.info("MetaLearner shutdown complete")
