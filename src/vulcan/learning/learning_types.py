"""
Learning types, configurations, and data structures for VULCAN-AGI
"""

import numpy as np
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import time

from ..config import ModalityType, EMBEDDING_DIM, HIDDEN_DIM, LATENT_DIM

# ============================================================
# LEARNING TYPES AND CONFIGS
# ============================================================


class LearningMode(Enum):
    """Learning modes supported by the system."""

    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    REINFORCEMENT = "reinforcement"
    META = "meta"
    CONTINUAL = "continual"
    CURRICULUM = "curriculum"
    TRANSFER = "transfer"
    FEDERATED = "federated"
    RLHF = "rlhf"  # Reinforcement Learning from Human Feedback
    ONLINE = "online"  # Online/streaming learning


@dataclass
class LearningConfig:
    """Configuration for learning systems."""

    learning_rate: float = 0.001
    batch_size: int = 32
    ewc_lambda: float = 100.0
    meta_lr: float = 0.001
    inner_lr: float = 0.01
    replay_buffer_size: int = 10000
    consolidation_threshold: int = 100
    curriculum_stages: int = 5
    task_detection_threshold: float = 0.3
    memory_strength: float = 0.5
    adaptation_steps: int = 5
    meta_batch_size: int = 4
    # RLHF parameters
    rlhf_enabled: bool = True
    feedback_buffer_size: int = 5000
    reward_model_update_freq: int = 100
    ppo_epochs: int = 4
    ppo_clip: float = 0.2
    kl_penalty: float = 0.01
    # Parameter history
    checkpoint_frequency: int = 1000
    max_checkpoints: int = 100
    audit_trail_enabled: bool = True


@dataclass
class TaskInfo:
    """Information about a learning task."""

    task_id: str
    task_type: str
    difficulty: float = 0.5
    created_at: float = field(default_factory=time.time)
    samples_seen: int = 0
    performance: float = 0.0
    signature: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeedbackData:
    """Human feedback data structure"""

    feedback_id: str
    timestamp: float
    feedback_type: str  # 'rating', 'comparison', 'correction', 'preference'
    content: Any
    context: Dict[str, Any]
    agent_response: Any
    human_preference: Any
    reward_signal: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningTrajectory:
    """Complete learning trajectory for auditing"""

    trajectory_id: str
    start_time: float
    end_time: Optional[float]
    task_id: str
    agent_id: str
    states: List[np.ndarray]
    actions: List[Any]
    rewards: List[float]
    losses: List[float]
    parameter_snapshots: List[str]  # Paths to parameter files
    metadata: Dict[str, Any] = field(default_factory=dict)
