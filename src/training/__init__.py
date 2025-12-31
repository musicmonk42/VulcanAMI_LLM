# ============================================================
# VULCAN Training Package
# Training components for VULCAN LLM
# ============================================================
#
# MODULES:
#     governed_trainer        - Consensus-governed training loop
#     self_improving_training - Meta-learning and self-improvement
#     gpt_model               - GPT model architecture
#     causal_loss             - Causal language modeling loss
#     data_loader             - Training data loading utilities
#     metrics                 - Training metrics and evaluation
#     self_awareness          - Self-awareness training module
#
# ============================================================

import logging

__version__ = "1.0.0"
__author__ = "VULCAN-AGI Team"

logger = logging.getLogger(__name__)

# Governed Trainer
try:
    from src.training.governed_trainer import GovernedTrainer
    GOVERNED_TRAINER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"GovernedTrainer not available: {e}")
    GOVERNED_TRAINER_AVAILABLE = False
    GovernedTrainer = None

# Self-Improving Training
try:
    from src.training.self_improving_training import SelfImprovingTraining
    SELF_IMPROVING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"SelfImprovingTraining not available: {e}")
    SELF_IMPROVING_AVAILABLE = False
    SelfImprovingTraining = None

# GPT Model
try:
    from src.training.gpt_model import GPTModel, GPTConfig
    GPT_MODEL_AVAILABLE = True
except ImportError as e:
    logger.debug(f"GPTModel not available: {e}")
    GPT_MODEL_AVAILABLE = False
    GPTModel = None
    GPTConfig = None

# Causal Loss
try:
    from src.training.causal_loss import CausalLoss
    CAUSAL_LOSS_AVAILABLE = True
except ImportError as e:
    logger.debug(f"CausalLoss not available: {e}")
    CAUSAL_LOSS_AVAILABLE = False
    CausalLoss = None

# Data Loader
try:
    from src.training.data_loader import TrainingDataLoader
    DATA_LOADER_AVAILABLE = True
except ImportError as e:
    logger.debug(f"TrainingDataLoader not available: {e}")
    DATA_LOADER_AVAILABLE = False
    TrainingDataLoader = None

__all__ = [
    "__version__",
    "__author__",
    # Governed Trainer
    "GovernedTrainer",
    "GOVERNED_TRAINER_AVAILABLE",
    # Self-Improving
    "SelfImprovingTraining",
    "SELF_IMPROVING_AVAILABLE",
    # GPT Model
    "GPTModel",
    "GPTConfig",
    "GPT_MODEL_AVAILABLE",
    # Loss
    "CausalLoss",
    "CAUSAL_LOSS_AVAILABLE",
    # Data
    "TrainingDataLoader",
    "DATA_LOADER_AVAILABLE",
]

logger.debug(f"VULCAN Training package v{__version__} loaded")
