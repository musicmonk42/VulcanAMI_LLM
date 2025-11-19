"""
Unified world model with dynamics and reward prediction
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from typing import Any, Dict, Tuple, List, Optional, Union
from collections import deque, defaultdict
from dataclasses import dataclass
import logging
import time
import numpy as np
from pathlib import Path
import pickle
import math
import random
from enum import Enum
import threading
import concurrent.futures

# ADDED for atomic save
import tempfile
import os
import io
# time was already imported

from ..config import EMBEDDING_DIM, HIDDEN_DIM
from .parameter_history import ParameterHistoryManager

logger = logging.getLogger(__name__)

# ============================================================
# ATOMIC WRITE UTILITY
# (Copied from deployment.py for robust saving)
# ============================================================

def atomic_write_with_retry(data: bytes, 
                            target_path: str, 
                            max_retries: int = 5,
                            retry_delay: float = 0.1,
                            suffix: str = '.pt') -> bool:
    """
    Atomic file write with Windows-compatible retry logic
    
    Handles Windows file locking issues by:
    - Writing to temporary file first
    - Properly closing file handles
    - Implementing exponential backoff retry
    - Cleaning up on failure
    - Using os.replace for Windows-safe atomic operations
    """
    
    # FIXED: Convert to absolute path immediately and ensure parent exists
    target_path_obj = Path(target_path).resolve()
    target_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    temp_fd = None
    temp_path = None
    
    try:
        # Create temporary file in same directory for atomic operation
        temp_fd, temp_path = tempfile.mkstemp(
            dir=str(target_path_obj.parent),  # FIXED: Convert to string
            prefix='.tmp_model_',
            suffix=suffix # Use appropriate suffix
        )
        
        # Write data to temporary file
        try:
            os.write(temp_fd, data)
            os.fsync(temp_fd)  # Ensure data is written to disk
        finally:
            # Always close the file descriptor
            os.close(temp_fd)
            temp_fd = None
        
        # Attempt atomic rename with retry logic
        for attempt in range(max_retries):
            try:
                # FIXED: Use os.replace for Windows-safe atomic replacement
                os.replace(temp_path, str(target_path_obj))
                
                # Success!
                return True
                
            except PermissionError as e:
                if attempt < max_retries - 1:
                    logger.debug(
                        f"File locked on attempt {attempt + 1}/{max_retries}, "
                        f"retrying in {retry_delay * (2 ** attempt):.2f}s: {e}"
                    )
                    time.sleep(retry_delay * (2 ** attempt))
                else:
                    logger.error(f"Failed to write file after {max_retries} attempts: {e}")
                    # Re-raise the exception after max retries
                    raise PermissionError(f"Failed to replace file after retries: {e}") from e
            except Exception as e:
                logger.error(f"Unexpected error during atomic write: {e}")
                raise # Re-raise other unexpected exceptions
        
        return False
        
    except Exception as e:
        logger.error(f"Failed to perform atomic write: {e}", exc_info=True)
        return False
        
    finally:
        # Cleanup: Close file descriptor if still open
        if temp_fd is not None:
            try:
                os.close(temp_fd)
            except Exception:
                pass
        
        # Cleanup: Remove temporary file if it still exists
        if temp_path and Path(temp_path).exists():
            try:
                Path(temp_path).unlink()
            except Exception as e:
                logger.debug(f"Failed to cleanup temporary file {temp_path}: {e}")

# ============================================================
# WORLD MODEL TYPES
# ============================================================

class PlanningAlgorithm(Enum):
    """Planning algorithms available"""
    GREEDY = "greedy"
    BEAM_SEARCH = "beam_search"
    MCTS = "mcts"  # Monte Carlo Tree Search
    CEM = "cem"  # Cross Entropy Method
    MPPI = "mppi"  # Model Predictive Path Integral

@dataclass
class WorldState:
    """Represents a state in the world model"""
    embedding: torch.Tensor
    uncertainty: float = 0.0
    value: float = 0.0
    visit_count: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class MCTSNode:
    """Node for Monte Carlo Tree Search"""
    
    def __init__(self, state: torch.Tensor, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.prior = 0
    
    @property
    def value(self):
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0
    
    def is_expanded(self):
        return len(self.children) > 0
    
    def ucb_score(self, c_puct: float = 1.4):
        """Calculate UCB score for tree traversal"""
        if self.visit_count == 0:
            return float('inf')
        
        if self.parent is None:
            return self.value
        
        exploration = c_puct * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return self.value + exploration

# ============================================================
# ATTENTION MODULES
# ============================================================

class MultiHeadAttention(nn.Module):
    """Multi-head attention for state-action processing"""
    
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project and reshape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        return self.out_proj(attn_output)

# ============================================================
# ENHANCED WORLD MODEL
# ============================================================

class UnifiedWorldModel(nn.Module):
    """Enhanced world model with dynamics and reward prediction."""
    
    def __init__(self, processor=None, state_dim: int = EMBEDDING_DIM, 
                 ensemble_size: int = 5, use_attention: bool = True):
        super().__init__()
        self.processor = processor
        self.state_dim = state_dim
        self.ensemble_size = ensemble_size
        self.use_attention = use_attention
        
        # FIXED: Get device for all operations
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # State history
        self.state_history = deque(maxlen=1000)
        self.transition_buffer = deque(maxlen=10000)
        
        # Create ensemble of models for uncertainty estimation
        self.dynamics_ensemble = nn.ModuleList([
            self._create_dynamics_model() for _ in range(ensemble_size)
        ])
        
        self.reward_ensemble = nn.ModuleList([
            self._create_reward_model() for _ in range(ensemble_size)
        ])
        
        # Uncertainty estimation
        self.uncertainty_model = nn.Sequential(
            nn.Linear(state_dim * 2, HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.LayerNorm(HIDDEN_DIM // 2),
            nn.Linear(HIDDEN_DIM // 2, 1),
            nn.Sigmoid()
        )
        
        # Value function for planning
        self.value_function = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.LayerNorm(HIDDEN_DIM),
            nn.Linear(HIDDEN_DIM, 1)
        )
        
        # Inverse dynamics model (state, next_state -> action)
        self.inverse_dynamics = nn.Sequential(
            nn.Linear(state_dim * 2, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, state_dim)
        )
        
        # Curiosity module
        self.curiosity_module = CuriosityModule(state_dim)
        
        # State abstraction
        self.state_abstractor = StateAbstractor(state_dim)
        
        # Contrastive learning projection head
        self.projection_head = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 128)
        )
        
        # Move all modules to device
        self.to(self.device)
        
        # Optimizers
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=100)
        
        # Parameter tracking
        self.param_history = ParameterHistoryManager(base_path="world_model_params")
        
        # Training statistics
        self.training_stats = {
            'dynamics_losses': deque(maxlen=100),
            'reward_losses': deque(maxlen=100),
            'inverse_losses': deque(maxlen=100),
            'contrastive_losses': deque(maxlen=100),
            'curiosity_rewards': deque(maxlen=100),
            'total_steps': 0
        }
        
        # Planning cache
        self.planning_cache = {}
        
        # MCTS tree for planning
        self.mcts_trees = {}
        
        # Thread safety
        self._lock = threading.RLock()
    
    def _create_dynamics_model(self) -> nn.Module:
        """Create a single dynamics model"""
        if self.use_attention:
            return nn.Sequential(
                nn.Linear(self.state_dim * 2, HIDDEN_DIM),
                nn.LayerNorm(HIDDEN_DIM),
                nn.ReLU(),
                nn.Dropout(0.1),
                AttentionBlock(HIDDEN_DIM),
                nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
                nn.ReLU(),
                nn.Linear(HIDDEN_DIM, self.state_dim)
            )
        else:
            return nn.Sequential(
                nn.Linear(self.state_dim * 2, HIDDEN_DIM),
                nn.LayerNorm(HIDDEN_DIM),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
                nn.ReLU(),
                nn.Linear(HIDDEN_DIM, self.state_dim)
            )
    
    def _create_reward_model(self) -> nn.Module:
        """Create a single reward model"""
        return nn.Sequential(
            nn.Linear(self.state_dim * 2, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM // 2, 1)
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor, 
                model_idx: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """FIXED: Forward dynamics prediction with ensemble and device management"""
        # Ensure tensors on correct device
        state = state.to(self.device)
        action = action.to(self.device)
        
        # FIXED: Ensure both state and action have proper dimensions for concatenation
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        
        # Ensure both have same batch dimension
        if state.shape[0] != action.shape[0]:
            # Broadcast to match batch sizes
            if state.shape[0] == 1:
                state = state.expand(action.shape[0], -1)
            elif action.shape[0] == 1:
                action = action.expand(state.shape[0], -1)
        
        state_action = torch.cat([state, action], dim=-1)
        
        if model_idx is not None:
            # Use specific model from ensemble
            next_state = self.dynamics_ensemble[model_idx](state_action)
            reward = self.reward_ensemble[model_idx](state_action)
            ensemble_uncertainty = torch.zeros(1, device=self.device)
        else:
            # Use ensemble mean
            next_states = []
            rewards = []
            
            for dynamics_model, reward_model in zip(self.dynamics_ensemble, self.reward_ensemble):
                next_states.append(dynamics_model(state_action))
                rewards.append(reward_model(state_action))
            
            next_state = torch.stack(next_states).mean(dim=0)
            reward = torch.stack(rewards).mean(dim=0)
            
            # FIXED: Safe uncertainty calculation with zero check
            if len(next_states) > 1:
                next_state_std = torch.stack(next_states).std(dim=0).mean()
                reward_std = torch.stack(rewards).std(dim=0).mean()
            else:
                next_state_std = torch.tensor(0.0, device=self.device)
                reward_std = torch.tensor(0.0, device=self.device)
            
            ensemble_uncertainty = (next_state_std + reward_std) / 2
            # Add epsilon to prevent issues
            ensemble_uncertainty = torch.clamp(ensemble_uncertainty, min=1e-6)
        
        # Compute uncertainty
        uncertainty = self.uncertainty_model(state_action)
        
        if model_idx is None:
            # Combine uncertainties
            uncertainty = 0.5 * uncertainty + 0.5 * ensemble_uncertainty.unsqueeze(-1)
        
        return next_state, reward, uncertainty
    
    def predict_value(self, state: torch.Tensor) -> torch.Tensor:
        """Predict state value"""
        state = state.to(self.device)
        return self.value_function(state)
    
    def predict_inverse_dynamics(self, state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        """Predict action that led from state to next_state"""
        state = state.to(self.device)
        next_state = next_state.to(self.device)
        state_pair = torch.cat([state, next_state], dim=-1)
        return self.inverse_dynamics(state_pair)
    
    def compute_curiosity_reward(self, state: torch.Tensor, action: torch.Tensor, 
                                 next_state: torch.Tensor) -> torch.Tensor:
        """Compute curiosity-driven intrinsic reward"""
        state = state.to(self.device)
        action = action.to(self.device)
        next_state = next_state.to(self.device)
        return self.curiosity_module(state, action, next_state)
    
    def abstract_state(self, state: torch.Tensor) -> torch.Tensor:
        """Get abstract representation of state"""
        state = state.to(self.device)
        return self.state_abstractor(state)
    
    def update_state(self, state: Any, action: Any, reward: float, next_state: Any):
        """Update world model state with tracking"""
        with self._lock:
            self.state_history.append({
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'timestamp': time.time()
            })
            
            # Add to transition buffer for training
            self.transition_buffer.append({
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state
            })
            
            self.training_stats['total_steps'] += 1
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """FIXED: Enhanced training step with device management"""
        state = batch['state'].to(self.device)
        action = batch['action'].to(self.device)
        next_state = batch['next_state'].to(self.device)
        reward = batch['reward'].to(self.device)
        
        losses = {}
        
        # Train each model in ensemble
        dynamics_losses = []
        reward_losses = []
        
        for i, (dynamics_model, reward_model) in enumerate(zip(self.dynamics_ensemble, self.reward_ensemble)):
            # Bootstrap sampling for ensemble diversity
            mask = torch.bernoulli(torch.ones(len(state), device=self.device) * 0.8).bool()
            
            if mask.sum() == 0:
                continue
            
            # Forward pass
            state_action = torch.cat([state[mask], action[mask]], dim=-1)
            pred_next = dynamics_model(state_action)
            pred_reward = reward_model(state_action)
            
            # Compute losses
            dynamics_loss = F.mse_loss(pred_next, next_state[mask])
            reward_loss = F.mse_loss(pred_reward, reward[mask].unsqueeze(-1) if reward[mask].dim() == 1 else reward[mask])
            
            dynamics_losses.append(dynamics_loss)
            reward_losses.append(reward_loss)
        
        # Average losses
        if dynamics_losses:
            losses['dynamics'] = torch.stack(dynamics_losses).mean()
            losses['reward'] = torch.stack(reward_losses).mean()
        
        # Train inverse dynamics
        pred_action = self.predict_inverse_dynamics(state, next_state)
        losses['inverse'] = F.mse_loss(pred_action, action)
        
        # Contrastive loss for representation learning
        losses['contrastive'] = self._contrastive_loss(state, next_state)
        
        # Train value function
        with torch.no_grad():
            # TD target
            next_value = self.predict_value(next_state)
            target_value = reward.unsqueeze(-1) if reward.dim() == 1 else reward
            target_value = target_value + 0.99 * next_value
        
        pred_value = self.predict_value(state)
        losses['value'] = F.mse_loss(pred_value, target_value)
        
        # Curiosity loss
        curiosity_reward = self.compute_curiosity_reward(state, action, next_state)
        losses['curiosity'] = -curiosity_reward.mean()  # Maximize curiosity
        
        # Total loss
        total_loss = sum(losses.values())
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        # Update statistics
        with self._lock:
            self.training_stats['dynamics_losses'].append(losses.get('dynamics', torch.tensor(0.0)).item())
            self.training_stats['reward_losses'].append(losses.get('reward', torch.tensor(0.0)).item())
            self.training_stats['inverse_losses'].append(losses['inverse'].item())
            self.training_stats['contrastive_losses'].append(losses['contrastive'].item())
            self.training_stats['curiosity_rewards'].append(curiosity_reward.mean().item())
            
            # Save checkpoint periodically
            if self.training_stats['total_steps'] % 1000 == 0:
                try:
                    self.param_history.save_checkpoint(
                        self,
                        metadata={
                            'losses': {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()},
                            'step': self.training_stats['total_steps']
                        }
                    )
                except Exception as e:
                    logger.error(f"Failed to save checkpoint: {e}")
        
        return {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}
    
    def _contrastive_loss(self, state: torch.Tensor, next_state: torch.Tensor, 
                         temperature: float = 0.1) -> torch.Tensor:
        """Contrastive loss for representation learning"""
        batch_size = state.shape[0]
        
        if batch_size < 2:
            return torch.tensor(0.0, device=self.device)
        
        # Project states
        z1 = self.projection_head(state)
        z2 = self.projection_head(next_state)
        
        # Normalize
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(z1, z2.T) / temperature
        
        # Labels: positive pairs are on diagonal
        labels = torch.arange(batch_size, device=self.device)
        
        # Cross entropy loss
        loss = F.cross_entropy(similarity, labels)
        
        return loss
    
    def imagine_rollout(self, initial_state: torch.Tensor, 
                       action_sequence: List[torch.Tensor],
                       horizon: int = 10,
                       use_ensemble: bool = True) -> Dict[str, Any]:
        """Imagine future trajectory with ensemble uncertainty"""
        initial_state = initial_state.to(self.device)
        
        # FIXED: Ensure initial state has batch dimension
        if initial_state.dim() == 1:
            initial_state = initial_state.unsqueeze(0)
        
        states = [initial_state]
        rewards = []
        uncertainties = []
        values = []
        curiosity_rewards = []
        
        current_state = initial_state
        
        for i in range(min(horizon, len(action_sequence))):
            action = action_sequence[i].to(self.device)
            
            # FIXED: Ensure action has batch dimension
            if action.dim() == 1:
                action = action.unsqueeze(0)
            
            if use_ensemble:
                # Sample random model from ensemble for diversity
                model_idx = np.random.randint(0, self.ensemble_size)
            else:
                model_idx = None
            
            # Predict next state and reward
            with torch.no_grad():
                next_state, reward, uncertainty = self.forward(current_state, action, model_idx)
                value = self.predict_value(next_state)
                curiosity = self.compute_curiosity_reward(current_state, action, next_state)
            
            states.append(next_state)
            rewards.append(reward.item())
            uncertainties.append(uncertainty.item())
            values.append(value.item())
            curiosity_rewards.append(curiosity.mean().item() if curiosity.numel() > 1 else curiosity.item())
            
            current_state = next_state
        
        return {
            'states': states,
            'rewards': rewards,
            'uncertainties': uncertainties,
            'values': values,
            'curiosity_rewards': curiosity_rewards,
            'cumulative_reward': sum(rewards),
            'cumulative_curiosity': sum(curiosity_rewards)
        }
    
    def plan_actions(self, current_state: torch.Tensor,
                    candidate_actions: List[torch.Tensor],
                    horizon: int = 5,
                    algorithm: PlanningAlgorithm = PlanningAlgorithm.MCTS,
                    **kwargs) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Plan best action sequence using specified algorithm"""
        
        current_state = current_state.to(self.device)
        candidate_actions = [a.to(self.device) for a in candidate_actions]
        
        if algorithm == PlanningAlgorithm.GREEDY:
            return self._plan_greedy(current_state, candidate_actions, horizon)
        elif algorithm == PlanningAlgorithm.BEAM_SEARCH:
            return self._plan_beam_search(current_state, candidate_actions, horizon, **kwargs)
        elif algorithm == PlanningAlgorithm.MCTS:
            return self._plan_mcts(current_state, candidate_actions, horizon, **kwargs)
        elif algorithm == PlanningAlgorithm.CEM:
            return self._plan_cem(current_state, candidate_actions, horizon, **kwargs)
        elif algorithm == PlanningAlgorithm.MPPI:
            return self._plan_mppi(current_state, candidate_actions, horizon, **kwargs)
        else:
            return self._plan_greedy(current_state, candidate_actions, horizon)
    
    def _plan_greedy(self, current_state: torch.Tensor,
                    candidate_actions: List[torch.Tensor],
                    horizon: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Greedy planning"""
        best_reward = -float('inf')
        best_action = None
        best_rollout = None
        
        for action in candidate_actions:
            rollout = self.imagine_rollout(
                current_state,
                [action] * horizon,
                horizon
            )
            
            # Evaluate based on cumulative reward and uncertainty
            score = rollout['cumulative_reward'] - 0.1 * sum(rollout['uncertainties'])
            
            if score > best_reward:
                best_reward = score
                best_action = action
                best_rollout = rollout
        
        return best_action if best_action is not None else candidate_actions[0], best_rollout
    
    def _plan_beam_search(self, current_state: torch.Tensor,
                         candidate_actions: List[torch.Tensor],
                         horizon: int,
                         beam_width: int = 5) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """FIXED: Beam search planning with proper dimension handling"""
        # FIXED: Ensure current_state has batch dimension
        if current_state.dim() == 1:
            current_state = current_state.unsqueeze(0)
        
        # Initialize beam with all candidate actions
        beam = []
        for action in candidate_actions[:beam_width]:
            # FIXED: Ensure action has proper dimension
            if action.dim() == 1:
                action = action.unsqueeze(0)
            
            with torch.no_grad():
                next_state, reward, uncertainty = self.forward(current_state, action)
            
            beam.append({
                'actions': [action],
                'state': next_state,
                'cumulative_reward': reward.item(),
                'uncertainty': uncertainty.item()
            })
        
        # Expand beam for horizon steps
        for step in range(1, horizon):
            new_beam = []
            
            for beam_item in beam:
                for action in candidate_actions:
                    # FIXED: Ensure action has proper dimension
                    if action.dim() == 1:
                        action = action.unsqueeze(0)
                    
                    with torch.no_grad():
                        next_state, reward, uncertainty = self.forward(beam_item['state'], action)
                    
                    new_item = {
                        'actions': beam_item['actions'] + [action],
                        'state': next_state,
                        'cumulative_reward': beam_item['cumulative_reward'] + reward.item(),
                        'uncertainty': beam_item['uncertainty'] + uncertainty.item()
                    }
                    new_beam.append(new_item)
            
            # Keep top beam_width items
            new_beam.sort(key=lambda x: x['cumulative_reward'] - 0.1 * x['uncertainty'], reverse=True)
            beam = new_beam[:beam_width]
        
        # Return best action sequence
        best = beam[0]
        
        # **************************************************************************
        # FIXED: The bug was here. Removed the .squeeze(0) to ensure the
        # returned action maintains its (1, dim) shape, matching the greedy
        # algorithm and the test's expectation.
        first_action = best['actions'][0]
        # if first_action.dim() > 1 and first_action.shape[0] == 1:
        #     first_action = first_action.squeeze(0) <--- BUGGY LINE REMOVED
        
        return first_action, {
            'cumulative_reward': best['cumulative_reward'],
            'uncertainty': best['uncertainty'],
            'action_sequence': best['actions']
        }
        # **************************************************************************
    
    def _plan_mcts(self, current_state: torch.Tensor,
                  candidate_actions: List[torch.Tensor],
                  horizon: int,
                  num_simulations: int = 100,
                  c_puct: float = 1.4) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """FIXED: Monte Carlo Tree Search planning with proper gradient context"""
        
        # FIXED: Ensure current_state has batch dimension
        if current_state.dim() == 1:
            current_state = current_state.unsqueeze(0)
        
        # Create root node
        root = MCTSNode(current_state)
        
        for _ in range(num_simulations):
            # Selection
            node = root
            path = [root]
            
            while node.is_expanded() and len(path) < horizon:
                # Select best child using UCB
                best_child = max(node.children.values(), 
                               key=lambda n: n.ucb_score(c_puct))
                node = best_child
                path.append(node)
            
            # Expansion
            if not node.is_expanded() and len(path) < horizon:
                for action in candidate_actions:
                    # FIXED: Ensure action has proper dimension
                    if action.dim() == 1:
                        action = action.unsqueeze(0)
                    
                    with torch.no_grad():
                        next_state, _, _ = self.forward(node.state, action)
                    
                    child = MCTSNode(next_state, parent=node, action=action)
                    # Use tensor id as key since tensors aren't hashable
                    child_key = id(action)
                    node.children[child_key] = child
                
                if node.children:
                    # Select random child for simulation
                    node = random.choice(list(node.children.values()))
                    path.append(node)
            
            # Simulation with no_grad context
            sim_state = node.state
            sim_reward = 0
            
            with torch.no_grad():
                for _ in range(horizon - len(path)):
                    action = random.choice(candidate_actions)
                    # FIXED: Ensure action has proper dimension
                    if action.dim() == 1:
                        action = action.unsqueeze(0)
                    sim_state, reward, _ = self.forward(sim_state, action)
                    sim_reward += reward.item()
            
            # Backpropagation
            for node in reversed(path):
                node.visit_count += 1
                node.value_sum += sim_reward
        
        # Select best action
        if root.children:
            best_child = max(root.children.values(), key=lambda n: n.visit_count)
            
            # **************************************************************************
            # FIXED: The bug was here. Removed the .squeeze(0) to ensure the
            # returned action maintains its (1, dim) shape, matching the greedy
            # algorithm and the test's expectation.
            best_action = best_child.action
            # if best_action.dim() > 1 and best_action.shape[0] == 1:
            #     best_action = best_action.squeeze(0) <--- BUGGY LINE REMOVED
            
            return best_action, {
                'visit_count': best_child.visit_count,
                'value': best_child.value,
                'num_simulations': num_simulations
            }
            # **************************************************************************
        else:
            # Fallback if no expansion happened
            return candidate_actions[0], {
                'visit_count': 0,
                'value': 0,
                'num_simulations': num_simulations
            }
    
    def _plan_cem(self, current_state: torch.Tensor,
                 candidate_actions: List[torch.Tensor],
                 horizon: int,
                 population_size: int = 100,
                 elite_frac: float = 0.2,
                 num_iters: int = 10) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """FIXED: Cross Entropy Method planning with proper dimension handling"""
        
        action_dim = candidate_actions[0].shape[-1]
        
        # Initialize distribution
        mean = torch.zeros(horizon, action_dim, device=self.device)
        std = torch.ones(horizon, action_dim, device=self.device)
        
        best_action_sequence = None
        best_reward = -float('inf')
        
        for _ in range(num_iters):
            # Sample population
            population = []
            rewards = []
            
            for _ in range(population_size):
                # Sample action sequence
                action_sequence = []
                for t in range(horizon):
                    action = mean[t] + std[t] * torch.randn(action_dim, device=self.device)
                    # FIXED: Ensure action has batch dimension
                    if action.dim() == 1:
                        action = action.unsqueeze(0)
                    action_sequence.append(action)
                
                # Evaluate
                rollout = self.imagine_rollout(current_state, action_sequence, horizon)
                reward = rollout['cumulative_reward']
                
                population.append(action_sequence)
                rewards.append(reward)
            
            # Select elites
            elite_size = max(1, int(population_size * elite_frac))
            elite_indices = np.argsort(rewards)[-elite_size:]
            
            # Update distribution
            elite_actions = [population[i] for i in elite_indices]
            # FIXED: Handle dimension properly when computing mean and std
            mean = torch.stack([torch.stack([a[t].squeeze(0) if a[t].dim() > 1 else a[t] 
                                            for a in elite_actions]).mean(dim=0) 
                               for t in range(horizon)])
            std = torch.stack([torch.stack([a[t].squeeze(0) if a[t].dim() > 1 else a[t] 
                                          for a in elite_actions]).std(dim=0) 
                              for t in range(horizon)])
            std = torch.clamp(std, min=0.01)  # Prevent collapse
            
            # Track best
            best_idx = elite_indices[-1]
            if rewards[best_idx] > best_reward:
                best_reward = rewards[best_idx]
                best_action_sequence = population[best_idx]
        
        # **************************************************************************
        # FIXED: The bug was here. Removed the .squeeze(0) to ensure the
        # returned action maintains its (1, dim) shape, matching the greedy
        # algorithm and the test's expectation.
        if best_action_sequence:
            first_action = best_action_sequence[0]
            # if first_action.dim() > 1 and first_action.shape[0] == 1:
            #     first_action = first_action.squeeze(0) <--- BUGGY LINE REMOVED
            return first_action, {
                'cumulative_reward': best_reward,
                'action_sequence': best_action_sequence
            }
        # **************************************************************************
        else:
            return candidate_actions[0], {
                'cumulative_reward': best_reward,
                'action_sequence': None
            }
    
    def _plan_mppi(self, current_state: torch.Tensor,
                  candidate_actions: List[torch.Tensor],
                  horizon: int,
                  num_samples: int = 100,
                  temperature: float = 1.0) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """FIXED: Model Predictive Path Integral planning with proper dimension handling"""
        
        action_dim = candidate_actions[0].shape[-1]
        
        # Sample random action sequences
        action_sequences = []
        costs = []
        
        for _ in range(num_samples):
            # Random walk in action space
            action_sequence = []
            for _ in range(horizon):
                action = candidate_actions[np.random.randint(len(candidate_actions))]
                noise = torch.randn_like(action) * 0.1
                action_sequence.append(action + noise)
            
            # Evaluate trajectory
            rollout = self.imagine_rollout(current_state, action_sequence, horizon)
            
            # Cost = negative reward + uncertainty penalty
            cost = -rollout['cumulative_reward'] + 0.1 * sum(rollout['uncertainties'])
            
            action_sequences.append(action_sequence)
            costs.append(cost)
        
        # Convert to tensors
        costs = torch.tensor(costs, device=self.device)
        
        # Compute weights using path integral
        weights = F.softmax(-costs / temperature, dim=0)
        
        # Weighted average of action sequences
        weighted_action = torch.zeros(horizon, action_dim, device=self.device)
        for i, action_seq in enumerate(action_sequences):
            for t, action in enumerate(action_seq):
                # FIXED: Ensure action is 1D for proper accumulation
                if action.dim() > 1:
                    action = action.squeeze(0)
                weighted_action[t] += weights[i] * action
        
        return weighted_action[0], {
            'expected_cost': (weights * costs).sum().item(),
            'temperature': temperature
        }
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get comprehensive training statistics"""
        with self._lock:
            stats = {
                'total_steps': self.training_stats['total_steps'],
                'state_history_size': len(self.state_history),
                'transition_buffer_size': len(self.transition_buffer)
            }
            
            # Add average losses
            for loss_type in ['dynamics', 'reward', 'inverse', 'contrastive']:
                losses = self.training_stats[f'{loss_type}_losses']
                if losses:
                    stats[f'avg_{loss_type}_loss'] = sum(losses) / len(losses)
            
            # Add curiosity stats
            if self.training_stats['curiosity_rewards']:
                stats['avg_curiosity_reward'] = sum(self.training_stats['curiosity_rewards']) / len(self.training_stats['curiosity_rewards'])
        
        return stats
    
    def save_model(self, path: str):
        """FIXED: Save model state with atomic write and retry"""
        try:
            # Convert deques to lists for serialization
            training_stats_to_save = {}
            for key, value in self.training_stats.items():
                if isinstance(value, deque):
                    training_stats_to_save[key] = list(value)
                else:
                    training_stats_to_save[key] = value

            data_to_save = {
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'training_stats': training_stats_to_save,
                'ensemble_size': self.ensemble_size
            }
            
            # Serialize model data to bytes
            buffer = io.BytesIO()
            # Use pickle_module=pickle to ensure deques are handled if any remain
            torch.save(data_to_save, buffer, pickle_module=pickle)
            model_data = buffer.getvalue()
            buffer.close()
            
            # Use atomic write with retry
            success = atomic_write_with_retry(
                data=model_data,
                target_path=path,
                max_retries=5,
                retry_delay=0.1,
                suffix='.pt' # Ensure correct suffix
            )
            
            if not success:
                # The exception would have been raised by atomic_write_with_retry
                raise RuntimeError(f"Failed to save model to {path} after retries")

            logger.info(f"Saved world model to {path}")

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise # Re-raise the exception so callers are aware
    
    def load_model(self, path: str):
        """FIXED: Load model state with proper deserialization"""
        # Load with weights_only=False to handle deque objects
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Reconstruct deques from lists
        saved_stats = checkpoint['training_stats']
        for key in ['dynamics_losses', 'reward_losses', 'inverse_losses', 
                    'contrastive_losses', 'curiosity_rewards']:
            if key in saved_stats and isinstance(saved_stats[key], list):
                self.training_stats[key] = deque(saved_stats[key], maxlen=100)
            elif key in saved_stats:
                self.training_stats[key] = saved_stats[key]
        
        # Restore scalar values
        self.training_stats['total_steps'] = saved_stats.get('total_steps', 0)
        
        logger.info(f"Loaded world model from {path}")
    
    def shutdown(self):
        """Clean shutdown"""
        logger.info("Shutting down world model...")
        if hasattr(self, 'param_history'):
            try:
                self.param_history.shutdown()
            except Exception as e:
                logger.error(f"Failed to shutdown param history: {e}")
        logger.info("World model shutdown complete")

# ============================================================
# ATTENTION BLOCK
# ============================================================

class AttentionBlock(nn.Module):
    """Attention block for dynamics model"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.attention = MultiHeadAttention(dim, num_heads=4)
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """FIXED: Robustly handle dimensions"""
        original_shape = x.shape
        
        # Ensure 3D: [batch, seq, dim]
        if len(x.shape) == 1:
            x = x.unsqueeze(0).unsqueeze(0)
            needs_squeeze = (True, True)
        elif len(x.shape) == 2:
            x = x.unsqueeze(1)
            needs_squeeze = (False, True)
        else:
            needs_squeeze = (False, False)
        
        # Self-attention with residual
        attn_out = self.attention(x)
        x = self.norm(x + attn_out)
        
        # Feedforward with residual
        ff_out = self.ff(x)
        x = self.norm(x + ff_out)
        
        # Restore original dimensions
        if needs_squeeze[1]:
            x = x.squeeze(1)
        if needs_squeeze[0]:
            x = x.squeeze(0)
        
        return x

# ============================================================
# CURIOSITY MODULE
# ============================================================

class CuriosityModule(nn.Module):
    """Intrinsic curiosity module for exploration"""
    
    def __init__(self, state_dim: int):
        super().__init__()
        self.state_dim = state_dim
        
        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2)
        )
        
        # FIXED: Forward model with correct dimensions
        self.forward_model = nn.Sequential(
            nn.Linear(HIDDEN_DIM // 2 + state_dim, HIDDEN_DIM),  # encoded_state + action
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2)  # Output matches feature dimension
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor, 
                next_state: torch.Tensor) -> torch.Tensor:
        """Compute curiosity reward based on prediction error"""
        
        # Encode states
        state_feat = self.feature_encoder(state)
        next_state_feat = self.feature_encoder(next_state)
        
        # Predict next state features
        state_action = torch.cat([state_feat, action], dim=-1)
        pred_next_feat = self.forward_model(state_action)
        
        # Curiosity = prediction error
        curiosity = F.mse_loss(pred_next_feat, next_state_feat, reduction='none').mean(dim=-1)
        
        return curiosity

# ============================================================
# STATE ABSTRACTION
# ============================================================

class StateAbstractor(nn.Module):
    """Learn hierarchical state abstractions"""
    
    def __init__(self, state_dim: int, num_levels: int = 3):
        super().__init__()
        self.state_dim = state_dim
        self.num_levels = num_levels
        
        # Hierarchical encoders
        self.encoders = nn.ModuleList()
        current_dim = state_dim
        
        for level in range(num_levels):
            next_dim = max(current_dim // 2, 32)  # Ensure minimum dimension
            self.encoders.append(nn.Sequential(
                nn.Linear(current_dim, next_dim),
                nn.LayerNorm(next_dim),
                nn.ReLU()
            ))
            current_dim = next_dim
        
        # Decoders for reconstruction
        self.decoders = nn.ModuleList()
        
        for level in reversed(range(num_levels)):
            in_dim = state_dim // (2 ** (level + 1))
            in_dim = max(in_dim, 32)
            out_dim = state_dim // (2 ** level) if level > 0 else state_dim
            out_dim = max(out_dim, 32)
            
            self.decoders.append(nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.ReLU()
            ))
    
    def forward(self, state: torch.Tensor, level: int = -1) -> torch.Tensor:
        """Get abstract representation at specified level"""
        
        # Encode to desired level
        x = state
        abstractions = [x]
        
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            abstractions.append(x)
            
            if i == level:
                break
        
        # Return requested abstraction level
        if level == -1:
            return abstractions[-1]  # Most abstract
        else:
            return abstractions[min(level + 1, len(abstractions) - 1)]
    
    def reconstruct(self, abstract_state: torch.Tensor, level: int) -> torch.Tensor:
        """Reconstruct state from abstraction"""
        
        x = abstract_state
        
        # Decode from level
        start_idx = max(0, self.num_levels - level - 1)
        for decoder in self.decoders[start_idx:]:
            x = decoder(x)
        
        return x