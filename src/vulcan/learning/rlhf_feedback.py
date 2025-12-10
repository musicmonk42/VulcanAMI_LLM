"""
RLHF (Reinforcement Learning from Human Feedback) and live feedback processing
"""

import asyncio
import logging
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ..config import EMBEDDING_DIM, HIDDEN_DIM
from .learning_types import FeedbackData, LearningConfig

logger = logging.getLogger(__name__)

# ============================================================
# RLHF IMPLEMENTATION
# ============================================================


class RLHFManager:
    """Reinforcement Learning from Human Feedback manager"""

    def __init__(self, base_model: nn.Module, config: LearningConfig = None):
        self.config = config or LearningConfig()
        self.base_model = base_model

        # FIXED: Initialize thread safety locks FIRST before any threads
        self._lock = threading.RLock()
        self._buffer_lock = threading.RLock()

        # FIXED: Get device from base model
        self.device = (
            next(base_model.parameters()).device
            if hasattr(base_model, "parameters")
            else torch.device("cpu")
        )

        # FIX: Get embedding dimension from base model, fallback to global constant
        self.embedding_dim = getattr(base_model, "embedding_dim", EMBEDDING_DIM)

        # Feedback buffer
        self.feedback_buffer = deque(maxlen=self.config.feedback_buffer_size)
        self.processed_feedback = deque(maxlen=10000)

        # Reward model
        self.reward_model = self._build_reward_model()
        self.reward_optimizer = optim.Adam(self.reward_model.parameters(), lr=0.0001)

        # PPO components for policy optimization
        self.value_model = self._build_value_model()
        self.value_optimizer = optim.Adam(self.value_model.parameters(), lr=0.0001)

        # Policy head for action distribution
        self.policy_head = self._build_policy_head()
        self.policy_optimizer = optim.Adam(
            list(self.base_model.parameters()) + list(self.policy_head.parameters()),
            lr=self.config.learning_rate,
        )

        # Feedback statistics
        self.feedback_stats = {
            "total_feedback": 0,
            "positive_feedback": 0,
            "negative_feedback": 0,
            "corrections": 0,
            "preferences": 0,
            "reward_model_updates": 0,
            "policy_updates": 0,
            "api_fetches": 0,
        }

        # API configuration
        self.feedback_api_endpoint = "/api/feedback"
        self.governance_api_endpoint = "/api/governance"
        self.api_base_url = "http://localhost:8000"  # Configurable
        self.api_key = None  # Set via environment variable
        self.api_session = None

        # Feedback ranking for preference learning
        self.preference_pairs = deque(maxlen=5000)

        # FIXED: Track if we're shutting down for async operations
        self._is_shutdown = False

        # Background processor - Start AFTER locks are initialized
        self.feedback_processor = ThreadPoolExecutor(max_workers=2)
        self._shutdown_event = threading.Event()
        self._processing_thread = self._start_feedback_processing()

    def _build_reward_model(self) -> nn.Module:
        """Build reward model for human preferences"""
        model = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM // 2, 1),
            nn.Tanh(),  # Bound rewards to [-1, 1]
        ).to(self.device)
        # Ensure all parameters have gradients enabled
        for param in model.parameters():
            param.requires_grad = True
        return model

    def _build_value_model(self) -> nn.Module:
        """Build value model for PPO"""
        model = nn.Sequential(
            nn.Linear(self.embedding_dim, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM // 2, 1),
        ).to(self.device)
        # Ensure all parameters have gradients enabled
        for param in model.parameters():
            param.requires_grad = True
        return model

    def _build_policy_head(self) -> nn.Module:
        """Build policy head for action distribution"""
        model = nn.Sequential(
            nn.Linear(self.embedding_dim, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, self.embedding_dim),  # Output action space
            nn.Tanh(),
        ).to(self.device)
        # Ensure all parameters have gradients enabled
        for param in model.parameters():
            param.requires_grad = True
        return model

    def receive_feedback(self, feedback: FeedbackData):
        """FIXED: Receive and queue human feedback with proper locking"""
        with self._buffer_lock:
            self.feedback_buffer.append(feedback)

            with self._lock:
                self.feedback_stats["total_feedback"] += 1

                # Categorize feedback
                if feedback.reward_signal > 0:
                    self.feedback_stats["positive_feedback"] += 1
                else:
                    self.feedback_stats["negative_feedback"] += 1

                if feedback.feedback_type == "correction":
                    self.feedback_stats["corrections"] += 1
                elif feedback.feedback_type == "preference":
                    self.feedback_stats["preferences"] += 1

            # Store preference pairs for ranking
            if (
                feedback.feedback_type == "preference"
                and "preferred_over" in feedback.metadata
            ):
                self.preference_pairs.append(
                    {
                        "preferred": feedback.agent_response,
                        "rejected": feedback.metadata["preferred_over"],
                        "context": feedback.context,
                    }
                )

            # Check if should trigger processing
            should_process = (
                len(self.feedback_buffer) >= self.config.reward_model_update_freq
            )

        # Trigger processing outside the lock
        if should_process and not self._shutdown_event.is_set():
            try:
                self.feedback_processor.submit(self._process_feedback_batch)
            except Exception as e:
                logger.error(f"Failed to submit feedback processing: {e}")

    def _process_feedback_batch(self):
        """FIXED: Process a batch of feedback with proper locking"""
        with self._buffer_lock:
            if len(self.feedback_buffer) < 10:
                return

            # Sample feedback batch
            batch_size = min(self.config.batch_size, len(self.feedback_buffer))
            batch = [self.feedback_buffer.popleft() for _ in range(batch_size)]

        # Process outside the lock to avoid blocking
        try:
            # Separate preference pairs and direct rewards
            preference_batch = []
            reward_batch = []

            for feedback in batch:
                if (
                    feedback.feedback_type == "preference"
                    and "preferred_over" in feedback.metadata
                ):
                    preference_batch.append(feedback)
                else:
                    reward_batch.append(feedback)

            # Update reward model with direct rewards
            if reward_batch:
                self._update_reward_model_direct(reward_batch)

            # Update reward model with preferences
            if preference_batch:
                self._update_reward_model_preferences(preference_batch)

            with self._lock:
                self.feedback_stats["reward_model_updates"] += 1

            with self._buffer_lock:
                self.processed_feedback.extend(batch)
        except Exception as e:
            logger.error(f"Error processing feedback batch: {e}")

    def _update_reward_model_direct(self, batch: List[FeedbackData]):
        """Update reward model with direct reward signals"""
        inputs = []
        targets = []

        for feedback in batch:
            try:
                # Extract features from feedback
                agent_feat = self._extract_features(feedback.agent_response)
                human_feat = self._extract_features(feedback.human_preference)

                combined = torch.cat([agent_feat, human_feat])
                inputs.append(combined)
                targets.append(feedback.reward_signal)
            except Exception as e:
                logger.warning(f"Failed to extract features from feedback: {e}")
                continue

        if not inputs:
            return

        inputs = (
            torch.stack(inputs).to(self.device).detach()
        )  # Detach inputs, they're not trainable
        targets = torch.tensor(
            targets, dtype=torch.float32, device=self.device
        ).unsqueeze(1)

        # Update reward model
        self.reward_model.train()  # Ensure model is in training mode
        for _ in range(self.config.ppo_epochs):
            pred_rewards = self.reward_model(inputs)
            loss = F.mse_loss(pred_rewards, targets)

            self.reward_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.reward_model.parameters(), 1.0)
            self.reward_optimizer.step()

        logger.info(
            f"Updated reward model with {len(batch)} direct feedback, loss: {loss.item():.4f}"
        )

    def _update_reward_model_preferences(self, batch: List[FeedbackData]):
        """Update reward model using preference learning (Bradley-Terry model)"""
        preference_loss = 0
        num_pairs = 0

        # Ensure model is in training mode
        self.reward_model.train()

        for feedback in batch:
            try:
                if "preferred_over" not in feedback.metadata:
                    continue

                # Get features for preferred and rejected
                preferred_feat = self._extract_features(
                    feedback.agent_response
                ).detach()
                rejected_feat = self._extract_features(
                    feedback.metadata["preferred_over"]
                ).detach()
                context_feat = self._extract_features(feedback.context).detach()

                # Compute rewards
                preferred_input = torch.cat([preferred_feat, context_feat]).to(
                    self.device
                )
                rejected_input = torch.cat([rejected_feat, context_feat]).to(
                    self.device
                )

                preferred_reward = self.reward_model(preferred_input.unsqueeze(0))
                rejected_reward = self.reward_model(rejected_input.unsqueeze(0))

                # Bradley-Terry loss: -log(sigma(r_preferred - r_rejected))
                loss = -F.logsigmoid(preferred_reward - rejected_reward).mean()
                preference_loss += loss
                num_pairs += 1
            except Exception as e:
                logger.warning(f"Failed to process preference pair: {e}")
                continue

        if num_pairs > 0:
            preference_loss = preference_loss / num_pairs

            self.reward_optimizer.zero_grad()
            preference_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.reward_model.parameters(), 1.0)
            self.reward_optimizer.step()

            logger.info(
                f"Updated reward model with {num_pairs} preference pairs, loss: {preference_loss.item():.4f}"
            )

    def _extract_features(self, data: Any) -> torch.Tensor:
        """FIXED: Extract features with device management and consistent dimensions"""
        if isinstance(data, torch.Tensor):
            # Ensure 1D tensor of size embedding_dim
            if data.numel() == self.embedding_dim:
                result = data.reshape(self.embedding_dim)
            elif data.numel() < self.embedding_dim:
                # Pad with zeros
                result = F.pad(data.flatten(), (0, self.embedding_dim - data.numel()))
            else:
                # Truncate
                result = data.flatten()[: self.embedding_dim]
            return result.to(self.device)
        elif isinstance(data, np.ndarray):
            # Convert numpy array to tensor and resize
            flat_data = data.flatten()
            if len(flat_data) < self.embedding_dim:
                flat_data = np.pad(flat_data, (0, self.embedding_dim - len(flat_data)))
            elif len(flat_data) > self.embedding_dim:
                flat_data = flat_data[: self.embedding_dim]
            return torch.tensor(flat_data, dtype=torch.float32, device=self.device)
        elif isinstance(data, dict) and "embedding" in data:
            return self._extract_features(data["embedding"])
        else:
            # Default to random features (should be replaced with proper encoding)
            return torch.randn(self.embedding_dim, device=self.device)

    def update_policy_with_ppo(self, trajectories: List[Dict[str, Any]]):
        """FIXED: Update policy using PPO with proper dimension handling and gradient management"""
        if not trajectories:
            return

        all_states = []
        all_actions = []
        all_old_log_probs = []
        all_rewards = []
        all_advantages = []

        # Collect all trajectory data
        for trajectory in trajectories:
            try:
                # FIXED: Ensure proper dimensions for states and actions
                states = trajectory["states"]
                actions = trajectory["actions"]
                log_probs = trajectory["log_probs"]

                # FIXED: Convert to tensors properly using detach().clone()
                if not isinstance(states, torch.Tensor):
                    states = torch.stack(
                        [
                            s.detach().clone()
                            if isinstance(s, torch.Tensor)
                            else torch.tensor(s, dtype=torch.float32)
                            for s in states
                        ]
                    )
                if not isinstance(actions, torch.Tensor):
                    actions = torch.stack(
                        [
                            a.detach().clone()
                            if isinstance(a, torch.Tensor)
                            else torch.tensor(a, dtype=torch.float32)
                            for a in actions
                        ]
                    )
                if not isinstance(log_probs, torch.Tensor):
                    log_probs = torch.stack(
                        [
                            lp.detach().clone()
                            if isinstance(lp, torch.Tensor)
                            else torch.tensor(lp, dtype=torch.float32)
                            for lp in log_probs
                        ]
                    )

                states = states.to(self.device)
                actions = actions.to(self.device)
                log_probs = log_probs.to(self.device)

                # FIXED: Ensure states and actions have correct dimensions
                # States should be [seq_len, embedding_dim]
                if states.dim() == 1:
                    states = states.unsqueeze(0)
                if actions.dim() == 1:
                    actions = actions.unsqueeze(0)

                # Ensure correct feature dimension
                if states.shape[-1] != self.embedding_dim:
                    # Pad or truncate to embedding_dim
                    if states.shape[-1] < self.embedding_dim:
                        states = F.pad(
                            states, (0, self.embedding_dim - states.shape[-1])
                        )
                    else:
                        states = states[..., : self.embedding_dim]

                if actions.shape[-1] != self.embedding_dim:
                    if actions.shape[-1] < self.embedding_dim:
                        actions = F.pad(
                            actions, (0, self.embedding_dim - actions.shape[-1])
                        )
                    else:
                        actions = actions[..., : self.embedding_dim]

                # Get rewards from reward model
                with torch.no_grad():
                    # FIXED: Concatenate for reward model (expects 2*embedding_dim)
                    state_action_pairs = torch.cat([states, actions], dim=-1)
                    human_rewards = self.reward_model(state_action_pairs).squeeze(-1)

                # Compute advantages using GAE
                values = self.value_model(states).squeeze(-1)
                advantages = self._compute_advantages(human_rewards, values)

                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

                all_states.append(states)
                all_actions.append(actions)
                all_old_log_probs.append(log_probs)
                all_rewards.append(human_rewards)
                all_advantages.append(advantages)
            except Exception as e:
                logger.error(f"Failed to process trajectory: {e}")
                continue

        if not all_states:
            logger.warning("No valid trajectories to process")
            return

        # Concatenate all data - detach since they're inputs
        all_states = torch.cat(all_states, dim=0).detach()
        all_actions = torch.cat(all_actions, dim=0).detach()
        all_old_log_probs = torch.cat(all_old_log_probs, dim=0).detach()
        all_rewards = torch.cat(all_rewards, dim=0).detach()
        all_advantages = torch.cat(all_advantages, dim=0).detach()

        # Ensure models are in training mode
        self.base_model.train()
        self.policy_head.train()
        self.value_model.train()

        # PPO epochs
        for epoch in range(self.config.ppo_epochs):
            # Shuffle data for each epoch
            perm = torch.randperm(len(all_states))

            for i in range(0, len(all_states), self.config.batch_size):
                batch_indices = perm[i : i + self.config.batch_size]

                batch_states = all_states[batch_indices]
                batch_actions = all_actions[batch_indices]
                batch_old_log_probs = all_old_log_probs[batch_indices]
                batch_advantages = all_advantages[batch_indices]
                batch_rewards = all_rewards[batch_indices]

                # Compute new log probs and entropy
                new_log_probs, entropy = self._compute_log_probs(
                    batch_states, batch_actions
                )

                # Compute ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # Clipped objective
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(
                        ratio, 1 - self.config.ppo_clip, 1 + self.config.ppo_clip
                    )
                    * batch_advantages
                )

                policy_loss = -torch.min(surr1, surr2).mean()

                # FIXED: Value loss with proper dimension handling
                value_pred = self.value_model(batch_states).squeeze()
                # Ensure batch_rewards has same shape as value_pred
                if value_pred.dim() != batch_rewards.dim():
                    if batch_rewards.dim() == 0:
                        batch_rewards = batch_rewards.unsqueeze(0)
                    elif value_pred.dim() == 0:
                        value_pred = value_pred.unsqueeze(0)
                value_loss = F.mse_loss(value_pred, batch_rewards)

                # Entropy bonus for exploration
                entropy_loss = -entropy.mean() * 0.01

                # Total loss
                total_loss = policy_loss + 0.5 * value_loss + entropy_loss

                # Optimization step
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()

                # FIXED: Single backward pass without retain_graph
                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.base_model.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.policy_head.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), 1.0)

                self.policy_optimizer.step()
                self.value_optimizer.step()

        with self._lock:
            self.feedback_stats["policy_updates"] += 1
        logger.info(
            f"PPO update complete: policy_loss={policy_loss.item():.4f}, "
            f"value_loss={value_loss.item():.4f}"
        )

    def _compute_log_probs(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """FIXED: Compute log probabilities with proper dimension handling"""
        # Process states through base model
        if hasattr(self.base_model, "forward"):
            # FIXED: Handle different model architectures
            try:
                state_features = self.base_model(states)
            except Exception as e:
                # If base model expects different input, use states directly
                logger.debug(f"Base model forward failed: {e}, using states directly")
                state_features = states
        else:
            state_features = states

        # Ensure state_features has correct dimensions
        if state_features.shape[-1] != self.embedding_dim:
            if state_features.shape[-1] < self.embedding_dim:
                state_features = F.pad(
                    state_features, (0, self.embedding_dim - state_features.shape[-1])
                )
            else:
                state_features = state_features[..., : self.embedding_dim]

        # Get action distribution from policy head
        action_means = self.policy_head(state_features)

        # Assume Gaussian distribution for continuous actions
        action_std = torch.ones_like(action_means) * 0.5  # Could be learned

        # Create distribution
        action_dist = dist.Normal(action_means, action_std)

        # Compute log probabilities
        log_probs = action_dist.log_prob(actions).sum(dim=-1)

        # Compute entropy
        entropy = action_dist.entropy().sum(dim=-1)

        return log_probs, entropy

    def _compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> torch.Tensor:
        """Compute GAE (Generalized Advantage Estimation) advantages"""
        advantages = torch.zeros_like(rewards)
        last_advantage = 0

        for t in reversed(range(len(rewards))
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + gamma * next_value - values[t]
            advantages[t] = last_advantage = delta + gamma * lam * last_advantage

        return advantages

    def _start_feedback_processing(self) -> threading.Thread:
        """Start background feedback processing with shutdown support"""

        def process_loop():
            while not self._shutdown_event.is_set():
                try:
                    # Check for feedback to process
                    with self._buffer_lock:
                        should_process = (
                            len(self.feedback_buffer) >= self.config.batch_size
                        )

                    if should_process:
                        self._process_feedback_batch()

                    # Sleep with shutdown check
                    for _ in range(10):  # Check every second for 10 seconds
                        if self._shutdown_event.is_set():
                            break
                        time.sleep(1)

                except Exception as e:
                    if not self._shutdown_event.is_set():
                        logger.error(f"Feedback processing error: {e}")

            logger.info("Feedback processing thread stopped")

        thread = threading.Thread(target=process_loop, daemon=True)
        thread.start()
        return thread

    def sync_with_governance(self, governance_decisions: List[Dict[str, Any]]):
        """Synchronize with governance decisions"""
        for decision in governance_decisions:
            try:
                # Convert governance decision to feedback
                feedback = FeedbackData(
                    feedback_id=f"gov_{decision.get('id', time.time())}",
                    timestamp=time.time(),
                    feedback_type="governance",
                    content=decision,
                    context={"source": "governance"},
                    agent_response=decision.get("agent_action"),
                    human_preference=decision.get("approved_action"),
                    reward_signal=1.0 if decision.get("approved", False) else -1.0,
                    metadata=decision,
                )

                self.receive_feedback(feedback)
            except Exception as e:
                logger.error(f"Failed to process governance decision: {e}")

    async def fetch_feedback_from_api(
        self, since_timestamp: Optional[float] = None, limit: int = 100
    ) -> List[FeedbackData]:
        """Fetch feedback from API endpoint"""
        if self._is_shutdown:
            logger.warning("Cannot fetch feedback: manager is shutdown")
            return []

        if not self.api_session:
            self.api_session = aiohttp.ClientSession()

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        params = {"limit": limit}
        if since_timestamp:
            params["since"] = since_timestamp

        try:
            url = f"{self.api_base_url}{self.feedback_api_endpoint}"
            async with self.api_session.get(
                url, headers=headers, params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()

                    # Convert API response to FeedbackData objects
                    feedback_list = []
                    for item in data.get("feedback", []):
                        try:
                            feedback = FeedbackData(
                                feedback_id=item["id"],
                                timestamp=item["timestamp"],
                                feedback_type=item["type"],
                                content=item["content"],
                                context=item.get("context", {}),
                                agent_response=item.get("agent_response"),
                                human_preference=item.get("human_preference"),
                                reward_signal=item.get("reward_signal", 0.0),
                                metadata=item.get("metadata", {}),
                            )
                            feedback_list.append(feedback)
                        except Exception as e:
                            logger.warning(f"Failed to parse feedback item: {e}")
                            continue

                    with self._lock:
                        self.feedback_stats["api_fetches"] += 1
                    logger.info(f"Fetched {len(feedback_list)} feedback items from API")

                    # Add to buffer
                    for feedback in feedback_list:
                        self.receive_feedback(feedback)

                    return feedback_list
                else:
                    logger.error(f"API request failed with status {response.status}")
                    return []

        except Exception as e:
            logger.error(f"Failed to fetch feedback from API: {e}")
            return []

    async def push_metrics_to_api(self, metrics: Dict[str, Any]):
        """Push performance metrics to API for dashboard"""
        if self._is_shutdown:
            return

        if not self.api_session:
            self.api_session = aiohttp.ClientSession()

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            url = f"{self.api_base_url}/api/metrics"
            async with self.api_session.post(
                url, json=metrics, headers=headers
            ) as response:
                if response.status == 200:
                    logger.debug("Successfully pushed metrics to API")
                else:
                    logger.error(f"Failed to push metrics: {response.status}")

        except Exception as e:
            logger.error(f"Failed to push metrics to API: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get RLHF statistics"""
        with self._lock:
            stats = self.feedback_stats.copy()

        with self._buffer_lock:
            stats["buffer_size"] = len(self.feedback_buffer)
            stats["processed_count"] = len(self.processed_feedback)
            stats["preference_pairs"] = len(self.preference_pairs)

        return stats

    def shutdown(self):
        """FIXED: Clean shutdown of RLHF manager"""
        logger.info("Shutting down RLHF manager...")

        # Mark as shutdown to prevent new operations
        self._is_shutdown = True

        # Signal shutdown
        self._shutdown_event.set()

        # Wait for processing thread
        if self._processing_thread and self._processing_thread.is_alive():
            logger.info("Waiting for processing thread...")
            self._processing_thread.join(timeout=5)
            if self._processing_thread.is_alive():
                logger.warning("Processing thread did not terminate in time")

        # Shutdown executor and wait for pending tasks
        logger.info("Shutting down feedback processor...")
        self.feedback_processor.shutdown(wait=True)

        # FIXED: Close API session properly
        if self.api_session and not self.api_session.closed:
            logger.info("Closing API session...")
            try:
                # Create a new event loop for cleanup if needed
                try:
                    loop = asyncio.get_running_loop()
                    # If we're in a running loop, we can't use run_until_complete
                    # Schedule the close and continue
                    asyncio.create_task(self.api_session.close())
                    logger.info("Scheduled API session close")
                except RuntimeError:
                    # No running loop, safe to create one
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(self.api_session.close())
                        logger.info("API session closed successfully")
                    finally:
                        loop.close()
            except Exception as e:
                logger.error(f"Failed to close API session: {e}")

        logger.info("RLHF manager shutdown complete")


# ============================================================
# LIVE FEEDBACK PROCESSOR
# ============================================================


class LiveFeedbackProcessor:
    """Process live feedback and performance data"""

    def __init__(self, model: nn.Module, config: LearningConfig = None):
        self.config = config or LearningConfig()
        self.model = model
        self.feedback_queue = None  # Will be created when async loop starts
        self.performance_buffer = deque(maxlen=1000)

        # Real-time metrics
        self.realtime_metrics = {
            "latency_ms": deque(maxlen=100),
            "accuracy": deque(maxlen=100),
            "user_satisfaction": deque(maxlen=100),
            "memory_usage_mb": deque(maxlen=100),
            "throughput_qps": deque(maxlen=100),
        }

        # Adaptive learning rates based on performance
        self.adaptive_lr = self.config.learning_rate
        self.lr_adjustment_history = []

        # Performance tracking
        self.performance_tracker = {
            "start_time": time.time(),
            "total_predictions": 0,
            "correct_predictions": 0,
            "total_latency": 0,
            "feedback_processed": 0,
        }

        # Alert thresholds
        self.alert_thresholds = {
            "latency_ms": 100,
            "accuracy": 0.7,
            "user_satisfaction": 0.5,
            "memory_usage_mb": 1000,
        }

        # Monitoring state
        self._monitoring_task = None
        self._shutdown_event = None  # Will be created in async context

        # Retraining configuration
        self.retraining_config = {
            "enabled": True,
            "min_samples": 1000,
            "accuracy_threshold": 0.75,
            "degradation_threshold": 0.1,
        }

        # Thread safety
        self._lock = threading.RLock()

    async def start_monitoring(self):
        """Start performance monitoring"""
        if self._shutdown_event is None:
            self._shutdown_event = asyncio.Event()

        if self.feedback_queue is None:
            self.feedback_queue = asyncio.Queue()

        if not self._monitoring_task:
            self._monitoring_task = asyncio.create_task(
                self.performance_monitoring_loop()
            )
            logger.info("Started performance monitoring")

    async def stop_monitoring(self):
        """Stop performance monitoring"""
        if self._shutdown_event:
            self._shutdown_event.set()

        if self._monitoring_task:
            try:
                await asyncio.wait_for(self._monitoring_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Monitoring task did not complete in time")
                self._monitoring_task.cancel()
            logger.info("Stopped performance monitoring")

    async def process_live_feedback(self, feedback: Dict[str, Any]):
        """Process feedback in real-time"""
        if self.feedback_queue is None:
            self.feedback_queue = asyncio.Queue()

        await self.feedback_queue.put(feedback)

        # Update tracker
        with self._lock:
            self.performance_tracker["feedback_processed"] += 1

        # Immediate adjustment for critical feedback
        if feedback.get("priority") == "critical":
            await self._immediate_adjustment(feedback)

        # Regular processing
        await self._process_feedback(feedback)

    async def _process_feedback(self, feedback: Dict[str, Any]):
        """Process individual feedback item"""
        feedback_type = feedback.get("type")

        with self._lock:
            if feedback_type == "prediction_result":
                # Update accuracy tracking
                is_correct = feedback.get("correct", False)
                if is_correct:
                    self.performance_tracker["correct_predictions"] += 1
                self.performance_tracker["total_predictions"] += 1

                # Update accuracy metric
                if self.performance_tracker["total_predictions"] > 0:
                    accuracy = (
                        self.performance_tracker["correct_predictions"]
                        / self.performance_tracker["total_predictions"]
                    )
                    self.realtime_metrics["accuracy"].append(accuracy)

            elif feedback_type == "user_rating":
                # Update user satisfaction
                rating = feedback.get("rating", 0.5)
                self.realtime_metrics["user_satisfaction"].append(rating)

            elif feedback_type == "latency":
                # Update latency tracking
                latency = feedback.get("value", 0)
                self.performance_tracker["total_latency"] += latency
                self.realtime_metrics["latency_ms"].append(latency)

    async def _immediate_adjustment(self, feedback: Dict[str, Any]):
        """Make immediate adjustments based on critical feedback"""
        adjustment_made = False
        reason = feedback.get("type", "unknown")

        with self._lock:
            if feedback.get("type") == "error":
                # Reduce learning rate for stability
                self.adaptive_lr *= 0.5
                adjustment_made = True
                logger.warning(
                    f"Critical error feedback: reducing LR to {self.adaptive_lr}"
                )

            elif feedback.get("type") == "performance":
                performance = feedback.get("value", 0.5)
                if performance < 0.3:
                    # Poor performance - increase exploration
                    self.adaptive_lr = min(self.adaptive_lr * 1.2, 0.1)
                    adjustment_made = True
                elif performance > 0.9:
                    # Great performance - fine-tune
                    self.adaptive_lr = max(self.adaptive_lr * 0.9, 1e-5)
                    adjustment_made = True

            elif feedback.get("type") == "memory_pressure":
                # Memory issue - reduce batch size or model complexity
                logger.warning("Memory pressure detected - triggering optimization")
                adjustment_made = True
                reason = "memory_pressure"

            if adjustment_made:
                self.lr_adjustment_history.append(
                    {
                        "timestamp": time.time(),
                        "old_lr": self.config.learning_rate,
                        "new_lr": self.adaptive_lr,
                        "reason": reason,
                    }
                )

    async def performance_monitoring_loop(self):
        """Continuous performance monitoring"""
        while self._shutdown_event and not self._shutdown_event.is_set():
            try:
                # Collect performance metrics
                current_metrics = await self._collect_metrics()
                self.performance_buffer.append(current_metrics)

                # Update realtime metrics
                with self._lock:
                    for key, value in current_metrics.items():
                        if key in self.realtime_metrics:
                            self.realtime_metrics[key].append(value)

                # Check for alerts
                alerts = self._check_alerts(current_metrics)
                if alerts:
                    await self._handle_alerts(alerts)

                # Check for performance degradation
                if self._detect_degradation():
                    await self._trigger_retraining()

                # Wait before next check
                await asyncio.sleep(1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")

        logger.info("Performance monitoring loop stopped")

    async def _collect_metrics(self) -> Dict[str, float]:
        """Collect current performance metrics"""
        import gc

        import psutil

        try:
            # Get process info
            process = psutil.Process()

            # Collect real metrics
            metrics = {
                "latency_ms": self._get_average_latency(),
                "accuracy": self._get_current_accuracy(),
                "user_satisfaction": self._get_average_satisfaction(),
                "memory_usage_mb": process.memory_info().rss / 1024 / 1024,
                "cpu_percent": process.cpu_percent(),
                "throughput_qps": self._calculate_throughput(),
            }

            # Add model-specific metrics if available
            if hasattr(self.model, "training"):
                metrics["is_training"] = self.model.training

            # Garbage collection stats
            gc_stats = gc.get_stats()
            if gc_stats:
                metrics["gc_collections"] = sum(
                    s.get("collections", 0) for s in gc_stats
                )

            return metrics
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
            return {}

    def _get_average_latency(self) -> float:
        """Calculate average latency"""
        with self._lock:
            if self.realtime_metrics["latency_ms"]:
                return np.mean(list(self.realtime_metrics["latency_ms"]))
        return 0.0

    def _get_current_accuracy(self) -> float:
        """Get current accuracy"""
        with self._lock:
            if self.performance_tracker["total_predictions"] > 0:
                return (
                    self.performance_tracker["correct_predictions"]
                    / self.performance_tracker["total_predictions"]
                )
        return 1.0  # Assume perfect accuracy if no predictions yet

    def _get_average_satisfaction(self) -> float:
        """Get average user satisfaction"""
        with self._lock:
            if self.realtime_metrics["user_satisfaction"]:
                return np.mean(list(self.realtime_metrics["user_satisfaction"]))
        return 1.0  # Assume perfect satisfaction if no ratings

    def _calculate_throughput(self) -> float:
        """Calculate queries per second"""
        with self._lock:
            elapsed = time.time() - self.performance_tracker["start_time"]
            if elapsed > 0:
                return self.performance_tracker["total_predictions"] / elapsed
        return 0.0

    def _check_alerts(self, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Check if any metrics exceed alert thresholds"""
        alerts = []

        for metric, threshold in self.alert_thresholds.items():
            if metric in metrics:
                value = metrics[metric]

                # Different comparison for different metrics
                if metric in ["accuracy", "user_satisfaction"]:
                    # Lower is bad
                    if value < threshold:
                        alerts.append(
                            {
                                "metric": metric,
                                "value": value,
                                "threshold": threshold,
                                "type": "below_threshold",
                            }
                        )
                else:
                    # Higher is bad
                    if value > threshold:
                        alerts.append(
                            {
                                "metric": metric,
                                "value": value,
                                "threshold": threshold,
                                "type": "above_threshold",
                            }
                        )

        return alerts

    async def _handle_alerts(self, alerts: List[Dict[str, Any]]):
        """Handle performance alerts"""
        for alert in alerts:
            logger.warning(
                f"Performance alert: {alert['metric']} = {alert['value']:.2f} "
                f"({alert['type']} threshold {alert['threshold']})"
            )

            # Take action based on alert type
            if (
                alert["metric"] == "memory_usage_mb"
                and alert["type"] == "above_threshold"
            ):
                # Trigger garbage collection
                import gc

                gc.collect()
                logger.info("Triggered garbage collection due to memory pressure")

            elif alert["metric"] == "accuracy" and alert["type"] == "below_threshold":
                # Consider retraining
                if self.retraining_config["enabled"]:
                    await self._trigger_retraining()

    def _detect_degradation(self) -> bool:
        """Detect performance degradation"""
        with self._lock:
            if len(self.realtime_metrics["accuracy"]) < 20:
                return False

            recent = list(self.realtime_metrics["accuracy"])[-10:]
            older = list(self.realtime_metrics["accuracy"])[-20:-10]

        recent_mean = np.mean(recent)
        older_mean = np.mean(older)

        # Check if recent performance is significantly worse
        degradation = older_mean - recent_mean

        return degradation > self.retraining_config["degradation_threshold"]

    async def _trigger_retraining(self):
        """Trigger model retraining"""
        # Check if enough samples for retraining
        with self._lock:
            total_predictions = self.performance_tracker["total_predictions"]

        if total_predictions < self.retraining_config["min_samples"]:
            logger.info("Insufficient samples for retraining")
            return

        logger.info("Performance degradation detected, triggering retraining")

        # Create retraining task
        retraining_info = {
            "triggered_at": time.time(),
            "reason": "performance_degradation",
            "current_accuracy": self._get_current_accuracy(),
            "samples_available": total_predictions,
        }

        # In a real system, this would trigger a retraining pipeline
        # For now, just log the event
        logger.info(f"Retraining triggered: {retraining_info}")

        # Could emit event or call webhook
        await self._notify_retraining(retraining_info)

    async def _notify_retraining(self, info: Dict[str, Any]):
        """Notify external systems about retraining"""
        # This could send to a message queue, webhook, or API

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        with self._lock:
            summary = {
                "uptime_seconds": time.time() - self.performance_tracker["start_time"],
                "total_predictions": self.performance_tracker["total_predictions"],
                "accuracy": self._get_current_accuracy(),
                "average_latency_ms": self._get_average_latency(),
                "user_satisfaction": self._get_average_satisfaction(),
                "throughput_qps": self._calculate_throughput(),
                "adaptive_lr": self.adaptive_lr,
                "lr_adjustments": len(self.lr_adjustment_history),
                "feedback_processed": self.performance_tracker["feedback_processed"],
            }

            # Add recent metrics
            for metric_name, values in self.realtime_metrics.items():
                if values:
                    summary[f"{metric_name}_recent"] = list(values)[-10:]

        return summary

    def reset_metrics(self):
        """Reset performance metrics"""
        with self._lock:
            self.performance_tracker = {
                "start_time": time.time(),
                "total_predictions": 0,
                "correct_predictions": 0,
                "total_latency": 0,
                "feedback_processed": 0,
            }

            for metric in self.realtime_metrics.values():
                metric.clear()

        logger.info("Performance metrics reset")
