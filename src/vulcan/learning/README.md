VULCAN-AMI Learning Module
Overview
The Learning Module in VULCAN-AMI provides a unified framework for advanced learning paradigms, integrating continual learning, curriculum progression, meta-learning, reinforcement learning from human feedback (RLHF), world modeling, metacognition, and parameter auditing. It enables AI systems to learn continuously from experiences, adapt to new tasks, self-improve through reflection, plan actions via predictive models, and incorporate human preferences—all while maintaining audit trails for transparency and safety.
The core orchestrator, UnifiedLearningSystem, coordinates these components for , scalable learning. It supports modes like supervised, reinforcement, and online learning, with features for task detection, adaptive pacing, and hierarchical memory.
Key Features

Continual Learning: Handles catastrophic forgetting via Elastic Weight Consolidation (EWC), experience replay, progressive networks, and hierarchical memory.
Curriculum Learning: Adaptive difficulty progression with pacing strategies (e.g., threshold-based, self-paced) and learned difficulty estimation.
Meta-Learning: Fast adaptation to new tasks using algorithms like MAML, Reptile, and Prototypical Networks, with automatic task detection.
RLHF: Incorporates human feedback through PPO optimization, reward modeling, and live processing of ratings/preferences.
World Modeling: Predicts dynamics, rewards, and uncertainties for planning (e.g., MCTS, beam search) in simulated environments.
Metacognition: Monitors reasoning quality, detects weaknesses, and suggests self-improvements; includes compositional concept understanding.
Parameter Auditing: Tracks model parameter history with checkpoints, trajectories, and diff-based storage for compliance and debugging.
Mathematical Accuracy: Integrates verification feedback to improve mathematical reasoning and calculation accuracy.
Unified Orchestration: Seamless integration with async processing, thread safety, and persistence for long-running systems.

Architecture and Components
The module is structured around interconnected classes, each focusing on a learning aspect, unified in __init__.py:

__init__.py: Defines UnifiedLearningSystem as the main entry point, coordinating all subsystems with async/parallel processing.
continual_learning.py: Implements EnhancedContinualLearner for lifelong learning, with EWC, replay buffers, progressive growth, and RLHF integration. Includes ContinualLearner for backward compatibility.
curriculum_learning.py: Provides CurriculumLearner for staged learning, with difficulty estimators (e.g., LearnedDifficultyEstimator) and pacing enums like PacingStrategy.
learning_types.py: Defines core types like LearningConfig, TaskInfo, FeedbackData, LearningMode, and LearningTrajectory.
meta_learning.py: Features MetaLearner for meta-algorithms (e.g., MAML via MetaLearningAlgorithm) and TaskDetector for clustering-based task identification.
metacognition.py: Includes MetaCognitiveMonitor for self-assessment and improvement strategies, plus CompositionalUnderstanding for concept decomposition/composition.
parameter_history.py: Manages ParameterHistoryManager for versioned checkpoints, trajectories, and async saving with compression.
rlhf_feedback.py: Handles RLHFManager for PPO-based optimization and LiveFeedbackProcessor for real-time feedback ingestion (e.g., via APIs).
world_model.py: Builds UnifiedWorldModel with dynamics/reward prediction, planning algorithms (e.g., MCTS via PlanningAlgorithm), and state abstraction.
mathematical_accuracy_integration.py: Connects mathematical verification with learning system for feedback on calculation accuracy.

The system uses locks for thread safety, deques for efficient buffering, and optional libraries for advanced features (with fallbacks).
Installation and Dependencies
This module is part of the VULCAN-AMI project. To use it:

Clone the repository (or integrate into your project).
Install required dependencies:

bash pip install torch numpy scipy scikit-learn networkx aiohttp psutil
Core: torch, numpy, logging, typing, dataclasses, collections, time, pathlib, enum, threading, asyncio, aiohttp, json, pickle, queue.
Optional: scipy (stats), sklearn (clustering), networkx (graphs), psutil (monitoring).
Fallbacks for missing optionals.

Import the module:

python from vulcan.learning import UnifiedLearningSystem
Usage Example
Basic Setup
pythonimport logging
import torch
from vulcan.learning import UnifiedLearningSystem, LearningConfig, LearningMode

# Set up logging
logging.basicConfig(level=logging.INFO)

# Custom config (optional)
config = LearningConfig(
 learning_rate=0.0005, 
 replay_buffer_size=5000,
 rlhf_enabled=True
)

# Initialize the system (use context manager for proper cleanup)
with UnifiedLearningSystem(config=config, embedding_dim=512) as system:
 # Process an experience (e.g., in reinforcement mode)
 experience = {
 'embedding': torch.randn(512),
 'action': torch.tensor([1]),
 'reward': 1.0,
 'next_embedding': torch.randn(512),
 'modality': 'TEXT',
 'metadata': {'task_id': 'navigation'}
 }
 result = system.process_experience(experience)
 print("Learning Result:", result)
 
 # System will shutdown automatically when context exits
Processing Query Outcomes
pythonimport asyncio
import time

async def process_learning():
 # Learning from query outcomes (typical usage in VULCAN)
 outcome = {
 'query_id': 'q123',
 'status': 'success',
 'routing_ms': 45,
 'total_ms': 150,
 'tools': ['probabilistic', 'mathematical'],
 'query_type': 'bayesian_inference',
 'confidence': 0.95,
 'timestamp': time.time()
 }
 
 await system.process_outcome(outcome)

# Run async processing
asyncio.run(process_learning())
World Model Planning
python# Get a plan using world model (requires enable_world_model=True)
from vulcan.learning.world_model import PlanningAlgorithm

initial_state = torch.randn(512)
candidate_actions = [torch.randn(512) for _ in range(5)]

best_action, plan_info = system.plan_with_world_model(
 initial_state, 
 candidate_actions,
 algorithm=PlanningAlgorithm.MCTS,
 horizon=5
)
print("Best Action:", best_action)
print("Plan Info:", plan_info)
Human Feedback Integration
pythonfrom vulcan.learning import FeedbackData
import time

# Submit feedback
feedback = FeedbackData(
 feedback_id="fb_001",
 timestamp=time.time(),
 feedback_type="thumbs_up",
 content="Great response!",
 context={'query_id': 'q123'},
 agent_response="Generated text",
 human_preference=None,
 reward_signal=1.0,
 metadata={'source': 'web_ui'}
)

system.continual_learner.receive_feedback(feedback)
Mathematical Accuracy Integration
pythonfrom vulcan.learning.mathematical_accuracy_integration import (
 MathematicalAccuracyIntegration
)
from vulcan.reasoning.mathematical_verification import (
 BayesianProblem,
 MathematicalVerificationEngine
)

# Initialize math verification
math_engine = MathematicalVerificationEngine()
math_integration = MathematicalAccuracyIntegration(math_engine)

# Verify and learn from mathematical calculation
problem = BayesianProblem(
 prior=0.01,
 sensitivity=0.95,
 specificity=0.90,
 description="Medical test accuracy"
)

# Tool produced this answer
claimed_answer = 0.087

# Verify and apply learning feedback
verification, feedback = await math_integration.verify_and_learn(
 problem=problem,
 claimed_answer=claimed_answer,
 tool_name="probabilistic",
 learning_system=system
)

print(f"Verified: {verification.is_valid()}")
print(f"Feedback applied: {feedback.verified}")
Save and Load State
python# Save complete state
save_path = system.save_complete_state()
print(f"Saved to: {save_path}")

# Load state (in new session)
# Note: Ensure system is initialized with same configuration
system.continual_learner.load_state("path/to/continual_state.pkl")
Configuration
Learning Parameters
Tune parameters in LearningConfig:

learning_rate: Base learning rate (default: 0.001)
ewc_lambda: EWC regularization strength (default: 100.0)
ppo_epochs: PPO training epochs (default: 4)
meta_lr: Meta-learning rate (default: 0.001)
inner_lr: Inner loop learning rate for meta-learning (default: 0.01)

Buffers and Limits

replay_buffer_size: Experience replay capacity (default: 10,000)
max_checkpoints: Parameter history limit (default: 100)
feedback_buffer_size: RLHF feedback capacity (default: 5,000)
consolidation_threshold: Steps before EWC consolidation (default: 100)

Algorithm Selection
Choose via enums:

MetaLearningAlgorithm.MAML, .FOMAML, .REPTILE, .ANIL
PlanningAlgorithm.MCTS, .CEM, .MPPI, .BEAM_SEARCH, .GREEDY
PacingStrategy.THRESHOLD, .ADAPTIVE, .SELF_PACED, .FIXED, .EXPONENTIAL

Feature Toggles
Enable/disable components in UnifiedLearningSystem:

enable_world_model: World model planning (default: True)
enable_curriculum: Curriculum learning (default: True)
enable_metacognition: Self-monitoring (default: True)

Persistence
Configure storage paths:

ParameterHistoryManager(base_path="checkpoints")
CurriculumLearner auto-saves to curriculum_states/
UnifiedLearningSystem saves to unified_learning_state/

Tool Weight Management
Control automatic weight adjustment:

WEIGHT_ADJUSTMENT_SUCCESS: Reward for successful tool use (default: 0.01)
WEIGHT_ADJUSTMENT_FAILURE: Penalty for tool failure (default: -0.005)
MIN_TOOL_WEIGHT: Minimum weight bound (default: -0.1)
MAX_TOOL_WEIGHT: Maximum weight bound (default: 0.2)
WEIGHT_DECAY_FACTOR: Periodic decay factor (default: 0.95)

Best Practices
Recommended Usage Patterns

Always Use Context Manager

python with UnifiedLearningSystem(config) as system:
 # Your code here
 pass
 # Automatic cleanup on exit

Monitor Statistics Regularly

python stats = system.get_unified_stats()
 print(f"Total samples: {stats['continual']['total_experiences']}")
 print(f"Tool weights: {system.tool_weight_adjustments}")
 
 # Check for weight corruption
 persistence_stats = system.get_persistence_stats()
 print(f"Persistence status: {persistence_stats}")

Handle Async Properly

python # Process outcomes asynchronously
 async def learn():
 await system.process_outcome(outcome)
 
 # Don't block the main thread
 asyncio.create_task(learn())

Checkpoint Regularly

python # Save state every N steps
 if step % 1000 == 0:
 system.save_complete_state(f"checkpoint_{step}")

Clean Shutdown

python try:
 # Your learning loop
 pass
 finally:
 system.shutdown(timeout=30.0) # Graceful shutdown with timeout
What to Avoid
❌ Don't create multiple UnifiedLearningSystem instances simultaneously
❌ Don't load untrusted checkpoint files without validation
❌ Don't ignore warning logs about weight corruption or slow routing
❌ Don't manually modify tool weights while system is running
❌ Don't use very large buffer sizes without monitoring memory usage
Performance Considerations
Memory Usage
Typical memory footprint (embedding_dim=512):

Base system: ~500 MB
With replay buffer (10k): +2-5 GB
With world model: +1 GB
Per checkpoint: ~100 MB

Total for full system: 5-10 GB RAM recommended
Thread Count
Active background threads:

Event loop thread: 1
Checkpoint worker: 1
RLHF processor: 2
Feedback processor: 1
Parameter history: 1

Total: ~6 daemon threads (cleaned up on shutdown)
Performance Tuning
python# For low-latency applications
config = LearningConfig(
 replay_buffer_size=1000,
 consolidation_threshold=50, # Consolidate more frequently
 checkpoint_frequency=0 # Disable auto-checkpointing
)

# For memory-constrained environments
system = UnifiedLearningSystem(
 config=config,
 enable_world_model=False, # Save ~1 GB
 enable_metacognition=False # Reduce complexity
)
Advanced Features
Curriculum Learning
python# Start curriculum with automatic task clustering
tasks = [generate_task(difficulty=d) for d in range(100)]
system.start_curriculum(tasks, auto_cluster=True)

# Get curriculum statistics
curriculum_stats = system.get_curriculum_stats()
print(f"Current stage: {curriculum_stats['current_stage']}")
print(f"Stage performance: {curriculum_stats['avg_stage_performance']}")
Meta-Learning
python# Fast adaptation to new task
support_set = {
 'x': torch.randn(32, 512), # 32 examples
 'y': torch.randint(0, 10, (32,)) # Labels
}

adapted_model, adaptation_stats = system.meta_learner.adapt(
 support_set,
 num_steps=5,
 task_id='new_task_123'
)

print(f"Adaptation loss: {adaptation_stats['final_loss']}")
Metacognitive Monitoring
python# Analyze reasoning quality
reasoning_trace = [
 {'phase': 'perception', 'confidence': 0.8, 'content': '...', 'timestamp': time.time()},
 {'phase': 'planning', 'confidence': 0.9, 'content': '...', 'timestamp': time.time()},
 {'phase': 'execution', 'confidence': 0.85, 'content': '...', 'timestamp': time.time()}
]

if system.continual_learner and hasattr(system.continual_learner, 'meta_cognitive'):
 analysis = system.continual_learner.meta_cognitive.introspect_reasoning(reasoning_trace)
 print(f"Reasoning quality: {analysis['quality_score']}")
 print(f"Issues: {analysis['issues']}")
 print(f"Suggestions: {analysis['suggestions']}")
Knowledge Crystallization
python# Crystallize knowledge from successful execution
if system.continual_learner:
 crystallization_result = system.continual_learner.crystallize_from_execution(
 query="How to solve X?",
 response="Solution involves Y and Z",
 success=True,
 tools_used=['mathematical', 'symbolic'],
 strategy='multi_step_reasoning',
 metadata={'complexity': 0.8}
 )
 
 if crystallization_result:
 print(f"Crystallized {crystallization_result['principles']} principles")
Monitoring and Diagnostics
System Statistics
python# Get comprehensive statistics
unified_stats = system.get_unified_stats()

print("=== Continual Learning ===")
print(f"Tasks learned: {unified_stats['continual']['num_tasks']}")
print(f"Total experiences: {unified_stats['continual']['total_experiences']}")

print("\n=== Curriculum ===")
if 'curriculum' in unified_stats:
 print(f"Current stage: {unified_stats['curriculum']['current_stage']}")
 print(f"Stages completed: {unified_stats['curriculum']['stages_completed']}")

print("\n=== Meta-Learning ===")
if 'meta_learning' in unified_stats:
 print(f"Adaptations: {unified_stats['meta_learning']['num_adaptations']}")
 print(f"Avg task loss: {unified_stats['meta_learning']['avg_task_loss']}")

print("\n=== RLHF ===")
if 'rlhf' in unified_stats:
 print(f"Total feedback: {unified_stats['rlhf']['total_feedback']}")
 print(f"Positive/Negative: {unified_stats['rlhf']['positive_feedback']}/{unified_stats['rlhf']['negative_feedback']}")
Recovery Statistics
python# Monitor automatic recovery from slow routing
recovery_stats = system.get_recovery_stats()
print(f"Slow routing events: {recovery_stats['slow_routing_count']}")
print(f"Recoveries attempted: {recovery_stats['total_recoveries_attempted']}")
print(f"Recoveries successful: {recovery_stats['total_recoveries_successful']}")
Tool Weight Analysis
python# Analyze learned tool weights
for tool, adjustment in system.tool_weight_adjustments.items():
 base_weight = 1.0
 actual_weight = base_weight + adjustment
 print(f"{tool}: {actual_weight:.4f} (adjustment: {adjustment:+.4f})")
Mathematical Accuracy Stats
python# Get mathematical verification statistics
if hasattr(system, 'math_integration'):
 math_stats = system.math_integration.get_statistics()
 print(f"Total verifications: {math_stats['total_verifications']}")
 print(f"Overall accuracy: {math_stats['overall_accuracy']:.2%}")
 print(f"Tool accuracy: {math_stats['tool_accuracy']}")
Debugging
Enable Verbose Logging
pythonimport logging
logging.basicConfig(
 level=logging.DEBUG,
 format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
Common Issues and Solutions
Issue: Tool weights become negative or near-zero
Solution: Weight bounds and decay prevent death spiral. Check logs for automatic resets.
Issue: Slow routing persists
Solution: System automatically triggers recovery after 3 consecutive slow events. Monitor get_recovery_stats().
Issue: Memory usage grows over time
Solution: Reduce buffer sizes in config. System uses bounded deques to prevent unbounded growth.
Issue: CUDA out of memory
Solution:
python# Reduce batch size
config.batch_size = 16 # Down from 32

# Disable resource-intensive features
system = UnifiedLearningSystem(
 config=config,
 enable_world_model=False
)
Issue: Checkpoints fail to save
Solution: Check disk space and permissions. System uses atomic writes with retry logic.
Diagnostics Commands
python# Check if continual learner is available
if system.continual_learner:
 print("Continual learning: ENABLED")
 
 # Check RLHF status
 feedback_stats = system.continual_learner.get_feedback_stats()
 print(f"RLHF enabled: {feedback_stats['rlhf_enabled']}")
 
 # Check knowledge crystallizer
 crystallizer_stats = system.continual_learner.get_crystallizer_stats()
 print(f"Crystallizer available: {crystallizer_stats['available']}")

# Check world model
if system.world_model:
 world_stats = system.world_model.get_training_stats()
 print(f"World model steps: {world_stats['total_steps']}")
 print(f"Avg dynamics loss: {world_stats.get('avg_dynamics_loss', 'N/A')}")
Migration Guide
From Legacy ContinualLearner
python# Old code
from vulcan.learning.continual_learning import ContinualLearner
learner = ContinualLearner()

# New code
from vulcan.learning import UnifiedLearningSystem
system = UnifiedLearningSystem(
 enable_world_model=False,
 enable_curriculum=False,
 enable_metacognition=False
)
# Access continual learner
learner = system.continual_learner
Updating Checkpoints
Old checkpoints may not be compatible. To migrate:
python# Load old checkpoint manually
import pickle
with open('old_checkpoint.pkl', 'rb') as f:
 old_state = pickle.load(f)

# Extract relevant state
tool_weights = old_state.get('tool_weights', {})

# Initialize new system and restore weights
system = UnifiedLearningSystem()
system.tool_weight_adjustments = tool_weights

# Save in new format
system.save_complete_state()
Testing
Unit Tests
bashpytest tests/learning/test_unified_learning.py -v
pytest tests/learning/test_continual_learning.py -v
pytest tests/learning/test_curriculum_learning.py -v
pytest tests/learning/test_meta_learning.py -v
pytest tests/learning/test_world_model.py -v
Integration Tests
bashpytest tests/integration/test_learning_pipeline.py -v
Coverage
bashpytest tests/learning/ --cov=vulcan.learning --cov-report=html
API Reference
UnifiedLearningSystem
Constructor:
pythonUnifiedLearningSystem(
 config: Optional[LearningConfig] = None,
 embedding_dim: int = 384,
 enable_world_model: bool = True,
 enable_curriculum: bool = True,
 enable_metacognition: bool = True
)
Key Methods:

process_experience(experience: Dict[str, Any]) -> Dict[str, Any]: Process learning experience
async process_outcome(outcome: Dict[str, Any]) -> None: Learn from query outcome
get_tool_weight_adjustment(tool: str) -> float: Get learned tool weight
reset_tool_weights() -> None: Reset all tool weights to zero
save_complete_state(base_path: str) -> str: Save all component states
shutdown(timeout: float = 15.0) -> None: Clean shutdown with timeout
get_unified_stats() -> Dict[str, Any]: Get comprehensive statistics
get_recovery_stats() -> Dict[str, Any]: Get recovery attempt statistics
get_persistence_stats() -> Dict[str, Any]: Get persistence layer statistics

LearningConfig
Parameters:
python@dataclass
class LearningConfig:
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
 rlhf_enabled: bool = True
 feedback_buffer_size: int = 5000
 reward_model_update_freq: int = 100
 ppo_epochs: int = 4
 ppo_clip: float = 0.2
 kl_penalty: float = 0.01
 checkpoint_frequency: int = 1000
 max_checkpoints: int = 100
 audit_trail_enabled: bool = True
Contributing
When contributing to this module:

Add thread safety: All methods accessing shared state must use locks
Add error handling: Use try/except with proper logging
Add type hints: All functions must have type annotations
Add tests: Unit tests required for all new features
Update this README: Document new features and configuration options
Follow existing patterns: Match the style of similar components

Code Review Checklist

 Thread-safe operations (proper use of locks)
 Proper resource cleanup (context managers, shutdown methods)
 Type hints added
 Tests passing
 Documentation updated
 No security issues (pickle, torch.load with weights_only=True)
 Memory usage considered (bounded buffers)

Support
For issues or questions:

Bug Reports: Open GitHub issue with [Learning Module] prefix
Feature Requests: Discussion in project repository
Security Issues: Report privately to security team

License
Part of VULCAN-AMI project. See main repository for license details.

Last Updated: 2025-01-06
Version: 2.0.0
Maintainers: VULCAN-AMI Core TeamClaude is AI and can make mistakes. Please double-check responses.
