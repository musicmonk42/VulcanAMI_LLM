# ============================================================
# VULCAN-AGI Configuration Module
# Dynamic, layered, validated configuration system with runtime updates
# Enhanced with Tool Selection Configuration
# FULLY DEBUGGED VERSION - All critical issues resolved
# FIXED: Reordered definitions to prevent NameError (AgentConfig defined before usage)
# FIXED: Added missing SafetyPolicies.names_to_versions and other orchestrator dependencies
# FIXED: get_config() now returns AgentConfig instance when called without parameters
# ADDED: Intrinsic Drives Configuration for Self-Improvement
# INTEGRATED: Complete intrinsic drives configuration with profile support
# FIXED: _setup_file_watcher defined unconditionally to prevent AttributeError
# FIXED: Added initialize_config function to resolve startup NameError
# ============================================================

import os
import json
import yaml
import logging
import hashlib
import threading
import asyncio
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable, Type
from dataclasses import dataclass, field, asdict, is_dataclass
from enum import Enum
from pathlib import Path
from datetime import datetime
import re
from collections import defaultdict, deque
import copy
import importlib.util
import sys
import atexit

# Try to import validation libraries
try:
    from pydantic import BaseModel, Field, validator, ValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    
try:
    from cerberus import Validator as CerberusValidator
    CERBERUS_AVAILABLE = True
except ImportError:
    CERBERUS_AVAILABLE = False

try:
    import scipy
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import networkx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileModifiedEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False

logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION ENUMS
# ============================================================

class ConfigLayer(Enum):
    """Configuration layers in order of precedence."""
    DEFAULT = 0
    FILE = 1
    PROFILE = 2
    ENVIRONMENT = 3
    RUNTIME = 4
    ADMIN = 5

class ConfigValidationLevel(Enum):
    """Validation strictness levels."""
    NONE = 0
    BASIC = 1
    STRICT = 2
    PARANOID = 3

class ProfileType(Enum):
    """System operation profiles."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    RESEARCH = "research"
    MINIMAL = "minimal"
    HIGH_PERFORMANCE = "high_performance"
    ENERGY_SAVING = "energy_saving"
    SAFETY_CRITICAL = "safety_critical"

class ModalityType(Enum):
    """Supported modality types."""
    TEXT = "text"
    VISION = "vision"
    AUDIO = "audio"
    VIDEO = "video"
    MULTIMODAL = "multimodal"
    SYMBOLIC = "symbolic"
    CODE = "code"
    UNKNOWN = "unknown"

class SafetyLevel(Enum):
    """Safety enforcement levels."""
    MINIMAL = 0
    STANDARD = 1
    ENHANCED = 2
    MAXIMUM = 3
    PARANOID = 4

class ActionType(Enum):
    """Action types the system can take."""
    EXPLORE = "explore"
    OPTIMIZE = "optimize"
    MAINTAIN = "maintain"
    LEARN = "learn"
    ADAPT = "adapt"
    COMMUNICATE = "communicate"
    WAIT = "wait"
    SAFE_FALLBACK = "safe_fallback"
    EMERGENCY_STOP = "emergency_stop"
    CHECKPOINT = "checkpoint"
    ROLLBACK = "rollback"
    OPTIMIZED_ACTION = "optimized_action"

class GoalType(Enum):
    """Types of system goals."""
    EXPLORATION = "exploration"
    OPTIMIZATION = "optimization"
    LEARNING = "learning"
    SAFETY = "safety"
    EFFICIENCY = "efficiency"
    ROBUSTNESS = "robustness"
    ALIGNMENT = "alignment"
    RESEARCH = "research"
    MAINTENANCE = "maintenance"

class ExecutionStrategy(Enum):
    """Portfolio execution strategies for tool selection."""
    SINGLE = "single"
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    SPECULATIVE_PARALLEL = "speculative_parallel"
    CASCADE = "cascade"
    COMMITTEE_CONSENSUS = "committee_consensus"
    SEQUENTIAL_REFINEMENT = "sequential_refinement"
    HEDGE = "hedge"
    ADAPTIVE = "adaptive"

class SelectionMode(Enum):
    """Tool selection optimization modes."""
    FAST = "fast"
    ACCURATE = "accurate"
    EFFICIENT = "efficient"
    BALANCED = "balanced"
    SAFE = "safe"

# ============================================================
# CONFIGURATION SCHEMAS
# ============================================================

class ConfigSchema:
    """Configuration validation schemas."""
    
    AGENT_SCHEMA = {
        'type': 'dict',
        'schema': {
            'agent_id': {'type': 'string', 'required': True, 'regex': r'^[A-Za-z0-9_-]+$'},
            'collective_id': {'type': 'string', 'required': True},
            'version': {'type': 'string', 'required': True, 'regex': r'^\d+\.\d+\.\d+$'},
            'profile': {'type': 'string', 'allowed': [p.value for p in ProfileType]},
            'enable_learning': {'type': 'boolean'},
            'enable_adaptation': {'type': 'boolean'},
            'enable_self_modification': {'type': 'boolean'},
            'enable_multi_agent': {'type': 'boolean'},
            'enable_explainability': {'type': 'boolean'},
            'enable_adversarial_testing': {'type': 'boolean'},
            'max_parallel_tasks': {'type': 'integer', 'min': 1, 'max': 1000},
            'checkpoint_interval': {'type': 'integer', 'min': 1},
            'log_level': {'type': 'string', 'allowed': ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']}
        }
    }
    
    RESOURCE_SCHEMA = {
        'type': 'dict',
        'schema': {
            'max_memory_mb': {'type': 'integer', 'min': 100, 'max': 1000000},
            'max_cpu_percent': {'type': 'float', 'min': 0.1, 'max': 100.0},
            'max_gpu_percent': {'type': 'float', 'min': 0.0, 'max': 100.0},
            'energy_budget_nj': {'type': 'float', 'min': 0.0},
            'max_network_bandwidth_mbps': {'type': 'float', 'min': 0.0},
            'max_disk_io_mbps': {'type': 'float', 'min': 0.0},
            'max_threads': {'type': 'integer', 'min': 1, 'max': 10000},
            'max_processes': {'type': 'integer', 'min': 1, 'max': 1000},
            'gpu_memory_mb': {'type': 'integer', 'min': 0},
            'enable_distributed': {'type': 'boolean'}
        }
    }
    
    SAFETY_SCHEMA = {
        'type': 'dict',
        'schema': {
            'safety_level': {'type': 'integer', 'min': 0, 'max': 4},
            'require_human_approval': {'type': 'boolean'},
            'max_autonomy_level': {'type': 'integer', 'min': 0, 'max': 10},
            'rollback_threshold': {'type': 'float', 'min': 0.0, 'max': 1.0},
            'uncertainty_threshold': {'type': 'float', 'min': 0.0, 'max': 1.0},
            'identity_drift_threshold': {'type': 'float', 'min': 0.0, 'max': 1.0},
            'compliance_standards': {'type': 'list', 'schema': {'type': 'string'}},
            'prohibited_actions': {'type': 'list', 'schema': {'type': 'string'}},
            'audit_everything': {'type': 'boolean'},
            'encryption_required': {'type': 'boolean'},
            'safety_validation_timeout_ms': {'type': 'integer', 'min': 10, 'max': 60000}
        }
    }
    
    TOOL_SELECTION_SCHEMA = {
        'type': 'dict',
        'schema': {
            'default_selection_mode': {'type': 'string', 'allowed': [m.value for m in SelectionMode]},
            'confidence_threshold': {'type': 'float', 'min': 0.0, 'max': 1.0},
            'veto_threshold': {'type': 'float', 'min': 0.0, 'max': 1.0},
            'max_reasoning_time': {'type': 'float', 'min': 0.1, 'max': 300.0},
            'enable_caching': {'type': 'boolean'},
            'enable_warm_start': {'type': 'boolean'},
            'enable_portfolio': {'type': 'boolean'},
            'enable_voi': {'type': 'boolean'},
            'enable_calibration': {'type': 'boolean'},
            'enable_distribution_monitoring': {'type': 'boolean'}
        }
    }
    
    INTRINSIC_DRIVES_SCHEMA = {
        'type': 'dict',
        'schema': {
            'enabled': {'type': 'boolean'},
            'config_file': {'type': 'string'},
            'state_file': {'type': 'string'},
            'approval_required': {'type': 'boolean'},
            'load_on_startup': {'type': 'boolean'},
            'check_interval_seconds': {'type': 'integer', 'min': 1, 'max': 3600},
            'max_cost_usd_per_day': {'type': 'float', 'min': 0.0, 'max': 10000.0},
            'max_cost_usd_per_session': {'type': 'float', 'min': 0.0, 'max': 1000.0}
        }
    }

# ============================================================
# CONFIGURATION VALIDATORS
# ============================================================

class ConfigValidator:
    """Configuration validation system."""
    
    def __init__(self, validation_level: ConfigValidationLevel = ConfigValidationLevel.STRICT):
        self.validation_level = validation_level
        self.validation_errors = []
        self.validation_warnings = []
        
    def validate(self, config: Dict[str, Any], schema: Dict[str, Any] = None) -> Tuple[bool, List[str], List[str]]:
        """Validate configuration against schema."""
        self.validation_errors = []
        self.validation_warnings = []
        
        if self.validation_level == ConfigValidationLevel.NONE:
            return True, [], []
            
        if self.validation_level.value >= ConfigValidationLevel.BASIC.value:
            self._validate_types(config)
            
        if self.validation_level.value >= ConfigValidationLevel.STRICT.value and schema:
            self._validate_schema(config, schema)
            
        if self.validation_level.value >= ConfigValidationLevel.PARANOID.value:
            self._validate_security(config)
            
        self._validate_business_logic(config)
        
        is_valid = len(self.validation_errors) == 0
        return is_valid, self.validation_errors, self.validation_warnings
        
    def _validate_types(self, config: Dict[str, Any]):
        """Validate basic types."""
        for key, value in config.items():
            if value is None:
                self.validation_warnings.append(f"Configuration key '{key}' is None")
            elif isinstance(value, dict):
                self._validate_types(value)
                
    def _validate_schema(self, config: Dict[str, Any], schema: Dict[str, Any]):
        """Validate against schema."""
        if CERBERUS_AVAILABLE and schema:
            validator = CerberusValidator()
            if not validator.validate(config, schema):
                self.validation_errors.extend([
                    f"Schema validation error in field '{field}': {err}"
                    for field, err_list in validator.errors.items()
                    for err in err_list
                ])
                
    def _validate_security(self, config: Dict[str, Any]):
        """Validate security aspects."""
        sensitive_patterns = [
            r'password',
            r'secret',
            r'token',
            r'api_key',
            r'private_key'
        ]
        
        config_str = json.dumps(config).lower()
        for pattern in sensitive_patterns:
            if re.search(pattern, config_str):
                self.validation_warnings.append(
                    f"Potential sensitive data found matching pattern '{pattern}'"
                )
                
        if config.get('safety_policies', {}).get('safety_level', 0) < 1:
            self.validation_warnings.append(
                "Safety level is set below recommended minimum (1)"
            )
            
    def _validate_business_logic(self, config: Dict[str, Any]):
        """Validate business logic constraints."""
        resources = config.get('resource_limits', {})
        if resources.get('max_memory_mb', 0) < 1000:
            self.validation_warnings.append(
                "Memory limit below 1GB may cause performance issues"
            )
            
        safety = config.get('safety_policies', {})
        if safety.get('safety_level', 0) >= 3 and not safety.get('audit_everything', False):
            self.validation_warnings.append(
                "High safety level without full auditing enabled"
            )
            
        profile = config.get('agent_config', {}).get('profile', '')
        if profile == 'production' and not config.get('agent_config', {}).get('enable_adversarial_testing', False):
            self.validation_warnings.append(
                "Production profile without adversarial testing enabled"
            )
        
        tool_selection = config.get('tool_selection_config', {})
        if tool_selection.get('enable_portfolio', False) and tool_selection.get('max_parallel_tools', 1) < 2:
            self.validation_warnings.append(
                "Portfolio execution enabled but max_parallel_tools < 2"
            )
        
        # Validate intrinsic drives configuration
        intrinsic_drives = config.get('intrinsic_drives_config', {})
        if intrinsic_drives.get('enabled', False):
            if intrinsic_drives.get('max_cost_usd_per_session', 0) > intrinsic_drives.get('max_cost_usd_per_day', 0):
                self.validation_warnings.append(
                    "Intrinsic drives: session cost limit exceeds daily cost limit"
                )
            
            config_file = intrinsic_drives.get('config_file', '')
            if config_file and not Path(config_file).exists():
                self.validation_warnings.append(
                    f"Intrinsic drives config file not found: {config_file}"
                )

# ============================================================
# FILE WATCHER
# ============================================================

if WATCHDOG_AVAILABLE:
    class ConfigFileEventHandler(FileSystemEventHandler):
        """Handle file system events for configuration files."""
        
        def __init__(self, config_manager, file_path: Path, layer: ConfigLayer):
            self.config_manager = config_manager
            self.file_path = file_path
            self.layer = layer
            self.last_modified = 0
            
        def on_modified(self, event):
            """Handle file modification event."""
            if event.src_path != str(self.file_path):
                return
                
            current_time = time.time()
            if current_time - self.last_modified < 1.0:
                return
            
            self.last_modified = current_time
            
            logger.info(f"Configuration file modified: {self.file_path}")
            
            try:
                self.config_manager.load_from_file(self.file_path, self.layer)
                logger.info(f"Configuration reloaded from {self.file_path}")
            except Exception as e:
                logger.error(f"Failed to reload configuration: {e}")

# ============================================================
# LEGACY DATACLASSES
# ============================================================

EMBEDDING_DIM = 384
LATENT_DIM = 128
HIDDEN_DIM = 512
BATCH_SIZE = 32
LEARNING_RATE = 0.001
GAMMA = 0.99
TAU = 0.005

@dataclass
class AgentConfig:
    """Legacy agent configuration."""
    agent_id: str = ''
    collective_id: str = ''
    version: str = ''
    enable_learning: bool = True
    enable_adaptation: bool = True
    enable_self_modification: bool = False
    enable_multi_agent: bool = False
    enable_explainability: bool = True
    enable_adversarial_testing: bool = False
    enable_multimodal: bool = True
    enable_symbolic: bool = True
    enable_distributed: bool = False
    max_parallel_tasks: int = 10
    checkpoint_interval: int = 100
    log_level: str = 'INFO'
    max_working_memory: int = 20
    enable_self_improvement: bool = False
    intrinsic_drives_config_file: str = ''
    intrinsic_drives_state_file: str = ''
    
    def __post_init__(self):
        """Initialize from global config if not set."""
        if not self.agent_id:
            self.agent_id = get_config('agent_config.agent_id', 'vulcan-001')
        if not self.collective_id:
            self.collective_id = get_config('agent_config.collective_id', 'COLLECTIVE-001')
        if not self.version:
            self.version = get_config('agent_config.version', '1.0.0')
        if self.max_working_memory == 20:
            self.max_working_memory = get_config('memory_config.max_working_memory', 20)
        if not self.enable_self_improvement:
            self.enable_self_improvement = get_config('intrinsic_drives_config.enabled', False)
        if not self.intrinsic_drives_config_file:
            self.intrinsic_drives_config_file = get_config('intrinsic_drives_config.config_file', 'configs/intrinsic_drives.json')
        if not self.intrinsic_drives_state_file:
            self.intrinsic_drives_state_file = get_config('intrinsic_drives_config.state_file', 'data/agent_state.json')
    
    @property
    def safety_policies(self):
        """Get safety policies configuration."""
        return SafetyPolicies()
    
    @property
    def resource_limits(self) -> Dict[str, Any]:
        return get_config('resource_limits', {})
    
    @property
    def tool_selection_config(self) -> Dict[str, Any]:
        return get_config('tool_selection_config', {})
    
    @property
    def intrinsic_drives_config(self) -> Dict[str, Any]:
        return get_config('intrinsic_drives_config', {})
    
    @property
    def world_model(self) -> 'WorldModelConfig':
        """Get world model configuration."""
        return WorldModelConfig()

@dataclass
class ResourceLimits:
    """Legacy resource limits configuration."""
    max_memory_mb: int = 0
    max_cpu_percent: float = 0.0
    max_gpu_percent: float = 0.0
    energy_budget_nj: float = 0.0
    
    def __post_init__(self):
        """Initialize from global config if not set."""
        if self.max_memory_mb == 0:
            self.max_memory_mb = get_config('resource_limits.max_memory_mb', 8000)
        if self.max_cpu_percent == 0.0:
            self.max_cpu_percent = get_config('resource_limits.max_cpu_percent', 80.0)
        if self.max_gpu_percent == 0.0:
            self.max_gpu_percent = get_config('resource_limits.max_gpu_percent', 90.0)
        if self.energy_budget_nj == 0.0:
            self.energy_budget_nj = get_config('resource_limits.energy_budget_nj', 1e9)

@dataclass
class SafetyPolicies:
    """Legacy safety policies configuration (FIXED - added names_to_versions)."""
    safety_level: Optional[SafetyLevel] = None
    require_human_approval: bool = False
    max_autonomy_level: int = 0
    rollback_threshold: float = 0.0
    safety_thresholds: Dict[str, float] = field(default_factory=dict)
    names_to_versions: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize from global config if not set."""
        if self.safety_level is None:
            self.safety_level = SafetyLevel(get_config('safety_policies.safety_level', 1))
        if not self.require_human_approval:
            self.require_human_approval = get_config('safety_policies.require_human_approval', False)
        if self.max_autonomy_level == 0:
            self.max_autonomy_level = get_config('safety_policies.max_autonomy_level', 5)
        if self.rollback_threshold == 0.0:
            self.rollback_threshold = get_config('safety_policies.rollback_threshold', 0.3)
        if not self.safety_thresholds:
            self.safety_thresholds = get_config('safety_policies.safety_thresholds', {})
        if not self.names_to_versions:
            self.names_to_versions = get_config('safety_policies.names_to_versions', {
                'ITU_F748_53': '1.0',
                'safety_validator': '1.0',
                'governance': '1.0'
            })

@dataclass
class LearningConfig:
    """Legacy learning configuration."""
    learning_rate: float = 0.0
    batch_size: int = 0
    memory_size: int = 0
    replay_ratio: float = 0.0
    
    def __post_init__(self):
        """Initialize from global config if not set."""
        if self.learning_rate == 0.0:
            self.learning_rate = get_config('learning_config.learning_rate', 0.001)
        if self.batch_size == 0:
            self.batch_size = get_config('learning_config.batch_size', 32)
        if self.memory_size == 0:
            self.memory_size = get_config('learning_config.memory_size', 10000)
        if self.replay_ratio == 0.0:
            self.replay_ratio = get_config('learning_config.replay_ratio', 0.5)

@dataclass
class ToolSelectionConfig:
    """Tool selection configuration."""
    default_selection_mode: Optional[SelectionMode] = None
    confidence_threshold: float = 0.0
    veto_threshold: float = 0.0
    max_reasoning_time: float = 0.0
    enable_caching: bool = True
    enable_warm_start: bool = True
    enable_portfolio: bool = True
    enable_voi: bool = True
    enable_calibration: bool = True
    max_parallel_tools: int = 0
    
    def __post_init__(self):
        """Initialize from global config if not set."""
        if self.default_selection_mode is None:
            mode_str = get_config('tool_selection_config.default_selection_mode', 'balanced')
            self.default_selection_mode = SelectionMode(mode_str)
        if self.confidence_threshold == 0.0:
            self.confidence_threshold = get_config('tool_selection_config.confidence_threshold', 0.5)
        if self.veto_threshold == 0.0:
            self.veto_threshold = get_config('tool_selection_config.veto_threshold', 0.8)
        if self.max_reasoning_time == 0.0:
            self.max_reasoning_time = get_config('tool_selection_config.max_reasoning_time', 30.0)
        if self.max_parallel_tools == 0:
            self.max_parallel_tools = get_config('tool_selection_config.max_parallel_tools', 4)
    
    @property
    def utility_weights(self) -> Dict[str, float]:
        return get_utility_weights()
    
    @property
    def portfolio_strategies(self) -> Dict[str, str]:
        return get_config('tool_selection_config.portfolio_strategies', {})
    
    @property
    def calibration_config(self) -> Dict[str, Any]:
        return get_config('tool_selection_config.calibration_config', {})

@dataclass
class IntrinsicDrivesConfig:
    """Intrinsic drives configuration for self-improvement."""
    enabled: bool = False
    config_file: str = ''
    state_file: str = ''
    approval_required: bool = True
    load_on_startup: bool = True
    check_interval_seconds: int = 60
    max_cost_usd_per_day: float = 20.0
    max_cost_usd_per_session: float = 5.0
    
    def __post_init__(self):
        """Initialize from global config if not set."""
        if not self.enabled:
            self.enabled = get_config('intrinsic_drives_config.enabled', False)
        if not self.config_file:
            self.config_file = get_config('intrinsic_drives_config.config_file', 'configs/intrinsic_drives.json')
        if not self.state_file:
            self.state_file = get_config('intrinsic_drives_config.state_file', 'data/agent_state.json')
        if self.approval_required:
            self.approval_required = get_config('intrinsic_drives_config.approval_required', True)
        if self.load_on_startup:
            self.load_on_startup = get_config('intrinsic_drives_config.load_on_startup', True)
        if self.check_interval_seconds == 60:
            self.check_interval_seconds = get_config('intrinsic_drives_config.check_interval_seconds', 60)
        if self.max_cost_usd_per_day == 20.0:
            self.max_cost_usd_per_day = get_config('intrinsic_drives_config.max_cost_usd_per_day', 20.0)
        if self.max_cost_usd_per_session == 5.0:
            self.max_cost_usd_per_session = get_config('intrinsic_drives_config.max_cost_usd_per_session', 5.0)

@dataclass
class WorldModelConfig:
    """World model configuration for meta-reasoning and self-improvement."""
    enable_meta_reasoning: bool = False
    enable_self_improvement: bool = False
    meta_reasoning_config: str = ''
    self_improvement_config: str = ''
    improvement_state: str = ''
    
    def __post_init__(self):
        """Initialize from global config if not set."""
        if not self.enable_meta_reasoning:
            self.enable_meta_reasoning = get_config('world_model.enable_meta_reasoning', False)
        if not self.enable_self_improvement:
            self.enable_self_improvement = get_config('world_model.enable_self_improvement', False)
        if not self.meta_reasoning_config:
            self.meta_reasoning_config = get_config('world_model.meta_reasoning_config', 'configs/intrinsic_drives.json')
        if not self.self_improvement_config:
            self.self_improvement_config = get_config('world_model.self_improvement_config', 'configs/intrinsic_drives.json')
        if not self.improvement_state:
            self.improvement_state = get_config('world_model.self_improvement_state', 'data/agent_state.json')

logger.info("WorldModelConfig defined successfully")

@dataclass
class HierarchicalGoalSystem:
    """Legacy hierarchical goal system configuration."""
    max_depth: int = 5
    goal_types: List[GoalType] = field(default_factory=lambda: list(GoalType))
    priority_decay: float = 0.9
    
    def decompose_goal(self, goal: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Decompose high-level goal into subgoals."""
        # Basic decomposition - return simple subgoal structure
        return [
            {
                'subgoal': f"analyze_{goal}",
                'priority': 1.0,
                'estimated_cost': 0.5
            },
            {
                'subgoal': f"execute_{goal}",
                'priority': 0.9,
                'estimated_cost': 0.7
            }
        ]
    
    def prioritize_goals(self, resources: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prioritize goals based on available resources."""
        # Return basic prioritized goal list
        return [
            {
                'subgoal': 'explore',
                'priority': 1.0,
                'feasible': True
            }
        ]
    
    def generate_plan(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate plan from context."""
        return {
            'actions': [],
            'estimated_duration': 0,
            'resources_needed': {}
        }
    
    def update_progress(self, goal_id: str, progress: float) -> None:
        """Update progress for a specific goal."""
        # Basic progress tracking - extend as needed
        if not hasattr(self, '_progress'):
            self._progress = {}
        self._progress[goal_id] = max(0.0, min(1.0, progress))
    
    def get_goal_status(self) -> Dict[str, Any]:
        """Get status of all goals."""
        return {
            'active_goals': 0,
            'completed_goals': 0,
            'failed_goals': 0
        }

# ============================================================
# CONFIGURATION MANAGER
# ============================================================

class ConfigurationManager:
    """Dynamic configuration management system."""
    
    def __init__(self, 
                 validation_level: ConfigValidationLevel = ConfigValidationLevel.STRICT,
                 auto_reload: bool = True,
                 config_dir: str = "configs"):
        self.validation_level = validation_level
        self.auto_reload = auto_reload
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.layers = {layer: {} for layer in ConfigLayer}
        
        self.current_config = {}
        
        self.metadata = {
            'version': '1.0.0',
            'last_updated': datetime.now().isoformat(),
            'loaded_files': [],
            'active_profile': ProfileType.DEVELOPMENT,
            'override_count': 0,
            'validation_status': 'not_validated'
        }
        
        self.change_history = deque(maxlen=1000)
        self.change_callbacks = []
        
        self.lock = threading.RLock()
        
        self.validator = ConfigValidator(validation_level)
        
        self.file_watchers = {}
        if WATCHDOG_AVAILABLE:
            self.observer = Observer()
            self.observer.start()
        else:
            self.observer = None
            if auto_reload:
                logger.warning("Watchdog not available, auto-reload disabled")
        
        self.admin_overrides_enabled = False
        self.admin_auth_tokens = set()
        
        self._load_defaults()
        self._merge_configurations()
        
    def _setup_file_watcher(self, file_path: Path, layer: ConfigLayer):
        """Set up file watcher for configuration file."""
        # Defensive: only set up watcher if watchdog available and observer initialized
        if not WATCHDOG_AVAILABLE or not self.observer:
            return
            
        try:
            if str(file_path) in self.file_watchers:
                return
                
            event_handler = ConfigFileEventHandler(self, file_path, layer)
            watch = self.observer.schedule(
                event_handler,
                str(file_path.parent),
                recursive=False
            )
            self.file_watchers[str(file_path)] = watch
            logger.info(f"File watcher set up for {file_path}")
        except Exception as e:
            logger.error(f"Failed to setup file watcher for {file_path}: {e}")
            
    def _load_defaults(self):
        """Load default configuration."""
        self.layers[ConfigLayer.DEFAULT] = {
            'agent_config': {
                'agent_id': 'vulcan-agi-001',
                'collective_id': 'VULCAN-COLLECTIVE-001',
                'version': '1.0.0',
                'profile': ProfileType.DEVELOPMENT.value,
                'enable_learning': True,
                'enable_adaptation': True,
                'enable_self_modification': False,
                'enable_multi_agent': False,
                'enable_explainability': True,
                'enable_adversarial_testing': False,
                'enable_multimodal': True,
                'enable_symbolic': True,
                'enable_distributed': False,
                'max_parallel_tasks': 10,
                'max_agents': 100,
                'min_agents': 10,
                'task_queue_type': 'custom',
                'checkpoint_interval': 100,
                'log_level': 'INFO',
                'slo_p95_latency_ms': 1000,
                'slo_max_error_rate': 0.1
            },
            'resource_limits': {
                'max_memory_mb': 8000,
                'max_cpu_percent': 80.0,
                'max_gpu_percent': 90.0,
                'energy_budget_nj': 1e9,
                'max_network_bandwidth_mbps': 100.0,
                'max_disk_io_mbps': 500.0,
                'max_threads': 100,
                'max_processes': 10,
                'gpu_memory_mb': 8000,
                'enable_distributed': False
            },
            'safety_policies': {
                'safety_level': SafetyLevel.STANDARD.value,
                'require_human_approval': False,
                'max_autonomy_level': 5,
                'rollback_threshold': 0.3,
                'uncertainty_threshold': 0.8,
                'identity_drift_threshold': 0.5,
                'compliance_standards': ['ITU_F748_53'],
                'prohibited_actions': [],
                'audit_everything': False,
                'encryption_required': False,
                'safety_validation_timeout_ms': 1000,
                'safety_thresholds': {
                    'uncertainty_max': 0.9,
                    'identity_drift_max': 0.5,
                    'energy_usage_max': 0.9
                },
                'names_to_versions': {
                    'ITU_F748_53': '1.0',
                    'safety_validator': '1.0',
                    'governance': '1.0'
                }
            },
            'tool_selection_config': {
                'default_selection_mode': SelectionMode.BALANCED.value,
                'confidence_threshold': 0.5,
                'veto_threshold': 0.8,
                'max_reasoning_time': 30.0,
                'enable_caching': True,
                'enable_warm_start': True,
                'enable_portfolio': True,
                'enable_voi': True,
                'enable_calibration': True,
                'enable_distribution_monitoring': True,
                'max_parallel_tools': 4,
                'cache_ttl_seconds': 300,
                'warm_pool_size': 10,
                'portfolio_strategies': {
                    'default': ExecutionStrategy.ADAPTIVE.value,
                    'fast': ExecutionStrategy.SPECULATIVE_PARALLEL.value,
                    'accurate': ExecutionStrategy.COMMITTEE_CONSENSUS.value,
                    'efficient': ExecutionStrategy.CASCADE.value
                },
                'utility_weights': {
                    'quality': 1.0,
                    'time_penalty': 1.0,
                    'energy_penalty': 0.5,
                    'risk_penalty': 0.8,
                    'exploration_bonus': 0.1
                },
                'calibration_config': {
                    'method': 'isotonic',
                    'n_bins': 10,
                    'update_frequency': 100,
                    'min_samples': 50
                },
                'voi_config': {
                    'voi_threshold': 0.1,
                    'max_iterations': 3,
                    'myopic': True,
                    'information_sources': [
                        'tier2_features',
                        'tier3_features',
                        'probe_tool',
                        'memory_lookup'
                    ]
                },
                'distribution_monitor_config': {
                    'window_size': 1000,
                    'detection_threshold': 0.05,
                    'check_interval': 100
                },
                'cost_model_config': {
                    'max_observations': 1000,
                    'cold_start_time_ms': 100,
                    'cold_start_energy_mj': 50
                },
                'safety_governor_config': {
                    'veto_threshold': 0.8,
                    'require_consensus': False,
                    'rate_limit_window': 60,
                    'max_requests_per_tool': 100
                }
            },
            'learning_config': {
                'learning_rate': 0.001,
                'batch_size': 32,
                'memory_size': 10000,
                'replay_ratio': 0.5,
                'update_frequency': 10,
                'gradient_clip': 1.0,
                'enable_continual_learning': True,
                'enable_meta_learning': False,
                'knowledge_distillation_temperature': 3.0
            },
            'reasoning_config': {
                'max_reasoning_depth': 10,
                'min_confidence_threshold': 0.7,
                'enable_symbolic_reasoning': True,
                'enable_causal_reasoning': True,
                'enable_counterfactual_reasoning': True,
                'enable_analogical_reasoning': True,
                'reasoning_timeout_ms': 5000,
                'max_hypothesis_count': 100,
                'proof_search_depth': 20
            },
            'planning_config': {
                'planning_horizon': 20,
                'max_plan_length': 50,
                'replan_threshold': 0.3,
                'enable_hierarchical_planning': True,
                'enable_monte_carlo_planning': True,
                'mcts_simulations': 1000,
                'enable_contingency_planning': True,
                'plan_diversity_bonus': 0.1
            },
            'processing_config': {
                'default_quality': 'balanced',
                'cache_size': 10000,
                'cache_ttl_seconds': 300,
                'max_batch_size': 64,
                'processing_timeout_ms': 1000,
                'enable_gpu_acceleration': True,
                'enable_model_quantization': False,
                'enable_mixed_precision': True,
                'prefetch_size': 10
            },
            'memory_config': {
                'short_term_capacity': 100,
                'long_term_capacity': 100000,
                'episodic_capacity': 10000,
                'working_memory_size': 20,
                'consolidation_interval': 1000,
                'max_working_memory': 20,
                'forgetting_rate': 0.01,
                'importance_threshold': 0.3,
                'enable_semantic_compression': True
            },
            'communication_config': {
                'api_port': 8080,
                'websocket_port': 8081,
                'grpc_port': 50051,
                'enable_rest_api': True,
                'enable_graphql': False,
                'enable_websocket': True,
                'enable_grpc': False,
                'max_message_size_mb': 10,
                'request_timeout_seconds': 30,
                'enable_compression': True
            },
            'monitoring_config': {
                'metrics_port': 9090,
                'enable_prometheus': True,
                'enable_jaeger': False,
                'enable_logging': True,
                'log_retention_days': 30,
                'metric_retention_days': 90,
                'alert_thresholds': {
                    'error_rate': 0.05,
                    'latency_p99_ms': 1000,
                    'memory_usage_percent': 90
                },
                'health_check_interval_seconds': 30
            },
            'security_config': {
                'enable_encryption': True,
                'enable_authentication': True,
                'enable_authorization': True,
                'jwt_secret': os.getenv('JWT_SECRET', 'CHANGE_THIS_SECRET'),
                'token_expiry_seconds': 3600,
                'max_failed_auth_attempts': 5,
                'enable_rate_limiting': True,
                'rate_limit_requests_per_minute': 100,
                'allowed_origins': ['http://localhost:3000'],
                'enable_audit_logging': True
            },
            'deployment_config': {
                'environment': 'development',
                'region': 'us-east-1',
                'availability_zones': ['us-east-1a', 'us-east-1b'],
                'enable_auto_scaling': False,
                'min_instances': 1,
                'max_instances': 10,
                'target_cpu_utilization': 70,
                'enable_blue_green': False,
                'enable_canary': False,
                'canary_percentage': 10
            },
            'intrinsic_drives_config': {
                'enabled': False,  # Set to True to enable self-improvement
                'config_file': 'configs/intrinsic_drives.json',
                'state_file': 'data/agent_state.json',
                'approval_required': True,
                'load_on_startup': True,
                'check_interval_seconds': 60,
                'max_cost_usd_per_day': 20.0,
                'max_cost_usd_per_session': 5.0
            }
        }
        
    def load_from_file(self, file_path: Union[str, Path], 
                      layer: ConfigLayer = ConfigLayer.FILE) -> bool:
        """Load configuration from file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"Configuration file not found: {file_path}")
            return False
            
        try:
            with open(file_path, 'r') as f:
                if file_path.suffix == '.json':
                    config = json.load(f)
                elif file_path.suffix in ['.yaml', '.yml']:
                    config = yaml.safe_load(f)
                else:
                    logger.error(f"Unsupported file format: {file_path.suffix}")
                    return False
                    
            with self.lock:
                self.layers[layer].update(config)
                if str(file_path) not in self.metadata['loaded_files']:
                    self.metadata['loaded_files'].append(str(file_path))
                
            self._merge_configurations()
            
            if self.auto_reload:
                self._setup_file_watcher(file_path, layer)
                
            logger.info(f"Loaded configuration from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {file_path}: {e}")
            return False
            
    def load_from_environment(self, prefix: str = 'VULCAN_') -> int:
        """Load configuration from environment variables."""
        loaded_count = 0
        env_config = {}
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower().replace('_', '.')
                
                try:
                    parsed_value = json.loads(value)
                except json.JSONDecodeError:
                    parsed_value = value
                    
                self._set_nested_config(env_config, config_key, parsed_value)
                loaded_count += 1
        
        # FIX: Add explicit env setup as requested
        self._set_nested_config(env_config, 'core.env_setup', True)
        self._set_nested_config(env_config, 'distributed.enabled', os.getenv('VULCAN_DISTRIBUTED_ENABLED', False))
                
        with self.lock:
            self.layers[ConfigLayer.ENVIRONMENT] = env_config
            
        self._merge_configurations()
        logger.info(f"Loaded {loaded_count} configuration values from environment")
        return loaded_count
        
    def load_profile(self, profile: ProfileType) -> bool:
        """Load profile-specific configuration."""
        profile_file = self.config_dir / f"profile_{profile.value}.json"
        
        if not profile_file.exists():
            self._create_default_profile(profile)
            
        success = self.load_from_file(profile_file, ConfigLayer.PROFILE)
        
        if success:
            with self.lock:
                self.metadata['active_profile'] = profile
            
        return success
        
    def _create_default_profile(self, profile: ProfileType):
        """Create default profile configuration."""
        profile_configs = {
            ProfileType.DEVELOPMENT: {
                'agent_config': {
                    'log_level': 'DEBUG',
                    'enable_adversarial_testing': True
                },
                'safety_policies': {
                    'safety_level': SafetyLevel.MINIMAL.value
                },
                'monitoring_config': {
                    'enable_logging': True
                },
                'tool_selection_config': {
                    'default_selection_mode': SelectionMode.BALANCED.value,
                    'enable_caching': True,
                    'enable_warm_start': False
                },
                'intrinsic_drives_config': {
                    'enabled': True,
                    'approval_required': False,
                    'max_cost_usd_per_session': 2.0
                }
            },
            ProfileType.TESTING: {
                'agent_config': {
                    'log_level': 'DEBUG',
                    'enable_adversarial_testing': True
                },
                'resource_limits': {
                    'max_memory_mb': 4000,
                    'max_cpu_percent': 50.0
                },
                'tool_selection_config': {
                    'default_selection_mode': SelectionMode.FAST.value,
                    'max_reasoning_time': 10.0
                },
                'intrinsic_drives_config': {
                    'enabled': False,
                    'approval_required': True
                }
            },
            ProfileType.STAGING: {
                'agent_config': {
                    'log_level': 'INFO',
                    'enable_adversarial_testing': True
                },
                'safety_policies': {
                    'safety_level': SafetyLevel.ENHANCED.value,
                    'audit_everything': True
                },
                'tool_selection_config': {
                    'default_selection_mode': SelectionMode.SAFE.value,
                    'confidence_threshold': 0.7,
                    'enable_portfolio': True
                },
                'intrinsic_drives_config': {
                    'enabled': True,
                    'approval_required': True,
                    'max_cost_usd_per_session': 3.0,
                    'max_cost_usd_per_day': 10.0
                }
            },
            ProfileType.PRODUCTION: {
                'agent_config': {
                    'log_level': 'WARNING',
                    'enable_adversarial_testing': True,
                    'enable_self_modification': False
                },
                'safety_policies': {
                    'safety_level': SafetyLevel.MAXIMUM.value,
                    'audit_everything': True,
                    'require_human_approval': True
                },
                'security_config': {
                    'enable_encryption': True,
                    'enable_authentication': True
                },
                'tool_selection_config': {
                    'default_selection_mode': SelectionMode.ACCURATE.value,
                    'confidence_threshold': 0.8,
                    'enable_portfolio': True,
                    'enable_calibration': True,
                    'enable_distribution_monitoring': True
                },
                'intrinsic_drives_config': {
                    'enabled': True,
                    'approval_required': True,
                    'max_cost_usd_per_session': 5.0,
                    'max_cost_usd_per_day': 20.0
                }
            },
            ProfileType.RESEARCH: {
                'agent_config': {
                    'enable_learning': True,
                    'enable_adaptation': True,
                    'enable_self_modification': True
                },
                'learning_config': {
                    'enable_meta_learning': True
                },
                'tool_selection_config': {
                    'default_selection_mode': SelectionMode.BALANCED.value,
                    'enable_voi': True,
                    'utility_weights': {
                        'exploration_bonus': 0.3
                    }
                },
                'intrinsic_drives_config': {
                    'enabled': True,
                    'approval_required': False,
                    'max_cost_usd_per_session': 10.0,
                    'max_cost_usd_per_day': 50.0
                }
            },
            ProfileType.MINIMAL: {
                'resource_limits': {
                    'max_memory_mb': 1000,
                    'max_cpu_percent': 25.0,
                    'max_threads': 10
                },
                'tool_selection_config': {
                    'default_selection_mode': SelectionMode.EFFICIENT.value,
                    'max_parallel_tools': 1,
                    'enable_warm_start': False,
                    'enable_portfolio': False
                },
                'intrinsic_drives_config': {
                    'enabled': False
                }
            },
            ProfileType.HIGH_PERFORMANCE: {
                'resource_limits': {
                    'max_memory_mb': 32000,
                    'max_cpu_percent': 100.0,
                    'max_gpu_percent': 100.0,
                    'enable_distributed': True
                },
                'processing_config': {
                    'max_batch_size': 256,
                    'enable_gpu_acceleration': True
                },
                'tool_selection_config': {
                    'default_selection_mode': SelectionMode.FAST.value,
                    'max_parallel_tools': 10,
                    'enable_warm_start': True,
                    'warm_pool_size': 20
                },
                'intrinsic_drives_config': {
                    'enabled': True,
                    'approval_required': True,
                    'check_interval_seconds': 30
                }
            },
            ProfileType.ENERGY_SAVING: {
                'resource_limits': {
                    'energy_budget_nj': 1e8,
                    'max_cpu_percent': 50.0
                },
                'processing_config': {
                    'default_quality': 'fast',
                    'enable_model_quantization': True
                },
                'tool_selection_config': {
                    'default_selection_mode': SelectionMode.EFFICIENT.value,
                    'utility_weights': {
                        'energy_penalty': 2.0
                    },
                    'enable_warm_start': False
                },
                'intrinsic_drives_config': {
                    'enabled': False
                }
            },
            ProfileType.SAFETY_CRITICAL: {
                'safety_policies': {
                    'safety_level': SafetyLevel.PARANOID.value,
                    'require_human_approval': True,
                    'rollback_threshold': 0.1,
                    'audit_everything': True,
                    'encryption_required': True
                },
                'tool_selection_config': {
                    'default_selection_mode': SelectionMode.SAFE.value,
                    'confidence_threshold': 0.9,
                    'veto_threshold': 0.5,
                    'enable_portfolio': True,
                    'portfolio_strategies': {
                        'default': ExecutionStrategy.COMMITTEE_CONSENSUS.value
                    }
                },
                'intrinsic_drives_config': {
                    'enabled': True,
                    'approval_required': True,
                    'max_cost_usd_per_session': 1.0,
                    'max_cost_usd_per_day': 5.0
                }
            }
        }
        
        profile_config = profile_configs.get(profile, {})
        profile_file = self.config_dir / f"profile_{profile.value}.json"
        
        try:
            with open(profile_file, 'w') as f:
                json.dump(profile_config, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to create default profile file: {e}")
            
    def set_runtime_override(self, key: str, value: Any, 
                            admin_token: str = None) -> bool:
        """Set runtime configuration override."""
        if self.admin_overrides_enabled and admin_token not in self.admin_auth_tokens:
            logger.warning(f"Unauthorized runtime override attempt for key: {key}")
            return False
            
        with self.lock:
            runtime_config = self.layers[ConfigLayer.RUNTIME]
            self._set_nested_config(runtime_config, key, value)
            
            self._track_change(key, value, ConfigLayer.RUNTIME)
            
            self.metadata['override_count'] += 1
            self.metadata['last_updated'] = datetime.now().isoformat()
            
        self._merge_configurations()
        
        self._notify_change_callbacks(key, value)
        
        logger.info(f"Runtime override set: {key} = {value}")
        return True
        
    def remove_runtime_override(self, key: str, admin_token: str = None) -> bool:
        """Remove runtime configuration override."""
        if self.admin_overrides_enabled and admin_token not in self.admin_auth_tokens:
            logger.warning(f"Unauthorized override removal attempt for key: {key}")
            return False
            
        with self.lock:
            runtime_config = self.layers[ConfigLayer.RUNTIME]
            self._remove_nested_config(runtime_config, key)
            
        self._merge_configurations()
        logger.info(f"Runtime override removed: {key}")
        return True
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        with self.lock:
            return self._get_nested_config(self.current_config, key, default)
            
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration."""
        with self.lock:
            return copy.deepcopy(self.current_config)
            
    def validate(self) -> Tuple[bool, List[str], List[str]]:
        """Validate current configuration."""
        with self.lock:
            all_errors = []
            all_warnings = []
            
            is_valid, errors, warnings = self.validator.validate(
                self.current_config.get('agent_config', {}),
                ConfigSchema.AGENT_SCHEMA
            )
            all_errors.extend(errors)
            all_warnings.extend(warnings)
            
            is_valid, errors, warnings = self.validator.validate(
                self.current_config.get('resource_limits', {}),
                ConfigSchema.RESOURCE_SCHEMA
            )
            all_errors.extend(errors)
            all_warnings.extend(warnings)
            
            is_valid, errors, warnings = self.validator.validate(
                self.current_config.get('safety_policies', {}),
                ConfigSchema.SAFETY_SCHEMA
            )
            all_errors.extend(errors)
            all_warnings.extend(warnings)
            
            is_valid, errors, warnings = self.validator.validate(
                self.current_config.get('tool_selection_config', {}),
                ConfigSchema.TOOL_SELECTION_SCHEMA
            )
            all_errors.extend(errors)
            all_warnings.extend(warnings)
            
            is_valid, errors, warnings = self.validator.validate(
                self.current_config.get('intrinsic_drives_config', {}),
                ConfigSchema.INTRINSIC_DRIVES_SCHEMA
            )
            all_errors.extend(errors)
            all_warnings.extend(warnings)
            
            self.metadata['validation_status'] = 'valid' if len(all_errors) == 0 else 'invalid'
            
            return len(all_errors) == 0, all_errors, all_warnings
            
    def export(self, file_path: Union[str, Path], 
              include_metadata: bool = True) -> bool:
        """Export current configuration to file."""
        file_path = Path(file_path)
        
        try:
            with self.lock:
                export_data = {
                    'configuration': self.current_config
                }
                
                if include_metadata:
                    export_data['metadata'] = self.metadata
                    
            with open(file_path, 'w') as f:
                if file_path.suffix == '.json':
                    json.dump(export_data, f, indent=2, default=str)
                elif file_path.suffix in ['.yaml', '.yml']:
                    yaml.dump(export_data, f, default_flow_style=False)
                else:
                    logger.error(f"Unsupported export format: {file_path.suffix}")
                    return False
                    
            logger.info(f"Configuration exported to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export configuration: {e}")
            return False
            
    def get_layer_config(self, layer: ConfigLayer) -> Dict[str, Any]:
        """Get configuration for specific layer."""
        with self.lock:
            return copy.deepcopy(self.layers[layer])
            
    def get_metadata(self) -> Dict[str, Any]:
        """Get configuration metadata."""
        with self.lock:
            return copy.deepcopy(self.metadata)
            
    def get_change_history(self) -> List[Dict[str, Any]]:
        """Get configuration change history."""
        with self.lock:
            return list(self.change_history)
            
    def register_change_callback(self, callback: Callable[[str, Any], None]):
        """Register callback for configuration changes."""
        self.change_callbacks.append(callback)
    
    # FIX: Add new validate_env method as requested
    def validate_env(self):
        if not os.getenv('OPENAI_API_KEY'):
            logger.warning("OPENAI_API_KEY missing - set it for real AI")
        logger.info("Env validated")
        return True
        
    def enable_admin_overrides(self, auth_token: str):
        """Enable admin overrides with authentication."""
        self.admin_overrides_enabled = True
        self.admin_auth_tokens.add(auth_token)
        logger.info("Admin overrides enabled")
        
    def disable_admin_overrides(self):
        """Disable admin overrides."""
        self.admin_overrides_enabled = False
        self.admin_auth_tokens.clear()
        logger.info("Admin overrides disabled")
        
    def _merge_configurations(self):
        """Merge configuration layers in order of precedence."""
        with self.lock:
            merged = {}
            
            for layer in ConfigLayer:
                if self.layers[layer]:
                    merged = self._deep_merge(merged, self.layers[layer])
                    
            self.current_config = merged
            
    def _deep_merge(self, base: Dict[str, Any], 
                   update: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries with None handling."""
        if base is None:
            base = {}
        if update is None:
            return copy.deepcopy(base)
            
        result = copy.deepcopy(base)
        
        for key, value in update.items():
            if (key in result and 
                isinstance(result[key], dict) and 
                isinstance(value, dict) and 
                value is not None):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value) if value is not None else None
                
        return result
        
    def _set_nested_config(self, config: Dict[str, Any], 
                          key: str, value: Any):
        """Set nested configuration value."""
        keys = key.split('.')
        current = config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            elif not isinstance(current[k], dict):
                current[k] = {}
            current = current[k]
            
        current[keys[-1]] = value
        
    def _get_nested_config(self, config: Dict[str, Any], 
                          key: str, default: Any = None) -> Any:
        """Get nested configuration value."""
        keys = key.split('.')
        current = config
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
                
        return current
        
    def _remove_nested_config(self, config: Dict[str, Any], key: str):
        """Remove nested configuration value."""
        keys = key.split('.')
        current = config
        
        for k in keys[:-1]:
            if k not in current:
                return
            current = current[k]
            
        if keys[-1] in current:
            del current[keys[-1]]
            
    def _track_change(self, key: str, value: Any, layer: ConfigLayer):
        """Track configuration change."""
        change = {
            'timestamp': datetime.now().isoformat(),
            'key': key,
            'value': value,
            'layer': layer.name,
            'user': os.getenv('USER', 'unknown')
        }
        
        self.change_history.append(change)
        
    def _notify_change_callbacks(self, key: str, value: Any):
        """Notify registered callbacks of configuration change."""
        for callback in self.change_callbacks:
            try:
                callback(key, value)
            except Exception as e:
                logger.error(f"Error in change callback: {e}")
                
    def cleanup(self):
        """Cleanup resources."""
        if self.observer:
            try:
                self.observer.stop()
                self.observer.join(timeout=5)
            except Exception as e:
                logger.error(f"Error stopping file observer: {e}")

# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def _dict_to_agent_config(raw: dict) -> AgentConfig:
    """
    Given a config dict (result from get_all()), produce an AgentConfig with correct fields from both root and agent_config blocks.
    """
    agent_cfg = raw.get("agent_config", {}).copy()  # Make a copy to avoid modifying original
    # Add root-level and nested overrides
    enable_self_improvement = raw.get("enable_self_improvement", False)
    intrinsic_drives = raw.get("intrinsic_drives_config", {})
    
    # Define valid AgentConfig fields (excluding fields we'll set explicitly)
    valid_fields = {
        'agent_id', 'collective_id', 'version', 'enable_learning', 'enable_adaptation',
        'enable_self_modification', 'enable_multi_agent', 'enable_explainability',
        'enable_adversarial_testing', 'enable_multimodal', 'enable_symbolic',
        'enable_distributed', 'max_parallel_tasks', 'checkpoint_interval', 'log_level',
        'max_working_memory'
        # Note: enable_self_improvement, intrinsic_drives_config_file, and intrinsic_drives_state_file
        # are handled separately below
    }
    
    # Filter agent_cfg to only include valid fields
    filtered_cfg = {k: v for k, v in agent_cfg.items() if k in valid_fields}
    
    # Provide more fields as needed for new features
    return AgentConfig(
        **filtered_cfg,
        enable_self_improvement=enable_self_improvement or raw.get("agent_config", {}).get("enable_self_improvement", False),
        intrinsic_drives_config_file=intrinsic_drives.get("config_file", "configs/intrinsic_drives.json"),
        intrinsic_drives_state_file=intrinsic_drives.get("state_file", "data/agent_state.json")
    )

def get_config(key: str = None, default: Any = None) -> Any:
    """Get configuration value. Returns AgentConfig when called with string profile name or no args."""
    config_manager = _get_config_manager()
    
    # Handle string profile names (e.g., "development", "production", "testing")
    if isinstance(key, str) and key in [p.value for p in ProfileType]:
        try:
            profile_type = ProfileType(key)
            config_manager.load_profile(profile_type)
            raw = config_manager.get_all()
            return _dict_to_agent_config(raw)
        except Exception as e:
            logger.error(f"Failed to construct AgentConfig from loaded profile: {e}")
            return AgentConfig()
    
    # If no key provided, return AgentConfig instance
    if key is None:
        raw = config_manager.get_all()
        return _dict_to_agent_config(raw)
    
    # Otherwise get the specific config value
    return config_manager.get(key, default)

def set_config(key: str, value: Any) -> bool:
    """Set configuration value."""
    config_manager = _get_config_manager()
    return config_manager.set_runtime_override(key, value)

def load_profile(profile: ProfileType) -> bool:
    """Load configuration profile."""
    config_manager = _get_config_manager()
    return config_manager.load_profile(profile)

def validate_config() -> Tuple[bool, List[str], List[str]]:
    """Validate current configuration."""
    config_manager = _get_config_manager()
    return config_manager.validate()

def export_config(file_path: Union[str, Path]) -> bool:
    """Export configuration to file."""
    config_manager = _get_config_manager()
    return config_manager.export(file_path)

def get_tool_selection_config() -> Dict[str, Any]:
    """Get tool selection configuration."""
    return get_config('tool_selection_config', {})

def get_utility_weights() -> Dict[str, float]:
    """Get utility weights for tool selection."""
    return get_config('tool_selection_config.utility_weights', {
        'quality': 1.0,
        'time_penalty': 1.0,
        'energy_penalty': 0.5,
        'risk_penalty': 0.8
    })

def get_portfolio_strategy(mode: str = 'default') -> str:
    """Get portfolio execution strategy for given mode."""
    strategies = get_config('tool_selection_config.portfolio_strategies', {})
    return strategies.get(mode, ExecutionStrategy.ADAPTIVE.value)

def get_intrinsic_drives_config() -> Dict[str, Any]:
    """Get intrinsic drives configuration."""
    return get_config('intrinsic_drives_config', {})

def load_intrinsic_drives_from_file(file_path: str = None) -> Dict[str, Any]:
    """Load intrinsic drives configuration from file."""
    if file_path is None:
        file_path = get_config('intrinsic_drives_config.config_file', 'configs/intrinsic_drives.json')
    
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load intrinsic drives config from {file_path}: {e}")
        return {}

def enable_self_improvement(enabled: bool = True) -> bool:
    """Enable or disable self-improvement system."""
    return set_config('intrinsic_drives_config.enabled', enabled)

def is_self_improvement_enabled() -> bool:
    """Check if self-improvement is enabled."""
    return get_config('intrinsic_drives_config.enabled', False)

# ============================================================
# GLOBAL CONFIGURATION INSTANCE
# ============================================================

_config_manager = None
_config_lock = threading.Lock()

def _get_config_manager():
    """Get or create global configuration manager (thread-safe)."""
    global _config_manager
    if _config_manager is None:
        with _config_lock:
            if _config_manager is None:
                _config_manager = ConfigurationManager()
    return _config_manager

# ============================================================
# DEPENDENCY VALIDATION
# ============================================================

def _validate_dependencies():
    """Validate optional dependencies and update config with degraded features if any are missing."""
    config_manager = _get_config_manager()
    missing = []
    degraded_features = config_manager.get('degraded_features', [])
    
    if not SCIPY_AVAILABLE:
        missing.append('scipy')
        degraded_features.extend(['dynamics_fitting', 'causal_tests'])
        
    if not NETWORKX_AVAILABLE:
        missing.append('networkx')
        degraded_features.extend(['graph_cycles'])
    
    if not WATCHDOG_AVAILABLE:
        missing.append('watchdog')
        degraded_features.extend(['config_auto_reload'])
        
    if missing:
        unique_degraded_features = list(set(degraded_features))
        config_manager.set_runtime_override('degraded_features', unique_degraded_features)
        
        logger.warning(f"Missing optional dependencies: {list(set(missing))}. "
                       f"Degraded features: {unique_degraded_features}")

# ============================================================
# CONFIGURATION API
# ============================================================

class ConfigurationAPI:
    """REST API for configuration management."""
    
    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager
        
    async def get_config(self, key: str = None) -> Dict[str, Any]:
        """Get configuration via API."""
        if key:
            value = self.config_manager.get(key)
            return {'key': key, 'value': value}
        else:
            return self.config_manager.get_all()
            
    async def set_config(self, key: str, value: Any, 
                        admin_token: str = None) -> Dict[str, Any]:
        """Set configuration via API."""
        success = self.config_manager.set_runtime_override(key, value, admin_token)
        return {
            'success': success,
            'key': key,
            'value': value if success else None
        }
        
    async def delete_config(self, key: str, 
                           admin_token: str = None) -> Dict[str, Any]:
        """Delete configuration override via API."""
        success = self.config_manager.remove_runtime_override(key, admin_token)
        return {'success': success, 'key': key}
        
    async def validate_config(self) -> Dict[str, Any]:
        """Validate configuration via API."""
        is_valid, errors, warnings = self.config_manager.validate()
        return {
            'valid': is_valid,
            'errors': errors,
            'warnings': warnings
        }
        
    async def get_metadata(self) -> Dict[str, Any]:
        """Get configuration metadata via API."""
        return self.config_manager.get_metadata()
        
    async def get_change_history(self) -> List[Dict[str, Any]]:
        """Get configuration change history via API."""
        return self.config_manager.get_change_history()
        
    async def export_config(self, format: str = 'json') -> Dict[str, Any]:
        """Export configuration via API."""
        file_path = Path(f'/tmp/config_export.{format}')
        success = self.config_manager.export(file_path)
        
        if success:
            with open(file_path, 'r') as f:
                content = f.read()
            return {'success': True, 'content': content}
        else:
            return {'success': False, 'error': 'Export failed'}

# ============================================================
# MODULE INITIALIZATION
# ============================================================

def initialize_config(profile: ProfileType = ProfileType.DEVELOPMENT,
                      config_file: str = None,
                      load_env: bool = True,
                      validate: bool = True) -> bool:
    """
    Initializes the configuration manager, loads the specified profile, config file,
    and optionally environment variables, and validates the configuration.
    """
    config_manager = _get_config_manager()
    success = True
    
    # Load profile
    if not config_manager.load_profile(profile):
        logger.warning(f"Failed to load profile: {profile}")
        success = False
        
    # Load config file if specified
    if config_file:
        if not config_manager.load_from_file(config_file):
            logger.warning(f"Failed to load config file: {config_file}")
            success = False
            
    # Load environment
    if load_env:
        count = config_manager.load_from_environment()
        logger.info(f"Loaded {count} configuration values from environment")
        env_si = os.getenv("VULCAN_ENABLE_SELF_IMPROVEMENT")
        if env_si and str(env_si).lower() in ("1", "true", "yes", "on"):
            config_manager.set_runtime_override("intrinsic_drives_config.enabled", True)
            
    # Validation
    if validate:
        is_valid, errors, warnings = config_manager.validate()
        if warnings:
            for warning in warnings:
                logger.warning(f"Configuration warning: {warning}")
        if not is_valid:
            for error in errors:
                logger.error(f"Configuration error: {error}")
            if config_manager.validator.validation_level.value >= ConfigValidationLevel.STRICT.value:
                raise ValueError(f"Configuration validation failed with {len(errors)} errors")
                
    return success

def _lazy_init():
    """Lazy initialization of configuration."""
    try:
        initialize_config(validate=False)
        _validate_dependencies()
        # FIX: Call the new env validator as requested
        config_manager = _get_config_manager()
        config_manager.validate_env()
    except Exception as e:
        logger.error(f"Failed to initialize configuration: {e}")

def _ensure_initialized():
    """Ensure configuration is initialized."""
    global _config_manager
    if _config_manager is None:
        _lazy_init()

def _cleanup_on_exit():
    """Cleanup on module exit."""
    global _config_manager
    if _config_manager:
        try:
            _config_manager.cleanup()
        except Exception as e:
            logger.error(f"Error during config cleanup: {e}")

atexit.register(_cleanup_on_exit)

DEPENDENCIES_VALIDATED = False

def validate_all_dependencies():
    """Validate all dependencies (call this explicitly after import)."""
    global DEPENDENCIES_VALIDATED
    if not DEPENDENCIES_VALIDATED:
        _ensure_initialized()
        _validate_dependencies()
        DEPENDENCIES_VALIDATED = True

_ensure_initialized()
