# ============================================================
# VULCAN-AGI Planning Module - ENHANCED VERSION
# Hierarchical goals, resource-aware compute, distributed coordination
# ENHANCED: Real resource monitoring, network management, power control, survival protocols
# ============================================================

import numpy as np
import time
from typing import Any, Dict, List, Optional, Tuple, Set, Union, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import logging
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
import heapq
from queue import PriorityQueue, Queue
import copy
import json
import asyncio
import weakref
import gc
import platform
import os
import socket
import subprocess
from datetime import datetime, timedelta

# System monitoring imports
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available - using fallback resource monitoring")

# GPU monitoring imports
try:
    import pynvml
    NVIDIA_AVAILABLE = True
except ImportError:
    NVIDIA_AVAILABLE = False

# Making module standalone - define required types locally
class ActionType(Enum):
    """Action types for the system."""
    EXPLORE = "explore"
    OPTIMIZE = "optimize"
    MAINTAIN = "maintain"
    LEARN = "learn"
    OPTIMIZED_ACTION = "optimized_action"

class GoalType(Enum):
    """Goal types for planning."""
    LEARNING = "learning"
    OPTIMIZATION = "optimization"
    MAINTENANCE = "maintenance"
    EXPLORATION = "exploration"

# Base class for hierarchical planning
class HierarchicalGoalSystem:
    """Base hierarchical goal system."""
    def decompose_goal(self, goal: str, context: Dict) -> List[Dict]:
        """Decompose goal into subgoals."""
        subgoals = []
        if 'optimize' in goal.lower():
            subgoals.extend([
                {'subgoal': 'explore_solution_space', 'priority': 0.8, 'resources_required': {'cpu': 1}},
                {'subgoal': 'learn_patterns', 'priority': 0.9, 'resources_required': {'memory': 2}},
                {'subgoal': 'optimize_parameters', 'priority': 1.0, 'resources_required': {'compute': 3}}
            ])
        elif 'learn' in goal.lower():
            subgoals.extend([
                {'subgoal': 'explore_data', 'priority': 0.7, 'resources_required': {'cpu': 1}},
                {'subgoal': 'learn_representation', 'priority': 0.9, 'resources_required': {'memory': 3}}
            ])
        else:
            subgoals.append({'subgoal': 'maintain_state', 'priority': 0.5, 'resources_required': {'cpu': 0.5}})
        return subgoals

# Module imports with graceful fallback
AgentConfig = None
SafetyValidator = None
UnifiedRuntime = None

logger = logging.getLogger(__name__)

# ============================================================
# ENHANCED MONITORING TYPES
# ============================================================

class ResourceType(Enum):
    """Types of system resources to monitor."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    DISK = "disk"
    NETWORK = "network"
    POWER = "power"
    TEMPERATURE = "temperature"

class OperationalMode(Enum):
    """System operational modes for graceful degradation."""
    FULL = "full"
    PERFORMANCE = "performance"
    BALANCED = "balanced"
    POWER_SAVING = "power_saving"
    LIMITED = "limited"
    LOCAL = "local"
    SURVIVAL = "survival"
    EMERGENCY = "emergency"

class ConnectivityLevel(Enum):
    """Network connectivity levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    DEGRADED = "degraded"
    INTERMITTENT = "intermittent"
    LOCAL_ONLY = "local_only"
    OFFLINE = "offline"

# ============================================================
# PLANNING TYPES
# ============================================================

class PlanningMethod(Enum):
    """Planning methods available."""
    HIERARCHICAL = "hierarchical"
    MCTS = "mcts"
    A_STAR = "a_star"
    STRIPS = "strips"
    PARTIAL_ORDER = "partial_order"
    TEMPORAL = "temporal"
    PROBABILISTIC = "probabilistic"

@dataclass
class PlanStep:
    """Single step in a plan."""
    step_id: str
    action: str
    preconditions: List[str] = field(default_factory=list)
    effects: List[str] = field(default_factory=list)
    resources: Dict[str, float] = field(default_factory=dict)
    duration: float = 1.0
    probability: float = 1.0
    status: str = "pending"
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Plan:
    """Complete plan representation."""
    plan_id: str
    goal: str
    context: Dict[str, Any]
    steps: List[PlanStep] = field(default_factory=list)
    total_cost: float = 0.0
    expected_duration: float = 0.0
    success_probability: float = 1.0
    created_at: float = field(default_factory=time.time)
    
    def add_step(self, step: PlanStep):
        """Add step to plan."""
        self.steps.append(step)
        self.total_cost += sum(step.resources.values())
        self.expected_duration += step.duration
        self.success_probability *= step.probability
    
    def optimize(self):
        """Optimize step ordering based on dependencies."""
        dependency_graph = defaultdict(list)
        in_degree = defaultdict(int)
        
        for step in self.steps:
            for dep in step.dependencies:
                dependency_graph[dep].append(step.step_id)
                in_degree[step.step_id] += 1
        
        queue = deque([step.step_id for step in self.steps 
                      if in_degree[step.step_id] == 0])
        
        optimized_order = []
        while queue:
            current = queue.popleft()
            optimized_order.append(current)
            
            for neighbor in dependency_graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        step_map = {step.step_id: step for step in self.steps}
        self.steps = [step_map[step_id] for step_id in optimized_order 
                     if step_id in step_map]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert plan to dictionary."""
        return {
            'plan_id': self.plan_id,
            'goal': self.goal,
            'num_steps': len(self.steps),
            'total_cost': self.total_cost,
            'expected_duration': self.expected_duration,
            'success_probability': self.success_probability,
            'steps': [s.__dict__ for s in self.steps]
        }

@dataclass
class SystemState:
    """Complete system state snapshot."""
    timestamp: float
    cpu_percent: float
    cpu_freq: float
    cpu_temp: Optional[float]
    memory_used_mb: float
    memory_percent: float
    gpu_percent: Optional[float]
    gpu_memory_mb: Optional[float]
    gpu_temp: Optional[float]
    disk_usage_percent: float
    network_quality: str
    power_watts: Optional[float]
    operational_mode: OperationalMode
    
    def get_critical_resources(self) -> List[str]:
        """Identify resources under critical load."""
        critical = []
        if self.cpu_percent > 90:
            critical.append("cpu")
        if self.memory_percent > 90:
            critical.append("memory")
        if self.gpu_percent and self.gpu_percent > 90:
            critical.append("gpu")
        if self.disk_usage_percent > 90:
            critical.append("disk")
        return critical

# ============================================================
# ENHANCED RESOURCE MONITOR
# ============================================================

class EnhancedResourceMonitor:
    """Real system resource monitoring with predictive analytics."""
    
    def __init__(self, sampling_interval: float = 1.0):
        self.sampling_interval = sampling_interval
        self.history = defaultdict(lambda: deque(maxlen=100))
        self.current_state = None
        self.gpu_initialized = self._init_gpu_monitoring()
        self.platform = platform.system()
        self.monitor_thread = None
        self.stop_monitoring = threading.Event()
        self.start_monitoring()
    
    def _init_gpu_monitoring(self) -> bool:
        """Initialize GPU monitoring if available."""
        if not NVIDIA_AVAILABLE:
            return False
        
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count > 0:
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                logger.info(f"GPU monitoring initialized for {device_count} device(s)")
                return True
        except Exception as e:
            logger.debug(f"GPU monitoring not available: {e}")
        
        return False
    
    def start_monitoring(self):
        """Start background monitoring thread."""
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self.stop_monitoring.clear()
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while not self.stop_monitoring.is_set():
            try:
                self.current_state = self.collect_metrics()
                self._update_history(self.current_state)
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
            
            time.sleep(self.sampling_interval)
    
    def collect_metrics(self) -> SystemState:
        """Collect real system metrics."""
        timestamp = time.time()
        
        if PSUTIL_AVAILABLE:
            # Real metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_freq = psutil.cpu_freq()
            cpu_freq_current = cpu_freq.current if cpu_freq else 0
            
            memory = psutil.virtual_memory()
            memory_used_mb = memory.used / (1024 * 1024)
            memory_percent = memory.percent
            
            disk = psutil.disk_usage('/')
            disk_usage_percent = disk.percent
            
            # CPU temperature
            cpu_temp = self._get_cpu_temperature()
            
        else:
            # Fallback to simulated metrics
            cpu_percent = np.random.uniform(20, 80)
            cpu_freq_current = 2400
            memory_used_mb = np.random.uniform(1000, 7000)
            memory_percent = np.random.uniform(20, 80)
            disk_usage_percent = np.random.uniform(30, 70)
            cpu_temp = np.random.uniform(40, 70)
        
        # GPU metrics
        gpu_percent, gpu_memory_mb, gpu_temp = self._get_gpu_metrics()
        
        # Power metrics
        power_watts = self._get_power_consumption()
        
        # Network quality
        network_quality = self._assess_network_quality()
        
        # Determine operational mode
        operational_mode = self._determine_operational_mode(
            cpu_percent, memory_percent, gpu_percent
        )
        
        return SystemState(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            cpu_freq=cpu_freq_current,
            cpu_temp=cpu_temp,
            memory_used_mb=memory_used_mb,
            memory_percent=memory_percent,
            gpu_percent=gpu_percent,
            gpu_memory_mb=gpu_memory_mb,
            gpu_temp=gpu_temp,
            disk_usage_percent=disk_usage_percent,
            network_quality=network_quality,
            power_watts=power_watts,
            operational_mode=operational_mode
        )
    
    def _get_cpu_temperature(self) -> Optional[float]:
        """Get CPU temperature."""
        if not PSUTIL_AVAILABLE:
            return np.random.uniform(40, 70)
        
        try:
            if self.platform == "Linux":
                temps = psutil.sensors_temperatures()
                if temps:
                    for name, entries in temps.items():
                        if entries:
                            return entries[0].current
            elif self.platform == "Windows":
                # Windows requires additional tools
                return None
        except:
            pass
        
        return None
    
    def _get_gpu_metrics(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Get GPU metrics."""
        if not self.gpu_initialized:
            return None, None, None
        
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            temp = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
            
            return util.gpu, mem_info.used / (1024 * 1024), temp
        except:
            return None, None, None
    
    def _get_power_consumption(self) -> Optional[float]:
        """Estimate power consumption."""
        if self.gpu_initialized:
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle)
                return power_mw / 1000.0
            except:
                pass
        
        # Estimate based on CPU usage
        if self.current_state:
            return 10 + (self.current_state.cpu_percent * 0.5)
        
        return None
    
    def _assess_network_quality(self) -> str:
        """Enhanced network quality assessment with failure detection."""
        # Test multiple endpoints for redundancy
        test_endpoints = [
            ("8.8.8.8", 53),      # Google DNS
            ("1.1.1.1", 53),      # Cloudflare DNS
            ("208.67.222.222", 53)  # OpenDNS
        ]
        
        successful_tests = 0
        total_latency = 0
        
        for host, port in test_endpoints:
            try:
                start_time = time.time()
                socket.create_connection((host, port), timeout=2).close()
                latency = (time.time() - start_time) * 1000  # ms
                successful_tests += 1
                total_latency += latency
            except:
                pass
        
        # Track network quality in history
        success_rate = successful_tests / len(test_endpoints)
        self.history['network_success'].append(success_rate)
        
        # Determine quality based on success rate and latency
        if successful_tests == 0:
            return ConnectivityLevel.OFFLINE.value
        elif successful_tests == 1:
            return ConnectivityLevel.INTERMITTENT.value
        elif successful_tests == 2:
            avg_latency = total_latency / successful_tests
            if avg_latency > 500:
                return ConnectivityLevel.DEGRADED.value
            else:
                return ConnectivityLevel.GOOD.value
        else:  # All 3 successful
            avg_latency = total_latency / successful_tests
            if avg_latency < 100:
                return ConnectivityLevel.EXCELLENT.value
            else:
                return ConnectivityLevel.GOOD.value
    
    def _determine_operational_mode(self, cpu: float, memory: float, 
                                   gpu: Optional[float]) -> OperationalMode:
        """Determine appropriate operational mode based on resources."""
        if cpu > 90 or memory > 90:
            return OperationalMode.EMERGENCY
        elif cpu > 80 or memory > 80:
            return OperationalMode.SURVIVAL
        elif cpu > 70 or memory > 70:
            return OperationalMode.LIMITED
        elif cpu > 50 or memory > 50:
            return OperationalMode.BALANCED
        else:
            return OperationalMode.FULL
    
    def _update_history(self, state: SystemState):
        """Update historical data."""
        self.history['cpu'].append(state.cpu_percent)
        self.history['memory'].append(state.memory_percent)
        if state.gpu_percent is not None:
            self.history['gpu'].append(state.gpu_percent)
    
    def get_resource_availability(self) -> Dict[str, float]:
        """Get current resource availability."""
        if not self.current_state:
            return {'cpu': 50, 'memory': 4000, 'gpu': 50}
        
        return {
            'cpu': 100 - self.current_state.cpu_percent,
            'memory': 8000 - self.current_state.memory_used_mb,
            'gpu': 100 - (self.current_state.gpu_percent or 100),
            'disk': 100 - self.current_state.disk_usage_percent,
            'energy': 1000000 - (self.current_state.power_watts or 0) * 1000
        }
    
    def cleanup(self):
        """Cleanup monitoring."""
        self.stop_monitoring.set()
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        
        if self.gpu_initialized:
            try:
                pynvml.nvmlShutdown()
            except:
                pass

# ============================================================
# SURVIVAL PROTOCOL
# ============================================================

class SurvivalProtocol:
    """Manages system survival under resource constraints."""
    
    def __init__(self):
        self.current_mode = OperationalMode.FULL
        self.connectivity_level = ConnectivityLevel.GOOD
        self.mode_history = deque(maxlen=100)
        self.capabilities = self._initialize_capabilities()
        self.resource_monitor = EnhancedResourceMonitor()
        self.degradation_configs = self._define_degradation_configs()
    
    def _initialize_capabilities(self) -> Dict[str, Dict]:
        """Define system capabilities and their resource requirements."""
        return {
            'mcts_planning': {
                'enabled': True,
                'cpu_cost': 30,
                'memory_cost': 500,
                'fallback': 'simple_planning',
                'min_mode': OperationalMode.BALANCED
            },
            'gpu_inference': {
                'enabled': True,
                'gpu_cost': 50,
                'memory_cost': 1000,
                'fallback': 'cpu_inference',
                'min_mode': OperationalMode.BALANCED
            },
            'distributed_coordination': {
                'enabled': True,
                'cpu_cost': 20,
                'network_required': True,
                'min_mode': OperationalMode.FULL
            },
            'telemetry': {
                'enabled': True,
                'cpu_cost': 5,
                'network_required': True,
                'min_mode': OperationalMode.FULL
            },
            'advanced_optimization': {
                'enabled': True,
                'cpu_cost': 25,
                'memory_cost': 200,
                'min_mode': OperationalMode.BALANCED
            }
        }
    
    def _define_degradation_configs(self) -> Dict[OperationalMode, Dict]:
        """Define configuration for each operational mode."""
        return {
            OperationalMode.FULL: {
                'all_capabilities': True,
                'cache_size': 2048,
                'parallel_workers': 8,
                'use_gpu': True
            },
            OperationalMode.BALANCED: {
                'disable': ['telemetry'],
                'cache_size': 1024,
                'parallel_workers': 4,
                'use_gpu': True
            },
            OperationalMode.POWER_SAVING: {
                'disable': ['telemetry', 'advanced_optimization', 'gpu_inference'],
                'cache_size': 512,
                'parallel_workers': 2,
                'use_gpu': False
            },
            OperationalMode.LIMITED: {
                'disable': ['mcts_planning', 'distributed_coordination', 'telemetry', 
                          'advanced_optimization', 'gpu_inference'],
                'cache_size': 256,
                'parallel_workers': 1,
                'use_gpu': False
            },
            OperationalMode.SURVIVAL: {
                'disable': ['mcts_planning', 'distributed_coordination', 'telemetry',
                          'advanced_optimization', 'gpu_inference'],
                'use_fallbacks': True,
                'cache_size': 128,
                'parallel_workers': 1,
                'use_gpu': False
            },
            OperationalMode.EMERGENCY: {
                'disable_all_except': ['core_planning'],
                'cache_size': 64,
                'parallel_workers': 1,
                'use_gpu': False,
                'emergency_mode': True
            }
        }
    
    def assess_and_adapt(self):
        """Assess situation and adapt operational mode."""
        state = self.resource_monitor.current_state
        if not state:
            return
        
        new_mode = state.operational_mode
        
        if new_mode != self.current_mode:
            self.change_mode(new_mode)
    
    def change_mode(self, new_mode: OperationalMode):
        """Change operational mode with proper degradation."""
        if new_mode == self.current_mode:
            return
        
        logger.info(f"Changing operational mode: {self.current_mode.value} -> {new_mode.value}")
        
        # Record change
        self.mode_history.append({
            'timestamp': time.time(),
            'from': self.current_mode,
            'to': new_mode
        })
        
        # Apply degradation config
        config = self.degradation_configs.get(new_mode, {})
        
        # Disable capabilities based on mode
        if 'disable' in config:
            for cap in config['disable']:
                if cap in self.capabilities:
                    self.capabilities[cap]['enabled'] = False
                    # Enable fallback if available
                    fallback = self.capabilities[cap].get('fallback')
                    if fallback and fallback in self.capabilities:
                        self.capabilities[fallback] = {'enabled': True}
        
        elif 'disable_all_except' in config:
            keep = config['disable_all_except']
            for cap in self.capabilities:
                self.capabilities[cap]['enabled'] = cap in keep
        
        self.current_mode = new_mode
    
    def is_capability_available(self, capability: str) -> bool:
        """Check if a capability is available in current mode."""
        if capability not in self.capabilities:
            return False
        
        cap_info = self.capabilities[capability]
        
        # Check if enabled
        if not cap_info.get('enabled', False):
            return False
        
        # Check minimum mode requirement
        min_mode = cap_info.get('min_mode', OperationalMode.FULL)
        mode_order = [
            OperationalMode.EMERGENCY,
            OperationalMode.SURVIVAL,
            OperationalMode.LIMITED,
            OperationalMode.POWER_SAVING,
            OperationalMode.BALANCED,
            OperationalMode.FULL
        ]
        
        current_idx = mode_order.index(self.current_mode)
        min_idx = mode_order.index(min_mode)
        
        return current_idx >= min_idx
    
    def detect_network_failure(self) -> Dict[str, Any]:
        """Detect and analyze network failures.
        
        Returns:
            Dict containing failure status, connectivity level, and recommended actions
        """
        state = self.resource_monitor.current_state
        if not state:
            return {
                'failure_detected': False,
                'connectivity': 'unknown',
                'actions': []
            }
        
        connectivity = state.network_quality
        history = self.resource_monitor.history.get('network_success', deque())
        
        # Analyze network history for patterns
        recent_failures = 0
        if len(history) >= 3:
            recent_failures = sum(1 for rate in list(history)[-3:] if rate < 0.5)
        
        failure_detected = connectivity in ['offline', 'intermittent'] or recent_failures >= 2
        
        # Determine recommended actions
        actions = []
        if failure_detected:
            actions.append('switch_to_local_mode')
            if connectivity == 'offline':
                actions.append('disable_network_capabilities')
                actions.append('activate_survival_mode')
            elif connectivity == 'intermittent':
                actions.append('enable_retry_logic')
                actions.append('reduce_network_dependencies')
        
        # Update connectivity level
        if connectivity == 'offline':
            self.connectivity_level = ConnectivityLevel.OFFLINE
        elif connectivity == 'intermittent':
            self.connectivity_level = ConnectivityLevel.INTERMITTENT
        elif connectivity in ['degraded', 'local_only']:
            self.connectivity_level = ConnectivityLevel.DEGRADED
        else:
            self.connectivity_level = ConnectivityLevel.GOOD
        
        return {
            'failure_detected': failure_detected,
            'connectivity': connectivity,
            'recent_failures': recent_failures,
            'actions': actions,
            'connectivity_level': self.connectivity_level.value
        }
    
    def apply_graceful_degradation(self, failure_info: Dict[str, Any]):
        """Apply graceful degradation based on detected failures.
        
        Args:
            failure_info: Network failure information from detect_network_failure()
        """
        if not failure_info['failure_detected']:
            return
        
        logger.warning(f"Applying graceful degradation: {failure_info}")
        
        # Execute recommended actions
        for action in failure_info['actions']:
            if action == 'switch_to_local_mode':
                self._switch_to_local_mode()
            elif action == 'disable_network_capabilities':
                self._disable_network_capabilities()
            elif action == 'activate_survival_mode':
                self.change_mode(OperationalMode.SURVIVAL)
            elif action == 'enable_retry_logic':
                self._enable_network_retry()
            elif action == 'reduce_network_dependencies':
                self._reduce_network_load()
    
    def _switch_to_local_mode(self):
        """Switch system to local-only operation."""
        logger.info("Switching to local-only mode")
        for cap_name, cap_info in self.capabilities.items():
            if cap_info.get('network_required', False):
                cap_info['enabled'] = False
                # Activate fallback if available
                fallback = cap_info.get('fallback')
                if fallback and fallback in self.capabilities:
                    self.capabilities[fallback]['enabled'] = True
    
    def _disable_network_capabilities(self):
        """Disable all network-dependent capabilities."""
        logger.info("Disabling network-dependent capabilities")
        disabled = []
        for cap_name, cap_info in self.capabilities.items():
            if cap_info.get('network_required', False) and cap_info.get('enabled', False):
                cap_info['enabled'] = False
                disabled.append(cap_name)
        logger.info(f"Disabled capabilities: {disabled}")
    
    def _enable_network_retry(self):
        """Enable retry logic for network operations."""
        logger.info("Enabling network retry logic")
        # This would be implemented by the specific network clients
        # Here we just set a flag that can be checked
        self.network_retry_enabled = True
    
    def _reduce_network_load(self):
        """Reduce network load by batching and prioritizing operations."""
        logger.info("Reducing network load")
        # This would adjust batch sizes and request priorities
        self.network_batch_size = 10  # Reduced from default
        self.network_priority_threshold = 0.8  # Only critical operations

# ============================================================
# POWER MANAGER
# ============================================================

class PowerManager:
    """Manages power consumption and thermal profiles."""
    
    def __init__(self):
        self.power_profiles = {
            "performance": {
                "cpu_limit": 100,
                "gpu_enabled": True,
                "frequency_scaling": "performance",
                "thermal_limit": 85
            },
            "balanced": {
                "cpu_limit": 70,
                "gpu_enabled": True,
                "frequency_scaling": "ondemand",
                "thermal_limit": 75
            },
            "power_saver": {
                "cpu_limit": 40,
                "gpu_enabled": False,
                "frequency_scaling": "powersave",
                "thermal_limit": 65
            },
            "survival": {
                "cpu_limit": 20,
                "gpu_enabled": False,
                "frequency_scaling": "powersave",
                "thermal_limit": 60
            }
        }
        self.current_profile = "balanced"
        self.thermal_throttle_active = False
        self.battery_available = self._detect_battery()
        self.on_battery_power = False
        self.battery_percent = 100
        self.emergency_shutdown_threshold = 5  # percent
    
    def _detect_battery(self) -> bool:
        """Detect if system has battery (laptop/mobile)."""
        if not PSUTIL_AVAILABLE:
            return False
        
        try:
            battery = psutil.sensors_battery()
            return battery is not None
        except:
            return False
    
    def check_power_status(self) -> Dict[str, Any]:
        """Check current power status including battery.
        
        Returns:
            Dict with power status, battery level, and recommended actions
        """
        if not self.battery_available or not PSUTIL_AVAILABLE:
            return {
                'on_battery': False,
                'battery_percent': None,
                'power_warning': False,
                'actions': []
            }
        
        try:
            battery = psutil.sensors_battery()
            if battery:
                self.on_battery_power = not battery.power_plugged
                self.battery_percent = battery.percent
                
                actions = []
                power_warning = False
                
                # Critical battery level
                if self.battery_percent <= self.emergency_shutdown_threshold:
                    actions.append('emergency_shutdown')
                    power_warning = True
                # Low battery
                elif self.battery_percent <= 15:
                    actions.append('activate_survival_mode')
                    actions.append('save_state')
                    power_warning = True
                # Moderate battery on battery power
                elif self.battery_percent <= 30 and self.on_battery_power:
                    actions.append('activate_power_saver')
                    power_warning = True
                
                return {
                    'on_battery': self.on_battery_power,
                    'battery_percent': self.battery_percent,
                    'power_warning': power_warning,
                    'time_remaining': battery.secsleft if hasattr(battery, 'secsleft') else None,
                    'actions': actions
                }
        except Exception as e:
            logger.error(f"Error checking battery status: {e}")
        
        return {
            'on_battery': False,
            'battery_percent': None,
            'power_warning': False,
            'actions': []
        }
    
    def apply_power_management(self, power_status: Dict[str, Any]):
        """Apply power management based on power status.
        
        Args:
            power_status: Power status from check_power_status()
        """
        if not power_status['power_warning']:
            return
        
        logger.warning(f"Applying power management: {power_status}")
        
        for action in power_status['actions']:
            if action == 'emergency_shutdown':
                self._emergency_shutdown()
            elif action == 'activate_survival_mode':
                self.set_power_profile('survival')
            elif action == 'activate_power_saver':
                self.set_power_profile('power_saver')
            elif action == 'save_state':
                logger.info("Saving system state for emergency recovery")
    
    def _emergency_shutdown(self):
        """Perform emergency shutdown to prevent data loss.
        
        This should trigger graceful shutdown of all operations.
        """
        logger.critical(f"EMERGENCY SHUTDOWN: Battery at {self.battery_percent}%")
        # In a real implementation, this would trigger cleanup and shutdown
        # For now, just set the most restrictive profile
        self.set_power_profile('survival')
    
    def set_power_profile(self, profile: str):
        """Set power management profile."""
        if profile not in self.power_profiles:
            return
        
        self.current_profile = profile
        config = self.power_profiles[profile]
        
        # Apply CPU frequency scaling (Linux only)
        if platform.system() == "Linux" and os.path.exists("/sys/devices/system/cpu/cpu0/cpufreq"):
            try:
                governor = config["frequency_scaling"]
                for cpu in range(psutil.cpu_count()):
                    governor_path = f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor"
                    if os.path.exists(governor_path):
                        with open(governor_path, 'w') as f:
                            f.write(governor)
            except Exception as e:
                logger.debug(f"Could not set CPU governor: {e}")
    
    def check_thermal_status(self, current_temp: Optional[float]) -> bool:
        """Check thermal status and throttle if needed."""
        if not current_temp:
            return False
        
        profile = self.power_profiles[self.current_profile]
        thermal_limit = profile["thermal_limit"]
        
        if current_temp > thermal_limit:
            if not self.thermal_throttle_active:
                logger.warning(f"Thermal throttling activated: {current_temp}°C > {thermal_limit}°C")
                self.thermal_throttle_active = True
            return True
        elif current_temp < (thermal_limit - 10):
            if self.thermal_throttle_active:
                logger.info("Thermal throttling deactivated")
                self.thermal_throttle_active = False
        
        return self.thermal_throttle_active
    
    def get_power_budget(self) -> Dict[str, Any]:
        """Get current power budget based on profile."""
        profile = self.power_profiles[self.current_profile]
        return {
            'cpu_percent': profile['cpu_limit'],
            'gpu_enabled': profile['gpu_enabled'],
            'thermal_headroom': not self.thermal_throttle_active
        }

# ============================================================
# MONTE CARLO TREE SEARCH (Fixed - same as original)
# ============================================================

class MCTSNode:
    """Node in MCTS tree with proper memory management."""
    
    def __init__(self, state: Any, parent: Optional['MCTSNode'] = None, 
                 action: Optional[Any] = None):
        self.state = state
        self._parent_ref = weakref.ref(parent) if parent is not None else None
        self.action = action
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.untried_actions = None
        self._lock = threading.Lock()
    
    @property
    def parent(self):
        """Get parent node (may be None if parent was garbage collected)."""
        if self._parent_ref is None:
            return None
        return self._parent_ref()
    
    def ucb1(self, c: float = 1.414) -> float:
        """Upper Confidence Bound calculation."""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.value / self.visits
        
        parent = self.parent
        if parent:
            exploration = c * np.sqrt(np.log(parent.visits) / self.visits)
        else:
            exploration = 0
        
        return exploitation + exploration
    
    def best_child(self, c: float = 1.414) -> 'MCTSNode':
        """Select best child using UCB1."""
        return max(self.children.values(), key=lambda n: n.ucb1(c))
    
    def update(self, reward: float):
        """Update node statistics."""
        with self._lock:
            self.visits += 1
            self.value += reward
    
    def cleanup(self):
        """Explicit cleanup of node and children."""
        with self._lock:
            for child in self.children.values():
                child.cleanup()
            self.children.clear()
            self._parent_ref = None
            self.state = None
            self.action = None
            self.untried_actions = None

class MonteCarloTreeSearch:
    """Enhanced MCTS for planning with proper memory management."""
    
    def __init__(self, simulation_budget: int = 1000, max_depth: int = 50,
                 exploration_constant: float = 1.414):
        self.simulation_budget = simulation_budget
        self.max_depth = max_depth
        self.exploration_constant = exploration_constant
        self.root = None
        self.action_generator = None
        self.state_evaluator = None
        self.transition_model = None
        self._cleanup_lock = threading.Lock()
    
    def search(self, initial_state: Any, 
              available_actions: Optional[List[Any]] = None) -> Tuple[Any, float]:
        """Run MCTS synchronously to find best action."""
        self._cleanup_tree()
        
        self.root = MCTSNode(initial_state)
        
        if available_actions:
            self.root.untried_actions = available_actions.copy()
        else:
            self.root.untried_actions = self._get_actions(initial_state)
        
        for sim in range(self.simulation_budget):
            node = self._tree_policy(self.root)
            reward = self._default_policy(node.state)
            self._backup(node, reward)
            
            if sim % 100 == 0 and sim > 0:
                self._progressive_widening(self.root)
            
            if sim % 200 == 0 and sim > 0:
                self._prune_tree()
        
        if self.root.children:
            best_child = max(self.root.children.values(), key=lambda n: n.visits)
            return best_child.action, best_child.value / max(best_child.visits, 1)
        
        if self.root.untried_actions:
            return self.root.untried_actions[0], 0.0
        
        return None, 0.0
    
    async def search_async(self, initial_state: Any, 
                          available_actions: Optional[List[Any]] = None) -> Tuple[Any, float]:
        """Async MCTS with parallel simulations."""
        self._cleanup_tree()
        
        self.root = MCTSNode(initial_state)
        
        if available_actions:
            self.root.untried_actions = available_actions.copy()
        else:
            self.root.untried_actions = self._get_actions(initial_state)
        
        batch_size = min(self.simulation_budget, 50)
        total_simulations = 0
        
        while total_simulations < self.simulation_budget:
            current_batch = min(batch_size, self.simulation_budget - total_simulations)
            tasks = []
            
            for _ in range(current_batch):
                tasks.append(self._simulate_async())
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Simulation error: {result}")
                    continue
                    
                node, reward = result
                self._backup(node, reward)
            
            total_simulations += current_batch
            
            if total_simulations % 100 == 0 and total_simulations > 0:
                self._progressive_widening(self.root)
            
            if total_simulations % 200 == 0:
                self._prune_tree()
                gc.collect()
        
        if self.root.children:
            best_child = max(self.root.children.values(), key=lambda n: n.visits)
            return best_child.action, best_child.value / max(best_child.visits, 1)
        
        if self.root.untried_actions:
            return self.root.untried_actions[0], 0.0
        
        return None, 0.0
    
    async def _simulate_async(self) -> Tuple[MCTSNode, float]:
        """Single async simulation."""
        try:
            loop = asyncio.get_event_loop()
            node = await loop.run_in_executor(None, self._tree_policy, self.root)
            reward = await loop.run_in_executor(None, self._default_policy, node.state)
            return node, reward
        except Exception as e:
            logger.error(f"Simulation error: {e}")
            return self.root, 0.0
    
    def _tree_policy(self, node: MCTSNode) -> MCTSNode:
        """Select or expand node using tree policy."""
        depth = 0
        
        while not self._is_terminal(node.state) and depth < self.max_depth:
            if node.untried_actions:
                return self._expand(node)
            elif node.children:
                node = node.best_child(self.exploration_constant)
                depth += 1
            else:
                node.untried_actions = self._get_actions(node.state)
                if node.untried_actions:
                    return self._expand(node)
                break
        
        return node
    
    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expand tree with new node."""
        if not node.untried_actions:
            return node
        
        action_idx = np.random.randint(len(node.untried_actions))
        action = node.untried_actions.pop(action_idx)
        
        next_state = self._simulate_action(node.state, action)
        
        child = MCTSNode(next_state, parent=node, action=action)
        node.children[action] = child
        
        child.untried_actions = self._get_actions(next_state)
        
        return child
    
    def _default_policy(self, state: Any) -> float:
        """Random rollout for value estimation."""
        current_state = state
        depth = 0
        cumulative_reward = 0
        discount = 0.99
        
        while not self._is_terminal(current_state) and depth < self.max_depth:
            actions = self._get_actions(current_state)
            if not actions:
                break
            
            action = np.random.choice(actions)
            current_state = self._simulate_action(current_state, action)
            
            reward = self._evaluate_state(current_state)
            cumulative_reward += (discount ** depth) * reward
            
            depth += 1
        
        terminal_value = self._evaluate_state(current_state)
        cumulative_reward += (discount ** depth) * terminal_value
        
        return cumulative_reward
    
    def _backup(self, node: MCTSNode, reward: float):
        """Backpropagate reward up the tree."""
        current = node
        discount = 0.99
        depth = 0
        
        while current is not None:
            current.update(reward * (discount ** depth))
            current = current.parent
            depth += 1
    
    def _progressive_widening(self, node: MCTSNode):
        """Add new actions progressively."""
        if node.visits > len(node.children) * 10:
            new_actions = self._get_new_actions(node.state)
            if new_actions and node.untried_actions is not None:
                node.untried_actions.extend(new_actions[:1])
    
    def _prune_tree(self):
        """Prune least visited branches."""
        with self._cleanup_lock:
            if not self.root or not self.root.children:
                return
            
            children_list = list(self.root.children.items())
            if len(children_list) > 10:
                children_list.sort(key=lambda x: x[1].visits, reverse=True)
                keep_count = max(5, len(children_list) // 2)
                
                for action, child in children_list[keep_count:]:
                    child.cleanup()
                    del self.root.children[action]
    
    def _cleanup_tree(self):
        """Clean up entire tree."""
        with self._cleanup_lock:
            if self.root:
                self.root.cleanup()
                self.root = None
            
            gc.collect()
    
    def _is_terminal(self, state: Any) -> bool:
        """Check if state is terminal."""
        if hasattr(state, 'is_terminal'):
            return state.is_terminal()
        return False
    
    def _get_actions(self, state: Any) -> List[Any]:
        """Get available actions for state."""
        if self.action_generator:
            return self.action_generator(state)
        
        if hasattr(state, 'get_actions'):
            return state.get_actions()
        
        return [ActionType.EXPLORE, ActionType.OPTIMIZE, ActionType.MAINTAIN]
    
    def _get_new_actions(self, state: Any) -> List[Any]:
        """Get additional actions for progressive widening."""
        return []
    
    def _simulate_action(self, state: Any, action: Any) -> Any:
        """Simulate action execution."""
        if self.transition_model:
            return self.transition_model(state, action)
        
        if hasattr(state, 'apply_action'):
            return state.apply_action(action)
        
        return copy.deepcopy(state)
    
    def _evaluate_state(self, state: Any) -> float:
        """Evaluate state value."""
        if self.state_evaluator:
            return self.state_evaluator(state)
        
        if hasattr(state, 'evaluate'):
            return state.evaluate()
        
        return np.random.random()
    
    def cleanup(self):
        """Explicit cleanup method."""
        self._cleanup_tree()
    
    def __del__(self):
        """Destructor."""
        try:
            self.cleanup()
        except:
            pass

# ============================================================
# ENHANCED HIERARCHICAL GOAL SYSTEM
# ============================================================

class EnhancedHierarchicalPlanner(HierarchicalGoalSystem):
    """Enhanced hierarchical planning with real resource awareness."""
    
    def __init__(self):
        super().__init__()
        self.plan_library = PlanLibrary()
        self.plan_monitor = PlanMonitor()
        self.plan_repairer = PlanRepairer()
        self.mcts = MonteCarloTreeSearch()
        
        # Enhanced components
        self.resource_monitor = EnhancedResourceMonitor()
        self.survival_protocol = SurvivalProtocol()
        self.power_manager = PowerManager()
        
        self._lock = threading.RLock()
    
    def generate_plan(self, goal: str, context: Dict[str, Any], 
                      method: PlanningMethod = PlanningMethod.HIERARCHICAL) -> Plan:
        """Generates plan with resource and mode awareness."""
        # Check operational mode
        self.survival_protocol.assess_and_adapt()
        
        # Check if MCTS is available in current mode
        if method == PlanningMethod.MCTS:
            if not self.survival_protocol.is_capability_available('mcts_planning'):
                logger.info("MCTS not available in current mode, falling back to hierarchical")
                method = PlanningMethod.HIERARCHICAL
        
        plan = self.create_plan(goal, context, method)
        
        if SafetyValidator and AgentConfig:
            try:
                config = AgentConfig()
                validator = SafetyValidator(config)
                
                is_safe, reason, _ = validator.validate_action(plan.to_dict(), context)
                
                if not is_safe:
                    logger.error(f"Generated plan for goal '{goal}' failed safety validation: {reason}")
                    raise ValueError(f"Unsafe plan generated: {reason}")
            except Exception as e:
                logger.error(f"An error occurred during plan validation: {e}")
                raise ValueError(f"Unsafe plan: validation failed. Reason: {e}")
        
        return plan
    
    def create_plan(self, goal: str, context: Dict[str, Any],
                   method: PlanningMethod = PlanningMethod.HIERARCHICAL) -> Plan:
        """Create plan with resource-aware selection."""
        # Get resource availability
        resources = self.resource_monitor.get_resource_availability()
        
        # Check plan library first
        cached_plan = self.plan_library.get_plan(goal, context)
        if cached_plan and self._is_plan_valid(cached_plan, context):
            logger.info(f"Using cached plan for goal: {goal}")
            return cached_plan
        
        # Adapt method based on resources
        if method == PlanningMethod.MCTS and resources['cpu'] < 30:
            logger.info("Insufficient CPU for MCTS, using A*")
            method = PlanningMethod.A_STAR
        
        if method == PlanningMethod.MCTS:
            return self._create_mcts_plan(goal, context)
        elif method == PlanningMethod.A_STAR:
            return self._create_astar_plan(goal, context)
        else:
            return self._create_hierarchical_plan(goal, context)
    
    # [Previous planning methods remain the same]
    def _create_hierarchical_plan(self, goal: str, context: Dict[str, Any]) -> Plan:
        """Create plan using hierarchical decomposition."""
        with self._lock:
            subgoals = self.decompose_goal(goal, context)
            
            plan = Plan(
                plan_id=f"plan_{goal}_{int(time.time())}",
                goal=goal,
                context=context
            )
            
            for i, subgoal in enumerate(subgoals):
                step = PlanStep(
                    step_id=f"step_{i}_{subgoal['subgoal']}",
                    action=subgoal['subgoal'],
                    preconditions=self._extract_preconditions(subgoal),
                    effects=self._extract_effects(subgoal),
                    resources=subgoal.get('resources_required', {}),
                    duration=self._estimate_duration(subgoal),
                    probability=self._estimate_success_probability(subgoal),
                    dependencies=self._identify_dependencies(subgoal['subgoal'], 
                                                            subgoals[:i])
                )
                plan.add_step(step)
            
            plan.optimize()
            self.plan_library.store_plan(plan)
            
            return plan
    
    def _create_mcts_plan(self, goal: str, context: Dict[str, Any]) -> Plan:
        """Create plan using MCTS."""
        initial_state = PlanningState(goal=goal, context=context)
        
        self.mcts.action_generator = lambda s: self._generate_actions(s)
        self.mcts.state_evaluator = lambda s: self._evaluate_planning_state(s)
        self.mcts.transition_model = lambda s, a: self._apply_planning_action(s, a)
        
        plan_steps = []
        current_state = initial_state
        
        try:
            loop = asyncio.get_running_loop()
            return asyncio.run(self._create_mcts_plan_async(goal, context))
        except RuntimeError:
            pass
        
        for step_num in range(20):
            action, value = self.mcts.search(current_state)
            
            if action is None:
                break
            
            step = self._action_to_step(action, step_num)
            plan_steps.append(step)
            
            current_state = self._apply_planning_action(current_state, action)
            
            if self._goal_achieved(current_state, goal):
                break
        
        plan = Plan(
            plan_id=f"mcts_plan_{goal}_{int(time.time())}",
            goal=goal,
            context=context
        )
        
        for step in plan_steps:
            plan.add_step(step)
        
        self.mcts.cleanup()
        
        return plan
    
    async def _create_mcts_plan_async(self, goal: str, context: Dict[str, Any]) -> Plan:
        """Create plan using async MCTS."""
        initial_state = PlanningState(goal=goal, context=context)
        
        self.mcts.action_generator = lambda s: self._generate_actions(s)
        self.mcts.state_evaluator = lambda s: self._evaluate_planning_state(s)
        self.mcts.transition_model = lambda s, a: self._apply_planning_action(s, a)
        
        plan_steps = []
        current_state = initial_state
        
        for step_num in range(20):
            action, value = await self.mcts.search_async(current_state)
            
            if action is None:
                break
            
            step = self._action_to_step(action, step_num)
            plan_steps.append(step)
            
            current_state = self._apply_planning_action(current_state, action)
            
            if self._goal_achieved(current_state, goal):
                break
        
        plan = Plan(
            plan_id=f"mcts_plan_{goal}_{int(time.time())}",
            goal=goal,
            context=context
        )
        
        for step in plan_steps:
            plan.add_step(step)
        
        self.mcts.cleanup()
        
        return plan
    
    def _create_astar_plan(self, goal: str, context: Dict[str, Any]) -> Plan:
        """Create plan using A* search."""
        initial_state = PlanningState(goal=goal, context=context)
        goal_state = PlanningState(goal=goal, context=context, achieved=True)
        
        open_set = [(0, initial_state)]
        closed_set = set()
        g_score = {initial_state: 0}
        f_score = {initial_state: self._heuristic(initial_state, goal_state)}
        came_from = {}
        
        max_iterations = 1000
        iteration = 0
        
        while open_set and iteration < max_iterations:
            iteration += 1
            current_f, current = heapq.heappop(open_set)
            
            if self._goal_achieved(current, goal):
                path = self._reconstruct_path(came_from, current)
                return self._path_to_plan(path, goal, context)
            
            closed_set.add(current)
            
            for action in self._generate_actions(current):
                neighbor = self._apply_planning_action(current, action)
                
                if neighbor in closed_set:
                    continue
                
                tentative_g = g_score[current] + self._action_cost(action)
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = (current, action)
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal_state)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        logger.warning(f"A* search failed to find plan for goal: {goal}")
        return Plan(plan_id=f"failed_{goal}", goal=goal, context=context)
    
    # Helper methods remain the same...
    def _extract_preconditions(self, subgoal: Dict) -> List[str]:
        preconditions = []
        for resource, amount in subgoal.get('resources_required', {}).items():
            preconditions.append(f"has_{resource}>={amount}")
        for dep in subgoal.get('dependencies', []):
            preconditions.append(f"completed_{dep}")
        return preconditions
    
    def _extract_effects(self, subgoal: Dict) -> List[str]:
        effects = [f"completed_{subgoal['subgoal']}"]
        for criterion, value in subgoal.get('success_criteria', {}).items():
            effects.append(f"{criterion}={value}")
        return effects
    
    def _estimate_duration(self, subgoal: Dict) -> float:
        base_duration = 1.0
        if 'optimize' in subgoal['subgoal']:
            base_duration *= 2.0
        if 'learn' in subgoal['subgoal']:
            base_duration *= 1.5
        resource_factor = sum(subgoal.get('resources_required', {}).values())
        base_duration *= (1 + resource_factor)
        return base_duration
    
    def _estimate_success_probability(self, subgoal: Dict) -> float:
        base_prob = 0.9
        priority = subgoal.get('priority', 0.5)
        base_prob = min(1.0, base_prob * (0.5 + priority))
        if subgoal.get('conflicts'):
            base_prob *= 0.8
        return base_prob
    
    def _identify_dependencies(self, subgoal: str, previous_subgoals: List[Dict]) -> List[str]:
        dependencies = []
        if 'learn' in subgoal:
            for prev in previous_subgoals:
                if 'explore' in prev['subgoal']:
                    dependencies.append(f"step_{previous_subgoals.index(prev)}_{prev['subgoal']}")
        if 'optimize' in subgoal:
            for prev in previous_subgoals:
                if 'explore' in prev['subgoal'] or 'learn' in prev['subgoal']:
                    dependencies.append(f"step_{previous_subgoals.index(prev)}_{prev['subgoal']}")
        return dependencies
    
    def _is_plan_valid(self, plan: Plan, context: Dict) -> bool:
        age = time.time() - plan.created_at
        if age > 3600:
            return False
        for key, value in context.items():
            if key in plan.context and plan.context[key] != value:
                return False
        return True
    
    def _generate_actions(self, state: 'PlanningState') -> List[Any]:
        actions = []
        base_actions = [ActionType.EXPLORE, ActionType.OPTIMIZE, ActionType.MAINTAIN]
        if not state.achieved:
            actions.extend(base_actions)
        if 'learning' in state.goal:
            actions.append(ActionType.LEARN)
        return actions
    
    def _evaluate_planning_state(self, state: 'PlanningState') -> float:
        score = 0.0
        if state.achieved:
            score += 10.0
        total_resources = sum(state.resources_used.values())
        score -= total_resources * 0.1
        score -= len(state.steps_taken) * 0.01
        return score
    
    def _apply_planning_action(self, state: 'PlanningState', action: Any) -> 'PlanningState':
        new_state = copy.deepcopy(state)
        action_str = str(action)
        new_state.steps_taken.append(action_str)
        
        if 'explore' in action_str.lower():
            new_state.resources_used['time'] = new_state.resources_used.get('time', 0) + 1.0
        elif 'optimize' in action_str.lower():
            new_state.resources_used['compute'] = new_state.resources_used.get('compute', 0) + 2.0
        
        if len(new_state.steps_taken) > 3:
            new_state.achieved = np.random.random() > 0.3
        
        return new_state
    
    def _goal_achieved(self, state: 'PlanningState', goal: str) -> bool:
        return state.achieved
    
    def _action_to_step(self, action: Any, step_num: int) -> PlanStep:
        action_str = str(action)
        return PlanStep(
            step_id=f"mcts_step_{step_num}",
            action=action_str,
            duration=1.0,
            probability=0.8,
            resources={'compute': 1.0}
        )
    
    def _heuristic(self, state: 'PlanningState', goal_state: 'PlanningState') -> float:
        distance = 0.0
        if not state.achieved:
            distance += 5.0
        for resource in ['time', 'compute', 'memory']:
            state_val = state.resources_used.get(resource, 0)
            goal_val = goal_state.resources_used.get(resource, 0)
            distance += abs(state_val - goal_val)
        return distance
    
    def _action_cost(self, action: Any) -> float:
        action_str = str(action).lower()
        if 'explore' in action_str:
            return 1.0
        elif 'optimize' in action_str:
            return 2.0
        elif 'maintain' in action_str:
            return 0.5
        return 1.0
    
    def _reconstruct_path(self, came_from: Dict, current: 'PlanningState') -> List[Tuple['PlanningState', Any]]:
        path = []
        while current in came_from:
            prev_state, action = came_from[current]
            path.append((current, action))
            current = prev_state
        return list(reversed(path))
    
    def _path_to_plan(self, path: List[Tuple['PlanningState', Any]], 
                     goal: str, context: Dict) -> Plan:
        plan = Plan(
            plan_id=f"astar_plan_{goal}_{int(time.time())}",
            goal=goal,
            context=context
        )
        
        for i, (state, action) in enumerate(path):
            step = PlanStep(
                step_id=f"astar_step_{i}",
                action=str(action),
                duration=1.0,
                probability=0.85,
                resources={'compute': self._action_cost(action)}
            )
            plan.add_step(step)
        
        return plan
    
    def execute_plan(self, plan: Plan, executor: Optional[Callable] = None) -> Dict[str, Any]:
        """Execute plan with monitoring and repair."""
        execution_trace = []
        failed_steps = []
        
        for step in plan.steps:
            if not self._check_preconditions(step.preconditions):
                repair_success = self.plan_repairer.repair_preconditions(step)
                if not repair_success:
                    failed_steps.append(step)
                    continue
            
            step.status = "executing"
            
            if executor:
                result = executor(step)
            else:
                result = self._execute_step(step)
            
            execution_trace.append(result)
            
            if result['success']:
                step.status = "completed"
            else:
                step.status = "failed"
                failed_steps.append(step)
                
                recovery_plan = self.plan_repairer.create_recovery_plan(
                    plan, step, result.get('error', 'Unknown error')
                )
                
                if recovery_plan:
                    logger.info(f"Executing recovery plan for step {step.step_id}")
                    recovery_result = self.execute_plan(recovery_plan, executor)
                    
                    if recovery_result['success']:
                        step.status = "recovered"
                    else:
                        break
        
        success = len(failed_steps) == 0
        
        return {
            'success': success,
            'executed_steps': len(execution_trace),
            'failed_steps': [s.step_id for s in failed_steps],
            'trace': execution_trace
        }
    
    def _execute_step(self, step: PlanStep) -> Dict[str, Any]:
        """Execute a single plan step."""
        if not UnifiedRuntime:
            logger.warning("UnifiedRuntime not available. Simulating.")
            success = np.random.random() < step.probability
            return {
                'step_id': step.step_id,
                'action': step.action,
                'success': success,
                'duration': step.duration,
                'resources_used': step.resources
            }
        
        try:
            runtime = UnifiedRuntime()
            graph = {
                "nodes": [{
                    "id": step.step_id,
                    "type": step.action,
                    "params": step.metadata
                }],
                "edges": []
            }
            
            result = asyncio.run(runtime.execute_graph(graph))
            
            success = result.get('status') == 'completed'
            
            return {
                'step_id': step.step_id,
                'action': step.action,
                'success': success,
                'duration': result.get('duration_ms', step.duration * 1000),
                'resources_used': step.resources,
                'runtime_output': result.get('output')
            }
        except Exception as e:
            logger.error(f"Error executing step {step.step_id}: {e}")
            return {
                'step_id': step.step_id,
                'action': step.action,
                'success': False,
                'error': str(e)
            }
    
    def _check_preconditions(self, preconditions: List[str]) -> bool:
        """Check if preconditions are satisfied."""
        # In production, check actual conditions
        return np.random.random() > 0.1
    
    def cleanup(self):
        """Cleanup resources."""
        with self._lock:
            if self.mcts:
                self.mcts.cleanup()
            if self.plan_library:
                self.plan_library.cleanup()
            if self.resource_monitor:
                self.resource_monitor.cleanup()
            gc.collect()
    
    def __del__(self):
        """Destructor."""
        try:
            self.cleanup()
        except:
            pass

# ============================================================
# PLANNING STATE
# ============================================================

@dataclass
class PlanningState:
    """State representation for planning."""
    goal: str
    context: Dict[str, Any]
    achieved: bool = False
    steps_taken: List[str] = field(default_factory=list)
    resources_used: Dict[str, float] = field(default_factory=dict)
    
    def __hash__(self):
        return hash((self.goal, tuple(self.steps_taken)))
    
    def __eq__(self, other):
        return (self.goal == other.goal and 
                self.steps_taken == other.steps_taken)
    
    def __lt__(self, other):
        """Less than comparison for heap operations."""
        return len(self.steps_taken) < len(other.steps_taken)

# ============================================================
# PLAN LIBRARY (Same as original)
# ============================================================

class PlanLibrary:
    """Library of reusable plans."""
    
    def __init__(self, max_size: int = 1000):
        self.plans = {}
        self.access_counts = defaultdict(int)
        self.max_size = max_size
        self._lock = threading.RLock()
    
    def get_plan(self, goal: str, context: Dict) -> Optional[Plan]:
        """Retrieve plan from library."""
        key = self._make_key(goal, context)
        
        with self._lock:
            if key in self.plans:
                self.access_counts[key] += 1
                return copy.deepcopy(self.plans[key])
        
        return None
    
    def store_plan(self, plan: Plan):
        """Store plan in library."""
        key = self._make_key(plan.goal, plan.context)
        
        with self._lock:
            if len(self.plans) >= self.max_size:
                self._evict_lru()
            
            self.plans[key] = plan
            self.access_counts[key] = 1
    
    def _make_key(self, goal: str, context: Dict) -> str:
        """Create key for plan lookup."""
        context_str = json.dumps(context, sort_keys=True, default=str)
        return hashlib.md5(f"{goal}_{context_str}".encode()).hexdigest()
    
    def _evict_lru(self):
        """Evict least recently used plan."""
        if not self.access_counts:
            return
        
        lru_key = min(self.access_counts.items(), key=lambda x: x[1])[0]
        
        del self.plans[lru_key]
        del self.access_counts[lru_key]
    
    def cleanup(self):
        """Cleanup library."""
        with self._lock:
            self.plans.clear()
            self.access_counts.clear()
    
    def __del__(self):
        """Destructor."""
        try:
            self.cleanup()
        except:
            pass

# ============================================================
# PLAN MONITOR (Same as original)
# ============================================================

class PlanMonitor:
    """Monitor plan execution."""
    
    def __init__(self):
        self.execution_history = deque(maxlen=100)
        self.performance_metrics = defaultdict(list)
        self._lock = threading.RLock()
    
    def monitor_step(self, step: PlanStep, result: Dict[str, Any]):
        """Monitor step execution."""
        with self._lock:
            self.execution_history.append({
                'step': step,
                'result': result,
                'timestamp': time.time()
            })
            
            self.performance_metrics[step.action].append(result['success'])
    
    def get_action_success_rate(self, action: str) -> float:
        """Get success rate for action type."""
        with self._lock:
            if action not in self.performance_metrics:
                return 0.5
            
            successes = self.performance_metrics[action]
            return sum(successes) / len(successes) if successes else 0.5

# ============================================================
# PLAN REPAIRER (Same as original)
# ============================================================

class PlanRepairer:
    """Repair failed plans."""
    
    def __init__(self):
        self.repair_strategies = {
            'precondition_failure': self._repair_preconditions_strategy,
            'resource_shortage': self._repair_resources_strategy,
            'timeout': self._repair_timeout_strategy
        }
    
    def repair_preconditions(self, step: PlanStep) -> bool:
        """Try to establish preconditions."""
        for precondition in step.preconditions:
            if not self._is_satisfied(precondition):
                if not self._achieve_condition(precondition):
                    return False
        return True
    
    def create_recovery_plan(self, original_plan: Plan, 
                            failed_step: PlanStep,
                            error: str) -> Optional[Plan]:
        """Create recovery plan from failure."""
        failure_type = self._classify_failure(error)
        
        if failure_type in self.repair_strategies:
            strategy = self.repair_strategies[failure_type]
            return strategy(original_plan, failed_step)
        
        return None
    
    def _is_satisfied(self, condition: str) -> bool:
        """Check if condition is satisfied."""
        return np.random.random() > 0.3
    
    def _achieve_condition(self, condition: str) -> bool:
        """Try to achieve a condition."""
        return np.random.random() > 0.5
    
    def _classify_failure(self, error: str) -> str:
        """Classify failure type from error."""
        error_lower = error.lower()
        
        if 'precondition' in error_lower:
            return 'precondition_failure'
        elif 'resource' in error_lower:
            return 'resource_shortage'
        elif 'timeout' in error_lower or 'time' in error_lower:
            return 'timeout'
        
        return 'unknown'
    
    def _repair_preconditions_strategy(self, plan: Plan, 
                                      failed_step: PlanStep) -> Plan:
        """Repair plan with precondition failure."""
        recovery_plan = Plan(
            plan_id=f"recovery_{plan.plan_id}",
            goal=f"recover_{failed_step.step_id}",
            context=plan.context
        )
        
        for precondition in failed_step.preconditions:
            recovery_step = PlanStep(
                step_id=f"establish_{precondition}",
                action=f"achieve_{precondition}",
                effects=[precondition],
                duration=0.5,
                probability=0.8
            )
            recovery_plan.add_step(recovery_step)
        
        recovery_plan.add_step(failed_step)
        
        return recovery_plan
    
    def _repair_resources_strategy(self, plan: Plan, 
                                  failed_step: PlanStep) -> Plan:
        """Repair plan with resource shortage."""
        recovery_plan = Plan(
            plan_id=f"resource_recovery_{plan.plan_id}",
            goal=f"acquire_resources_{failed_step.step_id}",
            context=plan.context
        )
        
        for resource, amount in failed_step.resources.items():
            acquire_step = PlanStep(
                step_id=f"acquire_{resource}",
                action=f"gather_{resource}",
                effects=[f"has_{resource}>={amount}"],
                duration=1.0,
                probability=0.7
            )
            recovery_plan.add_step(acquire_step)
        
        return recovery_plan
    
    def _repair_timeout_strategy(self, plan: Plan, 
                                failed_step: PlanStep) -> Plan:
        """Repair plan with timeout."""
        recovery_plan = Plan(
            plan_id=f"timeout_recovery_{plan.plan_id}",
            goal=f"expedite_{failed_step.step_id}",
            context=plan.context
        )
        
        fast_step = copy.deepcopy(failed_step)
        fast_step.step_id = f"fast_{failed_step.step_id}"
        fast_step.duration *= 0.5
        fast_step.probability *= 0.9
        
        recovery_plan.add_step(fast_step)
        
        return recovery_plan

# ============================================================
# ENHANCED RESOURCE-AWARE COMPUTE
# ============================================================

class ResourceAwareCompute:
    """Resource-aware computation with real monitoring."""
    
    def __init__(self):
        self.resource_monitor = EnhancedResourceMonitor()
        self.power_manager = PowerManager()
        self.survival_protocol = SurvivalProtocol()
        
        self.optimization_strategies = {
            'pruning': self._apply_pruning,
            'quantization': self._apply_quantization,
            'caching': self._apply_caching,
            'batching': self._apply_batching,
            'offloading': self._apply_offloading,
            'parallelization': self._apply_parallelization,
            'cpu_only': self._apply_cpu_only
        }
        
        self.cache = {}
        self._lock = threading.RLock()
    
    def get_resource_availability(self) -> Dict[str, float]:
        """Get real resource availability."""
        return self.resource_monitor.get_resource_availability()
    
    def plan_with_budget(self, problem: Dict, time_budget_ms: float, 
                        energy_budget_nJ: float) -> Dict[str, Any]:
        """Plan computation within resource constraints."""
        start_time = time.time()
        
        # Check cache
        cache_key = self._compute_cache_key(problem)
        with self._lock:
            if cache_key in self.cache:
                cached = self.cache[cache_key]
                if time.time() - cached['timestamp'] < 60:
                    return cached['result']
        
        # Get real resource availability
        resources = self.get_resource_availability()
        
        # Check operational mode
        self.survival_protocol.assess_and_adapt()
        
        # Check if GPU is available
        use_gpu = self.survival_protocol.is_capability_available('gpu_inference')
        
        # Estimate requirements
        estimated = self._estimate_requirements(problem, use_gpu)
        
        # Select optimization strategy
        strategy = self._select_optimization_strategy(
            problem, estimated, resources, time_budget_ms, energy_budget_nJ
        )
        
        if strategy:
            problem = self.optimization_strategies[strategy](problem)
            estimated = self._estimate_requirements(problem, use_gpu)
        
        # Execute plan
        plan = self._execute_plan(problem, time_budget_ms - (time.time() - start_time) * 1000)
        
        # Track actual usage
        actual_time = (time.time() - start_time) * 1000
        power = self.resource_monitor.current_state.power_watts if self.resource_monitor.current_state else 10
        actual_energy = power * actual_time
        
        plan['resource_usage'] = {
            'time_ms': actual_time,
            'energy_nJ': actual_energy,
            'within_budget': actual_time <= time_budget_ms and actual_energy <= energy_budget_nJ,
            'optimization_used': strategy,
            'operational_mode': self.survival_protocol.current_mode.value
        }
        
        # Cache result
        with self._lock:
            self.cache[cache_key] = {
                'result': plan,
                'timestamp': time.time()
            }
            
            if len(self.cache) > 1000:
                oldest_keys = sorted(self.cache.items(), key=lambda x: x[1]['timestamp'])[:200]
                for key, _ in oldest_keys:
                    del self.cache[key]
        
        return plan
    
    def _compute_cache_key(self, problem: Dict) -> str:
        """Compute cache key for problem."""
        key_str = json.dumps(problem, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _estimate_requirements(self, problem: Dict, use_gpu: bool) -> Dict[str, float]:
        """Estimate resource requirements."""
        complexity = problem.get('complexity', 1.0)
        data_size = problem.get('data_size', 1000)
        
        time_ms = 10 * complexity * np.log(data_size + 1)
        memory_mb = data_size / 1000 + complexity * 100
        energy_nJ = time_ms * 0.1 * complexity
        
        # Adjust for GPU vs CPU
        if use_gpu and problem.get('gpu_accelerated', False):
            time_ms *= 0.3
            energy_nJ *= 2
        
        return {
            'time_ms': time_ms,
            'memory_mb': memory_mb,
            'energy_nJ': energy_nJ
        }
    
    def _select_optimization_strategy(self, problem: Dict, estimated: Dict,
                                     resources: Dict, time_budget: float, 
                                     energy_budget: float) -> Optional[str]:
        """Select optimization strategy based on constraints."""
        # Check if CPU-only mode
        if self.survival_protocol.current_mode in [OperationalMode.SURVIVAL, 
                                                    OperationalMode.EMERGENCY]:
            return 'cpu_only'
        
        time_ratio = estimated['time_ms'] / time_budget
        energy_ratio = estimated['energy_nJ'] / energy_budget
        
        if resources.get('gpu', 0) < 10:
            return 'cpu_only'
        elif problem.get('data_size', 0) > 10000 and time_ratio > 1.5:
            return 'pruning'
        elif energy_ratio > 1.5:
            return 'quantization'
        elif problem.get('complexity', 1) > 2:
            return 'parallelization'
        elif time_budget < 100:
            return 'caching'
        else:
            return 'batching'
    
    def _execute_plan(self, problem: Dict, remaining_budget: float) -> Dict:
        """Execute computation plan."""
        action_type = ActionType.OPTIMIZED_ACTION
        
        if problem.get('goal', {}).get('type') == 'learning':
            action_type = ActionType.EXPLORE
        elif problem.get('goal', {}).get('type') == 'maintenance':
            action_type = ActionType.MAINTAIN
        
        return {
            'action': {
                'type': action_type.value,
                'details': problem,
                'executed_at': time.time()
            },
            'confidence': min(0.9, remaining_budget / 1000)
        }
    
    def _apply_pruning(self, problem: Dict) -> Dict:
        problem = problem.copy()
        current_size = problem.get('data_size', 1000)
        problem['data_size'] = int(current_size * 0.5)
        problem['pruned'] = True
        problem['complexity'] = problem.get('complexity', 1.0) * 0.8
        return problem
    
    def _apply_quantization(self, problem: Dict) -> Dict:
        problem = problem.copy()
        problem['precision'] = 'int8'
        problem['complexity'] = problem.get('complexity', 1.0) * 0.7
        problem['quantized'] = True
        return problem
    
    def _apply_caching(self, problem: Dict) -> Dict:
        problem = problem.copy()
        problem['use_cache'] = True
        problem['complexity'] = problem.get('complexity', 1.0) * 0.3
        return problem
    
    def _apply_batching(self, problem: Dict) -> Dict:
        problem = problem.copy()
        problem['batch_size'] = min(32, problem.get('data_size', 1000) // 10)
        problem['complexity'] = problem.get('complexity', 1.0) * 0.9
        return problem
    
    def _apply_offloading(self, problem: Dict) -> Dict:
        problem = problem.copy()
        problem['offloaded'] = True
        problem['complexity'] = problem.get('complexity', 1.0) * 0.5
        return problem
    
    def _apply_parallelization(self, problem: Dict) -> Dict:
        problem = problem.copy()
        problem['parallel'] = True
        
        # Adapt parallelization to operational mode
        if self.survival_protocol.current_mode == OperationalMode.FULL:
            problem['num_workers'] = 8
        elif self.survival_protocol.current_mode == OperationalMode.BALANCED:
            problem['num_workers'] = 4
        else:
            problem['num_workers'] = 2
        
        problem['complexity'] = problem.get('complexity', 1.0) * 0.6
        return problem
    
    def _apply_cpu_only(self, problem: Dict) -> Dict:
        """Apply CPU-only optimization for systems without GPU."""
        problem = problem.copy()
        problem['gpu_accelerated'] = False
        problem['cpu_optimized'] = True
        problem['use_vectorization'] = True
        problem['complexity'] = problem.get('complexity', 1.0) * 1.2
        return problem

# ============================================================
# RESOURCE ALLOCATOR (Same as original)
# ============================================================

class ResourceAllocator:
    """Allocate resources optimally."""
    
    def allocate(self, requirements: Dict[str, float], 
                available: Dict[str, float]) -> Dict[str, Any]:
        """Allocate resources based on requirements and availability."""
        allocation = {}
        feasible = True
        
        for resource, required in requirements.items():
            if resource in available:
                if required <= available[resource]:
                    allocation[resource] = required
                else:
                    allocation[resource] = available[resource]
                    feasible = False
            else:
                allocation[resource] = required
        
        return {
            'allocation': allocation,
            'feasible': feasible,
            'utilization': {r: allocation.get(r, 0) / available.get(r, 1) 
                          for r in available}
        }

# ============================================================
# DISTRIBUTED COORDINATOR (Same as original with minor fixes)
# ============================================================

class DistributedCoordinator:
    """Enhanced distributed coordination."""
    
    def __init__(self, max_agents: int = 8):
        self.max_agents = max_agents
        self.agents = {}
        self.message_queue = defaultdict(deque)
        self.task_queue = Queue()
        self.results_cache = {}
        self.consensus_protocol = ConsensusProtocol()
        self.shared_knowledge = {}
        self.executor = ThreadPoolExecutor(max_workers=max_agents)
        self.lock = threading.RLock()
        self._shutdown_event = threading.Event()
        self.heartbeat_thread = None
        self._start_heartbeat_monitor()
    
    def _start_heartbeat_monitor(self):
        """Start heartbeat monitoring thread."""
        def monitor():
            while not self._shutdown_event.is_set():
                try:
                    if self._shutdown_event.wait(timeout=5):
                        break
                    self._check_agent_health()
                except:
                    break
        
        self.heartbeat_thread = threading.Thread(target=monitor, daemon=True)
        self.heartbeat_thread.start()
    
    def _check_agent_health(self):
        """Check agent health based on heartbeats."""
        current_time = time.time()
        with self.lock:
            for agent_id, info in list(self.agents.items()):
                if current_time - info['last_heartbeat'] > 30:
                    info['status'] = 'inactive'
                    logger.warning(f"Agent {agent_id} marked inactive")
    
    def register_agent(self, agent_id: str, capabilities: List[str]) -> bool:
        """Register a new agent."""
        with self.lock:
            if len(self.agents) >= self.max_agents:
                return False
            
            self.agents[agent_id] = {
                'capabilities': capabilities,
                'status': 'active',
                'tasks_completed': 0,
                'last_heartbeat': time.time()
            }
            
            logger.info(f"Registered agent {agent_id} with capabilities {capabilities}")
            return True
    
    def distribute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Distribute task to available agents."""
        with self.lock:
            active_agents = [aid for aid, info in self.agents.items() 
                           if info['status'] == 'active']
        
        if not active_agents:
            return {'status': 'no_agents', 'result': None}
        
        subtasks = self._decompose_task(task)
        assignments = self._assign_subtasks(subtasks, active_agents)
        
        futures = []
        for agent_id, subtask in assignments.items():
            future = self.executor.submit(self._execute_subtask, agent_id, subtask)
            futures.append(future)
        
        results = []
        for future in as_completed(futures):
            try:
                result = future.result(timeout=10)
                results.append(result)
            except Exception as e:
                logger.error(f"Subtask execution failed: {e}")
        
        if results:
            consensus_result = self.consensus_protocol.achieve_consensus(
                {f"agent_{i}": r for i, r in enumerate(results)}
            )
            
            return {
                'status': 'distributed',
                'result': consensus_result,
                'assignments': assignments
            }
        
        return {'status': 'failed', 'result': None}
    
    def _decompose_task(self, task: Dict) -> List[Dict]:
        """Decompose task into subtasks."""
        task_type = task.get('type', 'default')
        
        if task_type == 'parallel':
            data = task.get('data', [])
            chunk_size = max(1, len(data) // self.max_agents)
            
            subtasks = []
            for i in range(0, len(data), chunk_size):
                subtask = {
                    'type': 'process_chunk',
                    'data': data[i:i+chunk_size],
                    'params': task.get('params', {})
                }
                subtasks.append(subtask)
            
            return subtasks
        
        return [task]
    
    def _assign_subtasks(self, subtasks: List[Dict], 
                        agents: List[str]) -> Dict[str, Dict]:
        """Assign subtasks to agents."""
        assignments = {}
        
        for i, subtask in enumerate(subtasks):
            agent_id = agents[i % len(agents)]
            assignments[agent_id] = subtask
        
        return assignments
    
    def _execute_subtask(self, agent_id: str, subtask: Dict) -> Any:
        """Execute subtask on agent."""
        time.sleep(np.random.uniform(0.1, 0.5))
        
        with self.lock:
            if agent_id in self.agents:
                self.agents[agent_id]['tasks_completed'] += 1
                self.agents[agent_id]['last_heartbeat'] = time.time()
        
        return {
            'agent_id': agent_id,
            'subtask': subtask,
            'result': f"Processed by {agent_id}",
            'success': np.random.random() > 0.1
        }
    
    def cleanup(self):
        """Cleanup coordinator resources."""
        self._shutdown_event.set()
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            self.heartbeat_thread.join(timeout=1)
        
        with self.lock:
            self.executor.shutdown(wait=False)
            self.agents.clear()
            self.message_queue.clear()
            self.results_cache.clear()
    
    def __del__(self):
        """Destructor."""
        try:
            self.cleanup()
        except:
            pass

# ============================================================
# CONSENSUS PROTOCOL (Same as original)
# ============================================================

class ConsensusProtocol:
    """Byzantine Fault Tolerant consensus protocol."""
    
    def achieve_consensus(self, proposals: Dict[str, Any]) -> Any:
        """Achieve consensus using BFT."""
        if not proposals:
            return None
        
        values = list(proposals.values())
        n = len(values)
        
        if n == 1:
            return values[0]
        
        f = (n - 1) // 3
        threshold = n - f
        
        from collections import Counter
        vote_counts = Counter(str(v) for v in values)
        
        for value_str, count in vote_counts.most_common():
            if count >= threshold:
                for v in values:
                    if str(v) == value_str:
                        return v
        
        return values[0]

# ============================================================
# MAIN TESTING
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("VULCAN-AGI Planning Module - Enhanced Version")
    print("=" * 50)
    
    # Create enhanced planner
    planner = EnhancedHierarchicalPlanner()
    
    # Wait for monitoring to initialize
    time.sleep(2)
    
    # Show current system state
    if planner.resource_monitor.current_state:
        state = planner.resource_monitor.current_state
        print(f"\nSystem State:")
        print(f"  CPU: {state.cpu_percent:.1f}%")
        print(f"  Memory: {state.memory_percent:.1f}%")
        if state.gpu_percent:
            print(f"  GPU: {state.gpu_percent:.1f}%")
        print(f"  Operational Mode: {state.operational_mode.value}")
        print(f"  Network: {state.network_quality}")
    
    # Test planning with different goals
    test_goals = [
        "optimize_system_performance",
        "learn_new_pattern",
        "maintain_stability"
    ]
    
    for goal in test_goals:
        print(f"\n{'='*50}")
        print(f"Planning for: {goal}")
        
        context = {
            'priority': 0.8,
            'time_limit': 100,
            'resources_available': planner.resource_monitor.get_resource_availability()
        }
        
        try:
            # Create plan
            plan = planner.generate_plan(goal, context)
            
            print(f"Plan created: {plan.plan_id}")
            print(f"  Steps: {len(plan.steps)}")
            print(f"  Expected duration: {plan.expected_duration:.2f}s")
            print(f"  Success probability: {plan.success_probability:.2%}")
            print(f"  Mode: {planner.survival_protocol.current_mode.value}")
            
            # Show steps
            for step in plan.steps[:3]:
                print(f"    - {step.action} (p={step.probability:.2f})")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    # Test resource-aware compute
    print(f"\n{'='*50}")
    print("Testing Resource-Aware Compute")
    
    compute = ResourceAwareCompute()
    
    problem = {
        'goal': {'type': 'optimization'},
        'data_size': 5000,
        'complexity': 2.0,
        'gpu_accelerated': True
    }
    
    result = compute.plan_with_budget(problem, time_budget_ms=100, energy_budget_nJ=10000)
    
    print(f"Computation result:")
    print(f"  Action: {result['action']['type']}")
    print(f"  Time used: {result['resource_usage']['time_ms']:.2f}ms")
    print(f"  Within budget: {result['resource_usage']['within_budget']}")
    print(f"  Mode: {result['resource_usage']['operational_mode']}")
    
    # Cleanup
    print(f"\n{'='*50}")
    print("Cleaning up...")
    planner.cleanup()
    compute.resource_monitor.cleanup()
    
    print("Test completed successfully!")
