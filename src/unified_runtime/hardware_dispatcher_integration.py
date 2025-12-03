"""
Hardware Dispatcher Integration Module for Graphix IR
Handles hardware dispatch, emulation, and optimization for heterogeneous compute
"""

import asyncio
import json
import time
import random
import hashlib
import os
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict
import logging
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# Optional imports with graceful fallback
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    from analog_photonic_emulator import analog_photonic_emulator
    EMULATOR_AVAILABLE = True
except ImportError:
    analog_photonic_emulator = None
    EMULATOR_AVAILABLE = False

try:
    from hardware_dispatcher import HardwareDispatcher
    HARDWARE_DISPATCHER_AVAILABLE = True
except ImportError:
    HardwareDispatcher = None
    HARDWARE_DISPATCHER_AVAILABLE = False

try:
    import llm_compressor
    COMPRESSOR_AVAILABLE = True
except ImportError:
    llm_compressor = None
    COMPRESSOR_AVAILABLE = False

logger = logging.getLogger(__name__)


class HardwareBackend(Enum):
    """Available hardware backends"""
    CPU = "cpu"
    GPU = "gpu"
    PHOTONIC = "photonic"
    MEMRISTOR = "memristor"
    FPGA = "fpga"
    QUANTUM = "quantum"
    EMULATOR = "emulator"


class DispatchStrategy(Enum):
    """Hardware dispatch strategies"""
    FASTEST = "fastest"
    MOST_EFFICIENT = "most_efficient"
    LOWEST_ENERGY = "lowest_energy"
    BEST_ACCURACY = "best_accuracy"
    BALANCED = "balanced"


@dataclass
class HardwareProfile:
    """Profile for a hardware backend"""
    backend: HardwareBackend
    max_tensor_size_mb: float
    latency_ms: float
    energy_per_op_nj: float
    accuracy: float
    throughput_tops: float
    available: bool = True
    health_score: float = 1.0
    temperature_c: Optional[float] = None
    utilization_percent: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DispatchResult:
    """Result from hardware dispatch"""
    backend: HardwareBackend
    result: Any
    latency_ms: float
    energy_nj: Optional[float] = None
    accuracy_score: Optional[float] = None
    fallback_used: bool = False
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class HardwareProfileManager:
    """Manages hardware profiles and capabilities"""
    
    def __init__(self, profiles_path: Optional[str] = None):
        self.profiles: Dict[HardwareBackend, HardwareProfile] = {}
        self._lock = threading.Lock()
        
        # Load profiles from file or use defaults
        if profiles_path and os.path.exists(profiles_path):
            self._load_profiles_from_file(profiles_path)
        else:
            self._load_default_profiles()
    
    def _load_profiles_from_file(self, path: str):
        """Load hardware profiles from JSON file"""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                
            for backend_name, profile_data in data.items():
                try:
                    backend = HardwareBackend(backend_name.lower())
                    self.profiles[backend] = HardwareProfile(
                        backend=backend,
                        max_tensor_size_mb=profile_data.get('max_tensor_size_mb', 1024),
                        latency_ms=profile_data.get('latency_ms', 1.0),
                        energy_per_op_nj=profile_data.get('energy_per_op_nj', 1.0),
                        accuracy=profile_data.get('accuracy', 0.99),
                        throughput_tops=profile_data.get('throughput_tops', 1.0),
                        available=profile_data.get('available', True)
                    )
                except Exception as e:
                    logger.warning(f"Failed to load profile for {backend_name}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to load hardware profiles from {path}: {e}")
            self._load_default_profiles()
    
    def _load_default_profiles(self):
        """Load default hardware profiles"""
        self.profiles = {
            HardwareBackend.CPU: HardwareProfile(
                backend=HardwareBackend.CPU,
                max_tensor_size_mb=8192,
                latency_ms=10.0,
                energy_per_op_nj=10.0,
                accuracy=1.0,
                throughput_tops=0.1,
                available=True
            ),
            HardwareBackend.GPU: HardwareProfile(
                backend=HardwareBackend.GPU,
                max_tensor_size_mb=16384,
                latency_ms=1.0,
                energy_per_op_nj=5.0,
                accuracy=0.9999,
                throughput_tops=10.0,
                available=TORCH_AVAILABLE
            ),
            HardwareBackend.PHOTONIC: HardwareProfile(
                backend=HardwareBackend.PHOTONIC,
                max_tensor_size_mb=128,
                latency_ms=0.1,
                energy_per_op_nj=0.01,
                accuracy=0.98,
                throughput_tops=100.0,
                available=False  # Requires real hardware
            ),
            HardwareBackend.MEMRISTOR: HardwareProfile(
                backend=HardwareBackend.MEMRISTOR,
                max_tensor_size_mb=256,
                latency_ms=0.5,
                energy_per_op_nj=0.1,
                accuracy=0.97,
                throughput_tops=50.0,
                available=False  # Requires real hardware
            ),
            HardwareBackend.FPGA: HardwareProfile(
                backend=HardwareBackend.FPGA,
                max_tensor_size_mb=512,
                latency_ms=2.0,
                energy_per_op_nj=1.0,
                accuracy=0.999,
                throughput_tops=5.0,
                available=False
            ),
            HardwareBackend.EMULATOR: HardwareProfile(
                backend=HardwareBackend.EMULATOR,
                max_tensor_size_mb=1024,
                latency_ms=5.0,
                energy_per_op_nj=2.0,
                accuracy=0.95,
                throughput_tops=1.0,
                available=EMULATOR_AVAILABLE
            )
        }
    
    def get_profile(self, backend: HardwareBackend) -> Optional[HardwareProfile]:
        """Get profile for a backend"""
        with self._lock:
            return self.profiles.get(backend)
    
    def update_health(self, backend: HardwareBackend, health_score: float,
                     temperature_c: Optional[float] = None,
                     utilization_percent: Optional[float] = None):
        """Update health metrics for a backend"""
        with self._lock:
            if backend in self.profiles:
                profile = self.profiles[backend]
                profile.health_score = health_score
                if temperature_c is not None:
                    profile.temperature_c = temperature_c
                if utilization_percent is not None:
                    profile.utilization_percent = utilization_percent
    
    def get_available_backends(self, min_health: float = 0.5) -> List[HardwareBackend]:
        """Get list of available and healthy backends"""
        with self._lock:
            return [
                backend for backend, profile in self.profiles.items()
                if profile.available and profile.health_score >= min_health
            ]
    
    def select_backend(self, tensor_size_mb: float, strategy: DispatchStrategy) -> Optional[HardwareBackend]:
        """Select best backend based on strategy"""
        available = self.get_available_backends()
        if not available:
            return None
        
        # Filter by tensor size capability
        capable = []
        for backend in available:
            profile = self.profiles[backend]
            if profile.max_tensor_size_mb >= tensor_size_mb:
                capable.append(backend)
        
        if not capable:
            return None
        
        # Select based on strategy
        if strategy == DispatchStrategy.FASTEST:
            return min(capable, key=lambda b: self.profiles[b].latency_ms)
        elif strategy == DispatchStrategy.LOWEST_ENERGY:
            return min(capable, key=lambda b: self.profiles[b].energy_per_op_nj)
        elif strategy == DispatchStrategy.BEST_ACCURACY:
            return max(capable, key=lambda b: self.profiles[b].accuracy)
        elif strategy == DispatchStrategy.MOST_EFFICIENT:
            return max(capable, key=lambda b: self.profiles[b].throughput_tops)
        else:  # BALANCED
            # Score each backend
            scores = {}
            for backend in capable:
                profile = self.profiles[backend]
                # Normalize metrics (lower is better for latency/energy)
                latency_score = 1.0 / (1.0 + profile.latency_ms)
                energy_score = 1.0 / (1.0 + profile.energy_per_op_nj)
                accuracy_score = profile.accuracy
                throughput_score = profile.throughput_tops / 100.0  # Normalize to 0-1
                
                # Weighted average
                scores[backend] = (
                    0.25 * latency_score +
                    0.25 * energy_score +
                    0.25 * accuracy_score +
                    0.25 * throughput_score
                ) * profile.health_score
            
            return max(scores.keys(), key=lambda k: scores[k])


class HardwareDispatcherIntegration:
    """
    Integration layer for hardware dispatch with emulation fallback
    """
    
    def __init__(self, 
                 enable_hardware: bool = True,
                 enable_emulator: bool = True,
                 profiles_path: Optional[str] = None,
                 cache_size: int = 100,
                 strategy: DispatchStrategy = DispatchStrategy.BALANCED): # Added
        
        self.enable_hardware = enable_hardware and HARDWARE_DISPATCHER_AVAILABLE
        self.enable_emulator = enable_emulator and EMULATOR_AVAILABLE
        self.profile_manager = HardwareProfileManager(profiles_path)
        self.strategy = strategy # Added
        self._lock = threading.Lock() # Added
        
        # Initialize hardware dispatcher if available
        if self.enable_hardware:
            try:
                self.hardware_dispatcher = HardwareDispatcher(
                    enable_metrics=True,
                    enable_health_checks=True,
                    prefer_grpc=False
                )
                logger.info("Hardware dispatcher initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize hardware dispatcher: {e}")
                self.hardware_dispatcher = None
                self.enable_hardware = False
        else:
            self.hardware_dispatcher = None
        
        # Execution cache
        self.cache: Dict[str, DispatchResult] = {}
        self.cache_size = cache_size
        
        # Metrics
        self.dispatch_count = defaultdict(int)
        self.fallback_count = defaultdict(int)
        self.total_energy_nj = 0.0
        self.total_latency_ms = 0.0
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info(f"HardwareDispatcherIntegration initialized. Hardware: {self.enable_hardware}, Emulator: {self.enable_emulator}")
    
    async def dispatch_to_hardware(self, operation: str, *args, **kwargs) -> DispatchResult:
        """
        Main hardware dispatch function with intelligent routing
        """
        start_time = time.perf_counter()
        
        # Check cache
        cache_key = self._compute_cache_key(operation, args, kwargs)
        if cache_key in self.cache:
            logger.debug(f"Cache hit for operation {operation}")
            return self.cache[cache_key]
        
        # Determine backend based on operation and data
        backend = self._select_backend_for_operation(operation, args, kwargs)
        
        # Track dispatch
        self.dispatch_count[backend] += 1
        
        # Try hardware dispatch first
        if self.enable_hardware and self.hardware_dispatcher and backend != HardwareBackend.EMULATOR:
            try:
                result = await self._dispatch_to_real_hardware(operation, backend, *args, **kwargs)
                if not result.error:
                    self._update_cache(cache_key, result)
                    self._update_metrics(result)
                    return result
                else:
                    logger.warning(f"Hardware dispatch failed: {result.error}")
            except Exception as e:
                logger.error(f"Hardware dispatch exception: {e}")
                self.fallback_count[backend] += 1
        
        # Fallback to emulator
        if self.enable_emulator:
            logger.info(f"Falling back to emulator for {operation}")
            result = await self.dispatch_to_emulator_fallback(operation, *args, **kwargs)
            self._update_cache(cache_key, result)
            self._update_metrics(result)
            return result
        
        # Final fallback to CPU
        logger.warning(f"All dispatch methods failed, using CPU fallback for {operation}")
        result = await self._cpu_fallback(operation, *args, **kwargs)
        self._update_metrics(result)
        return result
    
    async def _dispatch_to_real_hardware(self, operation: str, backend: HardwareBackend, 
                                        *args, **kwargs) -> DispatchResult:
        """Dispatch to real hardware backend"""
        start_time = time.perf_counter()
        
        try:
            # Map backend to hardware dispatcher method
            if backend == HardwareBackend.PHOTONIC:
                hw_result = await asyncio.to_thread(
                    self.hardware_dispatcher.dispatch,
                    "photonic",
                    operation,
                    *args,
                    **kwargs
                )
            elif backend == HardwareBackend.MEMRISTOR:
                hw_result = await asyncio.to_thread(
                    self.hardware_dispatcher.dispatch,
                    "memristor",
                    operation,
                    *args,
                    **kwargs
                )
            elif backend == HardwareBackend.FPGA:
                hw_result = await asyncio.to_thread(
                    self.hardware_dispatcher.dispatch,
                    "fpga",
                    operation,
                    *args,
                    **kwargs
                )
            else:
                # Default CPU/GPU dispatch
                hw_result = await asyncio.to_thread(
                    self.hardware_dispatcher.dispatch,
                    backend.value,
                    operation,
                    *args,
                    **kwargs
                )
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            # Check for hardware errors
            if isinstance(hw_result, dict) and 'error_code' in hw_result:
                return DispatchResult(
                    backend=backend,
                    result=None,
                    latency_ms=latency_ms,
                    error=hw_result.get('message', 'Unknown hardware error'),
                    fallback_used=True
                )
            
            # Get energy estimate from profile
            profile = self.profile_manager.get_profile(backend)
            energy_nj = None
            if profile:
                # Estimate energy based on operation complexity
                ops_count = self._estimate_operation_count(operation, args)
                energy_nj = profile.energy_per_op_nj * ops_count
            
            return DispatchResult(
                backend=backend,
                result=hw_result,
                latency_ms=latency_ms,
                energy_nj=energy_nj,
                accuracy_score=profile.accuracy if profile else None,
                fallback_used=False,
                metadata={'hardware': 'real'}
            )
            
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return DispatchResult(
                backend=backend,
                result=None,
                latency_ms=latency_ms,
                error=str(e),
                fallback_used=True
            )
    
    async def dispatch_to_emulator_fallback(self, operation: str, *args, **kwargs) -> DispatchResult:
        """
        Fallback to emulator when hardware is unavailable
        """
        start_time = time.perf_counter()
        
        try:
            if operation == "photonic_mvm" and len(args) >= 2:
                result = await self._emulate_photonic_mvm(args[0], args[1], kwargs.get('params', {}))
                backend = HardwareBackend.PHOTONIC
            elif operation == "memristor_mvm" and len(args) >= 2:
                result = await self._emulate_memristor_mvm(args[0], args[1])
                backend = HardwareBackend.MEMRISTOR
            elif operation == "photonic_fused":
                subgraph = args[0] if args else kwargs.get('subgraph')
                result = await self.dispatch_to_emulator(subgraph, backend="photonic", **kwargs)
                backend = HardwareBackend.PHOTONIC
            else:
                # Generic CPU emulation
                result = await self._cpu_fallback(operation, *args, **kwargs)
                backend = HardwareBackend.CPU
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            # Get energy estimate
            profile = self.profile_manager.get_profile(backend)
            energy_nj = None
            if profile:
                ops_count = self._estimate_operation_count(operation, args)
                energy_nj = profile.energy_per_op_nj * ops_count * 1.5  # Emulation overhead
            
            return DispatchResult(
                backend=backend,
                result=result,
                latency_ms=latency_ms,
                energy_nj=energy_nj,
                accuracy_score=0.95,  # Emulation accuracy
                fallback_used=True,
                metadata={'hardware': 'emulated'}
            )
            
        except Exception as e:
            logger.error(f"Emulator fallback failed: {e}")
            latency_ms = (time.perf_counter() - start_time) * 1000
            return DispatchResult(
                backend=HardwareBackend.EMULATOR,
                result=None,
                latency_ms=latency_ms,
                error=str(e),
                fallback_used=True
            )
    
    async def _emulate_photonic_mvm(self, matrix: Any, vector: Any, params: Dict[str, Any]) -> Any:
        """Emulate photonic matrix-vector multiplication"""
        if EMULATOR_AVAILABLE and analog_photonic_emulator:
            # Apply compression if specified
            if params.get('compression') == 'ITU-F.748-quantized' and COMPRESSOR_AVAILABLE:
                logger.debug("Applying ITU-F.748 compression")
                matrix = llm_compressor.quantize_tensor(matrix, config={"precision": "8bit"})
            
            # Ensure vector is 2D for emulator compatibility
            if hasattr(vector, 'ndim') and vector.ndim == 1:
                vector = vector.reshape(-1, 1)
            elif isinstance(vector, list) and vector and not isinstance(vector[0], list):
                vector = [[v] for v in vector]
            
            # Add photonic noise
            result = analog_photonic_emulator.emulate_photonic_mvm(matrix, vector)
            
            # Apply additional noise based on params
            noise_std = params.get('noise_std', 0.01)
            if noise_std > 0 and isinstance(result, np.ndarray):
                noise = np.random.normal(0, noise_std, result.shape)
                result = result + noise
            
            return result
        else:
            # Pure numpy/python fallback
            return self._emulate_mvm_pure(matrix, vector, noise_std=params.get('noise_std', 0.01))
    
    async def _emulate_memristor_mvm(self, tensor1: Any, tensor2: Any) -> Any:
        """
        Emulate memristor matrix-vector multiplication with CIM characteristics
        """
        if EMULATOR_AVAILABLE and analog_photonic_emulator:
            # Use analog emulator
            if TORCH_AVAILABLE:
                op_tensor = torch.tensor(tensor2, dtype=torch.float32)
                result = analog_photonic_emulator.emulate_memristor_cim(
                    tensor1, 
                    op=lambda t: t @ op_tensor
                )
                return result.cpu().numpy() if hasattr(result, "cpu") else result
            else:
                # Numpy-based emulation
                return analog_photonic_emulator.emulate_memristor_cim(tensor1, op=lambda t: np.dot(t, tensor2))
        else:
            # Pure fallback
            return self._emulate_mvm_pure(tensor1, tensor2, noise_std=0.02, scaling_factor=0.98)
    
    def _emulate_mvm_pure(self, tensor1: Any, tensor2: Any, 
                         noise_std: float = 0.01, 
                         scaling_factor: float = 1.0) -> Any:
        """
        Pure Python/NumPy fallback for MVM emulation
        """
        # Track if inputs were pure Python lists
        input_was_list = isinstance(tensor1, list) and isinstance(tensor2, list)
        
        if np:
            # NumPy implementation
            if not isinstance(tensor1, np.ndarray):
                tensor1 = np.array(tensor1)
            if not isinstance(tensor2, np.ndarray):
                tensor2 = np.array(tensor2)
            
            # Validate non-empty arrays
            if tensor1.size == 0 or tensor2.size == 0:
                raise ValueError("Cannot perform matrix multiplication on empty tensors")
            
            result = np.dot(tensor1 * scaling_factor, tensor2)
            
            # Add noise
            if noise_std > 0:
                noise = np.random.normal(0, noise_std, result.shape)
                result = result + noise
            
            # Convert back to list if inputs were lists
            if input_was_list:
                return result.tolist()
            
            return result
        else:
            # Pure Python fallback
            logger.warning("NumPy not available. Using pure Python MVM (slow)")
            
            if not isinstance(tensor1, list):
                tensor1 = [[tensor1]]
            if not isinstance(tensor2, list):
                tensor2 = [tensor2]
            
            # Check for empty tensors
            if not tensor1 or not tensor1[0]:
                raise ValueError("tensor1 is empty or has empty rows")
            
            if not tensor2:
                raise ValueError("tensor2 is empty")
            
            M = len(tensor1)
            N = len(tensor1[0]) if isinstance(tensor1[0], list) else 1
            
            # Handle vector
            if not isinstance(tensor2[0], list):
                tensor2 = [[x] for x in tensor2]
            
            P = len(tensor2[0]) if tensor2 and tensor2[0] else 0
            
            if N != len(tensor2):
                raise ValueError(f"Matrix dimensions incompatible: {N} vs {len(tensor2)}")
            
            # Matrix multiplication
            result = []
            for i in range(M):
                row = []
                for j in range(P):
                    val = 0.0
                    for k in range(N):
                        a = tensor1[i][k] if isinstance(tensor1[i], list) else tensor1[i]
                        b = tensor2[k][j] if isinstance(tensor2[k], list) else tensor2[k]
                        val += a * b * scaling_factor
                    
                    # Add noise
                    if noise_std > 0:
                        val += random.gauss(0, noise_std)
                    
                    row.append(val)
                result.append(row[0] if P == 1 else row)
            
            return result
    
    async def dispatch_to_emulator(self, subgraph: Dict, backend: str = "photonic", **kwargs) -> Any:
        """
        General emulator dispatch for subgraph execution
        """
        if not EMULATOR_AVAILABLE or not analog_photonic_emulator:
            logger.warning("Emulator not available, using CPU fallback")
            return await self._execute_subgraph_cpu(subgraph, **kwargs)
        
        # Extract tensors from subgraph
        node_map = {n['id']: n for n in subgraph.get("nodes", [])}
        inputs = {}
        
        for n in subgraph.get("nodes", []):
            if n.get("type", "").startswith("LOAD_TENSOR") or n.get("type") == "CONST":
                param = n.get("params", {})
                # Fix: Use explicit None check instead of 'or' operator
                value = param.get("value")
                if value is None:
                    value = param.get("tensor")
                if value is not None:
                    inputs[n['id']] = value
        
        if len(inputs) < 2:
            raise ValueError(f"Emulator dispatch requires at least 2 tensors, got {len(inputs)}")
        
        # Get tensors
        tensor_keys = sorted(inputs.keys())
        tensor1 = inputs[tensor_keys[0]]
        tensor2 = inputs[tensor_keys[1]]
        
        # Ensure tensor2 is 2D for photonic emulator
        if backend == "photonic":
            if hasattr(tensor2, 'ndim') and tensor2.ndim == 1:
                tensor2 = tensor2.reshape(-1, 1)
            elif isinstance(tensor2, list) and tensor2 and not isinstance(vector[0], list):
                tensor2 = [[v] for v in tensor2]
        
        logger.info(f"Dispatching to {backend} emulator: shapes {getattr(tensor1, 'shape', 'N/A')} x {getattr(tensor2, 'shape', 'N/A')}")
        
        # Backend-specific emulation
        if backend == "photonic":
            # In-situ training if available
            if hasattr(analog_photonic_emulator, "train_in_situ") and random.random() > 0.8:
                logger.info("Triggering in-situ training for photonic emulation")
                
                # Generate or get target
                target = kwargs.get('target')
                if target is None:
                    if TORCH_AVAILABLE:
                        target = torch.randn(tensor1.shape[0], tensor2.shape[-1] if hasattr(tensor2, 'shape') else 1)
                    elif np:
                        target = np.random.randn(tensor1.shape[0], tensor2.shape[-1] if hasattr(tensor2, 'shape') else 1)
                    else:
                        target = [[random.gauss(0, 1)] for _ in range(len(tensor1))]
                
                analog_photonic_emulator.train_in_situ(tensor1, target)
            
            # Execute photonic MVM
            result = analog_photonic_emulator.emulate_photonic_mvm(tensor1, tensor2)
            
        elif backend == "memristor":
            # Memristor CIM emulation
            if TORCH_AVAILABLE:
                op_tensor = torch.tensor(tensor2, dtype=torch.float32)
                result = analog_photonic_emulator.emulate_memristor_cim(
                    tensor1,
                    op=lambda t: t @ op_tensor
                )
            else:
                result = analog_photonic_emulator.emulate_memristor_cim(
                    tensor1,
                    op=lambda t: np.dot(t, tensor2) if np else self._emulate_mvm_pure(t, tensor2)
                )
        else:
            raise ValueError(f"Unknown emulator backend: {backend}")
        
        # Convert result to appropriate format
        if hasattr(result, "cpu"):
            result = result.cpu().numpy()
        
        return {"product": result, "backend": backend}
    
    async def _execute_subgraph_cpu(self, subgraph: Dict, **kwargs) -> Dict:
        """CPU execution of subgraph"""
        # Simple CPU execution - just extract and multiply tensors
        tensors = []
        
        for n in subgraph.get("nodes", []):
            if n.get("type") in ["CONST", "LOAD_TENSOR"]:
                value = n.get("params", {}).get("value")
                if value is not None:
                    tensors.append(value)
        
        if len(tensors) >= 2:
            result = self._emulate_mvm_pure(tensors[0], tensors[1])
            return {"product": result, "backend": "cpu"}
        
        return {"error": "Insufficient tensors for computation"}
    
    async def optimize_and_dispatch(self, subgraph: Dict, metrics: Dict[str, float]) -> DispatchResult:
        """
        Dynamically choose backend and dispatch based on subgraph characteristics
        """
        start_time = time.perf_counter()
        
        # Analyze subgraph
        tensor_size_mb = self._estimate_tensor_size(subgraph)
        operation_type = self._identify_operation_type(subgraph)
        
        # Determine dispatch strategy based on metrics
        strategy = self._select_strategy(metrics)
        
        # Select backend
        backend = self.profile_manager.select_backend(tensor_size_mb, strategy)
        
        if backend is None:
            logger.warning("No suitable backend found, using CPU")
            backend = HardwareBackend.CPU
        
        logger.info(f"Dynamic dispatch chose backend='{backend.value}' for {tensor_size_mb:.2f}MB tensor, strategy={strategy.value}")
        
        # Dispatch to selected backend
        try:
            if backend in [HardwareBackend.PHOTONIC, HardwareBackend.MEMRISTOR]:
                # Use emulator for analog backends
                result = await self.dispatch_to_emulator(subgraph, backend=backend.value)
                latency_ms = (time.perf_counter() - start_time) * 1000
                return DispatchResult(
                    backend=backend,
                    result=result,
                    latency_ms=latency_ms,
                    fallback_used=backend == HardwareBackend.EMULATOR
                )
            else:
                # CPU/GPU execution
                result = await self._execute_subgraph_cpu(subgraph)
                latency_ms = (time.perf_counter() - start_time) * 1000
                return DispatchResult(
                    backend=backend,
                    result=result,
                    latency_ms=latency_ms,
                    fallback_used=False
                )
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"optimize_and_dispatch failed: {e}")
            return DispatchResult(
                backend=backend,
                result=None,
                latency_ms=latency_ms,
                error=str(e),
                fallback_used=True
            )
    
    def _estimate_tensor_size(self, subgraph: Dict) -> float:
        """Estimate total tensor size in MB"""
        total_size = 0
        
        for n in subgraph.get("nodes", []):
            if "params" in n and ("tensor" in n["params"] or "value" in n["params"]):
                # Fix: Use explicit None check
                t = n["params"].get("tensor")
                if t is None:
                    t = n["params"].get("value")
                
                if t is None:
                    continue
                
                if hasattr(t, "shape"):
                    # NumPy/Torch tensor
                    elements = 1
                    for dim in t.shape:
                        elements *= dim
                    total_size += elements * 4 / 1e6  # Assume float32
                elif isinstance(t, list):
                    # Estimate for lists
                    def count_elements(lst):
                        if not isinstance(lst, list):
                            return 1
                        return sum(count_elements(item) for item in lst)
                    
                    elements = count_elements(t)
                    total_size += elements * 4 / 1e6
        
        return total_size
    
    def _identify_operation_type(self, subgraph: Dict) -> str:
        """Identify the primary operation type in subgraph"""
        node_types = set()
        for n in subgraph.get("nodes", []):
            node_types.add(n.get("type", ""))
        
        if "PhotonicMVMNode" in node_types or "PHOTONIC" in str(node_types):
            return "photonic_mvm"
        elif "MEMRISTOR" in str(node_types):
            return "memristor_mvm"
        elif "CONV" in str(node_types):
            return "convolution"
        elif "ATTENTION" in str(node_types):
            return "attention"
        else:
            return "general"
    
    def _select_backend_for_operation(self, operation: str, args: tuple, kwargs: dict) -> HardwareBackend:
        """Select backend for operation"""
        # Simple heuristic-based selection
        if operation in ["photonic_mvm", "photonic_fused"]:
            return HardwareBackend.PHOTONIC
        elif operation == "memristor_mvm":
            return HardwareBackend.MEMRISTOR
        else:
            return HardwareBackend.CPU
    
    def _select_strategy(self, metrics: Dict[str, float]) -> DispatchStrategy:
        """Select dispatch strategy based on metrics"""
        if not metrics:
            return self.strategy # Use class default
        
        # Analyze metrics to determine priority
        if metrics.get("latency_critical", False):
            return DispatchStrategy.FASTEST
        elif metrics.get("energy_constrained", False):
            return DispatchStrategy.LOWEST_ENERGY
        elif metrics.get("accuracy_required", 0) > 0.99:
            return DispatchStrategy.BEST_ACCURACY
        elif metrics.get("throughput_required", False):
            return DispatchStrategy.MOST_EFFICIENT
        else:
            return self.strategy # Use class default

    
    def _estimate_operation_count(self, operation: str, args: tuple) -> int:
        """Estimate number of operations for energy calculation"""
        if operation in ["photonic_mvm", "memristor_mvm"]:
            # Matrix-vector multiplication
            if len(args) >= 2:
                try:
                    if hasattr(args[0], "shape") and hasattr(args[1], "shape"):
                        return args[0].shape[0] * args[0].shape[1]
                    elif isinstance(args[0], list) and isinstance(args[1], list):
                        return len(args[0]) * len(args[0][0]) if args[0] else 0
                except Exception as e:
                    logger.error(f"Operation failed: {e}")
        elif operation == "photonic_fused":
            # Estimate from subgraph
            if args and isinstance(args[0], dict):
                return len(args[0].get("nodes", [])) * 100
        
        return 1000  # Default estimate
    
    def _compute_cache_key(self, operation: str, args: tuple, kwargs: dict) -> str:
        """Compute cache key for operation"""
        # Create hashable representation
        key_parts = [operation]
        
        for arg in args:
            if hasattr(arg, "shape"):
                key_parts.append(str(arg.shape))
            elif isinstance(arg, (list, dict)):
                key_parts.append(str(len(arg)))
            else:
                key_parts.append(str(type(arg)))
        
        for k, v in sorted(kwargs.items()):
            if k != "params":  # Skip variable params
                key_parts.append(f"{k}={v}")
        
        return hashlib.md5("|".join(key_parts).encode()).hexdigest()
    
    def _update_cache(self, key: str, result: DispatchResult):
        """Update result cache with LRU eviction"""
        if len(self.cache) >= self.cache_size:
            # Simple FIFO eviction (could be improved to LRU)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = result
    
    def _update_metrics(self, result: DispatchResult):
        """Update internal metrics"""
        if result.latency_ms:
            self.total_latency_ms += result.latency_ms
        if result.energy_nj:
            self.total_energy_nj += result.energy_nj
    
    async def _cpu_fallback(self, operation: str, *args, **kwargs) -> DispatchResult:
        """Final CPU fallback for any operation"""
        start_time = time.perf_counter()
        
        try:
            if operation in ["photonic_mvm", "memristor_mvm"] and len(args) >= 2:
                result = self._emulate_mvm_pure(args[0], args[1])
            elif operation == "photonic_fused":
                subgraph = args[0] if args else kwargs.get('subgraph')
                result = await self._execute_subgraph_cpu(subgraph)
            else:
                result = {"error": f"Unknown operation: {operation}"}
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            return DispatchResult(
                backend=HardwareBackend.CPU,
                result=result,
                latency_ms=latency_ms,
                energy_nj=latency_ms * 10,  # Rough estimate
                accuracy_score=1.0,
                fallback_used=True
            )
            
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return DispatchResult(
                backend=HardwareBackend.CPU,
                result=None,
                latency_ms=latency_ms,
                error=str(e),
                fallback_used=True
            )
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of dispatch metrics"""
        return {
            "dispatch_counts": dict(self.dispatch_count),
            "fallback_counts": dict(self.fallback_count),
            "total_energy_nj": self.total_energy_nj,
            "total_latency_ms": self.total_latency_ms,
            "cache_size": len(self.cache),
            "available_backends": [b.value for b in self.profile_manager.get_available_backends()],
            "backend_profiles": {
                backend.value: profile.to_dict()
                for backend, profile in self.profile_manager.profiles.items()
            }
        }

    # ========================================================================
    # ADDED METHODS PER REQUEST
    # ========================================================================

    def _estimate_tensor_size_mb(self, tensor: Any) -> float:
        """Estimate tensor size in MB"""
        try:
            if TORCH_AVAILABLE and isinstance(tensor, torch.Tensor):
                return tensor.element_size() * tensor.nelement() / (1024 * 1024)
            if np and isinstance(tensor, np.ndarray):
                return tensor.nbytes / (1024 * 1024)
            
            # Fallback for lists or other objects: serialize and measure
            # This is an expensive proxy, but safer than deep recursion
            json_str = json.dumps(tensor, default=str)
            return len(json_str.encode('utf-8')) / (1024 * 1024)
            
        except Exception as e:
            logger.warning(f"Could not estimate tensor size for type {type(tensor)}: {e}")
        return 1.0 # Default 1MB

    async def run_tensor_op(self, op: Callable[[], Any], estimated_tensor_mb: Optional[float] = None) -> DispatchResult:
        """
        Pick backend, maybe compress, then execute op on that backend.
        op should be a closure that already knows what to run (matmul, forward(), etc).
        """
        start = time.time()
        backend_choice = None
        try:
            # pick backend
            tensor_mb = estimated_tensor_mb if estimated_tensor_mb is not None else 1.0
            backend_choice = self.profile_manager.select_backend(tensor_mb, self.strategy)
            if backend_choice is None:
                backend_choice = HardwareBackend.CPU

            # optionally compress model/weights if COMPRESSOR_AVAILABLE and backend is constrained
            # (safe no-op if not available)
            # Example placeholder:
            # if COMPRESSOR_AVAILABLE and backend_choice == HardwareBackend.PHOTONIC:
            #    op = self._get_compressed_op(op) 

            # actually run it in a thread so we don't block asyncio
            result_val = await asyncio.get_running_loop().run_in_executor(
                self.executor, op # Use existing self.executor
            )

            latency_ms = (time.time() - start) * 1000.0
            prof = self.profile_manager.get_profile(backend_choice)

            return DispatchResult(
                backend=backend_choice,
                result=result_val,
                latency_ms=latency_ms,
                energy_nj=(prof.energy_per_op_nj if prof else None),
                accuracy_score=(prof.accuracy if prof else None),
                fallback_used=False,
                metadata={"backend_profile": prof.to_dict() if prof else {}}
            )
        except Exception as e:
            latency_ms = (time.time() - start) * 1000.0
            return DispatchResult(
                backend=backend_choice or HardwareBackend.CPU,
                result=None,
                latency_ms=latency_ms,
                fallback_used=True,
                error=str(e),
                metadata={"traceback": traceback.format_exc()}
            )
    
    def get_health_snapshot(self) -> Dict[str, Any]:
        """Expose health info back to runtime."""
        try:
            return {
                b.value: self.profile_manager.get_profile(b).to_dict()
                for b in self.profile_manager.get_available_backends()
            }
        except Exception as e:
            logger.error(f"Failed to get health snapshot: {e}")
            return {"error": str(e)}

    # ========================================================================
    
    def cleanup(self):
        """Cleanup resources"""
        # Shutdown hardware dispatcher if it exists
        if self.hardware_dispatcher is not None:
            try:
                self.hardware_dispatcher.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down hardware dispatcher: {e}")
        
        self.executor.shutdown(wait=False)
        self.cache.clear()


# ============================================================================
# MODULE-LEVEL FUNCTIONS FOR BACKWARD COMPATIBILITY
# ============================================================================

# Global instance for module-level functions
_global_dispatcher: Optional[HardwareDispatcherIntegration] = None

def _get_global_dispatcher() -> HardwareDispatcherIntegration:
    """Get or create global dispatcher instance"""
    global _global_dispatcher
    if _global_dispatcher is None:
        _global_dispatcher = HardwareDispatcherIntegration()
    return _global_dispatcher


async def dispatch_to_hardware(operation: str, *args, **kwargs) -> Any:
    """Module-level hardware dispatch function"""
    dispatcher = _get_global_dispatcher()
    result = await dispatcher.dispatch_to_hardware(operation, *args, **kwargs)
    return result.result if result.result is not None else result


async def dispatch_to_emulator_fallback(operation: str, *args, **kwargs) -> Any:
    """Module-level emulator fallback function"""
    dispatcher = _get_global_dispatcher()
    result = await dispatcher.dispatch_to_emulator_fallback(operation, *args, **kwargs)
    return result.result if result.result is not None else result


async def emulate_memristor_mvm(tensor1: Any, tensor2: Any) -> Any:
    """Module-level memristor emulation function"""
    dispatcher = _get_global_dispatcher()
    return await dispatcher._emulate_memristor_mvm(tensor1, tensor2)


async def dispatch_to_emulator(subgraph: Dict, backend: str = "photonic", **kwargs) -> Any:
    """Module-level emulator dispatch function"""
    dispatcher = _get_global_dispatcher()
    return await dispatcher.dispatch_to_emulator(subgraph, backend, **kwargs)


async def optimize_and_dispatch(subgraph: Dict, metrics: Dict[str, float]) -> Any:
    """Module-level optimization and dispatch function"""
    dispatcher = _get_global_dispatcher()
    result = await dispatcher.optimize_and_dispatch(subgraph, metrics)
    return result.result if result.result is not None else result
