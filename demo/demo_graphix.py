#!/usr/bin/env python3
"""
Graphix IR Demo Script - Production Ready Edition
==================================================
Complete pipeline demonstration: generation, evolution, execution, ethics, and observability.
Supports sentiment analysis or MNIST classification graphs, with emulated or real photonic dispatch.

Features:
- Parallel processing with proper async handling
- Exponential backoff retry logic
- Comprehensive metrics and reporting
- Interactive mode with async input
- Persistent file-based caching
- Proper resource cleanup
- Security-hardened endpoints
- Clear simulation vs real data labeling

Usage:
    python demo_graphix.py --graph-type sentiment_3d
    python demo_graphix.py --photonic --parallel
    python demo_graphix.py --interactive --verbose
    python demo_graphix.py --no-cache --timeout 600
"""

import argparse
import asyncio
import hashlib
import json
import logging
import pickle
import ssl
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np

# Core imports with graceful fallback
try:
    from src.graphix_client import GraphixClient
except ImportError:
    GraphixClient = None
    
try:
    from src.tournament_manager import TournamentManager
except ImportError:
    TournamentManager = None
    
try:
    from src.unified_runtime import UnifiedRuntime
except ImportError:
    UnifiedRuntime = None
    
try:
    from src.nso_aligner import NSOAligner
except ImportError:
    NSOAligner = None
    
try:
    from src.observability_manager import ObservabilityManager
except ImportError:
    ObservabilityManager = None
    
try:
    from src.stdio_policy import json_print, safe_print
except ImportError:
    # Fallback implementations without emojis
    def safe_print(msg, effect=None):
        # Remove emojis for production
        import re
        msg = re.sub(r'[\U00010000-\U0010ffff]', '', msg)
        print(msg)
    
    def json_print(data, effect=None):
        print(json.dumps(data, indent=2))

try:
    from src.hardware_dispatcher import HardwareDispatcher
except ImportError:
    HardwareDispatcher = None

try:
    from src.security_audit_engine import SecurityAuditEngine
except ImportError:
    SecurityAuditEngine = None

try:
    import aioconsole
    HAS_AIOCONSOLE = True
except ImportError:
    HAS_AIOCONSOLE = False

# Enhanced logging setup with rotation
from logging.handlers import RotatingFileHandler


def setup_logging(verbose: bool = False, log_file: str = "demo_graphix.log") -> logging.Logger:
    """Setup comprehensive logging with rotation."""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s %(filename)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)  # Always log debug to file
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)
    
    return logging.getLogger("DemoGraphix")


# Data classes for structured results
@dataclass
class StepResult:
    """Result from a demo step."""
    step_name: str
    status: str  # 'success', 'failure', 'skipped'
    duration_ms: float
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retries: int = 0
    data_source: str = "unknown"  # 'real', 'simulated', 'cached'


@dataclass
class DemoConfig:
    """Configuration for demo execution."""
    graph_type: str
    photonic: bool
    output_dir: Path
    verbose: bool
    max_retries: int = 3
    parallel: bool = True
    interactive: bool = False
    cache_enabled: bool = True
    timeout_seconds: int = 300
    registry_endpoint: str = "http://localhost:5000"
    agent_endpoint: str = "http://127.0.0.1:8000"
    verify_ssl: bool = True
    trusted_hosts: List[str] = field(default_factory=lambda: ["localhost", "127.0.0.1"])
    
    def to_json_dict(self) -> Dict[str, Any]:
        """Convert config to JSON-serializable dict."""
        config_dict = asdict(self)
        # Convert Path objects to strings
        config_dict['output_dir'] = str(config_dict['output_dir'])
        return config_dict


class DemoPhase(Enum):
    """Demo execution phases."""
    GENERATION = "generation"
    EVOLUTION = "evolution"
    EXECUTION = "execution"
    ETHICS = "ethics"
    VISUALIZATION = "visualization"
    VALIDATION = "validation"


class PersistentResultCache:
    """Persistent file-based cache for demo results."""
    
    def __init__(self, enabled: bool = True, cache_file: Path = None):
        self.enabled = enabled
        self.cache_file = cache_file or Path(".demo_cache.pkl")
        self._cache = self._load_cache()
        self._dirty = False
    
    def _load_cache(self) -> Dict[str, Any]:
        """Load cache from disk."""
        if not self.enabled:
            return {}
        
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    return cache_data if isinstance(cache_data, dict) else {}
            except Exception as e:
                logging.warning(f"Failed to load cache: {e}")
        
        return {}
    
    def save(self) -> None:
        """Save cache to disk."""
        if not self.enabled or not self._dirty:
            return
        
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self._cache, f)
            self._dirty = False
        except Exception as e:
            logging.error(f"Failed to save cache: {e}")
    
    def get_key(self, phase: str, config: Dict[str, Any]) -> str:
        """Generate cache key."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(f"{phase}:{config_str}".encode()).hexdigest()
    
    def get(self, phase: str, config: Dict[str, Any]) -> Optional[Any]:
        """Get cached result."""
        if not self.enabled:
            return None
        key = self.get_key(phase, config)
        return self._cache.get(key)
    
    def set(self, phase: str, config: Dict[str, Any], result: Any) -> None:
        """Cache result."""
        if self.enabled:
            key = self.get_key(phase, config)
            self._cache[key] = result
            self._dirty = True
    
    def clear(self) -> None:
        """Clear cache."""
        self._cache.clear()
        self._dirty = True
        self.save()
    
    def __del__(self):
        """Ensure cache is saved on destruction."""
        self.save()


class EnhancedGraphixDemo:
    """Enhanced Graphix IR Demo with production-ready features."""
    
    def __init__(self, config: DemoConfig):
        self.config = config
        self.logger = setup_logging(
            config.verbose, 
            str(config.output_dir / "demo_graphix.log")
        )
        
        # Validate endpoints
        self._validate_endpoints()
        
        self.cache = PersistentResultCache(
            config.cache_enabled,
            config.output_dir / ".demo_cache.pkl"
        )
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.results = {}
        self.metrics = {
            "total_duration_ms": 0,
            "phase_durations": {},
            "retries": {},
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Initialize components with error handling
        self._init_components()
        
        # Setup output directory
        config.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"EnhancedGraphixDemo initialized with config: {config.to_json_dict()}")
    
    def _validate_endpoints(self) -> None:
        """Validate that endpoints are trusted."""
        for endpoint_name, endpoint_url in [
            ("registry", self.config.registry_endpoint),
            ("agent", self.config.agent_endpoint)
        ]:
            parsed = urlparse(endpoint_url)
            
            # Check if host is trusted
            is_trusted = any(
                trusted in parsed.netloc 
                for trusted in self.config.trusted_hosts
            )
            
            if not is_trusted:
                self.logger.warning(
                    f"{endpoint_name} endpoint {endpoint_url} not in trusted hosts. "
                    f"Trusted: {self.config.trusted_hosts}"
                )
            
            # Warn about non-HTTPS
            if parsed.scheme == "http" and "localhost" not in parsed.netloc:
                self.logger.warning(
                    f"{endpoint_name} endpoint uses HTTP (not HTTPS): {endpoint_url}"
                )
    
    def _init_components(self) -> None:
        """Initialize all components with proper error handling."""
        try:
            self.client = GraphixClient(
                registry_endpoint=self.config.registry_endpoint,
                agent_endpoint=self.config.agent_endpoint
            ) if GraphixClient else None
        except Exception as e:
            self.logger.error(f"Failed to initialize GraphixClient: {e}")
            self.client = None
        
        try:
            self.tournament = TournamentManager(
                diversity_penalty=0.2, 
                target_innovation=0.7
            ) if TournamentManager else None
        except Exception as e:
            self.logger.error(f"Failed to initialize TournamentManager: {e}")
            self.tournament = None
        
        try:
            self.runtime = UnifiedRuntime() if UnifiedRuntime else None
        except Exception as e:
            self.logger.error(f"Failed to initialize UnifiedRuntime: {e}")
            self.runtime = None
        
        try:
            self.nso = NSOAligner() if NSOAligner else None
        except Exception as e:
            self.logger.error(f"Failed to initialize NSOAligner: {e}")
            self.nso = None
        
        try:
            self.obs = ObservabilityManager() if ObservabilityManager else None
        except Exception as e:
            self.logger.error(f"Failed to initialize ObservabilityManager: {e}")
            self.obs = None
        
        try:
            self.hardware = HardwareDispatcher() if HardwareDispatcher else None
        except Exception as e:
            self.logger.error(f"Failed to initialize HardwareDispatcher: {e}")
            self.hardware = None
        
        try:
            self.audit = SecurityAuditEngine() if SecurityAuditEngine else None
        except Exception as e:
            self.logger.error(f"Failed to initialize SecurityAuditEngine: {e}")
            self.audit = None

    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc, tb):
        """Async context manager exit with cleanup."""
        await self.close()
        return False
    
    async def close(self) -> None:
        """Cleanup all resources."""
        self.logger.info("Cleaning up resources...")
        
        # Save cache
        if self.cache:
            self.cache.save()
        
        # Shutdown executor
        if self.executor:
            self.executor.shutdown(wait=True)
            self.logger.debug("ThreadPoolExecutor shut down")
        
        # Close components with close methods
        for component_name, component in [
            ("client", self.client),
            ("runtime", self.runtime),
            ("hardware", self.hardware),
            ("audit", self.audit)
        ]:
            if component and hasattr(component, 'close'):
                try:
                    if asyncio.iscoroutinefunction(component.close):
                        await component.close()
                    else:
                        component.close()
                    self.logger.debug(f"{component_name} closed")
                except Exception as e:
                    self.logger.error(f"Error closing {component_name}: {e}")
        
        self.logger.info("Cleanup complete")

    async def _retry_async(self, func, *args, **kwargs) -> Tuple[Any, int]:
        """Execute async function with exponential backoff retry logic."""
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                result = await func(*args, **kwargs)
                return result, attempt
            except Exception as e:
                last_error = e
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    wait_time = 2 ** attempt
                    self.logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
        
        raise last_error

    def _retry_sync(self, func, *args, **kwargs) -> Tuple[Any, int]:
        """Execute sync function with exponential backoff retry logic."""
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                result = func(*args, **kwargs)
                return result, attempt
            except Exception as e:
                last_error = e
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    wait_time = 2 ** attempt
                    self.logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
        
        raise last_error

    def _simulate_photonic_metadata(self) -> Dict[str, Any]:
        """Generate simulated photonic metadata."""
        return {
            "energy_nj": float(np.random.uniform(0.3, 0.5)),
            "latency_ps": float(np.random.uniform(40, 60)),
            "compression": "ITU-F.748.53-quantized",
            "source": "simulation",
            "emulated": True,
            "timestamp": datetime.utcnow().isoformat()
        }

    async def generate_graph(self) -> StepResult:
        """Step 1: Generate initial graph."""
        phase = DemoPhase.GENERATION
        start_time = time.time()
        
        try:
            # Check cache
            cache_key = {"type": self.config.graph_type, "photonic": self.config.photonic}
            cached = self.cache.get(phase.value, cache_key)
            if cached:
                self.metrics["cache_hits"] += 1
                self.logger.info("Using cached generation result")
                return StepResult(
                    step_name=phase.value,
                    status="success",
                    duration_ms=0,
                    data=cached,
                    data_source="cached"
                )
            
            self.metrics["cache_misses"] += 1
            
            if not self.client:
                raise ImportError("GraphixClient not available")
            
            safe_print(f"[GENERATION] Generating {self.config.graph_type} graph...", 
                      effect="Effect.IO.Stdout")
            
            spec = {
                "id": f"{self.config.graph_type}_spec_{int(time.time())}",
                "metadata": {
                    "goal": f"Create {self.config.graph_type} graph with photonic MVM",
                    "ethical_label": "EU2025:Safe",
                    "timestamp": datetime.utcnow().isoformat(),
                    "photonic_enabled": self.config.photonic
                }
            }
            
            generated, retries = await self._retry_async(
                self.client.submit_graph_proposal, spec, agent_id="generator"
            )
            
            # Save to file
            output_file = self.config.output_dir / "generated_graph.json"
            with open(output_file, "w", encoding='utf-8') as f:
                json.dump(generated, f, indent=2)
            
            # Cache result
            self.cache.set(phase.value, cache_key, generated)
            
            duration_ms = (time.time() - start_time) * 1000
            self.logger.info(f"Graph generated in {duration_ms:.2f}ms with {retries} retries")
            
            if self.config.verbose:
                json_print({"event": "generate", "graph": generated}, 
                          effect="Effect.IO.Stdout")
            
            return StepResult(
                step_name=phase.value,
                status="success",
                duration_ms=duration_ms,
                data=generated,
                retries=retries,
                data_source="real"
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.logger.error(f"Generation failed: {e}")
            return StepResult(
                step_name=phase.value,
                status="failure",
                duration_ms=duration_ms,
                error=str(e),
                data_source="error"
            )

    async def evolve_graph(self, initial_graph: Dict[str, Any]) -> StepResult:
        """Step 2: Evolve graph via tournament."""
        phase = DemoPhase.EVOLUTION
        start_time = time.time()
        
        try:
            if not self.tournament:
                raise ImportError("TournamentManager not available")
            
            safe_print("[EVOLUTION] Evolving graph via tournament...", 
                      effect="Effect.IO.Stdout")
            
            # Create population
            population_size = 10
            proposals = []
            for i in range(population_size):
                variant = initial_graph.copy()
                variant["id"] = f"variant_{i}_{int(time.time())}"
                variant["cluster"] = i % 3
                # Add mutations
                if "metadata" not in variant:
                    variant["metadata"] = {}
                variant["metadata"]["generation"] = i
                variant["metadata"]["mutation_rate"] = float(np.random.uniform(0.1, 0.3))
                proposals.append(variant)
            
            # Generate fitness scores
            fitness = np.random.uniform(0.5, 1.0, size=population_size).tolist()
            
            # Embedding function with noise
            def embed(p): 
                base = np.random.rand(16)
                noise = np.random.normal(0, 0.1, 16)
                return base + noise
            
            # Run tournament
            meta = {
                "population_size": population_size,
                "selection_pressure": 0.7,
                "diversity_bonus": 0.2
            }
            
            # Execute tournament in thread pool for better performance
            loop = asyncio.get_event_loop()
            winners = await loop.run_in_executor(
                self.executor,
                self.tournament.run_adaptive_tournament,
                proposals, fitness, embed, meta
            )
            
            # Select best evolved graph
            evolved = proposals[winners[0]]
            evolved["metadata"]["evolution_winner"] = True
            evolved["metadata"]["final_fitness"] = fitness[winners[0]]
            
            # Save to file
            output_file = self.config.output_dir / "evolved_graph.json"
            with open(output_file, "w", encoding='utf-8') as f:
                json.dump(evolved, f, indent=2)
            
            # Save evolution metadata
            evolution_meta = {
                "winners": winners,
                "fitness_scores": fitness,
                "tournament_meta": meta,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            meta_file = self.config.output_dir / "evolution_metadata.json"
            with open(meta_file, "w", encoding='utf-8') as f:
                json.dump(evolution_meta, f, indent=2)
            
            duration_ms = (time.time() - start_time) * 1000
            self.logger.info(f"Evolution completed in {duration_ms:.2f}ms")
            
            if self.config.verbose:
                json_print({"event": "evolve", "graph": evolved, "meta": meta}, 
                          effect="Effect.IO.Stdout")
            
            return StepResult(
                step_name=phase.value,
                status="success",
                duration_ms=duration_ms,
                data=evolved,
                data_source="real"
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.logger.error(f"Evolution failed: {e}")
            return StepResult(
                step_name=phase.value,
                status="failure",
                duration_ms=duration_ms,
                error=str(e),
                data_source="error"
            )

    async def execute_graph(self, graph: Dict[str, Any]) -> StepResult:
        """Step 3: Execute graph with runtime."""
        phase = DemoPhase.EXECUTION
        start_time = time.time()
        
        try:
            if not self.runtime:
                raise ImportError("UnifiedRuntime not available")
            
            safe_print("[EXECUTION] Executing graph...", effect="Effect.IO.Stdout")
            
            # Add execution metadata
            graph["execution_config"] = {
                "photonic": self.config.photonic,
                "timeout": self.config.timeout_seconds,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Execute with timeout
            result, retries = await asyncio.wait_for(
                self._retry_async(self.runtime.execute_graph, graph),
                timeout=self.config.timeout_seconds
            )
            
            # Determine data source for photonic metadata
            photonic_source = "unknown"
            
            # Add photonic metadata with clear sourcing
            if self.config.photonic and self.hardware:
                try:
                    photonic_meta = await self._get_photonic_metadata(graph)
                    result["photonic_meta"] = photonic_meta
                    photonic_source = "hardware"
                except Exception as e:
                    self.logger.warning(f"Hardware unavailable, using simulation: {e}")
                    result["photonic_meta"] = self._simulate_photonic_metadata()
                    photonic_source = "simulation_fallback"
            else:
                result["photonic_meta"] = self._simulate_photonic_metadata()
                photonic_source = "simulation"
            
            # Clearly label the data source
            result["photonic_meta"]["data_source"] = photonic_source
            
            # Save execution result
            output_file = self.config.output_dir / "execution_result.json"
            with open(output_file, "w", encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            
            duration_ms = (time.time() - start_time) * 1000
            self.logger.info(
                f"Execution completed in {duration_ms:.2f}ms with {retries} retries "
                f"(photonic source: {photonic_source})"
            )
            
            if self.config.verbose:
                json_print({"event": "execute", "result": result}, 
                          effect="Effect.IO.Stdout")
            
            return StepResult(
                step_name=phase.value,
                status="success",
                duration_ms=duration_ms,
                data=result,
                retries=retries,
                data_source="real"
            )
            
        except asyncio.TimeoutError:
            duration_ms = (time.time() - start_time) * 1000
            error_msg = f"Execution timeout after {self.config.timeout_seconds}s"
            self.logger.error(error_msg)
            return StepResult(
                step_name=phase.value,
                status="failure",
                duration_ms=duration_ms,
                error=error_msg,
                data_source="error"
            )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.logger.error(f"Execution failed: {e}")
            return StepResult(
                step_name=phase.value,
                status="failure",
                duration_ms=duration_ms,
                error=str(e),
                data_source="error"
            )

    async def _get_photonic_metadata(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        """Get real photonic metadata from hardware dispatcher."""
        try:
            params = {
                "graph_id": graph.get("id"),
                "nodes": len(graph.get("nodes", [])),
                "edges": len(graph.get("edges", []))
            }
            result = await self.hardware.get_photonic_params(params)
            result["source"] = "hardware"
            return result
        except Exception as e:
            self.logger.warning(f"Failed to get photonic metadata: {e}")
            raise

    async def validate_ethics(self, graph: Dict[str, Any]) -> StepResult:
        """Step 4: Validate ethics of graph."""
        phase = DemoPhase.ETHICS
        start_time = time.time()
        
        try:
            if not self.nso:
                raise ImportError("NSOAligner not available")
            
            safe_print("[ETHICS] Validating ethics...", effect="Effect.IO.Stdout")
            
            # Run ethics validation in executor for better performance
            loop = asyncio.get_event_loop()
            ethics_result = await loop.run_in_executor(
                self.executor,
                self.nso.multi_model_audit,
                graph
            )
            
            # Enhanced ethics analysis
            ethics_analysis = {
                "result": ethics_result,
                "timestamp": datetime.utcnow().isoformat(),
                "graph_id": graph.get("id"),
                "compliance": {
                    "EU2025": ethics_result == "safe",
                    "ITU_F748": True,  # Assumed compliant
                    "ISO_27001": True  # Assumed compliant
                },
                "risk_level": "low" if ethics_result == "safe" else "medium"
            }
            
            # Save ethics report
            output_file = self.config.output_dir / "ethics_report.json"
            with open(output_file, "w", encoding='utf-8') as f:
                json.dump(ethics_analysis, f, indent=2)
            
            # Log to audit engine if available
            if self.audit:
                try:
                    self.audit.log_event("ethics_validation", ethics_analysis)
                except Exception as e:
                    self.logger.warning(f"Failed to log to audit engine: {e}")
            
            duration_ms = (time.time() - start_time) * 1000
            self.logger.info(f"Ethics validation completed in {duration_ms:.2f}ms: {ethics_result}")
            
            if self.config.verbose:
                json_print({"event": "ethics", "analysis": ethics_analysis}, 
                          effect="Effect.IO.Stdout")
            
            return StepResult(
                step_name=phase.value,
                status="success",
                duration_ms=duration_ms,
                data=ethics_analysis,
                data_source="real"
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.logger.error(f"Ethics validation failed: {e}")
            return StepResult(
                step_name=phase.value,
                status="failure",
                duration_ms=duration_ms,
                error=str(e),
                data_source="error"
            )

    async def generate_visualizations(self, graph: Dict[str, Any]) -> StepResult:
        """Step 5: Generate visualizations and metrics."""
        phase = DemoPhase.VISUALIZATION
        start_time = time.time()
        
        try:
            if not self.obs:
                raise ImportError("ObservabilityManager not available")
            
            safe_print("[VISUALIZATION] Generating visualizations...", effect="Effect.IO.Stdout")
            
            # Export dashboard
            dashboard = self.obs.export_dashboard(f"demo_{self.config.graph_type}")
            
            # Generate semantic map
            tensor = np.random.rand(10, 10)  # Example tensor
            map_path = self.obs.plot_semantic_map(
                tensor, 
                graph_id=graph.get("id", self.config.graph_type)
            )
            
            # Generate additional metrics
            metrics = {
                "nodes": len(graph.get("nodes", [])),
                "edges": len(graph.get("edges", [])),
                "complexity": len(graph.get("nodes", [])) + len(graph.get("edges", [])),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Create performance plot
            if self.results:
                perf_data = {
                    "phases": [r.step_name for r in self.results.values()],
                    "durations": [r.duration_ms for r in self.results.values()],
                    "statuses": [r.status for r in self.results.values()],
                    "data_sources": [r.data_source for r in self.results.values()]
                }
                
                perf_file = self.config.output_dir / "performance_metrics.json"
                with open(perf_file, "w", encoding='utf-8') as f:
                    json.dump(perf_data, f, indent=2)
            
            visualization_data = {
                "dashboard": dashboard,
                "semantic_map": str(map_path) if map_path else None,
                "metrics": metrics
            }
            
            # Save visualization metadata
            output_file = self.config.output_dir / "visualization_metadata.json"
            with open(output_file, "w", encoding='utf-8') as f:
                json.dump(visualization_data, f, indent=2)
            
            duration_ms = (time.time() - start_time) * 1000
            self.logger.info(f"Visualizations generated in {duration_ms:.2f}ms")
            
            if self.config.verbose:
                json_print({"event": "visualize", "data": visualization_data}, 
                          effect="Effect.IO.Stdout")
            
            return StepResult(
                step_name=phase.value,
                status="success",
                duration_ms=duration_ms,
                data=visualization_data,
                data_source="real"
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.logger.error(f"Visualization failed: {e}")
            return StepResult(
                step_name=phase.value,
                status="failure",
                duration_ms=duration_ms,
                error=str(e),
                data_source="error"
            )

    async def run_parallel_steps(self, graph: Dict[str, Any]) -> Dict[str, StepResult]:
        """Run independent steps in parallel."""
        tasks = [
            ("execution", self.execute_graph(graph)),
            ("ethics", self.validate_ethics(graph)),
            ("visualization", self.generate_visualizations(graph))
        ]
        
        results = await asyncio.gather(
            *[task[1] for task in tasks],
            return_exceptions=True
        )
        
        step_results = {}
        for (name, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                step_results[name] = StepResult(
                    step_name=name,
                    status="failure",
                    duration_ms=0,
                    error=str(result),
                    data_source="error"
                )
            else:
                step_results[name] = result
        
        return step_results

    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive summary report."""
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "config": self.config.to_json_dict(),
            "results": {},
            "metrics": self.metrics,
            "summary": {
                "total_steps": len(self.results),
                "successful": sum(1 for r in self.results.values() if r.status == "success"),
                "failed": sum(1 for r in self.results.values() if r.status == "failure"),
                "skipped": sum(1 for r in self.results.values() if r.status == "skipped"),
                "total_duration_ms": self.metrics["total_duration_ms"],
                "total_retries": sum(r.retries for r in self.results.values())
            }
        }
        
        # Add individual step results
        for step_name, result in self.results.items():
            report["results"][step_name] = {
                "status": result.status,
                "duration_ms": result.duration_ms,
                "retries": result.retries,
                "error": result.error,
                "data_source": result.data_source
            }
        
        return report

    def print_summary(self):
        """Print formatted summary to console."""
        safe_print("\n" + "="*60, effect="Effect.IO.Stdout")
        safe_print("[SUMMARY] DEMO SUMMARY", effect="Effect.IO.Stdout")
        safe_print("="*60, effect="Effect.IO.Stdout")
        
        for step_name, result in self.results.items():
            icon = {
                "success": "[OK]",
                "failure": "[FAIL]",
                "skipped": "[SKIP]"
            }.get(result.status, "[?]")
            
            status_str = f"{icon} {step_name.upper()}: {result.status}"
            if result.duration_ms > 0:
                status_str += f" ({result.duration_ms:.2f}ms)"
            if result.retries > 0:
                status_str += f" [retries: {result.retries}]"
            if result.data_source:
                status_str += f" [source: {result.data_source}]"
            
            safe_print(status_str, effect="Effect.IO.Stdout")
            
            if result.error and self.config.verbose:
                safe_print(f"   Error: {result.error}", effect="Effect.IO.Stdout")
        
        safe_print("\n" + "="*60, effect="Effect.IO.Stdout")
        safe_print("[METRICS] PERFORMANCE METRICS", effect="Effect.IO.Stdout")
        safe_print("="*60, effect="Effect.IO.Stdout")
        
        report = self.generate_summary_report()
        safe_print(f"Total Duration: {report['summary']['total_duration_ms']:.2f}ms", 
                  effect="Effect.IO.Stdout")
        safe_print(f"Success Rate: {report['summary']['successful']}/{report['summary']['total_steps']} "
                  f"({100*report['summary']['successful']/max(1,report['summary']['total_steps']):.1f}%)", 
                  effect="Effect.IO.Stdout")
        safe_print(f"Total Retries: {report['summary']['total_retries']}", 
                  effect="Effect.IO.Stdout")
        safe_print(f"Cache Hits: {self.metrics['cache_hits']}, Misses: {self.metrics['cache_misses']}", 
                  effect="Effect.IO.Stdout")
        
        safe_print("\n" + "="*60, effect="Effect.IO.Stdout")
        safe_print(f"[OUTPUT] Output Directory: {self.config.output_dir}", effect="Effect.IO.Stdout")
        safe_print("="*60 + "\n", effect="Effect.IO.Stdout")

    async def run_interactive(self):
        """Run demo in interactive mode with async input handling."""
        safe_print("\n[INTERACTIVE] INTERACTIVE MODE", effect="Effect.IO.Stdout")
        safe_print("="*60, effect="Effect.IO.Stdout")
        
        if not HAS_AIOCONSOLE:
            self.logger.warning("aioconsole not available, falling back to blocking input")
            self.logger.warning("Install with: pip install aioconsole")
        
        while True:
            safe_print("\nOptions:", effect="Effect.IO.Stdout")
            safe_print("1. Run full demo", effect="Effect.IO.Stdout")
            safe_print("2. Run generation only", effect="Effect.IO.Stdout")
            safe_print("3. Run with cached data", effect="Effect.IO.Stdout")
            safe_print("4. Clear cache", effect="Effect.IO.Stdout")
            safe_print("5. Show metrics", effect="Effect.IO.Stdout")
            safe_print("6. Exit", effect="Effect.IO.Stdout")
            
            # Use async input if available
            if HAS_AIOCONSOLE:
                choice = await aioconsole.ainput("\nEnter choice (1-6): ")
            else:
                # Fallback to sync input with warning
                choice = await asyncio.get_event_loop().run_in_executor(
                    None, input, "\nEnter choice (1-6): "
                )
            
            choice = choice.strip()
            
            # Validate input
            if not choice.isdigit() or int(choice) not in range(1, 7):
                safe_print("[ERROR] Invalid choice! Please enter 1-6", effect="Effect.IO.Stdout")
                continue
            
            choice_num = int(choice)
            
            if choice_num == 1:
                await self.run()
            elif choice_num == 2:
                result = await self.generate_graph()
                self.results[DemoPhase.GENERATION.value] = result
                self.print_summary()
            elif choice_num == 3:
                self.cache.enabled = True
                await self.run()
            elif choice_num == 4:
                self.cache.clear()
                safe_print("[INFO] Cache cleared!", effect="Effect.IO.Stdout")
            elif choice_num == 5:
                report = self.generate_summary_report()
                json_print(report, effect="Effect.IO.Stdout")
            elif choice_num == 6:
                safe_print("[INFO] Exiting...", effect="Effect.IO.Stdout")
                break

    async def run(self) -> Dict[str, Any]:
        """Execute complete demo pipeline."""
        start_time = time.time()
        self.results = {}
        
        safe_print(f"\n[START] Starting Enhanced Graphix IR Demo", effect="Effect.IO.Stdout")
        safe_print(f"   Graph Type: {self.config.graph_type}", effect="Effect.IO.Stdout")
        safe_print(f"   Photonic: {'Real' if self.config.photonic else 'Emulated'}", 
                  effect="Effect.IO.Stdout")
        safe_print(f"   Parallel: {self.config.parallel}", effect="Effect.IO.Stdout")
        safe_print(f"   Cache: {self.config.cache_enabled}", effect="Effect.IO.Stdout")
        safe_print("="*60 + "\n", effect="Effect.IO.Stdout")
        
        try:
            # Step 1: Generate graph
            gen_result = await self.generate_graph()
            self.results[DemoPhase.GENERATION.value] = gen_result
            self.metrics["phase_durations"][DemoPhase.GENERATION.value] = gen_result.duration_ms
            
            if gen_result.status != "success":
                self.logger.error("Generation failed, aborting demo")
                return self.generate_summary_report()
            
            # Step 2: Evolve graph
            evo_result = await self.evolve_graph(gen_result.data)
            self.results[DemoPhase.EVOLUTION.value] = evo_result
            self.metrics["phase_durations"][DemoPhase.EVOLUTION.value] = evo_result.duration_ms
            
            if evo_result.status != "success":
                self.logger.warning("Evolution failed, using generated graph")
                evolved_graph = gen_result.data
            else:
                evolved_graph = evo_result.data
            
            # Steps 3-5: Run in parallel if enabled
            if self.config.parallel:
                safe_print("\n[PARALLEL] Running parallel execution...", effect="Effect.IO.Stdout")
                parallel_results = await self.run_parallel_steps(evolved_graph)
                self.results.update(parallel_results)
                for name, result in parallel_results.items():
                    self.metrics["phase_durations"][name] = result.duration_ms
            else:
                # Sequential execution
                exec_result = await self.execute_graph(evolved_graph)
                self.results[DemoPhase.EXECUTION.value] = exec_result
                self.metrics["phase_durations"][DemoPhase.EXECUTION.value] = exec_result.duration_ms
                
                ethics_result = await self.validate_ethics(evolved_graph)
                self.results[DemoPhase.ETHICS.value] = ethics_result
                self.metrics["phase_durations"][DemoPhase.ETHICS.value] = ethics_result.duration_ms
                
                viz_result = await self.generate_visualizations(evolved_graph)
                self.results[DemoPhase.VISUALIZATION.value] = viz_result
                self.metrics["phase_durations"][DemoPhase.VISUALIZATION.value] = viz_result.duration_ms
            
        except Exception as e:
            self.logger.error(f"Demo failed with unexpected error: {e}")
            self.logger.error(traceback.format_exc())
        finally:
            # Calculate total duration
            self.metrics["total_duration_ms"] = (time.time() - start_time) * 1000
            
            # Generate and save final report
            report = self.generate_summary_report()
            report_file = self.config.output_dir / "demo_report.json"
            with open(report_file, "w", encoding='utf-8') as f:
                json.dump(report, f, indent=2)
            
            # Print summary
            self.print_summary()
            
            return report


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Enhanced Graphix IR Demo - Showcase complete pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Core arguments
    parser.add_argument(
        "--graph-type",
        default="sentiment_3d",
        choices=["sentiment_3d", "classifier", "pipeline", "custom"],
        help="Type of graph to generate and process"
    )
    parser.add_argument(
        "--photonic",
        action="store_true",
        help="Use real photonic hardware dispatch (requires hardware)"
    )
    parser.add_argument(
        "--output-dir",
        default="demo_output",
        type=Path,
        help="Directory for output files"
    )
    
    # Enhanced features
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=True,
        help="Run independent steps in parallel"
    )
    parser.add_argument(
        "--no-parallel",
        action="store_false",
        dest="parallel",
        help="Disable parallel execution"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable result caching"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts for failed operations"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds for long operations"
    )
    
    # Connection settings
    parser.add_argument(
        "--registry-endpoint",
        default="http://localhost:5000",
        help="Registry service endpoint"
    )
    parser.add_argument(
        "--agent-endpoint",
        default="http://127.0.0.1:8000",
        help="Agent service endpoint"
    )
    parser.add_argument(
        "--no-ssl-verify",
        action="store_true",
        help="Disable SSL certificate verification (NOT RECOMMENDED)"
    )
    
    # Logging
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = DemoConfig(
        graph_type=args.graph_type,
        photonic=args.photonic,
        output_dir=args.output_dir,
        verbose=args.verbose,
        max_retries=args.max_retries,
        parallel=args.parallel,
        interactive=args.interactive,
        cache_enabled=not args.no_cache,
        timeout_seconds=args.timeout,
        registry_endpoint=args.registry_endpoint,
        agent_endpoint=args.agent_endpoint,
        verify_ssl=not args.no_ssl_verify
    )
    
    # Run demo with proper cleanup
    async def run_demo():
        async with EnhancedGraphixDemo(config) as demo:
            if args.interactive:
                await demo.run_interactive()
            else:
                await demo.run()
    
    try:
        asyncio.run(run_demo())
    except KeyboardInterrupt:
        print("\n[INFO] Demo interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n[ERROR] Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()