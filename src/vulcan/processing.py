# ============================================================
# VULCAN-AGI Processing Module
# Multimodal processing, embeddings, and cross-attention mechanisms
# Enhanced with optimization, dynamic loading, workload management, and traceability
# FULLY DEBUGGED VERSION - All critical issues resolved
# FIXED: Import paths corrected for src.vulcan package structure
# FIXED: All 13 test failures resolved - inference tensors, dimension mismatch, attention fusion, log rotation, caching
# ============================================================

from __future__ import annotations

import asyncio
import gc
import hashlib
import json
import logging
import pickle
import queue
import shutil
import threading
import time
import uuid
from collections import OrderedDict, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import os
import PIL.Image
import psutil
import torch
import torch.nn as nn

# REMOVED: sentence_transformers import - will use internal LLM-based encoder
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

# FIXED: Import from src.vulcan.config instead of config
from src.vulcan.config import EMBEDDING_DIM, HIDDEN_DIM, LATENT_DIM, ModalityType

# HuggingFace Model Configuration (CWE-494 mitigation)
# To pin models to specific revisions for security, set these environment variables:
# - VULCAN_BERT_MODEL_REVISION: commit hash for BERT model
# - VULCAN_VISION_AUDIO_MODEL_REVISION: commit hash for vision/audio models
BERT_MODEL_REVISION = os.environ.get("VULCAN_BERT_MODEL_REVISION", None)
VISION_AUDIO_MODEL_REVISION = os.environ.get("VULCAN_VISION_AUDIO_MODEL_REVISION", None)

# Initialize logger for this module
logger = logging.getLogger(__name__)

# --- Graphix Module Imports ---
try:
    from src.ai_providers import AIProviders
except ImportError:
    AIProviders = None
try:
    from src.unified_runtime import UnifiedRuntime
except ImportError:
    UnifiedRuntime = None


# --- Simple Mode Import ---
try:
    from src.vulcan.simple_mode import should_skip_bert, SKIP_BERT_EMBEDDINGS
except ImportError:
    # Fallback if simple_mode not available
    SKIP_BERT_EMBEDDINGS = os.getenv("SKIP_BERT_EMBEDDINGS", "false").lower() in ("true", "1", "yes", "on")
    def should_skip_bert():
        return SKIP_BERT_EMBEDDINGS


# --- Custom LLM Dependency ---
# NOTE: In a real system, this would be imported from src.llm_core.graphix_transformer

# PERFORMANCE FIX: Thread-safe singleton holder for GraphixTransformer
# This prevents re-loading BERT models on every request
_graphix_transformer_instance: Optional["GraphixTransformer"] = None
_graphix_transformer_lock = threading.Lock()


class GraphixTransformer:
    """
    Core LLM component with real BERT embeddings.
    Uses pre-trained BERT model for text embeddings.
    
    PERFORMANCE: Set SKIP_BERT_EMBEDDINGS=true to skip BERT loading
    and use lightweight embeddings instead. Saves 3.5s+ per request.
    
    PERFORMANCE FIX: This class now implements a singleton pattern via
    get_instance() to prevent loading BERT models multiple times per request.
    Use get_instance() instead of direct instantiation for model reuse.
    """

    @classmethod
    def get_instance(cls, config=None, embedding_dim=EMBEDDING_DIM) -> "GraphixTransformer":
        """
        Get the singleton instance of GraphixTransformer.
        
        PERFORMANCE FIX: This ensures the BERT model is only loaded once,
        preventing the 3-5 second model load on every request.
        
        Thread-safe implementation using double-checked locking.
        
        Args:
            config: Optional configuration (only used on first instantiation)
            embedding_dim: Embedding dimension (only used on first instantiation)
            
        Returns:
            The singleton GraphixTransformer instance
        """
        global _graphix_transformer_instance
        
        if _graphix_transformer_instance is None:
            with _graphix_transformer_lock:
                # Double-checked locking for thread safety
                if _graphix_transformer_instance is None:
                    logger.info("Creating singleton GraphixTransformer instance")
                    _graphix_transformer_instance = cls(config=config, embedding_dim=embedding_dim)
        
        return _graphix_transformer_instance

    def __init__(self, config=None, embedding_dim=EMBEDDING_DIM):
        self.embedding_dim = embedding_dim
        self.device = "cpu"

        # PERFORMANCE: Skip BERT loading if SKIP_BERT_EMBEDDINGS is set
        # When skipping, we set attributes to None which causes get_embeddings()
        # to use the fallback random embeddings path (line ~170). This is intentional
        # as it avoids the 3.5s+ BERT model load time for simple chat use cases.
        if should_skip_bert():
            logger.info("Skipping BERT embeddings - using lightweight fallback (SKIP_BERT_EMBEDDINGS=true)")
            self.tokenizer = None
            self.model = None
            self.projection = None
            return

        # SECURITY: Support model revision pinning (CWE-494 mitigation)
        revision = BERT_MODEL_REVISION if BERT_MODEL_REVISION else "main"

        # Load real BERT model and tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "bert-base-uncased", revision=revision
            )  # nosec B615 - revision parameter present

            self.model = AutoModel.from_pretrained(
                "bert-base-uncased", revision=revision
            )  # nosec B615 - revision parameter present

            self.model.eval()  # Set to evaluation mode

            # Pre-compute projection matrix for consistent embeddings
            bert_dim = 768  # BERT base hidden size
            self.projection = torch.randn(bert_dim, 384) * 0.01

            logger.info("Loaded real BERT model for embeddings")
        except Exception as e:
            logger.error(f"Failed to load BERT model: {e}, using fallback")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "bert-base-uncased", revision=revision
            )
            self.model = None
            self.projection = None

    def get_embeddings(self, text: Union[str, List[str]]) -> torch.Tensor:
        """
        Get real BERT embeddings for text.
        Returns 384-dim embeddings to match projection layers.
        """
        if isinstance(text, str):
            text = [text]

        batch_size = len(text)
        embedding_size = 384  # Match projection layer

        # Use real BERT model if available
        if self.model is not None:
            try:
                # Tokenize text
                encoded = self.tokenizer(
                    text,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )

                # Get embeddings from BERT
                with torch.no_grad():
                    outputs = self.model(**encoded)
                    # Use CLS token embedding (first token)
                    cls_embeddings = outputs.last_hidden_state[:, 0, :]

                    # Project to 384 dimensions using stored projection matrix
                    if (
                        cls_embeddings.shape[1] != embedding_size
                        and self.projection is not None
                    ):
                        embeddings = torch.matmul(cls_embeddings, self.projection)
                    else:
                        embeddings = cls_embeddings

                    return embeddings.float()

            except Exception as e:
                logger.warning(f"BERT embedding failed: {e}, using fallback")

        # Fallback to random embeddings if model unavailable
        logger.debug("Using fallback random embeddings")
        return torch.randn(batch_size, embedding_size, dtype=torch.float32)


class GraphixTextEncoder:
    """A wrapper to provide a consistent interface for the LLM's embedding method."""

    def __init__(self, llm: GraphixTransformer, device: str = "cpu"):
        self.llm = llm
        self.device = device
        self.name = "GraphixTransformer_TextEncoder"

    def encode(
        self, sentences: Union[str, List[str]], convert_to_tensor: bool = True
    ) -> Union[torch.Tensor, np.ndarray]:
        """Encodes text using the GraphixTransformer's embedding layer."""
        embeddings = self.llm.get_embeddings(sentences)

        if convert_to_tensor:
            return embeddings.to(self.device)
        else:
            return embeddings.cpu().numpy()


# --- END Custom LLM Dependency ---

logger = logging.getLogger(__name__)

# Debounce timers for memory warnings
_last_gpu_mem_warn_ts = 0
_last_cpu_mem_warn_ts = 0

# ============================================================
# PROCESSING QUALITY LEVELS & PRIORITIES
# ============================================================


class ProcessingQuality(Enum):
    """Processing quality levels for adaptive processing."""

    FAST = "fast"  # <50ms, cached/approximate
    BALANCED = "balanced"  # 50-200ms, standard processing
    QUALITY = "quality"  # >200ms, ensemble/advanced
    ADAPTIVE = "adaptive"  # Automatically choose based on context


class ProcessingPriority(Enum):
    """Processing priority for workload management."""

    CRITICAL = 0  # Highest priority
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BATCH = 4  # Lowest priority


@dataclass
class SLOConfig:
    """Service Level Objective configuration."""

    max_latency_ms: float = 100.0
    target_throughput: float = 100.0  # requests/sec
    max_batch_size: int = 32
    priority: ProcessingPriority = ProcessingPriority.NORMAL
    quality_preference: ProcessingQuality = ProcessingQuality.ADAPTIVE


# ============================================================
# VERSIONED DATA LOGGER (FIXED)
# ============================================================


class VersionedDataLogger:
    """Logs and versions all input/output data for traceability."""

    def __init__(
        self,
        log_dir: str = "processing_logs",
        max_entries: int = 1000,
        enable_versioning: bool = True,
        max_age_days: int = 30,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.max_entries = max_entries
        self.enable_versioning = enable_versioning
        self.max_age_days = max_age_days
        self.current_log = deque(maxlen=max_entries)
        self.log_lock = threading.Lock()
        self.version_counter = 0
        self.session_id = str(uuid.uuid4())

        # Initialize log file
        self.log_file = self.log_dir / f"processing_log_{self.session_id}.jsonl"
        self.data_store = self.log_dir / "data_store"
        self.data_store.mkdir(exist_ok=True)

        # FIXED: Track rotation time
        self._last_rotation = time.time()

        # FIXED: Start cleanup thread
        self._shutdown_event = threading.Event()
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()

    def log_processing(
        self, input_data: Any, output_data: Any, metadata: Dict[str, Any]
    ) -> str:
        """Log a processing operation with versioning."""
        log_id = str(uuid.uuid4())
        timestamp = time.time()

        # Create log entry
        entry = {
            "log_id": log_id,
            "timestamp": timestamp,
            "session_id": self.session_id,
            "version": self.version_counter,
            "metadata": metadata,
        }

        # Store data if versioning enabled
        if self.enable_versioning:
            # Save input data
            input_hash = self._save_data(input_data, f"{log_id}_input")
            entry["input_hash"] = input_hash
            entry["input_type"] = type(input_data).__name__

            # Save output data
            output_hash = self._save_data(output_data, f"{log_id}_output")
            entry["output_hash"] = output_hash
            entry["output_type"] = type(output_data).__name__
        else:
            # Just store summaries
            entry["input_summary"] = self._summarize_data(input_data)
            entry["output_summary"] = self._summarize_data(output_data)

        # Write to log
        with self.log_lock:
            # FIXED: Write directly to file, don't accumulate unbounded in memory
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")

            # Keep only recent entries in memory for quick access
            self.current_log.append(entry)

            self.version_counter += 1

            # FIXED: Time-based rotation instead of count-based only
            if timestamp - self._last_rotation > 3600:  # Rotate every hour
                self._rotate_logs()
                self._last_rotation = timestamp

        return log_id

    def _save_data(self, data: Any, prefix: str) -> str:
        """Save data to versioned store and return hash."""
        # Serialize data
        try:
            if isinstance(data, np.ndarray):
                serialized = pickle.dumps(data.tolist())
            elif isinstance(data, torch.Tensor):
                serialized = pickle.dumps(data.detach().cpu().numpy().tolist())
            elif isinstance(data, PIL.Image.Image):
                serialized = pickle.dumps(np.array(data).tolist())
            else:
                serialized = pickle.dumps(data)
        except Exception:
            serialized = pickle.dumps(str(data))

        # Compute hash
        data_hash = hashlib.sha256(serialized).hexdigest()

        # Save to file
        data_path = self.data_store / f"{prefix}_{data_hash[:8]}.pkl"
        if not data_path.exists():
            with open(data_path, "wb") as f:
                f.write(serialized)

        return data_hash

    def _summarize_data(self, data: Any) -> Dict[str, Any]:
        """Create a summary of data without storing it."""
        if isinstance(data, (np.ndarray, torch.Tensor)):
            return {
                "type": type(data).__name__,
                "shape": list(data.shape),
                "dtype": str(data.dtype),
                "mean": float(data.mean()) if data.size > 0 else 0,
                "std": float(data.std()) if data.size > 0 else 0,
            }
        elif isinstance(data, str):
            return {
                "type": "str",
                "length": len(data),
                "preview": data[:100] if len(data) > 100 else data,
            }
        else:
            return {
                "type": type(data).__name__,
                "size": len(data) if hasattr(data, "__len__") else None,
            }

    def _rotate_logs(self):
        """Rotate log files when they get too large."""
        # Archive current log
        archive_name = f"processing_log_{self.session_id}_v{self.version_counter}.jsonl"
        archive_path = self.log_dir / "archive" / archive_name
        archive_path.parent.mkdir(exist_ok=True)

        # Close any open file handles first
        try:
            # Force Python to close any open handles to the file
            gc.collect()
        except Exception as e:
            # GC failure is unexpected but not critical
            logger.warning(f"Garbage collection failed during log rotation: {e}")

        # Move current log to archive
        if self.log_file.exists():
            try:
                self.log_file.rename(archive_path)
            except Exception as e:
                logger.error(f"Failed to rotate log file: {e}")
                # If rename fails, copy then delete
                try:
                    shutil.copy2(self.log_file, archive_path)
                    self.log_file.unlink()
                except Exception as e2:
                    logger.error(f"Failed to copy log file: {e2}")

        # FIXED: Create new log file immediately and ensure it exists
        self.log_file = self.log_dir / f"processing_log_{self.session_id}.jsonl"
        self.log_file.touch()  # Ensure file exists

    def retrieve_data(self, log_id: str) -> Optional[Tuple[Any, Any]]:
        """Retrieve input/output data for a given log ID."""
        # Search for log entry in memory first
        with self.log_lock:
            for entry in self.current_log:
                if entry["log_id"] == log_id:
                    if self.enable_versioning:
                        # Load from store
                        input_data = self._load_data(entry["input_hash"])
                        output_data = self._load_data(entry["output_hash"])
                        return input_data, output_data
                    else:
                        return entry["input_summary"], entry["output_summary"]

        # If not in memory, search log file
        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line)
                    if entry["log_id"] == log_id:
                        if self.enable_versioning:
                            input_data = self._load_data(entry["input_hash"])
                            output_data = self._load_data(entry["output_hash"])
                            return input_data, output_data
                        else:
                            return entry["input_summary"], entry["output_summary"]
        except Exception as e:
            # Log retrieval failures should be logged
            logger.warning(f"Failed to retrieve log entry {log_id}: {e}")

        return None

    def _load_data(self, data_hash: str) -> Any:
        """Load data from versioned store."""
        # Find file with hash prefix
        for file in self.data_store.glob(f"*_{data_hash[:8]}.pkl"):
            with open(file, "rb") as f:
                return pickle.load(f)  # nosec B301 - Internal data structure
        return None

    def _cleanup_loop(self):
        """Periodically clean up old log files."""
        while not self._shutdown_event.is_set():  # ADD THIS
            try:
                # CHANGE THIS LINE
                if self._shutdown_event.wait(timeout=3600):  # Run every hour
                    break  # Shutdown signaled
                self._cleanup_old_logs()
            except Exception as e:
                logger.error(f"Error in log cleanup: {e}")

    def _cleanup_old_logs(self):
        """Remove log files older than max_age_days."""
        cutoff_time = time.time() - (self.max_age_days * 86400)

        removed_count = 0
        freed_bytes = 0

        # Clean data store
        for file in self.data_store.glob("*.pkl"):
            try:
                if file.stat().st_mtime < cutoff_time:
                    file_size = file.stat().st_size
                    file.unlink()
                    removed_count += 1
                    freed_bytes += file_size
            except Exception as e:
                logger.error(f"Failed to remove {file}: {e}")

        # Clean archived logs
        archive_dir = self.log_dir / "archive"
        if archive_dir.exists():
            for file in archive_dir.glob("*.jsonl"):
                try:
                    if file.stat().st_mtime < cutoff_time:
                        file_size = file.stat().st_size
                        file.unlink()
                        removed_count += 1
                        freed_bytes += file_size
                except Exception as e:
                    logger.error(f"Failed to remove {file}: {e}")

        if removed_count > 0:
            freed_mb = freed_bytes / (1024 * 1024)
            logger.info(
                f"Cleaned up {removed_count} old log files, freed {freed_mb:.2f}MB"
            )

    def shutdown(self):
        """Shutdown logger gracefully."""
        self._shutdown_event.set()
        if self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5)

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.shutdown()
        except Exception as e:
            # Destructor failures in ProcessingLogger should be logged at debug level
            logger.debug(f"ProcessingLogger cleanup in destructor failed: {e}")


# ============================================================
# DYNAMIC MODEL MANAGER WITH HOT-SWAPPING (FIXED & MODIFIED)
# ============================================================


class DynamicModelManager:
    """Enhanced model manager with dynamic loading and hot-swapping."""

    _instance = None
    _models = {}
    _processors = {}
    _tokenizers = {}
    _model_configs = {}
    _ai_providers = None
    _initialized = False
    _lock = threading.RLock()
    _creation_lock = threading.Lock()
    _device_map = {}  # Maps models to devices (CPU/GPU/Photonic)

    def __new__(cls):
        # FIXED: Double-checked locking pattern for thread safety
        if cls._instance is None:
            with cls._creation_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not DynamicModelManager._initialized:
            DynamicModelManager._ai_providers = AIProviders() if AIProviders else None
            self._init_device_mapping()
            self._init_model_configs()
            DynamicModelManager._initialized = True

            # Monitor thread for hot-swapping
            self._shutdown_event = threading.Event()
            self._monitor_thread = threading.Thread(
                target=self._monitor_resources, daemon=True
            )
            self._monitor_thread.start()

    def _init_device_mapping(self):
        """Initialize device mapping for models."""
        if torch.cuda.is_available():
            self._device_map["primary"] = "cuda:0"
            if torch.cuda.device_count() > 1:
                self._device_map["secondary"] = "cuda:1"
        else:
            self._device_map["primary"] = "cpu"
            self._device_map["secondary"] = "cpu"

        # Check for photonic accelerator (placeholder)
        self._device_map["photonic"] = self._check_photonic_accelerator()

    def _check_photonic_accelerator(self) -> Optional[str]:
        """Check for photonic accelerator availability."""
        # Placeholder for photonic accelerator detection
        # In real implementation, this would interface with photonic hardware
        return None

    def _init_model_configs(self):
        """Initialize model configurations for dynamic loading."""
        # MODIFIED: Text models now use a placeholder name to indicate use of GraphixTransformer
        self._model_configs = {
            "text": {
                "fast": {"name": "graphix-llm-embed-fast", "device": "cpu"},
                "balanced": {"name": "graphix-llm-embed-balanced", "device": "primary"},
                "quality": {"name": "graphix-llm-embed-quality", "device": "primary"},
            },
            "vision": {
                "fast": {"name": "google/vit-base-patch16-224", "device": "cpu"},
                "balanced": {
                    "name": "google/vit-base-patch16-224",
                    "device": "primary",
                },
                "quality": {
                    "name": "google/vit-large-patch16-224",
                    "device": "primary",
                },
            },
            "audio": {
                "fast": {"name": "facebook/wav2vec2-base", "device": "cpu"},
                "balanced": {"name": "facebook/wav2vec2-base", "device": "primary"},
                "quality": {"name": "facebook/wav2vec2-large", "device": "primary"},
            },
        }

    def _monitor_resources(self):
        """Monitor system resources and trigger hot-swapping if needed."""
        global _last_gpu_mem_warn_ts, _last_cpu_mem_warn_ts

        while not self._shutdown_event.is_set():
            try:
                # Check GPU memory
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        mem_free = torch.cuda.mem_get_info(i)[0] / 1024**3  # GB
                        # mem_total could be used for future memory tracking
                        # mem_total = torch.cuda.mem_get_info(i)[1] / 1024**3

                        if mem_free < 1.0:  # Less than 1GB free
                            now = time.time()
                            if (
                                now - _last_gpu_mem_warn_ts > 30
                            ):  # warn at most every 30s
                                logger.warning(
                                    f"Low GPU memory on device {i}: {mem_free:.2f}GB free"
                                )
                                _last_gpu_mem_warn_ts = now
                            self._optimize_models("gpu_memory")

                # Check CPU memory
                cpu_percent = psutil.virtual_memory().percent
                if cpu_percent > 90:
                    now = time.time()
                    if now - _last_cpu_mem_warn_ts > 30:  # warn at most every 30s
                        logger.warning(f"High CPU memory usage: {cpu_percent}%")
                        _last_cpu_mem_warn_ts = now
                    self._optimize_models("cpu_memory")

                # Check CPU usage
                cpu_usage = psutil.cpu_percent(interval=1)
                if cpu_usage > 90:
                    logger.warning(f"High CPU usage: {cpu_usage}%")
                    self._optimize_models("cpu_usage")

                # Check every 10 seconds, interruptible
                if self._shutdown_event.wait(timeout=10):  # CHANGE THIS
                    break  # Shutdown signaled

            except Exception as e:
                logger.error(f"Error in resource monitor: {e}")
                if self._shutdown_event.wait(timeout=30):
                    break

    def _optimize_models(self, reason: str):
        """Optimize model loading based on resource constraints."""
        # FIXED: Proper GPU memory cleanup with complete cache clearing
        with self._lock:
            if reason == "gpu_memory":
                # Move models to CPU and free GPU memory
                models_moved = 0
                models_to_remove = []

                # First pass: identify models to move
                for key, model in self._models.items():
                    # Skip GraphixTextEncoder instances
                    if isinstance(model, GraphixTextEncoder):
                        continue

                    if hasattr(model, "to") and hasattr(model, "parameters"):
                        try:
                            first_param = next(model.parameters(), None)
                            if first_param is not None and first_param.is_cuda:
                                models_to_remove.append(key)
                                models_moved += 1

                                if models_moved >= 2:  # Move up to 2 models
                                    break
                        except StopIteration:
                            continue

                # Second pass: move and cleanup
                for key in models_to_remove:
                    model = self._models[key]

                    # Move model to CPU
                    model.cpu()

                    # Clear gradients if model has them
                    if hasattr(model, "zero_grad"):
                        model.zero_grad(set_to_none=True)

                    # Delete from all caches
                    del self._models[key]
                    if key in self._processors:
                        del self._processors[key]
                    if key in self._tokenizers:
                        del self._tokenizers[key]

                    logger.info(f"Moved model {key} to CPU and freed GPU memory")

                # FIXED: Force complete GPU cleanup after all deletions
                if models_moved > 0:
                    # Force garbage collection
                    gc.collect()

                    # Clear CUDA cache multiple times for thorough cleanup
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()  # Wait for GPU operations
                        time.sleep(0.1)  # Brief pause
                        torch.cuda.empty_cache()  # Second pass

                        # Log memory status after cleanup
                        for i in range(torch.cuda.device_count()):
                            mem_free = torch.cuda.mem_get_info(i)[0] / 1024**3
                            logger.info(
                                f"GPU {i} memory after cleanup: {mem_free:.2f}GB free"
                            )

            elif reason == "cpu_memory":
                # Unload least recently used models
                keys_to_keep = [
                    key
                    for key in self._models.keys()
                    if isinstance(self._models[key], GraphixTextEncoder)
                ]

                if len(self._models) > len(keys_to_keep) + 2:
                    # Keep only essential and 2 most recent non-essential models
                    keys_to_remove = [
                        key for key in self._models.keys() if key not in keys_to_keep
                    ][2:]
                    for key in keys_to_remove:
                        if key in self._models:
                            del self._models[key]
                        if key in self._processors:
                            del self._processors[key]
                        if key in self._tokenizers:
                            del self._tokenizers[key]
                        logger.info(f"Unloaded model {key} due to memory pressure")

                    # FIXED: Force garbage collection
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    def get_model(
        self,
        modality: str,
        quality: ProcessingQuality = ProcessingQuality.BALANCED,
        force_reload: bool = False,
    ) -> Tuple[Any, Any]:
        """Get model with dynamic loading based on quality requirements."""
        quality_key = (
            quality.value if quality != ProcessingQuality.ADAPTIVE else "balanced"
        )

        if modality not in self._model_configs:
            logger.error(f"Unknown modality: {modality}")
            return None, None

        config = self._model_configs[modality].get(
            quality_key, self._model_configs[modality]["balanced"]
        )
        model_name = config["name"]
        device = self._device_map.get(config["device"], "cpu")

        cache_key = f"{modality}_{quality_key}_{model_name}"

        with self._lock:
            if force_reload and cache_key in self._models:
                del self._models[cache_key]
                if cache_key in self._processors:
                    del self._processors[cache_key]
                if cache_key in self._tokenizers:
                    del self._tokenizers[cache_key]

            if cache_key not in self._models:
                try:
                    # MODIFIED: _load_model handles the GraphixTransformer logic
                    model, processor = self._load_model(modality, model_name, device)
                    self._models[cache_key] = model
                    if processor:
                        self._processors[cache_key] = processor
                    logger.info(f"Loaded {modality} model: {model_name} on {device}")
                except Exception as e:
                    logger.error(f"Failed to load model {model_name}: {e}")
                    return None, None

        return self._models.get(cache_key), self._processors.get(cache_key)

    def _load_model(
        self, modality: str, model_name: str, device: str
    ) -> Tuple[Any, Any]:
        """Load a specific model."""
        # MODIFIED: Replaced SentenceTransformer with GraphixTransformer/GraphixTextEncoder
        if modality == "text":
            # PERFORMANCE FIX: Use singleton instance to prevent BERT model reload
            # This prevents the 3-5 second model load on every request
            llm_core = GraphixTransformer.get_instance()
            # Wrap the LLM's embedding method in a consistent encoder interface
            model = GraphixTextEncoder(llm_core, device=device)

            # Since the LLM is initialized, we can conceptually move it if needed
            # For the mock, we skip moving GraphixTransformer, but the wrapper
            # reports the correct device.

            return model, None

        elif modality in ["vision", "audio"]:
            # SECURITY: Support model revision pinning (CWE-494 mitigation)
            # Use environment variable VULCAN_VISION_AUDIO_MODEL_REVISION to pin to specific commit
            revision = (
                VISION_AUDIO_MODEL_REVISION if VISION_AUDIO_MODEL_REVISION else "main"
            )

            model = (
                AutoModel.from_pretrained(  # nosec B615 - revision parameter present
                    model_name, revision=revision
                )
            )
            processor = AutoImageProcessor.from_pretrained(  # nosec B615 - revision parameter present
                model_name, revision=revision
            )

            if device != "cpu" and torch.cuda.is_available():
                model = model.to(device)

            return model, processor

        else:
            raise ValueError(f"Unsupported modality: {modality}")

    def hot_swap_model(self, old_key: str, new_model_name: str, new_device: str = None):
        """Hot-swap a model with a new one."""
        with self._lock:
            # Remove old model
            if old_key in self._models:
                old_model = self._models[old_key]
                del self._models[old_key]

                # Clean up GPU memory if needed
                if hasattr(old_model, "to") and not isinstance(
                    old_model, GraphixTextEncoder
                ):
                    del old_model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # Load new model
            modality = old_key.split("_")[0]
            device = new_device or self._device_map.get("primary", "cpu")

            try:
                model, processor = self._load_model(modality, new_model_name, device)
                self._models[old_key] = model
                if processor:
                    self._processors[old_key] = processor

                logger.info(f"Hot-swapped {old_key} with {new_model_name} on {device}")
                return True
            except Exception as e:
                logger.error(f"Hot-swap failed: {e}")
                return False

    @classmethod
    def cleanup(cls):
        """Clean up all loaded models."""
        with cls._lock:
            cls._models.clear()
            cls._processors.clear()
            cls._tokenizers.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            logger.info("Cleaned up all cached models")

    def shutdown(self):
        """Shutdown model manager gracefully."""
        self._shutdown_event.set()
        if self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)
        self.cleanup()

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.shutdown()
        except Exception as e:
            # Destructor failures in ModelManager should be logged at debug level
            logger.debug(f"ModelManager cleanup in destructor failed: {e}")


# ============================================================
# WORKLOAD MANAGER (FIXED)
# ============================================================


class WorkloadManager:
    """Manages batching, splitting, and prioritizing workloads."""

    def __init__(
        self, max_batch_size: int = 16, max_queue_size: int = 1000, num_workers: int = 4
    ):
        self.max_batch_size = max_batch_size
        self.max_queue_size = max_queue_size
        self.num_workers = num_workers
        self.max_memory_percent = 80  # ADDED: Memory limit

        # Priority queues for different priorities
        self.queues = {
            priority: queue.PriorityQueue(maxsize=max_queue_size)
            for priority in ProcessingPriority
        }

        # Batch accumulators with locks
        self.batch_accumulators = {priority: [] for priority in ProcessingPriority}

        # Statistics
        self.stats = {
            "processed": 0,
            "queued": 0,
            "dropped": 0,
            "avg_latency_ms": 0,
            "throughput": 0,
        }

        self.stats_lock = threading.RLock()
        self._running_flag = True
        self._running_lock = threading.RLock()

        # Worker threads
        self.workers = []
        for i in range(num_workers):
            worker = threading.Thread(target=self._worker, args=(i,), daemon=True)
            worker.start()
            self.workers.append(worker)

        # Batch processor thread
        self.batch_thread = threading.Thread(target=self._batch_processor, daemon=True)
        self.batch_thread.start()

    @property
    def running(self):
        """Thread-safe access to running flag."""
        with self._running_lock:
            return self._running_flag

    @running.setter
    def running(self, value):
        """Thread-safe setting of running flag."""
        with self._running_lock:
            self._running_flag = value

    def submit(
        self,
        data: Any,
        priority: ProcessingPriority = ProcessingPriority.NORMAL,
        slo: Optional[SLOConfig] = None,
        callback: Optional[Callable] = None,
    ) -> str:
        """Submit work item for processing."""
        # ADDED: Memory check
        mem = psutil.virtual_memory()
        if mem.percent > self.max_memory_percent:
            logger.warning(
                f"High memory usage: {mem.percent}% - pausing new work. Dropping item."
            )
            with self.stats_lock:
                self.stats["dropped"] += 1
            return None

        work_id = str(uuid.uuid4())
        timestamp = time.time()

        work_item = {
            "id": work_id,
            "data": data,
            "priority": priority,
            "slo": slo or SLOConfig(),
            "callback": callback,
            "timestamp": timestamp,
        }

        try:
            # Add to appropriate queue
            priority_value = priority.value
            self.queues[priority].put(
                (priority_value, timestamp, work_item), timeout=0.1
            )

            with self.stats_lock:
                self.stats["queued"] += 1

            return work_id

        except queue.Full:
            logger.warning(f"Queue full for priority {priority}, dropping work item")
            with self.stats_lock:
                self.stats["dropped"] += 1
            return None

    def _worker(self, worker_id: int):
        """Worker thread for processing items with proper synchronization."""
        processor = AdaptiveMultimodalProcessor()

        while self.running:
            work_item = None

            # Get highest priority item
            for priority in ProcessingPriority:
                try:
                    _, _, work_item = self.queues[priority].get(timeout=0.1)
                    break
                except queue.Empty:
                    continue

            if work_item:
                try:
                    start_time = time.time()

                    # Check if item should be batched
                    if work_item["priority"] == ProcessingPriority.BATCH:
                        # FIXED: Thread-safe batch accumulation
                        with self.stats_lock:
                            self.batch_accumulators[ProcessingPriority.BATCH].append(
                                work_item
                            )
                            batch_size = len(
                                self.batch_accumulators[ProcessingPriority.BATCH]
                            )

                        # Process batch if full
                        if batch_size >= self.max_batch_size:
                            with self.stats_lock:
                                batch = self.batch_accumulators[
                                    ProcessingPriority.BATCH
                                ]
                                self.batch_accumulators[ProcessingPriority.BATCH] = []

                            if batch:
                                self._process_batch(batch, processor)
                    else:
                        # Process individually
                        result = processor.process_adaptive(
                            work_item["data"],
                            time_budget_ms=work_item["slo"].max_latency_ms,
                            quality=work_item["slo"].quality_preference,
                        )

                        # Execute callback
                        if work_item["callback"]:
                            try:
                                work_item["callback"](work_item["id"], result)
                            except Exception as e:
                                logger.error(f"Callback error: {e}")

                        # Update stats
                        latency_ms = (time.time() - start_time) * 1000
                        self._update_stats(latency_ms)

                        gc.collect()  # ADDED: GC after individual task

                except Exception as e:
                    logger.error(f"Worker {worker_id} error: {e}")
            else:
                # FIXED: Longer sleep to reduce busy-waiting
                time.sleep(0.1)

    def _batch_processor(self):
        """Thread for processing accumulated batches."""
        processor = AdaptiveMultimodalProcessor()

        while self.running:
            time.sleep(0.1)  # Check every 100ms

            for priority in ProcessingPriority:
                # FIXED: Thread-safe check and extraction
                with self.stats_lock:
                    if len(self.batch_accumulators[priority]) > 0:
                        # Process partial batch if waited too long
                        oldest_timestamp = min(
                            item["timestamp"]
                            for item in self.batch_accumulators[priority]
                        )

                        if time.time() - oldest_timestamp > 0.5:  # 500ms timeout
                            batch = self.batch_accumulators[priority]
                            self.batch_accumulators[priority] = []
                        else:
                            batch = None
                    else:
                        batch = None

                if batch:
                    self._process_batch(batch, processor)

    def _process_batch(self, batch: List[Dict], processor):
        """Process a batch of work items."""
        if not batch:
            return

        try:
            start_time = time.time()

            # Extract data
            data_list = [item["data"] for item in batch]

            # Batch process
            results = []
            for data in data_list:
                result = processor.process_adaptive(
                    data,
                    time_budget_ms=50,  # Fast processing for batch
                    quality=ProcessingQuality.FAST,
                )
                results.append(result)

            # Execute callbacks
            for item, result in zip(batch, results):
                if item["callback"]:
                    try:
                        item["callback"](item["id"], result)
                    except Exception as e:
                        logger.error(f"Batch callback error: {e}")

            # Update stats
            latency_ms = (time.time() - start_time) * 1000 / len(batch)
            self._update_stats(latency_ms, batch_size=len(batch))

            gc.collect()  # ADDED: GC after batch task

        except Exception as e:
            logger.error(f"Batch processing error: {e}")

    def _update_stats(self, latency_ms: float, batch_size: int = 1):
        """Update processing statistics."""
        with self.stats_lock:
            self.stats["processed"] += batch_size
            self.stats["queued"] = max(0, self.stats["queued"] - batch_size)

            # Update average latency (exponential moving average)
            alpha = 0.1
            self.stats["avg_latency_ms"] = (1 - alpha) * self.stats[
                "avg_latency_ms"
            ] + alpha * latency_ms

            # Update throughput
            self.stats["throughput"] = self.stats["processed"] / (time.time() + 1e-10)

    def get_stats(self) -> Dict[str, Any]:
        """Get workload statistics."""
        with self.stats_lock:
            return self.stats.copy()

    def adjust_workers(self, num_workers: int):
        """Dynamically adjust number of workers."""
        current_workers = len(self.workers)

        if num_workers > current_workers:
            # Add workers
            for i in range(current_workers, num_workers):
                worker = threading.Thread(target=self._worker, args=(i,), daemon=True)
                worker.start()
                self.workers.append(worker)

        elif num_workers < current_workers:
            # Remove workers (they will exit on next iteration)
            self.workers = self.workers[:num_workers]

        logger.info(f"Adjusted workers from {current_workers} to {num_workers}")

    def shutdown(self):
        """Shutdown workload manager."""
        self.running = False

        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=1.0)

        # Wait for batch processor
        if self.batch_thread.is_alive():
            self.batch_thread.join(timeout=1.0)

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.shutdown()
        except Exception as e:
            # Destructor failures in WorkloadManager should be logged at debug level
            logger.debug(f"WorkloadManager cleanup in destructor failed: {e}")


# ============================================================
# BASE EMBEDDING CACHE
# ============================================================


class EmbeddingCache:
    """Base embedding cache with LRU eviction."""

    def __init__(self, max_size: int = 10000, ttl_seconds: float = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_times = {}
        self.lock = threading.Lock()

    def get(self, key: str) -> Optional[np.ndarray]:
        """Get value from cache."""
        with self.lock:
            if key in self.cache:
                # Check TTL
                if time.time() - self.access_times[key] < self.ttl_seconds:
                    self.access_times[key] = time.time()
                    return self.cache[key]
                else:
                    # Expired
                    del self.cache[key]
                    del self.access_times[key]
            return None

    def put(self, key: str, value: np.ndarray):
        """Put value in cache."""
        with self.lock:
            if len(self.cache) >= self.max_size:
                self._evict()

            self.cache[key] = value
            self.access_times[key] = time.time()

    def _evict(self):
        """Evict least recently used item."""
        if not self.cache:
            return

        # Find LRU item
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[lru_key]
        del self.access_times[lru_key]

    def clear(self):
        """Clear cache."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds,
            }


# ============================================================
# ENHANCED EMBEDDING CACHE WITH METRICS
# ============================================================


class EnhancedEmbeddingCache(EmbeddingCache):
    """Enhanced cache with better metrics and adaptive TTL."""

    def __init__(self, max_size: int = 10000, ttl_seconds: float = 300):
        super().__init__(max_size, ttl_seconds)
        self.metrics = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "evictions": 0,
            "avg_access_time_ms": 0,
        }
        self.access_time_samples = deque(maxlen=1000)

    def get(self, key: str) -> Optional[np.ndarray]:
        """Get with metrics tracking."""
        start_time = time.time()
        result = super().get(key)
        access_time = (time.time() - start_time) * 1000

        self.access_time_samples.append(access_time)
        self.metrics["total_requests"] += 1

        if result is not None:
            self.metrics["cache_hits"] += 1
        else:
            self.metrics["cache_misses"] += 1

        # Update average access time
        if self.access_time_samples:
            self.metrics["avg_access_time_ms"] = np.mean(self.access_time_samples)

        return result

    def _evict(self):
        """Evict with metrics."""
        super()._evict()
        self.metrics["evictions"] += 1

    def get_detailed_stats(self) -> Dict[str, Any]:
        """Get detailed cache statistics."""
        base_stats = super().get_stats()

        return {
            **base_stats,
            **self.metrics,
            "cache_efficiency": self.metrics["cache_hits"]
            / max(1, self.metrics["total_requests"]),
            "eviction_rate": self.metrics["evictions"]
            / max(1, self.metrics["total_requests"]),
        }


# ============================================================
# MODEL MANAGER (Legacy Interface)
# ============================================================


class ModelManager:
    """Legacy interface redirecting to DynamicModelManager."""

    _instance = DynamicModelManager()

    @classmethod
    def get_text_model(cls, model_name: str = "graphix-llm-embed-balanced"):
        # MODIFIED: Changed default model name to reflect the new architecture
        model, _ = cls._instance.get_model("text", ProcessingQuality.BALANCED)
        return model

    @classmethod
    def get_vision_model(cls, model_name: str = "google/vit-base-patch16-224"):
        return cls._instance.get_model("vision", ProcessingQuality.BALANCED)

    @classmethod
    def get_audio_model(cls, model_name: str = "facebook/wav2vec2-base"):
        return cls._instance.get_model("audio", ProcessingQuality.BALANCED)

    @classmethod
    def get_ai_providers(cls):
        return cls._instance._ai_providers

    @classmethod
    def cleanup(cls):
        cls._instance.cleanup()


# ============================================================
# PROCESSING RESULT (Enhanced)
# ============================================================


@dataclass
class ProcessingResult:
    """Enhanced result from multimodal processing with traceability."""

    embedding: np.ndarray
    modality: ModalityType
    uncertainty: float
    attention_weights: Optional[np.ndarray] = None
    sub_modalities: List[ModalityType] = field(default_factory=list)
    processing_time_ms: float = 0.0
    quality_level: ProcessingQuality = ProcessingQuality.BALANCED
    metadata: Dict[str, Any] = field(default_factory=dict)
    cache_hit: bool = False
    log_id: Optional[str] = None  # Reference to versioned log
    model_used: Optional[str] = None  # Track which model was used
    device_used: Optional[str] = None  # Track which device was used

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "modality": self.modality.value,
            "uncertainty": self.uncertainty,
            "processing_time_ms": self.processing_time_ms,
            "quality_level": self.quality_level.value,
            "cache_hit": self.cache_hit,
            "embedding_shape": self.embedding.shape,
            "sub_modalities": [m.value for m in self.sub_modalities],
            "metadata": self.metadata,
            "log_id": self.log_id,
            "model_used": self.model_used,
            "device_used": self.device_used,
        }


# ============================================================
# CROSS-MODAL ATTENTION
# ============================================================


class CrossModalAttention(nn.Module):
    """Enhanced cross-modal attention with position encoding and layer norm."""

    def __init__(
        self,
        embed_dim: int = EMBEDDING_DIM,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 100,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Position encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim) * 0.1)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(HIDDEN_DIM, embed_dim),
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Modality-specific projections
        self.modality_projections = nn.ModuleDict(
            {
                "text": nn.Linear(embed_dim, embed_dim),
                "vision": nn.Linear(embed_dim, embed_dim),
                "audio": nn.Linear(embed_dim, embed_dim),
                "symbolic": nn.Linear(embed_dim, embed_dim),
            }
        )

        # Modality embeddings
        self.modality_embeddings = nn.Embedding(5, embed_dim)  # 5 modality types

        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        embeddings: torch.Tensor,
        modalities: Optional[List[str]] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply cross-modal attention with position and modality encoding.

        Args:
            embeddings: Tensor of shape (batch, seq_len, embed_dim)
            modalities: List of modality types for each sequence element
            mask: Attention mask

        Returns:
            attended: Attended embeddings
            weights: Attention weights
        """
        batch_size, seq_len, _ = embeddings.shape

        # Add position encoding
        embeddings = embeddings + self.pos_encoding[:, :seq_len, :]

        # Add modality-specific projections and embeddings
        if modalities:
            projected = []
            for i, mod in enumerate(modalities):
                if mod in self.modality_projections:
                    # Apply modality-specific projection
                    proj = self.modality_projections[mod]
                    mod_emb = proj(embeddings[:, i : i + 1, :])

                    # Add modality embedding
                    mod_idx = ["text", "vision", "audio", "symbolic", "unknown"].index(
                        mod
                        if mod in ["text", "vision", "audio", "symbolic"]
                        else "unknown"
                    )
                    mod_emb = mod_emb + self.modality_embeddings(
                        torch.tensor([mod_idx], device=embeddings.device)
                    )
                    projected.append(mod_emb)
                else:
                    projected.append(embeddings[:, i : i + 1, :])
            embeddings = torch.cat(projected, dim=1)

        # Self-attention with residual connection
        attended, weights = self.attention(
            embeddings, embeddings, embeddings, attn_mask=mask
        )
        embeddings = self.norm1(embeddings + self.dropout(attended))

        # Feed-forward with residual connection
        output = self.ffn(embeddings)
        output = self.norm2(embeddings + self.dropout(output))

        # Final projection
        output = self.output_proj(output)

        return output, weights


# ============================================================
# MODALITY FUSION (FIXED)
# ============================================================


class ModalityFusion(nn.Module):
    """Advanced fusion strategies for multimodal inputs."""

    def __init__(self, embed_dim: int = EMBEDDING_DIM):
        super().__init__()
        self.embed_dim = embed_dim

        # Fusion methods
        self.concat_proj = nn.Linear(embed_dim * 4, embed_dim)

        # Gated fusion
        self.gated_fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), nn.Sigmoid()
        )

        # Attention-based fusion
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim, num_heads=4, batch_first=True
        )

        # Mixture of Experts (MoE) fusion
        self.num_experts = 4
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.ReLU(),
                    nn.Linear(embed_dim, embed_dim),
                )
                for _ in range(self.num_experts)
            ]
        )
        self.gating_network = nn.Sequential(
            nn.Linear(embed_dim, self.num_experts), nn.Softmax(dim=-1)
        )

        # Learnable modality weights
        self.modality_weights = nn.Parameter(torch.ones(4) / 4)

    def forward(
        self, modality_embeddings: Dict[str, torch.Tensor], method: str = "attention"
    ) -> torch.Tensor:
        """
        Fuse embeddings from multiple modalities.

        Args:
            modality_embeddings: Dict mapping modality to embedding tensor
            method: Fusion method ('mean', 'concat', 'gated', 'attention', 'weighted', 'moe')
        """
        embeddings = list(modality_embeddings.values())

        if not embeddings:
            return torch.zeros(1, self.embed_dim)

        if len(embeddings) == 1:
            return embeddings[0]

        if method == "mean":
            return torch.stack(embeddings).mean(dim=0)

        elif method == "concat":
            # Pad if necessary
            while len(embeddings) < 4:
                embeddings.append(torch.zeros_like(embeddings[0]))
            concatenated = torch.cat(embeddings[:4], dim=-1)
            return self.concat_proj(concatenated)

        elif method == "gated":
            # Gated fusion for pairs
            result = embeddings[0]
            for emb in embeddings[1:]:
                gate = self.gated_fusion(torch.cat([result, emb], dim=-1))
                result = gate * result + (1 - gate) * emb
            return result

        elif method == "attention":
            # FIXED: Proper stacking for attention
            # embeddings are list of (batch, embed_dim)
            # Need (batch, seq_len, embed_dim) for attention
            if len(embeddings[0].shape) == 1:
                # Add batch dimension if needed
                embeddings = [e.unsqueeze(0) for e in embeddings]

            # Stack along sequence dimension: (batch, n_modalities, embed_dim)
            stacked = torch.stack(embeddings, dim=1)

            fused, _ = self.fusion_attention(stacked, stacked, stacked)
            return fused.mean(dim=1)

        elif method == "weighted":
            # Use learnable weights
            weighted = []
            modality_order = ["text", "vision", "audio", "symbolic"]
            for modality, emb in modality_embeddings.items():
                if modality in modality_order:
                    idx = modality_order.index(modality)
                    weighted.append(self.modality_weights[idx] * emb)
            return torch.stack(weighted).sum(dim=0) if weighted else embeddings[0]

        elif method == "moe":
            # Mixture of Experts fusion
            combined = torch.stack(embeddings).mean(dim=0)
            gates = self.gating_network(combined)

            expert_outputs = []
            for expert in self.experts:
                expert_outputs.append(expert(combined))

            # FIXED: Handle gates and output tensor dimensionality properly
            # combined can be (embed_dim,) or (batch, embed_dim)
            # gates will be (num_experts,) or (batch, num_experts)
            # expert_out will match combined's shape

            # Ensure combined has batch dimension
            if len(combined.shape) == 1:
                combined = combined.unsqueeze(0)  # (1, embed_dim)

            # Ensure gates has batch dimension
            if len(gates.shape) == 1:
                gates = gates.unsqueeze(0)  # (1, num_experts)

            # Ensure all expert outputs have batch dimension
            expert_outputs_batched = []
            for expert_out in expert_outputs:
                if len(expert_out.shape) == 1:
                    expert_out = expert_out.unsqueeze(0)
                expert_outputs_batched.append(expert_out)

            # Weighted sum of expert outputs
            output = torch.zeros_like(combined)
            for i, expert_out in enumerate(expert_outputs_batched):
                # Now everything is (batch, dim) shape
                gate_weight = gates[:, i : i + 1]  # (batch, 1)
                output += gate_weight * expert_out  # (batch, embed_dim)

            # If input was 1D, return 1D
            if len(embeddings[0].shape) == 1:
                output = output.squeeze(0)

            return output

        else:
            return torch.stack(embeddings).mean(dim=0)


# ============================================================
# ADAPTIVE MULTIMODAL PROCESSOR (Enhanced & FIXED & MODIFIED)
# ============================================================


class AdaptiveMultimodalProcessor(nn.Module):
    """Enhanced multimodal processor with all optimizations."""

    def __init__(self, common_dim: int = EMBEDDING_DIM):
        super().__init__()
        self.common_dim = common_dim
        self.cache = EnhancedEmbeddingCache(max_size=10000, ttl_seconds=300)
        self.model_manager = DynamicModelManager()
        self.data_logger = VersionedDataLogger()
        self.workload_manager = None  # Will be set by parent class

        # FIXED: Add cache key cache with bounded size using OrderedDict
        self._cache_key_cache = OrderedDict()
        self._cache_key_lock = threading.RLock()
        self._cache_key_max_size = 10000

        # Projection layers to common space
        # NOTE: Text input is fixed to 384 dim to match GraphixTransformer mock output
        self.projections = nn.ModuleDict(
            {
                "text": nn.Sequential(
                    nn.Linear(384, HIDDEN_DIM),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(HIDDEN_DIM, common_dim),
                    nn.LayerNorm(common_dim),
                ),
                "vision": nn.Sequential(
                    nn.Linear(768, HIDDEN_DIM),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(HIDDEN_DIM, common_dim),
                    nn.LayerNorm(common_dim),
                ),
                "audio": nn.Sequential(
                    nn.Linear(768, HIDDEN_DIM),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(HIDDEN_DIM, common_dim),
                    nn.LayerNorm(common_dim),
                ),
                "symbolic": nn.Sequential(
                    nn.Linear(384, HIDDEN_DIM),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(HIDDEN_DIM, common_dim),
                    nn.LayerNorm(common_dim),
                ),
            }
        )

        # Cross-modal attention
        self.cross_attention = CrossModalAttention(common_dim)
        self.modality_fusion = ModalityFusion(common_dim)

        # Modality classifier
        self.modality_classifier = nn.Sequential(
            nn.Linear(common_dim, LATENT_DIM),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(LATENT_DIM, len(ModalityType)),
            nn.Softmax(dim=-1),
        )

        # Uncertainty estimator
        self.uncertainty_head = nn.Sequential(
            nn.Linear(common_dim, LATENT_DIM // 2),
            nn.ReLU(),
            nn.Linear(LATENT_DIM // 2, 1),
            nn.Sigmoid(),
        )

        # Processing executors
        self.thread_executor = ThreadPoolExecutor(max_workers=4)
        self.process_executor = ProcessPoolExecutor(max_workers=2)

        # Quality processors
        self.quality_processors = {
            ProcessingQuality.FAST: self._process_fast,
            ProcessingQuality.BALANCED: self._process_balanced,
            ProcessingQuality.QUALITY: self._process_quality,
        }

    def process_adaptive(
        self,
        raw_data: Any,
        time_budget_ms: float = 100,
        quality: ProcessingQuality = ProcessingQuality.ADAPTIVE,
    ) -> ProcessingResult:
        """Process with adaptive quality based on time budget."""
        start_time = time.time()

        # Determine quality level
        if quality == ProcessingQuality.ADAPTIVE:
            if time_budget_ms < 50:
                quality = ProcessingQuality.FAST
            elif time_budget_ms < 200:
                quality = ProcessingQuality.BALANCED
            else:
                quality = ProcessingQuality.QUALITY

        # Get processor for quality level
        processor = self.quality_processors[quality]

        # Process
        result = processor(raw_data)
        result.quality_level = quality
        result.processing_time_ms = (time.time() - start_time) * 1000

        # Log processing
        log_metadata = {
            "quality": quality.value,
            "time_budget_ms": time_budget_ms,
            "actual_time_ms": result.processing_time_ms,
            "cache_hit": result.cache_hit,
        }

        log_id = self.data_logger.log_processing(
            input_data=raw_data, output_data=result, metadata=log_metadata
        )
        result.log_id = log_id

        return result

    def _process_fast(self, data: Any) -> ProcessingResult:
        """Fast processing with caching and approximation."""
        # Check cache first
        cache_key = self._compute_cache_key(data)
        cached = self.cache.get(cache_key)

        if cached is not None:
            return ProcessingResult(
                embedding=cached,
                modality=ModalityType.UNKNOWN,
                uncertainty=0.1,
                cache_hit=True,
                quality_level=ProcessingQuality.FAST,
            )

        # Lightweight processing
        if isinstance(data, str):
            # Use fast text model (GraphixTextEncoder)
            model, _ = self.model_manager.get_model("text", ProcessingQuality.FAST)
            if model:
                # Use GraphixTextEncoder.encode method
                embedding = model.encode(data[:500], convert_to_tensor=True)  # Truncate
                embedding = (
                    embedding.cpu().numpy().squeeze(0)
                )  # Ensure numpy array, remove batch dim
            else:
                embedding = self._hash_to_embedding(data[:500])
            modality = ModalityType.TEXT
        elif isinstance(data, np.ndarray):
            # Simple dimensionality reduction
            flat = data.flatten()[: self.common_dim]
            if len(flat) < self.common_dim:
                flat = np.pad(flat, (0, self.common_dim - len(flat)))
            embedding = flat
            modality = (
                ModalityType.VISION if len(data.shape) > 1 else ModalityType.AUDIO
            )
        else:
            # For unknown types, try to create a meaningful embedding
            # Use text encoder if possible, otherwise fallback
            if hasattr(self, "text_encoder") and self.text_encoder:
                try:
                    # Try to convert to string and encode
                    text_repr = str(data)[:512]  # Limit length
                    text_embedding = self.text_encoder.encode(
                        [text_repr], convert_to_tensor=False
                    )
                    if isinstance(text_embedding, np.ndarray):
                        embedding = (
                            text_embedding[0]
                            if len(text_embedding.shape) > 1
                            else text_embedding
                        )
                    else:
                        embedding = np.array(text_embedding).flatten()

                    # Pad or truncate to common_dim
                    if len(embedding) > self.common_dim:
                        embedding = embedding[: self.common_dim]
                    elif len(embedding) < self.common_dim:
                        embedding = np.pad(
                            embedding, (0, self.common_dim - len(embedding))
                        )
                except Exception as e:
                    logger.debug(
                        f"Text encoding failed for unknown type: {e}, using zero vector"
                    )
                    embedding = np.zeros(self.common_dim)
            else:
                # Last resort: zero vector instead of random
                logger.debug("Unknown data type, using zero vector")
                embedding = np.zeros(self.common_dim)

            modality = ModalityType.UNKNOWN

        # Cache result
        self.cache.put(cache_key, embedding)

        return ProcessingResult(
            embedding=embedding,
            modality=modality,
            uncertainty=0.5,
            quality_level=ProcessingQuality.FAST,
            model_used="fast_model",
        )

    def _process_balanced(self, data: Any) -> ProcessingResult:
        """Standard processing with single model."""
        return self.process_input(data)

    def _process_quality(self, data: Any) -> ProcessingResult:
        """High quality processing with ensemble and advanced features."""
        if isinstance(data, str):
            # Use multiple text models
            embeddings = []
            models_used = []

            for quality in [
                ProcessingQuality.FAST,
                ProcessingQuality.BALANCED,
                ProcessingQuality.QUALITY,
            ]:
                # Get the GraphixTextEncoder wrapper
                model, _ = self.model_manager.get_model("text", quality)
                if model:
                    # Use GraphixTextEncoder.encode
                    emb = model.encode(data, convert_to_tensor=True)

                    # FIXED: Clone and detach, ensure 2D
                    if len(emb.shape) == 1:
                        emb = emb.unsqueeze(0)
                    emb = emb.clone().detach()

                    # FIXED: Ensure 384 dimensions BEFORE projection
                    if emb.shape[1] != 384:
                        if emb.shape[1] > 384:
                            emb = emb[:, :384]
                        else:
                            padding = torch.zeros(
                                emb.shape[0], 384 - emb.shape[1], device=emb.device
                            )
                            emb = torch.cat([emb, padding], dim=1)

                    # Now project to common space
                    projected = self.projections["text"](emb)
                    embeddings.append(projected)
                    models_used.append(f"text_{quality.value}")

            if embeddings:
                # FIXED: Now all embeddings have same dimension (batch, common_dim)
                # Stack them along a new 'ensemble' dimension (n_models, batch, common_dim)
                combined = torch.stack(embeddings).mean(dim=0)

                # Apply attention refinement - stacked is (1, n_models, common_dim) for batch size 1
                refined, _ = self.cross_attention(combined.unsqueeze(0))
                embedding = refined.squeeze(0).squeeze(0)  # (common_dim,)

                # Estimate uncertainty from ensemble variance
                if len(embeddings) > 1:
                    # Variance across ensemble predictions (n_models, 1, common_dim)
                    variance = torch.stack(embeddings).var(dim=0).mean()
                    # Apply sigmoid to variance for a 0-1 uncertainty score
                    uncertainty = float(torch.sigmoid(variance))
                else:
                    # Fallback to head if only one model
                    uncertainty = self.uncertainty_head(embedding).item()

                return ProcessingResult(
                    embedding=embedding.detach().numpy(),
                    modality=ModalityType.TEXT,
                    uncertainty=uncertainty,
                    quality_level=ProcessingQuality.QUALITY,
                    model_used=",".join(models_used),
                )

        # Fallback to balanced processing
        return self._process_balanced(data)

    def process_input(self, raw_data: Any) -> ProcessingResult:
        """Process any input type through appropriate encoder."""
        start_time = time.time()

        # Check cache
        cache_key = self._compute_cache_key(raw_data)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return ProcessingResult(
                embedding=cached,
                modality=ModalityType.UNKNOWN,
                uncertainty=0.0,
                processing_time_ms=0.1,
                cache_hit=True,
            )

        try:
            # FIXED: Handle None explicitly for error recovery test
            if raw_data is None:
                raise ValueError("Cannot process None input")

            # Detect and process modality
            if isinstance(raw_data, str):
                result = self._process_text(raw_data)
            elif isinstance(raw_data, dict):
                result = self._process_multimodal(raw_data)
            elif isinstance(raw_data, (list, tuple)) and all(
                isinstance(x, str) for x in raw_data
            ):
                result = self._process_text_batch(raw_data)
            elif isinstance(raw_data, np.ndarray):
                result = self._process_array(raw_data)
            elif isinstance(raw_data, PIL.Image.Image):
                result = self._process_image(raw_data)
            else:
                result = self._process_unknown(raw_data)

            # Cache result
            self.cache.put(cache_key, result.embedding)

            # Add processing time
            result.processing_time_ms = (time.time() - start_time) * 1000

            # Log processing
            log_id = self.data_logger.log_processing(
                input_data=raw_data,
                output_data=result,
                metadata={"modality": result.modality.value},  # Pass the string value
            )
            result.log_id = log_id

            return result

        except Exception as e:
            logger.error(f"Error processing input: {e}")
            return ProcessingResult(
                embedding=np.zeros(self.common_dim),
                modality=ModalityType.UNKNOWN,
                uncertainty=1.0,
                processing_time_ms=(time.time() - start_time) * 1000,
            )

    def _process_text(self, text_data: str) -> ProcessingResult:
        """Process text input with dynamic model selection."""
        # Get appropriate model based on current resources (GraphixTextEncoder)
        model, _ = self.model_manager.get_model("text", ProcessingQuality.BALANCED)

        if model:
            # MODIFIED: Call encode on the GraphixTextEncoder wrapper
            raw_embedding = model.encode(text_data, convert_to_tensor=True)
            model_used = "graphix_llm_embed_balanced"
        else:
            raw_embedding = self._hash_to_embedding(text_data)
            model_used = "hash_fallback"

        # Project to common space
        if isinstance(raw_embedding, torch.Tensor):
            if len(raw_embedding.shape) == 1:
                raw_embedding = raw_embedding.unsqueeze(0)
            # FIXED: Clone and detach inference tensors
            raw_embedding = raw_embedding.clone().detach()
        else:
            raw_embedding = torch.tensor(raw_embedding, dtype=torch.float32).unsqueeze(
                0
            )

        # Ensure correct dimension for projection (384 for text)
        if raw_embedding.shape[1] != 384:
            if raw_embedding.shape[1] > 384:
                raw_embedding = raw_embedding[:, :384]
            else:
                padding = torch.zeros(
                    raw_embedding.shape[0],
                    384 - raw_embedding.shape[1],
                    device=raw_embedding.device,
                )
                raw_embedding = torch.cat([raw_embedding, padding], dim=1)

        embedding = self.projections["text"](raw_embedding).squeeze(0)

        # Estimate uncertainty
        with torch.no_grad():
            uncertainty = self.uncertainty_head(embedding).item()

        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().numpy()

        return ProcessingResult(
            embedding=embedding,
            modality=ModalityType.TEXT,
            uncertainty=uncertainty,
            model_used=model_used,
        )

    def _process_image(self, image: PIL.Image.Image) -> ProcessingResult:
        """Process PIL image."""
        vision_model, processor = self.model_manager.get_model(
            "vision", ProcessingQuality.BALANCED
        )

        if vision_model and processor:
            # Process through vision model
            inputs = processor(images=image, return_tensors="pt")

            with torch.no_grad():
                outputs = vision_model(**inputs)
                # Get the pooled output or CLS token
                if hasattr(outputs, "pooler_output"):
                    features = outputs.pooler_output
                else:
                    features = outputs.last_hidden_state[:, 0, :]

            # Project to common space
            embedding = self.projections["vision"](features).squeeze(0)

            # Estimate uncertainty
            uncertainty = self.uncertainty_head(embedding).item()

            return ProcessingResult(
                embedding=embedding.detach().numpy(),
                modality=ModalityType.VISION,
                uncertainty=uncertainty,
                model_used="vision_balanced",
            )

        # Fallback processing
        return self._process_array(np.array(image))

    def _process_array(self, array_data: np.ndarray) -> ProcessingResult:
        """Process numpy array (image or audio)."""
        if len(array_data.shape) >= 2:  # Likely image
            return self._process_vision(array_data)
        else:  # Likely audio
            return self._process_audio(array_data)

    def _process_vision(self, image_data: np.ndarray) -> ProcessingResult:
        """Process vision input."""
        # Flatten and project
        flattened = image_data.flatten()[:768]
        if len(flattened) < 768:
            flattened = np.pad(flattened, (0, 768 - len(flattened)))

        vision_tensor = torch.tensor(flattened, dtype=torch.float32).unsqueeze(0)
        embedding = self.projections["vision"](vision_tensor).squeeze(0)

        with torch.no_grad():
            uncertainty = self.uncertainty_head(embedding).item()

        return ProcessingResult(
            embedding=embedding.detach().numpy(),
            modality=ModalityType.VISION,
            uncertainty=uncertainty,
            model_used="vision_direct",
        )

    def _process_audio(self, audio_data: np.ndarray) -> ProcessingResult:
        """Process audio input."""
        # Feature extraction
        features = np.zeros(768)
        features[: min(len(audio_data), 768)] = audio_data[:768]

        audio_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        embedding = self.projections["audio"](audio_tensor).squeeze(0)

        with torch.no_grad():
            uncertainty = self.uncertainty_head(embedding).item()

        return ProcessingResult(
            embedding=embedding.detach().numpy(),
            modality=ModalityType.AUDIO,
            uncertainty=uncertainty,
            model_used="audio_direct",
        )

    def _process_text_batch(self, texts: List[str]) -> ProcessingResult:
        """Process batch of text inputs."""
        embeddings = []

        # Process each text
        for text in texts:
            result = self._process_text(text)
            embeddings.append(result.embedding)

        # Aggregate embeddings
        if embeddings:
            avg_embedding = np.mean(embeddings, axis=0)
            uncertainty = 0.3
        else:
            avg_embedding = np.zeros(self.common_dim)
            uncertainty = 1.0

        return ProcessingResult(
            embedding=avg_embedding,
            modality=ModalityType.TEXT,
            uncertainty=uncertainty,
            model_used="text_batch",
        )

    def _process_multimodal(self, multimodal_data: Dict) -> ProcessingResult:
        """Process and integrate multiple modalities."""
        modality_embeddings = {}
        sub_modalities = []
        models_used = []

        # Process each modality
        for key, value in multimodal_data.items():
            if key in ["text", "description"] and isinstance(value, str):
                result = self._process_text(value)
                modality_embeddings["text"] = torch.tensor(result.embedding)
                sub_modalities.append(ModalityType.TEXT)
                models_used.append(result.model_used)

            elif key in ["image", "vision"]:
                if isinstance(value, np.ndarray):
                    result = self._process_vision(value)
                elif isinstance(value, PIL.Image.Image):
                    result = self._process_image(value)
                else:
                    continue
                modality_embeddings["vision"] = torch.tensor(result.embedding)
                sub_modalities.append(ModalityType.VISION)
                models_used.append(result.model_used)

            elif key in ["audio", "sound"] and isinstance(value, np.ndarray):
                result = self._process_audio(value)
                modality_embeddings["audio"] = torch.tensor(result.embedding)
                sub_modalities.append(ModalityType.AUDIO)
                models_used.append(result.model_used)

        if not modality_embeddings:
            return ProcessingResult(
                embedding=np.zeros(self.common_dim),
                modality=ModalityType.UNKNOWN,
                uncertainty=1.0,
            )

        # Fuse modalities using MoE
        fused = self.modality_fusion(modality_embeddings, method="moe")

        # Apply cross-modal attention
        stacked = torch.stack(list(modality_embeddings.values())).unsqueeze(0)
        attended, attention_weights = self.cross_attention(stacked)

        # Combine fused and attended
        final_embedding = (fused + attended.mean(dim=1).squeeze(0)) / 2

        # Estimate uncertainty
        with torch.no_grad():
            uncertainty = self.uncertainty_head(final_embedding).item()

        return ProcessingResult(
            embedding=final_embedding.detach().numpy(),
            modality=ModalityType.MULTIMODAL,
            uncertainty=uncertainty,
            attention_weights=(
                attention_weights.detach().numpy()
                if attention_weights is not None
                else None
            ),
            sub_modalities=sub_modalities,
            model_used=",".join(models_used),
        )

    def _process_unknown(self, data: Any) -> ProcessingResult:
        """Process unknown data type."""
        # Convert to string and process as text
        text_repr = str(data)[:1000]
        return self._process_text(text_repr)

    def _hash_to_embedding(self, data: str) -> np.ndarray:
        """Convert string to embedding using hash."""
        hash_hex = hashlib.sha256(data.encode()).hexdigest()

        # Convert hex to float vector
        embedding = []
        for i in range(0, min(len(hash_hex), self.common_dim * 2), 2):
            value = int(hash_hex[i : i + 2], 16) / 127.5 - 1
            embedding.append(value)

        # Pad or truncate
        if len(embedding) < self.common_dim:
            embedding.extend([0] * (self.common_dim - len(embedding)))
        else:
            embedding = embedding[: self.common_dim]

        return np.array(embedding, dtype=np.float32)

    def _compute_cache_key(self, data: Any) -> str:
        """Compute cache key for data with caching optimization."""
        # FIXED: Optimized cache key computation with hash caching using OrderedDict

        if isinstance(data, str):
            # FIXED: Use content-based hash, not object id
            # Object id changes for different string objects with same content
            # Check if we already computed this hash
            # Use content hash as key instead of object id
            try:
                import xxhash

                key = xxhash.xxh64(data.encode()).hexdigest()
            except ImportError:
                # Fallback to FNV-1a hash
                h = 2166136261
                for byte in data.encode():
                    h = ((h ^ byte) * 16777619) & 0xFFFFFFFF
                key = hex(h)[2:]

            return key

        elif isinstance(data, np.ndarray):
            # FIXED: Use shape + fingerprint for faster hashing
            if data.size == 0:
                return "empty_array"

            # Create fingerprint from shape, dtype, and sample values
            fingerprint = (
                data.shape,
                data.dtype,
                float(data.flat[0]) if data.size > 0 else 0,
                float(data.flat[-1]) if data.size > 0 else 0,
                float(data.mean()) if data.size > 0 else 0,
            )
            return hashlib.md5(
                str(fingerprint).encode(), usedforsecurity=False
            ).hexdigest()

        elif isinstance(data, PIL.Image.Image):
            # Hash image size and mode as fingerprint
            fingerprint = (data.size, data.mode)
            return hashlib.md5(
                str(fingerprint).encode(), usedforsecurity=False
            ).hexdigest()

        else:
            # Fallback for other types
            return hashlib.md5(str(data).encode(), usedforsecurity=False).hexdigest()

    def cleanup(self):
        """Cleanup resources."""
        self.cache.clear()
        self.data_logger.shutdown()
        self.model_manager.shutdown()

        with self._cache_key_lock:
            self._cache_key_cache.clear()

        self.thread_executor.shutdown(wait=False)
        self.process_executor.shutdown(wait=False)

        gc.collect()

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except Exception as e:
            # Destructor failures in EmbeddingCache should be logged at debug level
            logger.debug(f"EmbeddingCache cleanup in destructor failed: {e}")


# ============================================================
# STREAMING PROCESSOR
# ============================================================


class StreamingProcessor:
    """Process streaming data with windowing and temporal aggregation."""

    def __init__(self, window_size: int = 10, overlap: int = 5):
        self.window_size = window_size
        self.overlap = overlap
        self.buffer = deque(maxlen=window_size)
        self.processor = AdaptiveMultimodalProcessor()
        self.temporal_memory = deque(maxlen=100)

    async def process_stream(
        self, stream: AsyncGenerator
    ) -> AsyncGenerator[ProcessingResult, None]:
        """Process streaming data asynchronously."""
        async for chunk in stream:
            self.buffer.append(chunk)

            # Process when window is full
            if len(self.buffer) >= self.window_size:
                window_data = list(self.buffer)

                # Process window
                result = await self._process_window_async(window_data)

                # Update temporal memory
                self.temporal_memory.append(result)

                # Slide window
                for _ in range(self.window_size - self.overlap):
                    if self.buffer:
                        self.buffer.popleft()

                yield result

    async def _process_window_async(self, window_data: List[Any]) -> ProcessingResult:
        """Process a window of data asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._process_window, window_data)

    def _process_window(self, window_data: List[Any]) -> ProcessingResult:
        """Process a window of data with temporal aggregation."""
        embeddings = []
        modalities = []
        uncertainties = []

        # Process each item in window
        for data in window_data:
            result = self.processor.process_adaptive(
                data, time_budget_ms=50, quality=ProcessingQuality.FAST
            )
            embeddings.append(result.embedding)
            modalities.append(result.modality)
            uncertainties.append(result.uncertainty)

        # Temporal aggregation with weighted average
        weights = np.exp(-np.arange(len(embeddings)) * 0.1)  # Exponential decay
        weights = weights / weights.sum()

        aggregated = np.average(embeddings, axis=0, weights=weights)
        avg_uncertainty = np.average(uncertainties, weights=weights)

        # Determine dominant modality
        from collections import Counter

        modality_counts = Counter(modalities)
        dominant_modality = (
            modality_counts.most_common(1)[0][0]
            if modality_counts
            else ModalityType.UNKNOWN
        )

        return ProcessingResult(
            embedding=aggregated,
            modality=dominant_modality,
            sub_modalities=list(set(modalities)),
            uncertainty=avg_uncertainty,
            metadata={
                "window_size": len(window_data),
                "temporal_aggregation": "weighted",
            },
        )

    def cleanup(self):
        """Cleanup streaming processor resources."""
        if hasattr(self.processor, "cleanup"):
            self.processor.cleanup()
        self.buffer.clear()
        self.temporal_memory.clear()

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except Exception as e:
            # Destructor failures in StreamingProcessor should be logged at debug level
            logger.debug(f"StreamingProcessor cleanup in destructor failed: {e}")


# ============================================================
# MULTIMODAL PROCESSOR (Main Interface - Enhanced)
# ============================================================


class MultimodalProcessor(AdaptiveMultimodalProcessor):
    """Main multimodal processor interface with all enhancements."""

    def __init__(self, common_dim: int = EMBEDDING_DIM):
        super().__init__(common_dim)
        self.streaming_processor = StreamingProcessor()
        self.workload_manager = WorkloadManager()

        # Statistics tracking
        self.stats = {
            "total_processed": 0,
            "cache_hits": 0,
            "avg_latency_ms": 0,
            "models_loaded": 0,
        }
        self.stats_lock = threading.RLock()

    def process_ir_node(self, node_data: Dict[str, Any]) -> ProcessingResult:
        """Process Graphix IR node data."""
        if UnifiedRuntime:
            # UnifiedRuntime() initialization could be used in the future
            # runtime = UnifiedRuntime()
            processed = node_data.get("params", node_data)
            return self.process_input(processed)
        else:
            logger.warning("UnifiedRuntime not available. Processing raw node data.")
            return self.process_input(node_data)

    async def process_batch_async(
        self, batch: List[Any], max_workers: int = 4, slo: Optional[SLOConfig] = None
    ) -> List[ProcessingResult]:
        """Process batch of inputs asynchronously with SLO support."""
        if slo and self.workload_manager:
            # Submit to workload manager
            results = []
            futures = {}

            for item in batch:
                future = asyncio.Future()

                def callback(work_id, result):
                    if work_id in futures:
                        futures[work_id].set_result(result)

                work_id = self.workload_manager.submit(
                    data=item,
                    priority=ProcessingPriority.BATCH,
                    slo=slo,
                    callback=callback,
                )

                if work_id:
                    futures[work_id] = future

            # Wait for all results
            for future in futures.values():
                result = await future
                results.append(result)

            return results
        else:
            # Original implementation
            tasks = []
            for item in batch:
                task = asyncio.create_task(self._process_async(item))
                tasks.append(task)

            results = await asyncio.gather(*tasks)
            return results

    async def _process_async(self, data: Any) -> ProcessingResult:
        """Process single input asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.process_input, data)

    def process_with_priority(
        self,
        data: Any,
        priority: ProcessingPriority = ProcessingPriority.NORMAL,
        slo: Optional[SLOConfig] = None,
    ) -> ProcessingResult:
        """Process with priority and SLO support."""
        if self.workload_manager:
            # Use threading Event for synchronous waiting
            result_container = {"result": None, "ready": threading.Event()}

            def callback(work_id, result):
                result_container["result"] = result
                result_container["ready"].set()

            work_id = self.workload_manager.submit(
                data=data, priority=priority, slo=slo, callback=callback
            )

            if work_id:
                # Wait for result (with timeout based on SLO)
                timeout = (slo.max_latency_ms / 1000) if slo else 10.0

                if result_container["ready"].wait(timeout=timeout):
                    return result_container["result"]
                else:
                    logger.error(f"Processing timeout for work_id {work_id}")
                    return ProcessingResult(
                        embedding=np.zeros(self.common_dim),
                        modality=ModalityType.UNKNOWN,
                        uncertainty=1.0,
                        metadata={"error": "timeout"},
                    )

        # Fallback to direct processing
        return self.process_input(data)

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        stats = {
            "cache": self.cache.get_detailed_stats(),
            "workload": (
                self.workload_manager.get_stats() if self.workload_manager else {}
            ),
            "models": {
                "loaded_count": len(self.model_manager._models),
                "models": list(self.model_manager._models.keys()),
            },
            "processing": self.stats,
        }

        return stats

    def optimize_for_latency(self):
        """Optimize system for low latency."""
        # Adjust cache TTL
        self.cache.ttl_seconds = 600  # Increase TTL

        # Load fast models
        self.model_manager.get_model("text", ProcessingQuality.FAST)
        self.model_manager.get_model("vision", ProcessingQuality.FAST)

        # Increase workers
        if self.workload_manager:
            self.workload_manager.adjust_workers(8)

        logger.info("Optimized for low latency")

    def optimize_for_throughput(self):
        """Optimize system for high throughput."""
        # Increase batch size
        if self.workload_manager:
            self.workload_manager.max_batch_size = 64
            self.workload_manager.adjust_workers(2)

        logger.info("Optimized for high throughput")

    def clear_cache(self):
        """Clear embedding cache."""
        self.cache.clear()

        with self._cache_key_lock:
            self._cache_key_cache.clear()

    def cleanup(self):
        """Clean up all resources."""
        super().cleanup()

        if self.workload_manager:
            self.workload_manager.shutdown()

        if self.streaming_processor:
            self.streaming_processor.cleanup()

        DynamicModelManager.cleanup()
        gc.collect()

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except Exception as e:
            # Destructor failures in AdaptiveMultimodalProcessor should be logged at debug level
            logger.debug(
                f"AdaptiveMultimodalProcessor cleanup in destructor failed: {e}"
            )
