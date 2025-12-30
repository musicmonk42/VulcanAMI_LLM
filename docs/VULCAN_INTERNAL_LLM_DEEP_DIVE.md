# Ultra Deep Dive Examination of Vulcan's Internal LLM

**Version:** 3.0.0  
**Last Updated:** December 30, 2024

This document provides an ultra deep dive examination into the VulcanAMI platform's internal Large Language Model (LLM) architecture, covering all components from the low-level IR representation to the high-level cognitive reasoning systems. This is an exhaustive technical reference for developers, researchers, and operators.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Core LLM Components](#3-core-llm-components)
   - [3.1 GraphixTransformer](#31-graphixtransformer)
   - [3.2 IR (Intermediate Representation) System](#32-ir-intermediate-representation-system)
   - [3.3 GraphixExecutor - Production IR Engine](#33-graphixexecutor---production-ir-engine)
   - [3.4 Local GPT Provider](#34-local-gpt-provider)
   - [3.5 GPT Training Model](#35-gpt-training-model)
4. [Attention Mechanisms](#4-attention-mechanisms)
5. [Feed-Forward Networks](#5-feed-forward-networks)
6. [Normalization Layers](#6-normalization-layers)
7. [Embedding System](#7-embedding-system)
8. [Persistent Context Management](#8-persistent-context-management)
9. [Memory Architecture](#9-memory-architecture)
   - [9.1 Hierarchical Memory System](#91-hierarchical-memory-system)
   - [9.2 Persistent Memory with ZK Proofs](#92-persistent-memory-with-zk-proofs)
   - [9.3 Tool Selection Memory](#93-tool-selection-memory)
10. [Reasoning Systems](#10-reasoning-systems)
    - [10.1 Unified Reasoning Interface](#101-unified-reasoning-interface)
    - [10.2 Reasoning Types](#102-reasoning-types)
    - [10.3 Mathematical Verification](#103-mathematical-verification)
11. [World Model Integration](#11-world-model-integration)
    - [11.1 Core Components](#111-core-components)
    - [11.2 Meta-Reasoning](#112-meta-reasoning)
12. [Training Infrastructure](#12-training-infrastructure)
    - [12.1 Governed Trainer](#121-governed-trainer)
    - [12.2 Continual Learning](#122-continual-learning)
    - [12.3 Meta-Learning](#123-meta-learning)
    - [12.4 RLHF Integration](#124-rlhf-integration)
13. [Tool Selection System](#13-tool-selection-system)
    - [13.1 Production Tool Selector](#131-production-tool-selector)
    - [13.2 Cost Model](#132-cost-model)
    - [13.3 Utility Model](#133-utility-model)
    - [13.4 Safety Governor](#134-safety-governor)
14. [Unified Generation System](#14-unified-generation-system)
15. [Graph Compiler Infrastructure](#15-graph-compiler-infrastructure)
16. [Self-Improvement System](#16-self-improvement-system)
17. [Safety and Governance](#17-safety-and-governance)
18. [Performance Instrumentation](#18-performance-instrumentation)
19. [Configuration and Tuning](#19-configuration-and-tuning)
20. [Code Examples](#20-code-examples)
21. [Future Directions](#21-future-directions)

---

## 1. Executive Summary

The Vulcan Internal LLM is a sophisticated, multi-layered language model architecture that integrates:

- **Custom Transformer Architecture**: Built using an Intermediate Representation (IR) system for maximum flexibility
- **Production IR Execution Engine**: GraphixExecutor with Flash Attention, quantization, and KV caching
- **Hybrid Attention Mechanisms**: Combining traditional attention with linear attention, sparse patterns, and GQA
- **SwiGLU Feed-Forward Networks**: With Mixture-of-Experts (MoE) dynamic gating capabilities
- **RoPE Positional Encoding**: Rotary Position Embeddings for improved context handling
- **Multi-Modal Reasoning**: 8+ reasoning types with unified orchestration
- **Self-Improving World Model**: Autonomous improvement with safety constraints and kill switches
- **Production-Ready RAG**: Hierarchical memory with GraphRAG, Merkle LSM compaction, and ZK proofs
- **Mathematical Verification**: Bayesian calculation verification with learning integration
- **Local GPT Provider**: Fine-tuned model serving with streaming, perplexity, and confidence calibration

**Key Statistics:**
- GraphixExecutor: ~1,500 lines of production code
- GraphixTransformer: ~1,300 lines with tokenizer
- UnifiedReasoner: ~3,600 lines with tool selection
- HierarchicalMemory: ~1,900 lines with embeddings
- LocalGPTProvider: ~430 lines with calibration
- GPTModel (Training): ~580 lines with generation
- GovernedTrainer: ~1,380 lines with governance
- EnhancedContinualLearner: ~2,100 lines with RLHF
- MetaLearner: ~1,200 lines with MAML/Reptile
- UnifiedGeneration: ~1,200 lines with fusion strategies
- GraphCompiler: ~800 lines with LLVM backend
- Tool Selection System: ~1,500 lines with utility optimization
- VULCAN Cognitive System: 285,000+ lines of code across 256 files
- Supported Reasoning Types: 8+ (Symbolic, Causal, Probabilistic, Analogical, Multimodal, Mathematical, etc.)
- Memory Systems: 10+ specialized subsystems with persistent storage
- Meta-Learning Algorithms: 5 (MAML, FOMAML, Reptile, PROTO, ANIL)

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    VULCAN INTERNAL LLM ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    APPLICATION LAYER                             │   │
│  │   Text Generation │ Reasoning │ Question Answering │ Analysis    │   │
│  └───────────────────────────┬─────────────────────────────────────┘   │
│                              │                                          │
│  ┌───────────────────────────▼─────────────────────────────────────┐   │
│  │                    UNIFIED REASONING LAYER                       │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌───────────┐  │   │
│  │  │ Symbolic   │  │ Causal     │  │ Probabilis │  │ Analogical│  │   │
│  │  │ Reasoning  │  │ Reasoning  │  │ tic        │  │ Reasoning │  │   │
│  │  └────────────┘  └────────────┘  └────────────┘  └───────────┘  │   │
│  └───────────────────────────┬─────────────────────────────────────┘   │
│                              │                                          │
│  ┌───────────────────────────▼─────────────────────────────────────┐   │
│  │                    WORLD MODEL LAYER                             │   │
│  │   Causal DAG │ Prediction Engine │ Intervention Manager          │   │
│  │   Correlation Tracker │ Self-Improvement Drive                   │   │
│  └───────────────────────────┬─────────────────────────────────────┘   │
│                              │                                          │
│  ┌───────────────────────────▼─────────────────────────────────────┐   │
│  │                    GRAPHIX TRANSFORMER CORE                      │   │
│  │  ┌────────────────────────────────────────────────────────────┐ │   │
│  │  │  Token Embeddings (RoPE) → Transformer Layers (N layers)   │ │   │
│  │  │  → Layer Norm → Output Projection → Token Prediction       │ │   │
│  │  └────────────────────────────────────────────────────────────┘ │   │
│  └───────────────────────────┬─────────────────────────────────────┘   │
│                              │                                          │
│  ┌───────────────────────────▼─────────────────────────────────────┐   │
│  │                    IR EXECUTION ENGINE                           │   │
│  │   IR Attention │ IR FeedForward │ IR LayerNorm │ IR Embeddings   │   │
│  │   KV Cache │ Sparse Patterns │ Hybrid Attention │ MoE Gating    │   │
│  └───────────────────────────┬─────────────────────────────────────┘   │
│                              │                                          │
│  ┌───────────────────────────▼─────────────────────────────────────┐   │
│  │                    MEMORY & CONTEXT LAYER                        │   │
│  │   Persistent Context │ Graph RAG │ Hierarchical Memory           │   │
│  │   Embedding Cache │ Relevance Scoring │ Compression              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Core LLM Components

### 3.0 GraphixVulcanLLM - Main Orchestrator

**Location:** `graphix_vulcan_llm.py` (root level)

The GraphixVulcanLLM is the **main entry point** and orchestrator for the entire Vulcan LLM system. It integrates all components into a unified, production-ready interface.

#### Architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                    GRAPHIX VULCAN LLM                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Application API                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ generate() │ stream() │ generate_async() │ generate_text()│  │
│  │ train() │ fine_tune_step() │ self_improve()              │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│  ┌───────────────────────────▼───────────────────────────────┐ │
│  │                COGNITIVE LOOP                              │ │
│  │  Prompt → Encode → Generate → Safety → Consensus → Output  │ │
│  └───────────────────────────┬───────────────────────────────┘ │
│                              │                                  │
│  ┌──────────────┬────────────┼────────────┬──────────────────┐ │
│  │              │            │            │                   │ │
│  ▼              ▼            ▼            ▼                   │ │
│  GraphixTransformer  GraphixVulcanBridge  SafeGeneration      │ │
│  (Model Core)        (World Model)        (Safety)            │ │
│  │              │            │            │                   │ │
│  ▼              ▼            ▼            ▼                   │ │
│  LanguageReasoning   HierarchicalContext  EnhancedValidator   │ │
│  (Generation)        (Memory)             (Validation)        │ │
│                                                                │ │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ UTILITIES: Cache │ Monitor │ Explainer │ Self-Improvement │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Component Integrations:

| Component | Module | Purpose |
|-----------|--------|---------|
| `GraphixTransformer` | `src/llm_core/graphix_transformer.py` | Core transformer model |
| `GraphixVulcanBridge` | `src/integration/graphix_vulcan_bridge.py` | World model + reasoning bridge |
| `SafeGeneration` | `src/generation/safe_generation.py` | Safety-aware generation |
| `EnhancedSafetyValidator` | `src/vulcan/safety/llm_validators.py` | Token/sequence validation |
| `CognitiveLoop` | `src/integration/cognitive_loop.py` | Main generation loop |
| `HierarchicalContext` | `src/context/hierarchical_context.py` | Memory management |
| `CausalContext` | `src/context/causal_context.py` | Causal reasoning context |
| `LanguageReasoning` | `src/vulcan/reasoning/language_reasoning.py` | Language-based reasoning |
| `ExplainableGeneration` | `src/generation/explainable_generation.py` | Explanation generation |
| `GovernedTrainer` | `src/training/governed_trainer.py` | Training with governance |
| `SelfImprovingTraining` | `src/training/self_improving_training.py` | Autonomous improvement |

#### Configuration:

```python
default_config = {
    "transformer": GraphixTransformerConfig(
        num_layers=6,
        hidden_size=512,
        num_heads=8,
        vocab_size=50257,
        max_position_embeddings=2048,
        dropout=0.1,
    ),
    "generation": {
        "max_tokens": 2000,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.9,
        "repetition_penalty": 1.0,
        "enable_streaming": True,
    },
    "safety": {
        "mode": "first_safe",
        "enable_validation": True,
        "max_retries": 3,
    },
    "training": {
        "max_grad_norm": 5.0,
        "learning_rate": 1e-4,
        "batch_size": 8,
    },
    "performance": {
        "enable_caching": True,
        "cache_size_mb": 512,
        "enable_batching": True,
    },
}
```

#### Generation API:

```python
# Basic generation
result = llm.generate(
    prompt="Hello, world!",
    max_tokens=100,
    explain=True,
    use_cache=True,
)

# Streaming generation
for token in llm.stream(prompt, max_tokens=100, callback=my_callback):
    print(token, end="")

# Async generation
result = await llm.generate_async(prompt, max_tokens=100)

# Quick generation (lightweight)
tokens = llm.quick_generate(prompt, max_tokens=32)
```

#### GenerationResult Structure:

```python
@dataclass
class GenerationResult:
    tokens: List[int]              # Generated token IDs
    text: str                      # Decoded text
    reasoning_trace: List[Dict]    # Step-by-step reasoning
    safety_events: List[Dict]      # Safety interventions
    explanation: Optional[Dict]    # Explainability data
    metrics: Dict[str, Any]        # Performance metrics
    stopped_reason: str            # Why generation stopped
    duration_seconds: float        # Total time
    metadata: Dict[str, Any]       # Additional data
```

#### Training API:

```python
# Full training loop
logs = llm.train(dataset, epochs=3, batch_size=8)

# Single step training
record = llm.fine_tune_step(batch)

# Self-improvement cycle
improvement = llm.self_improve(context)
```

#### Performance Monitoring:

```python
class PerformanceMonitor:
    """Tracks system performance metrics."""
    
    # Metrics tracked:
    # - total_tokens
    # - total_duration
    # - generation_count
    # - error_count
    # - cache_hits / cache_misses
    # - avg_tokens_per_generation
    # - overall_throughput_tokens_per_sec
    # - error_rate
```

#### Cache Management:

```python
class CacheManager:
    """LRU cache for generation results."""
    
    # Features:
    # - LRU eviction policy
    # - Thread-safe operations
    # - Configurable max_size
    # - Key includes prompt + generation params
```

#### Async Generator Handling (Critical Fix):

The v2.0.2 release includes a critical fix for handling async generators from CognitiveLoop:

```python
def generate(self, prompt, ...):
    # CognitiveLoop.generate() may return:
    # 1. Direct result object (has .tokens attribute)
    # 2. Coroutine (needs await)
    # 3. Async generator (needs iteration)
    
    gen_result = self.cog_loop.generate(prompt, ...)
    
    # Detect and handle each case:
    if inspect.iscoroutine(gen_result):
        loop_result = loop.run_until_complete(gen_result)
        # May return async generator after await!
        if hasattr(loop_result, "__anext__"):
            # Consume nested async generator
            loop_result = loop.run_until_complete(consume_nested_generator())
    elif hasattr(gen_result, "__anext__"):
        # Direct async generator
        loop_result = loop.run_until_complete(consume_generator())
    else:
        # Direct result
        loop_result = gen_result
```

### 3.1 GraphixTransformer

**Location:** `src/llm_core/graphix_transformer.py`

The GraphixTransformer is the main transformer model class that orchestrates all LLM operations. It provides:

#### Key Features:
- **IR-Based Execution**: Uses intermediate representation for flexible computation
- **LoRA Support**: Low-Rank Adaptation for efficient fine-tuning
- **Top-P Sampling**: Nucleus sampling for generation
- **Gradient Checkpointing**: Memory-efficient training
- **KV Caching**: Efficient inference through key-value caching

#### Configuration Parameters:

```python
@dataclass
class GraphixTransformerConfig:
    num_layers: int = 6              # Number of transformer layers
    hidden_size: int = 256           # Hidden dimension size
    num_heads: int = 4               # Number of attention heads
    vocab_size: int = 4096           # Vocabulary size
    max_position_embeddings: int = 1024  # Maximum sequence length
    dropout: float = 0.1             # Dropout probability
    layer_norm_eps: float = 1e-5     # Layer norm epsilon
    seed: Optional[int] = 1234       # Random seed
    gradient_checkpointing: bool = False  # Memory optimization
    dtype: str = "float32"           # Data type
    lora_rank: int = 0               # LoRA rank (0 = disabled)
    lora_alpha: float = 1.0          # LoRA scaling factor
```

#### Core Methods:

| Method | Description |
|--------|-------------|
| `forward(tokens)` | Standard forward pass through transformer |
| `generate(prompt, max_new_tokens, temperature, top_p)` | Autoregressive text generation |
| `get_embeddings(text)` | Get embeddings for input text |
| `get_logits(input)` | Get logits for next token prediction |
| `add_adapter(name, rank, alpha)` | Add LoRA adapter |
| `save(path)` / `load(path)` | Model persistence |

#### SimpleTokenizer:

The transformer includes a built-in word-level tokenizer with:
- Pre-populated vocabulary of 400+ common English words
- Special tokens: `<pad>`, `<unk>`, `<bos>`, `<eos>`
- Dynamic vocabulary expansion
- Encode/decode functionality

### 3.2 IR (Intermediate Representation) System

The IR system provides a graph-based representation of transformer operations, enabling:
- **Hardware Abstraction**: Same IR can target CPU, GPU, or custom accelerators
- **Optimization**: Graph-level optimizations like fusion and dead code elimination
- **Debugging**: Clear visibility into computation structure
- **Extensibility**: Easy to add new operations or modify existing ones

#### IR Node Types:

| Type | Description | Example |
|------|-------------|---------|
| `transform` | Linear transformations | Q/K/V projections |
| `lookup` | Embedding lookups | Token embeddings |
| `filter` | Activation functions | SwiGLU |
| `combine` | Tensor combination | Residual connections |
| `reduce_*` | Reduction operations | Mean, variance |
| `cache` | KV caching | Attention cache |
| `dropout` | Regularization | Dropout layers |
| `moe_gate` | Expert gating | MoE routing |
| `spike` | Sparse activation | Spiking gate |

### 3.3 GraphixExecutor - Production IR Engine

**Location:** `src/llm_core/graphix_executor.py`

The GraphixExecutor is the production-grade IR execution engine with comprehensive optimizations:

#### Enterprise Features:

```
✅ CORE EXECUTION
- Multi-stage IR graph execution (embeddings, attention, FFN, layer norm)
- Dynamic graph optimization and fusion
- Automatic kernel selection and optimization
- Memory-efficient execution with gradient checkpointing

✅ ADVANCED OPTIMIZATIONS
- Flash Attention implementation
- Kernel fusion (attention + feedforward)
- Mixed precision (FP16, BF16, INT8)
- Dynamic quantization and dequantization
- KV cache management with eviction policies
- Sparse attention patterns (sliding window, block-sparse)

✅ LORA & PEFT
- LoRA adapter fusion and application
- Multi-adapter support with dynamic switching
- Adapter merging and quantization
- Fine-grained adapter control per layer

✅ DISTRIBUTED & SCALING
- Tensor parallelism hints
- Pipeline parallelism support
- Expert parallelism for MoE
- Gradient accumulation

✅ OBSERVABILITY
- Performance profiling and tracing
- Memory tracking and optimization
- Execution graph visualization
- Audit logging and compliance

✅ ROBUSTNESS
- Automatic error recovery
- Fallback execution paths
- Checkpointing and state management
- Numerical stability checks
```

#### Execution Modes:

| Mode | Description |
|------|-------------|
| `TRAINING` | Full gradient computation |
| `INFERENCE` | Optimized for speed |
| `EVALUATION` | Metrics collection |
| `PROFILING` | Detailed timing |

#### Precision Modes:

| Mode | Description | Memory Savings |
|------|-------------|----------------|
| `FP32` | Full precision | Baseline |
| `FP16` | Half precision | 50% |
| `BF16` | Brain float | 50% |
| `INT8` | 8-bit quantized | 75% |
| `MIXED` | Dynamic selection | Variable |

#### KV Cache Management:

```python
class KVCacheManager:
    """Manages KV cache with eviction policies."""
    
    # Eviction Policies:
    # - LRU: Least Recently Used (default)
    # - LFU: Least Frequently Used
    # - FIFO: First In First Out
    # - ADAPTIVE: Weighted recency + frequency
    
    def get(self, layer_idx, head_idx, position) -> Optional[KVCacheEntry]
    def put(self, layer_idx, head_idx, position, keys, values) -> None
    def get_stats(self) -> Dict[str, Any]
```

#### Quantization Support:

```python
class QuantizationManager:
    """Manages quantization and dequantization."""
    
    def quantize(tensor, key) -> Tuple[List[int], float, float]:
        """Quantize tensor to INT8 with scale/zero-point."""
        
    def dequantize(quantized, scale, zero_point) -> List[float]:
        """Dequantize back to float."""
```

#### Performance Instrumentation:

```python
# Decorator for timing operations
@timed_operation("GraphixExecutor.execute")
def execute(self, graph_ir, inputs):
    ...

# Global performance stats
_PERF_STATS = {
    "operation_name": {
        "count": 0,
        "total_ms": 0.0,
        "min_ms": float("inf"),
        "max_ms": 0.0
    }
}

# Performance summary
def log_performance_summary():
    """Log hottest operations sorted by total time."""
```

#### Executor Configuration:

```python
@dataclass
class ExecutorConfig:
    mode: ExecutionMode = ExecutionMode.INFERENCE
    precision: PrecisionMode = PrecisionMode.FP32
    attention_impl: AttentionImpl = AttentionImpl.FLASH
    use_flash_attention: bool = True
    use_kernel_fusion: bool = True
    use_kv_cache: bool = True
    kv_cache_size: int = 2048
    kv_cache_eviction: CacheEvictionPolicy = CacheEvictionPolicy.LRU
    max_batch_size: int = 32
    gradient_checkpointing: bool = False
    enable_profiling: bool = False
    enable_audit: bool = True
    use_quantization: bool = False
    quantization_bits: int = 8
    compile_graphs: bool = True
    optimize_memory: bool = True
```

#### Execution Pipeline:

```
┌─────────────────────────────────────────────────────────────────┐
│                    GRAPHIX EXECUTOR PIPELINE                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Inputs ──► Audit Log ──► Stage 1: Embeddings                   │
│                               │                                 │
│                               ▼                                 │
│                      Stage 2: Transformer Layers (Loop)         │
│                         ┌─────────────────────────────┐         │
│                         │ Pre-LN ──► Attention        │         │
│                         │    │         │              │         │
│                         │    │    KV Cache Lookup     │         │
│                         │    │         │              │         │
│                         │    └──► Residual            │         │
│                         │              │              │         │
│                         │ Pre-LN ──► FFN (SwiGLU)     │         │
│                         │    │         │              │         │
│                         │    └──► Residual ─────────►│         │
│                         └─────────────────────────────┘         │
│                               │                                 │
│                               ▼                                 │
│                      Stage 3: Final Layer Norm                  │
│                               │                                 │
│                               ▼                                 │
│                      Metrics Update ──► Result                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.4 Local GPT Provider

**Location:** `src/local_llm/provider/local_gpt_provider.py`

The LocalGPTProvider serves fine-tuned GPT models with production features:

#### Provider Configuration:

```python
@dataclass
class ProviderInitConfig:
    model_path: str              # Path to llm_best_model.pt
    vocab_path: str              # Path to vocab.json
    device: str = "cpu"          # cpu, cuda, cuda:0, etc.
    seq_len: int = 256           # Maximum sequence length
    dim: int = 384               # Model dimension
    n_layers: int = 6            # Number of layers
    n_heads: int = 8             # Number of attention heads
    ff_mult: int = 4             # FFN expansion factor
    dropout: float = 0.0         # Dropout rate
    dtype: str = "float32"       # float32, float16, bfloat16
    use_autocast: bool = False   # Enable automatic mixed precision
    calibration_path: Optional[str] = None  # Confidence calibration
    temperature: float = 0.9     # Sampling temperature
    top_k: int = 64              # Top-K sampling
    top_p: float = 0.95          # Nucleus sampling
    repetition_penalty: float = 1.05  # Repetition penalty
    eos_token: Optional[str] = None   # End of sequence token
```

#### API Methods:

| Method | Description |
|--------|-------------|
| `generate(prompt, ...)` | Single text generation with metadata |
| `generate_batch(prompts, ...)` | Batch generation |
| `generate_stream(prompt, ...)` | Streaming generator yielding partial text |
| `perplexity(text)` | Compute perplexity score |
| `batch_perplexity(texts)` | Batch perplexity computation |

#### Confidence Calibration:

```python
class OptionalCalibrator:
    """Remaps model confidences for better calibration."""
    
    # Supported calibration types:
    # - kind="scale": p' = scale * p + bias
    # - kind="temperature": p' = sigmoid(logit(p) / T)
    # - kind="isotonic": Piecewise linear mapping (future)
    
    def calibrate_prob(self, p: float) -> float:
        """Apply calibration to probability."""
```

#### Streaming Generation:

```python
def generate_stream(
    self,
    prompt: str,
    max_new_tokens: int = 128,
    chunk_size: int = 1,
) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
    """
    Yields (partial_text, meta) every chunk_size new tokens.
    
    Meta includes:
    - new_ids: Newly generated token IDs
    - total_len: Total sequence length
    - emitted: Total tokens emitted
    """
```

### 3.5 GPT Training Model

**Location:** `src/training/gpt_model.py`

A patched minimal causal Transformer (GPT-like) for token-level language modeling:

#### Architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                    GPT MODEL ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Token IDs ──► Token Embedding ──► Positional Embedding         │
│                                           │                     │
│                                           ▼                     │
│                                      Dropout                    │
│                                           │                     │
│                     ┌─────────────────────┼─────────────────┐   │
│                     │         N × Transformer Blocks         │   │
│                     │  ┌──────────────────────────────────┐ │   │
│                     │  │ Pre-LN ──► Multi-Head Self-Attn  │ │   │
│                     │  │     (with causal mask caching)   │ │   │
│                     │  │              + Residual          │ │   │
│                     │  ├──────────────────────────────────┤ │   │
│                     │  │ Pre-LN ──► FeedForward (GELU)    │ │   │
│                     │  │              + Residual          │ │   │
│                     │  └──────────────────────────────────┘ │   │
│                     └───────────────────┬───────────────────┘   │
│                                         │                       │
│                                         ▼                       │
│                                  Final LayerNorm                │
│                                         │                       │
│                                         ▼                       │
│                              Output Projection (LM Head)        │
│                                  (weight-tied)                  │
│                                         │                       │
│                                         ▼                       │
│                                      Logits                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Configuration:

```python
@dataclass
class GPTConfig:
    vocab_size: int
    seq_len: int
    dim: int = 512
    n_layers: int = 6
    n_heads: int = 8
    ff_mult: int = 4
    dropout: float = 0.1
    tied_embeddings: bool = True       # Weight tying
    layer_norm_eps: float = 1e-5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Training options
    loss_reduction: str = "mean"       # 'mean' or 'sum'
    label_smoothing: float = 0.0       # Label smoothing coefficient
    return_loss_dict: bool = False     # Return detailed loss metrics
    enforce_safe_softmax: bool = True  # Guard against NaN in generation
```

#### Training Features:

| Feature | Description |
|---------|-------------|
| Loss Reduction | `mean` for per-token avg, `sum` for total |
| Label Smoothing | Optional smoothing (0.0 to disable) |
| Perplexity | Computed automatically in loss dict |
| Gradient Safety | NaN/Inf checks with safe softmax |
| Mixed Precision | Compatible with external AMP |

#### Generation Features:

| Feature | Description |
|---------|-------------|
| Temperature | Softmax temperature scaling |
| Top-K | Keep only top-k logits |
| Top-P (Nucleus) | Cumulative probability threshold |
| Repetition Penalty | Logarithmic penalty for repeats |
| Greedy Mode | Argmax when temperature ≤ threshold |

#### Generation Method:

```python
@torch.no_grad()
def generate(
    self,
    start_ids: List[int],
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
    eos_id: Optional[int] = None,
    greedy_threshold: float = 1e-8,
) -> List[int]:
    """
    Autoregressive token generation.
    
    - Context cropped to last config.seq_len tokens
    - NaN-safe softmax with fallback to uniform
    - Logarithmic repetition penalty
    """
```

#### Stable Initialization (GPT-2 Style):

```python
# Token and positional embeddings
nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.01)

# Attention projections
nn.init.normal_(self.qkv.weight, mean=0.0, std=0.02)
nn.init.zeros_(self.qkv.bias)

# FFN layers
nn.init.normal_(m.weight, mean=0.0, std=0.02)
nn.init.zeros_(m.bias)
```

---

## 4. Attention Mechanisms

**Location:** `src/llm_core/ir_attention.py`

The attention system implements a sophisticated hybrid attention mechanism:

### Architecture Details:

```
┌─────────────────────────────────────────────────────────────────┐
│                    ATTENTION MECHANISM                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input ──► Q Projection ──┐                                     │
│                           ▼                                     │
│  Input ──► K Projection ──► Hybrid Attention ──► Softmax        │
│                           ▲       │                  │          │
│  Input ──► V Projection ──┘       │                  │          │
│                                   │                  │          │
│                                   ▼                  ▼          │
│                           KV Cache          Attention Dropout   │
│                                                      │          │
│                                                      ▼          │
│                                              Weighted Values    │
│                                                      │          │
│                                                      ▼          │
│                                              Spike Gate         │
│                                                      │          │
│                                                      ▼          │
│                                              Output Projection  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Features:

#### 1. Grouped Query Attention (GQA)
- Supports fewer KV heads than query heads
- Reduces memory bandwidth requirements
- Parameter: `num_kv_heads` (if < `num_heads`, enables GQA)

#### 2. Hybrid Attention
- Combines quadratic (softmax) and linear attention
- `linear_ratio`: 0.5 (configurable blend)
- Reduces computational complexity for long sequences

#### 3. KV Caching
- Dedicated cache nodes for K and V tensors
- Enables efficient autoregressive generation
- Cache axis: 0 (sequence dimension)

#### 4. Sparse Patterns
- Windowed attention support
- `window_size`: 128 tokens
- Pattern type: `windowed`

#### 5. Spiking Gate
- Threshold-based sparse activation
- `threshold`: 0.3
- Improves efficiency by pruning weak attention

#### 6. Causal Masking
- Built-in causal mask support
- Prevents attending to future tokens

### IR Structure:

```json
{
  "type": "attention_subgraph",
  "params": {
    "num_heads": 4,
    "hidden_size": 256,
    "num_kv_heads": null,
    "causal_masking": true,
    "sparse": "windowed",
    "window_size": 128
  },
  "metadata": {
    "semantic": "hybrid_grouped_attention",
    "inference": "kv_cache_supported",
    "sampling": {"type": "top_p", "p": 0.9}
  }
}
```

---

## 5. Feed-Forward Networks

**Location:** `src/llm_core/ir_feedforward.py`

The FFN uses a SwiGLU architecture with Mixture-of-Experts (MoE) capabilities:

### Architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                    FEED-FORWARD NETWORK (SwiGLU + MoE)          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input ──► Dynamic Gate ──┬──► Linear Gate ──┐                  │
│               (MoE)       │                  ▼                  │
│                           └──► Linear Expand ──► SwiGLU         │
│                                                    │            │
│                                                    ▼            │
│                                              Linear Project     │
│                                                    │            │
│                                                    ▼            │
│                                               Dropout           │
│                                                    │            │
│  Input ─────────────────────────────────────────► + (Residual)  │
│                                                    │            │
│                                                    ▼            │
│                                                 Output          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Features:

#### 1. SwiGLU Activation
- Combines Swish activation with gating
- Superior to standard ReLU/GELU for transformers
- Formula: `SwiGLU(x) = Swish(W_gate * x) ⊗ (W_up * x)`

#### 2. Mixture-of-Experts (MoE) Gating
- Dynamic expert selection
- Parameters:
  - `moe_experts`: 4 (number of experts)
  - `moe_top_k`: 2 (experts activated per token)
- Mode: `top_k_select`

#### 3. Scaled Residual Connection
- Residual scale: 0.5
- Improves training stability for deep networks

#### 4. Intermediate Size
- Default: `hidden_size * 4`
- Standard transformer expansion ratio

### Configuration:

```python
"params": {
    "hidden_size": 256,
    "intermediate": 1024,  # hidden_size * 4
    "dropout_p": 0.1,
    "residual_scale": 0.5,
    "moe_experts": 4,
    "moe_top_k": 2
}
```

---

## 6. Normalization Layers

**Location:** `src/llm_core/ir_layer_norm.py`

Supports multiple normalization strategies:

### Normalization Types:

#### 1. RMSNorm (Default)
- Root Mean Square normalization
- Omits mean calculation and shift parameter
- More efficient than LayerNorm
- Formula: `RMSNorm(x) = x / sqrt(mean(x²) + eps) * scale`

#### 2. Standard LayerNorm
- Full mean and variance normalization
- Includes shift (bias) parameter
- Formula: `LayerNorm(x) = (x - mean) / sqrt(var + eps) * scale + shift`

#### 3. GroupNorm (Optional)
- Groups channels for normalization
- Applied after base normalization
- Useful for certain architectural patterns

### Features:

- **Learned Epsilon Scheduling**: Adaptive epsilon for training stability
- **Pre-Norm Position**: Normalization applied before attention/FFN
- **Configurable Groups**: For GroupNorm integration

### IR Structure:

```json
{
  "type": "layer_norm_subgraph",
  "params": {
    "hidden_size": 256,
    "eps": 1e-5,
    "type": "rmsnorm",
    "groups": null,
    "eps_schedule": "learned",
    "position": "pre"
  }
}
```

---

## 7. Embedding System

**Location:** `src/llm_core/ir_embeddings.py`

### Rotary Position Embeddings (RoPE)

Unlike traditional additive position embeddings, Vulcan uses RoPE:

#### Advantages of RoPE:
- Better extrapolation to longer sequences
- Applied during Q/K projection in attention
- No separate position lookup table needed
- Preserves relative position information

#### Configuration:

```python
"params": {
    "vocab_size": 4096,
    "hidden_size": 256,
    "max_positions": 1024,
    "pos_mode": "rotary",
    "pos_base": 10000,  # Base frequency for RoPE
    "prune_vocab_below_freq": 5  # Dynamic vocab pruning hint
}
```

### Embedding Pipeline:

```
Token IDs ──► Token Lookup ──► Dropout ──► Norm Scale ──► Output
                  │
                  └── (RoPE applied later in Attention Q/K)
```

---

## 8. Persistent Context Management

**Location:** `src/llm_core/persistant_context.py`

A production-ready RAG (Retrieval-Augmented Generation) implementation:

### Architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│               PERSISTENT CONTEXT MANAGER                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Query ──► Embedding ──► Retrieval ──► Chunking ──► Scoring     │
│                              │             │           │        │
│                              ▼             ▼           ▼        │
│                         Memory System   Strategies  Relevance   │
│                              │             │           │        │
│                              └─────────────┴───────────┘        │
│                                           │                     │
│                                           ▼                     │
│                      Parent-Child Expansion ──► Reranking       │
│                                                      │          │
│                                                      ▼          │
│                                              Compression        │
│                                                      │          │
│                                                      ▼          │
│                                              Token Budget Fit   │
│                                                      │          │
│                                                      ▼          │
│                                              Final Context      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components:

#### 1. Chunking Strategies

| Strategy | Description |
|----------|-------------|
| `FIXED_SIZE` | Fixed character chunks |
| `SEMANTIC` | Sentence-boundary aware |
| `SLIDING_WINDOW` | Overlapping windows |
| `HIERARCHICAL` | Paragraphs → sentences |
| `PARAGRAPH` | Paragraph-based |

#### 2. Reranking Methods

| Method | Description |
|--------|-------------|
| `NONE` | No reranking |
| `CROSS_ENCODER` | Neural reranking |
| `RECIPROCAL_RANK_FUSION` | Multi-signal fusion |
| `DIVERSITY` | MMR-based diversity |

#### 3. Compression Methods

| Method | Description |
|--------|-------------|
| `NONE` | No compression |
| `EXTRACTIVE` | Select important sentences |
| `ABSTRACTIVE` | Use summaries |
| `HYBRID` | Mix of both |

### Configuration:

```python
@dataclass
class ContextConfig:
    max_context_tokens: int = 8192
    retrieval_k: int = 50
    rerank_top_k: int = 20
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.HIERARCHICAL
    chunk_size: int = 512
    chunk_overlap: int = 128
    reranking_method: RerankingMethod = RerankingMethod.RECIPROCAL_RANK_FUSION
    compression_method: CompressionMethod = CompressionMethod.EXTRACTIVE
    compression_ratio: float = 0.5
    use_parent_child_context: bool = True
    temporal_decay_factor: float = 0.95
    diversity_threshold: float = 0.8
    cache_size: int = 1000
```

---

## 9. Memory Architecture

### 9.1 Hierarchical Memory System

**Location:** `src/vulcan/memory/hierarchical.py`

The HierarchicalMemory class implements a multi-level memory system with tool selection history:

#### Memory Levels:

| Level | Capacity | Decay Rate | Consolidation Threshold | Use Case |
|-------|----------|------------|------------------------|----------|
| Sensory | 50 | 0.5 | 0.7 | Raw input buffer |
| Working | 20 | 0.1 | 0.6 | Active processing |
| Short-Term | 1000 | 0.05 | 0.5 | Recent memories |
| Long-Term | 100000 | 0.001 | 0.8 | Persistent storage |

#### Memory Level Data Structure:

```python
@dataclass
class MemoryLevel:
    name: str
    capacity: int
    decay_rate: float
    consolidation_threshold: float
    memories: Dict[str, Memory]
    access_queue: deque  # Track access order for LRU
    
    def add(self, memory: Memory) -> bool
    def remove_least_salient(self, n: int) -> List[Memory]
    def get_candidates_for_consolidation(self) -> List[Memory]
```

#### Embedding Models:

The system supports multiple embedding backends with automatic fallback:

```python
# Priority order:
1. Global Model Registry (singleton, shared across components)
2. SentenceTransformer models:
   - "all-MiniLM-L6-v2" (fast, 384 dimensions)
   - "all-mpnet-base-v2" (better quality, 768 dimensions)
   - "paraphrase-MiniLM-L6-v2" (alternative)
3. Hash-based fallback (128 dimensions)
```

#### Memory Operations:

| Operation | Description |
|-----------|-------------|
| `store(content, type, importance)` | Store content at appropriate level |
| `retrieve(query)` | Retrieve memories matching query |
| `forget(memory_id)` | Remove specific memory |
| `consolidate()` | Promote memories between levels |

#### Attention-Based Retrieval:

```python
class AttentionMechanism:
    """Computes attention weights for memory retrieval."""
    
    def __init__(self, hidden_dim, input_dim):
        # Learnable attention weights
        
    def compute_attention(self, query_embedding, memory_embeddings):
        """Compute attention-weighted relevance scores."""
```

#### LLM Context Integration:

```python
def retrieve_context_for_generation(self, query_tokens, max_tokens=2048):
    """
    Retrieve relevant context from memory for LLM generation.
    
    Returns merged context from:
    1. Episodic: Recent conversation (short_term level)
    2. Semantic: Relevant concepts (long_term level)
    3. Procedural: Matching patterns with tools and utility scores
    """

def store_generation(self, prompt, generated, reasoning_trace):
    """
    Store generation in memory, updating all levels:
    - Procedural: Pattern extraction from reasoning trace
    - Episodic: Full prompt/response with timestamp
    - Semantic: Extracted concepts
    """
```

### 9.2 Persistent Memory with ZK Proofs

**Location:** `src/vulcan/memory/hierarchical.py` (PersistentHierarchicalMemory)

The PersistentHierarchicalMemory extends hierarchical memory with durable storage:

#### Storage Tiers:

```
┌─────────────────────────────────────────────────────────────────┐
│                    MEMORY STORAGE TIERS                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  HOT TIER (Episodic Memory)                                     │
│  ├── In-memory deque                                            │
│  ├── Recent items (last 24 hours)                               │
│  └── Fastest access                                             │
│                                                                 │
│  WARM TIER (Local Disk)                                         │
│  ├── Pickle serialization                                       │
│  ├── Frequently accessed items                                  │
│  └── ./local_memory_cache/*.pkl                                 │
│                                                                 │
│  COLD TIER (Persistent Store)                                   │
│  ├── PackfileStore with S3/CloudFront                           │
│  ├── MerkleLSM compaction                                       │
│  └── Long-term archival                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Components:

| Component | Purpose |
|-----------|---------|
| `PackfileStore` | S3-backed persistent storage |
| `MerkleLSM` | Merkle tree + LSM compaction |
| `GraphRAG` | Semantic retrieval augmented generation |
| `UnlearningEngine` | Privacy-preserving data deletion |
| `ZKProver` | Zero-knowledge proofs for verification |
| `LearningStatePersistence` | Learning state recovery |

#### Interaction Storage:

```python
def store_interaction(
    self,
    query_id: str,
    query: str,
    answer: str,
    tools_used: Optional[List[str]] = None,
    success: bool = True,
    latency_ms: float = 0.0,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Store interaction across all tiers:
    1. GraphRAG for semantic retrieval
    2. Episodic memory for quick access
    3. LearningStatePersistence for recovery
    """
```

### 9.3 Tool Selection Memory

The HierarchicalMemory includes specialized storage for tool selection decisions:

#### Tool Selection Record:

```python
@dataclass
class ToolSelectionRecord:
    record_id: str
    timestamp: float
    problem_features: np.ndarray
    problem_description: str
    selected_tools: List[str]
    execution_strategy: str
    performance_metrics: Dict[str, float]  # latency, accuracy, energy
    context: Dict[str, Any]
    success: bool
    utility_score: float
    metadata: Dict[str, Any]
```

#### Problem Pattern Discovery:

```python
@dataclass
class ProblemPattern:
    pattern_id: str
    feature_signature: np.ndarray  # Centroid of similar problems
    typical_tools: List[str]       # Most successful tools
    success_rate: float            # Historical success rate
    avg_utility: float             # Average utility score
    occurrence_count: int          # Number of matching problems
    examples: List[str]            # Sample problem IDs
```

#### Tool Recommendation:

```python
def get_recommended_tools(
    self,
    problem_features: np.ndarray,
    problem_description: Optional[str] = None,
    max_recommendations: int = 5,
) -> List[Dict[str, Any]]:
    """
    Get tool recommendations based on:
    1. Similar successful problems
    2. Discovered problem patterns
    
    Returns recommendations with:
    - tool: Tool name
    - confidence: Utility-based confidence
    - success_rate: Historical success rate
    - occurrence_count: How often used
    - pattern_match: Whether matches a known pattern
    - score: Combined ranking score
    """
```

#### Background Tasks:

The system runs two background threads:

1. **Consolidation Thread**: Periodically promotes memories between levels
2. **Pattern Mining Thread**: Discovers problem patterns from tool selection history

```python
# Pattern mining algorithm:
1. Group records by feature similarity (threshold: 0.8)
2. For each group with >= 3 records:
   - Compute centroid feature signature
   - Calculate success rate and average utility
   - Identify typical tools from successful records
   - Create ProblemPattern
```

---

## 10. Reasoning Systems

### 10.1 Unified Reasoning Interface

**Location:** `src/vulcan/reasoning/unified_reasoning.py`

The UnifiedReasoner orchestrates all reasoning capabilities with production-grade features:

#### Initialization Parameters:

```python
class UnifiedReasoner:
    def __init__(
        self,
        enable_learning: bool = True,     # Enable continuous learning
        enable_safety: bool = True,       # Enable safety validation
        max_workers: Optional[int] = None,  # Thread pool size (env: VULCAN_MAX_WORKERS)
        config: Optional[Dict[str, Any]] = None,
    ):
```

#### Reasoning Strategies:

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `SEQUENTIAL` | Execute reasoning types in sequence | Simple problems |
| `PARALLEL` | Execute reasoning types concurrently | Independent analyses |
| `ENSEMBLE` | Weighted voting across reasoners | High confidence needed |
| `HIERARCHICAL` | Dependency-based execution | Complex reasoning chains |
| `ADAPTIVE` | Dynamic strategy selection | Unknown problem types |
| `HYBRID` | Custom combination approach | Specialized pipelines |
| `PORTFOLIO` | Complementary reasoner selection | Diverse perspectives |
| `UTILITY_BASED` | Utility-optimized selection | Resource-constrained |

#### Singleton Tool Weight Manager:

```python
class ToolWeightManager:
    """Singleton manager for tool weights shared between Learning and Ensemble.
    
    BUG #5 FIX: Learning system was updating tool weights in its own dictionary,
    but Ensemble was reading from a separate dictionary. Weights never propagated.
    This singleton ensures both systems use the same weight storage.
    """
    
    def get_weight(self, tool: str, default: float = 1.0) -> float
    def set_weight(self, tool: str, value: float) -> None
    def adjust_weight(self, tool: str, delta: float) -> None
    def get_all_weights(self, tools: List[str]) -> Dict[str, float]

# Usage:
get_weight_manager().adjust_weight("causal", 0.01)
```

#### Environment Detection:

```python
def _is_test_environment() -> bool:
    """
    Determine if running in test environment.
    
    Bug #3 Fix: Default to PRODUCTION for safety.
    Only return True if explicitly in test mode.
    
    Explicit TEST indicators:
    - PYTEST_CURRENT_TEST is set
    - VULCAN_TEST_MODE=true
    - VULCAN_ENV=test
    
    Explicit PRODUCTION indicators (override test):
    - VULCAN_PRODUCTION=true
    - VULCAN_FORCE_PRODUCTION_REASONING=true
    - VULCAN_ENV=production
    - ENVIRONMENT=production
    """
```

#### Utility Context:

```python
@dataclass
class UtilityContext:
    mode: ContextMode      # RUSH, ACCURATE, EFFICIENT, BALANCED
    time_budget: float     # Milliseconds
    energy_budget: float   # Millijoules
    min_quality: float     # Minimum acceptable quality
    max_risk: float        # Maximum acceptable risk
    user_preferences: Dict[str, Any]
```

#### Main Reasoning Method:

```python
def reason(
    self,
    input_data: Any,
    query: Optional[Dict[str, Any]] = None,
    reasoning_type: Optional[ReasoningType] = None,
    strategy: ReasoningStrategy = ReasoningStrategy.ADAPTIVE,
    confidence_threshold: Optional[float] = None,
    constraints: Optional[Dict[str, Any]] = None,
) -> ReasoningResult:
    """
    Enhanced reasoning interface with production tool selection.
    
    Pipeline:
    1. Shutdown check
    2. Create reasoning chain
    3. Safety input validation
    4. Distribution shift detection
    5. Reasoning type determination
    6. VOI (Value of Information) gate
    7. Plan optimization
    8. Tool selection for portfolio/utility strategies
    9. Strategy execution with timeout
    10. Confidence calibration
    11. Post-processing with mathematical verification
    12. Safety output validation
    13. Learning update
    14. Metrics and caching
    """
```

### 10.2 Reasoning Types

#### Available Reasoners:

| Type | Class | Description |
|------|-------|-------------|
| `PROBABILISTIC` | `ProbabilisticReasoner` | Uncertainty quantification |
| `SYMBOLIC` | `SymbolicReasoner` | Logic and formal reasoning |
| `CAUSAL` | `EnhancedCausalReasoning` | Cause-effect relationships |
| `ANALOGICAL` | `AnalogicalReasoner` | Pattern matching, similarity |
| `MULTIMODAL` | `MultiModalReasoningEngine` | Cross-modal reasoning |
| `MATHEMATICAL` | `MathematicalComputationTool` | Computation and proofs |
| `COUNTERFACTUAL` | `CounterfactualReasoner` | What-if scenarios |
| `LANGUAGE` | `LanguageReasoner` | Natural language understanding |

#### Specialized Reasoners:

```python
# Counterfactual reasoning (from causal reasoner)
self.counterfactual = CounterfactualReasoner(causal_reasoner)

# Cross-modal reasoning
self.cross_modal = CrossModalReasoner()

# Multimodal with registered modality reasoners
self.multimodal = MultiModalReasoningEngine(enable_learning=enable_learning)
self.multimodal.register_modality_reasoner(ModalityType.TEXT, symbolic_reasoner)
self.multimodal.register_modality_reasoner(ModalityType.CODE, symbolic_reasoner)
```

#### Ensemble Weighting:

```python
def _ensemble_reasoning(self, plan, reasoning_chain) -> ReasoningResult:
    """Ensemble reasoning with weighted voting."""
    
    for reasoning_type, result in results:
        # Base weight from confidence
        base_weight = result.confidence
        
        # Type weight from historical performance
        type_weight = self._get_reasoning_type_weight(reasoning_type)
        
        # Utility weight if context available
        if utility_context:
            execution_time_ms = result.metadata.get("execution_time_ms", 100)
            utility_weight = self._calculate_result_utility(
                result, utility_context, execution_time_ms
            )
            weights.append(base_weight * type_weight * utility_weight)
        else:
            weights.append(base_weight * type_weight)
    
    # Issue #53 Fix: Handle zero weights gracefully
    if sum(weights) <= 0:
        weights = [1.0 / len(results)] * len(results)
```

### 10.3 Mathematical Verification

**Location:** `src/vulcan/reasoning/mathematical_verification.py`

The MathematicalVerificationEngine validates calculation results:

#### Verification Types:

```python
class MathVerificationStatus(Enum):
    VERIFIED = "verified"
    ERROR_DETECTED = "error_detected"
    UNCERTAIN = "uncertain"
    NOT_APPLICABLE = "not_applicable"
```

#### Bayesian Calculation Verification:

```python
@dataclass
class BayesianProblem:
    prior: float           # P(H)
    sensitivity: float     # P(E|H)
    specificity: float     # P(¬E|¬H)

def verify_bayesian_calculation(
    self,
    problem: BayesianProblem,
    computed_posterior: float,
) -> VerificationResult:
    """
    Verify Bayesian calculation using Bayes' theorem:
    P(H|E) = P(E|H) * P(H) / P(E)
    
    where P(E) = P(E|H)*P(H) + P(E|¬H)*P(¬H)
    """
```

#### Arithmetic Verification:

```python
def verify_arithmetic(
    self,
    expression: str,
    computed_result: float,
    variables: Dict[str, float],
) -> VerificationResult:
    """Verify arithmetic expression evaluation."""
```

#### Learning Integration:

```python
# Constants for learning feedback
MATH_VERIFICATION_CONFIDENCE_BOOST = 1.1  # Boost for verified correct
MATH_ERROR_CONFIDENCE_PENALTY = 0.5       # Penalty for error detected
MATH_ACCURACY_REWARD = 0.015              # Learning reward for accuracy
MATH_ACCURACY_PENALTY = -0.01             # Learning penalty for errors
MATH_WEIGHT_ADJUSTMENT_PENALTY = -0.01    # Tool weight adjustment

def _apply_verification_to_result(self, result, verification, task):
    """Apply verification results and update learning system."""
    
    if verification.status == MathVerificationStatus.VERIFIED:
        # Boost confidence
        result.confidence *= MATH_VERIFICATION_CONFIDENCE_BOOST
        
        # Reward tool through learning
        if self._math_accuracy_integration:
            self._math_accuracy_integration.reward_tool(tool_name, self.learner)
            
    elif verification.status == MathVerificationStatus.ERROR_DETECTED:
        # Apply corrections
        result.conclusion['math_correction'] = {
            'original': result.conclusion,
            'corrected': verification.corrections.get('correct_result'),
            'errors': [e.value for e in verification.errors],
        }
        
        # Reduce confidence
        result.confidence *= MATH_ERROR_CONFIDENCE_PENALTY
        
        # Penalize tool through learning
        for error in verification.errors:
            self._math_accuracy_integration.penalize_tool(tool_name, error, self.learner)
        
        # Update shared weight manager
        get_weight_manager().adjust_weight(tool_name, MATH_WEIGHT_ADJUSTMENT_PENALTY)
```

---

## 11. World Model Integration

**Location:** `src/vulcan/world_model/`

The World Model provides the cognitive foundation for understanding and predicting the environment.

### 11.1 Core Components

| Component | File | Description |
|-----------|------|-------------|
| `world_model_core.py` | Core world model | Main orchestration |
| `causal_graph.py` | Causal DAG | Directed acyclic graph of causes |
| `correlation_tracker.py` | Correlation tracking | Statistical relationships |
| `prediction_engine.py` | Prediction | Ensemble prediction with uncertainty |
| `intervention_manager.py` | Interventions | Prioritized intervention scheduling |
| `dynamics_model.py` | Dynamics | Temporal state evolution |
| `confidence_calibrator.py` | Confidence | Probability calibration |
| `invariant_detector.py` | Invariants | Stable pattern detection |
| `world_model_router.py` | Routing | Update routing and prioritization |

#### Causal DAG:

```python
class CausalGraph:
    """Directed acyclic graph of causal relationships."""
    
    # Evidence types for edge creation
    evidence_types = ["observational", "intervention", "linguistic"]
    
    # Methods
    def add_edge(self, cause, effect, strength, evidence_type)
    def remove_edge(self, cause, effect)
    def get_causal_path(self, start, end) -> List[str]
    def detect_cycles(self) -> List[List[str]]
    def break_cycles(self) -> int
```

#### Prediction Engine:

```python
class PredictionEngine:
    """Ensemble prediction with uncertainty quantification."""
    
    def predict(self, variables, context) -> Prediction:
        """
        Returns:
        - value: Predicted value
        - confidence: Calibrated confidence score
        - uncertainty: Uncertainty estimate
        - causal_paths: Relevant causal paths used
        """
```

#### Intervention Manager:

```python
class InterventionManager:
    """Prioritized intervention scheduling with safety validation."""
    
    def schedule_intervention(self, intervention) -> bool:
        """
        Validates safety constraints before scheduling.
        Returns True if intervention was scheduled.
        """
    
    def execute_pending(self) -> List[InterventionResult]:
        """Execute scheduled interventions in priority order."""
```

### 11.2 Meta-Reasoning

**Location:** `src/vulcan/world_model/meta_reasoning/`

The meta-reasoning system enables self-reflection and improvement:

#### EXAMINE → SELECT → APPLY → REMEMBER Pattern:

```
┌─────────────────────────────────────────────────────────────────┐
│                    COGNITIVE CYCLE                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. EXAMINE                                                     │
│     - Validate and analyze observations                         │
│     - Extract variables and patterns                            │
│     - Detect interventions and anomalies                        │
│     - Analyze characteristics (complexity, uncertainty)         │
│                                                                 │
│  2. SELECT                                                      │
│     - Route updates through WorldModelRouter                    │
│     - Prioritize based on constraints                           │
│     - Choose optimal execution plan                             │
│     - Apply VOI (Value of Information) gating                   │
│                                                                 │
│  3. APPLY                                                       │
│     - Execute selected updates                                  │
│     - Update causal graph                                       │
│     - Run interventions if scheduled                            │
│     - Track correlations                                        │
│                                                                 │
│  4. REMEMBER                                                    │
│     - Update state counters                                     │
│     - Validate consistency                                      │
│     - Persist changes                                           │
│     - Update learning weights                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 12. Training Infrastructure

**Location:** `src/training/`

### 12.1 Governed Trainer

**Location:** `src/training/governed_trainer.py`

The GovernedTrainer provides a comprehensive training loop with governance:

#### Key Features:

```
✅ GOVERNANCE
- Every weight update goes through consensus engine
- Proposals include gradient metadata, loss, timing
- Approval required before parameter updates

✅ OPTIMIZATION
- Full Adam optimizer with momentum
- Gradient clipping and scaling
- Gradient accumulation support
- Learning rate scheduling (cosine, linear, exponential, constant)

✅ SAFETY GATES
- Pre-update safety validation
- Post-update divergence detection
- Automatic rollback mechanism
- Anomaly detection (NaN/Inf gradients, explosions, vanishing)

✅ INTEGRATIONS
- World model updates
- RLHF feedback processing
- Memory system writes
- Curriculum progression
- Dynamic architecture adaptation
```

#### Training Step Pipeline:

```
┌─────────────────────────────────────────────────────────────────┐
│                    GOVERNED TRAINING STEP                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Forward Pass ──► Compute Loss                               │
│                         │                                       │
│  2. Gradient Computation ──► Anomaly Detection                  │
│                         │                                       │
│  3. Gradient Clipping ──► Accumulation (if enabled)             │
│                         │                                       │
│  4. Safety Gate ──► Pre-Update Validation                       │
│                         │                                       │
│  5. Optimizer Step ──► Compute Adam Updates                     │
│                         │                                       │
│  6. Governance Proposal ──► Consensus Engine                    │
│                         │                                       │
│  7. Approval Check ──► Apply/Reject Update                      │
│                         │                                       │
│  8. Post-Update ──► Divergence Check                            │
│                         │                                       │
│  9. Rollback (if needed) ──► Restore Checkpoint                 │
│                         │                                       │
│  10. Integration Updates ──► World Model, RLHF, Memory          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Configuration:

```python
trainer = GovernedTrainer(
    agent_id="trainer-0",
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    weight_decay=0.01,
    max_grad_norm=1.0,
    gradient_accumulation_steps=4,
    lr_schedule="cosine",
    warmup_steps=1000,
    total_steps=100000,
    safety_check_interval=10,
    rollback_window=50,
    divergence_threshold=2.0,
)
```

### 12.2 Continual Learning

**Location:** `src/vulcan/learning/continual_learning.py`

#### ContinualLearner (Basic):

```python
class ContinualLearner:
    """Basic continual learner for backward compatibility."""
    
    def process_experience(self, experience: Dict) -> Dict
    def update(self, experience: Dict)
    def consolidate_knowledge()
    async def learn_from_outcome(self, outcome: Dict)
```

#### EnhancedContinualLearner:

```python
class EnhancedContinualLearner(nn.Module):
    """Enhanced continual learning with:
    - Task detection and clustering
    - EWC (Elastic Weight Consolidation)
    - Progressive Neural Networks
    - PackNet parameter isolation
    - Experience replay with intelligent sampling
    - RLHF integration
    - Knowledge Crystallizer
    """
```

#### Continual Learning Metrics:

```python
@dataclass
class ContinualMetrics:
    backward_transfer: float  # Performance change on old tasks
    forward_transfer: float   # Performance on new tasks
    average_accuracy: float   # Average across all tasks
    forgetting_measure: float # Amount of forgetting
    task_accuracies: Dict[str, float]
```

#### EWC (Elastic Weight Consolidation):

```
1. Train on Task A
2. Compute Fisher Information Matrix for Task A
3. Store optimal parameters θ_A*
4. Train on Task B with penalty:
   Loss_B + λ * Σ F_A * (θ - θ_A*)²
5. Repeat for each new task
```

#### Progressive Neural Networks:

```
┌─────────────────────────────────────────────────────────────────┐
│                PROGRESSIVE NEURAL NETWORK                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Column 1 (Task A)    Column 2 (Task B)    Column 3 (Task C)    │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Layer 3    │────│   Layer 3    │────│   Layer 3    │      │
│  ├──────────────┤    ├──────────────┤    ├──────────────┤      │
│  │   Layer 2    │────│   Layer 2    │────│   Layer 2    │      │
│  ├──────────────┤    ├──────────────┤    ├──────────────┤      │
│  │   Layer 1    │────│   Layer 1    │────│   Layer 1    │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│        ↑              ↑ (+ lateral)       ↑ (+ lateral)        │
│      Input           Input                Input                 │
│                                                                 │
│  - Each task gets frozen column                                 │
│  - New tasks add lateral connections                            │
│  - No catastrophic forgetting                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 12.3 Meta-Learning

**Location:** `src/vulcan/learning/meta_learning.py`

#### Supported Algorithms:

| Algorithm | Description |
|-----------|-------------|
| `MAML` | Model-Agnostic Meta-Learning (second-order) |
| `FOMAML` | First-Order MAML (more stable) |
| `REPTILE` | Interpolation-based meta-learning |
| `PROTO` | Prototypical Networks |
| `ANIL` | Almost No Inner Loop (only final layer) |

#### Task Detection:

```python
class TaskDetector:
    """Detect and track learning tasks with persistence and clustering."""
    
    def detect_task(self, experience: Dict) -> str
    def get_related_tasks(self, task_id: str, k: int = 5) -> List[str]
    def predict_next_task(self) -> Optional[str]
    def get_task_difficulty(self, task_id: str) -> float
```

#### MetaLearner:

```python
class MetaLearner:
    """Enhanced Model-Agnostic Meta-Learning."""
    
    def adapt(self, support_set, num_steps, task_id) -> Tuple[nn.Module, Dict]
    def meta_update(self, tasks: List[Dict])
    def online_meta_update(self, experience: Dict)
```

#### Task-Specific Learning Rates:

```python
# Adaptive learning rate per task based on adaptation performance
def _update_task_learning_rate(self, task_id, adapt_stats):
    losses = [step["loss"] for step in adapt_stats["trajectory"]]
    loss_decrease = losses[0] - losses[-1]
    
    if loss_decrease < 0.01:  # Not decreasing enough
        new_lr = min(current_lr * 1.2, 0.1)
    elif loss_decrease > 1.0:  # Too fast (unstable)
        new_lr = max(current_lr * 0.8, 1e-5)
```

### 12.4 RLHF Integration

**Location:** `src/vulcan/learning/rlhf_feedback.py`

#### FeedbackData:

```python
@dataclass
class FeedbackData:
    feedback_id: str
    timestamp: float
    feedback_type: str  # "thumbs_up", "thumbs_down", "correction", etc.
    content: Optional[str]
    context: Dict[str, Any]
    agent_response: Optional[str]
    human_preference: Optional[str]
    reward_signal: float
    metadata: Dict[str, Any]
```

#### RLHFManager:

```python
class RLHFManager:
    """Manages RLHF feedback processing."""
    
    def receive_feedback(self, feedback: FeedbackData)
    def process_batch()
    def compute_reward(self, context, response) -> float
    def update_policy(self, rewards)
```

#### Auto-Detection of Feedback:

```python
# Patterns detected from user messages:
CORRECTION_PATTERNS = [
    r"\b(no|wrong|incorrect|mistake|error)\b",
    r"\b(actually|correction|should be)\b",
]

POSITIVE_PATTERNS = [
    r"\b(thanks|great|perfect|exactly|correct)\b",
    r"\b(awesome|excellent|helpful)\b",
]

NEGATIVE_PATTERNS = [
    r"\b(bad|terrible|useless|unhelpful)\b",
    r"\b(doesn't work|broken|failed)\b",
]
```

#### Live Feedback Processing:

```python
class LiveFeedbackProcessor:
    """Process real-time feedback from user messages."""
    
    async def process_live_feedback(self, feedback_data: Dict)
    def detect_implicit_feedback(self, user_message: str) -> Optional[FeedbackData]
```

---

## 13. Context Management System

**Location:** `src/context/`

The context management system provides sophisticated memory and context management for LLM generation.

### 13.1 Hierarchical Context Memory

**Location:** `src/context/hierarchical_context.py`

A three-tier memory system for comprehensive context management:

```
┌─────────────────────────────────────────────────────────────────────┐
│                   HIERARCHICAL CONTEXT MEMORY                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    EPISODIC MEMORY                           │   │
│  │  Recent prompt/response pairs with full reasoning traces     │   │
│  │  - Access tracking (count, last_accessed)                    │   │
│  │  - Importance scoring                                        │   │
│  │  - Consolidation flags                                       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼ consolidate                          │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    SEMANTIC MEMORY                           │   │
│  │  Concept index with clustering and relationships             │   │
│  │  - Term indexing for fast lookup                             │   │
│  │  - Frequency and importance tracking                         │   │
│  │  - Related concepts graph                                    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼ patterns                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                   PROCEDURAL MEMORY                          │   │
│  │  Learned patterns, strategies, and procedures                │   │
│  │  - Strategy signatures                                       │   │
│  │  - Success rate tracking                                     │   │
│  │  - Latency metrics                                           │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### Memory Data Structures:

```python
@dataclass
class EpisodicItem:
    """Episodic memory with comprehensive metadata."""
    prompt: Any
    token: Any
    trace: Any
    ts: float = field(default_factory=time.time)
    importance: float = 1.0
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    consolidated: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SemanticEntry:
    """Semantic memory entry with relationships."""
    concept: str
    terms: List[str]
    freq: int = 1
    last_seen: float = field(default_factory=time.time)
    importance: float = 1.0
    cluster_id: Optional[int] = None
    related_concepts: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None

@dataclass
class ProceduralPattern:
    """Procedural memory pattern."""
    name: str
    signature_terms: List[str]
    freq: int = 1
    success_rate: float = 1.0
    avg_latency_ms: float = 0.0
```

#### Retrieval Strategies:

| Strategy | Description |
|----------|-------------|
| `RECENT` | Most recent episodic items |
| `RELEVANT` | Highest overlap with query terms |
| `DIVERSE` | Maximize concept diversity |
| `BALANCED` | Combine recent + relevant |

#### Consolidation Strategies:

| Strategy | Criteria |
|----------|----------|
| `FREQUENCY` | Items accessed >= min_freq times |
| `RECENCY` | Items within half-life window |
| `IMPORTANCE` | Items with importance > 0.7 |
| `HYBRID` | Combined scoring of all factors |

#### Pruning Strategies:

| Strategy | Description |
|----------|-------------|
| `DECAY` | Time-weighted importance decay |
| `LRU` | Least recently accessed |
| `FREQUENCY` | Least frequently accessed |
| `IMPORTANCE` | Lowest importance score |

#### Key Methods:

```python
class HierarchicalContext:
    def retrieve(self, query, max_items=10, strategy=RetrievalStrategy.BALANCED):
        """Retrieve relevant items from all memory tiers."""
    
    def retrieve_context_for_generation(self, query_tokens, max_tokens=2048):
        """Get generation-ready context bundle with flat concatenation."""
    
    def store(self, prompt, token, reasoning_trace, importance=1.0):
        """Store interaction and update all memory tiers."""
    
    def consolidate_memory(self, strategy=ConsolidationStrategy.HYBRID):
        """Move episodic memories to semantic memory."""
    
    def prune_memory(self, strategy=PruningStrategy.DECAY, target_reduction=0.2):
        """Intelligent memory pruning."""
    
    def export_memory(self) -> Dict[str, Any]:
        """Export for persistence."""
    
    def import_memory(self, data: Dict[str, Any]):
        """Import from exported data."""
```

### 13.2 Causal Context Selector

**Location:** `src/context/causal_context.py`

Advanced causal reasoning for context selection:

#### Causal Capabilities:

| Capability | Description |
|------------|-------------|
| Multi-hop traversal | Navigate causal graph up to N hops |
| Temporal reasoning | Time-series analysis with decay |
| Intervention tracking | Do-calculus operations |
| Counterfactual analysis | "What-if" scenarios |
| Confounder detection | Identify common causes |
| Mediator identification | Find intermediate variables |

#### Causal Strength Measurement:

```python
class CausalStrengthType(Enum):
    CORRELATION = "correlation"        # Simple correlation
    GRANGER = "granger"               # Granger causality
    TRANSFER_ENTROPY = "transfer_entropy"  # Information-theoretic
    INTERVENTION = "intervention"      # Do-calculus based
    COUNTERFACTUAL = "counterfactual" # Counterfactual queries
```

#### Temporal Decay Functions:

```python
class TemporalDecayFunction(Enum):
    EXPONENTIAL = "exponential"  # 0.5^(t/half_life)
    HYPERBOLIC = "hyperbolic"    # 1/(1 + t/3600)
    POWER_LAW = "power_law"      # 1/(1 + t/3600)^alpha
    LINEAR = "linear"            # max(0, 1 - t/half_life)
```

#### Key Methods:

```python
class CausalContext:
    def select(self, world_model, query) -> Dict[str, Any]:
        """Select causally-relevant context."""
        # Returns: causal_context, concepts, causal_graph,
        #          interventions, confounders, mediators
    
    def record_intervention(self, variable, value, effect_on=None):
        """Record a causal intervention."""
    
    def compute_counterfactual(self, world_model, variable, original_value,
                                counterfactual_value, outcome_variable):
        """Compute counterfactual scenario."""
```

#### Output Structure:

```python
{
    "causal_context": [
        {
            "source": "episodic|semantic|procedural",
            "score": float,
            "item": <original>,
            "reason": str,
            "causal_path": List[str],
            "causal_strength": float,
            "temporal_relevance": float,
        }
    ],
    "concepts": List[str],
    "causal_graph": Dict,
    "interventions": List[Dict],
    "confounders": List[str],
    "mediators": List[str],
    "statistics": CausalStatistics,
}
```

---

## 14. Generation Systems

**Location:** `src/generation/`

### 14.1 Safe Generation

**Location:** `src/generation/safe_generation.py`

Multi-layered safety filtering for token generation:

#### Risk Levels:

```python
class RiskLevel(Enum):
    SAFE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
```

#### Validation Categories:

| Category | Description |
|----------|-------------|
| `TOXICITY` | Harmful, offensive content |
| `HALLUCINATION` | Factually incorrect claims |
| `PROMPT_INJECTION` | Injection attacks |
| `PII` | Personal identifiable information |
| `BIAS` | Discriminatory content |
| `CONSISTENCY` | Logical coherence |
| `PROFANITY` | Explicit language |
| `VIOLENCE` | Violent content |
| `HATE_SPEECH` | Hate speech |
| `SEXUAL_CONTENT` | Adult content |
| `MEDICAL_HARM` | Dangerous medical advice |
| `LEGAL_VIOLATION` | Illegal content |

#### Validator Integration:

```python
# Integrates with VULCAN validators when available
from src.vulcan.safety.llm_validators import (
    ToxicityValidator,
    HallucinationValidator,
    PromptInjectionValidator,
    PIIValidator,
    BiasValidator,
    EnhancedSafetyValidator,
)
```

#### Features:

- **Multi-tier validation**: Token-level and sequence-level
- **Adaptive thresholds**: Domain and context-aware
- **Real-time monitoring**: Alerting for safety events
- **Audit trails**: Complete provenance tracking
- **Safe alternatives**: Suggest replacement tokens
- **Caching**: Performance optimization

### 14.2 Explainable Generation

**Location:** `src/generation/explainable_generation.py`

Comprehensive AI explainability system:

#### Explanation Levels:

```python
class ExplanationLevel(Enum):
    MINIMAL = "minimal"           # Just the choice
    BASIC = "basic"              # Choice + top alternatives
    STANDARD = "standard"        # Basic + factors + confidence
    DETAILED = "detailed"        # Standard + attributions + context
    COMPREHENSIVE = "comprehensive"  # Everything + counterfactuals
```

#### Attribution Methods:

```python
class AttributionMethod(Enum):
    GRADIENT = "gradient"                    # Gradient-based
    ATTENTION = "attention"                  # Attention weights
    INTEGRATED_GRADIENTS = "integrated_gradients"
    SHAPLEY = "shapley"                      # SHAP values
    LIME = "lime"                            # Local interpretable
```

#### Explanation Components:

```python
@dataclass
class DecisionSummary:
    token: Token
    token_str: str
    position: Optional[int]
    prob: Optional[float]
    confidence: Optional[float]
    entropy: Optional[float]
    strategy: Optional[str]
    temperature: Optional[float]
    perplexity: Optional[float]
    uncertainty: Optional[float]

@dataclass
class FeatureAttribution:
    feature_name: str
    importance: float
    contribution: float
    method: AttributionMethod
    details: Dict[str, Any]

@dataclass
class CounterfactualAnalysis:
    alternative_token: Token
    alternative_prob: float
    scenario_description: str
    outcome_difference: str
    plausibility: float
```

#### Key Methods:

```python
class ExplainableGeneration:
    def explain(self, token, chain, hidden_state=None, logits=None,
                candidates=None, prompt_tokens=None, level=ExplanationLevel.STANDARD):
        """Generate comprehensive explanation for a token decision."""
    
    def explain_sequence(self, tokens, explanations):
        """Generate explanation for full sequence."""
    
    def get_feature_attributions(self, token, hidden_state, method=AttributionMethod.ATTENTION):
        """Get feature importance attributions."""
    
    def generate_counterfactuals(self, token, alternatives, context):
        """Generate what-if scenarios for alternatives."""
```

### 14.3 Unified Generation

**Location:** `src/generation/unified_generation.py`

Multi-strategy ensemble generation (covered in Section 15).

---

## 15. Tool Selection System

**Location:** `src/vulcan/reasoning/selection/`

### 13.1 Production Tool Selector

**Location:** `src/vulcan/reasoning/selection/tool_selector.py`

The tool selection system determines which reasoning tools to use for a given problem:

#### Selection Pipeline:

```
┌─────────────────────────────────────────────────────────────────┐
│                    TOOL SELECTION PIPELINE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Admission Control ──► Rate limiting, resource checks        │
│                         │                                       │
│  2. Feature Extraction ──► Problem embedding                    │
│                         │                                       │
│  3. Cache Lookup ──► Check for cached selection                 │
│                         │                                       │
│  4. Memory Prior ──► Similar past problems                      │
│                         │                                       │
│  5. Semantic Matching ──► Tool-problem similarity               │
│                         │                                       │
│  6. Cost Estimation ──► Predict execution cost                  │
│                         │                                       │
│  7. Utility Computation ──► Expected utility per tool           │
│                         │                                       │
│  8. Safety Check ──► Validate tool safety                       │
│                         │                                       │
│  9. Portfolio Assembly ──► Select complementary tools           │
│                         │                                       │
│  10. Warm Pool Priming ──► Pre-initialize selected tools        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Components:

| Component | File | Purpose |
|-----------|------|---------|
| `tool_selector.py` | Main selector | Orchestrates selection |
| `cost_model.py` | Cost estimation | Predicts latency/energy |
| `utility_model.py` | Utility scoring | Expected value calculation |
| `safety_governor.py` | Safety checks | Validates tool safety |
| `memory_prior.py` | Historical patterns | Past problem similarity |
| `semantic_tool_matcher.py` | Semantic matching | Tool-problem alignment |
| `selection_cache.py` | Caching | Result caching |
| `warm_pool.py` | Pre-initialization | Tool warm-up |
| `portfolio_executor.py` | Parallel execution | Multi-tool execution |
| `admission_control.py` | Rate limiting | Resource management |

### 13.2 Cost Model

**Location:** `src/vulcan/reasoning/selection/cost_model.py`

```python
@dataclass
class CostEstimate:
    latency_ms: float      # Expected execution time
    energy_mj: float       # Expected energy consumption
    memory_mb: float       # Expected memory usage
    confidence: float      # Estimation confidence
    breakdown: Dict[str, float]  # Per-component costs
```

#### Cost Factors:

```python
COST_FACTORS = {
    "symbolic": {
        "base_latency_ms": 50,
        "complexity_factor": 1.2,
        "memory_factor": 0.8,
    },
    "probabilistic": {
        "base_latency_ms": 100,
        "complexity_factor": 1.5,
        "memory_factor": 1.2,
    },
    "causal": {
        "base_latency_ms": 150,
        "complexity_factor": 2.0,
        "memory_factor": 1.5,
    },
}
```

### 13.3 Utility Model

**Location:** `src/vulcan/reasoning/selection/utility_model.py`

```python
@dataclass
class UtilityEstimate:
    expected_utility: float   # Overall expected utility
    quality_score: float      # Expected quality
    risk_score: float         # Risk estimate
    confidence: float         # Estimation confidence
    breakdown: Dict[str, float]
```

#### Utility Calculation:

```python
def compute_utility(
    tool: str,
    problem_features: np.ndarray,
    context: UtilityContext,
) -> UtilityEstimate:
    """
    Utility = quality * (1 - risk) - cost_penalty
    
    Where:
    - quality: Expected result quality
    - risk: Probability of failure
    - cost_penalty: Weighted cost (time, energy)
    """
```

### 13.4 Safety Governor

**Location:** `src/vulcan/reasoning/selection/safety_governor.py`

```python
class SafetyGovernor:
    """Validates tool safety before execution."""
    
    def validate_tool(self, tool: str, context: Dict) -> SafetyResult
    def check_rate_limits(self, tool: str) -> bool
    def validate_inputs(self, tool: str, inputs: Dict) -> bool
    def log_safety_event(self, event: SafetyEvent)
```

#### Safety Checks:

| Check | Description |
|-------|-------------|
| Rate Limits | Tool-specific rate limiting |
| Input Validation | Sanitize and validate inputs |
| Resource Limits | Memory, CPU, time limits |
| Output Validation | Verify output safety |
| Audit Trail | Log all tool executions |

---

## 14. Unified Generation System

**Location:** `src/generation/unified_generation.py`

The UnifiedGeneration system combines multiple reasoning strategies into an ensemble:

### Fusion Strategies:

| Strategy | Description |
|----------|-------------|
| `WEIGHTED_SUM` | Sum of weighted module scores |
| `PRODUCT` | Geometric mean of probabilities |
| `MAX` | Take highest weighted score |
| `RANK_FUSION` | Reciprocal Rank Fusion (RRF) |
| `BORDA_COUNT` | Voting-based scoring |

### Normalization Methods:

| Method | Description |
|--------|-------------|
| `SOFTMAX` | Standard softmax normalization |
| `MIN_MAX` | Scale to [0, 1] range |
| `Z_SCORE` | Z-score standardization |
| `RANK` | Rank-based normalization |

### Configuration:

```python
@dataclass
class UnifiedGenConfig:
    max_candidates: int = 10
    default_module_weights: Dict[str, float] = {
        "language": 1.0,
        "symbolic": 1.0,
        "probabilistic": 1.0,
        "causal": 1.2,       # Causal reasoning boosted
        "analogical": 0.8,   # Creative, slightly risky
        "meta_cognitive": 1.1,
        "evolutionary": 0.7,
        "adversarial": 0.6,
        "hierarchical": 0.9,
    }
    fusion_strategy: FusionStrategy = FusionStrategy.WEIGHTED_SUM
    temperature: float = 1.0
    diversity_penalty: float = 0.0
    enable_cross_module_interaction: bool = True
    enable_dynamic_weights: bool = True
```

### Cross-Module Interaction:

```python
def _model_cross_module_interactions(self, per_module, weights):
    """Boost candidates with cross-module agreement."""
    
    # Find tokens proposed by multiple modules
    for tok, modules in all_tokens.items():
        if len(modules) > 1:
            # Boost based on agreement count
            boost = 1.0 + 0.1 * (len(modules) - 1)
            prop["module_prob"] *= boost
```

### Candidate Metadata:

```python
@dataclass
class CandidateMetadata:
    token: Any
    score: float
    prob: float
    logit: float
    rank: int
    provenance: List[Dict[str, Any]]  # Which modules contributed
    confidence: float
    diversity_score: float
    module_agreement: int             # Number of modules proposing
    uncertainty: float
```

---

## 15. Graph Compiler Infrastructure

**Location:** `src/compiler/`

### Graph Compiler

**Location:** `src/compiler/graph_compiler.py`

Compiles JSON graph representations to optimized native machine code:

#### Node Types:

```python
class NodeType(Enum):
    INPUT = "InputNode"
    OUTPUT = "OutputNode"
    CONST = "CONST"
    ADD = "ADD"
    MUL = "MUL"
    MATRIX_MUL = "MATRIX_MUL"
    RELU = "RELU"
    SOFTMAX = "SOFTMAX"
    CONV2D = "CONV2D"
    BATCH_NORM = "BATCH_NORM"
    EMBEDDING = "EMBEDDING"
    ATTENTION = "ATTENTION"
    PHOTONIC_MVM = "PhotonicMVMNode"  # Analog photonic emulation
    DYNAMIC_CODE = "DynamicCodeNode"
    GENERATIVE_AI = "GenerativeAINode"
```

#### Graph Optimizer:

```python
class GraphOptimizer:
    """Optimizes graph before compilation."""
    
    def optimize(self, graph: nx.DiGraph) -> nx.DiGraph:
        graph = self._fuse_operations(graph)      # Op fusion
        graph = self._eliminate_dead_code(graph)  # DCE
        graph = self._constant_folding(graph)     # Constant evaluation
        graph = self._common_subexpression_elimination(graph)  # CSE
        return graph
```

#### Fusable Patterns:

```python
patterns = [
    ["CONV2D", "BATCH_NORM", "RELU"],  # Conv + BN + ReLU
    ["MATRIX_MUL", "ADD"],              # MatMul + Bias
    ["ADD", "ADD"],                     # Multiple adds
]
```

### LLVM Backend

**Location:** `src/compiler/llvm_backend.py`

#### Data Types:

```python
class DataType(Enum):
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    INT32 = "int32"
    INT64 = "int64"
    BOOL = "bool"
```

#### Compilation Pipeline:

```
┌─────────────────────────────────────────────────────────────────┐
│                    COMPILATION PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  JSON Graph ──► Parse ──► NetworkX DiGraph                      │
│                                │                                │
│                                ▼                                │
│  Graph Optimization ──► Fusion, DCE, CSE, Folding               │
│                                │                                │
│                                ▼                                │
│  LLVM IR Generation ──► Create Module, Functions                │
│                                │                                │
│                                ▼                                │
│  LLVM Optimization ──► Passes (O2, vectorization)               │
│                                │                                │
│                                ▼                                │
│  Code Generation ──► Native Machine Code                        │
│                                │                                │
│                                ▼                                │
│  JIT Execution or Binary Output                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Hybrid Executor

**Location:** `src/compiler/hybrid_executor.py`

Executes graphs using both compiled code and Python fallback:

```python
class HybridExecutor:
    """
    Executes compiled graphs with fallback for unsupported ops.
    
    Features:
    - JIT compilation for hot paths
    - Python fallback for dynamic ops
    - Caching of compiled functions
    - Profiling and optimization hints
    """
```

---

## 16. Self-Improvement System

**Location:** `src/vulcan/world_model/meta_reasoning/`

### Self-Improvement Drive:

The autonomous self-improvement system includes:

#### Trigger Types:
- Startup initialization
- Error detection
- Performance degradation
- Idle period optimization
- Resource availability

#### Execution Pipeline:

```
1. Context Building ──► 2. Trigger Check ──► 3. Action Generation
        │                      │                      │
        ▼                      ▼                      ▼
   Performance          Self-Improvement         LLM-Driven
   Metrics              Drive Evaluation         Code Generation
        │                      │                      │
        └──────────────────────┴──────────────────────┘
                               │
                               ▼
4. AST Validation ──► 5. Diff Application ──► 6. Git Commit
        │                      │                    │
        ▼                      ▼                    ▼
   Syntax Check           File System          Version Control
        │                   Update                  │
        └──────────────────────┴──────────────────┘
                               │
                               ▼
                    7. Learning/Outcome Recording
```

#### Safety Controls:
- Kill switch: `VULCAN_ENABLE_SELF_IMPROVEMENT=0`
- Auto-commit disable: `VULCAN_SELF_IMPROVEMENT_AUTO_COMMIT=false`
- Configurable check interval (default: 24 hours)

---

## 17. Safety and Governance

### Safety Layers:

1. **Input Validation**
   - Sanitization of user inputs
   - Pattern-based threat detection
   - Content filtering

2. **Reasoning Validation**
   - SafetyAwareReasoning wrapper
   - Confidence threshold filtering
   - Output safety checks

3. **Causal Safety**
   - Causal edge validation
   - Intervention safety checks
   - Path validation

4. **Self-Improvement Safety**
   - AST validation before code application
   - Git-based rollback capability
   - Manual approval options

### Governance Features:

- **Trust-Weighted Consensus**: For proposals and changes
- **Audit Trail**: Comprehensive logging of all operations
- **CSIU Framework**: Communication, Safety, Identity, Utility constraints
- **Alignment Validation**: Motivational introspection for proposal alignment

---

## 18. LLM Backend Package

**Location:** `src/vulcan/llm/`

The LLM backend package provides multi-backend LLM integration.

### 18.1 Package Structure

```
src/vulcan/llm/
├── __init__.py          # Package exports
├── hybrid_executor.py   # Multi-backend executor
├── mock_llm.py          # Mock implementation for testing
└── openai_client.py     # OpenAI API client
```

### 18.2 OpenAI Client

**Location:** `src/vulcan/llm/openai_client.py`

Lazy-initialized OpenAI client with error tracking:

```python
# Configuration
# Environment Variable: OPENAI_API_KEY

# Client Functions
def get_openai_client() -> Optional[OpenAI]:
    """Get lazily-initialized OpenAI client."""

def get_openai_init_error() -> Optional[str]:
    """Get initialization error for diagnostics."""

def initialize_openai_client(api_key: Optional[str] = None) -> bool:
    """Explicitly initialize with optional key."""

def is_openai_ready() -> bool:
    """Check if client is ready for use."""

# Availability Check
OPENAI_AVAILABLE: bool  # True if openai package installed
```

### 18.3 Mock LLM

**Location:** `src/vulcan/llm/mock_llm.py`

Mock implementation for testing and fallback:

```python
class MockGraphixVulcanLLM:
    """Mock LLM for safe execution when real package unavailable."""
    
    def __init__(self, config_path: str):
        self.bridge = MagicMock()  # Mock bridge for reasoning
    
    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        """Return mock response."""
    
    def reason(self, query: str, context: Optional[dict] = None) -> str:
        """Mock reasoning operation."""
    
    def explain(self, concept: str) -> str:
        """Mock explanation operation."""
    
    def get_stats(self) -> dict:
        """Get generation statistics including is_mock flag."""

# Auto-fallback
GraphixVulcanLLM = RealGraphixVulcanLLM or MockGraphixVulcanLLM
GRAPHIX_LLM_AVAILABLE: bool
```

### 18.4 Hybrid LLM Executor

**Location:** `src/vulcan/llm/hybrid_executor.py`

Multi-backend execution with fallback and ensemble modes:

#### Execution Modes:

| Mode | Description |
|------|-------------|
| `local_first` | Try local LLM, fallback to OpenAI |
| `openai_first` | Try OpenAI, fallback to local |
| `parallel` | Race both, use first success |
| `ensemble` | Run both, combine/select best |

#### Key Features:

```python
class HybridLLMExecutor:
    # Constants
    MIN_MEANINGFUL_LENGTH = 10
    ENSEMBLE_LOCAL_RESPONSE_MAX_LENGTH = 500
    
    DEFAULT_SYSTEM_PROMPT = (
        "You are VULCAN, an advanced AI assistant. "
        "You SHOULD remember and reference information shared earlier..."
    )
    
    def __init__(
        self,
        local_llm: Optional[Any] = None,
        openai_client_getter: Optional[Callable] = None,
        mode: str = "parallel",
        timeout: float = 30.0,
        ensemble_min_confidence: float = 0.7,
        openai_max_tokens: int = 2000,
    )
    
    async def execute(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        enable_distillation: bool = True,  # Capture for training
        conversation_history: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """Execute with configured mode and distillation capture."""
```

#### Distillation Integration:

```python
# When enable_distillation=True and source includes OpenAI:
if enable_distillation and result.get("source") in ("openai", "parallel_both", "ensemble"):
    self._capture_for_distillation(prompt, result)
```

---

## 19. Knowledge Distillation System

**Location:** `src/vulcan/distillation/`

Comprehensive knowledge distillation from OpenAI to local LLM.

### 19.1 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                KNOWLEDGE DISTILLATION PIPELINE                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  User Request (with opt-in)                                         │
│         │                                                           │
│         ▼                                                           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                  CAPTURE LAYER                               │   │
│  │  OpenAIKnowledgeDistiller.capture_response()                 │   │
│  │                                                              │   │
│  │  Stage 1: Opt-In Check ─────► REJECT if not opted in         │   │
│  │  Stage 2: PII Redaction ────► Mask emails, phones, SSN       │   │
│  │  Stage 3: Secret Detection ─► REJECT if API keys found       │   │
│  │  Stage 4: Governance Check ─► REJECT if sensitive content    │   │
│  │  Stage 5: Quality Validation─► REJECT if low quality         │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼ (passed all stages)                  │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                  STORAGE LAYER                               │   │
│  │  DistillationStorageBackend                                  │   │
│  │  - JSONL format (appendable)                                 │   │
│  │  - Optional Fernet encryption                                │   │
│  │  - Provenance hashes                                         │   │
│  │  - Retention limits                                          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼ (training worker reads)              │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                  TRAINING LAYER                              │   │
│  │  GovernedTrainer + ConsensusEngine                           │   │
│  │  - Proposes weight updates                                   │   │
│  │  - Consensus approval/rejection                              │   │
│  │  - Rollback on regression                                    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                  EVALUATION LAYER                            │   │
│  │  ShadowModelEvaluator + PromotionGate                        │   │
│  │  - Golden test set evaluation                                │   │
│  │  - Regression detection                                      │   │
│  │  - Promotion/rollback decision                               │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 19.2 Distillation Example Model

**Location:** `src/vulcan/distillation/models.py`

```python
@dataclass
class DistillationExample:
    """Training example with full provenance tracking."""
    
    # Core content
    instruction: str          # Sanitized prompt (PII redacted)
    teacher_answer: str       # OpenAI response
    context: Dict[str, Any]   # Routing metadata, tools used
    labels: Dict[str, Any]    # Domain, difficulty, validation
    
    # Provenance tracking
    prompt_hash: str          # SHA256 of original prompt
    response_hash: str        # SHA256 of response
    teacher_model: str        # e.g., "gpt-3.5-turbo"
    timestamp: float
    
    # Quality metrics
    quality_score: float
    validation_passed: bool
    rejection_reasons: List[str]
    
    # Governance
    session_opted_in: bool
    retention_expires: Optional[float]
```

### 19.3 PII Redactor

**Location:** `src/vulcan/distillation/pii_redactor.py`

Detects and masks sensitive information:

#### PII Patterns:

| Type | Pattern |
|------|---------|
| Email | `user@domain.com` → `[EMAIL]` |
| Phone | `(555) 123-4567` → `[PHONE]` |
| SSN | `123-45-6789` → `[SSN]` |
| Credit Card | `4111-1111-1111-1111` → `[CREDIT_CARD]` |
| IP Address | `192.168.1.1` → `[IP_ADDRESS]` |

#### Secret Patterns (Hard Reject):

| Type | Pattern |
|------|---------|
| OpenAI Key | `sk-xxxx...` |
| AWS Access Key | `AKIA...` |
| GitHub Token | `ghp_...` |
| Bearer Token | `Bearer xxx...` |
| JWT | `eyJ...` |
| Password Fields | `password: xxx` |
| Connection Strings | `mongodb://...` |
| Private Keys | `-----BEGIN PRIVATE KEY-----` |

### 19.4 Governance Sensitivity Checker

**Location:** `src/vulcan/distillation/governance_checker.py`

Checks content against governance rules:

#### Sensitive Categories (Never Capture):

| Category | Examples |
|----------|----------|
| `auth_credentials` | Login, password, bearer tokens |
| `payment_info` | Credit cards, CVV, bank accounts |
| `medical_phi` | Diagnosis, prescriptions, HIPAA |
| `legal_privileged` | Attorney-client, legal advice |

#### Do-Not-Capture Markers:

```python
DO_NOT_CAPTURE_MARKERS = [
    "[CONFIDENTIAL]",
    "[DO NOT LOG]",
    "[SENSITIVE]",
    "[PRIVATE]",
    "[NO_TRAINING]",
    "[GOVERNANCE_RESTRICTED]",
]
```

### 19.5 Quality Validator

**Location:** `src/vulcan/distillation/quality_validator.py`

Multi-stage quality filtering:

#### Thresholds:

| Threshold | Default | Description |
|-----------|---------|-------------|
| `MIN_RESPONSE_LENGTH` | 50 | Minimum chars |
| `MAX_RESPONSE_LENGTH` | 4000 | Maximum chars |
| `MIN_QUALITY_SCORE` | 0.65 | Quality threshold |
| `MAX_BOILERPLATE_RATIO` | 0.4 | Max filler content |

#### Refusal Patterns (Rejected):

```python
REFUSAL_PATTERNS = [
    "i cannot", "i can't", "i'm not able to",
    "as an ai", "as a language model",
    "i apologize, but", "i'm sorry, but i cannot",
]
```

#### Boilerplate Patterns (Reduce Score):

```python
BOILERPLATE_PATTERNS = [
    "sure, ", "of course, ", "certainly, ",
    "great question", "good question",
    "here's", "let me", "i'd be happy to",
    "i hope this helps", "feel free to ask",
]
```

### 19.6 Shadow Model Evaluator

**Location:** `src/vulcan/distillation/evaluator.py`

Evaluates improvements before promotion:

#### Golden Test Set:

```python
GOLDEN_PROMPTS = [
    {"prompt": "What is 2 + 2?", 
     "expected_contains": ["4"], "domain": "math"},
    {"prompt": "Write a Python function that adds two numbers.",
     "expected_contains": ["def", "return", "+"], "domain": "code"},
    {"prompt": "Explain machine learning in one sentence.",
     "expected_contains": ["learn", "data"], "domain": "explanation"},
    {"prompt": "What is the capital of France?",
     "expected_contains": ["Paris"], "domain": "factual"},
]
```

#### Regression Detection:

- Threshold: 10% drop from baseline triggers failure
- Domain-specific metrics tracked
- History maintained for trend analysis

### 19.7 Promotion Gate

**Location:** `src/vulcan/distillation/promotion_gate.py`

Explicit gate for weight promotion:

#### Requirements:

| Requirement | Default | Description |
|-------------|---------|-------------|
| `MIN_EVAL_SCORE` | 0.7 | Minimum evaluation score |
| `MAX_REGRESSION_COUNT` | 0 | No regressions allowed |
| Provenance Record | Required | Full audit trail |

#### Decision Flow:

```python
def evaluate_for_promotion(eval_results, training_metadata):
    """
    Returns (approved: bool, decision_details: Dict)
    
    Checks:
    1. eval_score >= MIN_EVAL_SCORE
    2. regression_count <= MAX_REGRESSION_COUNT
    3. provenance_valid == True
    
    All must pass for approval.
    """
```

### 19.8 Storage Backend

**Location:** `src/vulcan/distillation/storage.py`

JSONL storage with optional encryption:

```python
class DistillationStorageBackend:
    def __init__(
        self,
        storage_path: str = "data/distillation",
        use_encryption: bool = False,
        encryption_key: Optional[str] = None,  # Fernet key
        max_file_size_mb: int = 100,
    )
    
    def append_example(self, example: Dict[str, Any]) -> bool:
        """Thread-safe append to JSONL."""
    
    def read_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """Read batch for training."""
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
```

---

## 20. Performance Instrumentation

**Location:** `src/llm_core/graphix_executor.py`

### Timing Decorator:

```python
def timed_operation(operation_name: str) -> Callable:
    """Decorator to time operations and log performance."""
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            
            # Update global stats
            stats = _PERF_STATS[operation_name]
            stats["count"] += 1
            stats["total_ms"] += elapsed_ms
            stats["min_ms"] = min(stats["min_ms"], elapsed_ms)
            stats["max_ms"] = max(stats["max_ms"], elapsed_ms)
            
            # Log slow operations (>50ms)
            if elapsed_ms > 50.0:
                logger.debug("[PERF] %s: %.1fms", operation_name, elapsed_ms)
            
            return result
        return wrapper
    return decorator
```

### Performance Statistics API:

```python
# Get all performance stats
stats = get_performance_stats()
# Returns: {"op_name": {"count": N, "total_ms": T, "avg_ms": A, "min_ms": M, "max_ms": X}}

# Reset stats
reset_performance_stats()

# Log summary (sorted by total time, hottest first)
log_performance_summary()
```

### Execution Metrics:

```python
@dataclass
class ExecutionMetrics:
    total_executions: int = 0
    total_tokens_processed: int = 0
    total_time_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    memory_peak_mb: float = 0.0
    flops_total: float = 0.0
    layer_times: Dict[str, float]
    
    def get_avg_time_per_token(self) -> float
    def get_cache_hit_rate(self) -> float
    def to_dict(self) -> Dict[str, Any]
```

### Layer-Level Profiling:

```python
# Fine-grained layer timing (logged when total > 200ms)
timings = {
    "ln1_ms": ...,           # Pre-attention layer norm
    "attention_ms": ...,     # Attention computation
    "residual1_ms": ...,     # First residual connection
    "ln2_ms": ...,           # Pre-FFN layer norm
    "ffn_ms": ...,           # Feed-forward network
    "residual2_ms": ...,     # Second residual connection
}

# FFN sub-operation timing (logged when total > 100ms)
ffn_timings = {
    "gate_proj_ms": ...,     # Gate projection
    "up_proj_ms": ...,       # Up projection
    "swiglu_ms": ...,        # SwiGLU activation
    "down_proj_ms": ...,     # Down projection
}
```

### Profiler Integration:

```python
class PerformanceProfiler:
    """Profiles execution performance."""
    
    def __init__(self, enabled: bool = False):
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.memory_snapshots: List[Tuple[str, float]] = []
        
    def start_timer(self, name: str) -> float
    def end_timer(self, name: str, start_time: float) -> float
    def record_memory(self, name: str, bytes_used: float) -> None
    def get_summary(self) -> Dict[str, Any]
```

### Audit Logging:

```python
class AuditLogger:
    """Logs execution for compliance and debugging."""
    
    def log(self, event_type: str, details: Dict[str, Any]):
        entry = {
            "timestamp": time.time(),
            "type": event_type,
            "details": details
        }
        # Append to in-memory list and optionally write to file
```

---

## 19. Configuration and Tuning

### Environment Variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `VULCAN_MAX_WORKERS` | Thread pool size | 2 |
| `VULCAN_TEST_MODE` | Enable test mode | false |
| `VULCAN_PRODUCTION` | Force production mode | false |
| `VULCAN_ENV` | Environment (test/production) | - |
| `VULCAN_ENABLE_SELF_IMPROVEMENT` | Enable self-improvement | 1 |
| `SELF_IMPROVEMENT_INTERVAL` | Check interval (seconds) | 86400 |
| `VULCAN_LLM_API_KEY` | External LLM API key | - |
| `VULCAN_FORCE_PRODUCTION_REASONING` | Force production reasoning | false |

### Model Size Guidelines:

| Configuration | Layers | Hidden | Heads | Params (Est.) | Memory |
|---------------|--------|--------|-------|---------------|--------|
| Small | 3 | 128 | 4 | ~1M | ~4MB |
| Medium | 6 | 512 | 8 | ~30M | ~120MB |
| Large | 12 | 768 | 12 | ~120M | ~480MB |
| Default | 6 | 256 | 4 | ~10M | ~40MB |

### Tuning Guidelines:

#### For Faster Inference:
```python
config = ExecutorConfig(
    mode=ExecutionMode.INFERENCE,
    precision=PrecisionMode.FP16,
    attention_impl=AttentionImpl.FLASH,
    use_kv_cache=True,
    use_kernel_fusion=True,
)
```

#### For Better Quality:
```python
config = GraphixTransformerConfig(
    num_layers=12,
    hidden_size=768,
    num_heads=12,
)

# Generation settings
model.generate(
    prompt,
    temperature=0.7,
    top_p=0.9,
    max_new_tokens=256
)
```

#### For Memory Efficiency:
```python
config = GraphixTransformerConfig(
    gradient_checkpointing=True,
    lora_rank=8,  # Use LoRA instead of full fine-tuning
)

executor_config = ExecutorConfig(
    use_quantization=True,
    quantization_bits=8,
    optimize_memory=True,
)
```

---

## 20. Code Examples

### Basic Text Generation:

```python
from src.llm_core.graphix_transformer import GraphixTransformer, GraphixTransformerConfig

# Create model
config = GraphixTransformerConfig(
    num_layers=6,
    hidden_size=256,
    num_heads=4,
    vocab_size=4096,
)
model = GraphixTransformer(config)

# Generate text
text = model.generate(
    prompt="The future of AI is",
    max_new_tokens=50,
    temperature=0.8,
    top_p=0.9,
)
print(text)
```

### Using LocalGPTProvider:

```python
from src.local_llm.provider.local_gpt_provider import LocalGPTProvider, ProviderInitConfig

# Initialize from config
provider = LocalGPTProvider.from_config_file("provider_config.json")

# Single generation
text, meta = provider.generate(
    prompt="Explain quantum computing",
    max_new_tokens=200,
    temperature=0.9,
)

# Streaming generation
for partial_text, meta in provider.generate_stream(
    prompt="Write a story about",
    max_new_tokens=500,
    chunk_size=10,
):
    print(partial_text, end="", flush=True)

# Calculate perplexity
ppl = provider.perplexity("The quick brown fox jumps over the lazy dog.")
print(f"Perplexity: {ppl:.2f}")
```

### Unified Reasoning:

```python
from src.vulcan.reasoning.unified_reasoning import UnifiedReasoner, ReasoningStrategy
from src.vulcan.reasoning.reasoning_types import ReasoningType

# Initialize reasoner
reasoner = UnifiedReasoner(
    enable_learning=True,
    enable_safety=True,
)

# Reason with specific type
result = reasoner.reason(
    input_data={"observation": "A causes B"},
    query={"question": "What happens if A occurs?"},
    reasoning_type=ReasoningType.CAUSAL,
)

# Reason with ensemble strategy
result = reasoner.reason(
    input_data="Is the hypothesis valid?",
    strategy=ReasoningStrategy.ENSEMBLE,
    confidence_threshold=0.7,
)

print(f"Conclusion: {result.conclusion}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Explanation: {result.explanation}")
```

### Hierarchical Memory:

```python
from src.vulcan.memory.hierarchical import HierarchicalMemory
from src.vulcan.memory.base import MemoryConfig, MemoryType, MemoryQuery

# Initialize memory
config = MemoryConfig(max_long_term=10000)
memory = HierarchicalMemory(config)

# Store memory
mem = memory.store(
    content="Important fact about machine learning",
    memory_type=MemoryType.SEMANTIC,
    importance=0.8,
)

# Retrieve similar memories
query = MemoryQuery(
    query_type="semantic_search",
    content="machine learning facts",
    limit=10,
)
result = memory.retrieve(query)

for mem, score in zip(result.memories, result.scores):
    print(f"[{score:.2f}] {mem.content}")

# Store tool selection
record = memory.store_tool_selection(
    problem_features=np.random.randn(128),
    problem_description="Classification task",
    selected_tools=["probabilistic", "symbolic"],
    execution_strategy="ensemble",
    performance_metrics={"latency_ms": 150, "accuracy": 0.95},
    success=True,
    utility_score=0.85,
)

# Get tool recommendations
recommendations = memory.get_recommended_tools(
    problem_features=np.random.randn(128),
    max_recommendations=5,
)
```

### GraphixExecutor Direct Usage:

```python
from src.llm_core.graphix_executor import (
    GraphixExecutor, ExecutorConfig, ExecutionMode, PrecisionMode
)

# Configure executor
config = ExecutorConfig(
    mode=ExecutionMode.INFERENCE,
    precision=PrecisionMode.FP16,
    use_flash_attention=True,
    use_kv_cache=True,
    enable_profiling=True,
)

# Create executor
executor = GraphixExecutor(
    hidden_size=256,
    num_layers=6,
    num_heads=4,
    vocab_size=4096,
    config=config,
)

# Execute IR graph
result = executor.execute(
    graph_ir={"embedding": {}, "layers": []},
    inputs={"tokens": [1, 2, 3, 4, 5]},
)

print(f"Execution time: {result['execution_time_ms']:.2f}ms")
print(f"Cache stats: {result['cache_stats']}")

# Get performance stats
stats = executor.get_stats()
print(f"Total executions: {stats['metrics']['total_executions']}")
```

---

## 21. Future Directions

## 17. Future Directions

### Planned Enhancements:

1. **Advanced Attention Patterns**
   - Multi-scale attention with hierarchical windows
   - Adaptive sparse patterns based on content
   - Memory-efficient long context (>100K tokens)
   - Sliding window with global attention tokens

2. **Enhanced MoE (Mixture of Experts)**
   - Dynamic expert allocation based on input
   - Hierarchical expert organization
   - Expert specialization and routing optimization
   - Load balancing across experts

3. **Improved Generation**
   - Speculative decoding for faster inference
   - Beam search with diversity constraints
   - Constrained generation with grammar rules
   - Multi-modal generation (text + code + structured)

4. **Distributed Execution**
   - Model parallelism across GPUs
   - Pipeline parallelism for large models
   - Distributed inference with load balancing
   - Federated learning support

5. **Safety Improvements**
   - Formal verification integration
   - Constitutional AI methods
   - Interpretability and attribution tools
   - Red team testing automation

6. **Learning Enhancements**
   - Online learning from user feedback
   - Curriculum learning for complex tasks
   - Meta-learning for few-shot adaptation
   - Reinforcement learning from human feedback (RLHF)

7. **Memory System Evolution**
   - Hierarchical attention over memory
   - Episodic memory retrieval with temporal reasoning
   - Semantic memory with knowledge graph integration
   - Procedural memory for skill acquisition

8. **Quantization and Efficiency**
   - INT4 and INT2 quantization
   - Structured pruning
   - Knowledge distillation
   - Neural architecture search (NAS)

---

## Related Documentation

- [ARCHITECTURE_OVERVIEW.md](ARCHITECTURE_OVERVIEW.md) - Platform architecture
- [EXECUTION_ENGINE.md](EXECUTION_ENGINE.md) - Execution mechanics
- [COMPLETE_SERVICE_CATALOG.md](COMPLETE_SERVICE_CATALOG.md) - Service documentation
- [GOVERNANCE.md](GOVERNANCE.md) - Governance documentation
- [SECURITY.md](SECURITY.md) - Security architecture
- [AI_TRAINING_GUIDE.md](AI_TRAINING_GUIDE.md) - Training guide
- [AUTONOMOUS_LEARNING_ANALYSIS.md](AUTONOMOUS_LEARNING_ANALYSIS.md) - Learning analysis

---

## Glossary

| Term | Definition |
|------|------------|
| **IR** | Intermediate Representation - graph-based representation of computations |
| **RoPE** | Rotary Position Embeddings - relative positional encoding method |
| **SwiGLU** | Swish-Gated Linear Unit - activation function combining Swish and gating |
| **MoE** | Mixture of Experts - sparse model with multiple expert networks |
| **GQA** | Grouped Query Attention - attention with fewer KV heads than query heads |
| **KV Cache** | Key-Value Cache - stores previous attention keys/values for efficient generation |
| **LoRA** | Low-Rank Adaptation - parameter-efficient fine-tuning method |
| **Flash Attention** | Memory-efficient attention implementation with tiling |
| **GraphRAG** | Graph Retrieval Augmented Generation - semantic retrieval with graph structure |
| **ZK Proofs** | Zero-Knowledge Proofs - cryptographic verification without revealing data |
| **VOI** | Value of Information - decision-theoretic measure for information gathering |

---

**Document Version:** 3.0.0  
**Last Updated:** December 30, 2024  
**Author:** Vulcan Ultra Deep Dive Analysis
