# Knowledge Crystallizer

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/vulcan-ami/knowledge-crystallizer)
[![Python](https://img.shields.io/badge/python-3.8+-green.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-Proprietary-red.svg)](LICENSE)

## Overview

The **Knowledge Crystallizer** is a core module of the VULCAN-AMI system designed to distill raw execution traces into reusable, validated knowledge principles. It enables AI systems to:

- **Learn** generalizable "crystallized" knowledge from specific execution instances
- **Validate** principles across domains with multi-level testing
- **Track** contraindications (conditions where principles fail) with cascade analysis
- **Store** versioned knowledge with efficient retrieval and pruning
- **Apply** learned principles to solve new problems

The module follows the **EXAMINE → SELECT → APPLY → REMEMBER** pattern for methodical processing, ensuring robust extraction, validation, and application of knowledge.

## Table of Contents

- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Advanced Usage](#advanced-usage)
- [Performance Considerations](#performance-considerations)
- [Security](#security)
- [Contributing](#contributing)

## Key Features

### Principle Extraction
- **Pattern Detection**: Identifies sequential, conditional, iterative, hierarchical, and composite patterns
- **Success Factor Analysis**: Extracts factors contributing to successful outcomes
- **Abstraction Engine**: Converts patterns into reusable, generalizable principles
- **Multiple Strategies**: Conservative, balanced, aggressive, and exploratory extraction modes

### Multi-Level Validation
- **Basic Validation**: Structure and type checking
- **Consistency Validation**: Internal coherence and domain conflict detection
- **Domain-Specific Testing**: Sandboxed execution with generated test cases
- **Generalization Testing**: Cross-domain applicability verification
- **Domain Criticality Awareness**: Safety-critical domains receive stricter validation

### Contraindication Tracking
- **Failure Mode Detection**: Performance, correctness, stability, resource, and cascading failures
- **Cascade Analysis**: Multi-hop impact simulation with graph-based propagation
- **Severity Classification**: LOW, MEDIUM, HIGH, CRITICAL with automatic escalation
- **Mitigation Suggestions**: Automated workaround recommendations

### Intelligent Method Selection
- **Adaptive Selection**: Chooses crystallization method based on trace characteristics
- **Learning-Based Optimization**: Improves selection based on historical outcomes
- **Fallback Support**: Automatic fallback to simpler methods on failure

### Versioned Knowledge Storage
- **Multiple Backends**: Memory, SQLite, File, and Hybrid storage options
- **Version Control**: Git-like versioning with diff-based storage optimization
- **Vector Search**: FAISS-powered similarity search with fallback implementation
- **Automatic Pruning**: Removes outdated, low-confidence, or contradictory principles

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│ KnowledgeCrystallizer │
│ (Main Orchestrator) │
├─────────────────────────────────────────────────────────────────────┤
│ │
│ ┌──────────────┐ ┌──────────────┐ ┌──────────────────────────┐ │
│ │Crystallization│ │ Principle │ │ Knowledge │ │
│ │ Selector │──│ Extractor │──│ Validator │ │
│ └──────────────┘ └──────────────┘ └──────────────────────────┘ │
│ │ │ │ │
│ ▼ ▼ ▼ │
│ ┌──────────────┐ ┌──────────────┐ ┌──────────────────────────┐ │
│ │ Selection │ │ Pattern │ │ Domain │ │
│ │ Strategies │ │ Detector │ │ Validator │ │
│ └──────────────┘ └──────────────┘ └──────────────────────────┘ │
│ │
├─────────────────────────────────────────────────────────────────────┤
│ │
│ ┌──────────────────────────┐ ┌─────────────────────────────────┐ │
│ │ Contraindication │ │ Versioned Knowledge Base │ │
│ │ Tracker │ │ (SQLite/FAISS/Memory) │ │
│ │ ├─ Database │ │ ├─ Version Control │ │
│ │ ├─ Graph │ │ ├─ Knowledge Index │ │
│ │ └─ Cascade Analyzer │ │ └─ Knowledge Pruner │ │
│ └──────────────────────────┘ └─────────────────────────────────┘ │
│ │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Summary

| Component | File | Responsibility | Key Classes |
|-----------|------|----------------|-------------|
| **Core Orchestrator** | `knowledge_crystallizer_core.py` | Pipeline coordination, mode management | `KnowledgeCrystallizer`, `KnowledgeApplicator` |
| **Principle Extractor** | `principle_extractor.py` | Pattern detection, principle abstraction | `PrincipleExtractor`, `PatternDetector`, `AbstractionEngine` |
| **Validation Engine** | `validation_engine.py` | Multi-level validation, sandboxed execution | `KnowledgeValidator`, `DomainValidator` |
| **Knowledge Storage** | `knowledge_storage.py` | Versioned persistence, vector search | `VersionedKnowledgeBase`, `KnowledgeIndex` |
| **Method Selector** | `crystallization_selector.py` | Adaptive method selection | `CrystallizationSelector`, `SelectionStrategy` |
| **Contraindication Tracker** | `contraindication_tracker.py` | Failure tracking, cascade analysis | `ContraindicationDatabase`, `CascadeAnalyzer` |

## Installation

### Requirements

- Python 3.8+
- NumPy (required)

### Install Dependencies

```bash
# Core dependencies
pip install numpy

# Optional dependencies (recommended)
pip install scipy networkx faiss-cpu

# All dependencies
pip install numpy scipy networkx faiss-cpu
```

### Optional Dependencies

| Package | Purpose | Fallback |
|---------|---------|----------|
| `scipy` | Statistical significance testing | Heuristic-based estimation |
| `networkx` | Graph-based cascade analysis | `SimpleGraph` implementation |
| `faiss-cpu` | Fast vector similarity search | `SimpleVectorIndex` with L2 distance |

### Integration

```python
from vulcan.knowledge_crystallizer import (
 KnowledgeCrystallizer,
 ExecutionTrace,
 CrystallizationMode,
 KNOWLEDGE_CRYSTALLIZER_AVAILABLE
)

# Check availability
if KNOWLEDGE_CRYSTALLIZER_AVAILABLE:
 crystallizer = KnowledgeCrystallizer()
```

## Quick Start

### Basic Crystallization

```python
import logging
from vulcan.knowledge_crystallizer import (
 KnowledgeCrystallizer,
 ExecutionTrace,
)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize crystallizer
crystallizer = KnowledgeCrystallizer()

# Create an execution trace
trace = ExecutionTrace(
 trace_id="example_001",
 actions=[
 {"type": "fetch_data", "source": "api"},
 {"type": "transform", "operation": "normalize"},
 {"type": "validate", "schema": "v1"},
 {"type": "store", "destination": "db"}
 ],
 outcomes={"success": True, "records_processed": 1000},
 context={"domain": "data_pipeline", "environment": "production"},
 success=True,
 domain="data_pipeline"
)

# Crystallize the trace
result = crystallizer.crystallize(trace)

print(f"Extracted {len(result.principles)} principles")
print(f"Confidence: {result.confidence:.2f}")
print(f"Method used: {result.method_used}")
```

### Applying Knowledge

```python
# Define a problem
problem = {
 "domain": "data_pipeline",
 "type": "process",
 "context": {"scale": "large"}
}

# Apply crystallized knowledge
application = crystallizer.apply_knowledge(problem, confidence_required=0.7)

if application.principle_used:
 print(f"Applied principle: {application.principle_used.id}")
 print(f"Solution: {application.solution}")
 print(f"Confidence: {application.confidence:.2f}")
else:
 print("No applicable principles found")
```

### Providing Feedback

```python
# Update principle based on application outcome
crystallizer.update_from_feedback(
 principle_id=application.principle_used.id,
 outcome={
 "success": True,
 "performance": {"latency_ms": 150}
 }
)
```

## Core Concepts

### Execution Traces

An `ExecutionTrace` represents a recorded execution with:

```python
@dataclass
class ExecutionTrace:
 trace_id: str # Unique identifier
 actions: List[Dict[str, Any]] # Sequence of actions performed
 outcomes: Dict[str, Any] # Results and metrics
 context: Dict[str, Any] # Environmental context
 success: bool = True # Overall success status
 domain: str = "general" # Domain classification
 metadata: Dict[str, Any] = {} # Additional metadata
 iteration: Optional[int] = None # For incremental learning
```

### Crystallization Modes

| Mode | Description | Best For |
|------|-------------|----------|
| `STANDARD` | General-purpose extraction | Typical successful traces |
| `CASCADE_AWARE` | Analyzes failure chains | Traces with failures or dependencies |
| `INCREMENTAL` | Builds on previous iterations | Iterative refinement scenarios |
| `BATCH` | Processes multiple traces | High-volume trace processing |
| `ADAPTIVE` | Adjusts to uncertainty | Novel domains, exploration |
| `HYBRID` | Combines multiple methods | Complex, multi-faceted scenarios |

### Validation Levels

```python
class ValidationLevel(Enum):
 BASIC = "basic" # Structure and type checks
 CONSISTENCY = "consistency" # Internal coherence
 DOMAIN_SPECIFIC = "domain" # Domain compatibility
 GENERALIZATION = "general" # Cross-domain applicability
 COMPREHENSIVE = "full" # All validation levels
```

### Domain Criticality

The system automatically adjusts validation rigor based on domain criticality:

| Criticality | Domains | Validation Requirements |
|-------------|---------|------------------------|
| 0.95 (Critical) | `safety_critical`, `medical` | All levels + comprehensive testing |
| 0.90 (High) | `financial`, `legal` | Full domain + generalization |
| 0.85 | `control`, `autonomous_systems` | Domain-specific required |
| 0.70-0.80 | `security`, `planning` | Conditional domain testing |
| 0.50-0.65 | `prediction`, `reasoning` | Basic + consistency |
| 0.30 (Low) | `general`, `perception` | Basic validation sufficient |

## API Reference

### KnowledgeCrystallizer

```python
class KnowledgeCrystallizer:
 def __init__(
 self,
 vulcan_memory=None, # Optional VULCAN memory system
 semantic_bridge=None # Optional semantic bridge
 )
 
 def crystallize(
 self,
 execution_trace: ExecutionTrace,
 context: Optional[Dict[str, Any]] = None
 ) -> CrystallizationResult:
 """Main crystallization entry point."""
 
 def apply_knowledge(
 self,
 problem: Dict[str, Any],
 confidence_required: float = 0.7
 ) -> ApplicationResult:
 """Apply crystallized knowledge to solve a problem."""
 
 def update_from_feedback(
 self,
 principle_id: str,
 outcome: Dict[str, Any]
 ) -> None:
 """Update knowledge from application feedback."""
 
 def validate_stratified(
 self,
 candidate: Principle
 ) -> ValidationResult:
 """Perform multi-level validation."""
```

### PrincipleExtractor

```python
class PrincipleExtractor:
 def __init__(
 self,
 min_evidence_count: int = 3,
 min_confidence: float = 0.6,
 strategy: ExtractionStrategy = ExtractionStrategy.BALANCED
 )
 
 def extract_from_trace(
 self,
 execution_trace: ExecutionTrace
 ) -> List[CrystallizedPrinciple]:
 """Extract principles from a single trace."""
 
 def extract_from_batch(
 self,
 traces: List[ExecutionTrace]
 ) -> List[CrystallizedPrinciple]:
 """Extract principles from multiple traces."""
```

### VersionedKnowledgeBase

```python
class VersionedKnowledgeBase:
 def __init__(
 self,
 backend: StorageBackend = StorageBackend.MEMORY,
 storage_path: Optional[Path] = None,
 compression: CompressionType = CompressionType.NONE,
 auto_save: bool = True,
 max_versions: int = 100
 )
 
 def store(
 self,
 principle,
 author: Optional[str] = None,
 message: Optional[str] = None
 ) -> str:
 """Store a principle with version control."""
 
 def get(
 self,
 principle_id: str,
 version: Optional[int] = None
 ) -> Optional[Principle]:
 """Retrieve a principle by ID."""
 
 def rollback(
 self,
 principle_id: str,
 target_version: int
 ) -> bool:
 """Rollback to a previous version."""
 
 def find_similar(
 self,
 principle,
 threshold: float = 0.7
 ) -> List[Principle]:
 """Find similar principles using vector search."""
```

## Configuration

### Storage Configuration

```python
from vulcan.knowledge_crystallizer import (
 VersionedKnowledgeBase,
 StorageBackend,
 CompressionType
)

# SQLite backend with compression
kb = VersionedKnowledgeBase(
 backend=StorageBackend.SQLITE,
 storage_path=Path("./knowledge.db"),
 compression=CompressionType.GZIP,
 max_versions=50
)

# Hybrid backend (memory + persistence)
kb = VersionedKnowledgeBase(
 backend=StorageBackend.HYBRID,
 storage_path=Path("./knowledge"),
 auto_save=True
)
```

### Extraction Strategy

```python
from vulcan.knowledge_crystallizer import (
 PrincipleExtractor,
 ExtractionStrategy
)

# Conservative: High confidence required (0.8+)
extractor = PrincipleExtractor(
 strategy=ExtractionStrategy.CONSERVATIVE,
 min_evidence_count=5
)

# Exploratory: Lower thresholds for novel domains
extractor = PrincipleExtractor(
 strategy=ExtractionStrategy.EXPLORATORY,
 min_confidence=0.3
)
```

### Validation Configuration

```python
from vulcan.knowledge_crystallizer import KnowledgeValidator

validator = KnowledgeValidator(
 min_confidence=0.6,
 consistency_threshold=0.7
)

# Multi-level validation with context
results = validator.validate_principle_multilevel(
 principle,
 context={
 "time_budget_ms": 5000,
 "quality_requirement": "high",
 "force_comprehensive": True
 }
)
```

## Advanced Usage

### Custom Crystallization Context

```python
# Provide additional context for method selection
result = crystallizer.crystallize(
 trace,
 context={
 "batch_size": 10,
 "cascade_failures_detected": True,
 "previous_iterations": 5,
 "refinement_requested": True,
 "known_domains": ["optimization", "analysis"]
 }
)
```

### Cascade Analysis

```python
from vulcan.knowledge_crystallizer import (
 CascadeAnalyzer,
 ContraindicationDatabase,
 ContraindicationGraph
)

# Initialize cascade analyzer
db = ContraindicationDatabase()
graph = ContraindicationGraph()
analyzer = CascadeAnalyzer(db, graph)

# Analyze cascade impact
impact = analyzer.analyze_cascade_impact(principle, max_depth=3)

print(f"Risk Level: {impact.get_risk_level()}")
print(f"Affected Principles: {impact.blast_radius}")
print(f"Recovery Time Estimate: {impact.recovery_time_estimate}s")
print(f"Mitigation Strategies: {impact.mitigation_strategies}")
```

### Batch Processing

```python
# Process multiple traces efficiently
traces = [trace1, trace2, trace3, ...]

result = crystallizer.crystallize(
 traces[0],
 context={
 "batch_traces_available": len(traces),
 "batch_size": 20
 }
)

# Or use batch extraction directly
principles = crystallizer.extractor.extract_from_batch(traces)
```

### Version Control Operations

```python
# Store with commit message
kb.store(principle, author="system", message="Initial extraction")

# Get version history
history = kb.get_history(principle_id)
for version in history:
 print(f"v{version.version}: {version.commit_message} by {version.author}")

# Rollback to previous version
kb.rollback(principle_id, target_version=2)

# Export/Import
kb.export(Path("./backup.json"), format="json")
kb.import_from(Path("./backup.json"))
```

## Performance Considerations

### Time Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Pattern Detection (sequential) | O(n²·m) | n=actions, m=max pattern length |
| Similarity Search (FAISS) | O(log n) | With IVF index |
| Similarity Search (fallback) | O(n·d) | n=principles, d=dimensions |
| Version Diff Computation | O(n·m) | Unified diff algorithm |
| Cascade Risk Calculation | O(p·d) | p=paths, d=max depth |

### Memory Management

- History deques capped at 1000 entries by default
- Simulation caches enforce 200 entry limits
- Diff-based version storage reduces memory by ~60%
- SQLite uses WAL mode with 5-connection pool

### Optimization Tips

1. **Use FAISS** for large knowledge bases (>1000 principles)
2. **Enable HYBRID storage** for production workloads
3. **Set appropriate `max_versions`** based on your retention needs
4. **Use batch extraction** for high-volume trace processing
5. **Configure domain criticality** to skip unnecessary validation

## Security

### Sandboxed Execution

The validation engine executes test code in a restricted environment:

- **Filtered Operations**: `import`, `eval`, `exec`, `open`, `__builtins__`, etc.
- **Resource Limits**: CPU time and memory limits (Linux: via `resource` module)
- **Temporary Isolation**: Tests run in auto-cleaned temporary directories

### Safe Serialization

- Uses `safe_pickle_load` wrapper for untrusted data
- Hash functions use `usedforsecurity=False` for non-cryptographic operations
- Internal pickle operations marked with security annotations

### Recommendations

- Use AST-based code analysis for additional security
- Run validation in containerized environments for production
- Audit principle execution logic before deployment

## Thread Safety

All major components use `threading.RLock` for concurrent access:

```python
# Safe for multi-threaded usage
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
 futures = [
 executor.submit(crystallizer.crystallize, trace)
 for trace in traces
 ]
 results = [f.result() for f in futures]
```

## Availability Flags

Check component availability before use:

```python
from vulcan.knowledge_crystallizer import (
 KNOWLEDGE_CRYSTALLIZER_AVAILABLE,
 PRINCIPLE_EXTRACTOR_AVAILABLE,
 VALIDATION_ENGINE_AVAILABLE,
 KNOWLEDGE_STORAGE_AVAILABLE,
 CRYSTALLIZATION_SELECTOR_AVAILABLE,
 CONTRAINDICATION_TRACKER_AVAILABLE
)

if not KNOWLEDGE_CRYSTALLIZER_AVAILABLE:
 print("Knowledge Crystallizer not available - check numpy installation")
```

## Error Handling

The module uses graceful degradation:

```python
try:
 result = crystallizer.crystallize(trace)
except Exception as e:
 # Automatic fallback to simpler methods
 # Check result.metadata for error details
 if result.confidence == 0.0:
 print(f"Crystallization failed: {result.metadata.get('error')}")
```

## Logging

Configure logging for debugging:

```python
import logging

# Detailed logging
logging.getLogger("vulcan.knowledge_crystallizer").setLevel(logging.DEBUG)

# Component-specific logging
logging.getLogger("vulcan.knowledge_crystallizer.principle_extractor").setLevel(logging.INFO)
logging.getLogger("vulcan.knowledge_crystallizer.validation_engine").setLevel(logging.WARNING)
```

### Code Style

- Follow PEP 8 guidelines
- Add type hints for all public methods
- Include docstrings with Args, Returns, and Raises sections
- Use `logging` instead of `print` statements

## License

Proprietary - See [LICENSE](LICENSE) for complete terms.

## Changelog

### v1.0.0 (2026-01)
- Initial release
- Core crystallization pipeline
- Multi-level validation framework
- Versioned knowledge storage with FAISS support
- Cascade-aware contraindication tracking
- Learning-based method selection

---

For issues, feature requests, or contributions, please refer to the [VULCAN-AMI project repository](https://github.com/vulcan-ami).
