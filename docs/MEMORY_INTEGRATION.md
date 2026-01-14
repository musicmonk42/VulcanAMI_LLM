# Memory System Integration Guide

**Version:** 1.0.0 
**Last Updated:** 2026-01-12 
---

## Overview

VulcanAMI_LLM features a unified memory integration layer that connects three distinct memory subsystems into a cohesive, production-grade storage and governance platform. This document describes the architecture, usage, and deployment considerations for the integrated memory system.

## Architecture

### Three Memory Subsystems

The system integrates three specialized memory components:

1. **`src/persistant_memory_v46/`** - Production Storage Backend
 - **GraphRAG**: Graph-based Retrieval Augmented Generation with semantic search
 - **MerkleLSM**: Log-structured merge tree with Merkle proofs for versioning
 - **PackfileStore**: S3-backed persistent storage with CloudFront CDN
 - **ZKProver**: Zero-knowledge proof generation for compliance
 - **UnlearningEngine**: GDPR-compliant machine unlearning

2. **`src/memory/`** - Governance & Optimization
 - **GovernedUnlearning**: Consensus-based unlearning with proposals
 - **CostOptimizer**: Budget-aware storage optimization

3. **`src/vulcan/memory/`** - Core Hierarchical Memory
 - **HierarchicalMemory**: Multi-level memory (short-term, working, long-term)
 - **Retrieval & Consolidation**: Memory search and consolidation
 - **Tool Selection**: Learning-based tool selection history

### Integration Bridges

Two bridge components provide unified access:

#### MemoryBridge (`src/integration/memory_bridge.py`)

**Purpose:** Single entry point for all memory operations

**Features:**
- Unified API across all memory subsystems
- Graceful degradation when components unavailable
- Thread-safe operations
- Context manager support logging and metrics

**Example Usage:**
```python
from src.integration import MemoryBridge, MemoryBridgeConfig

# Create bridge with configuration
config = MemoryBridgeConfig(
 s3_bucket="my-memory-bucket",
 enable_zk_proofs=True,
 enable_governed_unlearning=True,
 embedding_model="all-MiniLM-L6-v2" # Valid model
)

bridge = MemoryBridge(config)

# Store data across backends
bridge.store("doc1", "Important content", {"priority": "high"})

# Hybrid retrieval (semantic + contextual)
results = bridge.retrieve("search query", k=10)

# Request governed unlearning
result = bridge.unlearn("sensitive_pattern", urgency="high")

# Optimize storage costs
report = bridge.optimize_storage()

# Check status
status = bridge.get_status()
print(f"Components available: {status}")

# Clean shutdown
bridge.shutdown()
```

#### GVulcanBridge (`src/integration/gvulcan_bridge.py`)

**Purpose:** Access gvulcan's unique data quality and policy utilities

**Features:**
- **DQS (Data Quality Score)**: Multi-dimensional quality validation
- **OPA (Open Policy Agent)**: Policy-as-code enforcement
- Fail-safe defaults (permits operations when unavailable)
- Full type safety and documentation

**Example Usage:**
```python
from src.integration import GVulcanBridge

bridge = GVulcanBridge({
 "dqs_reject_threshold": 0.3,
 "dqs_quarantine_threshold": 0.4,
 "opa_cache_enabled": True
})

# Validate data quality
quality = bridge.validate_data_quality(
 pii_confidence=0.05,
 graph_completeness=0.95,
 syntactic_completeness=0.98
)

if quality["gate_decision"] == "accept":
 # Check policy compliance
 allowed = bridge.check_write_barrier(
 dqs_score=quality["score"],
 context={"source": "user_input"}
 )
 
 if allowed:
 # Proceed with write operation
 perform_write()
```

---

## Configuration

### MemoryBridge Configuration

```python
from src.integration import MemoryBridgeConfig

config = MemoryBridgeConfig(
 # Storage configuration
 s3_bucket="my-memory-bucket", # S3 bucket for PackfileStore
 region="us-east-1", # AWS region
 compression="zstd", # zstd, zlib, lz4, none
 encryption="AES256", # AES256 or aws:kms
 
 # Memory configuration
 max_memories=100000, # Maximum memories in hierarchical storage
 default_importance=0.5, # Default importance [0.0-1.0]
 decay_rate=0.001, # Memory decay rate
 embedding_model="all-MiniLM-L6-v2", # Valid sentence-transformers model
 
 # Operations configuration
 enable_governed_unlearning=True, # Enable governed unlearning
 enable_cost_optimization=True, # Enable cost optimization
 auto_optimize=False, # Auto-optimize in background
 
 # Feature flags
 enable_zk_proofs=True, # Enable zero-knowledge proofs
 enable_graph_rag=True, # Enable graph-based RAG
)
```

### Environment Variables

Configure via environment variables in Kubernetes/Docker:

```bash
# Memory System Configuration
MEMORY_BRIDGE_ENABLED=true
S3_BUCKET=vulcanami-memory
AWS_REGION=us-east-1
COMPRESSION=zstd
ENCRYPTION=AES256

# Hierarchical Memory
MAX_MEMORIES=100000
DEFAULT_IMPORTANCE=0.5
DECAY_RATE=0.001
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Operations
ENABLE_GOVERNED_UNLEARNING=true
ENABLE_COST_OPTIMIZATION=true
AUTO_OPTIMIZE=false

# Feature Flags
ENABLE_ZK_PROOFS=true
ENABLE_GRAPH_RAG=true

# GVulcan Configuration
DQS_REJECT_THRESHOLD=0.3
DQS_QUARANTINE_THRESHOLD=0.4
OPA_CACHE_ENABLED=true
OPA_BUNDLE_VERSION=1.0.0
```

---

## Kubernetes Deployment

### Helm Values

Update `helm/vulcanami/values.yaml`:

```yaml
memorySystem:
 enabled: true
 
 # Integration bridges
 bridges:
 memory:
 enabled: true
 s3Bucket: vulcanami-memory
 region: us-east-1
 compression: zstd
 encryption: AES256
 embeddingModel: all-MiniLM-L6-v2
 
 gvulcan:
 enabled: true
 dqsRejectThreshold: 0.3
 dqsQuarantineThreshold: 0.4
 opaCacheEnabled: true
 
 # Existing configuration...
 milvus:
 host: milvus-service
 port: 19530
 
 s3:
 bucket: vulcanami-memory
 region: us-east-1
 
 packfile:
 sizeMb: 32
 compression: zstd
 encryption: AES256
 
 lsm:
 compactionStrategy: adaptive
 bloomFilter: true
 
 indexType: disk_based_tier_c
 
 unlearning:
 method: gradient_surgery
 enableVerification: true
 
 zk:
 proofSystem: groth16
```

### ConfigMap

The Helm chart includes `configmap-memory.yaml` which configures the memory system. Ensure it includes bridge configuration:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
 name: {{ include "vulcanami.fullname" . }}-memory-config
data:
 # Memory Bridge Configuration
 MEMORY_BRIDGE_ENABLED: "true"
 EMBEDDING_MODEL: "all-MiniLM-L6-v2"
 ENABLE_GOVERNED_UNLEARNING: "true"
 ENABLE_COST_OPTIMIZATION: "true"
 ENABLE_ZK_PROOFS: "true"
 ENABLE_GRAPH_RAG: "true"
 
 # GVulcan Bridge Configuration
 GVULCAN_BRIDGE_ENABLED: "true"
 DQS_REJECT_THRESHOLD: "0.3"
 DQS_QUARANTINE_THRESHOLD: "0.4"
 OPA_CACHE_ENABLED: "true"
 
 # Existing configuration...
 MILVUS_HOST: {{ .Values.memorySystem.milvus.host | quote }}
 S3_BUCKET: {{ .Values.memorySystem.s3.bucket | quote }}
 # ... etc
```

---

## Docker Compose

Update `docker-compose.prod.yml` to include memory bridge environment variables:

```yaml
services:
 full-platform:
 environment:
 # Memory Bridge Configuration
 - MEMORY_BRIDGE_ENABLED=true
 - S3_BUCKET=vulcanami-memory
 - AWS_REGION=us-east-1
 - COMPRESSION=zstd
 - ENCRYPTION=AES256
 - EMBEDDING_MODEL=all-MiniLM-L6-v2
 - ENABLE_GOVERNED_UNLEARNING=true
 - ENABLE_COST_OPTIMIZATION=true
 - ENABLE_ZK_PROOFS=true
 - ENABLE_GRAPH_RAG=true
 
 # GVulcan Bridge Configuration
 - GVULCAN_BRIDGE_ENABLED=true
 - DQS_REJECT_THRESHOLD=0.3
 - DQS_QUARANTINE_THRESHOLD=0.4
 - OPA_CACHE_ENABLED=true
 
 # S3/MinIO Configuration
 - AWS_ACCESS_KEY_ID=${MINIO_ROOT_USER}
 - AWS_SECRET_ACCESS_KEY=${MINIO_ROOT_PASSWORD}
 - S3_ENDPOINT_URL=http://minio:9000
```

---

## Makefile Targets

New Makefile targets for memory integration:

```makefile
.PHONY: test-memory-integration
test-memory-integration: ## Test memory bridge integration
	@echo "$(GREEN)Testing memory bridge integration...$(NC)"
	pytest src/integration/tests/test_memory_bridge.py -v --tb=short

.PHONY: test-gvulcan-bridge
test-gvulcan-bridge: ## Test gvulcan bridge
	@echo "$(GREEN)Testing gvulcan bridge...$(NC)"
	pytest src/integration/tests/test_gvulcan_bridge.py -v --tb=short

.PHONY: memory-bridge-status
memory-bridge-status: ## Check memory bridge status
	@echo "$(GREEN)Checking memory bridge status...$(NC)"
	python -c "from src.integration import create_memory_bridge; \
	 bridge = create_memory_bridge(); \
	 status = bridge.get_status(); \
	 import json; \
	 print(json.dumps(status, indent=2))"

.PHONY: gvulcan-bridge-status
gvulcan-bridge-status: ## Check gvulcan bridge status
	@echo "$(GREEN)Checking gvulcan bridge status...$(NC)"
	python -c "from src.integration import create_gvulcan_bridge; \
	 bridge = create_gvulcan_bridge(); \
	 if bridge: \
	 status = bridge.get_status(); \
	 import json; \
	 print(json.dumps(status, indent=2))"
```

---

## Migration Guide

### Migrating from gvulcan Storage Components

**DEPRECATED:** `gvulcan.BloomFilter` and `gvulcan.MerkleLSMDAG` 
**USE INSTEAD:** `persistant_memory_v46.lsm.BloomFilter` and `persistant_memory_v46.lsm.MerkleLSM`

#### Before (Deprecated):
```python
from src.gvulcan import BloomFilter, MerkleLSMDAG

bloom = BloomFilter(capacity=10000)
lsm = MerkleLSMDAG()
```

#### After (Recommended):
```python
from src.persistant_memory_v46.lsm import BloomFilter, MerkleLSM

bloom = BloomFilter(capacity=10000)
lsm = MerkleLSM(packfile_size_mb=32)
```

**Note:** Deprecation warnings will guide migration. Storage components in gvulcan will be removed in a future version.

### Using MemoryBridge Instead of Direct Access

#### Before (Direct Access):
```python
from src.persistant_memory_v46 import GraphRAG, MerkleLSM
from src.memory import GovernedUnlearning

graph_rag = GraphRAG(embedding_model="all-MiniLM-L6-v2")
lsm = MerkleLSM()
unlearning = GovernedUnlearning(memory_ref)
```

#### After (Unified Bridge):
```python
from src.integration import create_memory_bridge

bridge = create_memory_bridge({
 "enable_graph_rag": True,
 "enable_governed_unlearning": True
})

# Unified API across all backends
bridge.store("key", "value", {"tags": ["important"]})
results = bridge.retrieve("query", k=10)
bridge.unlearn("pattern", urgency="high")
```

---

##Security Considerations

### Data Protection

1. **Encryption at Rest:** All data stored via PackfileStore uses AES256 encryption
2. **Encryption in Transit:** TLS/SSL for all S3 communications
3. **Access Control:** IAM roles for S3 access (never use long-lived credentials)
4. **Zero-Knowledge Proofs:** Cryptographic verification without revealing data

### GDPR Compliance

The governed unlearning system provides legally compliant data removal:

```python
# Submit unlearning request
result = bridge.unlearn(
 pattern="user_email@example.com",
 method="gradient_surgery",
 urgency="high",
 requester_id="gdpr_officer"
)

# Generate cryptographic proof
if result["status"] == "completed":
 proof = bridge.generate_unlearning_proof(
 items=[result["removed_items"]]
 )
 # Store proof for compliance audit
```

### Security Best Practices

1. **Secrets Management:** Use Kubernetes secrets or AWS Secrets Manager
2. **Least Privilege:** Grant minimum required permissions
3. **Audit Logging:** All unlearning operations are audited
4. **Rate Limiting:** Protect against DoS attacks
5. **Input Validation:** All inputs validated before processing

---

## Monitoring & Observability

### Metrics

MemoryBridge exposes comprehensive metrics:

```python
bridge = create_memory_bridge()

# Get detailed statistics
stats = bridge.get_statistics()

# Available metrics:
# - graph_rag: Retrieval statistics, cache hits, index size
# - lsm: Write amplification, compaction stats, bloom filter efficiency
# - unlearning: Proposals submitted, approved, rejected
# - cost: Storage costs, optimization savings, budget utilization
```

### Prometheus Integration

Add memory bridge metrics to your Prometheus scrape config:

```yaml
scrape_configs:
 - job_name: 'vulcanami-memory'
 static_configs:
 - targets: ['vulcanami-api:9148']
 metrics_path: '/metrics'
```

### Grafana Dashboards

Key metrics to monitor:

- **Storage Usage:** Total GB, growth rate, cost per GB
- **Retrieval Performance:** P50/P95/P99 latency, cache hit rate
- **Unlearning Queue:** Pending proposals, approval rate, average time
- **Cost Optimization:** Monthly spend, savings from optimization
- **Component Health:** Availability of GraphRAG, LSM, ZKProver

---

## Troubleshooting

### Bridge Initialization Failures

**Symptom:** MemoryBridge components not initializing

**Solution:**
1. Check logs for specific component failures
2. Verify dependencies installed (`numpy`, `sentence-transformers`, etc.)
3. Check S3 credentials and bucket access
4. Verify Milvus connectivity

```python
# Diagnostic check
bridge = create_memory_bridge()
status = bridge.get_status()

for component, initialized in status.items():
 if not initialized and component.endswith("_initialized"):
 print(f"⚠️ {component} failed to initialize")
```

### Graceful Degradation

MemoryBridge gracefully degrades when components unavailable:

```python
# Bridge still works with partial functionality
bridge = create_memory_bridge()

# Check what's available
status = bridge.get_status()
if status["graph_rag_initialized"]:
 # Use semantic search
 results = bridge.retrieve("query")
else:
 # Fall back to direct LSM lookup
 # (handled automatically by bridge)
```

### Performance Issues

**Slow Retrieval:**
1. Check GraphRAG cache size and hit rate
2. Verify embedding model is loaded (cached after first use)
3. Consider using smaller k value for retrieval

**High Storage Costs:**
1. Run optimization: `bridge.optimize_storage()`
2. Check budget status: `bridge.check_budget()`
3. Review retention policies

---

## Best Practices

### Configuration

1. **Use Specific Model Names:** Always use valid sentence-transformers models
 ```python
 # ✓ Good
 embedding_model="all-MiniLM-L6-v2"
 
 # ✗ Bad
 embedding_model="llm_embeddings" # Invalid model name
 ```

2. **Set Appropriate Limits:**
 ```python
 config = MemoryBridgeConfig(
 max_memories=100000, # Based on available memory
 default_importance=0.5, # Reasonable default
 decay_rate=0.001, # Slow decay for stable importance
 )
 ```

3. **Enable Features Selectively:**
 ```python
 config = MemoryBridgeConfig(
 enable_zk_proofs=True, # If compliance required
 enable_governed_unlearning=True, # If GDPR compliance needed
 enable_cost_optimization=True, # Always recommended
 )
 ```

### Operations

1. **Use Context Managers:**
 ```python
 with create_memory_bridge(config) as bridge:
 bridge.store("key", "value")
 results = bridge.retrieve("query")
 # Automatic cleanup on exit
 ```

2. **Handle Errors Gracefully:**
 ```python
 result = bridge.store("key", "value")
 if not result:
 logger.error("Storage failed, using fallback")
 # Implement fallback strategy
 ```

3. **Monitor Resource Usage:**
 ```python
 stats = bridge.get_statistics()
 if stats.get("cost", {}).get("usage_percentage", 0) > 80:
 bridge.optimize_storage()
 ```

### Testing

1. **Mock Unavailable Components:**
 ```python
 # Test with minimal configuration
 config = MemoryBridgeConfig(
 enable_zk_proofs=False, # Skip if not needed for test
 enable_graph_rag=False,
 )
 ```

2. **Verify Graceful Degradation:**
 ```python
 # Test that operations work even when components fail
 bridge = create_memory_bridge()
 assert bridge.store("key", "value") # Should work with any backend
 ```

---

## Performance Tuning

### Embedding Model Selection

Choose model based on requirements:

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| `all-MiniLM-L6-v2` | 80MB | Fast | Good | General purpose (recommended) |
| `all-mpnet-base-v2` | 420MB | Medium | Better | Higher quality needed |
| `all-MiniLM-L12-v2` | 120MB | Medium | Good | Balance speed/quality |

### Cache Configuration

```python
config = MemoryBridgeConfig(
 max_memories=100000, # Increase for more caching
)

# GraphRAG also has internal caching
# Configured via environment variables:
# GRAPHRAG_CACHE_SIZE=10000
# GRAPHRAG_CACHE_TTL=3600
```

### Compaction Strategy

```python
# Adaptive compaction (recommended for mixed workloads)
config.lsm_compaction_strategy = "adaptive"

# Tiered compaction (optimized for write-heavy)
config.lsm_compaction_strategy = "tiered"

# Leveled compaction (optimized for read-heavy)
config.lsm_compaction_strategy = "leveled"
```

---

## API Reference

See individual module documentation:

- [MemoryBridge API](../src/integration/memory_bridge.py) - Comprehensive docstrings
- [GVulcanBridge API](../src/integration/gvulcan_bridge.py) - Comprehensive docstrings
- [persistant_memory_v46 README](../src/persistant_memory_v46/README.md) - Detailed component docs
- [memory README](../src/memory/README.md) - Governance & optimization docs

---

## Related Documentation

- [Memory System v46](../src/persistant_memory_v46/README.md)
- [Governance & Unlearning](../src/memory/README.md)
- [Integration Architecture](INTEGRATION_ARCHITECTURE.md)
- [Deployment Guide](DEPLOYMENT.md)
- [Security Guide](SECURITY.md)

---

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review component-specific documentation
3. Check logs for detailed error messages
4. Open an issue on GitHub with logs and configuration

---

**Last Updated:** 2026-01-12 
**Version:** 1.0.0
