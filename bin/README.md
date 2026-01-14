# VulcanAMI Command-Line Tools

**Version**: 4.6.0 
**License**: Proprietary

## Overview

The VulcanAMI CLI provides a comprehensive suite of command-line tools for managing graph data packs, performing machine unlearning, verifying zero-knowledge proofs, and operating vector storage systems.

## Installation

```bash
# Copy tools to system bin directory
sudo cp vulcan-* /usr/local/bin/

# Make executable
sudo chmod +x /usr/local/bin/vulcan-*

# Verify installation
vulcan-cli --version
```

## Tools

### 1. vulcan-cli

**Main command-line interface** - Entry point for all VulcanAMI operations

```bash
# Show help
vulcan-cli --help

# Show version
vulcan-cli --version

# Check system configuration
vulcan-cli config

# Execute subcommands
vulcan-cli pack build -i data.json -o output.pack
vulcan-cli verify pack output.pack
vulcan-cli unlearn pattern "user_id:12345"
```

**Features**:
- Unified interface for all tools
- Color-coded output
- Verbose and debug modes
- Configuration file support
- Comprehensive help system

### 2. vulcan-pack

**Pack builder** - Create optimized packfiles with DQS integration

```bash
# Build pack from stdin
cat data.txt | vulcan-pack -o output.pack

# Build from JSON file
vulcan-pack -i data.json -o output.pack

# Build from directory
vulcan-pack -d /data -o output.pack --recursive

# High compression
vulcan-pack -i data.json -o output.pack --compression 9

# Skip DQS checks (faster)
vulcan-pack -i data.json -o output.pack --no-dqs

# Strict quality threshold
vulcan-pack -i data.json -o output.pack --dqs-threshold 0.90
```

**Features**:
- Multi-dimensional DQS integration
- PII detection and review
- Zstd compression (levels 1-22)
- Bloom filter generation
- Merkle tree construction
- Parallel processing
- Progress indicators
- Statistics output

**Options**:
- `-i, --input FILE` - Input file
- `-f, --file-list FILE` - List of files to pack
- `-d, --directory DIR` - Directory to pack
- `-o, --output FILE` - Output packfile
- `-c, --compression LEVEL` - Compression level (1-22)
- `--bloom-size SIZE` - Bloom filter size (KB)
- `--no-dqs` - Disable DQS checking
- `--dqs-threshold SCORE` - Minimum quality score
- `-r, --recursive` - Recursive directory processing
- `-v, --verbose` - Verbose output
- `--stats FILE` - Write statistics to JSON

### 3. vulcan-pack-verify

**Pack verifier** - Verify packfile integrity

```bash
# Basic verification
vulcan-pack-verify output.pack

# Full verification (all chunks)
vulcan-pack-verify output.pack --full

# Verify Merkle tree
vulcan-pack-verify output.pack --merkle

# Verify bloom filter
vulcan-pack-verify output.pack --bloom

# Verify all checksums
vulcan-pack-verify output.pack --checksums

# Output JSON results
vulcan-pack-verify output.pack --json results.json
```

**Features**:
- Header validation
- Merkle tree verification
- Bloom filter accuracy checks
- Checksum validation
- Chunk-level verification
- JSON output format

**Checks**:
- Magic number validation
- Version compatibility
- Chunk count verification
- Size validation
- Merkle root verification
- Bloom filter integrity
- Compression validation

### 4. vulcan-unlearn

**Unlearning engine** - Machine unlearning with ZK proofs

```bash
# Unlearn data matching pattern
vulcan-unlearn "user_id:12345"

# Fast lane (skip verification)
vulcan-unlearn "email:user@example.com" --fast-lane

# Specific packfile
vulcan-unlearn "pattern" --packfile pack-001.gpk2

# Different strategy
vulcan-unlearn "pattern" --strategy deletion

# Skip proof generation
vulcan-unlearn "pattern" --no-proof

# Verify after unlearning
vulcan-unlearn "pattern" --verify
```

**Features**:
- Gradient surgery unlearning
- Zero-knowledge proof generation
- Merkle DAG integration
- Audit trail logging
- Multiple strategies
- Verification support

**Strategies**:
- `gradient_surgery` - Neural network gradient updates (default)
- `deletion` - Simple data deletion
- `perturbation` - Data perturbation

**Options**:
- `-p, --packfile FILE` - Specific packfile
- `-s, --strategy TYPE` - Unlearning strategy
- `--fast-lane` - Skip verification
- `--no-proof` - Skip ZK proof generation
- `--verify` - Verify completeness
- `-v, --verbose` - Verbose output
- `--json FILE` - Output JSON results

### 5. vulcan-proof-verify-zk

**ZK proof verifier** - Verify zero-knowledge proofs

```bash
# Verify proof from file
vulcan-proof-verify-zk proof.json

# Verify with public inputs
vulcan-proof-verify-zk proof.json -p public_inputs.json

# Custom circuit
vulcan-proof-verify-zk proof.json --circuit custom.circom

# Custom verification key
vulcan-proof-verify-zk proof.json --vkey vkey.json

# JSON output
vulcan-proof-verify-zk proof.json --json results.json
```

**Features**:
- Groth16 proof verification
- Public input validation
- Custom circuit support
- Batch verification
- Performance metrics

**Options**:
- `-p, --public-inputs FILE` - Public inputs JSON
- `--circuit FILE` - Circuit file path
- `--vkey FILE` - Verification key path
- `-v, --verbose` - Verbose output
- `--json FILE` - Output JSON results

### 6. vulcan-repack

**Pack optimizer** - Repack and optimize packfiles

```bash
# Adaptive repacking
vulcan-repack pack-001.gpk2

# Aggressive compression
vulcan-repack pack-001.gpk2 --strategy aggressive

# Conservative (preserve compatibility)
vulcan-repack pack-001.gpk2 --strategy conservative

# High compression level
vulcan-repack pack-001.gpk2 --compression 9

# Custom output path
vulcan-repack pack-001.gpk2 -o optimized.pack

# Dry run (simulate)
vulcan-repack pack-001.gpk2 --dry-run
```

**Features**:
- Adaptive compression
- Space optimization
- Chunk deduplication
- Merkle tree rebuilding
- Statistics reporting

**Strategies**:
- `adaptive` - Balance size and speed (default)
- `aggressive` - Maximum compression
- `conservative` - Preserve compatibility

**Options**:
- `-s, --strategy TYPE` - Repacking strategy
- `-c, --compression LEVEL` - Compression level
- `-o, --output FILE` - Output path
- `--dry-run` - Simulate without writing
- `-v, --verbose` - Verbose output
- `--json FILE` - Output statistics

### 7. vulcan-prefetch-vectors

**Vector prefetcher** - Prefetch vectors for queries

```bash
# Prefetch for query
vulcan-prefetch-vectors query-123

# Hot tier prefetch
vulcan-prefetch-vectors query-123 --tier hot

# Top-k vectors
vulcan-prefetch-vectors query-123 --top-k 100

# ML-predicted strategy
vulcan-prefetch-vectors query-123 --strategy ml_predicted

# Popularity-based
vulcan-prefetch-vectors query-123 --strategy popularity
```

**Features**:
- ML-based prediction
- Tier-aware prefetching
- Redis caching
- Milvus integration
- Performance metrics

**Strategies**:
- `ml_predicted` - ML-based prediction (default)
- `popularity` - Popularity-based
- `recent` - Recent access patterns

**Options**:
- `-t, --tier TIER` - Storage tier (hot/warm/cold)
- `-k, --top-k N` - Number of vectors
- `--strategy TYPE` - Prefetch strategy
- `-v, --verbose` - Verbose output
- `--json FILE` - Output results

### 8. vulcan-vector-bootstrap

**Vector bootstrap** - Initialize vector storage

```bash
# Bootstrap all tiers
vulcan-vector-bootstrap

# Specific tier
vulcan-vector-bootstrap --tier hot

# Custom dimensions
vulcan-vector-bootstrap --dimension 256

# Different metric
vulcan-vector-bootstrap --metric COSINE

# HNSW index
vulcan-vector-bootstrap --index-type HNSW

# Drop existing collections
vulcan-vector-bootstrap --drop-existing
```

**Features**:
- Multi-tier initialization
- Milvus collection creation
- Index configuration
- Dimension customization
- Metric selection

**Tiers**:
- `hot` - Fast access, high memory
- `warm` - Balanced performance
- `cold` - Archive, low cost

**Index Types**:
- `FLAT` - Brute force (accurate)
- `IVF_FLAT` - Inverted file (fast)
- `IVF_SQ8` - Scalar quantization (memory efficient)
- `HNSW` - Hierarchical Navigable Small World (balanced)

**Options**:
- `-t, --tier TIER` - Storage tier
- `-d, --dimension N` - Vector dimension
- `-m, --metric TYPE` - Distance metric (L2/IP/COSINE)
- `--index-type TYPE` - Index type
- `--drop-existing` - Drop before creating
- `-v, --verbose` - Verbose output
- `--json FILE` - Output results

## Configuration

### Environment Variables

```bash
# Service endpoints
export DQS_ENDPOINT="http://dqs-service:8080"
export OPA_ENDPOINT="http://opa-service:8181"

# Database
export POSTGRES_HOST="postgres"
export POSTGRES_PORT="5432"
export POSTGRES_DB="vulcanami"
export POSTGRES_USER="vulcanami"
export POSTGRES_PASSWORD="password"

# Redis
export REDIS_HOST="redis"
export REDIS_PORT="6379"
export REDIS_DB="1"

# Milvus
export MILVUS_HOST="milvus"
export MILVUS_PORT="19530"

# Paths
export VULCAN_DATA_DIR="/var/lib/vulcanami"
export VULCAN_LOG_DIR="/var/log/vulcanami"
export VULCAN_CACHE_DIR="/var/cache/vulcanami"

# Verbosity
export VULCAN_VERBOSE="0"
export VULCAN_DEBUG="0"
export VULCAN_QUIET="0"
export VULCAN_NO_COLOR="0"
```

### Configuration File

Create `~/.vulcanrc`:

```bash
# VulcanAMI Configuration

# Service endpoints
DQS_ENDPOINT="http://dqs-service:8080"
OPA_ENDPOINT="http://opa-service:8181"

# Database
POSTGRES_HOST="postgres"
REDIS_HOST="redis"
MILVUS_HOST="milvus"

# Defaults
VULCAN_VERBOSE=0
VULCAN_DEBUG=0
```

## Examples

### Complete Workflow

```bash
# 1. Build a packfile
vulcan-pack -i dataset.json -o dataset.pack --compression 6

# 2. Verify the pack
vulcan-pack-verify dataset.pack --full

# 3. Bootstrap vector storage
vulcan-vector-bootstrap --tier all --dimension 128

# 4. Unlearn sensitive data
vulcan-unlearn "user_id:12345" --verify

# 5. Verify unlearning proof
vulcan-proof-verify-zk unlearning_proof.json -p public_inputs.json

# 6. Repack for optimization
vulcan-repack dataset.pack -o optimized.pack --strategy aggressive

# 7. Prefetch vectors for query
vulcan-prefetch-vectors query-456 --tier hot --top-k 100
```

### Automation

```bash
#!/bin/bash
# Daily pack maintenance

# Repack old packs
for pack in /data/packs/*.pack; do
 age=$(find "$pack" -mtime +7)
 if [ -n "$age" ]; then
 vulcan-repack "$pack" --strategy adaptive
 fi
done

# Verify all packs
for pack in /data/packs/*.pack; do
 vulcan-pack-verify "$pack" || echo "FAILED: $pack"
done
```

## Troubleshooting

### Common Issues

**Pack build fails with DQS error**
```bash
# Check DQS service
curl http://dqs-service:8080/health

# Disable DQS temporarily
vulcan-pack -i data.json -o output.pack --no-dqs
```

**Verification fails**
```bash
# Enable debug mode
vulcan-pack-verify output.pack --verbose

# Check specific component
vulcan-pack-verify output.pack --merkle
vulcan-pack-verify output.pack --checksums
```

**Unlearning slow**
```bash
# Use fast lane (skip verification)
vulcan-unlearn "pattern" --fast-lane

# Skip proof generation
vulcan-unlearn "pattern" --no-proof
```

## Performance Tips

1. **Pack Building**
 - Use `--no-dqs` for trusted data (10x faster)
 - Lower compression for speed: `--compression 1`
 - Use parallel workers: `--parallel 4`

2. **Verification**
 - Skip full verification for routine checks
 - Use bloom filters for membership tests

3. **Unlearning**
 - Use `--fast-lane` when confidence is high
 - Batch multiple patterns together
 - Skip proof generation for internal operations

4. **Vector Operations**
 - Prefetch during off-peak hours
 - Use appropriate tier for access patterns
 - Monitor cache hit rates

## Security

### Best Practices

1. **Credentials**
 - Never pass passwords as arguments
 - Use environment variables or secret managers
 - Rotate keys regularly

2. **Data Protection**
 - Always enable DQS for untrusted data
 - Use encryption for sensitive packs
 - Audit all unlearning operations

3. **Access Control**
 - Restrict tool access to authorized users
 - Log all operations
 - Review audit trails regularly

## Support

- **Documentation**: https://docs.vulcanami.io
- **Issues**: https://github.com/vulcanami/cli/issues
- **Email**: support@vulcanami.io
- **Slack**: #vulcanami-cli

## License

Copyright © 2025 VulcanAMI. All rights reserved.

Proprietary and confidential. Unauthorized use is prohibited.

---

**VulcanAMI CLI Tools** - Enterprise Graph Data Platform