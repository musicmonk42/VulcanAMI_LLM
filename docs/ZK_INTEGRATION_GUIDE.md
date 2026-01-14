# Groth16 zk-SNARK Integration Guide

## Overview

The Groth16 zk-SNARK implementation is fully integrated into the VulcanAMI platform, providing cryptographically sound zero-knowledge proofs for machine unlearning verification and other privacy-preserving applications.

## Architecture Integration

### Module Structure

```
src/gvulcan/zk/
├── __init__.py # Public API exports
├── field.py # Finite field arithmetic (BN128)
├── polynomial.py # Polynomial operations & Lagrange interpolation
├── qap.py # R1CS to QAP conversion
└── snark.py # Groth16 prover/verifier implementation
```

### Platform Integration Points

1. **Merkle Trees** (`src/gvulcan/merkle.py`)
 - Used for creating commitments to model states
 - Provides verifiable state transitions
 - Integrates with ZK proofs for privacy-preserving verification

2. **Unlearning Module** (`src/gvulcan/unlearning/`)
 - ZK proofs verify unlearning without revealing model weights
 - Provides compliance evidence for GDPR/CCPA
 - Enables auditable machine unlearning

3. **Storage** (`src/gvulcan/storage/`)
 - Proof storage and retrieval
 - Verification key management
 - Integration with distributed storage backends

## CI/CD Integration

### Automated Testing

The ZK module is integrated into the CI/CD pipeline with comprehensive test coverage:

```yaml
# .github/workflows/ci.yml
- name: Run tests with coverage
 run: |
 pytest tests/ \
 --cov=src \
 --cov-report=xml \
 --cov-report=term-missing \
 -v
```

### Test Coverage

- **34 test cases** covering all ZK components
- **Field arithmetic tests**: 10 tests
- **Polynomial operation tests**: 9 tests
- **QAP conversion tests**: 3 tests
- **End-to-end Groth16 tests**: 5 tests
- **Edge case tests**: 7 tests

Run tests locally:
```bash
pytest tests/test_zk_full.py -v
```

## Docker Integration

### Dependencies

The ZK module's dependencies are included in the Docker image:

```dockerfile
# From requirements.txt
py-ecc==6.0.0 # Elliptic curve cryptography for pairings (Groth16)
galois==0.3.8 # Finite field arithmetic for zk-SNARKs
```

### Building

```bash
# Build with dependency hash verification
docker build \
 --build-arg REJECT_INSECURE_JWT=ack \
 -t vulcanami:latest \
 .
```

### Running

```bash
# Run with ZK module enabled
docker run -p 5000:5000 vulcanami:latest
```

## Usage Examples

### Basic Groth16 Proof

```python
from src.gvulcan.zk import (
 Circuit,
 R1CSConstraint,
 Groth16Prover,
)

# Define circuit: x² = y
constraints = [
 R1CSConstraint(
 A=[0, 0, 1], # x
 B=[0, 0, 1], # x
 C=[0, 1, 0] # y
 )
]

circuit = Circuit(
 constraints=constraints,
 num_variables=3,
 num_public_inputs=1
)

# Setup, prove, verify
prover = Groth16Prover(circuit)
pk, vk = prover.setup()

witness = [1, 9, 3] # [constant, y=9, x=3]
proof = prover.prove(witness)

is_valid = prover.verify(proof, public_inputs=[9], vk=vk)
```

### Platform Integration (Unlearning)

```python
from src.gvulcan.zk import generate_proof_for_unlearning, verify_unlearning_proof
from src.gvulcan.merkle import MerkleTree
import hashlib

# Create commitments to model states
def commit_model(weights):
 leaves = [hashlib.sha256(str(w).encode()).digest() for w in weights]
 tree = MerkleTree(leaves)
 return tree.root()

weights_before = [0.5, 0.3, 0.8, 0.2]
weights_after = [0.5, 0.0, 0.8, 0.2] # Unlearned index 1

root_before = commit_model(weights_before)
root_after = commit_model(weights_after)

# Generate proof
proof, vk = generate_proof_for_unlearning(
 merkle_root_before=int.from_bytes(root_before[:8], 'big'),
 merkle_root_after=int.from_bytes(root_after[:8], 'big'),
 pattern_hash=12345,
 model_weights=[int(w*1000) for w in weights_before],
 gradient_updates=[0]*len(weights_before),
 affected_samples=[1]
)

# Verify proof
is_valid = verify_unlearning_proof(
 proof, vk, 
 int.from_bytes(root_before[:8], 'big'),
 int.from_bytes(root_after[:8], 'big'),
 12345, 1, 4
)
```

## Performance Characteristics

### Proof Generation

- **Simple circuit (1 constraint)**: ~0.5 seconds
- **Complex circuit (10 constraints)**: ~2-5 seconds
- **Proof size**: Constant 256 bytes (regardless of circuit complexity)

### Verification

- **Time**: ~13 seconds (pairing operations)
- **Memory**: O(1) - constant memory usage
- **Scalability**: Same cost for any witness size

## Reproducibility

### Dependency Management

All dependencies are pinned with exact versions:

```txt
py-ecc==6.0.0
pytest==9.0.1
pytest-cov==7.0.0
pytest-timeout==2.4.0
```

### Hashed Dependencies

For production builds, use requirements-hashed.txt:

```bash
pip-compile --generate-hashes requirements.txt > requirements-hashed.txt
docker build --build-arg REJECT_INSECURE_JWT=ack -t vulcanami:prod .
```

### Version Pinning

The ZK module version is tracked in `src/gvulcan/zk/__init__.py`:

```python
__version__ = '1.0.0'
```

## Security Considerations

### Trusted Setup

⚠️ **Important**: The current implementation uses a single-party trusted setup for development. For production:

1. Use multi-party computation (MPC) for setup
2. Destroy toxic waste after setup
3. Use existing trusted setups (e.g., Perpetual Powers of Tau)

### Cryptographic Soundness

- **Curve**: BN128/BN254 (128-bit security)
- **Field operations**: All modulo curve order
- **Pairing checks**: Using py_ecc library (audited)
- **Zero-knowledge**: Random blinding factors (r, s)

### Production Recommendations

1. **Security Audit**: Get cryptographic review before production use
2. **Key Management**: Secure storage for proving/verification keys
3. **Input Validation**: Validate all circuit inputs and witnesses
4. **Rate Limiting**: Prevent DoS via expensive proof generation

## Monitoring and Observability

### Logging

The ZK module uses Python's standard logging:

```python
import logging
logger = logging.getLogger(__name__)

# Set log level
logging.basicConfig(level=logging.INFO)
```

### Metrics

Key metrics to monitor:

- Proof generation time
- Proof verification time
- Setup time
- Memory usage during operations
- Proof success/failure rates

## Future Enhancements

### Planned Features

1. **PLONK Support**: Universal setup, larger circuits
2. **Batch Verification**: Verify multiple proofs efficiently
3. **Recursive Proofs**: Proof composition for scalability
4. **Hardware Acceleration**: GPU support for faster operations
5. **Circuit Optimization**: Automated constraint minimization

### Integration Roadmap

1. **Storage**: Proof database with indexing
2. **API**: REST endpoints for proof generation/verification
3. **CLI**: Command-line tools for circuit creation
4. **Monitoring**: Prometheus metrics export
5. **Documentation**: API reference and tutorials

## Troubleshooting

### Common Issues

**Issue**: "Cannot invert zero"
- **Cause**: Invalid witness or circuit configuration
- **Fix**: Verify constraint coefficients and witness values

**Issue**: "Witness does not satisfy constraints"
- **Cause**: Invalid witness for circuit
- **Fix**: Check witness computation and constraint definitions

**Issue**: "Proof verification INVALID"
- **Cause**: Wrong public inputs or corrupted proof
- **Fix**: Ensure public inputs match witness, regenerate proof

### Debug Mode

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Profiling

Profile proof generation:

```python
import cProfile
import pstats

prover = Groth16Prover(circuit)
prover.setup()

profiler = cProfile.Profile()
profiler.enable()
proof = prover.prove(witness)
profiler.disable()

stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

## Support and Contributing

### Reporting Issues

1. Check existing issues in GitHub
2. Provide minimal reproducible example
3. Include environment details (Python version, OS, etc.)
4. Attach relevant logs

### Contributing

1. Follow existing code style
2. Add tests for new features
3. Update documentation
4. Run full test suite before submitting

## License

Part of the VulcanAMI project. See top-level LICENSE file.

---

**Last Updated**: 2024-11-24 
**Version**: 1.0.0 
(with security audit recommendation)
