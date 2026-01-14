# Zero-Knowledge Proof System - Custom Implementation

⚠️ **IMPORTANT**: This is a **custom educational implementation**, NOT standard Groth16 or PLONK.

## Overview

This directory contains a custom zero-knowledge proof system that demonstrates ZK principles using real cryptographic primitives. It's designed for:
- 📚 **Education** - Learning how ZK proofs work
- 🔬 **Research** - Prototyping ZK integration patterns 
- 🧪 **Testing** - Internal development and testing
- ❌ **NOT for production** security-critical systems

## What's Included

### Core Files

- **`snark.py`** - Main ZK proof system with custom protocol
 - `Groth16Prover` class (custom, not standard Groth16)
 - R1CS constraint system
 - Proof generation and verification
 - Circuit creation utilities

- **`verify.py`** - Verification logic
 - Proof verification functions
 - Public input validation
 - Merkle tree verification

## Key Features

### ✅ What Works

- Real elliptic curve cryptography (BN128/BN254 via py_ecc)
- Merkle tree commitments for data integrity
- Hash-based zero-knowledge commitments
- R1CS constraint representation
- Proof generation and verification logic
- Integration with model unlearning

### ⚠️ Limitations

- **Not standard Groth16** - Custom protocol, not compatible with standard implementations
- **Simplified QAP** - Polynomial generation uses simplified approach
- **No security audit** - Not reviewed by cryptography experts
- **Educational focus** - Prioritizes clarity over optimization
- **Not blockchain-compatible** - Cannot be verified on Ethereum or other chains

## Quick Start

### Basic Usage

```python
from src.gvulcan.zk.snark import Groth16Prover, create_unlearning_circuit

# Create circuit
circuit = create_unlearning_circuit(num_samples=10, model_size=100)

# Initialize prover
prover = Groth16Prover(circuit)

# Setup
proving_key, verification_key = prover.setup()

# Generate proof
witness = [1, 2, 3] # Private data
proof = prover.prove(witness, proving_key)

# Verify proof
public_inputs = [1] # Public data
is_valid = prover.verify(proof, public_inputs, verification_key)
```

### Model Unlearning Example

```python
from src.gvulcan.zk.snark import create_unlearning_circuit
from src.gvulcan.merkle import MerkleTree
import hashlib

# Create Merkle commitments to model states
def commit_weights(weights):
 leaves = [hashlib.sha256(str(w).encode()).digest() for w in weights]
 tree = MerkleTree(leaves)
 return tree.get_root()

weights_before = [0.5, 0.3, 0.8, 0.2]
weights_after = [0.5, 0.0, 0.8, 0.2] # Unlearned

root_before = commit_weights(weights_before)
root_after = commit_weights(weights_after)

# Generate ZK proof
circuit = create_unlearning_circuit(
 num_samples=len(weights_before),
 model_size=len(weights_before)
)
prover = Groth16Prover(circuit)
proving_key, verification_key = prover.setup()

witness = [1] + weights_before + weights_after
proof = prover.prove(witness, proving_key)

# Verify (only roots are public, weights stay private)
public_inputs = [
 int.from_bytes(root_before[:8], 'big'),
 int.from_bytes(root_after[:8], 'big')
]
is_valid = prover.verify(proof, public_inputs, verification_key)

print(f"Unlearning verified: {is_valid}")
```

## Architecture

### Proof System Components

```
┌─────────────────────────────────────────────────────┐
│ User Application │
└──────────────────┬──────────────────────────────────┘
 │
┌──────────────────▼──────────────────────────────────┐
│ Groth16Prover (Custom) │
│ ┌──────────────────────────────────────────────┐ │
│ │ Circuit Definition (R1CS) │ │
│ │ - Constraints │ │
│ │ - Variables │ │
│ │ - Public inputs │ │
│ └──────────────────────────────────────────────┘ │
│ │
│ ┌──────────────────────────────────────────────┐ │
│ │ Setup Phase │ │
│ │ - Generate proving key │ │
│ │ - Generate verification key │ │
│ │ - (Simplified trusted setup) │ │
│ └──────────────────────────────────────────────┘ │
│ │
│ ┌──────────────────────────────────────────────┐ │
│ │ Prove Phase │ │
│ │ - Take private witness │ │
│ │ - Check constraints │ │
│ │ - Generate proof │ │
│ └──────────────────────────────────────────────┘ │
│ │
│ ┌──────────────────────────────────────────────┐ │
│ │ Verify Phase │ │
│ │ - Take proof + public inputs │ │
│ │ - Verify using verification key │ │
│ │ - Return true/false │ │
│ └──────────────────────────────────────────────┘ │
└─────────────────┬───────────────────────────────────┘
 │
┌─────────────────▼───────────────────────────────────┐
│ Cryptographic Primitives (py_ecc) │
│ - BN128/BN254 elliptic curve │
│ - Finite field arithmetic │
│ - (Partial) Pairing operations │
└──────────────────────────────────────────────────────┘
```

### Comparison to Standard Systems

| Component | This Implementation | Standard Groth16 | Standard PLONK |
|-----------|---------------------|------------------|----------------|
| **Circuit** | R1CS-inspired | Full R1CS | Custom gates |
| **Polynomials** | Simplified | Full QAP | Permutation polynomial |
| **Setup** | Basic | MPC ceremony | Universal setup |
| **Proof Size** | Variable | ~200 bytes | ~1KB |
| **Verification** | Hash-based | Pairing check | Pairing check |
| **Security** | Educational | Audited | Audited |

## Security Considerations

### What This Implementation Provides

✅ **Computational Hiding**: Private data is hidden using cryptographic hashes
✅ **Binding**: Merkle trees ensure data integrity 
✅ **Real Cryptography**: Uses industry-standard elliptic curves

### What This Implementation Does NOT Provide

❌ **Formal Security Proofs**: No mathematical proof of security
❌ **Audit Trail**: Not reviewed by cryptography experts
❌ **Standard Compliance**: Not compatible with Groth16/PLONK standards
❌ **Production Guarantees**: Not suitable for high-security applications

### Threat Model

**Protected Against:**
- Casual inspection of private data
- Accidental data leakage in logs
- Non-adversarial verification

**NOT Protected Against:**
- Determined adversary with cryptanalysis capabilities
- Side-channel attacks
- Implementation bugs in proof generation
- Malicious prover with unlimited resources

## Migration to Production ZK

If you need production-grade ZK proofs, follow this migration path:

### Phase 1: Evaluation (1 week)
- Evaluate circom + snarkjs vs arkworks vs libsnark
- Consider team expertise and performance needs
- Review existing trusted setups

### Phase 2: Circuit Design (2-3 weeks)
- Define public and private inputs formally
- Design proper constraint system
- Implement in chosen framework (Circom recommended)
- Test circuit correctness thoroughly

### Phase 3: Integration (1-2 weeks)
- Replace custom prover with standard library
- Update witness generation logic
- Modify verification endpoints
- Test end-to-end flows

### Phase 4: Security (2-3 weeks)
- Security audit by cryptography experts
- Penetration testing
- Fix identified issues
- Re-audit critical changes

### Total Time: 6-9 weeks with dedicated team

### Recommended Path

For most projects, we recommend:
1. **Use circom + snarkjs** (JavaScript, well-documented)
2. **Leverage existing trusted setups** (e.g., Perpetual Powers of Tau)
3. **Start simple** (small circuits, validate correctness)
4. **Scale gradually** (add complexity after basic system works)
5. **Get audit early** (before production deployment)

## FAQ

### Q: Can I use this in production?
**A:** No. This is for development and education only. Use standard Groth16/PLONK implementations for production.

### Q: Why is it called Groth16Prover if it's not standard Groth16?
**A:** Historical naming. The class implements a Groth16-inspired structure but uses simplified approaches. Consider it "Groth16-like" or "educational Groth16".

### Q: What's the performance?
**A:** 
- Setup: <1s
- Proof generation: ~0.1-1s (depends on circuit size)
- Verification: <0.1s

Real Groth16 is slower (1-10s for proof generation) but provides cryptographic guarantees.

### Q: Is the code secure?
**A:** The code uses real cryptographic primitives (py_ecc) but the overall protocol is **not audited** and **not suitable for production**. Use at your own risk for non-security-critical applications only.

### Q: Can I integrate this with Ethereum?
**A:** No. The proofs are not compatible with Ethereum's zk-SNARK verification contracts. For Ethereum, use circom + snarkjs.

### Q: What about PLONK or STARKs?
**A:** Those are alternative proof systems with different tradeoffs:
- **PLONK**: Transparent setup, larger proofs, universal setup
- **STARKs**: Post-quantum secure, very large proofs, no trusted setup

For most applications, Groth16 is still the best choice.

## Resources

### Documentation
- [Main ZK Implementation Notes](../../../docs/ZK_IMPLEMENTATION_NOTES.md) - Detailed technical documentation
- [py_ecc Documentation](https://github.com/ethereum/py_ecc) - Elliptic curve library

### Learning Resources
- [Zero-Knowledge Proofs MOOC](https://zk-learning.org/) - Free online course
- [Why and How zk-SNARK Works](https://arxiv.org/abs/1906.07221) - Comprehensive paper
- [Circom Documentation](https://docs.circom.io/) - For real implementations

### Production Libraries
- [snarkjs](https://github.com/iden3/snarkjs) - JavaScript (recommended)
- [arkworks](https://arkworks.rs/) - Rust
- [libsnark](https://github.com/scipr-lab/libsnark) - C++

## Contributing

This is a custom implementation for internal use. For contributions:

1. Understand this is **educational code**, not production
2. Maintain clarity and documentation
3. Add tests for any new functionality
4. Update this README with examples

For production ZK features, integrate a standard library instead of extending this custom implementation.

## License

Part of the VulcanAMI_LLM project. See top-level LICENSE file.

## Disclaimer

This zero-knowledge proof implementation is provided "as is" for educational and development purposes only. It has not been audited by cryptography experts and is not suitable for production use where cryptographic security guarantees are required. Use at your own risk.

For production systems requiring zero-knowledge proofs, use well-audited, standard implementations such as circom + snarkjs, arkworks, or libsnark.
