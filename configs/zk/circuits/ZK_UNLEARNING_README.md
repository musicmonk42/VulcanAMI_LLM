# Vulcan LLM Unlearning Verification Circuit

## Zero-Knowledge Proof System for Verifiable Data Unlearning

**Version**: 1.0.0 
**Last Updated**: 2025-11-14 
**Circuit Type**: Groth16 zkSNARK 
**Security Level**: 128-bit

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Quick Start](#quick-start)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Circuit Specification](#circuit-specification)
8. [Security](#security)
9. [Performance](#performance)
10. [Integration](#integration)
11. [Troubleshooting](#troubleshooting)
12. [FAQ](#faq)

---

## 🎯 Overview

The Vulcan LLM Unlearning Verification Circuit is a zero-knowledge proof system that enables **verifiable data unlearning** in machine learning systems. It allows proving that specific data has been properly removed from an AI model without revealing sensitive information about the data or model parameters.

### What Problem Does It Solve?

Modern data privacy regulations (GDPR, CCPA, etc.) require the ability to delete user data from AI systems. However, proving that data has been truly removed from a trained model is challenging because:

1. **Privacy**: You can't reveal the model's internal state
2. **Completeness**: You must prove all traces are removed
3. **Verifiability**: Third parties need to verify the deletion
4. **Efficiency**: Verification must be fast and scalable

### Our Solution

This circuit enables **cryptographic proof** that:
- ✅ Specific documents were marked for unlearning
- ✅ Corresponding embeddings were zeroed or modified
- ✅ Model parameters were appropriately updated
- ✅ Privacy budgets were respected
- ✅ **All without revealing any private data**

---

## ✨ Features

### Core Capabilities

- **Document Membership Verification**: Prove documents are in the unlearning set using Merkle proofs
- **Embedding Zeroing**: Verify embeddings have been properly zeroed or modified
- **Model Update Verification**: Prove model parameters changed appropriately
- **Privacy Budget Tracking**: Verify differential privacy guarantees
- **Commitment Binding**: Ensure integrity using cryptographic commitments
- **Timestamp Validation**: Ensure operations occurred in valid time ranges

### Security Properties

- **128-bit Security**: Based on discrete logarithm hardness on BN128 curve
- **Zero-Knowledge**: Reveals nothing about private inputs
- **Soundness**: <2^-128 probability of accepting false proofs
- **Perfect Completeness**: Valid proofs always accepted
- **Knowledge Soundness**: Witness extractable from valid proofs

### Performance

- **Proof Generation**: ~30 seconds
- **Proof Verification**: ~5 milliseconds
- **Proof Size**: 256 bytes (constant)
- **Batch Processing**: Up to 256 embeddings per proof

---

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────────┐
│ Unlearning Request │
│ (Document IDs, Timestamps, Privacy Requirements) │
└───────────────────┬─────────────────────────────────┘
 │
 ▼
┌─────────────────────────────────────────────────────┐
│ Witness Generator │
│ (Computes private inputs from system state) │
└───────────────────┬─────────────────────────────────┘
 │
 ▼
┌─────────────────────────────────────────────────────┐
│ ZK Prover │
│ (Generates cryptographic proof) │
│ • Input: Public + Private Signals │
│ • Output: Proof (256 bytes) │
└───────────────────┬─────────────────────────────────┘
 │
 ▼
┌─────────────────────────────────────────────────────┐
│ ZK Verifier │
│ (Verifies proof in ~5ms) │
│ • Input: Proof + Public Inputs │
│ • Output: Accept/Reject │
└─────────────────────────────────────────────────────┘
```

### Data Flow

1. **User Request**: Initiate unlearning for specific documents
2. **System Processing**: Update embeddings and model parameters
3. **Witness Generation**: Collect proof of proper unlearning
4. **Proof Generation**: Create ZK proof (~30s)
5. **Proof Verification**: Verify compliance (~5ms)
6. **Audit Trail**: Store proof on blockchain/audit log

---

## 🚀 Quick Start

### Prerequisites

```bash
# System requirements
- CPU: 16+ cores
- RAM: 64GB+
- Disk: 100GB+ free
- OS: Linux (Ubuntu 22.04+)

# Software dependencies
- Node.js 18+
- Circom 2.1.8
- snarkjs 0.7.0
- Rust 1.70+ (optional, for rapidsnark)
```

### Installation

```bash
# 1. Clone repository
git clone https://github.com/vulcanami/zk-unlearning
cd zk-unlearning

# 2. Install dependencies
./build_circuit.sh install-deps

# 3. Download Powers of Tau (1.2GB)
./build_circuit.sh download-ptau

# 4. Compile circuit (~15 minutes)
./build_circuit.sh compile

# 5. Run trusted setup (~1 hour)
./build_circuit.sh setup

# 6. Export keys
./build_circuit.sh export-keys
```

### Quick Test

```bash
# Run full test pipeline
./build_circuit.sh full-test

# Expected output:
# ✓ Witness computed
# ✓ Proof generated
# ✓ Proof verified successfully
```

---

## 📦 Installation

### Detailed Installation Steps

#### Step 1: Install System Dependencies

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y build-essential curl git wget

# Install Node.js 18
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# Install Rust (for rapidsnark)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

#### Step 2: Install Circom

```bash
# Download Circom 2.1.8
wget https://github.com/iden3/circom/releases/download/v2.1.8/circom-linux-amd64
sudo mv circom-linux-amd64 /usr/local/bin/circom
sudo chmod +x /usr/local/bin/circom

# Verify installation
circom --version
# Expected: circom compiler 2.1.8
```

#### Step 3: Install snarkjs

```bash
# Install globally via npm
sudo npm install -g snarkjs@0.7.0

# Verify installation
snarkjs --version
# Expected: 0.7.0
```

#### Step 4: Install rapidsnark (Optional, for faster proving)

```bash
git clone https://github.com/iden3/rapidsnark.git
cd rapidsnark
npm install
git submodule init
git submodule update
npm run task createFieldSources
npm run task buildProver
```

#### Step 5: Setup Project

```bash
# Create project directory
mkdir -p ~/zk-unlearning
cd ~/zk-unlearning

# Copy circuit files
cp unlearning_v1.0.circom circuits/
cp build_circuit.sh .
chmod +x build_circuit.sh

# Initialize
./build_circuit.sh install-deps
```

---

## 📖 Usage

### Basic Usage

#### 1. Compile Circuit

```bash
./build_circuit.sh compile
```

**Output:**
- `build/unlearning_v1.0.r1cs` - R1CS file (~500MB)
- `build/unlearning_v1.0_js/` - WASM witness generator
- `build/unlearning_v1.0.sym` - Symbol file

#### 2. Run Trusted Setup

```bash
./build_circuit.sh setup
```

**Output:**
- `keys/unlearning_v1.0_final.zkey` - Proving key (~2.5GB)
- `keys/verification_key.json` - Verification key (~2KB)
- `keys/verifier.sol` - Solidity verifier contract

#### 3. Generate Test Proof

```bash
# Create test input
./build_circuit.sh test-input

# Compute witness
./build_circuit.sh witness

# Generate proof
./build_circuit.sh prove

# Verify proof
./build_circuit.sh verify
```

### Advanced Usage

#### Custom Input Generation

Create `test/input.json`:

```json
{
 "requestId": "123456789",
 "timestamp": "1700000000",
 "versionBefore": "1",
 "versionAfter": "2",
 "unlearningSetRoot": "0x1234...",
 "embeddingTreeRootBefore": "0x5678...",
 "embeddingTreeRootAfter": "0x9abc...",
 "modelStateRootBefore": "0xdef0...",
 "modelStateRootAfter": "0x1234...",
 "privacyBudget": "100",
 "sensitivityParameter": "10",
 "embeddingCommitmentBefore": "0x5678...",
 "embeddingCommitmentAfter": "0x9abc...",
 "modelCommitmentBefore": "0xdef0...",
 "modelCommitmentAfter": "0x1234...",
 "requireStrictZeroing": "1",
 "requirePrivacyGuarantee": "1",
 "documentIds": ["1", "2", "3", ...],
 "documentMerkleProofs": [...],
 "documentMerkleIndices": [...],
 "embeddingsBefore": [...],
 "embeddingsAfter": [...],
 "embeddingBlindingBefore": [...],
 "embeddingBlindingAfter": [...],
 "modelParamsBefore": [...],
 "modelParamsAfter": [...],
 "modelBlindingBefore": "...",
 "modelBlindingAfter": "...",
 "unlearningMask": [...],
 "updateThreshold": "...",
 "queriesCountBefore": "...",
 "queriesCountAfter": "..."
}
```

Then run:

```bash
node build/unlearning_v1.0_js/generate_witness.js \
 build/unlearning_v1.0_js/unlearning_v1.0.wasm \
 test/input.json \
 build/witness.wtns
```

#### Proof Generation with rapidsnark

```bash
# Faster proof generation (requires rapidsnark)
rapidsnark keys/unlearning_v1.0_final.zkey \
 build/witness.wtns \
 build/proof.json \
 build/public.json
```

#### Batch Verification

```bash
# Verify multiple proofs
for proof in proofs/*.json; do
 snarkjs groth16 verify \
 keys/verification_key.json \
 ${proof%.json}_public.json \
 $proof
done
```

---

## 📐 Circuit Specification

### Circuit Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Embedding Dimension | 1536 | Size of each embedding vector |
| Num Embeddings | 256 | Number of embeddings per batch |
| Merkle Depth | 20 | Supports ~1M documents |
| Model Param Size | 128 | Number of model parameters to verify |
| Total Constraints | ~15.7M | Circuit complexity |
| Public Inputs | 17 | Publicly visible signals |
| Private Inputs | 394,496 | Secret witness values |

### Public Inputs

1. **requestId** - Unique unlearning request identifier
2. **timestamp** - Unix timestamp of operation
3. **versionBefore** - Model version before unlearning
4. **versionAfter** - Model version after unlearning
5. **unlearningSetRoot** - Merkle root of unlearning set
6. **embeddingTreeRootBefore** - Embedding tree root (before)
7. **embeddingTreeRootAfter** - Embedding tree root (after)
8. **modelStateRootBefore** - Model state hash (before)
9. **modelStateRootAfter** - Model state hash (after)
10. **privacyBudget** - Maximum privacy loss (ε)
11. **sensitivityParameter** - Privacy sensitivity
12. **embeddingCommitmentBefore** - Embedding commitment (before)
13. **embeddingCommitmentAfter** - Embedding commitment (after)
14. **modelCommitmentBefore** - Model commitment (before)
15. **modelCommitmentAfter** - Model commitment (after)
16. **requireStrictZeroing** - Strict zeroing flag
17. **requirePrivacyGuarantee** - Privacy verification flag

### Verification Steps

The circuit verifies the following in order:

1. ✅ **Timestamp & Version Validation**
2. ✅ **Document Membership** (Merkle proofs)
3. ✅ **Embedding Commitments** (cryptographic binding)
4. ✅ **Embedding Zeroing** (proper deletion)
5. ✅ **Model Parameter Updates** (appropriate changes)
6. ✅ **Privacy Budget** (differential privacy)
7. ✅ **Unlearning Effectiveness** (similarity metrics)
8. ✅ **Merkle Root Updates** (state changes)
9. ✅ **Completeness** (at least one embedding unlearned)
10. ✅ **Request Binding** (integrity of public inputs)

---

## 🔐 Security

### Security Properties

#### Soundness
- **Probability of accepting false proof**: <2^-128
- **Based on**: Discrete logarithm hardness on BN128 curve
- **Assumptions**: q-Power Discrete Logarithm (q-PDL)

#### Zero-Knowledge
- **Information leaked**: Computationally zero
- **Based on**: Decisional Diffie-Hellman on BN128
- **Property**: Simulator indistinguishable from real proofs

#### Completeness
- **Probability of rejecting valid proof**: 0
- **Property**: Perfect completeness (Groth16)
- **Guarantee**: All valid witnesses produce valid proofs

### Threat Model

#### What We Protect Against

✅ **Malicious Prover**
- Cannot forge proofs without valid witness
- Cannot prove false statements
- Cannot learn information from proof generation

✅ **Malicious Verifier**
- Cannot learn private inputs from proof
- Cannot forge verification results
- Cannot extract witness from proof

✅ **Man-in-the-Middle**
- Cannot modify proofs (cryptographic binding)
- Cannot replay old proofs (timestamp + request ID)
- Cannot correlate proofs to private data

#### What We Don't Protect Against

❌ **Trusted Setup Compromise**
- If setup is compromised, proofs can be forged
- Mitigation: Multi-party computation for setup
- Mitigation: Public randomness via blockchain

❌ **Side-Channel Attacks**
- Timing attacks on proof generation
- Mitigation: Constant-time implementations
- Mitigation: Secure hardware (SGX/TPU)

❌ **Quantum Computers**
- Groth16 not quantum-resistant
- Future: Migrate to post-quantum SNARKs

### Security Best Practices

1. **Multi-Party Trusted Setup**
 - Use at least 50 participants
 - Include diverse stakeholders
 - Public randomness from blockchain
 - Verify entire transcript

2. **Key Management**
 - Store proving keys securely
 - Rotate verification keys periodically
 - Use HSM for critical keys
 - Backup with encryption

3. **Proof Storage**
 - Store proofs on blockchain for auditability
 - Include timestamp and request ID
 - Maintain proof history
 - Regular integrity checks

4. **Integration Security**
 - Validate all public inputs
 - Rate limit proof generation
 - Monitor for unusual patterns
 - Log all operations

---

## ⚡ Performance

### Benchmarks

#### Hardware Configuration
```
CPU: AMD EPYC 7763 (64 cores)
RAM: 256GB DDR4
Storage: 2TB NVMe SSD
GPU: NVIDIA A100 (optional)
```

#### Performance Metrics

| Operation | Time | Memory | Notes |
|-----------|------|--------|-------|
| Circuit Compilation | 15-20 min | 16GB | One-time setup |
| Trusted Setup Phase 2 | 30-45 min | 32GB | One-time setup |
| Witness Generation | 5-10 sec | 32GB | Per proof |
| Proof Generation (CPU) | 25-35 sec | 64GB | Per proof |
| Proof Generation (GPU) | 8-12 sec | 32GB | With rapidsnark |
| Proof Verification | 5-8 ms | 1MB | Per proof |
| Batch Verification (100) | ~200 ms | 10MB | Amortized |

#### Scalability

- **Embeddings**: Linear in number of embeddings (256 max per batch)
- **Merkle Depth**: Linear in tree depth (20 levels = 1M documents)
- **Model Parameters**: Linear in parameter count (128 default)
- **Batch Processing**: Embarrassingly parallel (unlimited batches)

#### Optimization Tips

1. **Use GPU Acceleration**
 ```bash
 # Install rapidsnark with CUDA support
 npm run task buildPoverWithCUDA
 ```

2. **Parallel Witness Generation**
 ```bash
 # Process multiple witnesses in parallel
 parallel -j 16 ./generate_witness.sh ::: inputs/*.json
 ```

3. **Optimize Memory**
 ```bash
 # Use memory-mapped files for large witnesses
 export NODE_OPTIONS="--max-old-space-size=65536"
 ```

4. **Cache Powers of Tau**
 ```bash
 # Reuse across multiple circuits
 ln -s /shared/ptau/powersOfTau28_hez_final_24.ptau
 ```

---

## 🔌 Integration

### REST API

#### Generate Proof

```bash
POST /api/v1/proof/generate
Content-Type: application/json

{
 "requestId": "unlearn-123",
 "documentIds": ["doc1", "doc2"],
 "timestamp": 1700000000,
 "embeddings": {...},
 "modelState": {...}
}

Response:
{
 "proof": "0x...",
 "publicInputs": [...],
 "generationTime": 28.3,
 "verified": true
}
```

#### Verify Proof

```bash
POST /api/v1/proof/verify
Content-Type: application/json

{
 "proof": "0x...",
 "publicInputs": [...]
}

Response:
{
 "valid": true,
 "verificationTime": 0.005,
 "timestamp": 1700000100
}
```

### Python SDK

```python
from vulcan_zk import UnlearningProver, UnlearningVerifier

# Initialize prover
prover = UnlearningProver(
 circuit_path="circuits/unlearning_v1.0.circom",
 proving_key="keys/proving_key.zkey"
)

# Generate proof
proof = prover.generate_proof(
 request_id="unlearn-123",
 document_ids=["doc1", "doc2"],
 embeddings_before=embeddings_before,
 embeddings_after=embeddings_after,
 # ... other inputs
)

# Verify proof
verifier = UnlearningVerifier(
 verification_key="keys/verification_key.json"
)

is_valid = verifier.verify(
 proof=proof.proof,
 public_inputs=proof.public_inputs
)

print(f"Proof valid: {is_valid}")
```

### JavaScript/TypeScript SDK

```typescript
import { UnlearningProver, UnlearningVerifier } from '@vulcan/zk-unlearning';

// Generate proof
const prover = new UnlearningProver({
 circuitWasm: 'build/unlearning_v1.0.wasm',
 provingKey: 'keys/proving_key.zkey'
});

const proof = await prover.generateProof({
 requestId: 'unlearn-123',
 documentIds: ['doc1', 'doc2'],
 embeddingsBefore: embeddingsBefore,
 embeddingsAfter: embeddingsAfter,
 // ... other inputs
});

// Verify proof
const verifier = new UnlearningVerifier({
 verificationKey: 'keys/verification_key.json'
});

const isValid = await verifier.verify({
 proof: proof.proof,
 publicInputs: proof.publicInputs
});

console.log(`Proof valid: ${isValid}`);
```

### Smart Contract Integration

```solidity
// Deploy verifier contract
contract UnlearningVerifier {
 // Auto-generated from circuit
 function verifyProof(
 uint[2] memory a,
 uint[2][2] memory b,
 uint[2] memory c,
 uint[17] memory input
 ) public view returns (bool) {
 // Groth16 verification
 return verify(a, b, c, input);
 }
}

// Use in your contract
contract UnlearningRegistry {
 UnlearningVerifier public verifier;
 
 mapping(bytes32 => bool) public unlearningProofs;
 
 function registerUnlearning(
 bytes32 requestId,
 uint[2] memory a,
 uint[2][2] memory b,
 uint[2] memory c,
 uint[17] memory input
 ) public {
 require(
 verifier.verifyProof(a, b, c, input),
 "Invalid proof"
 );
 
 require(
 uint256(input[0]) == uint256(requestId),
 "Request ID mismatch"
 );
 
 unlearningProofs[requestId] = true;
 emit UnlearningVerified(requestId, block.timestamp);
 }
}
```

---

## 🐛 Troubleshooting

### Common Issues

#### Issue 1: Out of Memory During Compilation

**Symptoms:**
```
Error: Cannot allocate memory
```

**Solution:**
```bash
# Increase swap space
sudo fallocate -l 64G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Or use machine with more RAM (128GB+ recommended)
```

#### Issue 2: Witness Generation Fails

**Symptoms:**
```
Error: Signal not found: embeddingsBefore[0][0]
```

**Solution:**
```bash
# Verify input JSON matches circuit interface
./build_circuit.sh info # Check expected inputs

# Validate JSON syntax
jq . test/input.json

# Check array dimensions match
# embeddingsBefore should be [256][1536]
```

#### Issue 3: Proof Verification Fails

**Symptoms:**
```
Error: Invalid proof
```

**Solution:**
```bash
# Verify keys match circuit
./build_circuit.sh export-keys

# Regenerate witness and proof
./build_circuit.sh witness
./build_circuit.sh prove

# Check public inputs match
cat build/public.json
```

#### Issue 4: Slow Proof Generation

**Symptoms:**
```
Proof generation taking >5 minutes
```

**Solution:**
```bash
# Enable multi-threading
export OMP_NUM_THREADS=32

# Use GPU acceleration
# Install rapidsnark with CUDA

# Reduce batch size (if possible)
# Use fewer embeddings per proof
```

### Debug Commands

```bash
# Check circuit info
snarkjs r1cs info build/unlearning_v1.0.r1cs

# Print circuit constraints
snarkjs r1cs print build/unlearning_v1.0.r1cs build/unlearning_v1.0.sym

# Verify setup
snarkjs zkey verify \
 build/unlearning_v1.0.r1cs \
 ptau/powersOfTau28_hez_final_24.ptau \
 keys/unlearning_v1.0_final.zkey

# Test with minimal input
./build_circuit.sh test-input
./build_circuit.sh full-test
```

---

## ❓ FAQ

**Q: How long does proof generation take?** 
A: ~30 seconds on a 32-core CPU, ~10 seconds with GPU acceleration.

**Q: Can I batch multiple unlearning requests?** 
A: Yes, up to 256 embeddings per proof. For more, generate multiple proofs in parallel.

**Q: Is the proof size constant?** 
A: Yes, Groth16 proofs are always 256 bytes regardless of circuit size.

**Q: How do I verify proofs on-chain?** 
A: Deploy the generated Solidity verifier contract and call `verifyProof()`.

**Q: What happens if the trusted setup is compromised?** 
A: Attackers could forge proofs. Use multi-party computation with 50+ participants.

**Q: Can quantum computers break this?** 
A: Yes, Groth16 is not quantum-resistant. Future versions will use post-quantum SNARKs.

**Q: How much does it cost to verify on Ethereum?** 
A: ~250,000 gas (~$10 at 50 gwei, $2000/ETH). Use L2s for lower costs.

---

## 📚 Additional Resources

- **Circuit Code**: `unlearning_v1.0.circom`
- **Specification**: `circuit_specification.yaml`
- **Build Script**: `build_circuit.sh`
- **API Documentation**: See INTEGRATION.md
- **Security Analysis**: See SECURITY.md
- **Performance Guide**: See PERFORMANCE.md

## 📞 Support

- **Email**: zk-support@vulcanami.io
- **Slack**: #zk-circuits
- **Documentation**: https://docs.vulcanami.io/zk
- **GitHub**: https://github.com/vulcanami/zk-unlearning

---

**Copyright © 2025 VulcanAMI Team. All rights reserved.**