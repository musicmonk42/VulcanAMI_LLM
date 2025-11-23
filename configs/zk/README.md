# Zero-Knowledge Proof Configurations

This directory contains configurations for zero-knowledge proof circuits used in VulcanAMI's privacy-preserving operations.

## Contents

### circuits/
Contains circuit specifications and implementations for ZK proofs.

#### Files
- `circuit_specification.yaml` - Circuit specification and parameters
- `unlearning_v1.0.circom` - Circom circuit for unlearning verification
- `unlearning_v1.0.r1cs` - R1CS constraint system for the unlearning circuit
- `build_circuit.sh` - Build script for compiling circuits
- `ZK_UNLEARNING_README.md` - Detailed documentation on unlearning circuits

## Purpose

The ZK proof circuits enable:
- **Privacy-preserving unlearning**: Verify data removal without revealing the data
- **Compliance verification**: Prove compliance with data regulations (GDPR, etc.)
- **Audit trails**: Create verifiable proofs of operations without exposing sensitive data

## Circuit Specification

The circuit specification defines:
- Input/output parameters
- Constraint definitions
- Proof generation parameters
- Verification requirements

## Usage

To build circuits:
```bash
cd circuits
./build_circuit.sh
```

For detailed usage, see `circuits/ZK_UNLEARNING_README.md`.

## Security Considerations

- All circuits are designed to be zero-knowledge (reveal no information beyond validity)
- Constraint systems are thoroughly tested for soundness
- Proofs are non-interactive (using Groth16 or PLONK)
- Trusted setup ceremonies (if required) are documented

## Dependencies

- circom (circuit compiler)
- snarkjs (proof generation/verification)
- Node.js (for tooling)

## Version

Current circuit version: v1.0.0

## References

- [Circom Documentation](https://docs.circom.io/)
- [ZK-SNARKS Overview](https://z.cash/technology/zksnarks/)
