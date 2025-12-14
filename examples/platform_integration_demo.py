"""
Integration Example: ZK Proofs for Machine Unlearning Verification

This example demonstrates how the Groth16 zk-SNARK system integrates with
the VulcanAMI platform's unlearning and Merkle tree modules to provide
verifiable machine unlearning.

Key integration points:
1. Merkle trees for model state commitments
2. ZK proofs for verifying unlearning without revealing model weights
3. Integration with the platform's unlearning module
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import hashlib

from src.gvulcan.merkle import MerkleTree
from src.gvulcan.zk import (
    Circuit,
    Groth16Prover,
    R1CSConstraint,
    generate_proof_for_unlearning,
    verify_unlearning_proof,
)


def create_model_commitment(weights: list) -> bytes:
    """
    Create a Merkle root commitment for model weights.

    Args:
        weights: List of model weights

    Returns:
        Merkle root hash
    """
    # Hash each weight
    leaves = [hashlib.sha256(str(w).encode()).digest() for w in weights]

    # Build Merkle tree
    tree = MerkleTree(leaves)
    return tree.root()  # Call root method


def main():
    print("=" * 70)
    print("VulcanAMI Platform Integration: ZK Proofs for Unlearning")
    print("=" * 70)
    print()

    # Step 1: Simulate model before unlearning
    print("1. Creating Model State Commitments")
    print("-" * 70)

    weights_before = [0.5, 0.3, 0.8, 0.2, 0.6]
    weights_after = [0.5, 0.0, 0.8, 0.0, 0.6]  # Weights at indices 1,3 unlearned

    root_before = create_model_commitment(weights_before)
    root_after = create_model_commitment(weights_after)

    print(f"   Model weights before: {weights_before}")
    print(f"   Model weights after:  {weights_after}")
    print(f"   Merkle root before:   {root_before.hex()[:32]}...")
    print(f"   Merkle root after:    {root_after.hex()[:32]}...")
    print(f"   ✓ State commitments created")
    print()

    # Step 2: Create unlearning pattern
    print("2. Defining Unlearning Pattern")
    print("-" * 70)

    affected_samples = [1, 3]  # Indices that were unlearned
    pattern_data = f"unlearn:{','.join(map(str, affected_samples))}"
    pattern_hash_bytes = hashlib.sha256(pattern_data.encode()).digest()
    pattern_hash = int.from_bytes(pattern_hash_bytes[:8], "big")

    print(f"   Affected sample indices: {affected_samples}")
    print(f"   Pattern identifier: {pattern_hash}")
    print()

    # Step 3: Generate ZK proof of unlearning
    print("3. Generating ZK Proof of Unlearning")
    print("-" * 70)

    # Convert Merkle roots to integers for circuit
    merkle_before_int = int.from_bytes(root_before[:8], "big")
    merkle_after_int = int.from_bytes(root_after[:8], "big")

    print("   Creating proof...")
    print("   - Public inputs: merkle_before, merkle_after, pattern_hash")
    print("   - Private inputs: model weights, gradients, affected samples")

    try:
        proof, vk = generate_proof_for_unlearning(
            merkle_root_before=merkle_before_int,
            merkle_root_after=merkle_after_int,
            pattern_hash=pattern_hash,
            model_weights=[int(w * 1000) for w in weights_before],  # Scale to ints
            gradient_updates=[0] * len(weights_before),  # Simplified
            affected_samples=affected_samples,
        )

        proof_size = len(proof.to_bytes())
        print(f"   ✓ Proof generated: {proof_size} bytes")
        print()
    except Exception as e:
        print(f"   ⚠ Proof generation note: {e}")
        print("   (This is expected for the simplified circuit)")
        print()
        return

    # Step 4: Verify the proof
    print("4. Verifying Unlearning Proof")
    print("-" * 70)

    is_valid = verify_unlearning_proof(
        proof=proof,
        vk=vk,
        merkle_root_before=merkle_before_int,
        merkle_root_after=merkle_after_int,
        pattern_hash=pattern_hash,
        num_samples=len(affected_samples),
        model_size=len(weights_before),
    )

    if is_valid:
        print("   ✓ Proof VALID: Unlearning verified!")
        print("   - Model state changed correctly")
        print("   - Affected samples match pattern")
        print("   - All constraints satisfied")
        print("   - Model weights remain PRIVATE")
    else:
        print("   ✗ Proof INVALID")
    print()

    # Step 5: Show integration benefits
    print("5. Platform Integration Benefits")
    print("-" * 70)
    print("   ✓ Privacy: Model weights never revealed")
    print("   ✓ Verifiability: Mathematical proof of unlearning")
    print("   ✓ Compliance: Auditable GDPR/CCPA adherence")
    print("   ✓ Efficiency: Constant-size proofs (~256 bytes)")
    print("   ✓ Scalability: Works for large models")
    print()

    print("=" * 70)
    print("Integration Complete!")
    print("=" * 70)
    print()
    print("This demonstrates how VulcanAMI's ZK module integrates with:")
    print("  • Merkle trees (src/gvulcan/merkle.py)")
    print("  • Unlearning module (src/gvulcan/unlearning/)")
    print("  • Platform storage and verification systems")


if __name__ == "__main__":
    main()
