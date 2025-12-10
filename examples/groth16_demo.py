#!/usr/bin/env python3
"""
Example demonstrating the Groth16 zk-SNARK implementation.

This script shows how to:
1. Define a simple circuit (prove knowledge of square root)
2. Generate a proof
3. Verify the proof

The circuit proves: "I know a value x such that x^2 = y"
Without revealing x (zero-knowledge property).
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gvulcan.zk import Circuit, Groth16Prover, R1CSConstraint


def main():
    print("=" * 60)
    print("Groth16 zk-SNARK Demo: Proving Knowledge of Square Root")
    print("=" * 60)
    print()
    
    # Step 1: Define the circuit
    # Circuit: x * x = y (prove knowledge of square root)
    # Variables: [1, y, x] where:
    #   - index 0: constant 1
    #   - index 1: y (public - the square)
    #   - index 2: x (private - the square root)
    
    print("1. Defining Circuit")
    print("-" * 40)
    constraints = [
        R1CSConstraint(
            A=[0, 0, 1],  # x
            B=[0, 0, 1],  # x
            C=[0, 1, 0]   # y
        )
    ]
    
    circuit = Circuit(
        constraints=constraints,
        num_variables=3,
        num_public_inputs=1  # y is public
    )
    print(f"   Circuit: x² = y")
    print(f"   Variables: {circuit.num_variables}")
    print(f"   Constraints: {len(circuit.constraints)}")
    print(f"   Public inputs: {circuit.num_public_inputs}")
    print()
    
    # Step 2: Setup (trusted setup ceremony)
    print("2. Performing Trusted Setup")
    print("-" * 40)
    prover = Groth16Prover(circuit)
    pk, vk = prover.setup()
    print(f"   ✓ Proving key generated")
    print(f"   ✓ Verification key generated")
    print()
    
    # Step 3: Create witness and generate proof
    print("3. Generating Proof")
    print("-" * 40)
    # Private knowledge: x = 3
    # Public value: y = 9
    x = 3
    y = x * x
    witness = [1, y, x]  # [constant, public, private]
    
    print(f"   Secret value (x): {x}")
    print(f"   Public value (y): {y}")
    print(f"   Witness: {witness}")
    
    proof = prover.prove(witness)
    proof_size = len(proof.to_bytes())
    print(f"   ✓ Proof generated: {proof_size} bytes")
    print()
    
    # Step 4: Verify the proof
    print("4. Verifying Proof")
    print("-" * 40)
    public_inputs = [y]  # Only public value y
    is_valid = prover.verify(proof, public_inputs, vk)
    
    if is_valid:
        print(f"   ✓ Proof is VALID")
        print(f"   The prover knows x such that x² = {y}")
        print(f"   (without revealing that x = {x})")
    else:
        print(f"   ✗ Proof is INVALID")
    print()
    
    # Step 5: Try with wrong public input (should fail)
    print("5. Testing with Wrong Public Input")
    print("-" * 40)
    wrong_y = 10
    print(f"   Trying to verify with wrong y = {wrong_y}")
    is_valid_wrong = prover.verify(proof, [wrong_y], vk)
    
    if not is_valid_wrong:
        print(f"   ✓ Correctly rejected (y ≠ {wrong_y})")
    else:
        print(f"   ✗ Incorrectly accepted")
    print()
    
    print("=" * 60)
    print("Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
