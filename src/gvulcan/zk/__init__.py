"""
Zero-Knowledge Proof System using Groth16

This package implements Groth16 zk-SNARKs with proper QAP conversion.

Main components:
- FieldElement: Finite field arithmetic modulo BN128 curve order
- Polynomial: Polynomial arithmetic with Lagrange interpolation
- QAP: Quadratic Arithmetic Program for R1CS to polynomial conversion
- Groth16Prover: Complete Groth16 proof system (setup, prove, verify)

Example:
    from src.gvulcan.zk import FieldElement, Groth16Prover, Circuit, R1CSConstraint

    # Define circuit
    constraints = [R1CSConstraint(A=[...], B=[...], C=[...])]
    circuit = Circuit(constraints, num_variables=3, num_public_inputs=1)

    # Create prover and setup
    prover = Groth16Prover(circuit)
    pk, vk = prover.setup()

    # Generate proof
    witness = [1, 9, 3]  # [constant, public, private]
    proof = prover.prove(witness)

    # Verify
    is_valid = prover.verify(proof, public_inputs=[9], vk=vk)
"""

from .field import FieldElement, CURVE_ORDER
from .polynomial import Polynomial
from .qap import QAP, r1cs_to_qap, compute_h_polynomial
from .snark import (
    Circuit,
    R1CSConstraint,
    Groth16Prover,
    Groth16Proof,
    ProvingKey,
    VerificationKey,
    create_unlearning_circuit,
    generate_proof_for_unlearning,
    verify_unlearning_proof,
)

__all__ = [
    # Field arithmetic
    "FieldElement",
    "CURVE_ORDER",
    # Polynomials
    "Polynomial",
    # QAP
    "QAP",
    "r1cs_to_qap",
    "compute_h_polynomial",
    # Groth16
    "Circuit",
    "R1CSConstraint",
    "Groth16Prover",
    "Groth16Proof",
    "ProvingKey",
    "VerificationKey",
    # Unlearning integration
    "create_unlearning_circuit",
    "generate_proof_for_unlearning",
    "verify_unlearning_proof",
]

__version__ = "1.0.0"
