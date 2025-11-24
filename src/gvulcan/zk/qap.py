"""
Quadratic Arithmetic Program (QAP) for Groth16

This module converts R1CS (Rank-1 Constraint System) constraints into a
Quadratic Arithmetic Program using Lagrange interpolation. This is a key
step in creating zk-SNARKs.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List
from .field import FieldElement
from .polynomial import Polynomial


@dataclass
class QAP:
    """
    Quadratic Arithmetic Program representation.
    
    A QAP converts the R1CS constraints into polynomials. For each variable i,
    we have three polynomials:
    - A_i(x): encodes the left input coefficients
    - B_i(x): encodes the right input coefficients
    - C_i(x): encodes the output coefficients
    
    The witness satisfies the constraints if and only if:
    (Σ witness[i] * A_i(x)) * (Σ witness[i] * B_i(x)) - (Σ witness[i] * C_i(x))
    is divisible by the vanishing polynomial t(x).
    """
    A_polys: List[Polynomial]  # One per variable
    B_polys: List[Polynomial]  # One per variable
    C_polys: List[Polynomial]  # One per variable
    t_poly: Polynomial         # Vanishing polynomial
    num_variables: int
    num_public_inputs: int
    domain: List[FieldElement]  # Evaluation points


def r1cs_to_qap(circuit) -> QAP:
    """
    Convert R1CS circuit to Quadratic Arithmetic Program via Lagrange interpolation.
    
    For each variable i (0 to num_variables-1):
      - A_i(x) interpolates: at constraint j, value = A[j][i]
      - B_i(x) interpolates: at constraint j, value = B[j][i]  
      - C_i(x) interpolates: at constraint j, value = C[j][i]
    
    The domain is {1, 2, ..., num_constraints} for simplicity.
    The vanishing polynomial is t(x) = (x-1)(x-2)...(x-num_constraints).
    
    Args:
        circuit: Circuit object with R1CS constraints
        
    Returns:
        QAP representation
    """
    num_constraints = len(circuit.constraints)
    num_variables = circuit.num_variables
    num_public_inputs = circuit.num_public_inputs
    
    # Create domain: {1, 2, 3, ..., m} where m = num_constraints
    domain = [FieldElement(i + 1) for i in range(num_constraints)]
    
    # Initialize polynomial lists
    A_polys = []
    B_polys = []
    C_polys = []
    
    # For each variable i, interpolate A_i(x), B_i(x), C_i(x)
    for var_idx in range(num_variables):
        # Collect points for A_i(x): (constraint_j, A[j][i])
        A_points = []
        B_points = []
        C_points = []
        
        for constraint_idx, constraint in enumerate(circuit.constraints):
            x = domain[constraint_idx]  # x = constraint number (1, 2, 3, ...)
            
            # Get coefficient for this variable in this constraint
            # Pad with zeros if coefficient list is shorter
            if var_idx < len(constraint.A):
                a_val = FieldElement(constraint.A[var_idx])
            else:
                a_val = FieldElement.zero()
            
            if var_idx < len(constraint.B):
                b_val = FieldElement(constraint.B[var_idx])
            else:
                b_val = FieldElement.zero()
            
            if var_idx < len(constraint.C):
                c_val = FieldElement(constraint.C[var_idx])
            else:
                c_val = FieldElement.zero()
            
            A_points.append((x, a_val))
            B_points.append((x, b_val))
            C_points.append((x, c_val))
        
        # Interpolate polynomials
        A_i = Polynomial.lagrange_interpolation(A_points)
        B_i = Polynomial.lagrange_interpolation(B_points)
        C_i = Polynomial.lagrange_interpolation(C_points)
        
        A_polys.append(A_i)
        B_polys.append(B_i)
        C_polys.append(C_i)
    
    # Compute vanishing polynomial t(x)
    t_poly = Polynomial.vanishing_polynomial(domain)
    
    return QAP(
        A_polys=A_polys,
        B_polys=B_polys,
        C_polys=C_polys,
        t_poly=t_poly,
        num_variables=num_variables,
        num_public_inputs=num_public_inputs,
        domain=domain
    )


def compute_h_polynomial(
    qap: QAP,
    witness: List[FieldElement]
) -> Polynomial:
    """
    Compute h(x) = (A(x) * B(x) - C(x)) / t(x)
    
    Where:
      A(x) = Σ witness[i] * A_i(x)
      B(x) = Σ witness[i] * B_i(x)
      C(x) = Σ witness[i] * C_i(x)
    
    If the witness satisfies the constraints, then A(x)*B(x) - C(x) is exactly
    divisible by t(x), meaning the remainder is zero. This is the fundamental
    property that makes zk-SNARKs work.
    
    Args:
        qap: QAP representation
        witness: Witness values as FieldElements
        
    Returns:
        h(x) polynomial
        
    Raises:
        ValueError: If witness doesn't satisfy constraints (remainder != 0)
        
    Performance note:
        For large circuits, this is the most expensive operation in proof generation.
        The polynomial multiplication and division dominate the runtime.
    """
    if len(witness) != qap.num_variables:
        raise ValueError(
            f"Witness size {len(witness)} doesn't match QAP variables {qap.num_variables}"
        )
    
    # Compute A(x) = Σ witness[i] * A_i(x)
    A_x = Polynomial([FieldElement.zero()])
    for i, w in enumerate(witness):
        # Scale A_i(x) by witness[i]
        scaled_coeffs = [c * w for c in qap.A_polys[i].coeffs]
        scaled_poly = Polynomial(scaled_coeffs)
        A_x = A_x + scaled_poly
    
    # Compute B(x) = Σ witness[i] * B_i(x)
    B_x = Polynomial([FieldElement.zero()])
    for i, w in enumerate(witness):
        scaled_coeffs = [c * w for c in qap.B_polys[i].coeffs]
        scaled_poly = Polynomial(scaled_coeffs)
        B_x = B_x + scaled_poly
    
    # Compute C(x) = Σ witness[i] * C_i(x)
    C_x = Polynomial([FieldElement.zero()])
    for i, w in enumerate(witness):
        scaled_coeffs = [c * w for c in qap.C_polys[i].coeffs]
        scaled_poly = Polynomial(scaled_coeffs)
        C_x = C_x + scaled_poly
    
    # Compute numerator = A(x) * B(x) - C(x)
    numerator = (A_x * B_x) - C_x
    
    # Divide by t(x)
    h_poly, remainder = divmod(numerator, qap.t_poly)
    
    # Check that division is exact (remainder should be zero)
    if not remainder.is_zero():
        raise ValueError(
            "Witness does not satisfy constraints: "
            "A(x)*B(x) - C(x) is not divisible by t(x). "
            "This indicates the witness does not satisfy all R1CS constraints."
        )
    
    return h_poly
    
    return h_poly
