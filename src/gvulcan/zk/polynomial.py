"""
Polynomial Arithmetic for Groth16

This module implements polynomial operations over finite fields.
Polynomials are used in the Quadratic Arithmetic Program (QAP) representation
of arithmetic circuits.
"""

from __future__ import annotations

from typing import List, Tuple

from .field import FieldElement


class Polynomial:
    """
    Polynomial with FieldElement coefficients.

    Coefficients are stored in ascending order: coeffs[i] is the coefficient of x^i.
    For example, coeffs=[1, 2, 3] represents 1 + 2x + 3x^2.

    Example:
        >>> # Create polynomial 1 + 2x + 3x^2
        >>> p = Polynomial([FieldElement(1), FieldElement(2), FieldElement(3)])
        >>> # Evaluate at x=2
        >>> result = p.evaluate(FieldElement(2))
        >>> print(result)  # 1 + 2*2 + 3*4 = 17
    """

    def __init__(self, coeffs: List[FieldElement]):
        """
        Initialize polynomial.

        Args:
            coeffs: List of coefficients in ascending order (coeffs[i] = coeff of x^i)
        """
        # Strip trailing zeros
        while len(coeffs) > 1 and coeffs[-1] == FieldElement.zero():
            coeffs.pop()

        self.coeffs = coeffs if coeffs else [FieldElement.zero()]

    def degree(self) -> int:
        """
        Return the degree of the polynomial.

        The degree is the highest power with a non-zero coefficient.
        The zero polynomial has degree -1.
        """
        if len(self.coeffs) == 1 and self.coeffs[0] == FieldElement.zero():
            return -1
        return len(self.coeffs) - 1

    def evaluate(self, x: FieldElement) -> FieldElement:
        """
        Evaluate polynomial at a given point using Horner's method.

        Horner's method: p(x) = a0 + x(a1 + x(a2 + x(...)))
        This is more efficient than computing powers of x separately.

        Args:
            x: Point at which to evaluate

        Returns:
            p(x)
        """
        if not self.coeffs:
            return FieldElement.zero()

        # Horner's method: start from highest degree
        result = self.coeffs[-1]
        for i in range(len(self.coeffs) - 2, -1, -1):
            result = result * x + self.coeffs[i]

        return result

    def __add__(self, other: Polynomial) -> Polynomial:
        """Add two polynomials."""
        max_len = max(len(self.coeffs), len(other.coeffs))
        result = []

        for i in range(max_len):
            a = self.coeffs[i] if i < len(self.coeffs) else FieldElement.zero()
            b = other.coeffs[i] if i < len(other.coeffs) else FieldElement.zero()
            result.append(a + b)

        return Polynomial(result)

    def __sub__(self, other: Polynomial) -> Polynomial:
        """Subtract two polynomials."""
        max_len = max(len(self.coeffs), len(other.coeffs))
        result = []

        for i in range(max_len):
            a = self.coeffs[i] if i < len(self.coeffs) else FieldElement.zero()
            b = other.coeffs[i] if i < len(other.coeffs) else FieldElement.zero()
            result.append(a - b)

        return Polynomial(result)

    def __mul__(self, other: Polynomial) -> Polynomial:
        """Multiply two polynomials."""
        if self.degree() == -1 or other.degree() == -1:
            return Polynomial([FieldElement.zero()])

        # Result has degree = sum of degrees
        result = [FieldElement.zero()] * (len(self.coeffs) + len(other.coeffs) - 1)

        for i, a in enumerate(self.coeffs):
            for j, b in enumerate(other.coeffs):
                result[i + j] = result[i + j] + (a * b)

        return Polynomial(result)

    def __divmod__(self, divisor: Polynomial) -> Tuple[Polynomial, Polynomial]:
        """
        Polynomial long division.

        Returns (quotient, remainder) such that:
        self = quotient * divisor + remainder

        This is critical for computing h(x) = (A*B - C) / t(x) in Groth16.

        Args:
            divisor: Polynomial to divide by

        Returns:
            Tuple of (quotient, remainder)

        Raises:
            ValueError: If divisor is zero
        """
        if divisor.degree() == -1:
            raise ValueError("Cannot divide by zero polynomial")

        # Make a copy of dividend
        remainder = Polynomial(self.coeffs[:])
        quotient = Polynomial([FieldElement.zero()])

        # Polynomial long division
        while remainder.degree() >= divisor.degree():
            # Compute leading term of quotient
            degree_diff = remainder.degree() - divisor.degree()
            leading_coeff = remainder.coeffs[-1] / divisor.coeffs[-1]

            # Create monomial: leading_coeff * x^degree_diff
            term_coeffs = [FieldElement.zero()] * (degree_diff + 1)
            term_coeffs[-1] = leading_coeff
            term = Polynomial(term_coeffs)

            # Add to quotient
            quotient = quotient + term

            # Subtract term * divisor from remainder
            remainder = remainder - (term * divisor)

        return quotient, remainder

    def __eq__(self, other: object) -> bool:
        """Check equality of polynomials."""
        if not isinstance(other, Polynomial):
            return False
        return self.coeffs == other.coeffs

    def __repr__(self) -> str:
        """String representation."""
        if self.degree() == -1:
            return "Polynomial([0])"

        terms = []
        for i, c in enumerate(self.coeffs):
            if c != FieldElement.zero():
                if i == 0:
                    terms.append(str(c.value))
                elif i == 1:
                    terms.append(f"{c.value}x" if c.value != 1 else "x")
                else:
                    terms.append(f"{c.value}x^{i}" if c.value != 1 else f"x^{i}")

        return " + ".join(terms) if terms else "0"

    @classmethod
    def lagrange_interpolation(
        cls, points: List[Tuple[FieldElement, FieldElement]]
    ) -> Polynomial:
        """
        Lagrange interpolation to find polynomial passing through given points.

        Given points [(x0, y0), (x1, y1), ..., (xn, yn)], returns the unique
        polynomial of degree at most n that passes through all points.

        Formula: L(x) = Σ y_i * Π_{j≠i} (x - x_j) / (x_i - x_j)

        This is the core of R1CS→QAP conversion.

        Args:
            points: List of (x, y) tuples

        Returns:
            Interpolated polynomial
        """
        if not points:
            return cls([FieldElement.zero()])

        n = len(points)
        result = cls([FieldElement.zero()])

        for i in range(n):
            x_i, y_i = points[i]

            # Compute Lagrange basis polynomial L_i(x)
            # L_i(x) = Π_{j≠i} (x - x_j) / (x_i - x_j)
            basis = cls([FieldElement.one()])

            for j in range(n):
                if i != j:
                    x_j = points[j][0]

                    # Multiply by (x - x_j)
                    # This is equivalent to multiplying by polynomial [−x_j, 1]
                    term = cls([-x_j, FieldElement.one()])
                    basis = basis * term

                    # Divide by (x_i - x_j)
                    denominator = x_i - x_j
                    basis_coeffs = [c / denominator for c in basis.coeffs]
                    basis = cls(basis_coeffs)

            # Multiply by y_i and add to result
            scaled_coeffs = [c * y_i for c in basis.coeffs]
            scaled_basis = cls(scaled_coeffs)
            result = result + scaled_basis

        return result

    @classmethod
    def vanishing_polynomial(cls, domain: List[FieldElement]) -> Polynomial:
        """
        Compute vanishing polynomial for a domain.

        The vanishing polynomial is t(x) = (x - d[0]) * (x - d[1]) * ... * (x - d[n-1])
        It equals zero at all points in the domain.

        This is used as the divisor in h(x) = (A*B - C) / t(x).

        Args:
            domain: List of domain points

        Returns:
            Vanishing polynomial
        """
        if not domain:
            return cls([FieldElement.one()])

        # Start with t(x) = 1
        result = cls([FieldElement.one()])

        # Multiply by (x - d_i) for each domain point
        for d in domain:
            # (x - d) is represented as [-d, 1]
            term = cls([-d, FieldElement.one()])
            result = result * term

        return result

    def is_zero(self) -> bool:
        """Check if polynomial is zero."""
        return self.degree() == -1 or all(c == FieldElement.zero() for c in self.coeffs)
