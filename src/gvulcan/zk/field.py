"""
Finite Field Arithmetic for Groth16

This module implements field element operations modulo the BN128 curve order.
All arithmetic operations are performed in the scalar field F_r where r is the
curve order of the BN128/BN254 elliptic curve.
"""

from __future__ import annotations
from typing import Union
import secrets

# Import curve order from py_ecc
from py_ecc.bn128 import curve_order as CURVE_ORDER


class FieldElement:
    """
    Element in the scalar field of BN128.
    
    All operations are performed modulo CURVE_ORDER, which is the order of
    the BN128 elliptic curve. This is approximately 2^254.
    
    Example:
        >>> a = FieldElement(5)
        >>> b = FieldElement(3)
        >>> c = a + b
        >>> print(c)  # FieldElement(8)
        >>> d = a * b
        >>> print(d)  # FieldElement(15)
    """
    
    def __init__(self, value: Union[int, FieldElement]):
        """
        Initialize field element.
        
        Args:
            value: Integer value or another FieldElement
        """
        if isinstance(value, FieldElement):
            self.value = value.value
        else:
            self.value = int(value) % CURVE_ORDER
    
    def __add__(self, other: Union[FieldElement, int]) -> FieldElement:
        """Add two field elements."""
        if isinstance(other, int):
            other = FieldElement(other)
        return FieldElement((self.value + other.value) % CURVE_ORDER)
    
    def __radd__(self, other: Union[FieldElement, int]) -> FieldElement:
        """Right addition."""
        return self.__add__(other)
    
    def __sub__(self, other: Union[FieldElement, int]) -> FieldElement:
        """Subtract two field elements."""
        if isinstance(other, int):
            other = FieldElement(other)
        return FieldElement((self.value - other.value) % CURVE_ORDER)
    
    def __rsub__(self, other: Union[FieldElement, int]) -> FieldElement:
        """Right subtraction."""
        if isinstance(other, int):
            other = FieldElement(other)
        return FieldElement((other.value - self.value) % CURVE_ORDER)
    
    def __mul__(self, other: Union[FieldElement, int]) -> FieldElement:
        """Multiply two field elements."""
        if isinstance(other, int):
            other = FieldElement(other)
        return FieldElement((self.value * other.value) % CURVE_ORDER)
    
    def __rmul__(self, other: Union[FieldElement, int]) -> FieldElement:
        """Right multiplication."""
        return self.__mul__(other)
    
    def __truediv__(self, other: Union[FieldElement, int]) -> FieldElement:
        """
        Divide two field elements.
        
        Division is implemented as multiplication by the modular inverse.
        """
        if isinstance(other, int):
            other = FieldElement(other)
        return self * other.inverse()
    
    def __pow__(self, exponent: int) -> FieldElement:
        """
        Exponentiation in the field.
        
        Uses modular exponentiation for efficiency.
        """
        if exponent < 0:
            # For negative exponents, use inverse
            return self.inverse() ** (-exponent)
        return FieldElement(pow(self.value, exponent, CURVE_ORDER))
    
    def __neg__(self) -> FieldElement:
        """Negation in the field."""
        return FieldElement((CURVE_ORDER - self.value) % CURVE_ORDER)
    
    def __eq__(self, other: object) -> bool:
        """Check equality."""
        if isinstance(other, FieldElement):
            return self.value == other.value
        elif isinstance(other, int):
            return self.value == (other % CURVE_ORDER)
        return False
    
    def __ne__(self, other: object) -> bool:
        """Check inequality."""
        return not self.__eq__(other)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"FieldElement({self.value})"
    
    def __str__(self) -> str:
        """String conversion."""
        return str(self.value)
    
    def __hash__(self) -> int:
        """Hash for use in sets and dicts."""
        return hash(self.value)
    
    def __int__(self) -> int:
        """Convert to integer."""
        return self.value
    
    def inverse(self) -> FieldElement:
        """
        Compute modular inverse using Fermat's Little Theorem.
        
        For prime modulus p and a != 0: a^(p-1) = 1 (mod p)
        Therefore: a^(-1) = a^(p-2) (mod p)
        
        Returns:
            The multiplicative inverse of this element
            
        Raises:
            ValueError: If trying to invert zero
        """
        if self.value == 0:
            raise ValueError("Cannot invert zero")
        # Use Fermat's little theorem: a^(p-2) = a^(-1) mod p
        return FieldElement(pow(self.value, CURVE_ORDER - 2, CURVE_ORDER))
    
    @classmethod
    def zero(cls) -> FieldElement:
        """Return the additive identity (zero)."""
        return cls(0)
    
    @classmethod
    def one(cls) -> FieldElement:
        """Return the multiplicative identity (one)."""
        return cls(1)
    
    @classmethod
    def random(cls) -> FieldElement:
        """Generate a random field element."""
        return cls(secrets.randbelow(CURVE_ORDER))
