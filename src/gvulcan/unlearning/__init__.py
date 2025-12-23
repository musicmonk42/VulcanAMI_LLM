"""Unlearning Module - Gradient surgery implementation for machine unlearning."""

from .gradient_surgery import GradientSurgery, unlearn_with_gradient_surgery

__all__ = ["GradientSurgery", "unlearn_with_gradient_surgery"]
