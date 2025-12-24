"""Unlearning Module - Gradient surgery implementation for machine unlearning."""

from .gradient_surgery import GradientSurgeryUnlearner, unlearn_with_gradient_surgery

__all__ = ["GradientSurgeryUnlearner", "unlearn_with_gradient_surgery"]
