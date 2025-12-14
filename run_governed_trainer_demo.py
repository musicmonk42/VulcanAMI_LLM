#!/usr/bin/env python3
"""
Demo script for GovernedTrainer with safety checks and meta self-improvement.

This script demonstrates the usage of the GovernedTrainer class for training
neural network models with built-in safety mechanisms and governance features.
"""

import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

try:
    from src.training.governed_trainer import GovernedTrainer

    print("✓ GovernedTrainer loaded successfully")
except ImportError as e:
    print(f"Error importing GovernedTrainer: {e}")
    print(
        "Please ensure src/training/governed_trainer.py exists and dependencies are installed."
    )
    sys.exit(1)


def run_demo():
    """Run a simple demonstration of GovernedTrainer."""
    print("\n" + "=" * 60)
    print("GovernedTrainer Demo")
    print("=" * 60)

    # Initialize trainer with safety parameters
    trainer = GovernedTrainer(
        learning_rate=1e-3,
        lr_schedule="constant",
        warmup_steps=20,
        total_steps=200,
        log_interval=10,
        checkpoint_interval=100,
        safety_check_interval=10,
        gradient_accumulation_steps=1,
        detect_anomalies=True,
        enable_mixed_precision=False,
        random_seed=42,
        divergence_threshold=100.0,
    )

    print(f"\n✓ Initialized GovernedTrainer with:")
    print(f"  - Learning Rate: {trainer.learning_rate}")
    print(f"  - Safety Checks: Enabled")
    print(f"  - Gradient Accumulation: {trainer.gradient_accumulation_steps} steps")
    print(f"  - Divergence Threshold: {trainer.divergence_threshold}")

    print("\nDemo completed successfully!")
    print("For actual training, use: src/training/train_learnable_bigram.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_demo()
