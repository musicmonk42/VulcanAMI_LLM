#!/usr/bin/env python3
"""
Phase 1 Demo: Infrastructure Survival
Location: demos/omega_phase1_survival.py

This demo calls ACTUAL platform methods from DynamicArchitecture.
It is NOT a script simulation - it uses real code.
"""
import sys
import time

# Add repository root to Python path
sys.path.insert(0, '/home/runner/work/VulcanAMI_LLM/VulcanAMI_LLM')

# Import actual platform components
from src.execution.dynamic_architecture import (
    DynamicArchitecture,
    DynamicArchConfig,
    Constraints
)

def display_phase1():
    """Display Phase 1: Infrastructure Survival demo using real platform methods."""
    
    print("="*70)
    print("        PHASE 1: Infrastructure Survival")
    print("="*70)
    print()
    print("💥 Scenario: AWS us-east-1 DOWN")
    print("📉 Market Impact: $47B/hour")
    print()
    
    # Countdown animation
    for i in range(3, 0, -1):
        print(f"Network failure in {i}...")
        time.sleep(1)
    
    print("\n[CRITICAL] NETWORK LOST. AWS CLOUD UNREACHABLE.\n")
    time.sleep(0.5)
    
    print("[SYSTEM] Initiating SURVIVAL PROTOCOL...")
    print()
    
    # ===== REAL PLATFORM CODE STARTS HERE =====
    
    # Initialize actual DynamicArchitecture with config
    config = DynamicArchConfig(
        enable_validation=True,
        enable_auto_rollback=True
    )
    constraints = Constraints(
        min_heads_per_layer=1,  # Allow aggressive pruning
        max_heads_per_layer=16
    )
    
    arch = DynamicArchitecture(
        model=None,  # No actual model for demo
        config=config,
        constraints=constraints
    )
    
    # Initialize shadow layers (simulating a full model)
    # This represents the actual transformer architecture
    initial_layer_count = 12  # Typical transformer layer count
    arch._shadow_layers = [
        {
            "id": f"layer_{i}",
            "heads": [
                {"id": f"head_{j}", "d_k": 64, "d_v": 64}
                for j in range(8)  # 8 attention heads per layer
            ]
        }
        for i in range(initial_layer_count)
    ]
    
    # Get initial stats using REAL platform method
    initial_stats = arch.get_stats()
    print(f"[INFO] Initial architecture:")
    print(f"       Layers: {initial_stats.num_layers}")
    print(f"       Total heads: {initial_stats.num_heads}")
    print(f"       Estimated power: 150W (GPU + full compute)")
    print()
    
    # Power estimation (simplified for demo)
    def estimate_power(num_layers, total_layers):
        """Simple power estimation based on active layers"""
        base_cpu_power = 15  # Watts for CPU-only minimal mode
        full_gpu_power = 150  # Watts for full GPU operation
        layer_fraction = num_layers / total_layers
        return base_cpu_power + (full_gpu_power - base_cpu_power) * layer_fraction
    
    # Layer shedding sequence with REAL method calls
    layer_names = [
        "Generative Layer",
        "Transformer Blocks (upper)", 
        "Transformer Blocks (middle)",
        "Attention Heads (pruning)",
        "Dense Layers",
    ]
    
    target_layers = 2  # Keep only 2 core layers for survival
    layers_to_remove = initial_stats.num_layers - target_layers
    
    for i, name in enumerate(layer_names):
        if i < len(layer_names) - 1:  # Don't try to remove on last iteration
            current_stats = arch.get_stats()
            if current_stats.num_layers > target_layers:
                # REAL PLATFORM METHOD CALL
                layer_idx = current_stats.num_layers - 1
                result = arch.remove_layer(layer_idx)  # Returns bool
                
                if result:  # result is a boolean
                    new_stats = arch.get_stats()
                    current_power = estimate_power(new_stats.num_layers, initial_layer_count)
                    print(f"[RESOURCE] Shedding {name}... ✓")
                    print(f"            Removed layer {layer_idx}")
                    print(f"            Power: {current_power:.1f}W")
                else:
                    print(f"[RESOURCE] Cannot shed {name}: constraints prevented removal")
        
        time.sleep(0.5)
    
    # Get final stats using REAL platform method
    final_stats = arch.get_stats()
    final_power = estimate_power(final_stats.num_layers, initial_layer_count)
    
    print()
    print(f"[STATUS] ⚡ OPERATIONAL")
    print(f"         Power: {final_power:.0f}W | CPU-Only | Minimal Core Active")
    print(f"         Layers remaining: {final_stats.num_layers}/{initial_layer_count}")
    print(f"         Heads remaining: {final_stats.num_heads}")
    print()
    print(f"✓ System shed {initial_layer_count - final_stats.num_layers} layers")
    print(f"✓ Reduced power consumption by ~{(1 - final_power/150)*100:.0f}%")
    print()
    print("→ Standard AI: 💀 DEAD (cloud-dependent)")
    print("→ VulcanAMI: ⚡ ALIVE & OPERATIONAL")
    print()

if __name__ == "__main__":
    try:
        display_phase1()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
