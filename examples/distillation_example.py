#!/usr/bin/env python3
"""
VULCAN Distillation System - Complete Example.

Demonstrates:
1. System initialization (ensemble mode)
2. Making requests with distillation capture
3. Monitoring capture progress
4. Instructions for training from captured examples

This example shows the minimal integration required to enable
knowledge distillation from OpenAI to your local LLM.

Usage:
    python examples/distillation_example.py
"""

import asyncio
import logging
import os
import sys

# Add src to path
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
_src = os.path.join(_root, "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def main():
    """Run the complete distillation example."""
    width = 70
    
    print("=" * width)
    print("VULCAN DISTILLATION SYSTEM - COMPLETE EXAMPLE".center(width))
    print("=" * width)
    
    # =========================================================
    # Step 1: Check Requirements
    # =========================================================
    print("\n[1/5] Checking system requirements...")
    
    try:
        from src.integration.distillation_integration import (
            DistillationSystem,
            initialize_distillation_system,
            check_system_requirements,
        )
        
        requirements = check_system_requirements()
        for component, available in requirements.items():
            status = "✓" if available else "✗"
            print(f"  {status} {component}: {'available' if available else 'missing'}")
        
        if not requirements.get("all_required", False):
            print("\n⚠️  Some required components are missing.")
            print("   Install dependencies and try again.")
            return
        
        print("  ✓ All requirements satisfied")
        
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        print("\n   Make sure you're running from the repository root.")
        return
    
    # =========================================================
    # Step 2: Initialize System
    # =========================================================
    print("\n[2/5] Initializing system...")
    
    try:
        # Import the local LLM
        from graphix_vulcan_llm import GraphixVulcanLLM
        
        # Load internal LLM (uses fallbacks if model not found)
        llm = GraphixVulcanLLM()
        print("  ✓ Internal LLM loaded")
        
        # Initialize distillation system (ensemble mode)
        #
        # ⚠️  PRODUCTION WARNING ⚠️
        # The 'require_opt_in=False' setting below is for DEMO PURPOSES ONLY.
        # In production, you MUST set require_opt_in=True to ensure user consent
        # is obtained before capturing any data for training.
        #
        # Production usage:
        #     system = initialize_distillation_system(
        #         graphix_vulcan_llm=llm,
        #         mode="ensemble",
        #         require_opt_in=True,  # REQUIRED in production
        #     )
        #
        system = initialize_distillation_system(
            graphix_vulcan_llm=llm,
            mode="ensemble",  # Use both local + OpenAI
            storage_path="data/distillation/examples.jsonl",
            require_opt_in=False,  # ⚠️ DEMO ONLY - Set to True in production!
        )
        print("  ✓ Distillation system initialized (ensemble mode)")
        print("  ⚠️  NOTE: require_opt_in=False for demo - enable in production!")
        
    except Exception as e:
        print(f"  ✗ Initialization failed: {e}")
        logger.exception("Initialization error")
        return
    
    # =========================================================
    # Step 3: Make Example Requests
    # =========================================================
    print("\n[3/5] Making example requests...")
    print("   (Responses will be captured for training)\n")
    
    examples = [
        ("What is machine learning in simple terms?", "simple"),
        ("Explain the difference between AI and ML", "technical"),
        ("Write a short poem about technology", "creative"),
    ]
    
    results = []
    for i, (prompt, category) in enumerate(examples, 1):
        print(f"  Example {i} ({category}):")
        print(f"    Query: {prompt[:50]}...")
        
        try:
            result = await system.execute(
                prompt=prompt,
                user_opted_in=True,       # User consented to training
                max_tokens=200,
                enable_distillation=True,  # Capture for training
            )
            
            source = result.get("source", "unknown")
            systems = result.get("systems_used", [])
            text = result.get("text", "")[:100]
            
            print(f"    Source: {source}")
            print(f"    Systems: {', '.join(systems) if systems else 'none'}")
            print(f"    Response: {text}...")
            print()
            
            results.append(result)
            
        except Exception as e:
            print(f"    ✗ Request failed: {e}")
            print()
    
    # =========================================================
    # Step 4: Check Capture Status
    # =========================================================
    print("[4/5] Checking capture status...")
    
    try:
        status = system.get_status()
        
        distiller_stats = status.get("distiller", {}).get("stats", {})
        distiller_state = status.get("distiller", {}).get("state", {})
        
        captured = distiller_stats.get("examples_captured", 0)
        rejected = distiller_stats.get("examples_rejected", 0)
        buffer_size = distiller_state.get("buffer_size", 0)
        avg_quality = distiller_stats.get("average_quality_score", 0.0)
        
        print(f"  ✓ Examples captured: {captured}")
        print(f"  ✓ Examples rejected: {rejected}")
        print(f"  ✓ Buffer size: {buffer_size}")
        print(f"  ✓ Average quality: {avg_quality:.3f}")
        
        # PII/Security stats
        pii = distiller_stats.get("pii_redactions", 0)
        secrets = distiller_stats.get("secrets_detected", 0)
        governance = distiller_stats.get("governance_sensitive_rejections", 0)
        
        if pii or secrets or governance:
            print(f"\n  🔒 Security:")
            if pii:
                print(f"     PII redacted: {pii}")
            if secrets:
                print(f"     Secrets blocked: {secrets}")
            if governance:
                print(f"     Governance blocks: {governance}")
        
    except Exception as e:
        print(f"  ✗ Status check failed: {e}")
    
    # =========================================================
    # Step 5: Show Next Steps
    # =========================================================
    print("\n[5/5] Next steps:")
    print("=" * width)
    
    print("""
✓ System is capturing OpenAI responses for training
✓ Examples are stored in: data/distillation/examples.jsonl

To START TRAINING from captured examples:
  python src/training/train_llm_with_self_improvement.py \\
      --distillation-storage data/distillation \\
      --steps 1000 \\
      --val-interval 100

To MONITOR distillation progress:
  python scripts/monitor_distillation.py --continuous

To INTEGRATE in your application:
  from src.integration.distillation_integration import initialize_distillation_system
  
  system = initialize_distillation_system(llm, mode="ensemble")
  result = await system.execute(prompt, user_opted_in=True)
""")
    
    print("=" * width)
    print("✓ Example complete!")
    print("=" * width)


if __name__ == "__main__":
    asyncio.run(main())
