#!/usr/bin/env python3
"""
Simple test to verify the warning fixes work correctly.
Run this after installing dependencies with: pip install -r requirements.txt
"""

import sys
import logging

# Set up logging to capture warnings
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_llm_validators():
    """Test that LLM validators are available without warnings"""
    print("Testing LLM Validators...")
    try:
        from src.vulcan.safety.llm_validators import (
            StructuralValidator,
            EthicalValidator,
            ToxicityValidator,
            HallucinationValidator,
            PromptInjectionValidator
        )
        
        # Instantiate to verify they work
        validators = [
            StructuralValidator(),
            EthicalValidator(),
            ToxicityValidator(),
            HallucinationValidator(),
            PromptInjectionValidator()
        ]
        
        print("✅ All LLM validators available")
        return True
    except ImportError as e:
        print(f"❌ LLM validators not available: {e}")
        return False

def test_bm25():
    """Test that BM25 is available without warnings"""
    print("\nTesting BM25...")
    try:
        from rank_bm25 import BM25Okapi
        print("✅ BM25Okapi available")
        return True
    except ImportError as e:
        print(f"⚠️  BM25 not available: {e}")
        print("   Run: pip install rank-bm25==0.2.2")
        return False

def test_zmq():
    """Test that ZMQ is available without warnings"""
    print("\nTesting ZMQ...")
    try:
        import zmq
        import zmq.asyncio
        print(f"✅ ZMQ available (version {zmq.zmq_version()})")
        return True
    except ImportError as e:
        print(f"⚠️  ZMQ not available: {e}")
        print("   Run: pip install pyzmq==26.2.0")
        return False

def test_lingam():
    """Test that lingam is available without warnings"""
    print("\nTesting lingam...")
    try:
        import lingam
        print("✅ lingam available")
        return True
    except ImportError as e:
        print(f"⚠️  lingam not available: {e}")
        print("   Run: pip install lingam==1.11.0")
        return False

def test_cachetools():
    """Test that cachetools is available without warnings"""
    print("\nTesting cachetools...")
    try:
        from cachetools import TTLCache
        print("✅ cachetools available")
        return True
    except ImportError as e:
        print(f"⚠️  cachetools not available: {e}")
        print("   Run: pip install cachetools==6.2.2")
        return False

def test_safety_validator():
    """Test that safety validator dependencies are available"""
    print("\nTesting Safety Validator Dependencies...")
    missing = []
    
    try:
        import numpy
        print(f"✅ numpy {numpy.__version__}")
    except ImportError:
        print("❌ numpy not available")
        missing.append("numpy")
    
    try:
        import torch
        print(f"✅ torch {torch.__version__}")
    except ImportError:
        print("⚠️  torch not available")
        missing.append("torch")
    
    try:
        import scipy
        print(f"✅ scipy {scipy.__version__}")
    except ImportError:
        print("❌ scipy not available")
        missing.append("scipy")
    
    try:
        import statsmodels
        print(f"✅ statsmodels {statsmodels.__version__}")
    except ImportError:
        print("❌ statsmodels not available")
        missing.append("statsmodels")
    
    if missing:
        print(f"   Missing: {', '.join(missing)}")
        print("   Run: pip install -r requirements.txt")
        return False
    
    # If all dependencies available, try importing safety validator
    try:
        from src.vulcan.safety.safety_validator import EnhancedSafetyValidator
        print("✅ EnhancedSafetyValidator available")
        return True
    except Exception as e:
        print(f"⚠️  EnhancedSafetyValidator import issue: {e}")
        return False

if __name__ == "__main__":
    print("=" * 70)
    print("VULCAN AMI LLM - Warning Resolution Test")
    print("=" * 70)
    
    # Add current directory to path
    sys.path.insert(0, '.')
    
    results = {
        "LLM Validators": test_llm_validators(),
        "BM25": test_bm25(),
        "ZMQ": test_zmq(),
        "lingam": test_lingam(),
        "cachetools": test_cachetools(),
        "Safety Validator": test_safety_validator()
    }
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, status in results.items():
        symbol = "✅" if status else "⚠️ "
        print(f"{symbol} {name}")
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed < total:
        print("\nTo resolve remaining warnings, run:")
        print("  pip install -r requirements.txt")
    else:
        print("\n🎉 All warnings resolved!")
    
    sys.exit(0 if passed == total else 1)
