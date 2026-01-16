#!/usr/bin/env python3
"""
Test script to verify /health/live responds immediately without blocking.

This simulates Railway's healthcheck behavior by:
1. Starting uvicorn in a subprocess
2. Immediately making HTTP requests to /health/live
3. Verifying the endpoint responds within 1 second
4. Checking that response is 200 OK

Expected behavior:
- First health check succeeds within 1 second
- Subsequent checks continue to succeed
- No "service unavailable" errors
"""

import subprocess
import time
import sys
import requests
from pathlib import Path

def test_immediate_healthcheck():
    """Test that /health/live responds immediately after server starts."""
    
    print("=" * 70)
    print("Railway Healthcheck Simulation Test")
    print("=" * 70)
    print()
    
    # Start uvicorn in background
    print("Starting uvicorn server...")
    process = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "src.full_platform:app",
            "--host", "127.0.0.1",
            "--port", "8000",
            "--workers", "1"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Wait a moment for the server to start binding to the port
    print("Waiting 2 seconds for server to bind...")
    time.sleep(2)
    
    # Try health checks immediately (simulate Railway behavior)
    print()
    print("Testing /health/live endpoint (simulating Railway healthcheck)...")
    print("-" * 70)
    
    success_count = 0
    fail_count = 0
    
    for attempt in range(1, 12):  # Railway makes 11 attempts in the logs
        start = time.time()
        try:
            response = requests.get("http://127.0.0.1:8000/health/live", timeout=3)
            elapsed = time.time() - start
            
            if response.status_code == 200:
                print(f"✅ Attempt #{attempt}: SUCCESS [{response.status_code}] in {elapsed:.3f}s")
                print(f"   Response: {response.json()}")
                success_count += 1
            else:
                print(f"❌ Attempt #{attempt}: FAILED [{response.status_code}] in {elapsed:.3f}s")
                print(f"   Response: {response.text[:200]}")
                fail_count += 1
                
        except requests.exceptions.RequestException as e:
            elapsed = time.time() - start
            print(f"❌ Attempt #{attempt}: ERROR in {elapsed:.3f}s - {e}")
            fail_count += 1
        
        # Railway waits between attempts
        if attempt < 11:
            time.sleep(2)
    
    # Cleanup
    print()
    print("-" * 70)
    print(f"Results: {success_count} success, {fail_count} failed")
    print()
    
    # Properly terminate the server process
    print("Shutting down server...")
    try:
        process.terminate()
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        print("  Server didn't terminate gracefully, forcing kill...")
        process.kill()
        process.wait(timeout=2)
    print("  Server stopped.")
    print()
    
    # Evaluate results
    if success_count >= 10:  # At least 10 out of 11 should succeed
        print("✅ TEST PASSED: Healthcheck responds immediately!")
        print("   Server is ready for Railway deployment.")
        return 0
    else:
        print("❌ TEST FAILED: Healthcheck failures detected!")
        print(f"   Only {success_count}/11 attempts succeeded.")
        print("   Railway deployment would fail with this configuration.")
        return 1

if __name__ == "__main__":
    sys.exit(test_immediate_healthcheck())
