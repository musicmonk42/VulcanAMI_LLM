#!/usr/bin/env python3
"""
VulcanAMI Deployment Verification Script
=========================================

This script verifies that all services are running correctly after deployment.
Use it to diagnose issues with Railway or other deployments.

Usage:
    python scripts/verify_deployment.py [--base-url URL]
    
Examples:
    python scripts/verify_deployment.py
    python scripts/verify_deployment.py --base-url https://your-app.railway.app
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

# Try to import requests, provide fallback
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("⚠️  requests package not installed, using urllib fallback")
    import urllib.request
    import urllib.error
    import json


def check_endpoint(name: str, url: str, timeout: int = 10) -> Tuple[bool, str]:
    """Check if an endpoint is responding."""
    try:
        if REQUESTS_AVAILABLE:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                return True, f"OK (HTTP {response.status_code})"
            else:
                return False, f"HTTP {response.status_code}"
        else:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=timeout) as response:
                if response.status == 200:
                    return True, f"OK (HTTP {response.status})"
                else:
                    return False, f"HTTP {response.status}"
    except Exception as e:
        return False, str(e)


def verify_deployment(base_url: str) -> bool:
    """
    Verify all services are running.
    
    Returns:
        bool: True if all critical services pass, False otherwise
    """
    print("=" * 60)
    print("VulcanAMI Deployment Verification")
    print("=" * 60)
    print(f"Base URL: {base_url}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Define service checks
    # Format: (name, path, is_critical)
    CHECKS: List[Tuple[str, str, bool]] = [
        # Critical endpoints
        ("Health Check", "/health", True),
        ("VULCAN Health", "/vulcan/health", True),
        ("VULCAN Chat", "/vulcan/v1/chat", False),  # POST endpoint, may not respond to GET
        ("VULCAN LLM Status", "/vulcan/v1/llm/status", True),
        
        # Core services
        ("Registry Health", "/registry/health", False),
        ("Arena Health", "/arena/health", False),
        ("API Gateway Health", "/api-gateway/health", False),
        ("DQS Health", "/dqs/health", False),
        ("PII Health", "/pii/health", False),
        
        # Static files
        ("Chat Interface Root", "/", False),
    ]
    
    results: Dict[str, Tuple[bool, str]] = {}
    critical_failures = []
    
    print("\n📋 Checking Endpoints...\n")
    
    for name, path, is_critical in CHECKS:
        url = f"{base_url}{path}"
        status, message = check_endpoint(name, url)
        results[name] = (status, message)
        
        if status:
            print(f"  ✅ {name}: {message}")
        else:
            marker = "❌" if is_critical else "⚠️"
            print(f"  {marker} {name}: {message}")
            if is_critical:
                critical_failures.append(name)
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    total = len(CHECKS)
    passed = sum(1 for status, _ in results.values() if status)
    failed = total - passed
    
    print(f"Total Checks: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if critical_failures:
        print(f"\n❌ CRITICAL FAILURES: {', '.join(critical_failures)}")
        print("\nThese critical services must be running for the platform to function.")
        return False
    elif failed > 0:
        print(f"\n⚠️  Some non-critical services are not responding.")
        print("The platform may have reduced functionality.")
        return True
    else:
        print(f"\n✅ All services are running correctly!")
        return True


def check_environment() -> None:
    """Check and report environment variable status."""
    print("\n" + "=" * 60)
    print("Environment Variables")
    print("=" * 60)
    
    REQUIRED_VARS = ["OPENAI_API_KEY", "JWT_SECRET_KEY"]
    OPTIONAL_VARS = [
        "ANTHROPIC_API_KEY",
        "GRAPHIX_API_KEY",
        "VULCAN_LLM_API_KEY",
        "DATABASE_URL",
        "REDIS_URL",
        "PORT",
    ]
    
    def truncate_secret(value: str) -> str:
        """Truncate a secret value for safe display."""
        if len(value) > 12:
            return f"{value[:4]}...{value[-4:]}"
        return "(set)"
    
    print("\nRequired Variables:")
    for var in REQUIRED_VARS:
        value = os.getenv(var)
        if value:
            print(f"  ✅ {var}: {truncate_secret(value)}")
        else:
            print(f"  ❌ {var}: NOT SET")
    
    print("\nOptional Variables:")
    for var in OPTIONAL_VARS:
        value = os.getenv(var)
        if value:
            print(f"  ✓ {var}: {truncate_secret(value)}")
        else:
            print(f"  - {var}: not set")


def main():
    parser = argparse.ArgumentParser(
        description="Verify VulcanAMI deployment status"
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("BASE_URL", "http://localhost:8000"),
        help="Base URL of the deployment (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Also check environment variables"
    )
    
    args = parser.parse_args()
    
    # Remove trailing slash if present
    base_url = args.base_url.rstrip("/")
    
    # Check environment if requested
    if args.check_env:
        check_environment()
    
    # Verify deployment
    success = verify_deployment(base_url)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
