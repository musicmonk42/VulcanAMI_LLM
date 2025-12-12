#!/usr/bin/env python3
"""
Verification script for vulcan_unified.html
Checks that the HTML file has all necessary components for platform interaction
"""

import re
import sys
from pathlib import Path

def verify_vulcan_unified():
    """Verify vulcan_unified.html has all required components"""
    
    html_path = Path(__file__).parent / "vulcan_unified.html"
    
    if not html_path.exists():
        print("❌ ERROR: vulcan_unified.html not found!")
        return False
    
    print(f"✓ Found vulcan_unified.html ({html_path.stat().st_size} bytes)")
    
    with open(html_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check 1: Basic HTML structure
    checks = {
        "HTML doctype": content.startswith("<!DOCTYPE html>"),
        "HTML tag": "<html" in content,
        "Head section": "<head>" in content,
        "Body section": "<body>" in content,
        "Closing tags": "</html>" in content,
    }
    
    # Check 2: Required tabs
    required_tabs = [
        'dashboard', 'agents', 'vulcan', 'worldmodel', 'arena', 
        'registry', 'safety', 'llm', 'auth', 'tools', 'logs'
    ]
    
    for tab in required_tabs:
        checks[f"Tab: {tab}"] = f"switchTab('{tab}')" in content
    
    # Check 3: Critical functions
    critical_functions = [
        'connectPlatform', 'disconnectPlatform', 'getAuthHeaders',
        'callAPI', 'loadAgentPool', 'invokeVulcan', 'runArenaTask',
        'getNonce', 'registryLogin', 'getToken', 'refreshHealth'
    ]
    
    for func in critical_functions:
        checks[f"Function: {func}"] = f"function {func}(" in content or f"async function {func}(" in content
    
    # Check 4: API endpoints
    critical_endpoints = [
        '/api/status', '/health', '/auth/token',
        '/vulcan/health', '/vulcan/orchestrator/agents/status',
        '/arena/', '/registry/auth/', '/vulcan/llm/'
    ]
    
    for endpoint in critical_endpoints:
        checks[f"Endpoint: {endpoint}"] = endpoint in content
    
    # Check 5: UI components
    ui_components = [
        'connection-bar', 'tabs', 'panel', 'btn', 'form-group',
        'logs-container', 'json-display'
    ]
    
    for component in ui_components:
        checks[f"UI: {component}"] = component in content
    
    # Print results
    print("\n" + "="*60)
    print("VERIFICATION RESULTS")
    print("="*60)
    
    passed = 0
    failed = 0
    
    for check_name, result in checks.items():
        status = "✓" if result else "✗"
        print(f"{status} {check_name}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print("="*60)
    print(f"Total: {passed} passed, {failed} failed out of {len(checks)} checks")
    print("="*60)
    
    # Summary
    if failed == 0:
        print("\n✅ SUCCESS: vulcan_unified.html is complete and ready to use!")
        print("\nTo use the interface:")
        print("1. Start the platform: python src/full_platform.py")
        print("2. Open vulcan_unified.html in your browser")
        print("3. Connect to the platform and explore!")
        return True
    else:
        print(f"\n⚠️  WARNING: {failed} checks failed. Review the file.")
        return False

if __name__ == "__main__":
    success = verify_vulcan_unified()
    sys.exit(0 if success else 1)
