#!/usr/bin/env python3
"""
Static validation tests for new platform integration files
Tests syntax, structure, and documentation without requiring runtime dependencies
"""

import ast
import os
import sys
from pathlib import Path

def test_python_syntax(filepath):
    """Test that Python file has valid syntax"""
    print(f"Testing syntax: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code)
        print(f"  ✓ Valid Python syntax")
        return True
    except SyntaxError as e:
        print(f"  ✗ Syntax error at line {e.lineno}: {e.msg}")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_file_structure(filepath):
    """Test that Python file has proper structure"""
    print(f"Testing structure: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        
        tree = ast.parse(code)
        
        # Check for docstring
        has_docstring = (isinstance(tree.body[0], ast.Expr) and 
                        isinstance(tree.body[0].value, ast.Constant) and
                        isinstance(tree.body[0].value.value, str))
        
        if has_docstring:
            print(f"  ✓ Has module docstring")
        else:
            print(f"  ⚠ Missing module docstring")
        
        # Count functions and classes
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        
        print(f"  ✓ Functions: {len(functions)}")
        print(f"  ✓ Classes: {len(classes)}")
        
        return True
    except Exception as e:
        print(f"  ✗ Error analyzing structure: {e}")
        return False

def test_markdown_file(filepath):
    """Test that markdown file is well-formed"""
    print(f"Testing markdown: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for basic markdown elements
        has_title = content.startswith('#')
        line_count = len(content.split('\n'))
        word_count = len(content.split())
        
        print(f"  ✓ Has title heading: {has_title}")
        print(f"  ✓ Lines: {line_count}")
        print(f"  ✓ Words: {word_count}")
        
        # Check for common sections
        sections = {
            'Overview': '## Overview' in content or '## OVERVIEW' in content,
            'Configuration': '## Configuration' in content or 'configuration' in content.lower(),
            'Examples': '## Example' in content or 'example' in content.lower(),
        }
        
        for section, present in sections.items():
            if present:
                print(f"  ✓ Has {section} section")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_full_platform_modifications():
    """Test that full_platform.py has expected modifications"""
    print("Testing full_platform.py modifications...")
    filepath = Path("src/full_platform.py")
    
    if not filepath.exists():
        print(f"  ✗ File not found: {filepath}")
        return False
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for new configuration options
        checks = {
            "api_gateway_mount": "api_gateway_mount" in content,
            "dqs_mount": "dqs_mount" in content,
            "pii_mount": "pii_mount" in content,
            "api_server_port": "api_server_port" in content,
            "registry_grpc_port": "registry_grpc_port" in content,
            "listener_port": "listener_port" in content,
            "enable_api_gateway": "enable_api_gateway" in content,
            "enable_dqs_service": "enable_dqs_service" in content,
            "enable_pii_service": "enable_pii_service" in content,
            "background_processes": "background_processes" in content,
            "API Gateway import": "API Gateway" in content and "import_service_async" in content,
            "DQS Service import": "DQS Service" in content,
            "PII Service import": "PII Service" in content,
            "API Server subprocess": "api_server_proc" in content,
            "Registry gRPC subprocess": "registry_grpc_proc" in content,
            "Listener subprocess": "listener_proc" in content,
            "Process cleanup": "terminate" in content and "background_processes" in content,
        }
        
        passed = 0
        failed = 0
        for check_name, result in checks.items():
            if result:
                print(f"  ✓ {check_name}")
                passed += 1
            else:
                print(f"  ✗ Missing: {check_name}")
                failed += 1
        
        print(f"\n  Summary: {passed} passed, {failed} failed")
        return failed == 0
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def main():
    """Run all validation tests"""
    print("=" * 70)
    print("VulcanAMI Platform Integration - Validation Tests")
    print("=" * 70)
    print()
    
    results = []
    
    # Test new Python files
    python_files = [
        "test_platform_startup.py",
        "test_new_files.py",
    ]
    
    print("Testing Python Files")
    print("-" * 70)
    for filepath in python_files:
        if Path(filepath).exists():
            syntax_ok = test_python_syntax(filepath)
            structure_ok = test_file_structure(filepath)
            results.append(("Python: " + filepath, syntax_ok and structure_ok))
            print()
        else:
            print(f"Skipping {filepath} (not found)")
            print()
    
    # Test new markdown files
    markdown_files = [
        "PLATFORM_SERVICES_INVENTORY.md",
        "UNIFIED_STARTUP_GUIDE.md",
    ]
    
    print("Testing Markdown Documentation")
    print("-" * 70)
    for filepath in markdown_files:
        if Path(filepath).exists():
            md_ok = test_markdown_file(filepath)
            results.append(("Markdown: " + filepath, md_ok))
            print()
        else:
            print(f"Skipping {filepath} (not found)")
            print()
    
    # Test modifications to full_platform.py
    print("Testing Platform Modifications")
    print("-" * 70)
    platform_ok = test_full_platform_modifications()
    results.append(("full_platform.py modifications", platform_ok))
    print()
    
    # Summary
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print()
    print(f"Results: {passed}/{total} tests passed ({100*passed//total}%)")
    print()
    
    if passed == total:
        print("✅ All validation tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
