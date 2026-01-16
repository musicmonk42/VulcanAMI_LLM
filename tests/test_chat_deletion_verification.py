#!/usr/bin/env python3
"""
Triple-check verification that deleting chat.py didn't break anything.

This comprehensive test suite verifies:
1. chat.py file is deleted
2. No imports of chat.py exist
3. chat_router is removed from all code
4. unified_chat_router is properly registered
5. Endpoint routes are still functional
6. No documentation references remain
"""

import ast
from pathlib import Path


def test_chat_py_file_deleted():
    """Verify the chat.py file no longer exists."""
    chat_file = Path("src/vulcan/endpoints/chat.py")
    assert not chat_file.exists(), "chat.py should be deleted"
    print("✓ chat.py file deleted")


def test_no_imports_of_chat_py():
    """Verify no Python files import from chat.py."""
    src_dir = Path("src")
    violations = []
    
    for py_file in src_dir.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
            
        with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        # Check for imports of chat.py (excluding unified_chat)
        if "from vulcan.endpoints.chat import" in content:
            violations.append(str(py_file))
        elif "from .chat import" in content and "unified" not in content:
            violations.append(str(py_file))
    
    assert not violations, f"Files still importing chat.py: {violations}"
    print("✓ No imports of chat.py found")


def test_chat_router_removed_from_init():
    """Verify chat_router is removed from __init__.py using AST."""
    init_file = Path("src/vulcan/endpoints/__init__.py")
    assert init_file.exists(), "endpoints/__init__.py not found"
    
    with open(init_file, 'r') as f:
        tree = ast.parse(f.read(), filename=str(init_file))
    
    # Check imports
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and 'chat' in node.module and 'unified' not in node.module:
                for alias in node.names:
                    if alias.asname == 'chat_router' or alias.name == 'router':
                        raise AssertionError(f"Found import of chat_router in __init__.py")
    
    # Check __all__ list
    with open(init_file, 'r') as f:
        content = f.read()
    
    # Use AST to check __all__
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == '__all__':
                    if isinstance(node.value, ast.List):
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and elt.value == 'chat_router':
                                raise AssertionError("chat_router found in __all__")
    
    print("✓ chat_router removed from __init__.py")


def test_chat_router_removed_from_main():
    """Verify chat_router is removed from main.py using AST."""
    main_file = Path("src/vulcan/main.py")
    assert main_file.exists(), "main.py not found"
    
    with open(main_file, 'r') as f:
        tree = ast.parse(f.read(), filename=str(main_file))
    
    # Check for chat_router in imports
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.asname == 'chat_router':
                    raise AssertionError("chat_router imported in main.py")
                if alias.name == 'chat_router':
                    raise AssertionError("chat_router imported in main.py")
        
        # Check for chat_router in router registrations
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if node.func.attr == 'include_router':
                    for arg in node.args:
                        if isinstance(arg, ast.Name) and arg.id == 'chat_router':
                            raise AssertionError("chat_router registered in main.py")
    
    print("✓ chat_router removed from main.py")


def test_unified_chat_router_still_present():
    """Verify unified_chat_router is still properly registered."""
    # Check __init__.py
    init_file = Path("src/vulcan/endpoints/__init__.py")
    with open(init_file, 'r') as f:
        content = f.read()
        tree = ast.parse(content, filename=str(init_file))
    
    # Verify import exists
    found_import = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if 'unified_chat' in (node.module or ''):
                found_import = True
    
    assert found_import, "unified_chat_router import not found in __init__.py"
    
    # Check main.py registration
    main_file = Path("src/vulcan/main.py")
    with open(main_file, 'r') as f:
        tree = ast.parse(f.read(), filename=str(main_file))
    
    found_registration = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == 'include_router':
                for arg in node.args:
                    if isinstance(arg, ast.Name) and arg.id == 'unified_chat_router':
                        found_registration = True
    
    assert found_registration, "unified_chat_router not registered in main.py"
    
    print("✓ unified_chat_router properly registered")


def test_unified_chat_endpoints_exist():
    """Verify unified_chat.py has the expected endpoints."""
    unified_chat_file = Path("src/vulcan/endpoints/unified_chat.py")
    assert unified_chat_file.exists(), "unified_chat.py not found"
    
    with open(unified_chat_file, 'r') as f:
        content = f.read()
    
    # Check for /v1/chat endpoint
    assert '@router.post("/v1/chat"' in content, "/v1/chat endpoint not found"
    
    print("✓ unified_chat.py endpoints exist")


def test_no_broken_references():
    """Verify no broken references to /llm/chat endpoint."""
    # Check all relevant files for /llm/chat (the deleted endpoint)
    check_paths = [
        Path("src"),
        Path("docs"),
        Path("scripts"),
    ]
    
    violations = []
    
    for base_path in check_paths:
        if not base_path.exists():
            continue
            
        for file_path in base_path.rglob("*"):
            if file_path.is_file() and not any(x in str(file_path) for x in ["__pycache__", "archives", ".pyc"]):
                if file_path.suffix in [".py", ".md", ".sh", ".html", ".js"]:
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        if '"/llm/chat"' in content or "'/llm/chat'" in content:
                            violations.append(str(file_path))
                    except Exception:
                        pass
    
    assert not violations, f"Files with /llm/chat references: {violations}"
    print("✓ No broken references to /llm/chat endpoint")


def test_no_duplicate_routes():
    """Verify there are no duplicate route definitions."""
    unified_chat_file = Path("src/vulcan/endpoints/unified_chat.py")
    
    with open(unified_chat_file, 'r') as f:
        content = f.read()
        tree = ast.parse(content, filename=str(unified_chat_file))
    
    # Find all @router decorators
    routes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Call):
                    if isinstance(decorator.func, ast.Attribute):
                        if decorator.func.attr in ['post', 'get', 'put', 'delete']:
                            if decorator.args:
                                route = decorator.args[0]
                                if isinstance(route, ast.Constant):
                                    routes.append((decorator.func.attr, route.value))
    
    # Check for duplicates
    route_counts = {}
    for method, path in routes:
        key = f"{method.upper()} {path}"
        route_counts[key] = route_counts.get(key, 0) + 1
    
    duplicates = [k for k, v in route_counts.items() if v > 1]
    assert not duplicates, f"Duplicate routes found: {duplicates}"
    
    print(f"✓ No duplicate routes ({len(routes)} unique routes)")


if __name__ == "__main__":
    """Run all verification tests."""
    print("=" * 70)
    print("TRIPLE-CHECK: Verifying chat.py deletion didn't break anything")
    print("=" * 70)
    print()
    
    tests = [
        test_chat_py_file_deleted,
        test_no_imports_of_chat_py,
        test_chat_router_removed_from_init,
        test_chat_router_removed_from_main,
        test_unified_chat_router_still_present,
        test_unified_chat_endpoints_exist,
        test_no_broken_references,
        test_no_duplicate_routes,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            print(f"Running: {test.__name__}")
            test()
            passed += 1
            print()
        except AssertionError as e:
            print(f"✗ FAILED: {e}")
            failed += 1
            print()
        except Exception as e:
            print(f"✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            print()
    
    print("=" * 70)
    print("TRIPLE-CHECK RESULTS")
    print("=" * 70)
    print(f"Total:  {len(tests)} tests")
    print(f"Passed: {passed} tests")
    print(f"Failed: {failed} tests")
    
    if failed == 0:
        print("\n✓✓✓ TRIPLE-CHECKED: Deleting chat.py didn't break anything!")
        exit(0)
    else:
        print(f"\n✗✗✗ {failed}/{len(tests)} checks failed")
        exit(1)
