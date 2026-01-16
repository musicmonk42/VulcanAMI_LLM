#!/usr/bin/env python3
"""
Infrastructure and Documentation Verification

Verify that deleting chat.py didn't break any infrastructure or documentation:
- Docker files
- Helm charts
- Kubernetes manifests
- Makefile
- Markdown documentation
"""

from pathlib import Path
import re


def check_files_for_pattern(pattern, file_patterns, base_dirs, exclude_dirs=None):
    """
    Check multiple files for a pattern.
    
    Args:
        pattern: Regex pattern to search for
        file_patterns: List of file patterns (e.g., ["*.md", "*.yaml"])
        base_dirs: List of base directories to search
        exclude_dirs: List of directory names to exclude
    
    Returns:
        List of (file_path, line_number, line_content) tuples
    """
    if exclude_dirs is None:
        exclude_dirs = ["__pycache__", ".git", "node_modules", "archives"]
    
    matches = []
    regex = re.compile(pattern)
    
    for base_dir in base_dirs:
        if not base_dir.exists():
            continue
            
        for file_pattern in file_patterns:
            for file_path in base_dir.rglob(file_pattern):
                # Skip excluded directories
                if any(excl in str(file_path) for excl in exclude_dirs):
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        for line_num, line in enumerate(f, 1):
                            if regex.search(line):
                                matches.append((str(file_path), line_num, line.strip()))
                except Exception:
                    pass
    
    return matches


def test_docker_files():
    """Check Docker files for references to chat.py or chat_router."""
    print("Checking Docker files...")
    
    docker_patterns = ["Dockerfile*", "docker-compose*.yml", "docker-compose*.yaml", ".dockerignore"]
    base_dirs = [Path("."), Path("docker")]
    
    # Check for chat.py references (excluding unified_chat)
    matches = check_files_for_pattern(
        r'(?<!unified_)chat\.py|(?<!unified_)chat_router|"/llm/chat"',
        docker_patterns,
        base_dirs
    )
    
    if matches:
        print("✗ Found references in Docker files:")
        for file_path, line_num, line in matches:
            print(f"  {file_path}:{line_num}: {line}")
        raise AssertionError(f"Found {len(matches)} references in Docker files")
    
    print("✓ Docker files clean (no references to chat.py)")


def test_helm_files():
    """Check Helm charts for references to chat.py or chat_router."""
    print("Checking Helm charts...")
    
    helm_dir = Path("helm")
    if not helm_dir.exists():
        print("  (helm directory not found, skipping)")
        return
    
    helm_patterns = ["*.yaml", "*.yml", "*.tpl"]
    
    matches = check_files_for_pattern(
        r'(?<!unified_)chat\.py|(?<!unified_)chat_router|"/llm/chat"',
        helm_patterns,
        [helm_dir]
    )
    
    if matches:
        print("✗ Found references in Helm charts:")
        for file_path, line_num, line in matches:
            print(f"  {file_path}:{line_num}: {line}")
        raise AssertionError(f"Found {len(matches)} references in Helm charts")
    
    print("✓ Helm charts clean (no references to chat.py)")


def test_kubernetes_files():
    """Check Kubernetes manifests for references to chat.py or chat_router."""
    print("Checking Kubernetes manifests...")
    
    k8s_dirs = [Path("k8s"), Path("infra")]
    k8s_patterns = ["*.yaml", "*.yml"]
    
    matches = check_files_for_pattern(
        r'(?<!unified_)chat\.py|(?<!unified_)chat_router|"/llm/chat"',
        k8s_patterns,
        k8s_dirs
    )
    
    if matches:
        print("✗ Found references in Kubernetes manifests:")
        for file_path, line_num, line in matches:
            print(f"  {file_path}:{line_num}: {line}")
        raise AssertionError(f"Found {len(matches)} references in Kubernetes manifests")
    
    print("✓ Kubernetes manifests clean (no references to chat.py)")


def test_makefile():
    """Check Makefile for references to chat.py or chat_router."""
    print("Checking Makefile...")
    
    makefile = Path("Makefile")
    if not makefile.exists():
        print("  (Makefile not found, skipping)")
        return
    
    with open(makefile, 'r') as f:
        content = f.read()
    
    # Check for chat.py references (excluding unified_chat)
    problematic_patterns = [
        (r'(?<!unified_)chat\.py', 'chat.py'),
        (r'(?<!unified_)chat_router', 'chat_router'),
        (r'"/llm/chat"', '/llm/chat endpoint'),
    ]
    
    violations = []
    for pattern, name in problematic_patterns:
        if re.search(pattern, content):
            violations.append(name)
    
    if violations:
        print(f"✗ Found references in Makefile: {', '.join(violations)}")
        raise AssertionError(f"Makefile contains references to: {violations}")
    
    print("✓ Makefile clean (no references to chat.py)")


def test_markdown_docs():
    """Check Markdown documentation for problematic references."""
    print("Checking Markdown documentation...")
    
    md_patterns = ["*.md"]
    base_dirs = [Path("."), Path("docs")]
    
    # Check for chat.py or chat_router (but allow unified_chat references)
    matches = check_files_for_pattern(
        r'(?<!unified_)chat\.py(?!\w)|(?<!unified_)chat_router\b|"/llm/chat"',
        md_patterns,
        base_dirs
    )
    
    if matches:
        print("✗ Found references in Markdown files:")
        for file_path, line_num, line in matches:
            print(f"  {file_path}:{line_num}: {line}")
        raise AssertionError(f"Found {len(matches)} references in Markdown files")
    
    print("✓ Markdown documentation clean (no references to chat.py)")


def test_config_files():
    """Check configuration files for references."""
    print("Checking configuration files...")
    
    config_patterns = ["*.toml", "*.ini", "*.cfg", "*.json"]
    base_dirs = [Path("."), Path("config"), Path("configs")]
    
    matches = check_files_for_pattern(
        r'(?<!unified_)chat\.py|(?<!unified_)chat_router|"/llm/chat"',
        config_patterns,
        base_dirs
    )
    
    if matches:
        print("✗ Found references in config files:")
        for file_path, line_num, line in matches:
            print(f"  {file_path}:{line_num}: {line}")
        raise AssertionError(f"Found {len(matches)} references in config files")
    
    print("✓ Configuration files clean (no references to chat.py)")


def test_shell_scripts():
    """Check shell scripts for references."""
    print("Checking shell scripts...")
    
    script_patterns = ["*.sh"]
    base_dirs = [Path("."), Path("scripts"), Path("bin")]
    
    matches = check_files_for_pattern(
        r'(?<!unified_)chat\.py|(?<!unified_)chat_router|"/llm/chat"',
        script_patterns,
        base_dirs
    )
    
    if matches:
        print("✗ Found references in shell scripts:")
        for file_path, line_num, line in matches:
            print(f"  {file_path}:{line_num}: {line}")
        raise AssertionError(f"Found {len(matches)} references in shell scripts")
    
    print("✓ Shell scripts clean (no references to chat.py)")


if __name__ == "__main__":
    """Run all infrastructure and documentation checks."""
    print("=" * 70)
    print("INFRASTRUCTURE & DOCUMENTATION VERIFICATION")
    print("Checking Docker, Helm, K8s, Makefile, and Markdown docs")
    print("=" * 70)
    print()
    
    tests = [
        test_docker_files,
        test_helm_files,
        test_kubernetes_files,
        test_makefile,
        test_markdown_docs,
        test_config_files,
        test_shell_scripts,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
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
    print("INFRASTRUCTURE VERIFICATION RESULTS")
    print("=" * 70)
    print(f"Total:  {len(tests)} checks")
    print(f"Passed: {passed} checks")
    print(f"Failed: {failed} checks")
    
    if failed == 0:
        print("\n✓✓✓ All infrastructure and documentation files verified clean!")
        exit(0)
    else:
        print(f"\n✗✗✗ {failed}/{len(tests)} checks failed")
        exit(1)
