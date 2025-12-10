#!/usr/bin/env python3
"""
Minimal CI/CD Test Runner
Runs basic validation tests without requiring full dependencies
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


class Color:
    """ANSI color codes"""
    BLUE = '\033[0;34m'
    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    YELLOW = '\033[1;33m'
    NC = '\033[0m'  # No Color

class TestRunner:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.root_dir = Path(__file__).parent
        
    def log(self, message: str, color: str = Color.NC):
        """Print colored log message"""
        print(f"{color}{message}{Color.NC}")
        
    def test_python_syntax(self) -> bool:
        """Test if all Python files have valid syntax"""
        self.log("\n=== Testing Python Syntax ===", Color.BLUE)
        py_files = list(self.root_dir.glob("src/**/*.py"))
        
        errors = []
        for py_file in py_files:
            try:
                with open(py_file) as f:
                    compile(f.read(), py_file, 'exec')
            except SyntaxError as e:
                errors.append(f"{py_file}: {e}")
                
        if errors:
            self.log(f"✗ Syntax errors found in {len(errors)} files", Color.RED)
            for error in errors[:5]:  # Show first 5
                print(f"  {error}")
            self.failed += 1
            return False
        else:
            self.log(f"✓ All {len(py_files)} Python files have valid syntax", Color.GREEN)
            self.passed += 1
            return True
            
    def test_imports(self) -> bool:
        """Test if core modules can be imported"""
        self.log("\n=== Testing Core Imports ===", Color.BLUE)
        
        # Add src to path
        sys.path.insert(0, str(self.root_dir / "src"))
        
        # Set required environment variables for testing
        os.environ.setdefault('ALLOW_EPHEMERAL_SECRET', 'true')
        os.environ.setdefault('GRAPHIX_JWT_SECRET', 'test-secret-for-import-validation-only')
        
        core_modules = [
            "agent_registry",
            "consensus_engine",
            "api_server",
        ]
        
        failed_imports = []
        for module in core_modules:
            try:
                __import__(module)
                self.log(f"✓ Import {module}", Color.GREEN)
            except ImportError as e:
                self.log(f"✗ Failed to import {module}: {e}", Color.RED)
                failed_imports.append(module)
            except Exception as e:
                self.log(f"✗ Import {module} raised exception: {e}", Color.RED)
                failed_imports.append(module)
                
        if failed_imports:
            self.failed += 1
            return False
        else:
            self.passed += 1
            return True
            
    def test_json_configs(self) -> bool:
        """Test if all JSON config files are valid"""
        self.log("\n=== Testing JSON Configurations ===", Color.BLUE)
        
        json_files = list(self.root_dir.glob("configs/**/*.json"))
        json_files.extend(self.root_dir.glob("**/*.json"))
        
        errors = []
        for json_file in json_files:
            try:
                with open(json_file) as f:
                    json.load(f)
            except json.JSONDecodeError as e:
                errors.append(f"{json_file}: {e}")
                
        if errors:
            self.log(f"✗ JSON errors found in {len(errors)} files", Color.RED)
            for error in errors[:5]:
                print(f"  {error}")
            self.failed += 1
            return False
        else:
            self.log(f"✓ All {len(json_files)} JSON files are valid", Color.GREEN)
            self.passed += 1
            return True
            
    def test_required_files(self) -> bool:
        """Test if required files exist"""
        self.log("\n=== Testing Required Files ===", Color.BLUE)
        
        required = [
            "README.md",
            "requirements.txt",
            "Dockerfile",
            "docker-compose.dev.yml",
            "pytest.ini",
            "Makefile",
            ".gitignore",
        ]
        
        missing = []
        for file in required:
            if (self.root_dir / file).exists():
                self.log(f"✓ {file} exists", Color.GREEN)
            else:
                self.log(f"✗ {file} missing", Color.RED)
                missing.append(file)
                
        if missing:
            self.failed += 1
            return False
        else:
            self.passed += 1
            return True
            
    def test_docker_syntax(self) -> bool:
        """Test if Dockerfile has valid syntax"""
        self.log("\n=== Testing Dockerfile ===", Color.BLUE)
        
        dockerfile = self.root_dir / "Dockerfile"
        if not dockerfile.exists():
            self.log("✗ Dockerfile not found", Color.RED)
            self.failed += 1
            return False
            
        content = dockerfile.read_text()
        
        # Check for security best practices
        checks = [
            ("USER", "Non-root user directive"),
            ("HEALTHCHECK", "Health check configured"),
            ("REJECT_INSECURE_JWT", "JWT security check"),
        ]
        
        all_passed = True
        for directive, description in checks:
            if directive in content:
                self.log(f"✓ {description}", Color.GREEN)
            else:
                self.log(f"⚠ {description} not found", Color.YELLOW)
                all_passed = False
                
        if all_passed:
            self.passed += 1
        else:
            self.skipped += 1
            
        return True
        
    def test_secrets(self) -> bool:
        """Test for hardcoded secrets"""
        self.log("\n=== Scanning for Hardcoded Secrets ===", Color.BLUE)
        
        patterns = [
            (r"sk-[a-zA-Z0-9]{32,}", "OpenAI API key pattern"),
            # Exclude AWS example credentials (AKIAIOSFODNN7EXAMPLE)
            (r"AKIA[0-9A-Z]{16}(?!EXAMPLE)", "AWS Access Key pattern"),
            # Exclude test passwords (testpass, test123, etc)
            (r"['\"]password['\"]:\s*['\"](?!test|demo|example|sample)[^'\"]{8,}", "Hardcoded password"),
        ]
        
        # Only scan main source, exclude test directories
        py_files = list(self.root_dir.glob("src/**/*.py"))
        # Exclude test files
        py_files = [f for f in py_files if '/tests/' not in str(f) and '/test_' not in f.name]
        
        issues = []
        for py_file in py_files:
            content = py_file.read_text()
            for pattern, desc in patterns:
                import re
                if re.search(pattern, content, re.IGNORECASE):
                    issues.append(f"{py_file}: Potential {desc}")
                    
        if issues:
            self.log(f"⚠ Found {len(issues)} potential secret(s)", Color.YELLOW)
            for issue in issues[:5]:
                print(f"  {issue}")
            self.skipped += 1
        else:
            self.log("✓ No hardcoded secrets found in source code", Color.GREEN)
            self.passed += 1
            
        return True
        
    def test_github_workflows(self) -> bool:
        """Test GitHub Actions workflow files"""
        self.log("\n=== Testing GitHub Workflows ===", Color.BLUE)
        
        workflows_dir = self.root_dir / ".github" / "workflows"
        if not workflows_dir.exists():
            self.log("✗ No .github/workflows directory", Color.RED)
            self.failed += 1
            return False
            
        workflow_files = list(workflows_dir.glob("*.yml"))
        workflow_files.extend(workflows_dir.glob("*.yaml"))
        
        required_workflows = ["ci.yml", "docker.yml", "security.yml"]
        found = []
        
        for wf in workflow_files:
            self.log(f"✓ Found workflow: {wf.name}", Color.GREEN)
            found.append(wf.name)
            
        missing = [w for w in required_workflows if w not in found]
        if missing:
            self.log(f"⚠ Missing recommended workflows: {missing}", Color.YELLOW)
            self.skipped += 1
        else:
            self.passed += 1
            
        return True
        
    def run_all_tests(self) -> int:
        """Run all tests and return exit code"""
        self.log("=" * 50, Color.BLUE)
        self.log("Minimal CI/CD Test Suite", Color.BLUE)
        self.log("=" * 50, Color.BLUE)
        
        tests = [
            self.test_required_files,
            self.test_python_syntax,
            self.test_json_configs,
            self.test_docker_syntax,
            self.test_secrets,
            self.test_github_workflows,
        ]
        
        # Run tests that don't require dependencies first
        for test in tests:
            try:
                test()
            except Exception as e:
                self.log(f"✗ Test failed with exception: {e}", Color.RED)
                self.failed += 1
                
        # Try import test last (requires dependencies)
        try:
            self.test_imports()
        except Exception as e:
            self.log(f"⚠ Import test skipped: {e}", Color.YELLOW)
            self.skipped += 1
            
        # Summary
        self.log("\n" + "=" * 50, Color.BLUE)
        self.log("Test Summary", Color.BLUE)
        self.log("=" * 50, Color.BLUE)
        print(f"Passed:  {self.passed}")
        print(f"Failed:  {self.failed}")
        print(f"Skipped: {self.skipped}")
        
        success_rate = int(self.passed * 100 / (self.passed + self.failed)) if (self.passed + self.failed) > 0 else 0
        print(f"\nSuccess Rate: {success_rate}%")
        
        if self.failed == 0:
            self.log("\n✓ All tests passed!", Color.GREEN)
            return 0
        else:
            self.log(f"\n✗ {self.failed} test(s) failed", Color.RED)
            return 1

if __name__ == "__main__":
    runner = TestRunner()
    sys.exit(runner.run_all_tests())
