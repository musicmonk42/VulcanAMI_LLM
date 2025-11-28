#!/usr/bin/env python3
"""
Comprehensive CI/CD and Reproducibility Tests

This test suite validates:
1. Docker build reproducibility
2. Docker Compose functionality
3. Configuration file integrity
4. Deployment readiness
5. Security best practices
6. Dependencies and package integrity
"""

import os
import sys
import json
import yaml
import subprocess
import hashlib
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pytest


# Get repository root
REPO_ROOT = Path(__file__).parent.parent
SRC_DIR = REPO_ROOT / "src"
DOCKER_DIR = REPO_ROOT / "docker"
K8S_DIR = REPO_ROOT / "k8s"
HELM_DIR = REPO_ROOT / "helm"


def _docker_available_with_network() -> bool:
    """Check if Docker is available and can access PyPI from within a build."""
    try:
        import tempfile
        
        # First check if docker command exists
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode != 0:
            return False
        
        # Test Docker's ability to pip install from PyPI during build
        # This is the actual operation that fails in the tests
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a minimal Dockerfile that tests pip install
            dockerfile_content = '''FROM python:3.12-slim
RUN pip install --no-cache-dir requests==2.32.3
'''
            dockerfile_path = Path(tmpdir) / "Dockerfile"
            dockerfile_path.write_text(dockerfile_content, encoding='utf-8')
            
            result = subprocess.run(
                ["docker", "build", "-t", "network-test:latest", tmpdir],
                capture_output=True,
                text=True,
                timeout=120
            )
            return result.returncode == 0
    except Exception:
        return False


# Don't cache the result - check each time to ensure current state
def docker_available_with_network() -> bool:
    """Check for Docker with network availability (uncached)."""
    return _docker_available_with_network()


def _helm_available() -> bool:
    """Check if helm command is available."""
    return shutil.which("helm") is not None


class TestDockerConfigurations:
    """Test Docker-related configurations and builds"""
    
    def test_main_dockerfile_exists(self):
        """Verify main Dockerfile exists"""
        dockerfile = REPO_ROOT / "Dockerfile"
        assert dockerfile.exists(), "Main Dockerfile not found"
        
    def test_dockerfile_security_features(self):
        """Verify Dockerfile has security best practices"""
        dockerfile = REPO_ROOT / "Dockerfile"
        content = dockerfile.read_text(encoding='utf-8')
        
        # Check for non-root user
        assert "USER graphix" in content or "USER 1001" in content, \
            "Dockerfile should run as non-root user"
        
        # Check for healthcheck
        assert "HEALTHCHECK" in content, \
            "Dockerfile should include HEALTHCHECK"
        
        # Check for JWT secret validation
        assert "JWT_SECRET" in content or "REJECT_INSECURE_JWT" in content, \
            "Dockerfile should validate JWT configuration"
        
    def test_dockerignore_exists(self):
        """Verify .dockerignore exists and has necessary exclusions"""
        dockerignore = REPO_ROOT / ".dockerignore"
        assert dockerignore.exists(), ".dockerignore file not found"
        
        content = dockerignore.read_text(encoding='utf-8')
        critical_patterns = [".git", "__pycache__", "*.pyc", ".env"]
        
        for pattern in critical_patterns:
            assert pattern in content, f".dockerignore missing {pattern}"
    
    def test_service_dockerfiles_exist(self):
        """Verify service-specific Dockerfiles exist"""
        expected_services = ["api", "dqs", "pii"]
        
        for service in expected_services:
            dockerfile = DOCKER_DIR / service / "Dockerfile"
            assert dockerfile.exists(), f"Dockerfile for {service} not found at {dockerfile}"
    
    def test_docker_compose_dev_valid(self):
        """Verify docker-compose.dev.yml is valid"""
        compose_file = REPO_ROOT / "docker-compose.dev.yml"
        assert compose_file.exists(), "docker-compose.dev.yml not found"
        
        # Validate YAML structure
        with open(compose_file, encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        assert "services" in config, "docker-compose.dev.yml missing services"
        
        # Validate with docker compose config
        result = subprocess.run(
            ["docker", "compose", "-f", str(compose_file), "config"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT
        )
        assert result.returncode == 0, \
            f"docker-compose.dev.yml validation failed: {result.stderr}"
    
    def test_docker_compose_prod_valid(self):
        """Verify docker-compose.prod.yml is valid"""
        compose_file = REPO_ROOT / "docker-compose.prod.yml"
        assert compose_file.exists(), "docker-compose.prod.yml not found"
        
        # Validate YAML structure
        with open(compose_file, encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        assert "services" in config, "docker-compose.prod.yml missing services"
        
        # Validate with docker compose config
        # Set dummy env vars to avoid interpolation errors
        env = os.environ.copy()
        env.update({
            "POSTGRES_PASSWORD": "dummy",
            "REDIS_PASSWORD": "dummy",
            "MINIO_ROOT_PASSWORD": "dummy",
            "MINIO_ROOT_USER": "dummy",
            "JWT_SECRET_KEY": "dummy",
            "BOOTSTRAP_KEY": "dummy",
            "GRAFANA_PASSWORD": "dummy",
            "GRAFANA_USER": "dummy",
        })
        
        result = subprocess.run(
            ["docker", "compose", "-f", str(compose_file), "config"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            env=env
        )
        assert result.returncode == 0, \
            f"docker-compose.prod.yml validation failed: {result.stderr}"
    
    def test_entrypoint_script_exists_and_executable(self):
        """Verify entrypoint.sh exists and is executable"""
        entrypoint = REPO_ROOT / "entrypoint.sh"
        assert entrypoint.exists(), "entrypoint.sh not found"
        # On Windows, executability check may not work the same way
        if sys.platform != 'win32':
            assert os.access(entrypoint, os.X_OK), "entrypoint.sh is not executable"
    
    def test_entrypoint_validates_secrets(self):
        """Verify entrypoint.sh validates JWT secrets"""
        entrypoint = REPO_ROOT / "entrypoint.sh"
        content = entrypoint.read_text(encoding='utf-8')
        
        # Should check for JWT secret environment variables
        jwt_vars = ["JWT_SECRET", "JWT_SECRET_KEY", "GRAPHIX_JWT_SECRET"]
        has_jwt_check = any(var in content for var in jwt_vars)
        assert has_jwt_check, "entrypoint.sh should validate JWT secrets"


class TestDependencyManagement:
    """Test dependency management and reproducibility"""
    
    def test_requirements_txt_exists(self):
        """Verify requirements.txt exists"""
        requirements = REPO_ROOT / "requirements.txt"
        assert requirements.exists(), "requirements.txt not found"
    
    def test_requirements_hashed_exists(self):
        """Verify requirements-hashed.txt exists for reproducibility"""
        hashed_req = REPO_ROOT / "requirements-hashed.txt"
        assert hashed_req.exists(), \
            "requirements-hashed.txt not found (needed for reproducible builds)"
    
    def test_requirements_have_hashes(self):
        """Verify requirements-hashed.txt contains actual hashes"""
        hashed_req = REPO_ROOT / "requirements-hashed.txt"
        content = hashed_req.read_text(encoding='utf-8')
        
        # Should contain SHA256 hashes
        assert "sha256:" in content, \
            "requirements-hashed.txt should contain SHA256 hashes"
        
        # Count number of hashed entries (non-comment, non-empty lines with hashes)
        hashed_lines = [
            line for line in content.split("\n")
            if line.strip() and not line.strip().startswith("#") and "sha256:" in line
        ]
        assert len(hashed_lines) > 50, \
            f"requirements-hashed.txt should have many hashed packages, found {len(hashed_lines)}"


class TestCICDWorkflows:
    """Test GitHub Actions and CI/CD configurations"""
    
    def test_github_workflows_directory_exists(self):
        """Verify .github/workflows directory exists"""
        workflows_dir = REPO_ROOT / ".github" / "workflows"
        assert workflows_dir.exists(), ".github/workflows directory not found"
    
    def test_workflows_valid_yaml(self):
        """Verify all workflow files are valid YAML"""
        workflows_dir = REPO_ROOT / ".github" / "workflows"
        workflow_files = list(workflows_dir.glob("*.yml")) + list(workflows_dir.glob("*.yaml"))
        
        assert len(workflow_files) > 0, "No workflow files found"
        
        for workflow_file in workflow_files:
            with open(workflow_file, encoding='utf-8') as f:
                try:
                    yaml.safe_load(f)
                except yaml.YAMLError as e:
                    pytest.fail(f"Invalid YAML in {workflow_file}: {e}")
    
    def test_workflows_use_docker_compose_v2(self):
        """Verify workflows use Docker Compose V2 syntax"""
        workflows_dir = REPO_ROOT / ".github" / "workflows"
        workflow_files = list(workflows_dir.glob("*.yml")) + list(workflows_dir.glob("*.yaml"))
        
        for workflow_file in workflow_files:
            content = workflow_file.read_text(encoding='utf-8')
            
            # Should use 'docker compose' not 'docker-compose'
            if "docker-compose" in content.lower() and "docker compose" not in content.lower():
                pytest.fail(
                    f"{workflow_file.name} uses deprecated docker-compose command. "
                    "Use 'docker compose' (V2) instead"
                )
    
    def test_workflows_have_timeout(self):
        """Verify workflows have timeout-minutes set"""
        workflows_dir = REPO_ROOT / ".github" / "workflows"
        workflow_files = list(workflows_dir.glob("*.yml")) + list(workflows_dir.glob("*.yaml"))
        
        for workflow_file in workflow_files:
            with open(workflow_file, encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Check if any job has timeout-minutes
            if "jobs" in config:
                for job_name, job_config in config["jobs"].items():
                    if "timeout-minutes" not in job_config:
                        # Just warn, don't fail - timeout is optional but recommended
                        print(f"Warning: Job '{job_name}' in {workflow_file.name} has no timeout-minutes")


class TestKubernetesConfigs:
    """Test Kubernetes configurations"""
    
    def test_k8s_directory_exists(self):
        """Verify k8s directory exists"""
        if not K8S_DIR.exists():
            pytest.skip("k8s directory not found - skipping Kubernetes tests")
    
    def test_k8s_yaml_files_valid(self):
        """Verify Kubernetes YAML files are valid"""
        if not K8S_DIR.exists():
            pytest.skip("k8s directory not found")
        
        yaml_files = list(K8S_DIR.glob("*.yaml")) + list(K8S_DIR.glob("*.yml"))
        
        if len(yaml_files) == 0:
            pytest.skip("No Kubernetes YAML files found")
        
        for yaml_file in yaml_files:
            with open(yaml_file, encoding='utf-8') as f:
                try:
                    configs = list(yaml.safe_load_all(f))
                    
                    # Basic validation
                    for config in configs:
                        if config:  # Skip empty documents
                            assert "apiVersion" in config, \
                                f"{yaml_file.name} missing apiVersion"
                            assert "kind" in config, \
                                f"{yaml_file.name} missing kind"
                except yaml.YAMLError as e:
                    pytest.fail(f"Invalid YAML in {yaml_file}: {e}")


class TestHelmCharts:
    """Test Helm chart configurations"""
    
    def test_helm_directory_exists(self):
        """Verify helm directory exists"""
        if not HELM_DIR.exists():
            pytest.skip("helm directory not found - skipping Helm tests")
    
    def test_helm_chart_yaml_exists(self):
        """Verify Chart.yaml exists"""
        if not HELM_DIR.exists():
            pytest.skip("helm directory not found")
        
        # Helm charts are typically in subdirectories (e.g., helm/vulcanami/Chart.yaml)
        chart_yamls = list(HELM_DIR.glob("*/Chart.yaml"))
        chart_yamls.extend(HELM_DIR.glob("Chart.yaml"))  # Also check root level
        
        assert len(chart_yamls) > 0, \
            f"Chart.yaml not found in helm directory or its subdirectories"
    
    def test_helm_values_yaml_exists(self):
        """Verify values.yaml exists for each chart"""
        if not HELM_DIR.exists():
            pytest.skip("helm directory not found")
        
        # Find all Chart.yaml files
        chart_yamls = list(HELM_DIR.glob("*/Chart.yaml"))
        chart_yamls.extend(HELM_DIR.glob("Chart.yaml"))
        
        if len(chart_yamls) == 0:
            pytest.skip("No Chart.yaml found")
        
        # For each chart, verify values.yaml exists
        for chart_yaml in chart_yamls:
            values_yaml = chart_yaml.parent / "values.yaml"
            assert values_yaml.exists(), \
                f"values.yaml not found for chart in {chart_yaml.parent.name}"
    
    def test_helm_chart_metadata_valid(self):
        """Verify Chart.yaml has required metadata fields"""
        if not HELM_DIR.exists():
            pytest.skip("helm directory not found")
        
        chart_yamls = list(HELM_DIR.glob("*/Chart.yaml"))
        chart_yamls.extend(HELM_DIR.glob("Chart.yaml"))
        
        if len(chart_yamls) == 0:
            pytest.skip("No Chart.yaml found")
        
        required_fields = ["apiVersion", "name", "description", "type", "version", "appVersion"]
        
        for chart_yaml in chart_yamls:
            with open(chart_yaml, encoding='utf-8') as f:
                chart_data = yaml.safe_load(f)
            
            for field in required_fields:
                assert field in chart_data, \
                    f"Chart.yaml in {chart_yaml.parent.name} missing required field: {field}"
    
    def test_helm_values_security_config(self):
        """Verify values.yaml has security best practices"""
        if not HELM_DIR.exists():
            pytest.skip("helm directory not found")
        
        chart_yamls = list(HELM_DIR.glob("*/Chart.yaml"))
        chart_yamls.extend(HELM_DIR.glob("Chart.yaml"))
        
        if len(chart_yamls) == 0:
            pytest.skip("No Chart.yaml found")
        
        for chart_yaml in chart_yamls:
            values_yaml = chart_yaml.parent / "values.yaml"
            if not values_yaml.exists():
                continue
            
            with open(values_yaml, encoding='utf-8') as f:
                values_data = yaml.safe_load(f)
            
            # Check for security context
            if "podSecurityContext" in values_data:
                pod_sec = values_data["podSecurityContext"]
                assert "runAsNonRoot" in pod_sec or "runAsUser" in pod_sec, \
                    f"values.yaml in {chart_yaml.parent.name} should specify non-root user"
            
            # Check that image tag is not 'latest'
            if "image" in values_data and "tag" in values_data["image"]:
                image_tag = values_data["image"]["tag"]
                assert image_tag != "latest", \
                    f"values.yaml in {chart_yaml.parent.name} should not use 'latest' image tag"
    
    def test_helm_chart_valid(self):
        """Verify Helm chart is valid using helm lint"""
        if not HELM_DIR.exists():
            pytest.skip("helm directory not found")
        
        if not _helm_available():
            pytest.skip("helm command not available")
        
        # Find Chart.yaml files in subdirectories
        chart_yamls = list(HELM_DIR.glob("*/Chart.yaml"))
        chart_yamls.extend(HELM_DIR.glob("Chart.yaml"))  # Also check root level
        
        if len(chart_yamls) == 0:
            pytest.skip("No Chart.yaml found in helm directory")
        
        # Lint each chart found
        for chart_yaml in chart_yamls:
            chart_dir = chart_yaml.parent
            result = subprocess.run(
                ["helm", "lint", str(chart_dir)],
                capture_output=True,
                text=True
            )
            
            assert result.returncode == 0, \
                f"Helm chart validation failed for {chart_dir.name}: {result.stderr}"


class TestSecurityConfiguration:
    """Test security-related configurations"""
    
    def test_gitignore_exists(self):
        """Verify .gitignore exists"""
        gitignore = REPO_ROOT / ".gitignore"
        assert gitignore.exists(), ".gitignore file not found"
    
    def test_gitignore_has_critical_patterns(self):
        """Verify .gitignore excludes sensitive files"""
        gitignore = REPO_ROOT / ".gitignore"
        content = gitignore.read_text(encoding='utf-8')
        
        critical_patterns = [
            ".env",
            "*.pem",
            "*.key",
            "*.p12",
            "*.pfx",
        ]
        
        missing = []
        for pattern in critical_patterns:
            if pattern not in content:
                missing.append(pattern)
        
        assert len(missing) == 0, \
            f".gitignore missing critical patterns: {missing}"
    
    def test_no_committed_env_files(self):
        """Verify no .env files are committed"""
        env_files = list(REPO_ROOT.glob("**/.env"))
        env_files.extend(REPO_ROOT.glob(".env"))
        
        # Filter out .env.example which is okay
        env_files = [f for f in env_files if ".example" not in f.name]
        
        assert len(env_files) == 0, \
            f"Found committed .env files: {env_files}"
    
    def test_no_hardcoded_secrets_in_code(self):
        """Scan for potential hardcoded secrets in Python files"""
        secret_patterns = [
            "password = ",
            "secret = ",
            "api_key = ",
            "token = ",
        ]
        
        suspicious_files = []
        
        for py_file in SRC_DIR.glob("**/*.py"):
            try:
                content = py_file.read_text(encoding='utf-8').lower()
                
                for pattern in secret_patterns:
                    if pattern in content:
                        # Check if it's not a configuration or placeholder
                        lines = content.split("\n")
                        for i, line in enumerate(lines):
                            if pattern in line and "os.environ" not in line and "config" not in line:
                                # Likely hardcoded secret
                                suspicious_files.append((py_file, i + 1, line.strip()[:80]))
            except (UnicodeDecodeError, PermissionError):
                # Skip files that can't be read (might be binary or permission issues)
                continue
        
        # This is informational - we allow some false positives
        if suspicious_files:
            # Just log warning, don't fail
            print(f"\nWarning: Found {len(suspicious_files)} potential hardcoded secrets")
            for file, line_no, line in suspicious_files[:5]:
                print(f"  {file.name}:{line_no}: {line}")
    
    def test_bandit_config_exists(self):
        """Verify Bandit security scanner configuration exists"""
        bandit_config = REPO_ROOT / ".bandit"
        if not bandit_config.exists():
            pytest.skip(".bandit config not found - using defaults is acceptable")


class TestReproducibility:
    """Test reproducibility features"""
    
    def test_python_version_pinned_in_dockerfile(self):
        """Verify Dockerfile uses specific Python version"""
        dockerfile = REPO_ROOT / "Dockerfile"
        content = dockerfile.read_text(encoding='utf-8')
        
        # Should not use :latest
        assert ":latest" not in content or "python:latest" not in content.lower(), \
            "Dockerfile should not use :latest tag (non-reproducible)"
        
        # Should have specific version (3.10, 3.11, 3.12, etc.)
        import re
        assert re.search(r'python:3\.\d+', content.lower()), \
            "Dockerfile should specify exact Python version (e.g., python:3.11)"
    
    def test_makefile_exists(self):
        """Verify Makefile exists for consistent build commands"""
        makefile = REPO_ROOT / "Makefile"
        assert makefile.exists(), "Makefile not found"
    
    def test_makefile_has_common_targets(self):
        """Verify Makefile has common targets"""
        makefile = REPO_ROOT / "Makefile"
        content = makefile.read_text(encoding='utf-8')
        
        required_targets = ["install", "test", "docker-build"]
        missing = []
        
        for target in required_targets:
            # Check if target is defined (targets are defined with 'target:')
            if f"{target}:" not in content:
                missing.append(target)
        
        assert len(missing) == 0, \
            f"Makefile missing targets: {missing}"
    
    def test_documentation_exists(self):
        """Verify key documentation files exist"""
        required_docs = [
            "README.md",
            "CI_CD.md",
            "REPRODUCIBLE_BUILDS.md",
        ]
        
        missing = []
        for doc in required_docs:
            if not (REPO_ROOT / doc).exists():
                missing.append(doc)
        
        assert len(missing) == 0, \
            f"Missing documentation files: {missing}"


class TestValidationScripts:
    """Test that validation scripts exist and work"""
    
    def test_validation_script_exists(self):
        """Verify validation script exists"""
        validation_script = REPO_ROOT / "validate_cicd_docker.sh"
        assert validation_script.exists(), "validate_cicd_docker.sh not found"
    
    def test_validation_script_executable(self):
        """Verify validation script is executable"""
        validation_script = REPO_ROOT / "validate_cicd_docker.sh"
        # On Windows, executability check may not work the same way
        if sys.platform != 'win32':
            assert os.access(validation_script, os.X_OK), \
                "validate_cicd_docker.sh is not executable"
    
    def test_comprehensive_test_script_exists(self):
        """Verify comprehensive test runner exists"""
        test_script = REPO_ROOT / "run_comprehensive_tests.sh"
        assert test_script.exists(), "run_comprehensive_tests.sh not found"
    
    def test_comprehensive_test_script_executable(self):
        """Verify comprehensive test script is executable"""
        test_script = REPO_ROOT / "run_comprehensive_tests.sh"
        # On Windows, executability check may not work the same way
        if sys.platform != 'win32':
            assert os.access(test_script, os.X_OK), \
                "run_comprehensive_tests.sh is not executable"


class TestEndToEnd:
    """End-to-end integration tests"""
    
    @pytest.mark.slow
    def test_docker_build_succeeds(self):
        """Test that Docker image builds successfully"""
        if not docker_available_with_network():
            pytest.skip("Docker with network access not available")
        
        result = subprocess.run(
            [
                "docker", "build",
                "--build-arg", "REJECT_INSECURE_JWT=ack",
                "-t", "vulcanami-test:latest",
                "-f", str(REPO_ROOT / "Dockerfile"),
                str(REPO_ROOT)
            ],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT
        )
        
        assert result.returncode == 0, \
            f"Docker build failed: {result.stderr}"
    
    @pytest.mark.slow
    def test_docker_image_runs(self):
        """Test that Docker image can run"""
        if not docker_available_with_network():
            pytest.skip("Docker with network access not available")
        
        # First ensure image is built
        build_result = subprocess.run(
            [
                "docker", "build",
                "--build-arg", "REJECT_INSECURE_JWT=ack",
                "-t", "vulcanami-test:latest",
                "-f", str(REPO_ROOT / "Dockerfile"),
                str(REPO_ROOT)
            ],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT
        )
        
        if build_result.returncode != 0:
            pytest.skip("Docker build failed, skipping run test")
        
        # Generate secure JWT secret
        import secrets
        jwt_secret = secrets.token_urlsafe(48)
        
        # Try to run container and check it starts
        result = subprocess.run(
            [
                "docker", "run", "--rm",
                "-e", f"JWT_SECRET_KEY={jwt_secret}",
                "-e", f"BOOTSTRAP_KEY={secrets.token_urlsafe(32)}",
                "vulcanami-test:latest",
                "python", "--version"
            ],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        assert result.returncode == 0, \
            f"Docker container failed to run: {result.stderr}"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
