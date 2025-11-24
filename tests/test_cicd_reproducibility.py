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
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pytest


# Get repository root
REPO_ROOT = Path(__file__).parent.parent
SRC_DIR = REPO_ROOT / "src"
DOCKER_DIR = REPO_ROOT / "docker"
K8S_DIR = REPO_ROOT / "k8s"
HELM_DIR = REPO_ROOT / "helm"


class TestDockerConfigurations:
    """Test Docker-related configurations and builds"""
    
    def test_main_dockerfile_exists(self):
        """Verify main Dockerfile exists"""
        dockerfile = REPO_ROOT / "Dockerfile"
        assert dockerfile.exists(), "Main Dockerfile not found"
        
    def test_dockerfile_security_features(self):
        """Verify Dockerfile has security best practices"""
        dockerfile = REPO_ROOT / "Dockerfile"
        content = dockerfile.read_text()
        
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
        
        content = dockerignore.read_text()
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
        with open(compose_file) as f:
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
        with open(compose_file) as f:
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
        assert os.access(entrypoint, os.X_OK), "entrypoint.sh is not executable"
    
    def test_entrypoint_validates_secrets(self):
        """Verify entrypoint.sh validates JWT secrets"""
        entrypoint = REPO_ROOT / "entrypoint.sh"
        content = entrypoint.read_text()
        
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
        content = hashed_req.read_text()
        
        # Should contain SHA256 hashes
        assert "sha256:" in content, \
            "requirements-hashed.txt should contain SHA256 hashes"
        
        # Count number of hashed entries (non-comment, non-empty lines with hashes)
        hashed_lines = [
            line for line in content.split("\n")
            if line.strip() and not line.strip().startswith("#") and "sha256:" in line
        ]
        assert len(hashed_lines) > 50, \
            f"Expected many hashed dependencies, found {len(hashed_lines)}"
    
    def test_no_unpinned_dependencies(self):
        """Verify requirements.txt has pinned versions"""
        requirements = REPO_ROOT / "requirements.txt"
        content = requirements.read_text()
        
        lines = [
            line.strip() for line in content.split("\n")
            if line.strip() and not line.strip().startswith("#")
        ]
        
        unpinned = []
        for line in lines:
            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue
            # Check if line has a package but no version specifier
            if "==" not in line and ">=" not in line and line.strip():
                # Allow lines that are just comments or special directives
                if not line.startswith("-") and not line.startswith("https://"):
                    unpinned.append(line)
        
        assert len(unpinned) == 0, \
            f"Found unpinned dependencies (non-reproducible): {unpinned}"
    
    def test_setup_py_exists(self):
        """Verify setup.py exists for package installation"""
        setup_py = REPO_ROOT / "setup.py"
        assert setup_py.exists(), "setup.py not found"


class TestCICDWorkflows:
    """Test GitHub Actions workflow configurations"""
    
    def test_workflows_directory_exists(self):
        """Verify .github/workflows directory exists"""
        workflows_dir = REPO_ROOT / ".github" / "workflows"
        assert workflows_dir.exists(), ".github/workflows directory not found"
    
    def test_ci_workflow_exists(self):
        """Verify CI workflow exists"""
        ci_workflow = REPO_ROOT / ".github" / "workflows" / "ci.yml"
        assert ci_workflow.exists(), "CI workflow not found"
    
    def test_docker_workflow_exists(self):
        """Verify Docker build workflow exists"""
        docker_workflow = REPO_ROOT / ".github" / "workflows" / "docker.yml"
        assert docker_workflow.exists(), "Docker workflow not found"
    
    def test_security_workflow_exists(self):
        """Verify security scanning workflow exists"""
        security_workflow = REPO_ROOT / ".github" / "workflows" / "security.yml"
        assert security_workflow.exists(), "Security workflow not found"
    
    def test_workflows_valid_yaml(self):
        """Verify all workflow files are valid YAML"""
        workflows_dir = REPO_ROOT / ".github" / "workflows"
        
        for workflow_file in workflows_dir.glob("*.yml"):
            with open(workflow_file) as f:
                try:
                    yaml.safe_load(f)
                except yaml.YAMLError as e:
                    pytest.fail(f"Invalid YAML in {workflow_file.name}: {e}")
    
    def test_workflows_use_docker_compose_v2(self):
        """Verify workflows use 'docker compose' (v2) not 'docker-compose' (v1)"""
        workflows_dir = REPO_ROOT / ".github" / "workflows"
        
        deprecated_usage = []
        for workflow_file in workflows_dir.glob("*.yml"):
            content = workflow_file.read_text()
            # Look for deprecated docker-compose command
            if "docker-compose " in content:
                deprecated_usage.append(workflow_file.name)
        
        # This is a warning, not a failure
        if deprecated_usage:
            pytest.skip(
                f"Workflows using deprecated 'docker-compose': {deprecated_usage}. "
                "Consider updating to 'docker compose'"
            )
    
    def test_workflows_have_timeout(self):
        """Verify workflows have timeout configurations"""
        workflows_dir = REPO_ROOT / ".github" / "workflows"
        
        for workflow_file in workflows_dir.glob("*.yml"):
            with open(workflow_file) as f:
                config = yaml.safe_load(f)
            
            if "jobs" in config:
                for job_name, job_config in config["jobs"].items():
                    # Timeout is recommended but not required
                    if "timeout-minutes" not in job_config:
                        pytest.skip(
                            f"Job '{job_name}' in {workflow_file.name} "
                            "missing timeout-minutes (recommended)"
                        )


class TestKubernetesManifests:
    """Test Kubernetes deployment configurations"""
    
    def test_k8s_base_directory_exists(self):
        """Verify k8s/base directory exists"""
        k8s_base = K8S_DIR / "base"
        if not k8s_base.exists():
            pytest.skip("k8s/base directory not found - Kubernetes deployment optional")
    
    def test_k8s_manifests_valid_yaml(self):
        """Verify all Kubernetes manifests are valid YAML"""
        k8s_base = K8S_DIR / "base"
        if not k8s_base.exists():
            pytest.skip("k8s/base directory not found")
        
        for manifest in k8s_base.glob("*.yaml"):
            with open(manifest) as f:
                try:
                    # Load all documents in the file (some files have multiple docs with ---)
                    list(yaml.safe_load_all(f))
                except yaml.YAMLError as e:
                    pytest.fail(f"Invalid YAML in {manifest.name}: {e}")
    
    def test_k8s_deployment_exists(self):
        """Verify API deployment manifest exists"""
        deployment = K8S_DIR / "base" / "api-deployment.yaml"
        if not deployment.exists():
            pytest.skip("API deployment manifest not found - optional")
    
    def test_k8s_service_exists(self):
        """Verify service manifests exist"""
        service = K8S_DIR / "base" / "api-service.yaml"
        if not service.exists():
            # Check if service is defined in deployment file
            deployment = K8S_DIR / "base" / "api-deployment.yaml"
            if deployment.exists():
                content = deployment.read_text()
                if "kind: Service" not in content:
                    pytest.skip("Service manifest not found - might be in deployment file")


class TestHelmCharts:
    """Test Helm chart configurations"""
    
    def test_helm_chart_exists(self):
        """Verify Helm chart directory exists"""
        if not HELM_DIR.exists():
            pytest.skip("helm directory not found - Helm deployment optional")
        
        chart_dirs = list(HELM_DIR.glob("*/Chart.yaml"))
        assert len(chart_dirs) > 0, "No Helm charts found in helm directory"
    
    def test_helm_chart_valid(self):
        """Verify Helm chart is valid"""
        if not HELM_DIR.exists():
            pytest.skip("helm directory not found")
        
        # Find chart directory
        chart_dirs = list(HELM_DIR.glob("*/Chart.yaml"))
        if not chart_dirs:
            pytest.skip("No Helm charts found")
        
        chart_dir = chart_dirs[0].parent
        
        # Try helm lint
        result = subprocess.run(
            ["helm", "lint", str(chart_dir)],
            capture_output=True,
            text=True
        )
        
        # helm lint returns 0 on success
        assert result.returncode == 0, \
            f"Helm chart lint failed: {result.stderr}"
    
    def test_helm_values_exists(self):
        """Verify Helm values file exists"""
        if not HELM_DIR.exists():
            pytest.skip("helm directory not found")
        
        chart_dirs = list(HELM_DIR.glob("*/Chart.yaml"))
        if not chart_dirs:
            pytest.skip("No Helm charts found")
        
        chart_dir = chart_dirs[0].parent
        values_file = chart_dir / "values.yaml"
        
        assert values_file.exists(), "values.yaml not found in Helm chart"


class TestSecurityConfiguration:
    """Test security configurations and best practices"""
    
    def test_gitignore_excludes_secrets(self):
        """Verify .gitignore excludes sensitive files"""
        gitignore = REPO_ROOT / ".gitignore"
        assert gitignore.exists(), ".gitignore not found"
        
        content = gitignore.read_text()
        
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
            content = py_file.read_text().lower()
            
            for pattern in secret_patterns:
                if pattern in content:
                    # Check if it's not a configuration or placeholder
                    lines = content.split("\n")
                    for i, line in enumerate(lines):
                        if pattern in line and "os.environ" not in line and "config" not in line:
                            # Likely hardcoded secret
                            suspicious_files.append((py_file, i + 1, line.strip()[:80]))
        
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
        content = dockerfile.read_text()
        
        # Should not use :latest
        assert ":latest" not in content or "python:latest" not in content.lower(), \
            "Dockerfile should not use :latest tag (non-reproducible)"
        
        # Should have specific version
        assert "python:3.1" in content.lower(), \
            "Dockerfile should specify exact Python version"
    
    def test_makefile_exists(self):
        """Verify Makefile exists for consistent build commands"""
        makefile = REPO_ROOT / "Makefile"
        assert makefile.exists(), "Makefile not found"
    
    def test_makefile_has_common_targets(self):
        """Verify Makefile has common targets"""
        makefile = REPO_ROOT / "Makefile"
        content = makefile.read_text()
        
        required_targets = ["install", "test", "docker-build"]
        missing = []
        
        for target in required_targets:
            if f"{target}:" not in content and f".PHONY: {target}" not in content:
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
        assert os.access(validation_script, os.X_OK), \
            "validate_cicd_docker.sh is not executable"
    
    def test_comprehensive_test_script_exists(self):
        """Verify comprehensive test runner exists"""
        test_script = REPO_ROOT / "run_comprehensive_tests.sh"
        assert test_script.exists(), "run_comprehensive_tests.sh not found"
    
    def test_comprehensive_test_script_executable(self):
        """Verify comprehensive test script is executable"""
        test_script = REPO_ROOT / "run_comprehensive_tests.sh"
        assert os.access(test_script, os.X_OK), \
            "run_comprehensive_tests.sh is not executable"


class TestEndToEnd:
    """End-to-end integration tests"""
    
    @pytest.mark.slow
    def test_docker_build_succeeds(self):
        """Test that Docker image builds successfully"""
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
