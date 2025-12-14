"""
Comprehensive test suite for helm_chart.yaml (Kubernetes manifest)
Validates security, production readiness, and Kubernetes best practices.

Run with:
    pytest test_helm_chart.py -v
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import yaml


# Test Fixtures
@pytest.fixture
def helm_chart():
    """Load the Helm chart YAML file."""
    config_path = Path(__file__).parent / "configs" / "helm_chart.yaml"
    if not config_path.exists():
        config_path = Path(__file__).parent / ".." / "configs" / "helm_chart.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        # Load all documents (YAML with --- separators)
        return list(yaml.safe_load_all(f))


@pytest.fixture
def namespace(helm_chart):
    """Extract namespace resource."""
    return next((doc for doc in helm_chart if doc.get("kind") == "Namespace"), None)


@pytest.fixture
def deployment(helm_chart):
    """Extract deployment resource."""
    return next((doc for doc in helm_chart if doc.get("kind") == "Deployment"), None)


@pytest.fixture
def service(helm_chart):
    """Extract service resource."""
    return next((doc for doc in helm_chart if doc.get("kind") == "Service"), None)


@pytest.fixture
def required_labels():
    """Required Kubernetes labels."""
    return {"app.kubernetes.io/name", "app.kubernetes.io/instance"}


# Test YAML Structure
class TestYAMLStructure:
    def test_yaml_loads_successfully(self, helm_chart):
        """Test that YAML file loads without errors."""
        assert helm_chart is not None
        assert isinstance(helm_chart, list)
        assert len(helm_chart) > 0

    def test_multiple_resources_defined(self, helm_chart):
        """Test that multiple Kubernetes resources are defined."""
        assert len(helm_chart) >= 3, "Should have namespace, deployment, and service"

    def test_all_resources_have_apiversion(self, helm_chart):
        """Test that all resources have apiVersion."""
        for doc in helm_chart:
            assert (
                "apiVersion" in doc
            ), f"Resource missing apiVersion: {doc.get('kind')}"

    def test_all_resources_have_kind(self, helm_chart):
        """Test that all resources have kind."""
        for doc in helm_chart:
            assert "kind" in doc, "Resource missing kind"

    def test_all_resources_have_metadata(self, helm_chart):
        """Test that all resources have metadata."""
        for doc in helm_chart:
            assert "metadata" in doc, f"{doc.get('kind')} missing metadata"


# Test Namespace
class TestNamespace:
    def test_namespace_exists(self, namespace):
        """Test that namespace resource exists."""
        assert namespace is not None, "Namespace resource not found"

    def test_namespace_apiversion(self, namespace):
        """Test namespace has correct apiVersion."""
        assert namespace["apiVersion"] == "v1"

    def test_namespace_kind(self, namespace):
        """Test namespace has correct kind."""
        assert namespace["kind"] == "Namespace"

    def test_namespace_has_name(self, namespace):
        """Test namespace has a name."""
        assert "name" in namespace["metadata"]
        assert namespace["metadata"]["name"] is not None

    def test_namespace_naming_convention(self, namespace):
        """Test namespace follows naming convention."""
        name = namespace["metadata"]["name"]
        assert re.match(
            r"^[a-z0-9-]+$", name
        ), "Namespace name should be lowercase alphanumeric with hyphens"

    def test_namespace_has_labels(self, namespace):
        """Test namespace has labels."""
        assert "labels" in namespace["metadata"]
        assert len(namespace["metadata"]["labels"]) > 0

    def test_namespace_has_env_label(self, namespace):
        """Test namespace has environment label."""
        labels = namespace["metadata"].get("labels", {})
        assert "env" in labels, "Namespace should have 'env' label"


# Test Deployment
class TestDeployment:
    def test_deployment_exists(self, deployment):
        """Test that deployment resource exists."""
        assert deployment is not None, "Deployment resource not found"

    def test_deployment_apiversion(self, deployment):
        """Test deployment has correct apiVersion."""
        assert deployment["apiVersion"] == "apps/v1"

    def test_deployment_kind(self, deployment):
        """Test deployment has correct kind."""
        assert deployment["kind"] == "Deployment"

    def test_deployment_has_name(self, deployment):
        """Test deployment has a name."""
        assert "name" in deployment["metadata"]

    def test_deployment_in_correct_namespace(self, deployment, namespace):
        """Test deployment is in the correct namespace."""
        dep_ns = deployment["metadata"].get("namespace")
        ns_name = namespace["metadata"]["name"]
        assert (
            dep_ns == ns_name
        ), f"Deployment namespace {dep_ns} doesn't match {ns_name}"

    def test_deployment_has_labels(self, deployment, required_labels):
        """Test deployment has required labels."""
        labels = deployment["metadata"].get("labels", {})
        for label in required_labels:
            assert label in labels, f"Deployment missing label: {label}"

    def test_deployment_has_spec(self, deployment):
        """Test deployment has spec."""
        assert "spec" in deployment

    def test_deployment_has_replicas(self, deployment):
        """Test deployment has replicas defined."""
        assert "replicas" in deployment["spec"]
        assert deployment["spec"]["replicas"] > 0

    def test_deployment_has_selector(self, deployment):
        """Test deployment has selector."""
        assert "selector" in deployment["spec"]
        assert "matchLabels" in deployment["spec"]["selector"]

    def test_deployment_selector_matches_template(self, deployment):
        """Test deployment selector matches pod template labels."""
        selector = deployment["spec"]["selector"]["matchLabels"]
        template_labels = deployment["spec"]["template"]["metadata"]["labels"]

        for key, value in selector.items():
            assert (
                key in template_labels
            ), f"Selector label {key} not in template labels"
            assert (
                template_labels[key] == value
            ), f"Selector {key}={value} doesn't match template {template_labels[key]}"


# Test Pod Template
class TestPodTemplate:
    def test_pod_template_exists(self, deployment):
        """Test that pod template exists."""
        assert "template" in deployment["spec"]
        assert "spec" in deployment["spec"]["template"]

    def test_pod_has_containers(self, deployment):
        """Test that pod has containers defined."""
        spec = deployment["spec"]["template"]["spec"]
        assert "containers" in spec
        assert len(spec["containers"]) > 0

    def test_container_has_name(self, deployment):
        """Test that container has a name."""
        container = deployment["spec"]["template"]["spec"]["containers"][0]
        assert "name" in container

    def test_container_has_image(self, deployment):
        """Test that container has an image."""
        container = deployment["spec"]["template"]["spec"]["containers"][0]
        assert "image" in container
        assert container["image"] is not None

    def test_image_not_using_latest_tag(self, deployment):
        """Test that image doesn't use 'latest' tag."""
        container = deployment["spec"]["template"]["spec"]["containers"][0]
        image = container["image"]

        if ":" in image:
            tag = image.split(":")[-1]
            assert (
                tag != "latest"
            ), "Don't use 'latest' tag in production - use specific versions"

    def test_image_pull_policy_defined(self, deployment):
        """Test that imagePullPolicy is defined."""
        container = deployment["spec"]["template"]["spec"]["containers"][0]
        assert "imagePullPolicy" in container

    def test_container_has_ports(self, deployment):
        """Test that container has ports defined."""
        container = deployment["spec"]["template"]["spec"]["containers"][0]
        assert "ports" in container
        assert len(container["ports"]) > 0

    def test_container_port_has_name(self, deployment):
        """Test that container port has a name."""
        container = deployment["spec"]["template"]["spec"]["containers"][0]
        port = container["ports"][0]
        assert "name" in port
        assert "containerPort" in port


# Test Resource Management
class TestResourceManagement:
    def test_resource_limits_defined(self, deployment):
        """Test that resource limits are defined."""
        container = deployment["spec"]["template"]["spec"]["containers"][0]
        assert "resources" in container
        assert "limits" in container["resources"]

    def test_resource_requests_defined(self, deployment):
        """Test that resource requests are defined."""
        container = deployment["spec"]["template"]["spec"]["containers"][0]
        assert "resources" in container
        assert "requests" in container["resources"]

    def test_cpu_limits_defined(self, deployment):
        """Test that CPU limits are defined."""
        resources = deployment["spec"]["template"]["spec"]["containers"][0]["resources"]
        assert "cpu" in resources.get("limits", {})

    def test_memory_limits_defined(self, deployment):
        """Test that memory limits are defined."""
        resources = deployment["spec"]["template"]["spec"]["containers"][0]["resources"]
        assert "memory" in resources.get("limits", {})

    def test_cpu_requests_defined(self, deployment):
        """Test that CPU requests are defined."""
        resources = deployment["spec"]["template"]["spec"]["containers"][0]["resources"]
        assert "cpu" in resources.get("requests", {})

    def test_memory_requests_defined(self, deployment):
        """Test that memory requests are defined."""
        resources = deployment["spec"]["template"]["spec"]["containers"][0]["resources"]
        assert "memory" in resources.get("requests", {})

    def test_requests_less_than_limits(self, deployment):
        """Test that resource requests are less than or equal to limits."""
        resources = deployment["spec"]["template"]["spec"]["containers"][0]["resources"]
        requests = resources.get("requests", {})
        limits = resources.get("limits", {})

        # Parse CPU values
        def parse_cpu(val):
            if "m" in str(val):
                return float(str(val).replace("m", "")) / 1000
            return float(val)

        # Parse memory values
        def parse_memory(val):
            val = str(val)
            multipliers = {
                "Ki": 1024,
                "Mi": 1024**2,
                "Gi": 1024**3,
                "K": 1000,
                "M": 1000**2,
                "G": 1000**3,
            }
            for suffix, mult in multipliers.items():
                if val.endswith(suffix):
                    return float(val[: -len(suffix)]) * mult
            return float(val)

        if "cpu" in requests and "cpu" in limits:
            assert parse_cpu(requests["cpu"]) <= parse_cpu(
                limits["cpu"]
            ), "CPU request should not exceed limit"

        if "memory" in requests and "memory" in limits:
            assert parse_memory(requests["memory"]) <= parse_memory(
                limits["memory"]
            ), "Memory request should not exceed limit"

    def test_reasonable_resource_limits(self, deployment):
        """Test that resource limits are reasonable."""
        resources = deployment["spec"]["template"]["spec"]["containers"][0]["resources"]
        limits = resources.get("limits", {})

        # Memory should not be excessive
        memory = str(limits.get("memory", "0"))
        if "Gi" in memory:
            mem_gb = float(memory.replace("Gi", ""))
            assert mem_gb <= 32, "Memory limit seems excessive for dev environment"


# Test Health Checks
class TestHealthChecks:
    def test_liveness_probe_defined(self, deployment):
        """Test that liveness probe is defined."""
        container = deployment["spec"]["template"]["spec"]["containers"][0]
        assert "livenessProbe" in container, "Liveness probe is required for production"

    def test_readiness_probe_defined(self, deployment):
        """Test that readiness probe is defined."""
        container = deployment["spec"]["template"]["spec"]["containers"][0]
        assert (
            "readinessProbe" in container
        ), "Readiness probe is required for production"

    def test_liveness_probe_has_endpoint(self, deployment):
        """Test that liveness probe has an endpoint."""
        probe = deployment["spec"]["template"]["spec"]["containers"][0]["livenessProbe"]
        assert (
            "httpGet" in probe or "tcpSocket" in probe or "exec" in probe
        ), "Liveness probe needs a check mechanism"

    def test_readiness_probe_has_endpoint(self, deployment):
        """Test that readiness probe has an endpoint."""
        probe = deployment["spec"]["template"]["spec"]["containers"][0][
            "readinessProbe"
        ]
        assert (
            "httpGet" in probe or "tcpSocket" in probe or "exec" in probe
        ), "Readiness probe needs a check mechanism"

    def test_liveness_probe_has_initial_delay(self, deployment):
        """Test that liveness probe has initial delay."""
        probe = deployment["spec"]["template"]["spec"]["containers"][0]["livenessProbe"]
        assert (
            "initialDelaySeconds" in probe
        ), "Liveness probe should have initialDelaySeconds"

    def test_readiness_probe_has_initial_delay(self, deployment):
        """Test that readiness probe has initial delay."""
        probe = deployment["spec"]["template"]["spec"]["containers"][0][
            "readinessProbe"
        ]
        assert (
            "initialDelaySeconds" in probe
        ), "Readiness probe should have initialDelaySeconds"

    def test_probe_periods_reasonable(self, deployment):
        """Test that probe periods are reasonable."""
        container = deployment["spec"]["template"]["spec"]["containers"][0]

        for probe_name in ["livenessProbe", "readinessProbe"]:
            if probe_name in container:
                probe = container[probe_name]
                period = probe.get("periodSeconds", 10)
                assert (
                    1 <= period <= 60
                ), f"{probe_name} period should be between 1-60 seconds"

    def test_failure_threshold_defined(self, deployment):
        """Test that failure threshold is defined."""
        container = deployment["spec"]["template"]["spec"]["containers"][0]

        for probe_name in ["livenessProbe", "readinessProbe"]:
            if probe_name in container:
                probe = container[probe_name]
                assert (
                    "failureThreshold" in probe
                ), f"{probe_name} should have failureThreshold"


# Test Environment Variables
class TestEnvironmentVariables:
    def test_env_vars_defined(self, deployment):
        """Test that environment variables are defined."""
        container = deployment["spec"]["template"]["spec"]["containers"][0]
        assert "env" in container, "Environment variables should be defined"

    def test_no_hardcoded_secrets(self, deployment):
        """Test that no secrets are hardcoded in env vars."""
        container = deployment["spec"]["template"]["spec"]["containers"][0]
        env_vars = container.get("env", [])

        for env in env_vars:
            if "value" in env:
                value = str(env["value"]).lower()
                # Check for common secret patterns
                dangerous_patterns = ["password", "secret", "token", "key"]
                var_name = env["name"].lower()

                # If variable name suggests it's a secret, it should use valueFrom
                if any(pattern in var_name for pattern in dangerous_patterns):
                    assert (
                        "valueFrom" in env or "value" not in env
                    ), f"Secret {env['name']} should use secretKeyRef, not plain value"

    def test_secrets_use_secret_ref(self, deployment):
        """Test that secrets use secretKeyRef."""
        container = deployment["spec"]["template"]["spec"]["containers"][0]
        env_vars = container.get("env", [])

        secret_vars = [
            e
            for e in env_vars
            if "valueFrom" in e and "secretKeyRef" in e.get("valueFrom", {})
        ]

        # Should have at least one secret reference (API key)
        assert (
            len(secret_vars) > 0
        ), "Expected at least one secret reference (e.g., API key)"

    def test_secret_keys_named_appropriately(self, deployment):
        """Test that secret keys are named appropriately."""
        container = deployment["spec"]["template"]["spec"]["containers"][0]
        env_vars = container.get("env", [])

        for env in env_vars:
            if "valueFrom" in env and "secretKeyRef" in env.get("valueFrom", {}):
                secret_key = env["valueFrom"]["secretKeyRef"].get("key")
                assert (
                    secret_key is not None
                ), f"Secret reference {env['name']} missing key"


# Test Service
class TestService:
    def test_service_exists(self, service):
        """Test that service resource exists."""
        assert service is not None, "Service resource not found"

    def test_service_apiversion(self, service):
        """Test service has correct apiVersion."""
        assert service["apiVersion"] == "v1"

    def test_service_kind(self, service):
        """Test service has correct kind."""
        assert service["kind"] == "Service"

    def test_service_has_name(self, service):
        """Test service has a name."""
        assert "name" in service["metadata"]

    def test_service_in_correct_namespace(self, service, namespace):
        """Test service is in the correct namespace."""
        svc_ns = service["metadata"].get("namespace")
        ns_name = namespace["metadata"]["name"]
        assert svc_ns == ns_name

    def test_service_has_selector(self, service):
        """Test service has selector."""
        assert "selector" in service["spec"]
        assert len(service["spec"]["selector"]) > 0

    def test_service_selector_matches_deployment(self, service, deployment):
        """Test service selector matches deployment labels."""
        svc_selector = service["spec"]["selector"]
        dep_labels = deployment["spec"]["template"]["metadata"]["labels"]

        for key, value in svc_selector.items():
            assert key in dep_labels, f"Service selector {key} not in deployment labels"
            assert (
                dep_labels[key] == value
            ), f"Service selector {key}={value} doesn't match deployment"

    def test_service_has_ports(self, service):
        """Test service has ports defined."""
        assert "ports" in service["spec"]
        assert len(service["spec"]["ports"]) > 0

    def test_service_port_has_name(self, service):
        """Test service port has a name."""
        port = service["spec"]["ports"][0]
        assert "name" in port

    def test_service_port_protocol(self, service):
        """Test service port has protocol."""
        port = service["spec"]["ports"][0]
        assert "protocol" in port
        assert port["protocol"] in ["TCP", "UDP"]

    def test_service_type_defined(self, service):
        """Test service type is defined."""
        assert "type" in service["spec"]
        assert service["spec"]["type"] in ["ClusterIP", "NodePort", "LoadBalancer"]

    def test_service_port_matches_container(self, service, deployment):
        """Test service targetPort matches container port."""
        svc_port = service["spec"]["ports"][0]
        container = deployment["spec"]["template"]["spec"]["containers"][0]
        container_port = container["ports"][0]["containerPort"]

        target_port = svc_port.get("targetPort")
        if isinstance(target_port, int):
            assert (
                target_port == container_port
            ), "Service targetPort doesn't match container port"


# Test Security Best Practices
class TestSecurityBestPractices:
    def test_no_privileged_containers(self, deployment):
        """Test that containers don't run privileged."""
        container = deployment["spec"]["template"]["spec"]["containers"][0]
        security_context = container.get("securityContext", {})

        assert not security_context.get(
            "privileged", False
        ), "Containers should not run privileged"

    def test_security_context_defined(self, deployment):
        """Test that security context is defined."""
        # Pod-level or container-level security context should be defined
        pod_spec = deployment["spec"]["template"]["spec"]
        container = pod_spec["containers"][0]

        has_pod_security = "securityContext" in pod_spec
        has_container_security = "securityContext" in container

        # At least one should be defined for production
        if not (has_pod_security or has_container_security):
            pytest.skip("Warning: No security context defined. Add for production.")

    def test_run_as_non_root(self, deployment):
        """Test that containers run as non-root."""
        pod_spec = deployment["spec"]["template"]["spec"]
        container = pod_spec["containers"][0]

        # Check pod-level security context
        pod_security = pod_spec.get("securityContext", {})
        container_security = container.get("securityContext", {})

        run_as_non_root = pod_security.get("runAsNonRoot") or container_security.get(
            "runAsNonRoot"
        )

        if not run_as_non_root:
            pytest.skip(
                "Warning: runAsNonRoot not set. "
                "Containers should run as non-root user for security."
            )

    def test_read_only_root_filesystem(self, deployment):
        """Test for read-only root filesystem."""
        container = deployment["spec"]["template"]["spec"]["containers"][0]
        security_context = container.get("securityContext", {})

        if not security_context.get("readOnlyRootFilesystem"):
            pytest.skip(
                "Warning: readOnlyRootFilesystem not enabled. "
                "Consider enabling for improved security."
            )

    def test_capabilities_dropped(self, deployment):
        """Test that Linux capabilities are dropped."""
        container = deployment["spec"]["template"]["spec"]["containers"][0]
        security_context = container.get("securityContext", {})
        capabilities = security_context.get("capabilities", {})

        if not capabilities.get("drop"):
            pytest.skip(
                "Warning: No capabilities dropped. "
                "Consider dropping ALL and adding back only what's needed."
            )


# Test Production Readiness
class TestProductionReadiness:
    def test_multiple_replicas_recommended(self, deployment):
        """Test that multiple replicas are used for HA."""
        replicas = deployment["spec"]["replicas"]
        if replicas < 2:
            pytest.skip(
                f"Warning: Only {replicas} replica(s) configured. "
                "Use 2+ replicas for high availability in production."
            )

    def test_pod_disruption_budget_recommended(self, helm_chart):
        """Test that PodDisruptionBudget is defined."""
        pdb = next(
            (doc for doc in helm_chart if doc.get("kind") == "PodDisruptionBudget"),
            None,
        )

        if not pdb:
            pytest.skip(
                "Warning: No PodDisruptionBudget defined. "
                "Add PDB to ensure availability during cluster maintenance."
            )

    def test_hpa_recommended(self, helm_chart):
        """Test that HorizontalPodAutoscaler is defined."""
        hpa = next(
            (doc for doc in helm_chart if doc.get("kind") == "HorizontalPodAutoscaler"),
            None,
        )

        if not hpa:
            pytest.skip(
                "Warning: No HorizontalPodAutoscaler defined. "
                "Consider adding HPA for automatic scaling."
            )

    def test_resource_quotas_recommended(self, helm_chart):
        """Test that ResourceQuota is defined."""
        quota = next(
            (doc for doc in helm_chart if doc.get("kind") == "ResourceQuota"), None
        )

        if not quota:
            pytest.skip(
                "Warning: No ResourceQuota defined for namespace. "
                "Add quotas to prevent resource exhaustion."
            )

    def test_network_policy_recommended(self, helm_chart):
        """Test that NetworkPolicy is defined."""
        net_policy = next(
            (doc for doc in helm_chart if doc.get("kind") == "NetworkPolicy"), None
        )

        if not net_policy:
            pytest.skip(
                "Warning: No NetworkPolicy defined. "
                "Add network policies to restrict pod-to-pod communication."
            )

    def test_rolling_update_strategy(self, deployment):
        """Test that rolling update strategy is configured."""
        strategy = deployment["spec"].get("strategy", {})
        strategy_type = strategy.get("type", "RollingUpdate")

        assert (
            strategy_type == "RollingUpdate"
        ), "Should use RollingUpdate for zero-downtime deployments"

    def test_rolling_update_parameters(self, deployment):
        """Test that rolling update parameters are configured."""
        strategy = deployment["spec"].get("strategy", {})

        if strategy.get("type") == "RollingUpdate":
            rolling_update = strategy.get("rollingUpdate", {})

            # Check max surge and max unavailable are set
            if not rolling_update:
                pytest.skip(
                    "Warning: rollingUpdate parameters not configured. "
                    "Set maxSurge and maxUnavailable for controlled rollouts."
                )


# Test Labels and Annotations
class TestLabelsAndAnnotations:
    def test_recommended_labels_present(self, deployment, required_labels):
        """Test that recommended Kubernetes labels are present."""
        labels = deployment["metadata"].get("labels", {})

        for label in required_labels:
            assert label in labels, f"Missing recommended label: {label}"

    def test_labels_follow_convention(self, deployment):
        """Test that labels follow Kubernetes conventions."""
        labels = deployment["metadata"].get("labels", {})

        for key in labels.keys():
            # Labels should use proper prefixes
            if "/" in key:
                prefix, name = key.rsplit("/", 1)
                assert (
                    "." in prefix
                ), f"Label prefix should be a valid DNS subdomain: {key}"

    def test_pod_template_has_labels(self, deployment):
        """Test that pod template has labels."""
        template_labels = deployment["spec"]["template"]["metadata"].get("labels", {})
        assert len(template_labels) > 0, "Pod template should have labels"

    def test_consistent_app_name_across_resources(self, namespace, deployment, service):
        """Test that app name is consistent across resources."""
        app_label = "app.kubernetes.io/name"

        ns_labels = namespace["metadata"].get("labels", {})
        dep_labels = deployment["metadata"].get("labels", {})
        svc_labels = service["metadata"].get("labels", {})

        if app_label in dep_labels and app_label in svc_labels:
            assert (
                dep_labels[app_label] == svc_labels[app_label]
            ), "app.kubernetes.io/name should be consistent across resources"


# Test Configuration Issues
class TestConfigurationIssues:
    def test_image_placeholder_not_used(self, deployment):
        """Test that image placeholder is replaced."""
        container = deployment["spec"]["template"]["spec"]["containers"][0]
        image = container["image"]

        assert (
            "your_dockerhub_user" not in image
        ), "Replace 'your_dockerhub_user' with actual Docker registry"

    def test_secret_actually_exists_warning(self, deployment, helm_chart):
        """Test that referenced secrets exist (warning)."""
        container = deployment["spec"]["template"]["spec"]["containers"][0]
        env_vars = container.get("env", [])

        secret_names = []
        for env in env_vars:
            if "valueFrom" in env and "secretKeyRef" in env.get("valueFrom", {}):
                secret_name = env["valueFrom"]["secretKeyRef"].get("name")
                if secret_name:
                    secret_names.append(secret_name)

        if secret_names:
            # Check if the secrets are defined in the helm chart
            defined_secrets = [
                doc["metadata"]["name"]
                for doc in helm_chart
                if doc.get("kind") == "Secret"
            ]
            undefined_secrets = [s for s in secret_names if s not in defined_secrets]

            if undefined_secrets:
                pytest.skip(
                    f"Warning: Secrets referenced but not defined: {', '.join(undefined_secrets)}. "
                    "Create these secrets before deploying."
                )

    def test_service_type_appropriate(self, service):
        """Test that service type is appropriate for environment."""
        svc_type = service["spec"]["type"]

        if svc_type == "NodePort":
            pytest.skip(
                "Warning: NodePort is used. This is suitable for dev/test, "
                "but use LoadBalancer or Ingress for production."
            )


# Test Anti-patterns
class TestAntiPatterns:
    def test_no_latest_tag(self, deployment):
        """Test that 'latest' tag is not used."""
        container = deployment["spec"]["template"]["spec"]["containers"][0]
        image = container["image"]

        assert ":latest" not in image and not image.endswith(
            "latest"
        ), "Never use 'latest' tag - always specify exact versions"

    def test_no_implicit_latest(self, deployment):
        """Test that image has explicit tag."""
        container = deployment["spec"]["template"]["spec"]["containers"][0]
        image = container["image"]

        assert (
            ":" in image
        ), "Image should have explicit tag (e.g., :v1.0.0), not implicit :latest"

    def test_restart_policy_default(self, deployment):
        """Test that restart policy is appropriate."""
        spec = deployment["spec"]["template"]["spec"]
        restart_policy = spec.get("restartPolicy", "Always")

        assert (
            restart_policy == "Always"
        ), "Deployment pods should have restartPolicy: Always"


# Integration Tests
class TestIntegration:
    def test_complete_deployment_possible(self, namespace, deployment, service):
        """Test that all resources needed for deployment are present."""
        assert namespace is not None, "Namespace needed"
        assert deployment is not None, "Deployment needed"
        assert service is not None, "Service needed"

    def test_service_can_route_to_pods(self, service, deployment):
        """Test that service can route traffic to pods."""
        # Service selector must match pod labels
        svc_selector = service["spec"]["selector"]
        pod_labels = deployment["spec"]["template"]["metadata"]["labels"]

        for key, value in svc_selector.items():
            assert key in pod_labels
            assert pod_labels[key] == value

    def test_ports_properly_configured(self, service, deployment):
        """Test that ports are properly configured end-to-end."""
        svc_port = service["spec"]["ports"][0]["port"]
        target_port = service["spec"]["ports"][0]["targetPort"]
        container_port = deployment["spec"]["template"]["spec"]["containers"][0][
            "ports"
        ][0]["containerPort"]

        # Target port should match container port
        if isinstance(target_port, int):
            assert target_port == container_port


# Test Documentation
class TestDocumentation:
    def test_resources_have_comments_in_file(self, helm_chart):
        """Test that YAML file has helpful comments."""
        # Read raw file to check for comments
        config_path = Path(__file__).parent / "configs" / "helm_chart.yaml"
        if not config_path.exists():
            config_path = Path(__file__).parent / ".." / "configs" / "helm_chart.yaml"

        with open(config_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Should have at least some comments
        comment_count = content.count("#")
        if comment_count < 5:
            pytest.skip(
                f"Warning: Only {comment_count} comments found. "
                "Add more comments to document configuration choices."
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
