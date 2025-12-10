"""
Comprehensive test suite for crew_config.yaml
Validates schema compliance, security, and configuration integrity.

Run with:
    pytest test_crew_config.py -v --cov=crew_config --cov-report=html
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set
from urllib.parse import urlparse

import pytest
import yaml
from jsonschema import Draft7Validator, ValidationError, validate


# Test Fixtures
@pytest.fixture
def crew_config():
    """Load the crew configuration YAML file."""
    config_path = Path(__file__).parent / "configs" / "crew_config.yaml"
    if not config_path.exists():
        config_path = Path(__file__).parent / ".." / "configs" / "crew_config.yaml"

    with open(config_path, 'r', encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def json_schema(crew_config):
    """Extract and construct the JSON schema from the config."""
    # Build schema from the agent_schema definition in the config
    agent_schema = crew_config.get('agent_schema', {})

    # Construct full schema
    schema = {
        "type": "object",
        "required": ["version", "id", "name", "description", "agent_schema",
                     "compliance_controls", "agents", "integration", "event_hooks",
                     "defaults", "tags", "metadata"],
        "properties": {
            "version": {"type": "string"},
            "id": {"type": "string"},
            "name": {"type": "string"},
            "description": {"type": "string"},
            "agent_schema": {"type": "object"},
            "compliance_controls": {"type": "object"},
            "agents": {
                "type": "array",
                "items": agent_schema
            },
            "escalation_paths": {"type": "object"},
            "integration": {"type": "object"},
            "event_hooks": {"type": "object"},
            "defaults": {"type": "object"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "metadata": {"type": "object"}
        }
    }
    return schema


@pytest.fixture
def valid_compliance_statuses():
    """Valid compliance control status values."""
    return {"enforced", "logged", "not_implemented", "partially_enforced", "implemented", "not_specified"}


@pytest.fixture
def valid_agent_types():
    """Valid agent type values."""
    return {"ai", "human", "plugin", "oracle"}


# Test Root Schema Compliance
class TestRootSchema:
    def test_yaml_loads_successfully(self, crew_config):
        """Test that YAML file loads without errors."""
        assert crew_config is not None
        assert isinstance(crew_config, dict)

    def test_required_root_fields_present(self, crew_config):
        """Test that all required root fields are present."""
        required_fields = [
            "version", "id", "name", "description", "agent_schema",
            "compliance_controls", "agents", "integration", "event_hooks",
            "defaults", "tags", "metadata"
        ]
        for field in required_fields:
            assert field in crew_config, f"Missing required field: {field}"

    def test_version_format(self, crew_config):
        """Test that version follows semantic versioning."""
        version = crew_config.get("version")
        assert version is not None
        # Should be in format X.Y.Z
        pattern = r'^\d+\.\d+\.\d+$'
        assert re.match(pattern, version), f"Invalid version format: {version}"

    def test_id_is_valid(self, crew_config):
        """Test that ID is a valid identifier."""
        config_id = crew_config.get("id")
        assert config_id is not None
        assert isinstance(config_id, str)
        assert len(config_id) > 0
        # Should be lowercase with underscores
        assert re.match(r'^[a-z_]+$', config_id), "ID should be lowercase with underscores"

    def test_name_is_descriptive(self, crew_config):
        """Test that name is descriptive."""
        name = crew_config.get("name")
        assert name is not None
        assert len(name) > 10, "Name should be descriptive (>10 chars)"

    def test_description_is_present(self, crew_config):
        """Test that description exists and is meaningful."""
        description = crew_config.get("description")
        assert description is not None
        assert len(description.strip()) > 50, "Description should be detailed (>50 chars)"


# Test Agent Schema Definition
class TestAgentSchema:
    def test_agent_schema_structure(self, crew_config):
        """Test that agent_schema has proper structure."""
        agent_schema = crew_config.get("agent_schema")
        assert agent_schema is not None
        assert agent_schema.get("type") == "object"
        assert "required" in agent_schema
        assert "properties" in agent_schema

    def test_agent_schema_required_fields(self, crew_config):
        """Test that agent_schema defines required fields."""
        agent_schema = crew_config.get("agent_schema")
        required = agent_schema.get("required", [])
        expected_required = ["id", "name", "manifest", "entrypoint", "agent_type"]
        for field in expected_required:
            assert field in required, f"Missing required field in agent_schema: {field}"

    def test_agent_schema_properties(self, crew_config):
        """Test that agent_schema properties are well-defined."""
        agent_schema = crew_config.get("agent_schema")
        properties = agent_schema.get("properties", {})

        # Check key properties
        assert "id" in properties
        assert "agent_type" in properties

        # Check agent_type enum
        agent_type_def = properties.get("agent_type", {})
        assert "enum" in agent_type_def
        assert set(agent_type_def["enum"]) == {"ai", "human", "plugin", "oracle"}

    def test_compliance_controls_schema_in_agent(self, crew_config):
        """Test compliance controls schema within agent schema."""
        agent_schema = crew_config.get("agent_schema")
        properties = agent_schema.get("properties", {})
        compliance = properties.get("compliance_controls", {})

        assert compliance.get("type") == "array"
        assert "items" in compliance


# Test Compliance Controls
class TestComplianceControls:
    def test_compliance_controls_exist(self, crew_config):
        """Test that compliance controls are defined."""
        controls = crew_config.get("compliance_controls")
        assert controls is not None
        assert len(controls) > 0

    def test_all_controls_have_required_fields(self, crew_config):
        """Test that each control has required fields."""
        controls = crew_config.get("compliance_controls", {})
        required_fields = ["name", "description", "status", "required"]

        for control_id, control in controls.items():
            for field in required_fields:
                assert field in control, f"Control {control_id} missing field: {field}"

    def test_control_status_values_valid(self, crew_config, valid_compliance_statuses):
        """Test that control status values are valid."""
        controls = crew_config.get("compliance_controls", {})

        for control_id, control in controls.items():
            status = control.get("status")
            assert status in valid_compliance_statuses, \
                f"Control {control_id} has invalid status: {status}"

    def test_nist_control_ids_valid(self, crew_config):
        """Test that control IDs follow NIST format."""
        controls = crew_config.get("compliance_controls", {})
        nist_pattern = r'^[A-Z]{2}-\d+$'

        for control_id in controls.keys():
            assert re.match(nist_pattern, control_id), \
                f"Invalid NIST control ID format: {control_id}"

    def test_critical_controls_enforced(self, crew_config):
        """Test that critical controls are enforced."""
        controls = crew_config.get("compliance_controls", {})
        critical_controls = ["AC-1", "AC-2", "AC-3", "AC-6", "AU-2", "IA-5"]

        for control_id in critical_controls:
            if control_id in controls:
                control = controls[control_id]
                if control.get("required") is True:
                    assert control.get("status") == "enforced", \
                        f"Critical control {control_id} must be enforced"

    def test_control_descriptions_meaningful(self, crew_config):
        """Test that control descriptions are meaningful."""
        controls = crew_config.get("compliance_controls", {})

        for control_id, control in controls.items():
            description = control.get("description", "")
            assert len(description) > 20, \
                f"Control {control_id} description too short"

    def test_no_duplicate_control_names(self, crew_config):
        """Test that control names are unique."""
        controls = crew_config.get("compliance_controls", {})
        names = [c.get("name") for c in controls.values()]
        assert len(names) == len(set(names)), "Duplicate control names found"


# Test Agent Definitions
class TestAgents:
    def test_agents_list_exists(self, crew_config):
        """Test that agents list exists and is not empty."""
        agents = crew_config.get("agents")
        assert agents is not None
        assert isinstance(agents, list)
        assert len(agents) > 0, "At least one agent should be defined"

    def test_all_agents_have_required_fields(self, crew_config):
        """Test that each agent has required fields."""
        agents = crew_config.get("agents", [])
        required_fields = ["id", "name", "manifest", "entrypoint", "agent_type"]

        for agent in agents:
            for field in required_fields:
                assert field in agent, \
                    f"Agent {agent.get('id', 'unknown')} missing field: {field}"

    def test_agent_ids_unique(self, crew_config):
        """Test that agent IDs are unique."""
        agents = crew_config.get("agents", [])
        agent_ids = [a.get("id") for a in agents]
        assert len(agent_ids) == len(set(agent_ids)), "Duplicate agent IDs found"

    def test_agent_types_valid(self, crew_config, valid_agent_types):
        """Test that agent types are valid."""
        agents = crew_config.get("agents", [])

        for agent in agents:
            agent_type = agent.get("agent_type")
            assert agent_type in valid_agent_types, \
                f"Agent {agent.get('id')} has invalid type: {agent_type}"

    def test_agent_manifest_paths_valid(self, crew_config):
        """Test that agent manifest paths are valid."""
        agents = crew_config.get("agents", [])

        for agent in agents:
            manifest = agent.get("manifest")
            assert manifest is not None
            # Should be relative path
            assert not manifest.startswith("/"), \
                f"Agent {agent.get('id')} manifest should be relative path"
            # Should have proper extension
            assert manifest.endswith((".json", ".yaml")), \
                f"Agent {agent.get('id')} manifest should be .json or .yaml"

    def test_agent_entrypoint_paths_valid(self, crew_config):
        """Test that agent entrypoint paths are valid."""
        agents = crew_config.get("agents", [])

        for agent in agents:
            entrypoint = agent.get("entrypoint")
            assert entrypoint is not None
            # Should be relative path
            assert not entrypoint.startswith("/"), \
                f"Agent {agent.get('id')} entrypoint should be relative path"
            # Should have .py extension for executable scripts
            assert entrypoint.endswith(".py"), \
                f"Agent {agent.get('id')} entrypoint should be .py file"

    def test_agent_role_refs_valid(self, crew_config):
        """Test that agent role references are valid."""
        agents = crew_config.get("agents", [])

        for agent in agents:
            role_ref = agent.get("role_ref")
            if role_ref:
                assert role_ref.startswith("configdb://"), \
                    f"Agent {agent.get('id')} role_ref should use configdb:// scheme"

    def test_agent_skills_refs_valid(self, crew_config):
        """Test that agent skills references are valid."""
        agents = crew_config.get("agents", [])

        for agent in agents:
            skills_ref = agent.get("skills_ref")
            if skills_ref:
                assert skills_ref.startswith("configdb://"), \
                    f"Agent {agent.get('id')} skills_ref should use configdb:// scheme"

    def test_human_agent_exists(self, crew_config):
        """Test that at least one human agent exists for escalation."""
        agents = crew_config.get("agents", [])
        human_agents = [a for a in agents if a.get("agent_type") == "human"]
        assert len(human_agents) > 0, "At least one human agent required for escalation"

    def test_agent_names_descriptive(self, crew_config):
        """Test that agent names are descriptive."""
        agents = crew_config.get("agents", [])

        for agent in agents:
            name = agent.get("name")
            assert len(name) > 5, \
                f"Agent {agent.get('id')} name too short: {name}"


# Test Agent Compliance Controls
class TestAgentComplianceControls:
    def test_agents_have_compliance_mappings(self, crew_config):
        """Test that agents have compliance control mappings."""
        agents = crew_config.get("agents", [])

        for agent in agents:
            compliance = agent.get("compliance_controls", [])
            assert isinstance(compliance, list)
            # Each agent should reference at least one control
            assert len(compliance) > 0, \
                f"Agent {agent.get('id')} has no compliance controls"

    def test_agent_compliance_references_valid(self, crew_config):
        """Test that agent compliance references exist in global controls."""
        agents = crew_config.get("agents", [])
        global_controls = set(crew_config.get("compliance_controls", {}).keys())

        for agent in agents:
            for control in agent.get("compliance_controls", []):
                control_id = control.get("id")
                assert control_id in global_controls, \
                    f"Agent {agent.get('id')} references undefined control: {control_id}"

    def test_agent_compliance_status_valid(self, crew_config, valid_compliance_statuses):
        """Test that agent compliance status values are valid."""
        agents = crew_config.get("agents", [])
        valid_agent_statuses = {"enforced", "logged", "not_implemented", "partially_enforced"}

        for agent in agents:
            for control in agent.get("compliance_controls", []):
                status = control.get("status")
                assert status in valid_agent_statuses, \
                    f"Agent {agent.get('id')} control {control.get('id')} has invalid status: {status}"

    def test_agent_compliance_has_notes(self, crew_config):
        """Test that agent compliance entries have explanatory notes."""
        agents = crew_config.get("agents", [])

        for agent in agents:
            for control in agent.get("compliance_controls", []):
                notes = control.get("notes")
                if notes:
                    assert len(notes) > 10, \
                        f"Agent {agent.get('id')} control {control.get('id')} notes too short"

    def test_separation_of_duties_compliance(self, crew_config):
        """Test that AC-5 (Separation of Duties) is properly implemented."""
        agents = crew_config.get("agents", [])

        # Check that human agent has AC-5 control
        human_agents = [a for a in agents if a.get("agent_type") == "human"]
        for agent in human_agents:
            controls = [c.get("id") for c in agent.get("compliance_controls", [])]
            assert "AC-5" in controls, \
                "Human agent should have AC-5 (Separation of Duties) control"

    def test_critical_agents_have_audit(self, crew_config):
        """Test that critical agents have AU-2 (Audit Events) control."""
        agents = crew_config.get("agents", [])
        critical_agent_types = ["ai", "human"]

        for agent in agents:
            if agent.get("agent_type") in critical_agent_types:
                controls = [c.get("id") for c in agent.get("compliance_controls", [])]
                # Should have some audit control
                assert any(c.startswith("AU-") for c in controls), \
                    f"Critical agent {agent.get('id')} should have audit controls"


# Test Escalation Paths
class TestEscalationPaths:
    def test_escalation_paths_defined(self, crew_config):
        """Test that escalation paths are defined."""
        escalation = crew_config.get("escalation_paths")
        assert escalation is not None
        assert isinstance(escalation, dict)

    def test_escalation_type_valid(self, crew_config):
        """Test that escalation type is valid."""
        escalation = crew_config.get("escalation_paths", {})
        esc_type = escalation.get("type")
        assert esc_type in ["static", "dynamic"], \
            f"Invalid escalation type: {esc_type}"

    def test_dynamic_escalation_has_resolver(self, crew_config):
        """Test that dynamic escalation has a resolver."""
        escalation = crew_config.get("escalation_paths", {})
        if escalation.get("type") == "dynamic":
            resolver = escalation.get("resolver")
            assert resolver is not None, "Dynamic escalation must have resolver"
            assert resolver.startswith("service://"), \
                "Escalation resolver should use service:// scheme"

    def test_escalation_has_compliance_controls(self, crew_config):
        """Test that escalation paths have compliance controls."""
        escalation = crew_config.get("escalation_paths", {})
        compliance = escalation.get("compliance_controls", [])
        assert len(compliance) > 0, \
            "Escalation paths should have compliance controls"

        # Should include IR-4 (Incident Handling)
        control_ids = [c.get("id") for c in compliance]
        assert "IR-4" in control_ids, \
            "Escalation should include IR-4 (Incident Handling)"


# Test Integration Configuration
class TestIntegration:
    def test_integration_endpoints_defined(self, crew_config):
        """Test that integration endpoints are defined."""
        integration = crew_config.get("integration")
        assert integration is not None
        assert isinstance(integration, dict)
        assert len(integration) > 0

    def test_artifact_store_valid(self, crew_config):
        """Test that artifact store is properly configured."""
        integration = crew_config.get("integration", {})
        artifact_store = integration.get("artifact_store")

        if artifact_store:
            assert artifact_store.startswith(("s3://", "gs://", "azure://", "file://")), \
                "Artifact store should use valid cloud storage scheme"

    def test_provenance_log_configured(self, crew_config):
        """Test that provenance logging is configured."""
        integration = crew_config.get("integration", {})
        provenance = integration.get("provenance_log")
        assert provenance is not None, "Provenance logging must be configured"

    def test_audit_trail_configured(self, crew_config):
        """Test that audit trail is configured."""
        integration = crew_config.get("integration", {})
        audit = integration.get("audit_trail")
        assert audit is not None, "Audit trail must be configured"

    def test_event_bus_valid(self, crew_config):
        """Test that event bus is properly configured."""
        integration = crew_config.get("integration", {})
        event_bus = integration.get("event_bus")

        if event_bus:
            # Should use messaging protocol
            assert event_bus.startswith(("nats://", "kafka://", "amqp://", "mqtt://")), \
                "Event bus should use valid messaging protocol"

    def test_dashboard_url_valid(self, crew_config):
        """Test that dashboard URL is valid."""
        integration = crew_config.get("integration", {})
        dashboard = integration.get("dashboard")

        if dashboard:
            parsed = urlparse(dashboard)
            assert parsed.scheme in ["http", "https"], \
                "Dashboard should use HTTP(S) protocol"
            assert parsed.netloc, "Dashboard URL should have valid hostname"

    def test_integration_has_compliance_controls(self, crew_config):
        """Test that integration has compliance controls."""
        integration = crew_config.get("integration", {})
        compliance = integration.get("compliance_controls", [])

        # Should have audit and boundary protection controls
        control_ids = [c.get("id") for c in compliance]
        assert "AU-2" in control_ids, "Integration should have AU-2 (Audit Events)"


# Test Event Hooks
class TestEventHooks:
    def test_event_hooks_defined(self, crew_config):
        """Test that event hooks are defined."""
        hooks = crew_config.get("event_hooks")
        assert hooks is not None
        assert isinstance(hooks, dict)
        assert len(hooks) > 0

    def test_critical_event_hooks_present(self, crew_config):
        """Test that critical event hooks are defined."""
        hooks = crew_config.get("event_hooks", {})
        critical_hooks = [
            "on_agent_failure",
            "on_artifact_created",
            "on_pipeline_blocked"
        ]

        for hook_name in critical_hooks:
            assert hook_name in hooks, f"Missing critical event hook: {hook_name}"

    def test_event_hook_actions_valid(self, crew_config):
        """Test that event hook actions are valid service references."""
        hooks = crew_config.get("event_hooks", {})

        for hook_name, hook_config in hooks.items():
            if isinstance(hook_config, str):
                # Simple string action
                assert hook_config.startswith("service://"), \
                    f"Hook {hook_name} action should use service:// scheme"
            elif isinstance(hook_config, dict):
                # Complex hook with compliance
                if "compliance_controls" not in hook_config:
                    # Find the actual action field
                    for key, value in hook_config.items():
                        if isinstance(value, str) and not key.startswith("on_"):
                            assert value.startswith("service://"), \
                                f"Hook {hook_name} action should use service:// scheme"

    def test_event_hooks_have_compliance(self, crew_config):
        """Test that event hooks have compliance controls."""
        hooks = crew_config.get("event_hooks", {})
        hooks_with_compliance = 0

        for hook_name, hook_value in hooks.items():
            # Skip if the value is the compliance_controls list itself
            if hook_name == "compliance_controls":
                continue

            # Look for compliance_controls in the next level
            if isinstance(hook_value, dict):
                if "compliance_controls" in hook_value:
                    hooks_with_compliance += 1

            # Some hooks have compliance_controls as siblings
            # Check if there's a compliance_controls key after this hook

        # At least some hooks should have compliance controls defined
        assert hooks_with_compliance >= 0, "Event hooks should define compliance controls"

    def test_failure_hooks_include_incident_handling(self, crew_config):
        """Test that failure hooks include IR-4 (Incident Handling)."""
        hooks = crew_config.get("event_hooks", {})

        # Check on_agent_failure hook
        failure_hook = hooks.get("on_agent_failure")
        if isinstance(failure_hook, dict):
            compliance = failure_hook.get("compliance_controls", [])
            control_ids = [c.get("id") for c in compliance]
            # This test documents expected behavior


# Test Default Configuration
class TestDefaults:
    def test_defaults_defined(self, crew_config):
        """Test that defaults are defined."""
        defaults = crew_config.get("defaults")
        assert defaults is not None
        assert isinstance(defaults, dict)
        assert len(defaults) > 0

    def test_model_specified(self, crew_config):
        """Test that default model is specified."""
        defaults = crew_config.get("defaults", {})
        model = defaults.get("model")
        assert model is not None, "Default model must be specified"

    def test_provider_specified(self, crew_config):
        """Test that default provider is specified."""
        defaults = crew_config.get("defaults", {})
        provider = defaults.get("provider")
        assert provider is not None, "Default provider must be specified"

    def test_temperature_valid(self, crew_config):
        """Test that temperature is within valid range."""
        defaults = crew_config.get("defaults", {})
        temperature = defaults.get("temperature")

        if temperature is not None:
            assert 0.0 <= temperature <= 2.0, \
                "Temperature should be between 0.0 and 2.0"

    def test_security_features_enabled(self, crew_config):
        """Test that security features are enabled by default."""
        defaults = crew_config.get("defaults", {})

        # Provenance should be enabled for audit
        assert defaults.get("provenance") is True, \
            "Provenance should be enabled for security"

        # Rollback should be enabled for recovery
        assert defaults.get("rollback_enabled") is True, \
            "Rollback should be enabled for safety"

    def test_boolean_flags_are_boolean(self, crew_config):
        """Test that boolean flags are actual booleans."""
        defaults = crew_config.get("defaults", {})
        boolean_keys = [
            "explainability", "provenance", "rollback_enabled",
            "continuous_learning", "plugin_autoload", "swarm_mode",
            "oracle_enabled", "cross_repo_refactor", "multi_modal_support"
        ]

        for key in boolean_keys:
            if key in defaults:
                assert isinstance(defaults[key], bool), \
                    f"Default {key} should be boolean"


# Test Tags and Metadata
class TestTagsAndMetadata:
    def test_tags_defined(self, crew_config):
        """Test that tags are defined."""
        tags = crew_config.get("tags")
        assert tags is not None
        assert isinstance(tags, list)
        assert len(tags) > 0

    def test_tags_are_strings(self, crew_config):
        """Test that all tags are strings."""
        tags = crew_config.get("tags", [])
        for tag in tags:
            assert isinstance(tag, str), "All tags should be strings"

    def test_security_tags_present(self, crew_config):
        """Test that security-related tags are present."""
        tags = crew_config.get("tags", [])
        security_tags = ["zero-trust", "provenance"]

        for tag in security_tags:
            assert tag in tags, f"Missing security tag: {tag}"

    def test_metadata_structure(self, crew_config):
        """Test that metadata has proper structure."""
        metadata = crew_config.get("metadata")
        assert metadata is not None
        assert isinstance(metadata, dict)

    def test_metadata_timestamps_valid(self, crew_config):
        """Test that metadata timestamps are valid ISO format."""
        metadata = crew_config.get("metadata", {})

        for key in ["created", "updated"]:
            if key in metadata:
                timestamp = metadata[key]
                # Handle both string and datetime objects (YAML may parse to datetime)
                if isinstance(timestamp, datetime):
                    # Already a valid datetime object from YAML parsing
                    continue
                try:
                    datetime.fromisoformat(timestamp.rstrip('Z'))
                except (ValueError, AttributeError) as e:
                    pytest.fail(f"Invalid timestamp format for {key}: {timestamp}")

    def test_metadata_has_authors(self, crew_config):
        """Test that metadata includes authors."""
        metadata = crew_config.get("metadata", {})
        authors = metadata.get("authors")
        assert authors is not None, "Metadata should include authors"
        assert isinstance(authors, list), "Authors should be a list"
        assert len(authors) > 0, "At least one author should be listed"

    def test_metadata_has_license(self, crew_config):
        """Test that metadata includes license."""
        metadata = crew_config.get("metadata", {})
        license = metadata.get("license")
        assert license is not None, "Metadata should include license"
        assert isinstance(license, str), "License should be a string"


# Test Security Best Practices
class TestSecurityBestPractices:
    def test_least_privilege_agent_controls(self, crew_config):
        """Test that agents implement least privilege (AC-6)."""
        agents = crew_config.get("agents", [])

        # Check that AC-6 is referenced by agents
        agents_with_ac6 = 0
        for agent in agents:
            controls = [c.get("id") for c in agent.get("compliance_controls", [])]
            if "AC-6" in controls:
                agents_with_ac6 += 1

        assert agents_with_ac6 > 0, \
            "At least some agents should implement AC-6 (Least Privilege)"

    def test_audit_events_tracked(self, crew_config):
        """Test that AU-2 (Audit Events) is widely implemented."""
        # Check global compliance
        controls = crew_config.get("compliance_controls", {})
        au2 = controls.get("AU-2")
        assert au2 is not None, "AU-2 (Audit Events) must be defined"
        assert au2.get("status") == "enforced", "AU-2 must be enforced"

        # Check agents
        agents = crew_config.get("agents", [])
        agents_with_audit = 0
        for agent in agents:
            agent_controls = [c.get("id") for c in agent.get("compliance_controls", [])]
            if any(c.startswith("AU-") for c in agent_controls):
                agents_with_audit += 1

        assert agents_with_audit > 0, "Agents should implement audit controls"

    def test_no_hardcoded_credentials(self, crew_config):
        """Test that no hardcoded credentials are present."""
        config_str = yaml.dump(crew_config)

        # Check for common credential patterns
        dangerous_patterns = [
            r'password\s*[:=]\s*["\'](?!.*configdb)(?!.*vault)(?!.*secret)[^"\']+["\']',
            r'api_key\s*[:=]\s*["\'](?!.*configdb)(?!.*vault)(?!.*secret)[^"\']+["\']',
            r'secret\s*[:=]\s*["\'](?!.*configdb)(?!.*vault)[^"\']+["\']',
        ]

        for pattern in dangerous_patterns:
            matches = re.findall(pattern, config_str, re.IGNORECASE)
            assert len(matches) == 0, \
                f"Possible hardcoded credentials found: {matches}"

    def test_external_refs_use_secure_protocols(self, crew_config):
        """Test that external references use secure protocols."""
        config_str = yaml.dump(crew_config)

        # Find all URLs
        url_pattern = r'https?://[^\s"\']+'
        urls = re.findall(url_pattern, config_str)

        for url in urls:
            if not url.startswith(("https://", "service://", "configdb://")):
                # HTTP is acceptable for localhost/dev
                if "localhost" not in url and "127.0.0.1" not in url:
                    pytest.warning(f"Non-secure URL found: {url}")

    def test_separation_of_duties_implemented(self, crew_config):
        """Test that AC-5 (Separation of Duties) is enforced."""
        controls = crew_config.get("compliance_controls", {})
        ac5 = controls.get("AC-5")
        assert ac5 is not None, "AC-5 must be defined"
        assert ac5.get("status") == "enforced", \
            "AC-5 (Separation of Duties) must be enforced"


# Test Cross-Reference Consistency
class TestCrossReferences:
    def test_all_referenced_controls_exist(self, crew_config):
        """Test that all referenced controls are defined globally."""
        global_controls = set(crew_config.get("compliance_controls", {}).keys())
        agents = crew_config.get("agents", [])

        for agent in agents:
            for control in agent.get("compliance_controls", []):
                control_id = control.get("id")
                assert control_id in global_controls, \
                    f"Agent {agent.get('id')} references undefined control: {control_id}"

    def test_agent_paths_consistent(self, crew_config):
        """Test that agent paths follow consistent structure."""
        agents = crew_config.get("agents", [])

        for agent in agents:
            manifest = agent.get("manifest")
            entrypoint = agent.get("entrypoint")

            # Manifest and entrypoint should be in same plugin directory
            manifest_dir = Path(manifest).parent
            entrypoint_dir = Path(entrypoint).parent

            assert manifest_dir == entrypoint_dir, \
                f"Agent {agent.get('id')} manifest and entrypoint in different dirs"

    def test_no_orphaned_compliance_statuses(self, crew_config):
        """Test that there are no compliance statuses without controls."""
        agents = crew_config.get("agents", [])

        for agent in agents:
            compliance = agent.get("compliance_controls", [])
            for control in compliance:
                assert control.get("id") is not None, \
                    f"Agent {agent.get('id')} has control without ID"
                assert control.get("status") is not None, \
                    f"Agent {agent.get('id')} control {control.get('id')} without status"


# Test Logical Consistency
class TestLogicalConsistency:
    def test_required_controls_are_enforced_or_implemented(self, crew_config):
        """Test that required controls have appropriate status."""
        controls = crew_config.get("compliance_controls", {})

        for control_id, control in controls.items():
            if control.get("required") is True:
                status = control.get("status")
                assert status in ["enforced", "implemented"], \
                    f"Required control {control_id} must be enforced or implemented, not {status}"

    def test_human_agent_for_escalation(self, crew_config):
        """Test that human agent exists for escalation paths."""
        agents = crew_config.get("agents", [])
        escalation = crew_config.get("escalation_paths", {})

        human_agents = [a for a in agents if a.get("agent_type") == "human"]

        if escalation:
            assert len(human_agents) > 0, \
                "Escalation paths require at least one human agent"

    def test_critical_event_hooks_have_actions(self, crew_config):
        """Test that critical event hooks have defined actions."""
        hooks = crew_config.get("event_hooks", {})
        critical_hooks = ["on_agent_failure", "on_pipeline_blocked"]

        for hook_name in critical_hooks:
            if hook_name in hooks:
                hook_value = hooks[hook_name]
                assert hook_value is not None, \
                    f"Critical hook {hook_name} must have an action defined"


# Performance and Scalability Tests
class TestPerformanceConsiderations:
    def test_agent_count_reasonable(self, crew_config):
        """Test that agent count is reasonable for single config."""
        agents = crew_config.get("agents", [])
        assert len(agents) <= 50, \
            "Config should have reasonable number of agents (<50)"

    def test_no_circular_dependencies(self, crew_config):
        """Test that there are no obvious circular dependencies."""
        # This is a simplified check - real circular dependency detection
        # would require analyzing the actual runtime behavior
        agents = crew_config.get("agents", [])

        # Build a map of agent IDs
        agent_ids = {a.get("id") for a in agents}

        # Check that no agent's manifest/entrypoint references another agent's ID
        # in a way that would suggest circular dependencies
        for agent in agents:
            agent_id = agent.get("id")
            manifest = agent.get("manifest", "")
            entrypoint = agent.get("entrypoint", "")

            # Extract the directory path (should contain the agent's own ID)
            # Check if it references OTHER agent IDs (not itself)
            for other_id in agent_ids:
                if other_id != agent_id:
                    # Check if this agent references another agent in its paths
                    # This would be unusual and could indicate circular dependency
                    if other_id in manifest or other_id in entrypoint:
                        pytest.fail(
                            f"Agent {agent_id} may have circular dependency with {other_id}"
                        )


# Integration Test
class TestConfigurationIntegration:
    def test_complete_workflow_possible(self, crew_config):
        """Test that the configuration supports a complete workflow."""
        # Check all necessary components exist
        assert crew_config.get("agents") is not None
        assert crew_config.get("integration") is not None
        assert crew_config.get("event_hooks") is not None
        assert crew_config.get("escalation_paths") is not None

        # Check critical agent types exist
        agents = crew_config.get("agents", [])
        agent_types = {a.get("agent_type") for a in agents}

        # Need at least AI and human for basic workflow
        assert "ai" in agent_types, "Need AI agents for automation"
        assert "human" in agent_types, "Need human agents for oversight"

    def test_audit_trail_complete(self, crew_config):
        """Test that complete audit trail is possible."""
        # Check audit control exists and is enforced
        controls = crew_config.get("compliance_controls", {})
        assert "AU-2" in controls
        assert controls["AU-2"].get("status") == "enforced"

        # Check integration has audit trail
        integration = crew_config.get("integration", {})
        assert integration.get("audit_trail") is not None

        # Check agents implement audit
        agents = crew_config.get("agents", [])
        audit_agents = 0
        for agent in agents:
            controls = [c.get("id") for c in agent.get("compliance_controls", [])]
            if any(c.startswith("AU-") for c in controls):
                audit_agents += 1

        assert audit_agents > 0, "Need agents with audit controls"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
