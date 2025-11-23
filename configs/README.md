# VulcanAMI_LLM Configuration Directory

This directory contains all configuration files for the VulcanAMI_LLM system, including agent profiles, hardware backends, intrinsic drives, and system manifests.

## Overview

The configuration system is centralized, versioned, and validated. All configs integrate with `src/vulcan/config.py` (ConfigurationManager) and support:
- **Layered configuration**: Defaults → Files → Profiles → Environment → Runtime
- **Schema validation**: Type checking and business logic validation
- **Auto-reload**: File watchers for dynamic updates (when watchdog available)
- **Versioning**: Track changes and config history

## Directory Structure

```
configs/
├── __init__.py                      # Python package initialization
├── validate_configs.py              # Validation script
├── test_integration.py              # Integration tests
├── README.md                        # This file
│
├── Core Configurations
├── graphix_core_manifest.json       # Graphix IR component manifest
├── graphix_core_ontology.json       # Graphix IR ontology definitions
├── hardware_profiles.json           # Hardware backend profiles
├── intrinsic_drives.json            # Agent intrinsic motivation config
├── platform_mapping.json/.yaml      # API endpoint mappings
├── type_system_manifest.json        # Type system definitions
├── crew_config.yaml                 # Agent crew configuration
├── tool_selection.yaml              # Tool selection defaults
├── auto_apply_policy.yaml           # Auto-apply policies
├── helm_chart.yaml                  # Kubernetes Helm chart
│
├── Profiles
├── profile_development.json         # Development environment profile
├── profile_testing.json             # Testing environment profile
│
└── Subsystem Configurations
    ├── cloudfront/                  # CloudFront cache policies
    ├── dqs/                         # Data quality service configs
    ├── nginx/                       # Nginx configurations
    ├── opa/                         # Open Policy Agent bundles
    ├── packer/                      # Packer image build configs
    ├── redis/                       # Redis configurations
    ├── vector/                      # Vector database (Milvus) configs
    └── zk/                          # Zero-knowledge proof circuits
```

## Core Configuration Files

### graphix_core_manifest.json
Defines the core components of the Graphix IR execution system:
- Executor, validator, optimizer, scheduler
- Hardware backend preferences (CPU, GPU, vLLM, photonic, etc.)
- Execution parameters (parallelism, timeouts, retries)
- Optimization settings (JIT, caching, batching)

**Used by**: `src/unified_runtime/vulcan_integration.py`

### graphix_core_ontology.json
Formal ontology for Graphix IR semantic validation:
- Class definitions (Node, Edge, Agent, Objective, Concept)
- Property definitions with constraints
- Relationship definitions (causal, dependency, conflict, etc.)
- Global constraints (max nodes/edges, acyclic requirements)

**Used by**: Graphix IR semantic validator

### hardware_profiles.json
Performance profiles for different hardware backends:
- **cpu**: Conventional x86 CPU
- **gpu**: NVIDIA A100 40GB
- **vllm**: Tensor-parallel vLLM engine
- **photonic**: Analog-photonic MVM core (simulated)
- **memristor**: Memristor compute-in-memory (simulated)
- **aws-f1-fpga**: AWS F1 FPGA instance

Each profile includes: latency, throughput, energy per operation, max tensor size, dynamic metrics.

**Used by**: Tool selection system, resource allocation

### intrinsic_drives.json
Configuration for agent intrinsic motivation and self-improvement:
- **Drives**: self_improvement, exploration, optimization, maintenance
- **Triggers**: on_startup, on_error, periodic, on_low_activity
- **Objectives**: fix_circular_imports, optimize_performance, improve_test_coverage
- **Constraints**: rate limits, safety checks, resource limits
- **Validation**: pre-flight checks, testing requirements, rollback procedures

**Used by**: 
- `src/vulcan/world_model/meta_reasoning/self_improvement_drive.py`
- `src/vulcan/world_model/meta_reasoning/motivational_introspection.py`
- `src/vulcan/world_model/world_model_core.py`

### platform_mapping.json / .yaml
API endpoint mappings for different services:
- `submit_run`: POST /api/arena/run/generator
- `submit_feedback`: POST /api/arena/feedback_dispatch  
- `improve`: POST /improve
- `sse_stream`: GET /v1/stream
- `metrics`: GET /api/arena/metrics
- `rollback`: POST /rollback

**Used by**: API clients, service integrations

### type_system_manifest.json
Type system definitions for runtime type checking:
- Primitive types (int, float, str, bool, list, dict, set)
- NumPy types (ndarray)
- Domain types (Observation, CausalEdge, Prediction, Concept, Objective)
- Enums (UpdateType, GroundingStatus, ObjectiveType, ConflictType)

**Used by**: Type validation, serialization, API contracts

### crew_config.yaml
Multi-agent crew configuration with compliance controls:
- Agent definitions (refactor, judge, healer, ethics, simulation, CI/CD, human, oracle)
- Compliance controls (NIST 800-53 controls: AC-1, AC-2, AC-3, AU-2, CM-2, IR-4, etc.)
- Integration endpoints (artifact store, provenance log, event bus)
- Event hooks (on failure, on artifact created, on score below threshold)

**Used by**: Multi-agent orchestration system

### tool_selection.yaml
Default configuration for tool selection system:
- Utility weights (quality, time_penalty, energy_penalty, risk_penalty)
- Calibration settings (min_samples, retrain_interval, temperature_scaling)
- Portfolio strategies (single, speculative_parallel, sequential_refinement)
- Cost model (track_variance, cold_start_penalty, health_check_interval)

**Used by**: `src/vulcan/config.py` tool_selection_config

### auto_apply_policy.yaml
Policies for auto-applying agent changes:
- Budget limits (max_files, max_total_loc)
- Allowed/denied file patterns
- Pre-apply gates (linting, testing, type checking)
- NSO requirements (adversarial_detected, risk_score_max)

**Used by**: Self-improvement drive auto-apply system

## Profile Configurations

Profiles define environment-specific settings that override defaults:

### profile_development.json
**Purpose**: Local development with full debugging and self-improvement enabled

Key settings:
- `enable_self_improvement`: true (at root level)
- `log_level`: DEBUG
- `safety_level`: 0 (minimal)
- `require_human_approval`: false
- `enable_adversarial_testing`: true
- `intrinsic_drives_config.enabled`: true
- Resource limits: 8GB RAM, 80% CPU

### profile_testing.json
**Purpose**: Automated testing with reduced resources and no self-modification

Key settings:
- `enable_self_improvement`: false
- `log_level`: DEBUG
- `safety_level`: 1 (standard)
- `enable_adversarial_testing`: true
- Resource limits: 4GB RAM, 50% CPU
- Fast tool selection mode

## Subsystem Configurations

### cloudfront/
CloudFront CDN cache policies and schemas
- `cache_policy.json`: Cache behavior definitions
- `cache_policy.schema.json`: JSON schema for validation
- `README.md`: Documentation

### dqs/
Data Quality Service configurations
- `classifier.json`: Classification rules
- `rescore_cron.json`: Rescoring schedule
- `dqs_classifier.py`: Classifier implementation
- `setup_dqs.sh`: Setup script

### nginx/
Nginx web server and reverse proxy configs
- `origin.conf`: Origin server configuration
- `nginx-cache-manager.sh`: Cache management script
- `nginx-monitor.sh`: Monitoring script

### opa/
Open Policy Agent policy bundles
- `bundles/p2025.11/policy.rego`: Policy rules
- `bundles/p2025.11/data.json`: Policy data
- `version.txt`: OPA version

### packer/
HashiCorp Packer image build configurations
- `packer.toml`: Main Packer configuration
- `semver.txt`: Image version
- `README.md`: Build instructions

### redis/
Redis in-memory data store configs
- `redis.conf`: Redis server configuration
- `keys_ttl.yml`: TTL policies for keys
- `exporter.env`: Prometheus exporter config

### vector/
Vector database (Milvus) configurations
- `milvus/collections.yaml`: Collection schemas and indexes
- `README.md`: Usage guide

### zk/
Zero-knowledge proof circuit configurations
- `circuits/circuit_specification.yaml`: Circuit parameters
- `circuits/unlearning_v1.0.circom`: Circom circuit
- `circuits/build_circuit.sh`: Build script
- `circuits/ZK_UNLEARNING_README.md`: Detailed docs

## Usage

### Loading Configurations

```python
from vulcan.config import (
    ConfigurationManager,
    ProfileType,
    get_config,
    load_profile,
)

# Load a profile
load_profile(ProfileType.DEVELOPMENT)

# Get specific config value
agent_id = get_config('agent_config.agent_id', default='vulcan-001')

# Get nested config
max_memory = get_config('resource_limits.max_memory_mb', default=8000)

# Get intrinsic drives config
drives_config = get_config('intrinsic_drives_config')
```

### Validating Configurations

```bash
# Validate all configs
python configs/validate_configs.py

# Strict mode (exit on warnings)
python configs/validate_configs.py --strict
```

### Testing Integration

```bash
# Run integration tests
python configs/test_integration.py
```

### Environment Variables

Override configs using environment variables:

```bash
export VULCAN_AGENT_CONFIG_LOG_LEVEL=DEBUG
export VULCAN_RESOURCE_LIMITS_MAX_MEMORY_MB=16000
export VULCAN_ENABLE_SELF_IMPROVEMENT=true
```

## Validation

All configurations are validated on load:

1. **Syntax validation**: JSON/YAML parsing
2. **Schema validation**: Type checking and constraints
3. **Consistency checks**: Cross-file consistency
4. **Security checks**: Sensitive data patterns
5. **Business logic**: Resource limits, safety thresholds

## Versioning

Configuration files should include versioning information:

```json
{
  "versioning": {
    "config_version": "1.0.0",
    "schema_version": "1.0.0",
    "last_updated": "2025-11-23T10:00:00Z",
    "updated_by": "username",
    "changelog": [
      {
        "version": "1.0.0",
        "date": "2025-11-23",
        "changes": ["Initial version"]
      }
    ]
  }
}
```

## Security Considerations

1. **Never commit secrets**: Use environment variables or secret management
2. **Sensitive keywords**: Files containing "password", "secret", "api_key" should use references
3. **Schema validation**: Required for production profiles
4. **Audit logging**: Enabled for production profiles
5. **Encryption**: Required for safety-critical profiles

## Best Practices

1. **Profile-specific configs**: Use profiles instead of modifying defaults
2. **Version all changes**: Update versioning section on each change
3. **Validate before commit**: Run `validate_configs.py`
4. **Test integration**: Run `test_integration.py`
5. **Document changes**: Update README and inline comments
6. **Minimize changes**: Only change what's necessary
7. **Backup configs**: Keep backups before making changes

## Troubleshooting

### Config not loading
- Check file syntax with `python -m json.tool config.json`
- Verify file path in error messages
- Check file permissions

### Validation errors
- Run `validate_configs.py` for detailed error messages
- Check schema constraints in `src/vulcan/config.py`
- Verify required fields are present

### Integration issues
- Run `test_integration.py` to identify problems
- Check imports in `src/vulcan/config.py`
- Verify config paths match code expectations

## Contributing

When adding new configurations:

1. Add to appropriate subdirectory
2. Include JSON schema or validation rules
3. Update this README
4. Add validation to `validate_configs.py`
5. Add integration test to `test_integration.py`
6. Document in subsystem README
7. Test with all profiles

## References

- [Configuration Management Best Practices](https://12factor.net/config)
- [JSON Schema](https://json-schema.org/)
- [YAML Specification](https://yaml.org/spec/)
- [VulcanAMI Documentation](../README.md)
