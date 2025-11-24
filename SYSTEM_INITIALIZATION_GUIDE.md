# VulcanAMI_LLM System Initialization Guide

## Overview

The VulcanAMI_LLM system consists of three main services that are mounted at startup:

1. **VULCAN** (`/vulcan`) - The core AI system
2. **Graphix Arena** (`/arena`) - Agent arena and evolution platform
3. **Registry** (`/registry`) - Flask-based service registry with rate limiting

## System Components

### GraphixVulcanLLM v2.0.2
- **Model Architecture**: 512-dimensional embeddings
- **Layers**: 6 transformer layers
- **Attention Heads**: 8 heads per layer
- **Features**:
  - Transformer-based architecture with attention mechanisms
  - Cognitive loop integration for reasoning
  - Safe generation with explainability

### World Model
The world model provides causal reasoning and prediction capabilities:

- **Causal Graphs**: Bayesian structure learning and intervention analysis
- **Prediction Engine**: Multi-horizon forecasting with uncertainty quantification
- **Dynamics Model**: State transition modeling and trajectory prediction
- **Correlation Tracker**: Statistical dependency analysis

### Safety Layer
Multi-layered safety validation system:

- **Neural Safety Validators**: Deep learning-based constraint checking
- **Formal Verification**: SMT-based proof systems
- **Compliance/Bias Detection**: Fairness and regulatory compliance checks
- **CSIU Enforcement**: Consent, Safety, Integrity, Utility policy enforcement

### Meta-Reasoning System
Autonomous self-improvement with motivational introspection:

**Six Core Objectives:**
1. **Epistemic Curiosity**: Knowledge-seeking and exploration
2. **Competence Improvement**: Skill acquisition and refinement
3. **Social Collaboration**: Multi-agent coordination
4. **Efficiency Optimization**: Resource utilization
5. **Safety Preservation**: Risk mitigation
6. **Value Alignment**: Human preference learning

**Self-Improvement Drive Configuration:**
- Auto-apply enabled: **Yes**
- Human approval required: **No**
- Budget management: Cost-aware execution with configurable limits

### Hardware Backend
- **Analog Photonic Emulator**: CPU-based quantum-inspired optimization
- Supports energy-efficient analog computation simulation
- Fallback to digital computation when photonic hardware unavailable

## Startup Process

### Quick Start

To see the comprehensive startup information, run:

```bash
# Option 1: Run the demo script
python3 startup_demo.py

# Option 2: Use the startup logger module
python3 -c "import sys; sys.path.insert(0, 'src'); from startup_logger import log_vulcan_startup; log_vulcan_startup()"

# Option 3: Start the unified platform (includes startup logging)
python3 src/full_platform.py
```

### Startup Output Example

```
================================================================================
                       VulcanAMI_LLM System Initialization                       
================================================================================

Three services mounted:
-----------------------
✓ VULCAN (/vulcan)
  The core AI system with a world model, safety validators,
  reasoning modules, and self-improvement capabilities

✓ Graphix Arena (/arena)
  An agent arena with generator, evolver, and visualizer agents

✓ Registry (/registry)
  A Flask service with Redis-backed rate limiting

Key components initialized:
---------------------------
✓ GraphixVulcanLLM v2.0.2
  512-dimensional model with 6 layers, 8 heads
  Transformer-based architecture with attention mechanisms
  Cognitive loop integration for reasoning

✓ World Model
  ✓ Causal graphs (Bayesian structure learning)
  ✓ Prediction engine (multi-horizon forecasting)
  ✓ Dynamics model (state transitions)
  ✓ Correlation tracker (dependency analysis)

✓ Safety Layer
  ✓ Neural safety validators
  ✓ Formal verification
  ✓ Compliance/bias detection
  ✓ CSIU enforcement

✓ Meta-reasoning
  Motivational introspection with 6 objectives:
    1. Epistemic curiosity (knowledge-seeking)
    2. Competence improvement (skill acquisition)
    3. Social collaboration (multi-agent coordination)
    4. Efficiency optimization (resource utilization)
    5. Safety preservation (risk mitigation)
    6. Value alignment (human preference learning)
  Self-improvement drive:
    * Auto-apply enabled: Yes
    * Human approval required: No
    * Budget management: Cost-aware execution

✓ Hardware
  Analog Photonic emulator
  Backend: CPU
  Quantum-inspired optimization algorithms
  Energy-efficient computation simulation

Notable warnings:
-----------------
⚠️  Groth16 SNARK module unavailable (falling back to basic implementation)
   Note: py-ecc library not installed, using pure Python fallback

⚠️  spaCy model not loaded for analogical reasoning
   Note: Run: python -m spacy download en_core_web_sm

⚠️  FAISS loaded with AVX2 (AVX512 unavailable)
   Note: Vector operations will use AVX2 instructions

System initialization complete.
All services are ready to accept requests.
```

## Service Endpoints

Once the system is running, the following endpoints are available:

### VULCAN Service
- Base URL: `http://localhost:8080/vulcan`
- API Documentation: `http://localhost:8080/vulcan/docs`
- Health Check: `http://localhost:8080/vulcan/health`

### Graphix Arena
- Base URL: `http://localhost:8080/arena`
- API Documentation: `http://localhost:8080/arena/docs`
- Health Check: `http://localhost:8080/arena/health`

### Registry
- Base URL: `http://localhost:8080/registry`
- API Documentation: `http://localhost:8080/registry/`
- Health Check: `http://localhost:8080/registry/health`

## Configuration

### Environment Variables

Set the following environment variables before starting the system:

```bash
# API Keys (optional, for LLM features)
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GRAPHIX_API_KEY="your-graphix-key"

# Redis (optional, for distributed rate limiting)
export REDIS_HOST="localhost"
export REDIS_PORT="6379"

# JWT Authentication (optional)
export JWT_SECRET_KEY="your-secret-key"
export BOOTSTRAP_KEY="your-bootstrap-key"

# Deployment Mode
export DEPLOYMENT_MODE="development"  # or "production", "testing"

# Self-Improvement Configuration
export VULCAN_ENABLE_SELF_IMPROVEMENT="1"
export VULCAN_AUTO_APPLY_POLICY="configs/auto_apply_policy.yaml"
```

### Self-Improvement Configuration

The self-improvement drive can be configured in `configs/intrinsic_drives.json`:

```json
{
  "enabled": true,
  "approval_required": false,
  "check_interval_seconds": 120,
  "max_cost_usd_per_session": 2.0,
  "max_cost_usd_per_day": 10.0,
  "objectives": {
    "epistemic_curiosity": {"weight": 1.0, "enabled": true},
    "competence_improvement": {"weight": 1.0, "enabled": true},
    "social_collaboration": {"weight": 0.8, "enabled": true},
    "efficiency_optimization": {"weight": 0.9, "enabled": true},
    "safety_preservation": {"weight": 1.2, "enabled": true},
    "value_alignment": {"weight": 1.1, "enabled": true}
  }
}
```

## Troubleshooting

### Missing Dependencies

If you see warnings about missing components:

1. **Groth16 SNARK**: Install py-ecc library
   ```bash
   pip install py-ecc
   ```

2. **spaCy model**: Download the English model
   ```bash
   python -m spacy download en_core_web_sm
   ```

3. **FAISS**: Install with AVX2 support
   ```bash
   pip install faiss-cpu
   ```

### Component Not Loading

If a component fails to load, check:
- All dependencies are installed: `pip install -r requirements.txt`
- Python version is compatible (Python 3.10+)
- Environment variables are set correctly
- Configuration files exist in `configs/` directory

### Services Not Mounting

If services fail to mount:
- Check that port 8080 is available
- Verify no conflicting processes are running
- Review logs for specific error messages
- Ensure all required modules are in the `src/` directory

## Development

### Running Tests

```bash
# Run all tests
python3 -m pytest tests/ -v

# Run startup logger tests specifically
python3 -m pytest tests/test_startup_logger.py -v

# Run with coverage
python3 -m pytest tests/ --cov=src --cov-report=html
```

### Adding New Components

To add logging for a new component:

1. Import the startup logger:
   ```python
   from startup_logger import get_startup_logger
   ```

2. Log component initialization:
   ```python
   sl = get_startup_logger()
   sl.log_component_init(
       "NewComponent",
       version="1.0.0",
       details=["Feature 1", "Feature 2"],
       success=True
   )
   ```

3. Log warnings if needed:
   ```python
   sl.log_warning("Optional feature unavailable", note="Install xyz for full functionality")
   ```

## References

- GraphixVulcanLLM Documentation: See `graphix_vulcan_llm.py`
- World Model API: See `src/vulcan/world_model/`
- Safety Layer: See `src/vulcan/safety/`
- Meta-Reasoning: See `src/vulcan/world_model/meta_reasoning/`

## Support

For issues or questions:
- Check existing issues in the repository
- Review logs in `unified_platform.log`
- Enable debug logging: `export LOG_LEVEL=DEBUG`
