# VULCAN Distillation - Integration Guide

This guide explains how to enable OpenAI knowledge distillation with ensemble mode in your VULCAN application.

## Overview

Knowledge distillation allows VULCAN to:
1. **Serve users with OpenAI quality** - High-quality responses from OpenAI
2. **Capture responses for training** - Automatically save examples (with consent)
3. **Train the internal LLM** - Improve local model from captured data
4. **Reduce costs over time** - Eventually use local LLM for common queries

## Quick Start (3 Steps)

### Step 1: Initialize System

```python
from src.integration.distillation_integration import initialize_distillation_system
from graphix_vulcan_llm import GraphixVulcanLLM

# Load your local LLM
llm = GraphixVulcanLLM()

# Initialize distillation system (ensemble mode uses both LLMs)
system = initialize_distillation_system(llm, mode="ensemble")
```

### Step 2: Use in Your Application

```python
# In your request handler
async def handle_user_query(prompt: str, user_opted_in: bool):
    result = await system.execute(
        prompt=prompt,
        user_opted_in=user_opted_in,  # Required for capture
        enable_distillation=True,
    )
    
    # Return high-quality response
    return result["text"]
```

### Step 3: Start Training Worker

```bash
# Run training (separate process)
python src/training/train_llm_with_self_improvement.py \
    --distillation-storage data/distillation \
    --steps 10000 \
    --val-interval 300
```

**That's it!** Users get OpenAI quality, internal LLM trains automatically.

## Execution Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `ensemble` | Both LLMs, OpenAI quality (recommended) | Production with distillation |
| `local_first` | Try local first, fallback to OpenAI | Cost-sensitive deployment |
| `openai_first` | OpenAI primary, local backup | Maximum quality |
| `parallel` | Race both, use fastest | Low-latency requirement |

## Configuration

Edit `config/distillation_config.py` to customize settings:

```python
from config.distillation_config import get_config

# Get production configuration
config = get_config("production")

# Override specific settings
system = initialize_distillation_system(
    llm,
    mode="ensemble",
    storage_path="data/my_distillation.jsonl",
    require_opt_in=True,      # Always require in production
    enable_pii_redaction=True,
)
```

### Key Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `require_opt_in` | `True` | Require user consent for capture |
| `enable_pii_redaction` | `True` | Automatically redact PII |
| `enable_governance_check` | `True` | Check for sensitive content |
| `min_quality_score` | `0.65` | Minimum quality to capture |
| `executor_timeout` | `30.0` | Timeout for LLM calls (seconds) |

## Monitoring

### Check Status

```bash
# One-time status
python scripts/monitor_distillation.py

# Continuous monitoring
python scripts/monitor_distillation.py --continuous --interval 30
```

### Programmatic Status

```python
status = system.get_status()
print(f"Captured: {status['distiller']['stats']['examples_captured']}")
print(f"Quality: {status['distiller']['stats']['average_quality_score']:.2f}")
```

## Training from Captured Examples

The training script automatically checks the distillation storage:

```bash
python src/training/train_llm_with_self_improvement.py \
    --distillation-storage data/distillation \
    --distillation-interval 10 \  # Check every 10 steps
    --steps 5000 \
    --batch-size 16 \
    --val-interval 200
```

Training will:
1. Run regular training on your corpus
2. Every N steps, check for distillation examples
3. If found, train on OpenAI examples with same loss
4. Log both sources to training log

## Privacy & Security

The distillation pipeline includes 5-stage filtering:

1. **Opt-In Check** - Only captures if `user_opted_in=True`
2. **PII Redaction** - Automatically removes personal info
3. **Secret Detection** - Blocks API keys, passwords, etc.
4. **Governance Check** - Filters sensitive content
5. **Quality Validation** - Ensures minimum quality score

**Production Checklist:**
- [ ] `require_opt_in=True` in production
- [ ] `enable_pii_redaction=True`
- [ ] `enable_governance_check=True`
- [ ] Consider `use_encryption=True` for storage

## Troubleshooting

### No examples being captured?

1. Check `user_opted_in=True` is passed to `execute()`
2. Verify OpenAI API key is set
3. Run monitor to see rejection reasons:
   ```bash
   python scripts/monitor_distillation.py
   ```

### Training not reading examples?

1. Verify storage path exists:
   ```bash
   ls -la data/distillation/
   ```
2. Check examples.jsonl has content:
   ```bash
   wc -l data/distillation/examples.jsonl
   ```
3. Verify `--distillation-storage` path matches

### Quality scores too low?

Adjust thresholds in configuration:
```python
system = initialize_distillation_system(
    llm,
    min_quality_score=0.50,  # Lower threshold
)
```

## API Reference

### DistillationSystem

```python
class DistillationSystem:
    def __init__(
        self,
        graphix_vulcan_llm: Any,
        mode: str = "ensemble",
        config: Optional[Dict[str, Any]] = None,
    )
    
    async def execute(
        self,
        prompt: str,
        user_opted_in: bool = False,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        enable_distillation: bool = True,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]
    
    def get_status(self) -> Dict[str, Any]
    
    def set_mode(self, mode: str) -> None
```

### Response Format

```python
{
    "text": "Generated response text",
    "source": "ensemble",  # or "openai", "local"
    "systems_used": ["openai_llm", "vulcan_local_llm"],
    "distillation_enabled": True,
    "user_opted_in": True,
}
```

## Files Reference

| File | Purpose |
|------|---------|
| `config/distillation_config.py` | Configuration settings |
| `src/integration/distillation_integration.py` | Main integration API |
| `scripts/monitor_distillation.py` | Monitoring script |
| `examples/distillation_example.py` | Complete example |
| `src/training/train_llm_with_self_improvement.py` | Training with distillation |

## Support

For issues:
1. Check this guide's troubleshooting section
2. Run the example to verify setup
3. Check logs for detailed error messages
