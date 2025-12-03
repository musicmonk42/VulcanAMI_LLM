# LLM Training Workflow

This document describes the automated LLM training workflow that runs on pull requests.

## Overview

The LLM Training workflow (`.github/workflows/llm-training.yml`) automatically triggers training runs when pull requests are opened or updated. This enables continuous evaluation of model changes and provides immediate feedback on training performance.

## Features

- **Automatic Trigger**: Runs on PR open, synchronize, or reopen events
- **Manual Execution**: Can be triggered manually with custom parameters
- **Governed Training**: Uses the GovernedTrainer with safety checks and meta self-improvement
- **Artifact Management**: Automatically uploads training logs and model checkpoints
- **PR Comments**: Posts training results directly to the pull request
- **Resource Management**: Includes timeout protection and concurrent run cancellation

## Triggering the Workflow

### Automatic (on Pull Request)

The workflow automatically runs when:
- A pull request is opened to `main` or `develop`
- New commits are pushed to an open pull request
- A pull request is reopened

### Manual Execution

You can manually trigger the workflow with custom parameters:

1. Go to **Actions** → **LLM Training on Pull Request**
2. Click **Run workflow**
3. Select the branch
4. Configure parameters:
   - **Training Steps**: Number of training iterations (default: 1000)
   - **Sequence Length**: Input sequence length (default: 128)
   - **Meta Apply**: Enable meta self-improvement application (default: false)

## Training Configuration

### Default Settings

```yaml
Training Steps: 1000
Sequence Length: 128
Validation Interval: 200 steps
Meta Self-Improvement: Advisory mode (not applied)
```

### Training Script

The workflow executes `src/training/train_llm_with_self_improvement.py` with the following features:

- **Governance Loop**: Every optimizer step goes through consensus approval
- **Self-Awareness Metrics**: Entropy, calibration (ECE/MCE), perplexity, diversity
- **Safety Guardrails**: Learning rate change ratios, gradient clipping
- **Checkpointing**: Model and optimizer state saving
- **Structured Logging**: JSONL format for training and meta cycles

## Workflow Steps

1. **Checkout Code**: Retrieves the PR branch
2. **Set Up Python**: Installs Python 3.11 with pip caching
3. **Install Dependencies**: Installs PyTorch (CPU) and project requirements
4. **Create Directories**: Sets up data, logs, and checkpoints folders
5. **Prepare Corpus**: Creates a demo corpus if none exists
6. **Run Training**: Executes the training script with configured parameters
7. **Generate Summary**: Creates a markdown summary of training results
8. **Upload Artifacts**: Saves logs and models for later review
9. **Comment on PR**: Posts results directly to the pull request
10. **Check Status**: Reports any warnings or errors

## Artifacts

### Training Logs

Location: `training-logs-{run_number}`

Contains:
- `training_output.log`: Full training output
- `training_summary.md`: Formatted summary
- `logs/`: Detailed training logs (if generated)

Retention: 30 days

### Trained Models

Location: `trained-models-{run_number}`

Contains:
- Model checkpoints (`.pt` files)
- Optimizer states

Retention: 30 days

## PR Comments

The workflow automatically posts a comment on the pull request with:

- Training configuration (steps, sequence length, parameters)
- Training output (last 50 lines)
- Generated artifacts list
- Workflow run link
- Overall status (success/failure/warning)

Example comment:

```markdown
## 🤖 LLM Training Results

**PR:** #123
**Branch:** feature/new-model
**Training Steps:** 1000
**Sequence Length:** 128
**Meta Apply:** false

### Training Output (Last 50 lines)
...

### Generated Artifacts
- training_output.log
- llm_training_log.jsonl
- llm_last_model.pt

---
**Workflow Run:** #45
**Status:** success
```

## Best Practices

### For Model Changes

1. **Start with Low Steps**: Use 1000 steps for quick validation
2. **Enable Meta Apply Cautiously**: Only enable for trusted changes
3. **Review Artifacts**: Check training logs for anomalies
4. **Compare Baselines**: Compare with previous PR training results

### For Data Changes

1. **Verify Corpus**: Ensure data preparation is correct
2. **Adjust Sequence Length**: Match your data characteristics
3. **Monitor Validation**: Check validation metrics closely

### For Infrastructure Changes

1. **Test Locally First**: Run training script locally before PR
2. **Check Dependencies**: Ensure all requirements are met
3. **Monitor Timeout**: 120-minute limit for workflow

## Troubleshooting

### Training Fails to Start

**Issue**: Dependencies not installed
**Solution**: Check requirements.txt and ensure PyTorch installs correctly

### Corpus Too Short Error

**Issue**: Sequence length exceeds corpus size
**Solution**: Workflow auto-adjusts, but verify your training data

### Timeout Error

**Issue**: Training exceeds 120 minutes
**Solution**: Reduce training steps or optimize training script

### Artifacts Not Uploaded

**Issue**: Checkpoints directory empty
**Solution**: Verify checkpoint_interval in training script

## Configuration Reference

### Environment Variables

```yaml
PYTHON_VERSION: '3.11'         # Python version for training
TRAINING_STEPS: '1000'         # Default training steps
SEQ_LENGTH: '128'              # Default sequence length
```

### Job Configuration

```yaml
runs-on: ubuntu-latest         # GitHub-hosted runner
timeout-minutes: 120           # Maximum execution time
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true     # Cancel previous runs on new push
```

## Advanced Usage

### Custom Training Parameters

To run training with custom parameters via workflow_dispatch:

```bash
# Using GitHub CLI
gh workflow run llm-training.yml \
  -f training_steps=5000 \
  -f seq_length=256 \
  -f meta_apply=true
```

### Local Testing

To test the training script locally before PR:

```bash
# Basic training
python src/training/train_llm_with_self_improvement.py \
  --steps 1000 \
  --val-interval 200 \
  --seq-len 128 \
  --auto-create-corpus \
  --data-path data/corpus

# With meta self-improvement
python src/training/train_llm_with_self_improvement.py \
  --steps 3000 \
  --val-interval 300 \
  --seq-len 128 \
  --meta-interval 800 \
  --meta-apply \
  --meta-safe-types lr_adjustment,grad_clip_adjust \
  --data-path data/corpus
```

## Integration with Other Workflows

The LLM Training workflow complements:

- **CI Workflow**: Runs in parallel with linting and testing
- **Security Workflow**: Training artifacts can be scanned separately
- **Deployment Workflow**: Successful training can trigger staging deployment

## Monitoring and Metrics

### Training Metrics Tracked

- Loss (training and validation)
- Perplexity
- Entropy (prediction confidence)
- Calibration (ECE, MCE, Adaptive ECE)
- Diversity (distinct-n)
- Steps per second
- Drift detection

### Meta Self-Improvement Metrics

When enabled, tracks:
- Proposed changes (learning rate, gradient clipping, warmup)
- Approval/rejection decisions
- Applied changes and their effects
- Safety guardrail triggers

## Security Considerations

- Training runs in isolated GitHub Actions environment
- No external network access during training
- Artifacts stored securely in GitHub
- 30-day retention limit on artifacts
- No secrets required for basic training

## Future Enhancements

Potential improvements:
- GPU runner support for faster training
- Distributed training across multiple runners
- Integration with experiment tracking (Weights & Biases, MLflow)
- Automated model comparison with baseline
- Performance regression detection
- Training cost estimation and budgeting

## Related Documentation

- [CI_CD.md](../CI_CD.md) - Complete CI/CD pipeline documentation
- [TESTING_GUIDE.md](../TESTING_GUIDE.md) - Testing guidelines
- Training script: `src/training/train_llm_with_self_improvement.py`
- Governed trainer: `src/training/governed_trainer.py`

## Support

For issues or questions about the LLM training workflow:
1. Check workflow run logs in GitHub Actions
2. Review training artifacts
3. Consult the training script documentation
4. Contact the ML infrastructure team
