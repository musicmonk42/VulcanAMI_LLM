# Security & Compliance Framework

## 1. Objectives
Integrity, confidentiality, safe autonomy, forensic-grade auditability.

## 2. Threat Matrix
| Threat | Vector | Mitigation |
|--------|--------|-----------|
| Injection | Node params (eval/exec) | Regex pattern block; dangerous type gating |
| Replay | Artifact repeat | Replay window hash gating |
| Privilege Escalation | Trust inflation | Audit anomaly detection; threshold caps |
| Data Exfiltration | Generative outputs | Output whitelisting & redaction filters |
| Resource DoS | Oversized graphs/timeouts | Caps & adaptive timeouts |
| Supply Chain | Dependency compromise | Version pinning, SBOM, SCA scanning |
| Side-channel | Timing inference | Constant-time for crypto operations |
| Autonomous Mutation Abuse | Self-modifying nodes | NSO gates + risk scoring + allowlists |
| **URL Scheme Injection** | **file:// and custom schemes** | **URL scheme validation (http/https only)** |
| **Insecure Network Binding** | **Exposure to all interfaces** | **Default to 127.0.0.1 binding** |
| **Unsafe Deserialization** | **PyTorch model loading** | **weights_only=True parameter** |
| **Unpinned Model Downloads** | **HuggingFace model drift** | **Revision hash pinning** |

## 3. Recent Security Improvements (2024-12)

### 3.1 URL Handling (CWE-22 - Path Traversal)
**Status**: ✅ Fixed

**Issue**: `urllib.request.urlopen()` without scheme validation allowed dangerous schemes like `file://`, `ftp://`, `javascript://`

**Solution**:
- Created `src/utils/url_validator.py` with scheme allowlisting
- All URL requests now validate schemes before network access
- Only `http://` and `https://` schemes are allowed

**Implementation**:
```python
from src.utils.url_validator import validate_url_scheme

# Validate before making request
validate_url_scheme(url) # Raises URLValidationError if not http/https
response = urllib.request.urlopen(url, timeout=10)
```

**Files Updated**:
- `src/agent_interface.py`
- `src/api_server.py`
- `scripts/health_smoke.py`

### 3.2 Network Binding (CWE-605 - Multiple Binds to Same Port)
**Status**: ✅ Fixed

**Issue**: Services defaulted to binding `0.0.0.0` (all network interfaces), exposing them unnecessarily

**Solution**:
- Changed default bind address to `127.0.0.1` (localhost only)
- Configurable via environment variables: `HOST`, `API_HOST`
- Production deployments can override to `0.0.0.0` when needed

**Environment Variables**:
```bash
# Development (secure default)
export HOST=127.0.0.1
export API_HOST=127.0.0.1

# Production/Container deployment (when needed)
export HOST=0.0.0.0
export API_HOST=0.0.0.0
```

**Files Updated**:
- `main.py`
- `app.py`
- `src/gvulcan/config.py`
- `src/vulcan/main.py`
- `src/vulcan/api_gateway.py`

### 3.3 PyTorch Model Loading (CWE-502 - Deserialization of Untrusted Data)
**Status**: ✅ Fixed

**Issue**: `torch.load()` without `weights_only=True` can execute arbitrary code

**Solution**:
- Added `weights_only=True` parameter to all `torch.load()` calls
- Prevents arbitrary code execution during model deserialization

**Implementation**:
```python
# Before (unsafe)
model = torch.load(checkpoint_path, map_location="cpu")

# After (safe)
model = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
```

**Files Updated** (14 instances):
- `inspect_pt.py`
- `inspect_system_state.py`
- `simple_eval_pkl.py`
- `eval_state_dict_gpt.py`
- `src/vulcan/memory/retrieval.py`
- `src/vulcan/reasoning/contextual_bandit.py`
- `src/vulcan/reasoning/multimodal_reasoning.py`
- `src/vulcan/safety/neural_safety.py`
- `src/vulcan/safety/compliance_bias.py`

### 3.4 HuggingFace Model Pinning (CWE-494 - Download Without Integrity Check)
**Status**: ✅ Fixed

**Issue**: Models loaded without revision pinning can change unexpectedly

**Solution**:
- Added support for model revision pinning via environment variables
- Models can now be pinned to specific commit hashes
- Defaults to unpinned for backward compatibility

**Environment Variables**:
```bash
# Production: Pin models to specific commit hashes
export VULCAN_TEXT_MODEL_REVISION=86b5e0934494bd15c9632b12f734a8a67f723594
export VULCAN_AUDIO_MODEL_REVISION=abc123def456...
export VULCAN_BERT_MODEL_REVISION=def789ghi012...
export VULCAN_VISION_AUDIO_MODEL_REVISION=ghi345jkl678...
export ADVERSARIAL_DETECTOR_REVISION=main # or specific hash
```

**How to Find Revision Hashes**:
1. Visit model page: `https://huggingface.co/<model-name>/commits/main`
2. Copy the full commit hash of desired version
3. Set as environment variable

**Files Updated**:
- `configs/dqs/dqs_classifier.py`
- `src/vulcan/reasoning/multimodal_reasoning.py`
- `src/vulcan/processing.py`
- `src/nso_aligner.py` (already had revision support)

### 3.5 Temporary File Usage (CWE-377 - Insecure Temp File)
**Status**: ✅ Fixed

**Issue**: Hardcoded `/tmp/` paths are predictable and subject to race conditions

**Solution**:
- Replaced hardcoded paths with `tempfile.NamedTemporaryFile()`
- Ensures unique file names and proper cleanup

**Implementation**:
```python
import tempfile

# Secure temporary file with automatic cleanup
with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
 file_path = Path(tmp.name)
 # Use file_path...
# Clean up after use
file_path.unlink()
```

**Files Updated**:
- `src/vulcan/config.py`
- `src/gvulcan/storage/local_cache.py` (documentation)

## 4. Secrets & Key Management
Rotation cadence; external KMS; ephemeral dev vs production separation; no logging of secret material.

## 5. Authentication & Authorization
JWT short TTL + scope; API key for Arena fallback; trust-level stored server-side; rate limiting on login & propose endpoints.

## 6. NSO Gates & Risk Pipeline
Sequence: lint → type → tests → security → performance → smoke → risk score evaluation → adversarial check → apply or manual review.

## 7. Cryptographic Integrity
Audit chain: hash(prev_hash + event_json). Periodic integrity sweeps raising alerts on mismatch.

## 8. Audit & Forensics
Event schema: timestamp, actor, event_type, payload_hash, severity. Replay reconstruction + diff bundling for proposals.

## 9. Privacy Controls
Secret redaction (pattern-based), retention policies (TTL prune), sanitized export for compliance audits.

## 10. Hardening Checklist (Extended)
| Item | Status |
|------|--------|
| Secret rotation implemented | Yes |
| TLS + HSTS enforced | Yes |
| Replay guard active | Yes |
| NSO gates integrated CI | Yes |
| Audit chain integrity sweep | Scheduled |
| Dependency scanning | Enabled |
| Self-modification gating | Enforced |
| Config integrity hashing | Roadmap |
| **URL scheme validation** | **Yes ✅** |
| **Secure network binding defaults** | **Yes ✅** |
| **Safe PyTorch model loading** | **Yes ✅** |
| **HuggingFace model pinning** | **Yes ✅** |
| **Secure temporary file usage** | **Yes ✅** |

## 11. Incident Response
Detect → Classify → Contain → Eradicate → Recover → Postmortem → Preventive policy update.

## 12. Supply Chain Security
Pinned versions; SBOM generation; tamper detection via hash compare; periodic CVE triage; HuggingFace model revision pinning.

## 13. Future Enhancements
Secure enclaves, zero-trust workload identity, ML anomaly classification, policy DSL for declarative safety conditions.
