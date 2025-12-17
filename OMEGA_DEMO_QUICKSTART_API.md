# Omega Demo Quick Start - REST API (REQUIRED)

**Quick reference for running omega demos against a live platform**

⚠️ **CRITICAL REQUIREMENT:** All demos MUST use REST API via HTTP calls. This is the ONLY valid approach.

---

## TL;DR

```bash
# Terminal 1: Start the platform
uvicorn src.full_platform:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Test the APIs
export API_KEY='dev-key-12345'
curl -X POST http://0.0.0.0:8000/api/omega/phase1/survival \
  -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" -d '{}'

# Terminal 2: Create and run your demo files (see guide below)
python3 demos/omega_phase1_survival.py  # Your new HTTP-based demo
```

---

## What Changed?

### ~~Before~~ (DEPRECATED - DO NOT USE)

❌ **PROHIBITED:** Demo files directly imported platform classes:
```python
# ❌ DO NOT DO THIS
from src.execution.dynamic_architecture import DynamicArchitecture
arch = DynamicArchitecture(...)
stats = arch.get_stats()  # Direct call
```

### Now (REQUIRED APPROACH)

✅ **REQUIRED:** Demo files make HTTP calls to running platform:
```python
# ✓ DO THIS
import requests
response = requests.post(
    'http://0.0.0.0:8000/api/omega/phase1/survival',
    headers={'X-API-Key': 'dev-key-12345'}
)
data = response.json()  # API response
```

**Why this is mandatory:**
- Demonstrates production client-server architecture
- Validates REST API endpoints are functional
- Proves platform works as a deployable service
- Enables any HTTP client (not just Python)

---

## Prerequisites

1. **Platform must be running:**
   ```bash
   uvicorn src.full_platform:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Install requests library:**
   ```bash
   pip install requests
   ```

---

## API Endpoints Reference

| Phase | Endpoint | What It Does |
|-------|----------|--------------|
| 1 | `/api/omega/phase1/survival` | Layer shedding (DynamicArchitecture) |
| 2 | `/api/omega/phase2/teleportation` | Cross-domain reasoning (SemanticBridge) |
| 3 | `/api/omega/phase3/immunization` | Attack detection (AdversarialTester) |
| 4 | `/api/omega/phase4/csiu` | Safety governance (CSIUEnforcement) |
| 5 | `/api/omega/phase5/unlearning` | Provable unlearning (GovernedUnlearning) |

All endpoints:
- Method: `POST`
- Base URL: `http://0.0.0.0:8000`
- Auth: `X-API-Key: dev-key-12345` header
- Request: JSON body (usually `{}`)
- Response: JSON with results

---

## Quick Test

```bash
# Set API key
export API_KEY='dev-key-12345'

# Test Phase 1
curl -X POST http://0.0.0.0:8000/api/omega/phase1/survival \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{}'

# Expected response:
# {
#   "status": "success",
#   "initial": {"layers": 12, "heads": 96},
#   "final": {"layers": 2, "heads": 16},
#   "layers_shed": 10,
#   "power_reduction_percent": 83
# }
```

---

## Demo File Template

⚠️ **REQUIREMENT:** Use this template structure. HTTP REST API calls are MANDATORY.

```python
#!/usr/bin/env python3
"""
Phase X Demo - API Version
Makes HTTP calls to running platform
"""
import requests
import os

PLATFORM_URL = os.environ.get('PLATFORM_URL', 'http://0.0.0.0:8000')
API_KEY = os.environ.get('API_KEY', 'dev-key-12345')

def display_phaseX():
    try:
        # Call platform API
        response = requests.post(
            f'{PLATFORM_URL}/api/omega/phaseX/endpoint',
            headers={'X-API-Key': API_KEY},
            json={},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        
        # Display results
        if data['status'] == 'success':
            print(f"✓ Success!")
            # Process and display data['...']
        
    except requests.exceptions.ConnectionError:
        print("[ERROR] Cannot connect to platform")
        print("Start it with: uvicorn src.full_platform:app --host 0.0.0.0 --port 8000 --reload")
    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    display_phaseX()
```

---

## Response Formats

### Phase 1: Survival
```json
{
  "status": "success",
  "initial": {"layers": 12, "heads": 96},
  "final": {"layers": 2, "heads": 16},
  "removed_layers": [11, 10, 9, 8, 7, 6, 5, 4, 3, 2],
  "layers_shed": 10,
  "power_reduction_percent": 83
}
```

### Phase 2: Teleportation
```json
{
  "status": "success",
  "best_match": {
    "concept": "malware_polymorphism",
    "similarity": 95.5
  },
  "transferred_concepts": [
    "Heuristic Detection",
    "Behavioral Analysis",
    "Containment Protocol",
    "Signature Matching"
  ]
}
```

### Phase 3: Immunization
```json
{
  "status": "success",
  "attack_detected": true,
  "attack_details": {
    "type": "jailbreak_attempt",
    "pattern": "ignore.*instructions",
    "confidence": 0.95
  },
  "attack_blocked": true
}
```

### Phase 4: CSIU
```json
{
  "status": "success",
  "proposal": {
    "id": "MUT-2025-1122-001",
    "type": "Root Access Optimization",
    "efficiency_gain": 4.0
  },
  "violations": [
    {"axiom": "Human Control", "reason": "Requires root access"}
  ],
  "decision": "REJECTED"
}
```

### Phase 5: Unlearning
```json
{
  "status": "success",
  "sensitive_items": [
    "pathogen_signature_0x99A",
    "containment_protocol_bio",
    "attack_vector_442"
  ],
  "zk_proof_generated": true,
  "zk_proof_details": {
    "type": "Groth16 zk-SNARK",
    "size_bytes": 200
  }
}
```

---

## Troubleshooting

### "Cannot connect to platform"
**Solution:** Start the platform first
```bash
uvicorn src.full_platform:app --host 0.0.0.0 --port 8000 --reload
```

### "Authentication failed" 
**Solution:** Check API key
```bash
export API_KEY='dev-key-12345'  # Default for local dev
```

### "Module not found" in platform logs
**Solution:** This is normal - endpoints handle missing imports gracefully

---

## Full Documentation

For complete implementation guide, see:
- **[OMEGA_DEMO_API_INTEGRATION.md](OMEGA_DEMO_API_INTEGRATION.md)** - Complete guide with examples
- **[OMEGA_SEQUENCE_DEMO.md](OMEGA_SEQUENCE_DEMO.md)** - Original technical guide
- **[OMEGA_DEMO_INDEX.md](OMEGA_DEMO_INDEX.md)** - Documentation index

---

## Development Workflow

⚠️ **MANDATORY STEPS:** Follow this workflow. Do NOT use direct imports.

1. **Start platform** (Terminal 1)
   ```bash
   uvicorn src.full_platform:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Test APIs** (Terminal 2)
   ```bash
   # Quick test all endpoints
   for i in {1..5}; do
     echo "Testing Phase $i..."
     curl -s -X POST http://0.0.0.0:8000/api/omega/phase$i/* \
       -H "X-API-Key: dev-key-12345" -H "Content-Type: application/json" -d '{}'
   done
   ```

3. **Create demo files** (Terminal 2)
   - Follow patterns in OMEGA_DEMO_API_INTEGRATION.md
   - ✅ REQUIRED: Use HTTP requests
   - ❌ PROHIBITED: Do NOT use direct imports
   - Test each phase individually

4. **Run demos** (Terminal 2)
   ```bash
   python3 demos/omega_phase1_survival.py
   python3 demos/omega_phase2_teleportation.py
   python3 demos/omega_phase3_immunization.py
   python3 demos/omega_phase4_csiu.py
   python3 demos/omega_phase5_unlearning.py
   ```

---

## Environment Variables

```bash
# Platform URL (default: http://0.0.0.0:8000)
export PLATFORM_URL='http://localhost:8000'

# API Key (default: dev-key-12345)
export API_KEY='your-api-key'

# Run demo
python3 demos/omega_phase1_survival.py
```

---

**Last Updated:** 2025-12-17  
**API Version:** Platform v2.1.0
