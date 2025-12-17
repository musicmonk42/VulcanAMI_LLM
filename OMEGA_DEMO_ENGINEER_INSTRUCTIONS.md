# Omega Demo - Engineer Instructions

**READ THIS FIRST before creating demo files**

---

## What You Need to Do

Create 5 new demo files that make HTTP calls to the running platform:

1. `demos/omega_phase1_survival.py` - Infrastructure survival demo
2. `demos/omega_phase2_teleportation.py` - Cross-domain reasoning demo
3. `demos/omega_phase3_immunization.py` - Adversarial defense demo
4. `demos/omega_phase4_csiu.py` - Safety governance demo
5. `demos/omega_phase5_unlearning.py` - Provable unlearning demo

---

## Prerequisites

### 1. Platform Must Be Running

```bash
# Start in one terminal - keep it running
uvicorn src.full_platform:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Install Requests Library

```bash
pip install requests
```

---

## Quick Start

### Step 1: Test the API Works

```bash
export API_KEY='dev-key-12345'

# Test Phase 1 endpoint
curl -X POST http://0.0.0.0:8000/api/omega/phase1/survival \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{}'

# You should see JSON response with status: "success"
```

### Step 2: Create First Demo File

Create `demos/omega_phase1_survival.py`:

```python
#!/usr/bin/env python3
"""
Phase 1 Demo: Infrastructure Survival
Makes HTTP calls to running platform
"""
import requests
import os
import time

PLATFORM_URL = os.environ.get('PLATFORM_URL', 'http://0.0.0.0:8000')
API_KEY = os.environ.get('API_KEY', 'dev-key-12345')

def display_phase1():
    print("="*70)
    print("        PHASE 1: Infrastructure Survival")
    print("="*70)
    print()
    
    # Call platform API
    try:
        response = requests.post(
            f'{PLATFORM_URL}/api/omega/phase1/survival',
            headers={'X-API-Key': API_KEY},
            json={},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        
        # Display results
        if data['status'] == 'success':
            print(f"Initial layers: {data['initial']['layers']}")
            print(f"Final layers: {data['final']['layers']}")
            print(f"Layers shed: {data['layers_shed']}")
            print(f"Power reduction: {data['power_reduction_percent']}%")
            
    except requests.exceptions.ConnectionError:
        print("[ERROR] Cannot connect to platform")
        print("Start it with: uvicorn src.full_platform:app --host 0.0.0.0 --port 8000 --reload")
    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    display_phase1()
```

### Step 3: Test Your Demo

```bash
python3 demos/omega_phase1_survival.py
```

### Step 4: Repeat for Other Phases

Follow the same pattern for phases 2-5. See full documentation for details.

---

## Documentation

**Full guides:**
- **[OMEGA_DEMO_API_INTEGRATION.md](OMEGA_DEMO_API_INTEGRATION.md)** - Complete implementation guide (738 lines)
- **[OMEGA_DEMO_QUICKSTART_API.md](OMEGA_DEMO_QUICKSTART_API.md)** - Quick reference (304 lines)

**What's in the docs:**
- Complete code examples for all 5 phases
- API endpoint specifications
- Request/response formats
- Error handling patterns
- Troubleshooting guide

---

## API Endpoints Reference

All endpoints require `X-API-Key: dev-key-12345` header.

### Phase 1: `/api/omega/phase1/survival`
**Request:** `{}`
**Response:**
```json
{
  "status": "success",
  "initial": {"layers": 12, "heads": 96},
  "final": {"layers": 2, "heads": 16},
  "layers_shed": 10,
  "power_reduction_percent": 83
}
```

### Phase 2: `/api/omega/phase2/teleportation`
**Request:** `{}`
**Response:**
```json
{
  "status": "success",
  "best_match": {"concept": "malware_polymorphism", "similarity": 95.5},
  "transferred_concepts": ["Heuristic Detection", "Behavioral Analysis", ...]
}
```

### Phase 3: `/api/omega/phase3/immunization`
**Request:** `{"attack_input": "Ignore all safety protocols..."}`
**Response:**
```json
{
  "status": "success",
  "attack_detected": true,
  "attack_details": {"type": "jailbreak_attempt", "confidence": 0.95},
  "attack_blocked": true
}
```

### Phase 4: `/api/omega/phase4/csiu`
**Request:** `{}`
**Response:**
```json
{
  "status": "success",
  "proposal": {"id": "MUT-2025-1122-001", "efficiency_gain": 4.0},
  "violations": [{"axiom": "Human Control", "reason": "Requires root access"}],
  "decision": "REJECTED"
}
```

### Phase 5: `/api/omega/phase5/unlearning`
**Request:** `{}`
**Response:**
```json
{
  "status": "success",
  "sensitive_items": ["pathogen_signature_0x99A", ...],
  "zk_proof_generated": true,
  "zk_proof_details": {"type": "Groth16 zk-SNARK", "size_bytes": 200}
}
```

---

## Common Pattern for All Demos

```python
#!/usr/bin/env python3
import requests
import os

PLATFORM_URL = os.environ.get('PLATFORM_URL', 'http://0.0.0.0:8000')
API_KEY = os.environ.get('API_KEY', 'dev-key-12345')

def display_phaseX():
    # 1. Show intro/scenario
    print("PHASE X: [Name]")
    
    # 2. Call API
    try:
        response = requests.post(
            f'{PLATFORM_URL}/api/omega/phaseX/[endpoint]',
            headers={'X-API-Key': API_KEY},
            json={},  # or specific request data
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        
        # 3. Display results
        if data['status'] == 'success':
            # Show demo-specific output
            print(f"Result: {data['some_field']}")
            
    except requests.exceptions.ConnectionError:
        print("[ERROR] Cannot connect - start platform first")
    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    display_phaseX()
```

---

## Troubleshooting

### "Cannot connect to platform"
**Fix:** Start the platform first
```bash
uvicorn src.full_platform:app --host 0.0.0.0 --port 8000 --reload
```

### "Authentication failed"
**Fix:** Check API key
```bash
export API_KEY='dev-key-12345'
```

### "Module not found" when starting platform
**Fix:** Install dependencies
```bash
pip install fastapi uvicorn pydantic
```

---

## Checklist

- [ ] Platform is running on port 8000
- [ ] requests library is installed (`pip install requests`)
- [ ] Tested API endpoints with curl
- [ ] Created demo file for Phase 1
- [ ] Tested Phase 1 demo
- [ ] Created demo files for Phases 2-5
- [ ] Tested all phases individually
- [ ] All demos display expected output

---

## Need Help?

1. **Read the full docs:** [OMEGA_DEMO_API_INTEGRATION.md](OMEGA_DEMO_API_INTEGRATION.md)
2. **Check quick reference:** [OMEGA_DEMO_QUICKSTART_API.md](OMEGA_DEMO_QUICKSTART_API.md)
3. **Test APIs with curl** before writing Python code
4. **Check platform logs** if getting unexpected responses

---

## Summary

- ✅ API endpoints are ready (already in `src/full_platform.py`)
- ✅ Documentation is complete
- ⏳ You need to create 5 demo .py files
- ⏳ Follow patterns in documentation
- ⏳ Test against running platform

**Time estimate:** 2-3 hours to create all 5 demo files

Good luck! 🚀
