# Security Fix: Pickle Deserialization Vulnerability

## Issue Description

**Severity:** HIGH  
**CWE:** CWE-502 (Deserialization of Untrusted Data)  
**Impact:** Remote Code Execution (RCE)

The codebase uses Python's `pickle` module in several locations to serialize and deserialize data. Pickle is inherently unsafe when deserializing untrusted data as it can execute arbitrary code during the unpickling process.

## Affected Files

1. `./inspect_system_state.py:9` - Direct pickle.load usage
2. `./archive/orchestrator.py:2255` - pickle.load usage
3. `./archive/symbolic_reasoning.py:3865` - pickle.load usage
4. `./demo/demo_graphix.py:202` - pickle.load on cache data
5. `./src/unified_runtime/runtime_extensions.py:21` - pickle import
6. `./src/vulcan/world_model/world_model_router.py:1832` - pickle.load usage
7. `./src/processing.py:25` - pickle import

## Risk Analysis

### Attack Scenario
1. Attacker provides malicious pickle file (e.g., checkpoint, cache, state file)
2. Application loads pickle file using `pickle.load()`
3. Malicious code embedded in pickle executes with application privileges
4. Attacker gains remote code execution

### Real-World Impact
- Complete system compromise
- Data exfiltration
- Lateral movement in network
- Privilege escalation
- Ransomware deployment

## Recommended Solutions

### Option 1: Replace Pickle with Safer Alternatives (RECOMMENDED)

#### For Simple Data Structures
```python
# BEFORE (UNSAFE)
import pickle
with open('data.pkl', 'rb') as f:
    data = pickle.load(f)

# AFTER (SAFE)
import json
with open('data.json', 'r') as f:
    data = json.load(f)
```

#### For ML Model Weights
```python
# BEFORE (UNSAFE)
import pickle
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# AFTER (SAFE)
import torch
model = torch.load('model.pt', weights_only=True)

# OR use safetensors
from safetensors.torch import load_file
model.load_state_dict(load_file('model.safetensors'))
```

#### For Numpy Arrays
```python
# BEFORE (UNSAFE)
import pickle
with open('data.pkl', 'rb') as f:
    array = pickle.load(f)

# AFTER (SAFE)
import numpy as np
array = np.load('data.npy', allow_pickle=False)
```

### Option 2: Implement HMAC Signature Verification

If pickle must be used, implement integrity verification:

```python
import pickle
import hmac
import hashlib
import os

class SecurePickle:
    """Secure pickle wrapper with HMAC signature verification."""
    
    def __init__(self, secret_key: bytes = None):
        """Initialize with secret key for HMAC."""
        if secret_key is None:
            secret_key = os.environ.get('PICKLE_SECRET_KEY', '').encode()
            if not secret_key:
                raise ValueError("PICKLE_SECRET_KEY environment variable must be set")
        self.secret_key = secret_key
    
    def dumps(self, obj) -> bytes:
        """Serialize object with HMAC signature."""
        pickled_data = pickle.dumps(obj)
        signature = hmac.new(
            self.secret_key,
            pickled_data,
            hashlib.sha256
        ).digest()
        return signature + pickled_data
    
    def loads(self, data: bytes):
        """Deserialize object after verifying HMAC signature."""
        if len(data) < 32:  # SHA256 digest size
            raise ValueError("Invalid pickle data: too short")
        
        signature = data[:32]
        pickled_data = data[32:]
        
        expected_signature = hmac.new(
            self.secret_key,
            pickled_data,
            hashlib.sha256
        ).digest()
        
        if not hmac.compare_digest(signature, expected_signature):
            raise ValueError("Invalid pickle signature: data may be tampered")
        
        return pickle.loads(pickled_data)
    
    def dump(self, obj, file):
        """Serialize object to file with HMAC signature."""
        file.write(self.dumps(obj))
    
    def load(self, file):
        """Deserialize object from file after verifying HMAC signature."""
        return self.loads(file.read())

# Usage example
secure_pickle = SecurePickle()

# Saving
with open('secure_data.pkl', 'wb') as f:
    secure_pickle.dump(my_object, f)

# Loading (will raise exception if tampered)
with open('secure_data.pkl', 'rb') as f:
    my_object = secure_pickle.load(f)
```

### Option 3: Restricted Unpickler

Implement a restricted unpickler that only allows safe types:

```python
import pickle
import io

class RestrictedUnpickler(pickle.Unpickler):
    """Restricted unpickler that only allows safe types."""
    
    SAFE_MODULES = {
        'builtins': {'dict', 'list', 'tuple', 'set', 'frozenset', 
                     'int', 'float', 'str', 'bool', 'bytes', 'NoneType'},
        'collections': {'OrderedDict', 'defaultdict', 'Counter'},
        'datetime': {'datetime', 'date', 'time', 'timedelta'},
        'numpy': {'ndarray', 'dtype'},
        'torch': {'Tensor'},
    }
    
    def find_class(self, module, name):
        """Only allow safe classes to be unpickled."""
        if module in self.SAFE_MODULES:
            if name in self.SAFE_MODULES[module]:
                return super().find_class(module, name)
        
        raise pickle.UnpicklingError(
            f"Forbidden class: {module}.{name}"
        )

def restricted_loads(data: bytes):
    """Safely load pickle data with restricted types."""
    return RestrictedUnpickler(io.BytesIO(data)).load()

def restricted_load(file):
    """Safely load pickle file with restricted types."""
    return RestrictedUnpickler(file).load()

# Usage
with open('data.pkl', 'rb') as f:
    data = restricted_load(f)
```

## Implementation Plan

### Phase 1: Audit (Completed)
- [x] Identify all pickle usage locations
- [x] Assess risk level for each usage
- [x] Document current data formats

### Phase 2: Migration (Recommended)
- [ ] Replace simple data structures with JSON
- [ ] Migrate ML models to safetensors/torch with weights_only
- [ ] Implement SecurePickle wrapper for remaining cases
- [ ] Add integrity checks for checkpoint files

### Phase 3: Testing
- [ ] Verify all migrated code works correctly
- [ ] Test with existing data files
- [ ] Add security tests for pickle deserialization
- [ ] Document new data formats

### Phase 4: Deployment
- [ ] Deploy changes to staging
- [ ] Migrate existing pickle files
- [ ] Monitor for issues
- [ ] Deploy to production

## File-by-File Recommendations

### inspect_system_state.py
```python
# Current usage: Loading checkpoint state
# Recommendation: Use JSON for state, safetensors for model weights
# If model weights: Use torch.load(path, weights_only=True)
```

### demo/demo_graphix.py
```python
# Current usage: Cache data
# Recommendation: Replace with JSON cache
# Cache data is likely simple structures suitable for JSON
```

### src/vulcan/world_model/world_model_router.py
```python
# Current usage: World model state persistence
# Recommendation: Use JSON for state, implement SecurePickle if complex objects needed
```

### archive/orchestrator.py
```python
# Current usage: Checkpoint data
# Status: In archive folder, low priority if not in active use
# Recommendation: Update if brought back into active code
```

## Security Testing

Add these tests to verify fixes:

```python
# tests/test_pickle_security.py
import pytest
import pickle
import os

def test_pickle_not_used_for_untrusted_data():
    """Ensure pickle is not used to load untrusted data."""
    # Scan codebase for unsafe pickle usage
    import subprocess
    result = subprocess.run(
        ['grep', '-r', 'pickle.load', 'src/', '--include=*.py'],
        capture_output=True,
        text=True
    )
    
    # Allowlist for known secure usage (with HMAC)
    allowlist = [
        'src/utils/secure_pickle.py',  # Our secure wrapper
    ]
    
    unsafe_files = []
    for line in result.stdout.split('\n'):
        if line and not any(allowed in line for allowed in allowlist):
            unsafe_files.append(line)
    
    assert len(unsafe_files) == 0, f"Unsafe pickle.load found: {unsafe_files}"

def test_malicious_pickle_rejected():
    """Test that malicious pickle files are rejected."""
    # Create malicious pickle that tries to execute code
    import io
    import pickle
    
    class MaliciousObject:
        def __reduce__(self):
            import os
            return (os.system, ('echo pwned',))
    
    malicious_data = pickle.dumps(MaliciousObject())
    
    # Attempt to load with restricted unpickler
    from utils.secure_pickle import restricted_loads
    
    with pytest.raises(pickle.UnpicklingError):
        restricted_loads(malicious_data)

def test_secure_pickle_integrity():
    """Test that tampered pickle files are rejected."""
    from utils.secure_pickle import SecurePickle
    
    sp = SecurePickle(b'test-key-do-not-use')
    
    # Create signed pickle
    data = {'key': 'value'}
    signed = sp.dumps(data)
    
    # Tamper with data
    tampered = signed[:32] + b'X' + signed[33:]
    
    # Should raise exception
    with pytest.raises(ValueError, match="Invalid pickle signature"):
        sp.loads(tampered)
```

## Monitoring and Detection

Add monitoring for pickle usage:

```python
# In audit_log.py or similar
def log_pickle_usage(filepath: str, source: str):
    """Log whenever pickle files are loaded."""
    logger.warning(
        "Pickle file loaded",
        extra={
            'filepath': filepath,
            'source': source,
            'event': 'pickle_load',
            'severity': 'high'
        }
    )
```

## References

- [CWE-502: Deserialization of Untrusted Data](https://cwe.mitre.org/data/definitions/502.html)
- [Python Pickle Security](https://docs.python.org/3/library/pickle.html#restricting-globals)
- [OWASP Deserialization Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Deserialization_Cheat_Sheet.html)
- [HuggingFace SafeTensors](https://github.com/huggingface/safetensors)

## Timeline

- **Immediate:** Review all pickle usage with untrusted data
- **Week 1:** Implement SecurePickle wrapper
- **Week 2-3:** Migrate to safer formats where possible
- **Week 4:** Testing and validation
- **Week 5:** Production deployment

## Sign-off

This security fix must be reviewed and approved by:
- [ ] Security Team Lead
- [ ] Development Team Lead
- [ ] DevOps Team Lead

---

**Document Version:** 1.0  
**Created:** 2025-11-20  
**Status:** Pending Implementation
