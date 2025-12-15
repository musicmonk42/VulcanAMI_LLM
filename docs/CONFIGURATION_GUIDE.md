# FAISS and LLVM Configuration Guide

## Overview

This guide explains the enhanced configuration system for FAISS (Facebook AI Similarity Search) and LLVM backend components, including CPU capability detection and informative diagnostic messages.

## FAISS Configuration

### CPU Instruction Set Detection

The FAISS configuration module automatically detects CPU capabilities and provides informative messages about which instruction set is being used:

- **AVX-512**: Optimal performance for vector operations
- **AVX2**: Good performance, widely supported on modern CPUs
- **AVX**: Standard performance for older CPUs
- **Scalar**: Fallback mode without vector extensions

### Usage

```python
from src.utils.faiss_config import initialize_faiss, get_faiss_config_info

# Initialize FAISS with automatic CPU detection
faiss, available, instruction_set = initialize_faiss()

if available:
    # FAISS is available, use it for vector operations
    index = faiss.IndexFlatL2(dimension)
else:
    # Fall back to NumPy-based vector search
    pass

# Get detailed configuration information
config = get_faiss_config_info()
print(f"Instruction Set: {config['instruction_set']}")
print(f"CPU Capabilities: {config['cpu_capabilities']}")
print(f"Recommendations: {config['recommendations']}")
```

### Expected Messages

#### AVX512 Available
```
✓ FAISS initialized with AVX-512 (DQ, BW, VL) support (High Performance)
```

#### AVX2 Available (AVX512 not available)
```
✓ FAISS initialized with AVX2 support (Medium Performance). 
Note: AVX512 not available on this CPU. Performance is optimal for this hardware.
```

#### AVX Only
```
ℹ FAISS initialized with AVX support (Standard Performance). 
Note: AVX2/AVX512 not available. Consider upgrading hardware for better vector search performance.
```

#### FAISS Not Available
```
FAISS not available (ModuleNotFoundError). 
Falling back to NumPy-based vector search. 
Install faiss-cpu or faiss-gpu for better performance: pip install faiss-cpu
```

### Warning Suppression

The configuration module automatically suppresses expected FAISS internal warnings about `swigfaiss_avx512` module loading failures. These warnings are normal when AVX512 is not available and FAISS falls back to AVX2.

## LLVM Backend Configuration

### Execution Engine Diagnostics

The LLVM backend now provides detailed diagnostic information when the execution engine cannot be created:

```
LLVM execution engine creation failed (RuntimeError): ...
Optimization level: 2
Triple: x86_64-unknown-linux-gnu
Impact: JIT compilation unavailable. Compiler can still generate and analyze IR. 
For production JIT execution, ensure LLVM/MCJIT is properly configured.
CPU: x86_64, Best instruction set: AVX2
```

### Expected Behavior

#### Success
```
✓ LLVM execution engine created successfully (optimization level: 2)
```

#### Failure (Expected in Some Environments)
The execution engine creation may fail in certain environments (containers, test environments, etc.). This is expected and does not prevent:
- IR generation
- IR analysis
- IR optimization
- Compilation to object code

Only JIT (Just-In-Time) execution is unavailable when the execution engine fails to initialize.

## Performance Optimization Recommendations

### For Development

- **FAISS**: The system gracefully falls back to NumPy-based vector search when FAISS is not available. This is acceptable for development and testing.
- **LLVM**: The compiler works without an execution engine for IR generation and analysis tasks.

### For Production

1. **Install FAISS with CPU Optimization**:
   ```bash
   # For CPU-only systems
   pip install faiss-cpu
   
   # For GPU systems
   pip install faiss-gpu
   ```

2. **Verify CPU Capabilities**:
   ```python
   from src.utils.cpu_capabilities import get_capability_summary
   print(get_capability_summary())
   ```

3. **Compile FAISS with AVX512 Support** (Advanced):
   - If your CPU supports AVX512 but FAISS doesn't use it:
   - Build FAISS from source with AVX512 flags enabled
   - Refer to FAISS documentation for compilation instructions

4. **LLVM Configuration**:
   - Ensure llvmlite is properly installed: `pip install llvmlite`
   - Verify LLVM target initialization for your platform

## Troubleshooting

### FAISS Shows Wrong Instruction Set

**Problem**: System reports AVX2 but CPU has AVX512

**Solution**: 
1. Check CPU capabilities: `cat /proc/cpuinfo | grep avx512`
2. If AVX512 is present, FAISS may need recompilation with AVX512 support
3. This is a performance optimization, not a correctness issue

### LLVM Execution Engine Always Fails

**Problem**: Execution engine creation consistently fails

**Causes**:
- Platform limitations (some ARM platforms)
- Container restrictions
- Missing LLVM components

**Solutions**:
1. Verify llvmlite installation: `pip install --upgrade llvmlite`
2. Check platform compatibility with LLVM MCJIT
3. For containers, ensure proper LLVM libraries are available

### Performance Issues

**Problem**: Vector operations are slow

**Solutions**:
1. Install FAISS if not present
2. Verify CPU instruction set detection is correct
3. Consider hardware upgrade for AVX512 support
4. Profile code to identify bottlenecks

## Configuration Files

### Environment Variables

No environment variables are required. Configuration is automatic based on:
- Available libraries (FAISS, llvmlite)
- CPU capabilities (detected at runtime)
- Platform characteristics

### Manual Override (Advanced)

For testing or debugging, you can manually control initialization:

```python
import warnings

# Disable automatic warning suppression
warnings.filterwarnings('default')

# Then import
from src.utils.faiss_config import initialize_faiss
```

## Monitoring and Metrics

The configuration modules log important initialization events:

- INFO level: Successful initialization, capability detection
- WARNING level: Degraded performance mode, fallbacks
- DEBUG level: Detailed diagnostic information

Configure logging to capture these events:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## API Reference

### faiss_config Module

#### `initialize_faiss() -> Tuple[Optional[Any], bool, Optional[str]]`
Initialize FAISS with CPU capability detection.

**Returns**: (faiss_module, is_available, instruction_set)

#### `get_faiss() -> Optional[Any]`
Get FAISS module, initializing if necessary.

#### `is_faiss_available() -> bool`
Check if FAISS is available.

#### `get_faiss_instruction_set() -> Optional[str]`
Get the detected instruction set.

#### `get_faiss_config_info() -> dict`
Get comprehensive configuration information.

### cpu_capabilities Module

#### `get_cpu_capabilities() -> CPUCapabilities`
Get cached CPU capabilities (singleton pattern).

#### `detect_cpu_capabilities() -> CPUCapabilities`
Detect CPU capabilities for the current platform.

#### `get_capability_summary() -> str`
Get human-readable summary of CPU capabilities.

## Related Documentation

- [CPU Capabilities Module](../src/utils/cpu_capabilities.py)
- [FAISS Official Documentation](https://github.com/facebookresearch/faiss)
- [LLVM Documentation](https://llvm.org/docs/)
- [llvmlite Documentation](https://llvmlite.readthedocs.io/)
