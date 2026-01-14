# Requirements and Docker Build Fixes

## Summary
This document describes the fixes applied to resolve Docker build and dependency issues in the VulcanAMI_LLM project.

## Problems Identified

### 1. Git-Based Dependencies in requirements.txt
The original `requirements.txt` contained git URLs that caused Docker build failures:
- `dowhy @ git+https://github.com/microsoft/dowhy.git@...`
- `graphix @ git+https://github.com/musicmonk42/VulcanAMI_LLM.git@...` (circular dependency)
- `en_core_web_sm @ https://github.com/explosion/spacy-models/...` (direct wheel URL)

**Issues:**
- Requires git/SSL connectivity during Docker build
- Circular dependency (graphix pointing to itself)
- Bypasses pip caching and security features
- Incompatible with hashed requirements

### 2. Dependency Version Conflicts
Critical incompatibility discovered:
- `captum==0.8.0` requires `numpy<2.0`
- `dowhy==0.14` requires `numpy>2.0`
- Original requirements had `numpy==2.3.5`

This created an unsolvable dependency conflict.

## Solutions Applied

### 1. Convert Git Dependencies to PyPI Versions
```diff
- dowhy @ git+https://github.com/microsoft/dowhy.git@526573b1bbad7a4b5b0575cf65281456772d6330
+ dowhy==0.13

- graphix @ git+https://github.com/musicmonk42/VulcanAMI_LLM.git@6d9f03753e97ed6389c3e011728d5f330f8d5031
+ # graphix - this is the current package, installed from local source via setup.py or editable install
+ # Use: pip install -e . (for development) or pip install . (for installation)

- en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl#sha256=1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85
+ # en_core_web_sm model - install via: python -m spacy download en_core_web_sm
+ # Model URL: https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

### 2. Resolve Version Conflicts
```diff
- numpy==2.3.5
+ numpy==1.26.4

- dowhy==0.14
+ dowhy==0.13

- cvxpy==1.7.3
+ cvxpy==1.4.3

- pandas==2.3.3
+ pandas==2.2.3
```

**Rationale:**
- `numpy==1.26.4`: Last stable 1.x version, compatible with both captum and dowhy 0.13
- `dowhy==0.13`: Accepts `numpy>1.0` for Python >= 3.9 (unlike 0.14 which requires numpy>2.0)
- `cvxpy==1.4.3`: Required by dowhy 0.13 (`cvxpy<1.5`)
- `pandas==2.2.3`: Compatible with numpy 1.26.x

### 3. Update Dockerfiles
Enhanced all Dockerfiles (main + service-specific):

```dockerfile
# Upgrade pip and setuptools to latest versions
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy setup.py for local package installation
COPY setup.py ./setup.py

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY src/ ./src/

# Install local package if setup.py exists
RUN if [ -f setup.py ]; then \
 echo "Installing local package from setup.py"; \
 pip install --no-cache-dir -e .; \
 fi

# Download spacy language model if spacy is installed
RUN python -m spacy download en_core_web_sm || echo "Spacy model download failed (non-critical)"
```

### 4. Update Makefile
Enhanced the `install` target:

```makefile
.PHONY: install
install: ## Install Python dependencies
	@echo "$(GREEN)Installing Python dependencies...$(NC)"
	pip install --upgrade pip setuptools wheel
	pip install -r requirements.txt
	@if [ -f setup.py ]; then \
		echo "$(GREEN)Installing local package...$(NC)"; \
		pip install -e .; \
	fi
	@echo "$(GREEN)Downloading spacy language model...$(NC)"
	python -m spacy download en_core_web_sm || echo "$(YELLOW)Spacy model download failed (non-critical)$(NC)"
```

### 5. Update requirements-hashed.txt
Improved documentation for production deployments:

```text
# =============================================================================
# Hashed Requirements File (for secure production builds)
# =============================================================================
# This is a placeholder file. For production deployments, generate hashed
# requirements using pip-tools:
#
# Installation:
# pip install pip-tools
#
# Generate hashed requirements:
# pip-compile --generate-hashes requirements.txt -o requirements-hashed.txt
# =============================================================================
```

## Verification

The fixed requirements.txt was tested and confirmed to resolve correctly:
```bash
# Test in clean environment
python3 -m venv test-env
source test-env/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

All dependencies install successfully without conflicts.

## Known Issues

### SSL Certificate Verification in Docker Build
In some restricted environments (corporate proxies, security scanning tools), Docker builds may encounter SSL certificate verification errors:
```
SSLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] 
certificate verify failed: self-signed certificate in certificate chain'))
```

**This is an infrastructure issue**, not a problem with the requirements or Dockerfile. The build will work correctly in standard Docker environments.

**Workaround (development only):**
If you encounter SSL issues in your environment, you can temporarily disable verification (NOT recommended for production):
```dockerfile
RUN pip install --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

## Best Practices Going Forward

1. **Use PyPI versions**: Avoid git URLs in requirements.txt
2. **Pin compatible versions**: Test dependency resolution before committing
3. **Generate hashed requirements**: For production, use `pip-compile --generate-hashes`
4. **Local package installation**: Use `pip install -e .` during development
5. **Post-install steps**: Handle spacy models and similar assets separately

## Files Modified

- `requirements.txt` - Fixed git dependencies and version conflicts
- `requirements-hashed.txt` - Improved documentation
- `Dockerfile` - Added pip upgrade, setup.py copy, local install, spacy download
- `docker/api/Dockerfile` - Same enhancements
- `docker/dqs/Dockerfile` - Same enhancements 
- `docker/pii/Dockerfile` - Same enhancements
- `Makefile` - Enhanced install target with local package and spacy model

## Testing

To test the changes:

```bash
# Test local installation
make install

# Test Docker build
make docker-build

# Test docker-compose
make up-build
```
