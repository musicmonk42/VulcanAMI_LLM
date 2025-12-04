# Dependency Management Guide

This guide explains how to manage Python dependencies for the VulcanAMI/Graphix Vulcan project.

## Overview

The project uses multiple requirements files for different purposes:

- **requirements.txt**: Production dependencies with pinned versions
- **requirements-hashed.txt**: Production dependencies with SHA256 hashes for security
- **requirements-dev.txt**: Development tools (testing, linting, code quality)

## Dependency Files

### requirements.txt

Contains all production dependencies with exact version pins:

```
# Example format
Flask==3.1.2
pytest==9.0.1
numpy==1.26.4
```

**Purpose**: Ensures reproducible builds across all environments.

### requirements-hashed.txt

Contains the same dependencies as requirements.txt but with SHA256 hashes:

```
# Example format (auto-generated)
Flask==3.1.2 \
    --hash=sha256:abc123... \
    --hash=sha256:def456...
```

**Purpose**: 
- Prevents supply chain attacks by verifying package integrity
- Ensures downloaded packages match expected checksums
- Required for high-security deployments

### requirements-dev.txt

Contains additional development tools that are NOT needed in production:

```
# Code Formatting
black==24.10.0
isort==5.13.2

# Linting and Code Quality
flake8==7.1.1
pylint==3.3.2
mypy==1.13.0

# Security Scanning
bandit==1.7.10

# Dependency Management
pip-tools==7.4.1

# Type Checking Support
types-PyYAML==6.0.12.20240917
types-requests==2.32.0.20241016
types-redis==4.6.0.20241004

# Additional Development Tools
ipython==8.29.0
ipdb==0.13.13

# Documentation
sphinx==8.1.3
sphinx-rtd-theme==3.0.2
```

**Purpose**: Provides additional tools developers need for code quality, linting, and documentation.

**Note**: Testing tools (pytest, pytest-cov, pytest-asyncio, pytest-timeout, coverage) are already included in requirements.txt and do not need to be in requirements-dev.txt.

## Installation

### Production Environment

```bash
# Option 1: Standard installation
pip install -r requirements.txt

# Option 2: Secure installation with hash verification
pip install --require-hashes -r requirements-hashed.txt
```

### Development Environment

```bash
# Install both production and development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Or combine in one command
pip install -r requirements.txt -r requirements-dev.txt
```

## Using pip-tools

### Installation

pip-tools is included in requirements-dev.txt:

```bash
pip install pip-tools
```

### Regenerating Hashed Requirements

When you add or update dependencies in requirements.txt:

```bash
# Generate hashed requirements file
pip-compile --generate-hashes requirements.txt -o requirements-hashed.txt

# This creates a new requirements-hashed.txt with SHA256 hashes for all packages
```

### Upgrading Dependencies

To upgrade all dependencies to their latest compatible versions:

```bash
# Upgrade all packages
pip-compile --upgrade requirements.txt -o requirements-hashed.txt

# Upgrade specific package
pip-compile --upgrade-package flask requirements.txt -o requirements-hashed.txt
```

### Syncing Your Environment

To ensure your environment matches the requirements files:

```bash
# Sync production dependencies
pip-sync requirements.txt

# Sync with development dependencies
pip-sync requirements.txt requirements-dev.txt
```

## Adding New Dependencies

### Step 1: Add to requirements.txt

Edit requirements.txt and add the new dependency:

```
# Add at the appropriate location (keep alphabetical)
new-package==1.2.3
```

### Step 2: Regenerate Hashed Requirements

```bash
pip-compile --generate-hashes requirements.txt -o requirements-hashed.txt
```

### Step 3: Test the Changes

```bash
# Create a clean virtual environment
python -m venv test_env
source test_env/bin/activate

# Install from hashed requirements
pip install --require-hashes -r requirements-hashed.txt

# Run tests
pytest tests/

# Deactivate and remove test environment
deactivate
rm -rf test_env
```

### Step 4: Commit Both Files

```bash
git add requirements.txt requirements-hashed.txt
git commit -m "Add new-package==1.2.3"
```

## Security Best Practices

### 1. Always Use Pinned Versions

❌ **Bad**: `flask`  
❌ **Bad**: `flask>=2.0`  
✅ **Good**: `flask==3.1.2`

**Why**: Unpinned versions lead to non-reproducible builds and potential security issues.

### 2. Use Hashed Requirements in Production

```bash
# Always use --require-hashes in production
pip install --require-hashes -r requirements-hashed.txt
```

**Why**: Prevents package tampering and supply chain attacks.

### 3. Regular Security Audits

```bash
# Check for known vulnerabilities
pip-audit -r requirements.txt

# Update vulnerable packages
pip-compile --upgrade-package vulnerable-package requirements.txt -o requirements-hashed.txt
```

### 4. Never Commit Secrets

- Keep .env files out of version control
- Use environment variables or secret managers
- Review .gitignore to ensure secrets are excluded

### 5. Document Placeholder Values

All placeholder values in .env.example include comments:

```bash
# NOTE: This is a placeholder value only, not a real secret
JWT_SECRET_KEY=your-jwt-secret-key-here
```

**Why**: Helps prevent false positive security alerts during automated scanning.

## CI/CD Integration

### GitHub Actions

The CI/CD pipeline uses hashed requirements for security:

```yaml
- name: Install dependencies
  run: |
    pip install --upgrade pip
    pip install --require-hashes -r requirements-hashed.txt
```

### Docker Builds

Dockerfiles use hashed requirements:

```dockerfile
COPY requirements-hashed.txt .
RUN pip install --require-hashes -r requirements-hashed.txt
```

## Troubleshooting

### Hash Mismatch Error

**Error**: "Hash of <package> doesn't match expected hash"

**Solution**: Regenerate hashed requirements:

```bash
pip-compile --generate-hashes requirements.txt -o requirements-hashed.txt
```

### Dependency Conflicts

**Error**: "Cannot install package-a and package-b"

**Solution**: Use pip-compile to resolve conflicts:

```bash
pip-compile requirements.txt
# Review output for conflict resolution
pip-compile --generate-hashes requirements.txt -o requirements-hashed.txt
```

### Missing Dependencies

**Error**: "ModuleNotFoundError: No module named 'X'"

**Solution**: 
1. Check if the package is in requirements.txt
2. If missing, add it and regenerate hashed requirements
3. Reinstall dependencies

## Development Workflow

### Daily Development

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Ensure dependencies are up to date
pip-sync requirements.txt requirements-dev.txt

# 3. Make changes and run tests
pytest tests/

# 4. Run code quality checks
black src/ tests/
flake8 src/ tests/
mypy src/
```

### Before Committing

```bash
# Run all quality checks
make lint
make test-cov
make lint-security

# Verify hashed requirements are current
pip-compile --generate-hashes requirements.txt -o requirements-hashed.txt
git diff requirements-hashed.txt  # Should show no changes if current
```

### Before Releasing

```bash
# Update all dependencies (if appropriate)
pip-compile --upgrade requirements.txt -o requirements-hashed.txt

# Test with updated dependencies
pip-sync requirements.txt requirements-dev.txt
pytest tests/

# Run security audit
pip-audit -r requirements.txt

# Commit updated requirements
git add requirements.txt requirements-hashed.txt
git commit -m "Update dependencies for release X.Y.Z"
```

## Reference Commands

### Quick Reference

```bash
# Install dependencies
pip install -r requirements.txt                    # Production
pip install -r requirements-dev.txt                # Development
pip install --require-hashes -r requirements-hashed.txt  # Secure

# Update hashed requirements
pip-compile --generate-hashes requirements.txt -o requirements-hashed.txt

# Upgrade dependencies
pip-compile --upgrade requirements.txt -o requirements-hashed.txt

# Sync environment
pip-sync requirements.txt requirements-dev.txt

# Security audit
pip-audit -r requirements.txt

# Check for outdated packages
pip list --outdated
```

## Additional Resources

- [pip-tools Documentation](https://github.com/jazzband/pip-tools)
- [PEP 665: Specifying Installation Requirements](https://peps.python.org/pep-0665/)
- [Python Packaging User Guide](https://packaging.python.org/)
- [pip-audit Documentation](https://github.com/pypa/pip-audit)

## Getting Help

If you encounter issues with dependency management:

1. Check this guide first
2. Review the [TESTING_GUIDE.md](../TESTING_GUIDE.md)
3. Check the [CI_CD.md](../CI_CD.md) documentation
4. Open an issue in the repository with:
   - Your Python version
   - Your pip and pip-tools versions
   - The exact error message
   - Steps to reproduce
