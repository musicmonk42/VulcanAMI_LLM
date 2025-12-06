# Instructions to Update requirements-hashed.txt

## Overview

The `requirements-hashed.txt` file is generated using `pip-compile` with hash verification for enhanced security. After cleaning up `requirements.txt`, this file needs to be regenerated.

## Prerequisites

1. Python 3.12 environment
2. `pip-tools` installed
3. All dependencies available for download

## Steps to Regenerate

### 1. Install pip-tools

```bash
pip install pip-tools
```

### 2. Generate Hashed Requirements

```bash
pip-compile --generate-hashes --output-file=requirements-hashed.txt requirements.txt
```

This command will:
- Read `requirements.txt`
- Resolve all dependencies
- Download package metadata
- Generate SHA256 hashes for each package
- Write to `requirements-hashed.txt`

### 3. Verify the Output

```bash
# Check file was created
ls -lh requirements-hashed.txt

# Verify it contains hashes
head -50 requirements-hashed.txt
```

### 4. Test Installation

```bash
# In a fresh virtual environment
python -m venv test_env
source test_env/bin/activate  # or test_env\Scripts\activate on Windows

# Install using hashed requirements
pip install --require-hashes -r requirements-hashed.txt

# Test imports
python -c "import numpy, torch, transformers, fastapi"
```

## What Changed

After the cleanup:
- **Before:** 523 packages
- **After:** 440 packages
- **Reduction:** 81 packages (15.5%)

The hashed file will be correspondingly smaller and faster to install.

## Security Note

The `--generate-hashes` flag provides:
- Protection against package tampering
- Verification of package integrity
- Compliance with security best practices
- Reproducible builds

Each package version has its SHA256 hash verified during installation.

## Alternative: Automated Regeneration

You can add this to your CI/CD pipeline:

```yaml
# .github/workflows/update-hashed-requirements.yml
name: Update Hashed Requirements

on:
  push:
    paths:
      - 'requirements.txt'

jobs:
  update-hashed:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      
      - name: Install pip-tools
        run: pip install pip-tools
      
      - name: Generate hashed requirements
        run: pip-compile --generate-hashes --output-file=requirements-hashed.txt requirements.txt
      
      - name: Commit changes
        run: |
          git config --local user.email "action@github.com"
          git config --local user.name "GitHub Action"
          git add requirements-hashed.txt
          git commit -m "Auto-update requirements-hashed.txt" || echo "No changes"
          git push
```

## Troubleshooting

### Issue: Package not found
**Solution:** Ensure package exists on PyPI and version is correct

### Issue: Hash mismatch
**Solution:** Re-run pip-compile to regenerate hashes

### Issue: Slow generation
**Solution:** Normal for large dependency trees. Be patient.

## Next Steps

1. Run the regeneration command in your development environment
2. Test the new hashed requirements
3. Commit the updated `requirements-hashed.txt`
4. Update your Docker builds to use the new file
5. Update CI/CD pipelines if needed

## Questions?

Contact the development team or refer to:
- pip-tools documentation: https://pip-tools.readthedocs.io/
- REQUIREMENTS_CLEANUP_SUMMARY.md for details on what was removed
