# Security Scan Fixes Summary

This document summarizes the security improvements implemented to address vulnerabilities found in security scans.

## Terraform Infrastructure Changes (infra/terraform/main.tf)

### 1. Default VPC Security Group (CKV2_AWS_12)
**Issue**: Default VPC security group was not explicitly restricting all traffic.

**Fix**: Added `aws_default_security_group` resource with empty ingress and egress rules:
```hcl
resource "aws_default_security_group" "default" {
  vpc_id = aws_vpc.main.id
  ingress = []
  egress  = []
  tags = merge(local.common_tags, { Name = "${local.name_prefix}-default-sg" })
}
```

### 2. S3 Bucket Public Access Block (CKV2_AWS_6)
**Issue**: S3 buckets needed public access blocks.

**Status**: Already implemented for all buckets. No changes required.

### 3. S3 Bucket Versioning (CKV_AWS_21)
**Issue**: S3 bucket versioning needed to be enabled.

**Status**: Already implemented for all buckets. No changes required.

### 4. S3 Bucket Cross-Region Replication (CKV_AWS_144)
**Issue**: Log buckets lacked cross-region replication for disaster recovery.

**Fix**: 
- Created replica buckets in secondary region for both logs and CloudFront logs buckets
- Added KMS key in secondary region (`aws_kms_key.s3_secondary`) for proper encryption
- Configured replication with encryption, versioning, and delete marker replication
- Applied public access blocks to all replica buckets

## Kubernetes Security Improvements

### 5. Secrets as Files (CKV_K8S_35)
**Issue**: Secrets were exposed as environment variables, which is less secure.

**Fix**: Updated all deployments to mount secrets as files:

#### postgres-deployment.yaml
- Changed from `POSTGRES_PASSWORD` env var to `POSTGRES_PASSWORD_FILE`
- Mounted secret at `/etc/secrets/postgres-password`

#### redis-deployment.yaml
- Reads password from `/etc/secrets/redis-password` into a variable
- Uses variable in redis-server command to avoid process list exposure

#### api-deployment.yaml
- Changed all secrets to use `*_FILE` environment variables:
  - `JWT_SECRET_KEY_FILE`
  - `BOOTSTRAP_KEY_FILE`
  - `POSTGRES_PASSWORD_FILE`
  - `REDIS_PASSWORD_FILE`
  - `MINIO_SECRET_KEY_FILE`

#### configs/helm_chart.yaml
- Changed to `GRAPHIX_API_KEY_FILE`
- Mounted secret at `/etc/secrets/api-key`

### 6. Image Pinning (CKV_K8S_14, CKV_K8S_43)
**Issue**: Using `:latest` tags is not recommended for production.

**Fix**: Added documentation and comments in:
- `k8s/base/api-deployment.yaml`: Added comments showing how to use specific tags or digests
- `helm/vulcanami/values.yaml`: Added comprehensive comments with examples

**Recommendation**: Use image digests in production:
```yaml
image: ghcr.io/musicmonk42/vulcanami_llm-api@sha256:<digest>
```

### 7. High UID (CKV_K8S_40)
**Issue**: Containers should run as high UIDs (≥10000) for enhanced security.

**Fix**: Updated all deployments to use high UIDs:
- `api-deployment.yaml`: UID 10001
- `redis-deployment.yaml`: UID 10000
- `configs/helm_chart.yaml`: UID 10001
- `helm/vulcanami/values.yaml`: UID 10001
- `postgres-deployment.yaml`: UID 70 (documented exception - required by official PostgreSQL image)

### 8. Read-Only Root Filesystem (CKV_K8S_22)
**Issue**: Containers should use read-only root filesystems where possible.

**Status**: 
- Already implemented for `api-deployment.yaml` with emptyDir volumes for writable paths
- Documented as not feasible for postgres and redis due to their write requirements
- No changes required

## Application Code Changes Required

**Important**: The application code needs to be updated to support reading secrets from files instead of environment variables. The following changes are needed:

1. Update secret loading logic to check for `*_FILE` environment variables
2. If a `*_FILE` variable is set, read the secret from that file path
3. Fall back to the direct environment variable for backward compatibility

Example pseudocode:
```python
def get_secret(name):
    file_var = f"{name}_FILE"
    if file_var in os.environ:
        with open(os.environ[file_var], 'r') as f:
            return f.read().strip()
    return os.environ.get(name)
```

## Testing Recommendations

1. **Terraform**: Run `terraform plan` to verify no unexpected changes
2. **Kubernetes**: 
   - Test deployments in a development environment first
   - Verify secret mounts are working correctly
   - Ensure applications can read secrets from files
   - Test that high UIDs don't cause permission issues with volumes

## Notes

- All changes maintain backward compatibility where possible
- Comprehensive comments added for production deployment considerations
- Special attention needed for redis and postgres due to their specific UID requirements
- Consider building custom images with appropriate UIDs for production use
