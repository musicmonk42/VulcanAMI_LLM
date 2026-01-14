# Infrastructure Security Fixes - CI/CD Integration Guide

This document describes the critical security fixes applied to the VulcanAMI infrastructure and how they are validated in CI/CD pipelines.

## 🔒 Security Fixes Applied

### Helm Chart (`helm/vulcanami/values.yaml`)

#### 1. **Removed `latest` Tag** ✅
- **Issue**: Using `latest` tag causes unpredictable deployments and impossible rollbacks
- **Fix**: Image tag now requires explicit version specification
- **CI/CD Validation**: `infrastructure-validation.yml` workflow checks for `latest` tag
```yaml
image:
 tag: "" # REQUIRED: Must be set to specific version like "v1.0.0"
```

#### 2. **Removed Hardcoded MinIO Credentials** ✅
- **Issue**: Default credentials `minioadmin` were committed to git
- **Fix**: Replaced with secret references
- **CI/CD Validation**: Workflow checks for hardcoded credentials
```yaml
minio:
 accessKeySecretName: "" # REQUIRED: Reference to external secret
 accessKeySecretKey: "accessKey"
 secretKeySecretName: "" # REQUIRED: Reference to external secret
 secretKeySecretKey: "secretKey"
```

#### 3. **Enabled Read-Only Root Filesystem** ✅
- **Issue**: Writable filesystem increases attack surface
- **Fix**: Enabled `readOnlyRootFilesystem: true`
- **CI/CD Validation**: Workflow checks this setting
```yaml
securityContext:
 readOnlyRootFilesystem: true
```

#### 4. **Required Secrets Validation** ✅
- **Issue**: Empty secrets allowed vulnerable deployments
- **Fix**: Added REQUIRED comments and documented need for validation
```yaml
secrets:
 jwtSecretKey: "" # REQUIRED
 bootstrapKey: "" # REQUIRED
 postgresPassword: "" # REQUIRED
 redisPassword: "" # REQUIRED
```

#### 5. **Added Health Probes** ✅
- **Issue**: No liveness/readiness probes = no automatic recovery
- **Fix**: Added comprehensive health checks
- **CI/CD Validation**: Workflow checks for presence of probes
```yaml
livenessProbe:
 httpGet:
 path: /health
 port: 8000
 initialDelaySeconds: 30
 periodSeconds: 10

readinessProbe:
 httpGet:
 path: /ready
 port: 8000
 initialDelaySeconds: 10
 periodSeconds: 5
```

#### 6. **Added Pod Disruption Budget** ✅
- **Issue**: All pods could be terminated during maintenance
- **Fix**: Configured PDB to ensure availability
```yaml
podDisruptionBudget:
 enabled: true
 minAvailable: 1
```

#### 7. **Added Pod Anti-Affinity** ✅
- **Issue**: All replicas could run on same node
- **Fix**: Configured pod anti-affinity rules
```yaml
affinity:
 podAntiAffinity:
 preferredDuringSchedulingIgnoredDuringExecution:
 - weight: 100
 podAffinityTerm:
 topologyKey: kubernetes.io/hostname
```

#### 8. **Reduced Rate Limit** ✅
- **Issue**: 100 req/sec was too high for DoS protection
- **Fix**: Reduced to 10 req/sec
```yaml
nginx.ingress.kubernetes.io/rate-limit: "10"
```

---

### Terraform (`infra/terraform/`)

#### 1. **Fixed RDS `timestamp()` Bug** ✅
- **Issue**: Using `timestamp()` caused resource recreation on every apply
- **Fix**: Removed `timestamp()` from final snapshot identifier
- **CI/CD Validation**: Workflow checks for `timestamp()` usage
```hcl
# Before:
final_snapshot_identifier = "${local.name_prefix}-db-final-snapshot-${replace(timestamp(), ":", "-")}"

# After:
final_snapshot_identifier = "${local.name_prefix}-db-final-snapshot"
```

#### 2. **Removed Lambda@Edge Reserved Concurrency** ✅
- **Issue**: Lambda@Edge doesn't support `reserved_concurrent_executions`
- **Fix**: Removed the parameter with explanatory comment
- **CI/CD Validation**: Workflow checks Lambda@Edge configurations
```hcl
resource "aws_lambda_function" "edge_auth" {
 # NOTE: Lambda@Edge does not support reserved_concurrent_executions
 # ... other configuration
}
```

#### 3. **Fixed Cross-Region S3 Logging** ✅
- **Issue**: Secondary bucket tried to log to primary region bucket
- **Fix**: Created separate logs bucket in secondary region
- **CI/CD Validation**: Terraform validate checks resource dependencies
```hcl
# Added:
resource "aws_s3_bucket" "logs_secondary" {
 provider = aws.secondary
 bucket = local.bucket_logs_secondary
}
```

#### 4. **Fixed Lambda ZIP File Reference** ✅
- **Issue**: Referenced non-existent ZIP file
- **Fix**: Created Lambda function source and used `data.archive_file`
```hcl
data "archive_file" "edge_auth" {
 type = "zip"
 source_file = "${path.module}/lambda/edge-auth.js"
 output_path = "${path.module}/lambda/edge-auth.zip"
}
```

#### 5. **Split ACM Certificate Variables** ✅
- **Issue**: CloudFront needs us-east-1 certificate, ALB needs regional certificate
- **Fix**: Created separate variables for each
```hcl
variable "acm_certificate_arn" {
 description = "ACM certificate ARN for CloudFront (must be in us-east-1)"
}

variable "alb_certificate_arn" {
 description = "ACM certificate ARN for ALB (must be in the same region as the ALB)"
}
```

#### 6. **Fixed Default `allowed_ip_ranges`** ✅
- **Issue**: Default `0.0.0.0/0` allowed internet access
- **Fix**: Changed to empty list requiring explicit specification
- **CI/CD Validation**: Workflow checks for `0.0.0.0/0` in defaults
```hcl
variable "allowed_ip_ranges" {
 default = [] # Empty default requires explicit IP range specification
}
```

#### 7. **Added CloudFront Logs Bucket Policy** ✅
- **Issue**: Missing permissions for CloudFront to write logs
- **Fix**: Added bucket policy with CloudFront service principal
```hcl
data "aws_iam_policy_document" "cloudfront_logs_bucket" {
 statement {
 principals {
 type = "Service"
 identifiers = ["cloudfront.amazonaws.com"]
 }
 actions = ["s3:PutObject"]
 resources = ["${aws_s3_bucket.cloudfront_logs[0].arn}/*"]
 }
}
```

#### 8. **Enabled Redis Special Characters** ✅
- **Issue**: Redis auth token limited to alphanumeric (weaker security)
- **Fix**: Enabled special characters
```hcl
resource "random_password" "redis_auth_token" {
 special = true # Redis supports special characters in auth tokens
}
```

#### 9. **Added Health Check Documentation** ✅
- **Issue**: Health checks assumed curl was installed
- **Fix**: Added documentation notes
```hcl
# NOTE: Health check requires curl to be installed in the container image
healthCheck = {
 command = ["CMD-SHELL", "curl -f http://localhost:${var.port}/health || exit 1"]
}
```

#### 10. **Added Auto-Scaling Validation** ✅
- **Issue**: No validation that min <= max capacity
- **Fix**: Added lifecycle precondition
```hcl
lifecycle {
 precondition {
 condition = var.auto_scaling_min_capacity <= var.auto_scaling_max_capacity
 error_message = "Auto-scaling minimum capacity must be less than or equal to maximum capacity."
 }
}
```

#### 11. **Fixed CloudWatch Logs Retention** ✅
- **Issue**: Confusing behavior with default vs. enforced retention
- **Fix**: Updated default to 365 days and improved documentation
```hcl
variable "cloudwatch_retention_days" {
 description = "CloudWatch Logs retention in days. Note: Production deployments enforce minimum 365 days regardless of this value."
 default = 365
}
```

---

## 🔄 CI/CD Integration

### New Workflow: `infrastructure-validation.yml`

This workflow validates all infrastructure configurations:

#### Jobs:
1. **validate-helm**: Lint and validate Helm charts
2. **validate-terraform**: Validate Terraform syntax and security
3. **validate-docker**: Validate Dockerfiles and docker-compose files
4. **validate-reproducibility**: Check for pinned versions and reproducible builds
5. **summary**: Generate security checklist and summary

#### Triggers:
- Push to main/develop branches
- Pull requests
- Changes to infrastructure files
- Manual workflow dispatch

### Existing Workflows Enhanced

#### `ci.yml` - Already includes:
- Docker build validation
- Security linting with Bandit
- Dependency vulnerability scanning with pip-audit

#### `security.yml` - Already includes:
- Checkov scanning for Terraform, Docker, Kubernetes
- Kubesec scanning for Kubernetes manifests

#### `deploy.yml` - Already includes:
- Helm deployment with version-specific tags
- Environment-specific values files

---

## 🚀 Usage in CI/CD

### For Developers

#### Before Committing:
```bash
# Validate Helm charts locally
helm lint helm/vulcanami

# Format Terraform
cd infra/terraform
terraform fmt -recursive

# Validate Terraform
terraform init -backend=false
terraform validate
```

#### Required Environment Variables for Deployment:
```bash
# Helm deployment requires:
export IMAGE_TAG="v1.0.0" # Never use 'latest'
export JWT_SECRET_KEY="$(openssl rand -base64 48)"
export BOOTSTRAP_KEY="$(openssl rand -base64 32)"
export POSTGRES_PASSWORD="$(openssl rand -base64 32)"
export REDIS_PASSWORD="$(openssl rand -base64 32)"
export MINIO_ACCESS_KEY="$(openssl rand -base64 32)"
export MINIO_SECRET_KEY="$(openssl rand -base64 48)"

# Deploy with Helm
helm upgrade --install vulcanami ./helm/vulcanami \
 --set image.tag=$IMAGE_TAG \
 --set secrets.jwtSecretKey=$JWT_SECRET_KEY \
 --set secrets.bootstrapKey=$BOOTSTRAP_KEY \
 --set secrets.postgresPassword=$POSTGRES_PASSWORD \
 --set secrets.redisPassword=$REDIS_PASSWORD \
 --set minio.accessKeySecretName=minio-credentials \
 --set minio.secretKeySecretName=minio-credentials
```

### For CI/CD

#### GitHub Actions Secrets Required:
- `JWT_SECRET_KEY`
- `POSTGRES_PASSWORD`
- `REDIS_PASSWORD`
- `MINIO_ROOT_USER`
- `MINIO_ROOT_PASSWORD`
- `AWS_ACCESS_KEY_ID` (for Terraform)
- `AWS_SECRET_ACCESS_KEY` (for Terraform)

#### Automated Checks:
1. ✅ Helm chart linting
2. ✅ Terraform formatting and validation
3. ✅ Docker image security scanning
4. ✅ Dependency vulnerability scanning
5. ✅ Infrastructure security anti-pattern detection
6. ✅ Reproducible build verification

---

## 📋 Pre-Deployment Checklist

Before deploying to production, ensure:

### Helm:
- [ ] Image tag is set to specific version (not `latest`)
- [ ] All secrets are provided via external secret management
- [ ] Health probes are configured
- [ ] Resource limits are set
- [ ] Pod Disruption Budget is enabled
- [ ] Anti-affinity rules are configured

### Terraform:
- [ ] All variables are set in terraform.tfvars
- [ ] ACM certificates are in correct regions
- [ ] Allowed IP ranges are explicitly defined
- [ ] Auto-scaling min/max are validated
- [ ] Backup configurations are reviewed

### Docker:
- [ ] All images use specific version tags
- [ ] No hardcoded secrets in docker-compose files
- [ ] Health checks are defined for all services
- [ ] Resource limits are set

---

## 🔍 Monitoring and Validation

### Post-Deployment Validation:
```bash
# Check Helm release status
helm status vulcanami -n production

# Verify pods are running
kubectl get pods -n production

# Check pod security context
kubectl get pod <pod-name> -n production -o jsonpath='{.spec.securityContext}'

# Verify health probes
kubectl describe pod <pod-name> -n production | grep -A5 "Liveness\|Readiness"
```

### Terraform State Validation:
```bash
# Verify no drift
terraform plan

# Check for timestamp() related changes
terraform plan | grep -i "snapshot"

# Validate S3 logging configuration
aws s3api get-bucket-logging --bucket <secondary-bucket>
```

---

## 📚 Additional Resources

- [Helm Best Practices](https://helm.sh/docs/chart_best_practices/)
- [Terraform Security Best Practices](https://www.terraform.io/docs/language/modules/develop/index.html)
- [Docker Security Best Practices](https://docs.docker.com/develop/security-best-practices/)
- [Kubernetes Security Best Practices](https://kubernetes.io/docs/concepts/security/overview/)

---

## 🤝 Contributing

When adding new infrastructure:
1. Follow the security patterns established in this guide
2. Run local validation before committing
3. Ensure CI/CD checks pass
4. Update this documentation if adding new security measures

---

**Last Updated**: 2025-11-23 
**Maintainer**: Infrastructure Team 
**CI/CD Workflow Version**: v1.0.0
