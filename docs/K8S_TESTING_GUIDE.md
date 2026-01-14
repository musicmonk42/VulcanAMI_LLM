# Kubernetes Deployment Testing Guide

This guide provides comprehensive testing instructions for the VulcanAMI Kubernetes deployment package.

## Prerequisites

Before testing, ensure you have:

- ✅ Kubernetes cluster 1.24+ (local, EKS, AKS, GKE)
- ✅ kubectl 1.24+ installed and configured
- ✅ kustomize 4.5+ installed
- ✅ At least 235Gi available storage
- ✅ Sufficient compute resources (4+ CPUs, 16Gi+ RAM recommended for testing)

## Phase 1: Static Validation (No Cluster Required)

### Test 1.1: Kustomize Build Validation

```bash
# Test base manifests
kustomize build k8s/base > /tmp/base.yaml
echo "✅ Base build successful"

# Test development overlay
kustomize build k8s/overlays/development > /tmp/dev.yaml
echo "✅ Development overlay build successful"

# Test staging overlay
kustomize build k8s/overlays/staging > /tmp/staging.yaml
echo "✅ Staging overlay build successful"

# Test production overlay
kustomize build k8s/overlays/production > /tmp/prod.yaml
echo "✅ Production overlay build successful"
```

**Expected Result:** All builds complete without errors.

### Test 1.2: Manifest Content Validation

```bash
# Verify all required resources are present
for resource in Namespace ConfigMap Secret Service PersistentVolumeClaim Deployment StatefulSet NetworkPolicy Job; do
 count=$(grep "^kind: $resource" /tmp/dev.yaml | wc -l)
 echo "$resource: $count instances"
done

# Check for critical services
echo ""
echo "Checking for required services..."
grep "name: dev-milvus-service" /tmp/dev.yaml && echo "✅ Milvus service found"
grep "name: dev-minio-service" /tmp/dev.yaml && echo "✅ MinIO service found"
grep "name: dev-postgres-service" /tmp/dev.yaml && echo "✅ PostgreSQL service found"
grep "name: dev-redis-service" /tmp/dev.yaml && echo "✅ Redis service found"
grep "name: dev-vulcanami-api" /tmp/dev.yaml && echo "✅ API service found"
```

**Expected Result:** All services are present in the manifests.

### Test 1.3: Security Configuration Validation

```bash
# Check security contexts are defined
echo "Checking security contexts..."
grep -c "securityContext:" /tmp/dev.yaml
grep -c "runAsNonRoot: true" /tmp/dev.yaml
grep -c "readOnlyRootFilesystem: true" /tmp/dev.yaml

# Check network policies exist
echo ""
echo "Checking network policies..."
grep "kind: NetworkPolicy" /tmp/dev.yaml | wc -l

# Verify no privileged containers
echo ""
if grep "privileged: true" /tmp/dev.yaml; then
 echo "❌ FAILED: Found privileged containers"
 exit 1
else
 echo "✅ No privileged containers found"
fi
```

**Expected Result:** Security configurations are properly set, no privileged containers.

### Test 1.4: Shell Script Validation

```bash
# Check script syntax
bash -n scripts/deploy.sh && echo "✅ deploy.sh syntax valid"
bash -n scripts/validate-deployment.sh && echo "✅ validate-deployment.sh syntax valid"

# Check scripts are executable
[ -x scripts/deploy.sh ] && echo "✅ deploy.sh is executable" || echo "❌ deploy.sh not executable"
[ -x scripts/validate-deployment.sh ] && echo "✅ validate-deployment.sh is executable" || echo "❌ validate-deployment.sh not executable"

# Test help output
./scripts/deploy.sh --help | head -20
```

**Expected Result:** Scripts have valid syntax and are executable.

## Phase 2: Pre-Deployment Validation (Requires Cluster)

### Test 2.1: Cluster Accessibility

```bash
# Verify kubectl is configured
kubectl cluster-info

# Check available nodes
kubectl get nodes

# Verify sufficient resources
kubectl top nodes
```

**Expected Result:** Cluster is accessible and has sufficient resources.

### Test 2.2: Run Validation Script

```bash
# Run full validation for development
./scripts/validate-deployment.sh development

# Expected checks:
# ✅ kubectl installation
# ✅ Cluster connectivity
# ✅ Storage classes
# ✅ Namespace (or permission to create)
# ✅ Kustomize manifests
# ⚠️ Warnings for missing optional components (OK)
```

**Expected Result:** All critical checks pass (warnings are acceptable).

### Test 2.3: Storage Class Verification

```bash
# List available storage classes
kubectl get storageclasses

# Check for 'standard' storage class (used in manifests)
kubectl get storageclass standard || echo "⚠️ 'standard' storage class not found"

# Check default storage class
kubectl get storageclass -o jsonpath='{.items[?(@.metadata.annotations.storageclass\.kubernetes\.io/is-default-class=="true")].metadata.name}'
```

**Expected Result:** At least one storage class exists.

## Phase 3: Deployment Testing

### Test 3.1: Dry-Run Deployment

```bash
# Test deployment in dry-run mode (no actual changes)
./scripts/deploy.sh development --image-tag test-v1.0.0 --dry-run

# Or manually:
kubectl apply -k k8s/overlays/development --dry-run=server
```

**Expected Result:** Dry-run completes without errors.

### Test 3.2: Create Test Namespace and Secrets

```bash
# Create development namespace
kubectl create namespace vulcanami-development

# Create test secrets
kubectl create secret generic vulcanami-secrets \
 --from-literal=JWT_SECRET_KEY=$(openssl rand -base64 48) \
 --from-literal=BOOTSTRAP_KEY=$(openssl rand -base64 32) \
 --from-literal=POSTGRES_PASSWORD=$(openssl rand -base64 32) \
 --from-literal=REDIS_PASSWORD=$(openssl rand -base64 32) \
 --from-literal=MINIO_ROOT_USER=minioadmin \
 --from-literal=MINIO_ROOT_PASSWORD=$(openssl rand -base64 32) \
 --from-literal=MINIO_SECRET_KEY=$(openssl rand -base64 24) \
 --from-literal=OPENAI_API_KEY=${OPENAI_API_KEY:-} \
 --from-literal=ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-} \
 --from-literal=GRAPHIX_API_KEY=${GRAPHIX_API_KEY:-} \
 --from-literal=VULCAN_LLM_API_KEY=${OPENAI_API_KEY:-} \
 --from-literal=AWS_ACCESS_KEY_ID=minioadmin \
 --from-literal=AWS_SECRET_ACCESS_KEY=$(openssl rand -base64 32) \
 -n vulcanami-development

# Verify secrets created
kubectl get secret vulcanami-secrets -n vulcanami-development
```

**Expected Result:** Secrets created successfully.

### Test 3.3: Full Deployment

```bash
# Deploy using automation script
./scripts/deploy.sh development --image-tag v1.0.0 --wait-timeout 900

# Monitor deployment progress in another terminal
kubectl get pods -n vulcanami-development -w
```

**Expected Result:** Deployment completes successfully, all pods reach Running state.

## Phase 4: Post-Deployment Validation

### Test 4.1: Pod Status Check

```bash
# List all pods
kubectl get pods -n vulcanami-development

# Expected pods:
# - vulcanami-api-*
# - postgres-0
# - redis-*
# - milvus-0
# - minio-0
# - minio-bucket-setup-* (completed)

# Check for any unhealthy pods
kubectl get pods -n vulcanami-development --field-selector=status.phase!=Running,status.phase!=Succeeded

# Expected: No unhealthy pods (bucket-setup job may be Completed)
```

**Expected Result:** All pods are Running or Completed (bucket-setup job).

### Test 4.2: Service Connectivity Tests

```bash
# Test PostgreSQL connectivity
kubectl exec -n vulcanami-development deploy/vulcanami-api -- \
 sh -c 'timeout 10 nc -zv postgres-service 5432'

# Test Redis connectivity
kubectl exec -n vulcanami-development deploy/vulcanami-api -- \
 sh -c 'timeout 10 nc -zv redis-service 6379'

# Test Milvus connectivity
kubectl exec -n vulcanami-development deploy/vulcanami-api -- \
 sh -c 'timeout 10 nc -zv milvus-service 19530'

# Test MinIO connectivity
kubectl exec -n vulcanami-development deploy/vulcanami-api -- \
 sh -c 'timeout 10 curl -f -m 5 http://minio-service:9000/minio/health/live'
```

**Expected Result:** All connectivity tests pass.

### Test 4.3: API Health Checks

```bash
# Check API health endpoint
kubectl exec -n vulcanami-development deploy/vulcanami-api -- \
 curl -s http://localhost:8000/health/live

# Check readiness
kubectl exec -n vulcanami-development deploy/vulcanami-api -- \
 curl -s http://localhost:8000/health/ready

# Port forward and test from local machine
kubectl port-forward -n vulcanami-development svc/vulcanami-api 8000:8000 &
sleep 3
curl http://localhost:8000/health/live
curl http://localhost:8000/health/ready
```

**Expected Result:** Health endpoints return 200 OK.

### Test 4.4: Milvus Bootstrap Verification

```bash
# Check init container logs (should show successful bootstrap)
POD=$(kubectl get pod -n vulcanami-development -l app=vulcanami-api -o jsonpath='{.items[0].metadata.name}')
kubectl logs -n vulcanami-development $POD -c milvus-bootstrap

# Should show:
# - Connection to Milvus successful
# - Collections created or already exist
```

**Expected Result:** Milvus bootstrap completed successfully.

### Test 4.5: MinIO Bucket Verification

```bash
# Check bucket setup job logs
kubectl logs -n vulcanami-development job/minio-bucket-setup

# Should show:
# - Connection to MinIO successful
# - Bucket 'vulcanami-memory' created or already exists

# Verify bucket exists via API
kubectl exec -n vulcanami-development sts/minio -- \
 mc alias set local http://localhost:9000 minioadmin $(kubectl get secret vulcanami-secrets -n vulcanami-development -o jsonpath='{.data.MINIO_ROOT_PASSWORD}' | base64 -d)
kubectl exec -n vulcanami-development sts/minio -- \
 mc ls local/vulcanami-memory
```

**Expected Result:** Bucket exists and is accessible.

### Test 4.6: Network Policy Verification

```bash
# List network policies
kubectl get networkpolicies -n vulcanami-development

# Verify policies exist for:
# - vulcanami-api
# - postgres
# - redis
# - milvus
# - minio

# Test that external access to backend services is blocked
# (This test should fail, which is expected)
kubectl run test-external --image=curlimages/curl --rm -i --restart=Never -n default -- \
 curl -m 5 http://postgres-service.vulcanami-development:5432 || echo "✅ External access blocked (as expected)"
```

**Expected Result:** Network policies exist and block unauthorized access.

## Phase 5: Functional Testing

### Test 5.1: API Functionality

```bash
# Port forward to API
kubectl port-forward -n vulcanami-development svc/vulcanami-api 8000:8000 &

# Test basic endpoints
curl http://localhost:8000/health
curl http://localhost:8000/metrics | head -20

# Stop port forward
pkill -f "port-forward.*vulcanami-api"
```

**Expected Result:** API responds to basic requests.

### Test 5.2: Resource Usage Check

```bash
# Check resource usage
kubectl top pods -n vulcanami-development

# Compare against resource limits
kubectl describe pod -n vulcanami-development -l app=vulcanami-api | grep -A5 "Limits:"
```

**Expected Result:** Resource usage is within limits.

### Test 5.3: Logs Inspection

```bash
# Check API logs for errors
kubectl logs -n vulcanami-development -l app=vulcanami-api --tail=100 | grep -i error || echo "No errors found"

# Check Milvus logs
kubectl logs -n vulcanami-development sts/milvus --tail=50

# Check MinIO logs
kubectl logs -n vulcanami-development sts/minio --tail=50

# Check PostgreSQL logs
kubectl logs -n vulcanami-development sts/postgres --tail=50

# Check Redis logs
kubectl logs -n vulcanami-development deploy/redis --tail=50
```

**Expected Result:** No critical errors in logs.

## Phase 6: Stress Testing

### Test 6.1: Pod Restart Resilience

```bash
# Delete API pod and verify it restarts
kubectl delete pod -n vulcanami-development -l app=vulcanami-api

# Wait for new pod to be ready
kubectl wait --for=condition=ready pod -n vulcanami-development -l app=vulcanami-api --timeout=300s

# Verify service remains accessible
kubectl exec -n vulcanami-development deploy/vulcanami-api -- \
 curl -s http://localhost:8000/health/live
```

**Expected Result:** Pod restarts successfully, service remains available.

### Test 6.2: Scale Test

```bash
# Scale API deployment
kubectl scale deployment vulcanami-api -n vulcanami-development --replicas=3

# Wait for scale
kubectl wait --for=condition=available deployment/vulcanami-api -n vulcanami-development --timeout=300s

# Verify all replicas are running
kubectl get pods -n vulcanami-development -l app=vulcanami-api

# Scale back down
kubectl scale deployment vulcanami-api -n vulcanami-development --replicas=1
```

**Expected Result:** Deployment scales up and down successfully.

## Phase 7: Cleanup

### Test 7.1: Controlled Deletion

```bash
# Delete deployment using kubectl
kubectl delete -k k8s/overlays/development

# Verify resources are deleted
kubectl get all -n vulcanami-development

# Delete namespace
kubectl delete namespace vulcanami-development

# Verify complete cleanup
kubectl get namespace vulcanami-development || echo "✅ Namespace deleted"
```

**Expected Result:** All resources cleanly deleted.

## Test Results Summary

After completing all tests, document results:

```
✅ PASSED: Phase 1 - Static Validation
✅ PASSED: Phase 2 - Pre-Deployment Validation
✅ PASSED: Phase 3 - Deployment Testing
✅ PASSED: Phase 4 - Post-Deployment Validation
✅ PASSED: Phase 5 - Functional Testing
✅ PASSED: Phase 6 - Stress Testing
✅ PASSED: Phase 7 - Cleanup
```

## Troubleshooting Common Issues

### Issue: Pods stuck in Pending

**Solution:**
```bash
kubectl describe pod <pod-name> -n vulcanami-development
# Check events for resource or storage issues
# Verify storage class exists and has available capacity
```

### Issue: Image pull errors

**Solution:**
```bash
# Update image tag in overlay
cd k8s/overlays/development
kustomize edit set image ghcr.io/musicmonk42/vulcanami_llm-api:v1.0.0
kubectl apply -k .
```

### Issue: Init container fails

**Solution:**
```bash
# Check init container logs
kubectl logs <pod-name> -n vulcanami-development -c <init-container-name>
# Verify service dependencies are running
kubectl get pods -n vulcanami-development
```

## CI/CD Testing

The GitHub Actions workflows automatically run validation on every push:

- **infrastructure-validation.yml**: Validates Kubernetes manifests
- **deploy.yml**: Deploys to cluster (if KUBE_CONFIG secret is set)

To test CI/CD:
1. Push changes to a feature branch
2. Check workflow runs in GitHub Actions tab
3. Review validation results
4. Merge if all checks pass

## Next Steps

After successful testing:

1. Document any environment-specific configurations
2. Update secrets with production values
3. Configure monitoring and alerting
4. Set up backup procedures
5. Implement disaster recovery plan
6. Schedule regular health checks
7. Plan scaling strategy
8. Configure auto-scaling if needed

## Support

For issues or questions:
- Review [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for detailed deployment guide
- Check [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for common issues
- Create GitHub issue with test results and error logs
