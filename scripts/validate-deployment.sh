#!/bin/bash
#
# Pre-Deployment Validation Script for VulcanAMI
# 
# This script performs comprehensive pre-deployment validation checks to ensure
# the Kubernetes cluster and configuration are ready for VulcanAMI deployment.
#
# Usage: ./scripts/validate-deployment.sh [ENVIRONMENT]
#
# Arguments:
#   ENVIRONMENT    Target environment: development, staging, or production (default: development)
#
# Exit Codes:
#   0 - All validation checks passed
#   1 - One or more validation checks failed
#   2 - Critical error (cannot continue validation)
#

set -euo pipefail

# Script metadata
readonly SCRIPT_VERSION="1.0.0"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly K8S_BASE_DIR="${PROJECT_ROOT}/k8s"

# Color codes for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m'
readonly BOLD='\033[1m'

# Validation state
CHECKS_PASSED=0
CHECKS_FAILED=0
CHECKS_WARNED=0

# Configuration
ENVIRONMENT="${1:-development}"
NAMESPACE="vulcanami-${ENVIRONMENT}"

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_check() {
    echo -e "${CYAN}[CHECK]${NC} $*"
}

log_pass() {
    echo -e "${GREEN}[✓ PASS]${NC} $*"
    ((CHECKS_PASSED++))
}

log_fail() {
    echo -e "${RED}[✗ FAIL]${NC} $*"
    ((CHECKS_FAILED++))
}

log_warn() {
    echo -e "${YELLOW}[⚠ WARN]${NC} $*"
    ((CHECKS_WARNED++))
}

log_remediation() {
    echo -e "${YELLOW}  → Remediation:${NC} $*"
}

log_section() {
    echo
    echo -e "${BOLD}${CYAN}═══ $* ═══${NC}"
    echo
}

# Validate environment argument
validate_environment() {
    if [[ ! "$ENVIRONMENT" =~ ^(development|staging|production)$ ]]; then
        echo -e "${RED}ERROR: Invalid environment: $ENVIRONMENT${NC}" >&2
        echo "Must be one of: development, staging, production" >&2
        exit 2
    fi
}

# Check 1: kubectl installed and accessible
check_kubectl() {
    log_check "Checking kubectl installation"
    
    if ! command -v kubectl &> /dev/null; then
        log_fail "kubectl is not installed"
        log_remediation "Install kubectl: https://kubernetes.io/docs/tasks/tools/"
        return 1
    fi
    
    local version
    version=$(kubectl version --client -o json 2>/dev/null | grep -o '"gitVersion":"[^"]*"' | cut -d'"' -f4 || echo "unknown")
    log_pass "kubectl is installed (version: $version)"
    return 0
}

# Check 2: Kubernetes cluster accessibility
check_cluster_access() {
    log_check "Checking Kubernetes cluster connectivity"
    
    if ! kubectl cluster-info &> /dev/null; then
        log_fail "Cannot connect to Kubernetes cluster"
        log_remediation "Check your kubeconfig: kubectl config view"
        log_remediation "Verify cluster is running and accessible"
        return 1
    fi
    
    local cluster
    cluster=$(kubectl config current-context 2>/dev/null || echo "unknown")
    log_pass "Connected to cluster: $cluster"
    return 0
}

# Check 3: Required storage classes exist
check_storage_classes() {
    log_check "Checking storage classes"
    
    if ! kubectl get storageclass &> /dev/null; then
        log_fail "Cannot query storage classes"
        return 1
    fi
    
    local storage_classes
    storage_classes=$(kubectl get storageclass -o jsonpath='{.items[*].metadata.name}')
    
    if [[ -z "$storage_classes" ]]; then
        log_fail "No storage classes found"
        log_remediation "Create a storage class or configure a default storage provisioner"
        return 1
    fi
    
    # Check for standard storage class (used in manifests)
    if echo "$storage_classes" | grep -qw "standard"; then
        log_pass "Storage class 'standard' exists"
    else
        log_warn "Storage class 'standard' not found (available: $storage_classes)"
        log_remediation "Update PVC manifests to use an available storage class"
    fi
    
    # Check for default storage class
    local default_sc
    default_sc=$(kubectl get storageclass -o jsonpath='{.items[?(@.metadata.annotations.storageclass\.kubernetes\.io/is-default-class=="true")].metadata.name}')
    if [[ -n "$default_sc" ]]; then
        log_pass "Default storage class: $default_sc"
    else
        log_warn "No default storage class configured"
    fi
    
    return 0
}

# Check 4: NGINX Ingress Controller
check_ingress_controller() {
    log_check "Checking for NGINX Ingress Controller"
    
    # Check for nginx ingress controller pods
    if kubectl get pods -A -l app.kubernetes.io/name=ingress-nginx &> /dev/null; then
        local ingress_pods
        ingress_pods=$(kubectl get pods -A -l app.kubernetes.io/name=ingress-nginx -o jsonpath='{.items[*].metadata.name}' | wc -w)
        if [[ $ingress_pods -gt 0 ]]; then
            log_pass "NGINX Ingress Controller is installed ($ingress_pods pods)"
        else
            log_warn "NGINX Ingress Controller found but no pods running"
        fi
    else
        log_warn "NGINX Ingress Controller not detected"
        log_remediation "Install NGINX Ingress: kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/cloud/deploy.yaml"
        log_remediation "Or use your cloud provider's ingress controller"
    fi
    
    return 0
}

# Check 5: cert-manager for TLS certificates
check_cert_manager() {
    log_check "Checking for cert-manager"
    
    if kubectl get namespace cert-manager &> /dev/null; then
        log_pass "cert-manager namespace exists"
        
        # Check cert-manager pods
        local cm_pods
        cm_pods=$(kubectl get pods -n cert-manager -l app.kubernetes.io/instance=cert-manager -o jsonpath='{.items[*].metadata.name}' 2>/dev/null | wc -w)
        if [[ $cm_pods -gt 0 ]]; then
            log_pass "cert-manager is running ($cm_pods pods)"
        else
            log_warn "cert-manager namespace exists but no pods found"
        fi
    else
        log_warn "cert-manager is not installed"
        log_remediation "For TLS support, install cert-manager: kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml"
    fi
    
    return 0
}

# Check 6: Namespace exists or can be created
check_namespace() {
    log_check "Checking namespace: $NAMESPACE"
    
    if kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_pass "Namespace '$NAMESPACE' exists"
    else
        log_warn "Namespace '$NAMESPACE' does not exist (will be created during deployment)"
        
        # Check if we have permissions to create namespace
        if kubectl auth can-i create namespaces &> /dev/null; then
            log_pass "Have permission to create namespaces"
        else
            log_fail "Cannot create namespaces (insufficient permissions)"
            log_remediation "Ask cluster admin to create namespace: kubectl create namespace $NAMESPACE"
            return 1
        fi
    fi
    
    return 0
}

# Check 7: Secrets are configured
check_secrets() {
    log_check "Checking secrets configuration"
    
    # Check if namespace exists first
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_warn "Cannot check secrets (namespace doesn't exist yet)"
        return 0
    fi
    
    if kubectl get secret vulcanami-secrets -n "$NAMESPACE" &> /dev/null; then
        log_pass "Secret 'vulcanami-secrets' exists"
        
        # Validate critical secret keys
        local missing_keys=()
        for key in JWT_SECRET_KEY POSTGRES_PASSWORD REDIS_PASSWORD MINIO_ROOT_PASSWORD OPENAI_API_KEY; do
            if ! kubectl get secret vulcanami-secrets -n "$NAMESPACE" -o jsonpath="{.data.${key}}" &> /dev/null; then
                missing_keys+=("$key")
            else
                local value
                value=$(kubectl get secret vulcanami-secrets -n "$NAMESPACE" -o jsonpath="{.data.${key}}" | base64 -d)
                if [[ "$value" == *"REPLACE_WITH_ACTUAL_SECRET"* ]] || [[ -z "$value" ]]; then
                    log_warn "Secret key '$key' contains placeholder value"
                fi
            fi
        done
        
        if [[ ${#missing_keys[@]} -gt 0 ]]; then
            log_fail "Missing secret keys: ${missing_keys[*]}"
            return 1
        fi
    else
        log_warn "Secret 'vulcanami-secrets' does not exist"
        log_remediation "Create secrets using: ${SCRIPT_DIR}/generate-secrets.sh"
        log_remediation "Or manually: kubectl create secret generic vulcanami-secrets ..."
    fi
    
    return 0
}

# Check 8: Image tags are set (not placeholders)
check_image_tags() {
    log_check "Checking image tags in overlay"
    
    local overlay_dir="${K8S_BASE_DIR}/overlays/${ENVIRONMENT}"
    
    if [[ ! -f "${overlay_dir}/kustomization.yaml" ]]; then
        log_fail "Overlay kustomization.yaml not found: ${overlay_dir}"
        return 1
    fi
    
    # Check if IMAGE_TAG placeholder is in use
    if grep -q "newTag: IMAGE_TAG" "${overlay_dir}/kustomization.yaml"; then
        log_warn "Image tag is set to placeholder 'IMAGE_TAG'"
        log_remediation "Set image tag with: kustomize edit set image ghcr.io/musicmonk42/vulcanami_llm-api:v1.2.3"
        log_remediation "Or use --image-tag flag with deploy.sh script"
    else
        local tag
        tag=$(grep "newTag:" "${overlay_dir}/kustomization.yaml" | awk '{print $2}' | tr -d '"')
        log_pass "Image tag is set: $tag"
    fi
    
    return 0
}

# Check 9: PVC storage capacity available
check_storage_capacity() {
    log_check "Checking storage capacity requirements"
    
    # Calculate total storage requirements
    local total_storage_gb=0
    local requirements=(
        "postgres:20"
        "redis:5"
        "milvus:50"
        "minio:100"
        "memory-store:50"
        "cache:10"
    )
    
    for req in "${requirements[@]}"; do
        local size
        size=$(echo "$req" | cut -d: -f2)
        ((total_storage_gb += size))
    done
    
    log_info "Total storage required: ${total_storage_gb}Gi"
    log_pass "Storage requirements calculated"
    
    # Note: Actual capacity check would require cloud provider specific APIs
    log_warn "Cannot verify available storage capacity (cluster-specific)"
    log_remediation "Ensure your cluster has at least ${total_storage_gb}Gi of available storage"
    
    return 0
}

# Check 10: Kustomize manifests are valid
check_kustomize_build() {
    log_check "Validating Kustomize manifests"
    
    if ! command -v kustomize &> /dev/null; then
        log_fail "kustomize is not installed"
        log_remediation "Install kustomize: https://kubectl.docs.kubernetes.io/installation/kustomize/"
        return 1
    fi
    
    local overlay_dir="${K8S_BASE_DIR}/overlays/${ENVIRONMENT}"
    
    if [[ ! -d "$overlay_dir" ]]; then
        log_fail "Overlay directory not found: $overlay_dir"
        return 1
    fi
    
    # Try to build manifests
    if kustomize build "$overlay_dir" > /dev/null 2>&1; then
        log_pass "Kustomize build successful"
    else
        log_fail "Kustomize build failed"
        log_remediation "Run for details: kustomize build $overlay_dir"
        return 1
    fi
    
    return 0
}

# Check 11: No resource conflicts
check_resource_conflicts() {
    log_check "Checking for existing resource conflicts"
    
    # Check if namespace exists first
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_pass "No conflicts (namespace doesn't exist)"
        return 0
    fi
    
    # Check for existing deployments that might conflict
    local deployments
    deployments=$(kubectl get deployments -n "$NAMESPACE" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || echo "")
    
    if [[ -n "$deployments" ]]; then
        log_warn "Found existing deployments in namespace: $deployments"
        log_remediation "Review existing resources before deployment"
        log_remediation "Consider backing up or deleting existing resources if needed"
    else
        log_pass "No conflicting deployments found"
    fi
    
    return 0
}

# Check 12: RBAC permissions
check_rbac_permissions() {
    log_check "Checking RBAC permissions"
    
    local required_permissions=(
        "get:pods"
        "list:pods"
        "create:deployments"
        "create:services"
        "create:secrets"
        "create:configmaps"
    )
    
    local missing_permissions=()
    
    for perm in "${required_permissions[@]}"; do
        local verb
        verb=$(echo "$perm" | cut -d: -f1)
        local resource
        resource=$(echo "$perm" | cut -d: -f2)
        
        if ! kubectl auth can-i "$verb" "$resource" -n "$NAMESPACE" &> /dev/null; then
            missing_permissions+=("$verb $resource")
        fi
    done
    
    if [[ ${#missing_permissions[@]} -gt 0 ]]; then
        log_fail "Missing RBAC permissions: ${missing_permissions[*]}"
        log_remediation "Contact cluster administrator to grant required permissions"
        return 1
    else
        log_pass "Have required RBAC permissions"
    fi
    
    return 0
}

# Check 13: Network policies support
check_network_policies() {
    log_check "Checking network policy support"
    
    # Try to get network policies to see if CNI supports them
    if kubectl get networkpolicies -A &> /dev/null; then
        log_pass "Network policies are supported"
    else
        log_warn "Network policies may not be supported by CNI"
        log_remediation "Network policies in manifests may not be enforced"
        log_remediation "Consider using a CNI with network policy support (Calico, Cilium, etc.)"
    fi
    
    return 0
}

# Display summary
display_summary() {
    echo
    log_section "Validation Summary"
    
    local total_checks=$((CHECKS_PASSED + CHECKS_FAILED + CHECKS_WARNED))
    
    echo -e "${BOLD}Total Checks:${NC} $total_checks"
    echo -e "${GREEN}${BOLD}Passed:${NC} $CHECKS_PASSED"
    echo -e "${YELLOW}${BOLD}Warnings:${NC} $CHECKS_WARNED"
    echo -e "${RED}${BOLD}Failed:${NC} $CHECKS_FAILED"
    echo
    
    if [[ $CHECKS_FAILED -eq 0 ]]; then
        echo -e "${GREEN}${BOLD}✓ All critical checks passed!${NC}"
        if [[ $CHECKS_WARNED -gt 0 ]]; then
            echo -e "${YELLOW}  Note: ${CHECKS_WARNED} warning(s) were issued${NC}"
        fi
        echo
        echo "Ready to deploy with:"
        echo "  ${SCRIPT_DIR}/deploy.sh ${ENVIRONMENT}"
        echo
        return 0
    else
        echo -e "${RED}${BOLD}✗ Validation failed with ${CHECKS_FAILED} error(s)${NC}"
        echo
        echo "Please address the failures above before deployment"
        echo
        return 1
    fi
}

# Main execution
main() {
    echo -e "${BOLD}VulcanAMI Pre-Deployment Validation v${SCRIPT_VERSION}${NC}"
    echo -e "Environment: ${BOLD}${ENVIRONMENT}${NC}"
    echo -e "Namespace: ${BOLD}${NAMESPACE}${NC}"
    
    # Validate environment argument
    validate_environment
    
    log_section "Running Validation Checks"
    
    # Run all validation checks
    check_kubectl
    check_cluster_access
    check_storage_classes
    check_ingress_controller
    check_cert_manager
    check_namespace
    check_secrets
    check_image_tags
    check_storage_capacity
    check_kustomize_build
    check_resource_conflicts
    check_rbac_permissions
    check_network_policies
    
    # Display summary and return appropriate exit code
    if display_summary; then
        exit 0
    else
        exit 1
    fi
}

# Run main function
main "$@"
