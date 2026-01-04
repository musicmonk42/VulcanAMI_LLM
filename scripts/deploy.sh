#!/bin/bash
#
# Kubernetes Deployment Automation Script for VulcanAMI
# 
# This script automates the deployment of VulcanAMI to Kubernetes clusters with
# comprehensive validation, error handling, and rollback capabilities.
#
# Usage: ./scripts/deploy.sh [ENVIRONMENT] [OPTIONS]
#
# Arguments:
#   ENVIRONMENT    Target environment: development, staging, or production (default: development)
#
# Options:
#   --image-tag TAG       Override image tag (default: IMAGE_TAG from overlay)
#   --namespace NAME      Override namespace (default: vulcanami-ENVIRONMENT)
#   --skip-validation     Skip pre-deployment validation checks
#   --dry-run            Run kubectl apply in dry-run mode
#   --wait-timeout SECS  Timeout for rollout wait in seconds (default: 600)
#   --help               Display this help message
#
# Exit Codes:
#   0 - Deployment successful
#   1 - Validation failed
#   2 - Deployment failed
#   3 - Invalid arguments
#   4 - Prerequisites not met
#
# Examples:
#   ./scripts/deploy.sh development
#   ./scripts/deploy.sh production --image-tag v1.2.3
#   ./scripts/deploy.sh staging --dry-run
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
readonly NC='\033[0m' # No Color
readonly BOLD='\033[1m'

# Default configuration
ENVIRONMENT="${1:-development}"
IMAGE_TAG=""
NAMESPACE=""
SKIP_VALIDATION=false
DRY_RUN=false
WAIT_TIMEOUT=600

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

log_step() {
    echo -e "${CYAN}${BOLD}==>${NC} ${BOLD}$*${NC}"
}

# Display usage information
usage() {
    head -n 30 "$0" | grep "^#" | sed 's/^# \?//'
    exit 0
}

# Parse command line arguments
parse_args() {
    # First argument is environment if not a flag
    if [[ "${1:-}" != --* ]]; then
        ENVIRONMENT="${1:-development}"
        shift || true
    fi

    # Parse remaining flags
    while [[ $# -gt 0 ]]; do
        case $1 in
            --image-tag)
                IMAGE_TAG="$2"
                shift 2
                ;;
            --namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            --skip-validation)
                SKIP_VALIDATION=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --wait-timeout)
                WAIT_TIMEOUT="$2"
                shift 2
                ;;
            --help|-h)
                usage
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 3
                ;;
        esac
    done

    # Validate environment
    if [[ ! "$ENVIRONMENT" =~ ^(development|staging|production)$ ]]; then
        log_error "Invalid environment: $ENVIRONMENT"
        log_error "Must be one of: development, staging, production"
        exit 3
    fi

    # Set namespace if not provided
    if [[ -z "$NAMESPACE" ]]; then
        NAMESPACE="vulcanami-${ENVIRONMENT}"
    fi

    log_info "Configuration:"
    log_info "  Environment: ${ENVIRONMENT}"
    log_info "  Namespace: ${NAMESPACE}"
    log_info "  Image Tag: ${IMAGE_TAG:-<from overlay>}"
    log_info "  Dry Run: ${DRY_RUN}"
}

# Check if required tools are installed
validate_prerequisites() {
    log_step "Validating prerequisites"

    local missing_tools=()

    # Check for required commands
    for cmd in kubectl kustomize openssl; do
        if ! command -v "$cmd" &> /dev/null; then
            missing_tools+=("$cmd")
        fi
    done

    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_error "Please install missing tools and try again"
        return 4
    fi

    # Check kubectl cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        log_error "Please check your kubectl configuration"
        return 4
    fi

    # Check kubectl version compatibility (minimum 1.24)
    local kubectl_version
    kubectl_version=$(kubectl version --client -o json | grep -o '"gitVersion":"[^"]*"' | cut -d'"' -f4 | sed 's/v//')
    local kubectl_major
    kubectl_major=$(echo "$kubectl_version" | cut -d. -f1)
    local kubectl_minor
    kubectl_minor=$(echo "$kubectl_version" | cut -d. -f2)
    
    if [[ $kubectl_major -lt 1 ]] || [[ $kubectl_major -eq 1 && $kubectl_minor -lt 24 ]]; then
        log_warn "kubectl version ${kubectl_version} is older than recommended (1.24+)"
    fi

    log_success "All prerequisites met"
    return 0
}

# Validate overlay directory exists
validate_overlay() {
    log_step "Validating overlay configuration"

    local overlay_dir="${K8S_BASE_DIR}/overlays/${ENVIRONMENT}"
    
    if [[ ! -d "$overlay_dir" ]]; then
        log_error "Overlay directory not found: $overlay_dir"
        return 1
    fi

    if [[ ! -f "${overlay_dir}/kustomization.yaml" ]]; then
        log_error "kustomization.yaml not found in overlay directory"
        return 1
    fi

    log_success "Overlay configuration valid"
    return 0
}

# Generate or validate secrets
ensure_secrets() {
    log_step "Checking secrets configuration"

    # Check if secrets already exist in cluster
    if kubectl get secret vulcanami-secrets -n "$NAMESPACE" &> /dev/null; then
        log_info "Secret 'vulcanami-secrets' already exists in namespace $NAMESPACE"
        
        # Validate that secrets are not placeholders
        local has_placeholders=false
        for key in JWT_SECRET_KEY POSTGRES_PASSWORD REDIS_PASSWORD MINIO_ROOT_PASSWORD; do
            local value
            value=$(kubectl get secret vulcanami-secrets -n "$NAMESPACE" -o jsonpath="{.data.${key}}" 2>/dev/null | base64 -d)
            if [[ "$value" == *"REPLACE_WITH_ACTUAL_SECRET"* ]] || [[ -z "$value" ]]; then
                log_warn "Secret key '${key}' appears to be a placeholder"
                has_placeholders=true
            fi
        done

        if $has_placeholders; then
            log_warn "Some secrets contain placeholder values"
            log_warn "Please update secrets before deployment or delete and regenerate:"
            log_warn "  kubectl delete secret vulcanami-secrets -n $NAMESPACE"
            read -p "Continue anyway? (y/N) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                log_error "Deployment cancelled by user"
                exit 1
            fi
        fi
    else
        log_warn "Secret 'vulcanami-secrets' does not exist"
        log_info "You can create secrets using:"
        log_info "  ${SCRIPT_DIR}/generate-secrets.sh | kubectl apply -n $NAMESPACE -f -"
        log_info "Or manually with kubectl create secret"
        
        read -p "Create namespace and continue? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_error "Deployment cancelled by user"
            exit 1
        fi
    fi
}

# Set image tag in kustomization
set_image_tag() {
    if [[ -n "$IMAGE_TAG" ]]; then
        log_step "Setting image tag to: $IMAGE_TAG"
        
        # Validate image tag is not a placeholder
        if [[ "$IMAGE_TAG" == "IMAGE_TAG" ]]; then
            log_error "Image tag cannot be 'IMAGE_TAG' placeholder"
            log_error "Please specify a valid image tag with --image-tag"
            return 1
        fi

        local overlay_dir="${K8S_BASE_DIR}/overlays/${ENVIRONMENT}"
        cd "$overlay_dir"
        
        # Use kustomize edit to set image tag
        kustomize edit set image "ghcr.io/musicmonk42/vulcanami_llm-api:${IMAGE_TAG}"
        
        log_success "Image tag updated"
    else
        log_info "Using image tag from overlay configuration"
    fi
}

# Validate manifests with kustomize build and kubectl dry-run
validate_manifests() {
    log_step "Validating Kubernetes manifests"

    local overlay_dir="${K8S_BASE_DIR}/overlays/${ENVIRONMENT}"
    
    # Build manifests with kustomize
    log_info "Building manifests with kustomize..."
    if ! kustomize build "$overlay_dir" > /tmp/vulcanami-manifests.yaml; then
        log_error "Kustomize build failed"
        return 1
    fi
    log_success "Kustomize build successful"

    # Validate with kubectl dry-run
    log_info "Validating with kubectl dry-run..."
    if ! kubectl apply --dry-run=client -f /tmp/vulcanami-manifests.yaml &> /dev/null; then
        log_error "Kubectl validation failed"
        log_error "Run for details: kubectl apply --dry-run=client -f /tmp/vulcanami-manifests.yaml"
        return 1
    fi
    log_success "Manifest validation successful"

    # Clean up temp file
    rm -f /tmp/vulcanami-manifests.yaml

    return 0
}

# Deploy manifests to cluster
deploy() {
    log_step "Deploying VulcanAMI to ${ENVIRONMENT}"

    local overlay_dir="${K8S_BASE_DIR}/overlays/${ENVIRONMENT}"
    local apply_flags="-k $overlay_dir"

    if $DRY_RUN; then
        apply_flags="$apply_flags --dry-run=server"
        log_info "Running in DRY-RUN mode (no changes will be made)"
    fi

    # Create namespace if it doesn't exist
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_info "Creating namespace: $NAMESPACE"
        kubectl create namespace "$NAMESPACE"
    fi

    # Apply manifests
    log_info "Applying Kubernetes manifests..."
    if ! kubectl apply $apply_flags; then
        log_error "Deployment failed"
        return 2
    fi

    if $DRY_RUN; then
        log_success "Dry-run completed successfully"
        return 0
    fi

    log_success "Manifests applied successfully"

    # Wait for deployments to be ready
    log_step "Waiting for rollouts to complete (timeout: ${WAIT_TIMEOUT}s)"
    
    local deployments
    deployments=$(kubectl get deployments -n "$NAMESPACE" -o jsonpath='{.items[*].metadata.name}')
    
    for deployment in $deployments; do
        log_info "Waiting for deployment: $deployment"
        if ! kubectl rollout status deployment/"$deployment" -n "$NAMESPACE" --timeout="${WAIT_TIMEOUT}s"; then
            log_error "Deployment $deployment failed to roll out"
            return 2
        fi
    done

    # Wait for StatefulSets to be ready
    local statefulsets
    statefulsets=$(kubectl get statefulsets -n "$NAMESPACE" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || echo "")
    
    for sts in $statefulsets; do
        log_info "Waiting for StatefulSet: $sts"
        if ! kubectl rollout status statefulset/"$sts" -n "$NAMESPACE" --timeout="${WAIT_TIMEOUT}s"; then
            log_error "StatefulSet $sts failed to roll out"
            return 2
        fi
    done

    log_success "All rollouts completed successfully"
    return 0
}

# Perform post-deployment health checks
health_check() {
    log_step "Performing health checks"

    # Check pod status
    log_info "Checking pod status..."
    local unhealthy_pods
    unhealthy_pods=$(kubectl get pods -n "$NAMESPACE" --field-selector=status.phase!=Running,status.phase!=Succeeded -o jsonpath='{.items[*].metadata.name}')
    
    if [[ -n "$unhealthy_pods" ]]; then
        log_warn "Some pods are not in Running state:"
        kubectl get pods -n "$NAMESPACE" --field-selector=status.phase!=Running,status.phase!=Succeeded
    else
        log_success "All pods are running"
    fi

    # Check API endpoint if available
    log_info "Checking service endpoints..."
    local api_service
    api_service=$(kubectl get service -n "$NAMESPACE" -l app=vulcanami-api -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    
    if [[ -n "$api_service" ]]; then
        log_success "API service found: $api_service"
    else
        log_warn "API service not found"
    fi

    return 0
}

# Display deployment information and next steps
display_info() {
    log_step "Deployment Information"

    echo
    echo -e "${BOLD}Namespace:${NC} $NAMESPACE"
    echo -e "${BOLD}Environment:${NC} $ENVIRONMENT"
    echo
    
    log_info "Get pod status:"
    echo "  kubectl get pods -n $NAMESPACE"
    echo
    
    log_info "View logs:"
    echo "  kubectl logs -n $NAMESPACE -l app=vulcanami-api --tail=100 -f"
    echo
    
    log_info "Check service endpoints:"
    echo "  kubectl get services -n $NAMESPACE"
    echo
    
    log_info "Get ingress URL:"
    echo "  kubectl get ingress -n $NAMESPACE"
    echo
    
    log_info "Port forward to API (local testing):"
    echo "  kubectl port-forward -n $NAMESPACE svc/vulcanami-api 8000:8000"
    echo

    # Check if ingress exists
    local ingress
    ingress=$(kubectl get ingress -n "$NAMESPACE" -o jsonpath='{.items[0].spec.rules[0].host}' 2>/dev/null || echo "")
    if [[ -n "$ingress" ]]; then
        echo -e "${BOLD}Application URL:${NC} https://$ingress"
        echo
    fi
}

# Rollback deployment on failure
rollback() {
    log_error "Deployment failed, initiating rollback..."
    
    # Try to rollback each deployment
    local deployments
    deployments=$(kubectl get deployments -n "$NAMESPACE" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || echo "")
    
    for deployment in $deployments; do
        log_info "Rolling back deployment: $deployment"
        kubectl rollout undo deployment/"$deployment" -n "$NAMESPACE" || true
    done
}

# Main execution flow
main() {
    echo -e "${BOLD}VulcanAMI Kubernetes Deployment Script v${SCRIPT_VERSION}${NC}"
    echo

    # Parse arguments
    parse_args "$@"

    # Validate prerequisites
    if ! validate_prerequisites; then
        exit 4
    fi

    # Validate overlay
    if ! validate_overlay; then
        exit 1
    fi

    # Set image tag if provided
    if ! set_image_tag; then
        exit 1
    fi

    # Run validation unless skipped
    if ! $SKIP_VALIDATION; then
        if ! validate_manifests; then
            exit 1
        fi
    else
        log_warn "Skipping validation checks (--skip-validation flag set)"
    fi

    # Ensure secrets are configured
    if ! $DRY_RUN; then
        ensure_secrets
    fi

    # Deploy
    if ! deploy; then
        if ! $DRY_RUN; then
            rollback
        fi
        exit 2
    fi

    # Skip health checks and info display for dry-run
    if $DRY_RUN; then
        exit 0
    fi

    # Health checks
    health_check

    # Display info
    display_info

    log_success "Deployment completed successfully!"
    exit 0
}

# Trap errors and interrupts
trap 'log_error "Script interrupted"; exit 130' INT TERM

# Run main function
main "$@"
