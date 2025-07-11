#!/bin/bash

# Rental ML System Kubernetes Deployment Script
# This script automates the deployment of the Rental ML System to Kubernetes

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
K8S_DIR="$(dirname "$SCRIPT_DIR")"
ROOT_DIR="$(dirname "$K8S_DIR")"

# Default values
ENVIRONMENT="dev"
NAMESPACE=""
HELM_RELEASE_NAME="rental-ml"
IMAGE_TAG="latest"
DRY_RUN=false
SKIP_HEALTH_CHECK=false
WAIT_TIMEOUT=600
DEPLOY_METHOD="kustomize"  # Options: kustomize, helm
VERBOSE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

log_verbose() {
    if [[ "$VERBOSE" == "true" ]]; then
        echo -e "${BLUE}[VERBOSE]${NC} $1"
    fi
}

# Help function
show_help() {
    cat << EOF
Rental ML System Kubernetes Deployment Script

Usage: $0 [OPTIONS]

Options:
    -e, --environment ENV       Environment to deploy (dev, staging, prod) [default: dev]
    -n, --namespace NAMESPACE   Kubernetes namespace [default: auto-detect from environment]
    -r, --release-name NAME     Helm release name [default: rental-ml]
    -t, --image-tag TAG         Docker image tag to deploy [default: latest]
    -m, --method METHOD         Deployment method (kustomize, helm) [default: kustomize]
    -d, --dry-run              Perform a dry run without applying changes
    -s, --skip-health-check    Skip health checks after deployment
    -w, --wait-timeout SECONDS Timeout for waiting on deployments [default: 600]
    -v, --verbose              Enable verbose output
    -h, --help                 Show this help message

Examples:
    # Deploy to development environment
    $0 --environment dev
    
    # Deploy to production with specific image tag
    $0 --environment prod --image-tag v1.2.3
    
    # Deploy using Helm with dry run
    $0 --environment staging --method helm --dry-run
    
    # Deploy with custom namespace
    $0 --environment prod --namespace rental-ml-custom

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            -r|--release-name)
                HELM_RELEASE_NAME="$2"
                shift 2
                ;;
            -t|--image-tag)
                IMAGE_TAG="$2"
                shift 2
                ;;
            -m|--method)
                DEPLOY_METHOD="$2"
                shift 2
                ;;
            -d|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -s|--skip-health-check)
                SKIP_HEALTH_CHECK=true
                shift
                ;;
            -w|--wait-timeout)
                WAIT_TIMEOUT="$2"
                shift 2
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Validate prerequisites
validate_prerequisites() {
    log_info "Validating prerequisites..."
    
    # Check if kubectl is installed and configured
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check kubectl connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster. Please check your kubeconfig"
        exit 1
    fi
    
    # Check deployment method requirements
    case $DEPLOY_METHOD in
        "kustomize")
            if ! command -v kustomize &> /dev/null; then
                log_error "kustomize is not installed or not in PATH"
                exit 1
            fi
            ;;
        "helm")
            if ! command -v helm &> /dev/null; then
                log_error "helm is not installed or not in PATH"
                exit 1
            fi
            ;;
        *)
            log_error "Invalid deployment method: $DEPLOY_METHOD. Use 'kustomize' or 'helm'"
            exit 1
            ;;
    esac
    
    log_success "Prerequisites validated"
}

# Set namespace based on environment
set_namespace() {
    if [[ -z "$NAMESPACE" ]]; then
        case $ENVIRONMENT in
            "dev")
                NAMESPACE="rental-ml-dev"
                ;;
            "staging")
                NAMESPACE="rental-ml-staging"
                ;;
            "prod")
                NAMESPACE="rental-ml"
                ;;
            *)
                log_error "Invalid environment: $ENVIRONMENT. Use 'dev', 'staging', or 'prod'"
                exit 1
                ;;
        esac
    fi
    
    log_info "Using namespace: $NAMESPACE"
}

# Create namespace if it doesn't exist
create_namespace() {
    log_info "Ensuring namespace $NAMESPACE exists..."
    
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        if [[ "$DRY_RUN" == "true" ]]; then
            log_info "[DRY RUN] Would create namespace: $NAMESPACE"
        else
            kubectl create namespace "$NAMESPACE"
            log_success "Created namespace: $NAMESPACE"
        fi
    else
        log_info "Namespace $NAMESPACE already exists"
    fi
}

# Deploy using Kustomize
deploy_with_kustomize() {
    log_info "Deploying with Kustomize..."
    
    local overlay_dir="$K8S_DIR/overlays/$ENVIRONMENT"
    
    if [[ ! -d "$overlay_dir" ]]; then
        log_error "Overlay directory not found: $overlay_dir"
        exit 1
    fi
    
    # Build kustomization
    log_verbose "Building kustomization from $overlay_dir"
    local kustomize_output
    kustomize_output=$(kustomize build "$overlay_dir")
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Kustomize build output:"
        echo "$kustomize_output"
        return 0
    fi
    
    # Apply with server-side apply for better handling
    echo "$kustomize_output" | kubectl apply --server-side=true -f -
    
    log_success "Kustomize deployment completed"
}

# Deploy using Helm
deploy_with_helm() {
    log_info "Deploying with Helm..."
    
    local chart_dir="$K8S_DIR/helm/rental-ml"
    local values_file="$chart_dir/values-$ENVIRONMENT.yaml"
    
    if [[ ! -d "$chart_dir" ]]; then
        log_error "Helm chart directory not found: $chart_dir"
        exit 1
    fi
    
    # Check if values file exists
    if [[ ! -f "$values_file" ]]; then
        log_warning "Values file not found: $values_file. Using default values.yaml"
        values_file="$chart_dir/values.yaml"
    fi
    
    # Prepare Helm command
    local helm_cmd=("helm" "upgrade" "--install" "$HELM_RELEASE_NAME" "$chart_dir")
    helm_cmd+=("--namespace" "$NAMESPACE")
    helm_cmd+=("--create-namespace")
    helm_cmd+=("--values" "$values_file")
    helm_cmd+=("--set" "app.image.tag=$IMAGE_TAG")
    helm_cmd+=("--set" "environment=$ENVIRONMENT")
    helm_cmd+=("--wait" "--timeout" "${WAIT_TIMEOUT}s")
    
    if [[ "$DRY_RUN" == "true" ]]; then
        helm_cmd+=("--dry-run")
        log_info "[DRY RUN] Helm command: ${helm_cmd[*]}"
    fi
    
    if [[ "$VERBOSE" == "true" ]]; then
        helm_cmd+=("--debug")
    fi
    
    # Execute Helm command
    "${helm_cmd[@]}"
    
    if [[ "$DRY_RUN" != "true" ]]; then
        log_success "Helm deployment completed"
    fi
}

# Wait for deployments to be ready
wait_for_deployments() {
    if [[ "$SKIP_HEALTH_CHECK" == "true" ]] || [[ "$DRY_RUN" == "true" ]]; then
        return 0
    fi
    
    log_info "Waiting for deployments to be ready..."
    
    local deployments=("app" "worker" "scheduler")
    
    for deployment in "${deployments[@]}"; do
        log_verbose "Waiting for deployment $deployment in namespace $NAMESPACE"
        if ! kubectl rollout status deployment "$deployment" -n "$NAMESPACE" --timeout="${WAIT_TIMEOUT}s"; then
            log_error "Deployment $deployment failed to become ready within $WAIT_TIMEOUT seconds"
            return 1
        fi
    done
    
    # Wait for StatefulSets
    local statefulsets=("postgres" "redis")
    
    for statefulset in "${statefulsets[@]}"; do
        log_verbose "Waiting for StatefulSet $statefulset in namespace $NAMESPACE"
        if ! kubectl rollout status statefulset "$statefulset" -n "$NAMESPACE" --timeout="${WAIT_TIMEOUT}s"; then
            log_error "StatefulSet $statefulset failed to become ready within $WAIT_TIMEOUT seconds"
            return 1
        fi
    done
    
    log_success "All deployments are ready"
}

# Perform health checks
perform_health_checks() {
    if [[ "$SKIP_HEALTH_CHECK" == "true" ]] || [[ "$DRY_RUN" == "true" ]]; then
        return 0
    fi
    
    log_info "Performing health checks..."
    
    # Check if app service is responding
    local app_service="app-service"
    
    # Port forward to test the application
    log_verbose "Port forwarding to test application health endpoint"
    kubectl port-forward -n "$NAMESPACE" "service/$app_service" 8080:8000 &
    local pf_pid=$!
    
    # Wait a moment for port forward to establish
    sleep 3
    
    # Test health endpoint
    local health_check_passed=false
    for i in {1..10}; do
        if curl -f http://localhost:8080/health &> /dev/null; then
            health_check_passed=true
            break
        fi
        log_verbose "Health check attempt $i failed, retrying..."
        sleep 5
    done
    
    # Clean up port forward
    kill $pf_pid 2>/dev/null || true
    
    if [[ "$health_check_passed" == "true" ]]; then
        log_success "Health checks passed"
    else
        log_error "Health checks failed"
        return 1
    fi
}

# Show deployment status
show_deployment_status() {
    if [[ "$DRY_RUN" == "true" ]]; then
        return 0
    fi
    
    log_info "Deployment Status:"
    echo
    
    # Show pods
    kubectl get pods -n "$NAMESPACE" -o wide
    echo
    
    # Show services
    kubectl get services -n "$NAMESPACE"
    echo
    
    # Show ingress if exists
    if kubectl get ingress -n "$NAMESPACE" &> /dev/null; then
        kubectl get ingress -n "$NAMESPACE"
        echo
    fi
    
    # Show HPA if exists
    if kubectl get hpa -n "$NAMESPACE" &> /dev/null; then
        kubectl get hpa -n "$NAMESPACE"
        echo
    fi
}

# Cleanup function
cleanup() {
    # Kill any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
}

# Main function
main() {
    # Set up cleanup trap
    trap cleanup EXIT
    
    log_info "Starting Rental ML System deployment..."
    log_info "Environment: $ENVIRONMENT"
    log_info "Deployment method: $DEPLOY_METHOD"
    log_info "Image tag: $IMAGE_TAG"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_warning "DRY RUN MODE - No changes will be applied"
    fi
    
    # Execute deployment steps
    validate_prerequisites
    set_namespace
    create_namespace
    
    case $DEPLOY_METHOD in
        "kustomize")
            deploy_with_kustomize
            ;;
        "helm")
            deploy_with_helm
            ;;
    esac
    
    wait_for_deployments
    perform_health_checks
    show_deployment_status
    
    log_success "Deployment completed successfully!"
    
    if [[ "$ENVIRONMENT" == "dev" ]]; then
        log_info "For development access, you can use port-forward:"
        log_info "kubectl port-forward -n $NAMESPACE service/app-service 8000:8000"
        log_info "Then access the application at http://localhost:8000"
    fi
}

# Parse arguments and run main function
parse_args "$@"
main