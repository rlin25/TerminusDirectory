#!/bin/bash

# Rental ML System Health Check Script
# This script performs comprehensive health checks on the deployed system

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default values
ENVIRONMENT="dev"
NAMESPACE=""
VERBOSE=false
WAIT_TIMEOUT=300
CHECK_EXTERNAL=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Health check results
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0

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

# Check functions
check_passed() {
    ((TOTAL_CHECKS++))
    ((PASSED_CHECKS++))
    log_success "✓ $1"
}

check_failed() {
    ((TOTAL_CHECKS++))
    ((FAILED_CHECKS++))
    log_error "✗ $1"
}

check_warning() {
    ((TOTAL_CHECKS++))
    log_warning "⚠ $1"
}

# Help function
show_help() {
    cat << EOF
Rental ML System Health Check Script

Usage: $0 [OPTIONS]

Options:
    -e, --environment ENV       Environment to check (dev, staging, prod) [default: dev]
    -n, --namespace NAMESPACE   Kubernetes namespace [default: auto-detect from environment]
    -t, --timeout SECONDS       Timeout for checks [default: 300]
    -x, --check-external        Include external dependency checks
    -v, --verbose              Enable verbose output
    -h, --help                 Show this help message

Examples:
    # Basic health check for development
    $0 --environment dev
    
    # Comprehensive check for production including external dependencies
    $0 --environment prod --check-external --verbose
    
    # Check specific namespace
    $0 --namespace rental-ml-custom

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
            -t|--timeout)
                WAIT_TIMEOUT="$2"
                shift 2
                ;;
            -x|--check-external)
                CHECK_EXTERNAL=true
                shift
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
    
    log_info "Checking namespace: $NAMESPACE"
}

# Check Kubernetes connectivity
check_kubernetes_connectivity() {
    log_info "Checking Kubernetes connectivity..."
    
    if kubectl cluster-info &> /dev/null; then
        check_passed "Kubernetes cluster connectivity"
    else
        check_failed "Kubernetes cluster connectivity"
        return 1
    fi
    
    # Check namespace exists
    if kubectl get namespace "$NAMESPACE" &> /dev/null; then
        check_passed "Namespace '$NAMESPACE' exists"
    else
        check_failed "Namespace '$NAMESPACE' does not exist"
        return 1
    fi
}

# Check pod status
check_pod_status() {
    log_info "Checking pod status..."
    
    local pods_output
    pods_output=$(kubectl get pods -n "$NAMESPACE" --no-headers 2>/dev/null || echo "")
    
    if [[ -z "$pods_output" ]]; then
        check_failed "No pods found in namespace $NAMESPACE"
        return 1
    fi
    
    local total_pods=0
    local running_pods=0
    local pending_pods=0
    local failed_pods=0
    
    while IFS= read -r line; do
        if [[ -n "$line" ]]; then
            ((total_pods++))
            local status=$(echo "$line" | awk '{print $3}')
            local ready=$(echo "$line" | awk '{print $2}')
            local pod_name=$(echo "$line" | awk '{print $1}')
            
            case $status in
                "Running")
                    if [[ "$ready" == *"/"* ]]; then
                        local ready_count=$(echo "$ready" | cut -d'/' -f1)
                        local total_count=$(echo "$ready" | cut -d'/' -f2)
                        if [[ "$ready_count" == "$total_count" ]]; then
                            ((running_pods++))
                            log_verbose "Pod $pod_name is running and ready ($ready)"
                        else
                            check_warning "Pod $pod_name is running but not ready ($ready)"
                        fi
                    else
                        ((running_pods++))
                    fi
                    ;;
                "Pending")
                    ((pending_pods++))
                    check_warning "Pod $pod_name is pending"
                    ;;
                "Failed"|"Error"|"CrashLoopBackOff")
                    ((failed_pods++))
                    check_failed "Pod $pod_name is in failed state: $status"
                    ;;
                *)
                    check_warning "Pod $pod_name has unknown status: $status"
                    ;;
            esac
        fi
    done <<< "$pods_output"
    
    if [[ $failed_pods -eq 0 ]]; then
        check_passed "No failed pods found ($total_pods total, $running_pods running)"
    else
        check_failed "$failed_pods pods in failed state"
    fi
    
    if [[ $pending_pods -gt 0 ]]; then
        check_warning "$pending_pods pods still pending"
    fi
}

# Check service endpoints
check_service_endpoints() {
    log_info "Checking service endpoints..."
    
    local services=("app-service" "postgres-service" "redis-service")
    
    for service in "${services[@]}"; do
        if kubectl get service "$service" -n "$NAMESPACE" &> /dev/null; then
            local endpoints
            endpoints=$(kubectl get endpoints "$service" -n "$NAMESPACE" -o jsonpath='{.subsets[*].addresses[*].ip}' 2>/dev/null || echo "")
            
            if [[ -n "$endpoints" ]]; then
                local endpoint_count
                endpoint_count=$(echo "$endpoints" | wc -w)
                check_passed "Service $service has $endpoint_count endpoints"
                log_verbose "Service $service endpoints: $endpoints"
            else
                check_failed "Service $service has no endpoints"
            fi
        else
            check_failed "Service $service not found"
        fi
    done
}

# Check application health endpoints
check_application_health() {
    log_info "Checking application health endpoints..."
    
    # Get app service details
    local app_service="app-service"
    local service_port
    service_port=$(kubectl get service "$app_service" -n "$NAMESPACE" -o jsonpath='{.spec.ports[0].port}' 2>/dev/null || echo "")
    
    if [[ -z "$service_port" ]]; then
        check_failed "Cannot get port for service $app_service"
        return 1
    fi
    
    # Port forward to test the application
    log_verbose "Setting up port forward to test application health"
    kubectl port-forward -n "$NAMESPACE" "service/$app_service" 8080:"$service_port" &
    local pf_pid=$!
    
    # Wait for port forward to establish
    sleep 3
    
    # Test health endpoint
    local health_endpoints=("/health" "/api/health" "/health/ready" "/health/live")
    local health_check_passed=false
    
    for endpoint in "${health_endpoints[@]}"; do
        log_verbose "Testing health endpoint: $endpoint"
        if curl -f -s "http://localhost:8080$endpoint" &> /dev/null; then
            check_passed "Application health endpoint $endpoint is responding"
            health_check_passed=true
            break
        fi
    done
    
    if [[ "$health_check_passed" != "true" ]]; then
        check_failed "No application health endpoints are responding"
    fi
    
    # Test API endpoints
    local api_endpoints=("/docs" "/api/v1/properties" "/api/v1/users")
    for endpoint in "${api_endpoints[@]}"; do
        log_verbose "Testing API endpoint: $endpoint"
        local response_code
        response_code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:8080$endpoint" 2>/dev/null || echo "000")
        
        if [[ "$response_code" =~ ^[23] ]]; then
            check_passed "API endpoint $endpoint is accessible (HTTP $response_code)"
        elif [[ "$response_code" == "401" ]] || [[ "$response_code" == "403" ]]; then
            check_passed "API endpoint $endpoint is accessible but requires authentication (HTTP $response_code)"
        else
            check_warning "API endpoint $endpoint returned HTTP $response_code"
        fi
    done
    
    # Clean up port forward
    kill $pf_pid 2>/dev/null || true
    wait $pf_pid 2>/dev/null || true
}

# Check database connectivity
check_database_connectivity() {
    log_info "Checking database connectivity..."
    
    # Check PostgreSQL pod
    local postgres_pod
    postgres_pod=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/component=database --no-headers -o custom-columns=":metadata.name" | head -1)
    
    if [[ -z "$postgres_pod" ]]; then
        check_failed "PostgreSQL pod not found"
        return 1
    fi
    
    # Test database connection
    if kubectl exec -n "$NAMESPACE" "$postgres_pod" -- pg_isready -q; then
        check_passed "PostgreSQL database is ready"
    else
        check_failed "PostgreSQL database is not ready"
    fi
    
    # Test database query
    if kubectl exec -n "$NAMESPACE" "$postgres_pod" -- psql -U postgres -d rental_ml -c "SELECT 1;" &> /dev/null; then
        check_passed "PostgreSQL database query test successful"
    else
        check_failed "PostgreSQL database query test failed"
    fi
}

# Check Redis connectivity
check_redis_connectivity() {
    log_info "Checking Redis connectivity..."
    
    # Check Redis pod
    local redis_pod
    redis_pod=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/component=redis --no-headers -o custom-columns=":metadata.name" | head -1)
    
    if [[ -z "$redis_pod" ]]; then
        check_failed "Redis pod not found"
        return 1
    fi
    
    # Test Redis connection
    if kubectl exec -n "$NAMESPACE" "$redis_pod" -- redis-cli ping | grep -q "PONG"; then
        check_passed "Redis is responding to ping"
    else
        check_failed "Redis is not responding to ping"
    fi
    
    # Test Redis operations
    if kubectl exec -n "$NAMESPACE" "$redis_pod" -- redis-cli set health_check_test "ok" &> /dev/null && \
       kubectl exec -n "$NAMESPACE" "$redis_pod" -- redis-cli get health_check_test | grep -q "ok"; then
        check_passed "Redis read/write operations working"
        kubectl exec -n "$NAMESPACE" "$redis_pod" -- redis-cli del health_check_test &> /dev/null || true
    else
        check_failed "Redis read/write operations failed"
    fi
}

# Check resource usage
check_resource_usage() {
    log_info "Checking resource usage..."
    
    # Check node resources
    local node_info
    node_info=$(kubectl top nodes 2>/dev/null || echo "")
    
    if [[ -n "$node_info" ]]; then
        check_passed "Node metrics are available"
        if [[ "$VERBOSE" == "true" ]]; then
            echo "$node_info"
        fi
    else
        check_warning "Node metrics are not available (metrics-server may not be installed)"
    fi
    
    # Check pod resources
    local pod_info
    pod_info=$(kubectl top pods -n "$NAMESPACE" 2>/dev/null || echo "")
    
    if [[ -n "$pod_info" ]]; then
        check_passed "Pod metrics are available"
        if [[ "$VERBOSE" == "true" ]]; then
            echo "$pod_info"
        fi
    else
        check_warning "Pod metrics are not available"
    fi
}

# Check HPA status
check_hpa_status() {
    log_info "Checking HPA status..."
    
    local hpa_output
    hpa_output=$(kubectl get hpa -n "$NAMESPACE" --no-headers 2>/dev/null || echo "")
    
    if [[ -z "$hpa_output" ]]; then
        check_warning "No HPA found in namespace $NAMESPACE"
        return 0
    fi
    
    while IFS= read -r line; do
        if [[ -n "$line" ]]; then
            local hpa_name=$(echo "$line" | awk '{print $1}')
            local current_replicas=$(echo "$line" | awk '{print $6}')
            local min_replicas=$(echo "$line" | awk '{print $5}' | cut -d'/' -f1)
            local max_replicas=$(echo "$line" | awk '{print $5}' | cut -d'/' -f2)
            
            if [[ "$current_replicas" =~ ^[0-9]+$ ]] && [[ "$min_replicas" =~ ^[0-9]+$ ]] && [[ "$max_replicas" =~ ^[0-9]+$ ]]; then
                if [[ $current_replicas -ge $min_replicas ]] && [[ $current_replicas -le $max_replicas ]]; then
                    check_passed "HPA $hpa_name is within bounds ($current_replicas replicas, min: $min_replicas, max: $max_replicas)"
                else
                    check_warning "HPA $hpa_name may have scaling issues ($current_replicas replicas, min: $min_replicas, max: $max_replicas)"
                fi
            else
                check_warning "HPA $hpa_name status unclear"
            fi
        fi
    done <<< "$hpa_output"
}

# Check persistent volumes
check_persistent_volumes() {
    log_info "Checking persistent volumes..."
    
    local pvcs
    pvcs=$(kubectl get pvc -n "$NAMESPACE" --no-headers 2>/dev/null || echo "")
    
    if [[ -z "$pvcs" ]]; then
        check_warning "No PVCs found in namespace $NAMESPACE"
        return 0
    fi
    
    while IFS= read -r line; do
        if [[ -n "$line" ]]; then
            local pvc_name=$(echo "$line" | awk '{print $1}')
            local status=$(echo "$line" | awk '{print $2}')
            
            if [[ "$status" == "Bound" ]]; then
                check_passed "PVC $pvc_name is bound"
            else
                check_failed "PVC $pvc_name is not bound (status: $status)"
            fi
        fi
    done <<< "$pvcs"
}

# Check external dependencies
check_external_dependencies() {
    if [[ "$CHECK_EXTERNAL" != "true" ]]; then
        return 0
    fi
    
    log_info "Checking external dependencies..."
    
    # Test DNS resolution
    if kubectl run dns-test --image=busybox --rm -it --restart=Never -n "$NAMESPACE" -- nslookup google.com &> /dev/null; then
        check_passed "DNS resolution is working"
    else
        check_failed "DNS resolution is not working"
    fi
    
    # Test internet connectivity
    if kubectl run connectivity-test --image=busybox --rm -it --restart=Never -n "$NAMESPACE" -- wget -q -O- http://httpbin.org/ip &> /dev/null; then
        check_passed "Internet connectivity is working"
    else
        check_warning "Internet connectivity may be limited"
    fi
}

# Generate health report
generate_health_report() {
    echo
    log_info "=== HEALTH CHECK SUMMARY ==="
    echo
    
    local success_rate=0
    if [[ $TOTAL_CHECKS -gt 0 ]]; then
        success_rate=$((PASSED_CHECKS * 100 / TOTAL_CHECKS))
    fi
    
    echo "Total Checks: $TOTAL_CHECKS"
    echo "Passed: $PASSED_CHECKS"
    echo "Failed: $FAILED_CHECKS"
    echo "Success Rate: $success_rate%"
    echo
    
    if [[ $FAILED_CHECKS -eq 0 ]]; then
        log_success "All critical health checks passed! ✓"
        return 0
    elif [[ $success_rate -ge 80 ]]; then
        log_warning "Most health checks passed, but some issues detected"
        return 1
    else
        log_error "Multiple health check failures detected"
        return 2
    fi
}

# Main function
main() {
    log_info "Starting Rental ML System health checks..."
    log_info "Environment: $ENVIRONMENT"
    
    if [[ "$CHECK_EXTERNAL" == "true" ]]; then
        log_info "Including external dependency checks"
    fi
    
    echo
    
    # Execute all health checks
    check_kubernetes_connectivity || true
    check_pod_status || true
    check_service_endpoints || true
    check_application_health || true
    check_database_connectivity || true
    check_redis_connectivity || true
    check_resource_usage || true
    check_hpa_status || true
    check_persistent_volumes || true
    check_external_dependencies || true
    
    # Generate final report
    local exit_code
    generate_health_report
    exit_code=$?
    
    exit $exit_code
}

# Parse arguments and run main function
parse_args "$@"
set_namespace
main