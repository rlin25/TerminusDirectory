#!/bin/bash

# Redis Cluster Setup Script for Rental ML System
# Sets up production-ready Redis cluster with monitoring and backup

set -e

# Configuration
REDIS_PASSWORD="${REDIS_PASSWORD:-rental_ml_redis_pass}"
SENTINEL_PASSWORD="${SENTINEL_PASSWORD:-rental_ml_sentinel_pass}"
CLUSTER_MODE="${CLUSTER_MODE:-cluster}"  # cluster, sentinel, or standalone
ENVIRONMENT="${ENVIRONMENT:-production}"
BACKUP_ENABLED="${BACKUP_ENABLED:-true}"
MONITORING_ENABLED="${MONITORING_ENABLED:-true}"

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
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if Redis CLI is available (for cluster operations)
    if ! command -v redis-cli &> /dev/null; then
        log_warning "Redis CLI not found. Installing via Docker..."
        docker pull redis:7-alpine > /dev/null 2>&1
    fi
    
    log_success "Prerequisites check completed"
}

# Generate configuration files
generate_configs() {
    log_info "Generating Redis configuration files..."
    
    # Create directories
    mkdir -p config/redis/logs
    mkdir -p config/redis/data
    mkdir -p config/redis/backup
    
    # Set environment variables
    export REDIS_PASSWORD
    export SENTINEL_PASSWORD
    
    log_success "Configuration files generated"
}

# Start Redis cluster
start_cluster() {
    log_info "Starting Redis cluster in ${CLUSTER_MODE} mode..."
    
    case $CLUSTER_MODE in
        "cluster")
            start_redis_cluster
            ;;
        "sentinel")
            start_sentinel_setup
            ;;
        "standalone")
            start_standalone
            ;;
        *)
            log_error "Invalid cluster mode: $CLUSTER_MODE"
            exit 1
            ;;
    esac
}

# Start Redis cluster mode
start_redis_cluster() {
    log_info "Starting Redis cluster nodes..."
    
    # Start Redis nodes
    docker-compose -f config/redis/docker-compose-redis-cluster.yml up -d \
        redis-node-1 redis-node-2 redis-node-3 \
        redis-node-4 redis-node-5 redis-node-6
    
    # Wait for nodes to be ready
    log_info "Waiting for Redis nodes to be ready..."
    sleep 30
    
    # Check if cluster is already initialized
    if ! docker exec redis-node-1 redis-cli -a "$REDIS_PASSWORD" cluster nodes 2>/dev/null | grep -q master; then
        log_info "Initializing Redis cluster..."
        
        # Initialize cluster
        docker run --rm --network redis-cluster-net redis:7-alpine \
            redis-cli -a "$REDIS_PASSWORD" --cluster create \
            redis-node-1:7000 redis-node-2:7001 redis-node-3:7002 \
            redis-node-4:7003 redis-node-5:7004 redis-node-6:7005 \
            --cluster-replicas 1 --cluster-yes
    else
        log_info "Redis cluster already initialized"
    fi
    
    # Start monitoring if enabled
    if [ "$MONITORING_ENABLED" = "true" ]; then
        log_info "Starting monitoring services..."
        docker-compose -f config/redis/docker-compose-redis-cluster.yml up -d \
            redis-exporter redis-commander
    fi
    
    log_success "Redis cluster started successfully"
}

# Start Sentinel setup
start_sentinel_setup() {
    log_info "Starting Redis Sentinel setup..."
    
    # Start master and replicas
    docker-compose -f config/redis/docker-compose-redis-cluster.yml \
        --profile sentinel up -d \
        redis-master redis-replica-1 redis-replica-2
    
    # Wait for master and replicas
    log_info "Waiting for Redis master and replicas to be ready..."
    sleep 20
    
    # Start Sentinel nodes
    docker-compose -f config/redis/docker-compose-redis-cluster.yml \
        --profile sentinel up -d \
        redis-sentinel-1 redis-sentinel-2 redis-sentinel-3
    
    log_success "Redis Sentinel setup started successfully"
}

# Start standalone Redis
start_standalone() {
    log_info "Starting standalone Redis..."
    
    docker run -d \
        --name redis-standalone \
        --network redis-cluster-net \
        -p 6379:6379 \
        -v redis-standalone-data:/data \
        -e REDIS_PASSWORD="$REDIS_PASSWORD" \
        redis:7-alpine \
        redis-server --requirepass "$REDIS_PASSWORD"
    
    log_success "Standalone Redis started successfully"
}

# Verify cluster health
verify_cluster() {
    log_info "Verifying cluster health..."
    
    case $CLUSTER_MODE in
        "cluster")
            verify_redis_cluster
            ;;
        "sentinel")
            verify_sentinel_setup
            ;;
        "standalone")
            verify_standalone
            ;;
    esac
}

# Verify Redis cluster
verify_redis_cluster() {
    log_info "Checking Redis cluster status..."
    
    # Check cluster info
    if docker exec redis-node-1 redis-cli -a "$REDIS_PASSWORD" cluster info | grep -q "cluster_state:ok"; then
        log_success "Redis cluster is healthy"
        
        # Show cluster nodes
        log_info "Cluster nodes:"
        docker exec redis-node-1 redis-cli -a "$REDIS_PASSWORD" cluster nodes
        
        # Test cluster operations
        log_info "Testing cluster operations..."
        docker exec redis-node-1 redis-cli -a "$REDIS_PASSWORD" set test_key "cluster_test"
        if [ "$(docker exec redis-node-1 redis-cli -a "$REDIS_PASSWORD" get test_key)" = "cluster_test" ]; then
            log_success "Cluster operations working correctly"
            docker exec redis-node-1 redis-cli -a "$REDIS_PASSWORD" del test_key
        else
            log_error "Cluster operations test failed"
        fi
    else
        log_error "Redis cluster is not healthy"
        docker exec redis-node-1 redis-cli -a "$REDIS_PASSWORD" cluster info
        exit 1
    fi
}

# Verify Sentinel setup
verify_sentinel_setup() {
    log_info "Checking Redis Sentinel status..."
    
    # Check if master is monitored
    if docker exec redis-sentinel-1 redis-cli -p 26379 -a "$SENTINEL_PASSWORD" \
        sentinel masters | grep -q "rental-ml-master"; then
        log_success "Redis Sentinel is monitoring master"
        
        # Show master info
        log_info "Master info:"
        docker exec redis-sentinel-1 redis-cli -p 26379 -a "$SENTINEL_PASSWORD" \
            sentinel masters
        
        # Test failover (optional)
        # log_info "Testing sentinel failover..."
        # docker exec redis-sentinel-1 redis-cli -p 26379 -a "$SENTINEL_PASSWORD" \
        #     sentinel failover rental-ml-master
    else
        log_error "Redis Sentinel is not monitoring master properly"
        exit 1
    fi
}

# Verify standalone Redis
verify_standalone() {
    log_info "Checking standalone Redis status..."
    
    if docker exec redis-standalone redis-cli -a "$REDIS_PASSWORD" ping | grep -q "PONG"; then
        log_success "Standalone Redis is healthy"
        
        # Test operations
        docker exec redis-standalone redis-cli -a "$REDIS_PASSWORD" set test_key "standalone_test"
        if [ "$(docker exec redis-standalone redis-cli -a "$REDIS_PASSWORD" get test_key)" = "standalone_test" ]; then
            log_success "Redis operations working correctly"
            docker exec redis-standalone redis-cli -a "$REDIS_PASSWORD" del test_key
        fi
    else
        log_error "Standalone Redis is not healthy"
        exit 1
    fi
}

# Setup backup
setup_backup() {
    if [ "$BACKUP_ENABLED" = "true" ]; then
        log_info "Setting up Redis backup..."
        
        # Create backup script
        cat > config/redis/backup.sh << 'EOF'
#!/bin/bash
# Redis backup script
BACKUP_DIR="/backup/redis"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$BACKUP_DIR"

case "$CLUSTER_MODE" in
    "cluster")
        # Backup all master nodes
        for port in 7000 7001 7002; do
            docker exec redis-node-$((port - 6999)) redis-cli -a "$REDIS_PASSWORD" BGSAVE
            sleep 5
            docker cp redis-node-$((port - 6999)):/data/dump.rdb "$BACKUP_DIR/dump_${port}_${TIMESTAMP}.rdb"
        done
        ;;
    "sentinel")
        # Backup master
        docker exec redis-master redis-cli -a "$REDIS_PASSWORD" BGSAVE
        sleep 5
        docker cp redis-master:/data/dump.rdb "$BACKUP_DIR/dump_master_${TIMESTAMP}.rdb"
        ;;
    "standalone")
        # Backup standalone
        docker exec redis-standalone redis-cli -a "$REDIS_PASSWORD" BGSAVE
        sleep 5
        docker cp redis-standalone:/data/dump.rdb "$BACKUP_DIR/dump_standalone_${TIMESTAMP}.rdb"
        ;;
esac

# Cleanup old backups (keep last 7 days)
find "$BACKUP_DIR" -name "*.rdb" -mtime +7 -delete

echo "Backup completed: $TIMESTAMP"
EOF
        
        chmod +x config/redis/backup.sh
        
        # Setup cron job for daily backups
        log_info "Setting up daily backup cron job..."
        (crontab -l 2>/dev/null; echo "0 2 * * * $(pwd)/config/redis/backup.sh") | crontab -
        
        log_success "Backup setup completed"
    fi
}

# Display connection information
show_connection_info() {
    log_info "Redis connection information:"
    
    case $CLUSTER_MODE in
        "cluster")
            echo "Redis Cluster Endpoints:"
            echo "  Node 1: localhost:7000"
            echo "  Node 2: localhost:7001"
            echo "  Node 3: localhost:7002"
            echo "  Node 4: localhost:7003"
            echo "  Node 5: localhost:7004"
            echo "  Node 6: localhost:7005"
            echo ""
            echo "Connection string: redis://localhost:7000,localhost:7001,localhost:7002"
            ;;
        "sentinel")
            echo "Redis Sentinel Endpoints:"
            echo "  Sentinel 1: localhost:26379"
            echo "  Sentinel 2: localhost:26380"
            echo "  Sentinel 3: localhost:26381"
            echo ""
            echo "Master: rental-ml-master"
            echo "Redis Master: localhost:6379"
            echo "Redis Replica 1: localhost:6380"
            echo "Redis Replica 2: localhost:6381"
            ;;
        "standalone")
            echo "Redis Standalone:"
            echo "  Endpoint: localhost:6379"
            ;;
    esac
    
    echo "Password: $REDIS_PASSWORD"
    
    if [ "$MONITORING_ENABLED" = "true" ]; then
        echo ""
        echo "Monitoring:"
        echo "  Redis Commander: http://localhost:8081"
        echo "  Redis Exporter: http://localhost:9121/metrics"
    fi
}

# Cleanup function
cleanup() {
    log_info "Cleaning up Redis setup..."
    
    # Stop and remove all containers
    docker-compose -f config/redis/docker-compose-redis-cluster.yml down -v
    
    # Remove standalone container if exists
    docker rm -f redis-standalone 2>/dev/null || true
    
    # Remove network
    docker network rm redis-cluster-net 2>/dev/null || true
    
    log_success "Cleanup completed"
}

# Main function
main() {
    echo "Redis Cluster Setup for Rental ML System"
    echo "========================================"
    
    case "${1:-start}" in
        "start")
            check_prerequisites
            generate_configs
            start_cluster
            verify_cluster
            setup_backup
            show_connection_info
            ;;
        "stop")
            log_info "Stopping Redis cluster..."
            docker-compose -f config/redis/docker-compose-redis-cluster.yml down
            log_success "Redis cluster stopped"
            ;;
        "restart")
            $0 stop
            sleep 5
            $0 start
            ;;
        "status")
            verify_cluster
            ;;
        "cleanup")
            cleanup
            ;;
        "backup")
            if [ -f config/redis/backup.sh ]; then
                ./config/redis/backup.sh
            else
                log_error "Backup script not found. Run setup first."
            fi
            ;;
        *)
            echo "Usage: $0 {start|stop|restart|status|cleanup|backup}"
            echo ""
            echo "Commands:"
            echo "  start   - Start Redis cluster"
            echo "  stop    - Stop Redis cluster"
            echo "  restart - Restart Redis cluster"
            echo "  status  - Check cluster status"
            echo "  cleanup - Remove all Redis containers and data"
            echo "  backup  - Create backup of Redis data"
            echo ""
            echo "Environment variables:"
            echo "  CLUSTER_MODE     - cluster, sentinel, or standalone (default: cluster)"
            echo "  REDIS_PASSWORD   - Redis password (default: rental_ml_redis_pass)"
            echo "  SENTINEL_PASSWORD - Sentinel password (default: rental_ml_sentinel_pass)"
            echo "  BACKUP_ENABLED   - Enable automatic backups (default: true)"
            echo "  MONITORING_ENABLED - Enable monitoring (default: true)"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"