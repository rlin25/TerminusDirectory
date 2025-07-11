# Rental ML System - Kubernetes Deployment

This directory contains comprehensive Kubernetes deployment configurations for the Rental ML System, a machine learning platform for rental property recommendations.

## 📁 Directory Structure

```
k8s/
├── README.md                           # This file
├── 01-namespace.yaml                   # Namespace definitions
├── 02-rbac.yaml                        # RBAC configuration
├── 03-configmaps.yaml                  # Application configuration
├── 04-secrets.yaml                     # Sensitive data (template)
├── 05-storage.yaml                     # Storage classes and volumes
├── 06-postgres-deployment.yaml         # PostgreSQL database
├── 07-redis-deployment.yaml            # Redis cache
├── 08-app-deployment.yaml              # Main application
├── 09-worker-deployment.yaml           # Celery workers
├── 10-scheduler-deployment.yaml        # Celery scheduler
├── 11-services.yaml                    # Kubernetes services
├── 12-ingress.yaml                     # Ingress configuration
├── 13-hpa.yaml                         # Horizontal Pod Autoscaler
├── 14-network-policies.yaml            # Network security policies
├── helm/                               # Helm charts
│   └── rental-ml/
│       ├── Chart.yaml                  # Helm chart metadata
│       ├── values.yaml                 # Default values
│       ├── values-dev.yaml             # Development values
│       ├── values-prod.yaml            # Production values
│       └── templates/                  # Helm templates
├── overlays/                           # Kustomize overlays
│   ├── dev/                           # Development environment
│   ├── staging/                       # Staging environment
│   └── prod/                          # Production environment
├── monitoring/                         # Monitoring stack
│   ├── prometheus-deployment.yaml     # Prometheus monitoring
│   └── grafana-deployment.yaml        # Grafana dashboards
└── scripts/                           # Deployment utilities
    ├── deploy.sh                      # Main deployment script
    ├── health-check.sh                # Health check script
    ├── backup.sh                      # Backup utilities
    └── migrate.sh                     # Database migration
```

## 🚀 Quick Start

### Prerequisites

1. **Kubernetes Cluster**: A running Kubernetes cluster (v1.20+)
2. **kubectl**: Kubernetes CLI tool configured
3. **Storage**: Persistent volume support
4. **Ingress Controller**: NGINX Ingress Controller (optional but recommended)
5. **Cert Manager**: For SSL certificates (optional)

### Development Deployment

```bash
# Clone the repository
git clone <repository-url>
cd rental-ml-system/k8s

# Deploy to development environment
./scripts/deploy.sh --environment dev

# Check deployment status
./scripts/health-check.sh --environment dev

# Access the application (if using port-forward)
kubectl port-forward -n rental-ml-dev service/app-service 8000:8000
```

### Production Deployment

```bash
# Deploy to production environment
./scripts/deploy.sh --environment prod --image-tag v1.0.0

# Verify deployment
./scripts/health-check.sh --environment prod --check-external
```

## 🔧 Deployment Methods

### Method 1: Kustomize (Recommended)

```bash
# Development
kustomize build overlays/dev | kubectl apply -f -

# Staging
kustomize build overlays/staging | kubectl apply -f -

# Production
kustomize build overlays/prod | kubectl apply -f -
```

### Method 2: Helm

```bash
# Add dependencies
helm dependency update helm/rental-ml

# Development
helm upgrade --install rental-ml-dev helm/rental-ml \
  --namespace rental-ml-dev \
  --create-namespace \
  --values helm/rental-ml/values-dev.yaml

# Production
helm upgrade --install rental-ml helm/rental-ml \
  --namespace rental-ml \
  --create-namespace \
  --values helm/rental-ml/values-prod.yaml
```

### Method 3: Manual kubectl

```bash
# Apply manifests in order
kubectl apply -f 01-namespace.yaml
kubectl apply -f 02-rbac.yaml
kubectl apply -f 03-configmaps.yaml
kubectl apply -f 04-secrets.yaml
kubectl apply -f 05-storage.yaml
kubectl apply -f 06-postgres-deployment.yaml
kubectl apply -f 07-redis-deployment.yaml
kubectl apply -f 08-app-deployment.yaml
kubectl apply -f 09-worker-deployment.yaml
kubectl apply -f 10-scheduler-deployment.yaml
kubectl apply -f 11-services.yaml
kubectl apply -f 12-ingress.yaml
kubectl apply -f 13-hpa.yaml
kubectl apply -f 14-network-policies.yaml
```

## 🏗️ Architecture

### Components

- **Application (FastAPI)**: Main API server handling HTTP requests
- **Workers (Celery)**: Background task processors for ML training and data scraping
- **Scheduler (Celery Beat)**: Task scheduler for periodic jobs
- **Database (PostgreSQL)**: Primary data storage with ML models and user data
- **Cache (Redis)**: Session storage and task queue for Celery
- **Reverse Proxy (Nginx)**: Load balancing and SSL termination
- **Monitoring (Prometheus + Grafana)**: Metrics collection and visualization

### Network Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Internet      │────│  Ingress        │────│  Load Balancer  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                        ┌───────▼───────┐
                        │     Nginx     │
                        └───────┬───────┘
                                │
                    ┌───────────▼───────────┐
                    │    Application Pods   │
                    │  ┌─────┐ ┌─────┐ ┌─────┐│
                    │  │App-1││App-2││App-3││
                    │  └─────┘ └─────┘ └─────┘│
                    └───────────┬───────────┘
                                │
                ┌───────────────┼───────────────┐
                │               │               │
        ┌───────▼───────┐ ┌─────▼─────┐ ┌───────▼───────┐
        │  PostgreSQL   │ │   Redis   │ │    Workers    │
        │   Database    │ │   Cache   │ │   (Celery)    │
        └───────────────┘ └───────────┘ └───────────────┘
```

## 🔒 Security

### Security Features Implemented

1. **RBAC**: Role-based access control for service accounts
2. **Network Policies**: Pod-to-pod communication restrictions
3. **Security Contexts**: Non-root containers with restricted capabilities
4. **Secrets Management**: Encrypted storage for sensitive data
5. **Pod Security Policies**: Container security standards
6. **TLS/SSL**: Encrypted communication via Ingress

### Security Configuration

```bash
# Update secrets before deployment
vim 04-secrets.yaml

# Ensure secrets are base64 encoded
echo -n "your-secret-value" | base64
```

## 📊 Monitoring

### Prometheus Metrics

- Application metrics (HTTP requests, response times, errors)
- System metrics (CPU, memory, disk usage)
- Database metrics (connections, query performance)
- Cache metrics (Redis operations, memory usage)
- Worker metrics (Celery tasks, queue length)

### Grafana Dashboards

1. **Overview Dashboard**: System-wide metrics and health status
2. **Application Dashboard**: API performance and user activity
3. **Database Dashboard**: PostgreSQL performance metrics
4. **Worker Dashboard**: Celery task processing metrics
5. **Infrastructure Dashboard**: Kubernetes cluster metrics

### Accessing Monitoring

```bash
# Port forward to Grafana
kubectl port-forward -n rental-ml service/grafana-service 3000:3000

# Port forward to Prometheus
kubectl port-forward -n rental-ml service/prometheus-service 9090:9090
```

## 🔄 Auto-scaling

### Horizontal Pod Autoscaler (HPA)

The system includes HPA configurations for:

- **Application Pods**: Scale 3-10 replicas based on CPU/memory
- **Worker Pods**: Scale 2-8 replicas based on queue length

### Vertical Pod Autoscaler (VPA)

For automatic resource request/limit adjustments:

```bash
# Install VPA (if not available)
kubectl apply -f https://github.com/kubernetes/autoscaler/releases/latest/download/vpa-release.yaml

# Apply VPA configuration
kubectl apply -f vpa-config.yaml
```

## 💾 Storage

### Persistent Volumes

- **PostgreSQL**: 50Gi for production, 10Gi for development
- **Redis**: 20Gi for production, 2Gi for development
- **ML Models**: 50Gi shared storage for model artifacts
- **Application Data**: 100Gi for logs and temporary files
- **Monitoring**: 20Gi for Prometheus metrics, 5Gi for Grafana

### Backup Strategy

```bash
# Database backup
./scripts/backup.sh --type database --environment prod

# Application data backup
./scripts/backup.sh --type data --environment prod

# Full system backup
./scripts/backup.sh --type full --environment prod
```

## 🌍 Environment Configuration

### Development Environment

- **Namespace**: `rental-ml-dev`
- **Replicas**: Minimal (1 per service)
- **Resources**: Low resource requests/limits
- **Features**: Debug logging, development tools enabled
- **Storage**: Standard storage class
- **Monitoring**: Disabled for performance

### Staging Environment

- **Namespace**: `rental-ml-staging`
- **Replicas**: Medium (2-3 per service)
- **Resources**: Moderate resource allocation
- **Features**: Production-like configuration with testing tools
- **Storage**: SSD storage class
- **Monitoring**: Enabled

### Production Environment

- **Namespace**: `rental-ml`
- **Replicas**: High availability (3+ per service)
- **Resources**: Production resource allocation
- **Features**: Full monitoring, alerting, backup
- **Storage**: Fast SSD with replication
- **Monitoring**: Full monitoring stack

## 🚨 Troubleshooting

### Common Issues

1. **Pods Stuck in Pending**
   ```bash
   kubectl describe pod <pod-name> -n <namespace>
   # Check for resource constraints or storage issues
   ```

2. **Application Not Responding**
   ```bash
   # Check logs
   kubectl logs -f deployment/app -n <namespace>
   
   # Run health check
   ./scripts/health-check.sh --environment <env> --verbose
   ```

3. **Database Connection Issues**
   ```bash
   # Test database connectivity
   kubectl exec -it deployment/app -n <namespace> -- python -c "from src.infrastructure.data.config import db; print(db.engine.execute('SELECT 1').scalar())"
   ```

4. **Storage Issues**
   ```bash
   # Check PVC status
   kubectl get pvc -n <namespace>
   
   # Check available storage
   kubectl get pv
   ```

### Log Access

```bash
# Application logs
kubectl logs -f deployment/app -n <namespace>

# Worker logs
kubectl logs -f deployment/worker -n <namespace>

# Database logs
kubectl logs -f statefulset/postgres -n <namespace>

# All pods logs
kubectl logs -f -l app.kubernetes.io/name=rental-ml-system -n <namespace>
```

### Debug Mode

```bash
# Enable debug mode
kubectl patch deployment app -n <namespace> -p '{"spec":{"template":{"spec":{"containers":[{"name":"app","env":[{"name":"LOG_LEVEL","value":"DEBUG"}]}]}}}}'

# Port forward for debugging
kubectl port-forward deployment/app -n <namespace> 5678:5678
```

## 📈 Performance Tuning

### Resource Optimization

1. **Monitor resource usage**:
   ```bash
   kubectl top pods -n <namespace>
   kubectl top nodes
   ```

2. **Adjust resource requests/limits** based on actual usage
3. **Tune JVM/Python settings** for application containers
4. **Configure database** connection pooling and query optimization
5. **Optimize Redis** memory usage and eviction policies

### Scaling Recommendations

- **Application**: Start with 3 replicas, scale up to 10 based on traffic
- **Workers**: Start with 2 replicas, scale based on queue length
- **Database**: Single instance with read replicas for high read loads
- **Cache**: Single Redis instance, consider clustering for high availability

## 🔄 Updates and Rollbacks

### Rolling Updates

```bash
# Update application image
kubectl set image deployment/app app=rental-ml-system:v1.1.0 -n <namespace>

# Monitor rollout
kubectl rollout status deployment/app -n <namespace>
```

### Rollbacks

```bash
# View rollout history
kubectl rollout history deployment/app -n <namespace>

# Rollback to previous version
kubectl rollout undo deployment/app -n <namespace>

# Rollback to specific revision
kubectl rollout undo deployment/app --to-revision=2 -n <namespace>
```

## 📝 Migration Guide

### From Docker Compose

1. **Export data** from existing Docker containers
2. **Update secrets** in Kubernetes manifests
3. **Deploy** using preferred method
4. **Import data** to new deployment
5. **Update DNS** to point to new ingress

### Database Migration

```bash
# Run database migrations
./scripts/migrate.sh --environment <env>

# Manual migration
kubectl exec -it deployment/app -n <namespace> -- python -m alembic upgrade head
```

## 🆘 Support

For issues and questions:

1. **Check logs** using the troubleshooting section above
2. **Run health checks** to identify issues
3. **Review monitoring** dashboards for system metrics
4. **Consult documentation** in the `/docs` directory
5. **Open an issue** on the project repository

## 📚 Additional Resources

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Helm Documentation](https://helm.sh/docs/)
- [Kustomize Documentation](https://kustomize.io/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Celery Documentation](https://docs.celeryproject.org/)

---

**Note**: This deployment configuration includes production-ready security, monitoring, and scaling configurations. Always review and customize the settings according to your specific requirements and security policies.