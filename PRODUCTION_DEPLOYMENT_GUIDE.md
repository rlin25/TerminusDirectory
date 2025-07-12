# Production Deployment Guide - Rental ML System

This guide provides comprehensive instructions for deploying the Rental ML System to production with enterprise-grade security, monitoring, and scalability.

## 🚀 Overview

The Rental ML System production deployment includes:

- **Multi-stage Docker containers** with security hardening
- **Kubernetes orchestration** with Istio service mesh
- **Infrastructure as Code** (Terraform) for AWS/GCP
- **CI/CD pipelines** with automated testing and deployment
- **Comprehensive monitoring** with Prometheus, Grafana, and ELK stack
- **Security and compliance** with GDPR compliance and disaster recovery

## 📋 Prerequisites

### Required Tools
```bash
# Kubernetes and container tools
kubectl >= 1.27
helm >= 3.12
istioctl >= 1.18
docker >= 24.0

# Infrastructure tools
terraform >= 1.5
aws-cli >= 2.13 (for AWS deployment)
gcloud >= 434.0 (for GCP deployment)

# Development tools
git >= 2.40
make >= 4.3
```

### Required Permissions
- Kubernetes cluster admin access
- Cloud provider admin permissions (AWS/GCP)
- Container registry push permissions
- DNS management access

## 🏗️ Infrastructure Setup

### 1. Cloud Infrastructure Deployment

#### AWS Deployment
```bash
# Navigate to Terraform AWS directory
cd terraform/aws

# Initialize Terraform
terraform init

# Create terraform.tfvars file
cat > terraform.tfvars << EOF
project_name = "rental-ml-system"
environment = "production"
aws_region = "us-west-2"
domain_name = "rental-ml.com"

# Database configuration
rds_instance_class = "db.r6g.large"
rds_allocated_storage = 100
create_rds_replica = true

# Redis configuration
redis_node_type = "cache.r6g.large"
redis_num_cache_nodes = 3

# Security settings
enable_encryption_at_rest = true
enable_encryption_in_transit = true
enable_network_policy = true

# Monitoring settings
enable_observability_stack = true
cloudwatch_log_retention_days = 30
EOF

# Plan and apply infrastructure
terraform plan
terraform apply
```

#### GCP Deployment
```bash
# Navigate to Terraform GCP directory
cd terraform/gcp

# Initialize and configure similar to AWS
terraform init
# Configure terraform.tfvars for GCP
terraform plan
terraform apply
```

### 2. Kubernetes Cluster Setup

```bash
# Get cluster credentials (AWS EKS)
aws eks update-kubeconfig --region us-west-2 --name rental-ml-prod

# Verify cluster access
kubectl cluster-info
kubectl get nodes
```

### 3. Install Istio Service Mesh

```bash
# Download and install Istio
curl -L https://istio.io/downloadIstio | sh -
export PATH=$PATH:$PWD/istio-*/bin

# Install Istio with production configuration
istioctl install --set values.pilot.env.EXTERNAL_ISTIOD=false -y

# Enable Istio injection for production namespace
kubectl label namespace rental-ml-prod istio-injection=enabled
```

## 🐳 Container Image Build and Push

### 1. Build Production Images

```bash
# Set registry variables
export REGISTRY="ghcr.io/your-org"
export VERSION="v1.0.0"

# Build API service
docker build -f docker/Dockerfile.api -t $REGISTRY/rental-ml-api:$VERSION .
docker build -f docker/Dockerfile.api -t $REGISTRY/rental-ml-api:latest .

# Build ML training service
docker build -f docker/Dockerfile.ml-training -t $REGISTRY/rental-ml-ml-training:$VERSION .
docker build -f docker/Dockerfile.ml-training -t $REGISTRY/rental-ml-ml-training:latest .

# Build scraping service
docker build -f docker/Dockerfile.scraping -t $REGISTRY/rental-ml-scraping:$VERSION .
docker build -f docker/Dockerfile.scraping -t $REGISTRY/rental-ml-scraping:latest .
```

### 2. Push Images to Registry

```bash
# Login to container registry
docker login ghcr.io

# Push all images
docker push $REGISTRY/rental-ml-api:$VERSION
docker push $REGISTRY/rental-ml-api:latest
docker push $REGISTRY/rental-ml-ml-training:$VERSION
docker push $REGISTRY/rental-ml-ml-training:latest
docker push $REGISTRY/rental-ml-scraping:$VERSION
docker push $REGISTRY/rental-ml-scraping:latest
```

## ⚙️ Kubernetes Deployment

### 1. Create Production Namespace and Resources

```bash
# Apply namespace and basic resources
kubectl apply -f k8s/production/00-namespace.yaml

# Apply storage classes and PVCs
kubectl apply -f k8s/production/03-storage-pvc.yaml

# Apply ConfigMaps and Secrets
kubectl apply -f k8s/production/02-configmaps-secrets.yaml
```

### 2. Deploy Database Infrastructure

```bash
# Deploy PostgreSQL with high availability
kubectl apply -f k8s/production/04-database-statefulset.yaml

# Wait for database to be ready
kubectl wait --for=condition=ready pod -l app=postgres-primary -n rental-ml-prod --timeout=300s

# Verify database connectivity
kubectl exec -it postgres-primary-0 -n rental-ml-prod -- psql -U postgres -c "SELECT version();"
```

### 3. Deploy Application Services

```bash
# Update image tags in deployment manifests
cd k8s/production
kustomize edit set image \
  ghcr.io/your-org/rental-ml-api:$VERSION \
  ghcr.io/your-org/rental-ml-ml-training:$VERSION \
  ghcr.io/your-org/rental-ml-scraping:$VERSION

# Apply all application deployments
kubectl apply -k k8s/production/

# Wait for deployments to be ready
kubectl wait --for=condition=available deployment --all -n rental-ml-prod --timeout=600s
```

### 4. Configure Istio Service Mesh

```bash
# Apply Istio configuration
kubectl apply -f k8s/production/01-service-mesh-istio.yaml

# Verify Istio configuration
istioctl analyze -n rental-ml-prod
kubectl get gateway,virtualservice,destinationrule -n rental-ml-prod
```

### 5. Apply Security Policies

```bash
# Apply network policies
kubectl apply -f security/policies/network-policies.yaml

# Apply GDPR compliance configuration
kubectl apply -f security/compliance/gdpr-compliance.yaml

# Apply backup and disaster recovery
kubectl apply -f security/backup-disaster-recovery.yaml
```

## 📊 Monitoring and Observability Setup

### 1. Deploy Prometheus Stack

```bash
# Add Helm repositories
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

# Install Prometheus with custom values
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace rental-ml-prod \
  --values monitoring/prometheus/values-production.yaml

# Verify Prometheus deployment
kubectl get pods -l app.kubernetes.io/name=prometheus -n rental-ml-prod
```

### 2. Configure Grafana Dashboards

```bash
# Import custom dashboards
kubectl create configmap grafana-dashboards \
  --from-file=monitoring/grafana/dashboards/ \
  -n rental-ml-prod

# Restart Grafana to load dashboards
kubectl rollout restart deployment grafana -n rental-ml-prod
```

### 3. Deploy ELK Stack

```bash
# Install Elasticsearch
helm repo add elastic https://helm.elastic.co
helm install elasticsearch elastic/elasticsearch \
  --namespace rental-ml-prod \
  --values monitoring/elasticsearch/values-production.yaml

# Install Kibana
helm install kibana elastic/kibana \
  --namespace rental-ml-prod \
  --values monitoring/kibana/values-production.yaml

# Install Logstash
helm install logstash elastic/logstash \
  --namespace rental-ml-prod \
  --values monitoring/logstash/values-production.yaml
```

### 4. Deploy Jaeger Tracing

```bash
# Install Jaeger operator
kubectl create namespace observability
kubectl apply -f https://github.com/jaegertracing/jaeger-operator/releases/download/v1.47.0/jaeger-operator.yaml -n observability

# Deploy Jaeger instance
kubectl apply -f monitoring/jaeger/jaeger-production.yaml -n rental-ml-prod
```

## 🔐 Security Configuration

### 1. TLS Certificate Setup

```bash
# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.12.0/cert-manager.yaml

# Create ClusterIssuer for Let's Encrypt
kubectl apply -f security/tls/cluster-issuer.yaml

# Create certificate for domain
kubectl apply -f security/tls/certificate.yaml
```

### 2. HashiCorp Vault Setup

```bash
# Install Vault using Helm
helm repo add hashicorp https://helm.releases.hashicorp.com
helm install vault hashicorp/vault \
  --namespace rental-ml-prod \
  --values security/vault/values-production.yaml

# Initialize and unseal Vault
kubectl exec vault-0 -n rental-ml-prod -- vault operator init
kubectl exec vault-0 -n rental-ml-prod -- vault operator unseal <UNSEAL_KEY>
```

### 3. Secret Management

```bash
# Configure Vault authentication
kubectl exec vault-0 -n rental-ml-prod -- vault auth enable kubernetes

# Setup secret injection
kubectl apply -f security/vault/secret-injection.yaml
```

## 🔄 CI/CD Pipeline Setup

### 1. GitHub Actions Configuration

```bash
# Set up GitHub secrets
gh secret set PRODUCTION_KUBECONFIG --body "$(cat ~/.kube/config | base64)"
gh secret set DOCKER_REGISTRY_TOKEN --body "$DOCKER_TOKEN"
gh secret set AWS_ACCESS_KEY_ID --body "$AWS_ACCESS_KEY_ID"
gh secret set AWS_SECRET_ACCESS_KEY --body "$AWS_SECRET_ACCESS_KEY"

# Add environment-specific secrets
gh secret set DATABASE_PASSWORD --body "$DB_PASSWORD"
gh secret set REDIS_PASSWORD --body "$REDIS_PASSWORD"
gh secret set SECRET_KEY --body "$SECRET_KEY"
```

### 2. Setup Deployment Environments

```bash
# Create GitHub environments
gh api repos/:owner/:repo/environments/production -X PUT
gh api repos/:owner/:repo/environments/staging -X PUT

# Configure environment protection rules
gh api repos/:owner/:repo/environments/production/deployment-protection-rules \
  -X POST -f type=required_reviewers -f reviewers='[{"type":"User","id":123}]'
```

## 🚀 Application Deployment

### 1. Database Migration

```bash
# Run database migrations
kubectl exec -it deployment/rental-ml-api -n rental-ml-prod -- \
  python migrations/run_migrations.py

# Verify migration status
kubectl exec -it deployment/rental-ml-api -n rental-ml-prod -- \
  python -c "from migrations.run_migrations import check_migration_status; check_migration_status()"
```

### 2. ML Model Deployment

```bash
# Deploy initial ML models
kubectl cp models/ rental-ml-training-0:/app/models/ -n rental-ml-prod

# Trigger model training job
kubectl create job --from=cronjob/ml-model-training ml-training-initial -n rental-ml-prod
```

### 3. Health Verification

```bash
# Check application health
kubectl get pods -n rental-ml-prod
kubectl get services -n rental-ml-prod
kubectl get ingress -n rental-ml-prod

# Test API endpoints
curl -f https://api.rental-ml.com/health
curl -f https://api.rental-ml.com/api/v1/health
```

## 📈 Monitoring Setup Verification

### 1. Prometheus Targets

```bash
# Check Prometheus targets
kubectl port-forward svc/prometheus-server 9090:80 -n rental-ml-prod
# Open http://localhost:9090/targets
```

### 2. Grafana Dashboard Access

```bash
# Get Grafana admin password
kubectl get secret grafana -n rental-ml-prod -o jsonpath="{.data.admin-password}" | base64 -d

# Access Grafana
kubectl port-forward svc/grafana 3000:80 -n rental-ml-prod
# Open http://localhost:3000
```

### 3. Alerting Configuration

```bash
# Verify AlertManager
kubectl port-forward svc/alertmanager 9093:9093 -n rental-ml-prod
# Open http://localhost:9093
```

## 🔧 Operational Procedures

### 1. Scaling Applications

```bash
# Scale API service
kubectl scale deployment rental-ml-api --replicas=5 -n rental-ml-prod

# Scale ML training
kubectl scale deployment rental-ml-training --replicas=2 -n rental-ml-prod

# Enable HPA
kubectl apply -f k8s/production/hpa.yaml
```

### 2. Rolling Updates

```bash
# Update API service
kubectl set image deployment/rental-ml-api \
  rental-ml-api=ghcr.io/your-org/rental-ml-api:v1.1.0 \
  -n rental-ml-prod

# Monitor rollout
kubectl rollout status deployment/rental-ml-api -n rental-ml-prod
```

### 3. Backup and Recovery

```bash
# Trigger manual backup
kubectl create job --from=cronjob/database-full-backup manual-backup-$(date +%s) -n rental-ml-prod

# Test disaster recovery
kubectl apply -f security/disaster-recovery/test-procedure.yaml
```

## 🐛 Troubleshooting

### Common Issues

1. **Pod CrashLoopBackOff**
   ```bash
   kubectl logs -f deployment/rental-ml-api -n rental-ml-prod
   kubectl describe pod <pod-name> -n rental-ml-prod
   ```

2. **Database Connection Issues**
   ```bash
   kubectl exec -it postgres-primary-0 -n rental-ml-prod -- pg_isready
   kubectl get secret rental-ml-postgres-secrets -o yaml -n rental-ml-prod
   ```

3. **Istio Service Mesh Issues**
   ```bash
   istioctl proxy-status
   istioctl proxy-config cluster <pod-name>
   ```

4. **Storage Issues**
   ```bash
   kubectl get pv,pvc -n rental-ml-prod
   kubectl describe pvc <pvc-name> -n rental-ml-prod
   ```

### Performance Optimization

1. **Database Performance**
   ```bash
   # Check slow queries
   kubectl exec -it postgres-primary-0 -n rental-ml-prod -- \
     psql -U postgres -c "SELECT query, mean_time, calls FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;"
   ```

2. **API Performance**
   ```bash
   # Check API metrics
   kubectl port-forward svc/rental-ml-api 8000:8000 -n rental-ml-prod
   curl http://localhost:8000/metrics
   ```

## 📚 Additional Resources

### Documentation Links
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Istio Documentation](https://istio.io/latest/docs/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)

### Monitoring Dashboards
- Production Overview: https://grafana.rental-ml.com/d/rental-ml-overview
- Infrastructure Metrics: https://grafana.rental-ml.com/d/infrastructure
- Application Performance: https://grafana.rental-ml.com/d/application

### Support Contacts
- DevOps Team: devops@rental-ml.com
- Security Team: security@rental-ml.com
- ML Engineering: ml-team@rental-ml.com

## 🔄 Maintenance Schedule

### Daily Tasks
- Check application health and alerts
- Review error logs and metrics
- Verify backup completion

### Weekly Tasks
- Security patch updates
- Performance optimization review
- Capacity planning assessment

### Monthly Tasks
- Disaster recovery testing
- Security audit and compliance review
- Cost optimization analysis
- Infrastructure scaling review

---

**Note**: This guide provides a comprehensive production deployment setup. Always test changes in a staging environment before applying to production. Ensure all security configurations and compliance requirements are met for your specific use case.