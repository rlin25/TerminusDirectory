# Production Data Collection and Real-time Processing Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the enterprise-grade data collection and real-time processing system for the Rental ML platform. The system is designed to handle 1M+ property updates per day, 10,000+ concurrent user interactions, and provides <1 second latency for real-time processing.

## Architecture Overview

### Core Components

1. **Real-time Data Ingestion** (`src/data/ingestion/`)
   - Kafka-based streaming pipeline with aiokafka
   - WebSocket handler for real-time user interactions
   - Apache Pulsar integration for event-driven architecture
   - Schema registry with versioning and evolution
   - Backpressure control and flow management

2. **Enhanced Property Scraping** (`src/data/scraping/`)
   - Distributed scraping infrastructure
   - Anti-bot protection and CAPTCHA solving
   - Proxy rotation and IP management
   - Real-time quality control and monitoring

3. **Data Processing Pipeline** (`src/data/processing/`)
   - Apache Spark integration for large-scale processing
   - Real-time feature extraction and transformation
   - Data enrichment with external APIs
   - Duplicate detection and data fusion

4. **Event Streaming Architecture** (`src/data/events/`)
   - Event sourcing and CQRS patterns
   - Saga orchestration for distributed transactions
   - Event store with replay capabilities
   - Dead letter queues for error handling

5. **Data Lake Integration** (`src/data/lake/`)
   - Multi-tiered storage (hot, warm, cold, archive)
   - Delta Lake for ACID transactions
   - Automated lifecycle management
   - Cost optimization and tiering

6. **Real-time Analytics** (`src/data/analytics/`)
   - Apache Flink stream processing
   - Real-time aggregations and windowing
   - Complex event processing (CEP)
   - Anomaly detection in data streams

## Prerequisites

### Infrastructure Requirements

- **Kubernetes Cluster**: v1.25+ with minimum 32 GB RAM, 16 CPU cores
- **Apache Kafka**: v3.0+ cluster with minimum 3 brokers
- **Apache Pulsar**: v2.10+ with BookKeeper for persistence
- **Apache Spark**: v3.4+ cluster with minimum 4 executors
- **Apache Flink**: v1.17+ for stream processing
- **PostgreSQL**: v14+ with read replicas
- **Redis Cluster**: v7.0+ with 6 nodes (3 master, 3 replica)
- **Prometheus + Grafana**: For monitoring and alerting

### Software Dependencies

```bash
# Install production dependencies
pip install -r requirements/prod.txt

# Key components:
# - aiokafka>=0.8.0 (Kafka async client)
# - pulsar-client>=3.2.0 (Pulsar integration)
# - pyspark>=3.4.0 (Spark processing)
# - apache-flink>=1.17.0 (Stream processing)
# - websockets>=11.0.0 (WebSocket support)
# - jsonschema>=4.17.0 (Schema validation)
# - delta-spark>=2.4.0 (Delta Lake)
```

## Deployment Steps

### 1. Infrastructure Setup

#### Kafka Cluster Configuration

```yaml
# kafka-cluster.yaml
apiVersion: kafka.strimzi.io/v1beta2
kind: Kafka
metadata:
  name: rental-ml-kafka
spec:
  kafka:
    version: 3.5.0
    replicas: 3
    listeners:
      - name: plain
        port: 9092
        type: internal
        tls: false
      - name: tls
        port: 9093
        type: internal
        tls: true
    config:
      offsets.topic.replication.factor: 3
      transaction.state.log.replication.factor: 3
      transaction.state.log.min.isr: 2
      default.replication.factor: 3
      min.insync.replicas: 2
      inter.broker.protocol.version: "3.5"
    storage:
      type: persistent-claim
      size: 500Gi
      class: fast-ssd
  zookeeper:
    replicas: 3
    storage:
      type: persistent-claim
      size: 100Gi
      class: fast-ssd
```

#### Pulsar Cluster Configuration

```yaml
# pulsar-cluster.yaml
apiVersion: pulsar.apache.org/v1beta1
kind: PulsarCluster
metadata:
  name: rental-ml-pulsar
spec:
  pulsar:
    image: apachepulsar/pulsar:2.11.0
    broker:
      replicas: 3
      resources:
        requests:
          memory: "4Gi"
          cpu: "2"
    bookkeeper:
      replicas: 3
      resources:
        requests:
          memory: "4Gi"
          cpu: "2"
      storage:
        reclaimPolicy: Retain
        size: 200Gi
    zookeeper:
      replicas: 3
      resources:
        requests:
          memory: "2Gi"
          cpu: "1"
```

### 2. Application Deployment

#### Data Ingestion Service

```yaml
# ingestion-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: data-ingestion-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: data-ingestion
  template:
    metadata:
      labels:
        app: data-ingestion
    spec:
      containers:
      - name: ingestion
        image: rental-ml/data-ingestion:latest
        ports:
        - containerPort: 8001
        env:
        - name: KAFKA_BOOTSTRAP_SERVERS
          value: "rental-ml-kafka-bootstrap:9092"
        - name: PULSAR_SERVICE_URL
          value: "pulsar://rental-ml-pulsar:6650"
        - name: REDIS_URL
          value: "redis://redis-cluster:6379"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8001
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8001
          initialDelaySeconds: 5
          periodSeconds: 5
```

#### WebSocket Handler Service

```yaml
# websocket-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: websocket-handler
spec:
  replicas: 5
  selector:
    matchLabels:
      app: websocket-handler
  template:
    metadata:
      labels:
        app: websocket-handler
    spec:
      containers:
      - name: websocket
        image: rental-ml/websocket-handler:latest
        ports:
        - containerPort: 8002
        env:
        - name: MAX_CONNECTIONS
          value: "2000"
        - name: KAFKA_BOOTSTRAP_SERVERS
          value: "rental-ml-kafka-bootstrap:9092"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1"
```

#### Spark Processing Service

```yaml
# spark-deployment.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: spark-config
data:
  spark-defaults.conf: |
    spark.master                     spark://spark-master:7077
    spark.executor.memory            2g
    spark.executor.cores             2
    spark.executor.instances         4
    spark.sql.adaptive.enabled      true
    spark.sql.adaptive.coalescePartitions.enabled true
    spark.serializer                 org.apache.spark.serializer.KryoSerializer
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: spark-processing
spec:
  replicas: 2
  selector:
    matchLabels:
      app: spark-processing
  template:
    metadata:
      labels:
        app: spark-processing
    spec:
      containers:
      - name: processing
        image: rental-ml/spark-processing:latest
        ports:
        - containerPort: 8003
        volumeMounts:
        - name: spark-config
          mountPath: /opt/spark/conf
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
      volumes:
      - name: spark-config
        configMap:
          name: spark-config
```

### 3. Configuration Management

#### Environment Configuration

```yaml
# config-map.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: rental-ml-config
data:
  # Kafka Configuration
  KAFKA_BOOTSTRAP_SERVERS: "rental-ml-kafka-bootstrap:9092"
  KAFKA_SECURITY_PROTOCOL: "PLAINTEXT"
  KAFKA_ACKS: "all"
  KAFKA_RETRIES: "5"
  KAFKA_BATCH_SIZE: "16384"
  
  # Pulsar Configuration
  PULSAR_SERVICE_URL: "pulsar://rental-ml-pulsar:6650"
  PULSAR_ADMIN_URL: "http://rental-ml-pulsar:8080"
  PULSAR_TENANT: "rental-ml"
  PULSAR_NAMESPACE: "events"
  
  # Database Configuration
  DATABASE_URL: "postgresql://username:password@postgres:5432/rental_ml"
  DATABASE_POOL_SIZE: "20"
  DATABASE_MAX_OVERFLOW: "30"
  
  # Redis Configuration
  REDIS_URL: "redis://redis-cluster:6379"
  REDIS_CLUSTER_NODES: "redis-1:6379,redis-2:6379,redis-3:6379"
  
  # Processing Configuration
  SPARK_MASTER: "spark://spark-master:7077"
  FLINK_JOB_MANAGER: "flink-jobmanager:8081"
  
  # Quality Configuration
  DATA_QUALITY_THRESHOLD: "0.7"
  VALIDATION_ENABLED: "true"
  SCHEMA_REGISTRY_URL: "http://schema-registry:8081"
  
  # Monitoring Configuration
  PROMETHEUS_URL: "http://prometheus:9090"
  GRAFANA_URL: "http://grafana:3000"
  ALERTMANAGER_URL: "http://alertmanager:9093"
```

#### Secrets Management

```yaml
# secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: rental-ml-secrets
type: Opaque
data:
  # Base64 encoded values
  database-password: <base64-encoded-password>
  kafka-username: <base64-encoded-username>
  kafka-password: <base64-encoded-password>
  pulsar-token: <base64-encoded-token>
  api-keys: <base64-encoded-api-keys>
```

### 4. Monitoring Setup

#### Prometheus Configuration

```bash
# Deploy monitoring stack
kubectl apply -f config/monitoring/production-monitoring.yaml
kubectl apply -f config/monitoring/alerts/
```

#### Grafana Dashboards

```json
{
  "dashboard": {
    "title": "Rental ML Data Pipeline",
    "panels": [
      {
        "title": "Message Throughput",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(kafka_messages_produced_total[5m])",
            "legendFormat": "Messages/sec"
          }
        ]
      },
      {
        "title": "Data Quality Score",
        "type": "stat",
        "targets": [
          {
            "expr": "avg(data_quality_score)",
            "legendFormat": "Quality Score"
          }
        ]
      },
      {
        "title": "Processing Latency",
        "type": "heatmap",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(kafka_produce_latency_seconds_bucket[5m]))",
            "legendFormat": "95th Percentile"
          }
        ]
      }
    ]
  }
}
```

### 5. Performance Tuning

#### Kafka Optimization

```properties
# server.properties
num.network.threads=8
num.io.threads=16
socket.send.buffer.bytes=102400
socket.receive.buffer.bytes=102400
socket.request.max.bytes=104857600

# Producer optimization
batch.size=65536
linger.ms=5
compression.type=lz4
acks=all
retries=2147483647

# Consumer optimization
fetch.min.bytes=50000
fetch.max.wait.ms=500
max.partition.fetch.bytes=1048576
```

#### Spark Optimization

```python
# spark-optimization.py
spark_config = {
    "spark.sql.adaptive.enabled": "true",
    "spark.sql.adaptive.coalescePartitions.enabled": "true",
    "spark.sql.adaptive.skewJoin.enabled": "true",
    "spark.sql.cbo.enabled": "true",
    "spark.sql.cbo.joinReorder.enabled": "true",
    "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
    "spark.sql.execution.arrow.pyspark.enabled": "true",
    "spark.sql.execution.arrow.maxRecordsPerBatch": "10000"
}
```

### 6. Security Configuration

#### Network Policies

```yaml
# network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: rental-ml-network-policy
spec:
  podSelector:
    matchLabels:
      app: rental-ml
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: rental-ml
    ports:
    - protocol: TCP
      port: 8000-8010
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: kafka
    ports:
    - protocol: TCP
      port: 9092
```

#### RBAC Configuration

```yaml
# rbac.yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: rental-ml-role
rules:
- apiGroups: [""]
  resources: ["pods", "configmaps", "secrets"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources: ["deployments"]
  verbs: ["get", "list", "watch", "update"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: rental-ml-binding
subjects:
- kind: ServiceAccount
  name: rental-ml-service-account
roleRef:
  kind: Role
  name: rental-ml-role
  apiGroup: rbac.authorization.k8s.io
```

## Operations Guide

### Health Checks

```bash
# Check component health
kubectl get pods -l app=rental-ml
kubectl logs -l app=data-ingestion --tail=100
kubectl exec -it <pod-name> -- curl localhost:8001/health

# Check Kafka health
kubectl exec -it rental-ml-kafka-0 -- kafka-topics.sh --bootstrap-server localhost:9092 --list

# Check processing metrics
curl http://processing-service:8003/metrics | grep spark_jobs_executed
```

### Scaling Operations

```bash
# Scale ingestion service
kubectl scale deployment data-ingestion-service --replicas=5

# Scale WebSocket handlers
kubectl scale deployment websocket-handler --replicas=10

# Scale Spark executors
kubectl patch sparkcluster rental-ml-spark --type='merge' -p='{"spec":{"executor":{"instances":8}}}'
```

### Troubleshooting

#### Common Issues and Solutions

1. **High Kafka Consumer Lag**
   ```bash
   # Check consumer group status
   kubectl exec -it rental-ml-kafka-0 -- kafka-consumer-groups.sh \
     --bootstrap-server localhost:9092 --describe --group rental-ml-consumers
   
   # Scale consumers
   kubectl scale deployment data-processing --replicas=6
   ```

2. **WebSocket Connection Issues**
   ```bash
   # Check connection limits
   kubectl logs -l app=websocket-handler | grep "connection limit"
   
   # Scale WebSocket handlers
   kubectl scale deployment websocket-handler --replicas=8
   ```

3. **Data Quality Issues**
   ```bash
   # Check validation metrics
   curl http://quality-service:8005/metrics | grep validation_violations
   
   # Review quality reports
   kubectl exec -it <quality-pod> -- python -c "
   from src.data.ingestion.data_validator import RealTimeDataValidator
   validator = RealTimeDataValidator()
   print(validator.get_quality_report())
   "
   ```

### Maintenance Procedures

#### Rolling Updates

```bash
# Update ingestion service
kubectl set image deployment/data-ingestion-service \
  ingestion=rental-ml/data-ingestion:v2.0.0

# Monitor rollout
kubectl rollout status deployment/data-ingestion-service

# Rollback if needed
kubectl rollout undo deployment/data-ingestion-service
```

#### Backup and Recovery

```bash
# Backup Kafka topics
kubectl exec -it rental-ml-kafka-0 -- kafka-mirror-maker.sh \
  --consumer.config consumer.properties \
  --producer.config producer.properties \
  --whitelist="property-events|user-events"

# Backup PostgreSQL
kubectl exec -it postgres-0 -- pg_dump rental_ml > backup.sql

# Backup Redis cluster
kubectl exec -it redis-0 -- redis-cli --rdb /backup/dump.rdb
```

## Performance Benchmarks

### Expected Performance Metrics

- **Message Throughput**: 10,000+ messages/second
- **Processing Latency**: <100ms (p95)
- **WebSocket Connections**: 10,000+ concurrent
- **Data Quality Score**: >0.95
- **System Availability**: 99.9%
- **Error Rate**: <0.1%

### Load Testing

```python
# load-test.py
import asyncio
import aiohttp
from locust import HttpUser, task, between

class DataPipelineUser(HttpUser):
    wait_time = between(1, 3)
    
    @task(3)
    def send_property_event(self):
        self.client.post("/api/events/property", json={
            "property_id": "test-123",
            "event_type": "property_updated",
            "data": {"price": 2000}
        })
    
    @task(2)
    def send_user_event(self):
        self.client.post("/api/events/user", json={
            "user_id": "user-456",
            "event_type": "property_viewed",
            "data": {"property_id": "test-123"}
        })
    
    @task(1)
    def websocket_connection(self):
        # WebSocket load testing would be implemented separately
        pass
```

## Cost Optimization

### Resource Allocation

- **Development**: 4 vCPU, 16GB RAM, 100GB storage
- **Staging**: 8 vCPU, 32GB RAM, 500GB storage
- **Production**: 32 vCPU, 128GB RAM, 2TB storage

### Cost Monitoring

```yaml
# cost-monitoring.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: cost-alerts
data:
  prometheus-rules: |
    groups:
    - name: cost_alerts
      rules:
      - alert: HighResourceUsage
        expr: |
          sum(rate(container_cpu_usage_seconds_total[5m])) by (pod) > 2
        annotations:
          summary: "Pod {{ $labels.pod }} using high CPU"
          cost_impact: "High CPU usage increases costs"
```

## Compliance and Security

### Data Privacy

- GDPR compliance with data anonymization
- User consent tracking in events
- Right to erasure implementation
- Data retention policies

### Security Measures

- TLS encryption for all communications
- mTLS for internal service communication
- API rate limiting and authentication
- Regular security audits and penetration testing

## Conclusion

This deployment guide provides a comprehensive foundation for running the enterprise-grade data collection and real-time processing system. The architecture is designed for high availability, scalability, and performance while maintaining strong security and compliance standards.

For additional support, refer to the operational runbooks and contact the platform engineering team.