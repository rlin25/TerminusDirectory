# Performance Monitoring Setup Guide

This guide provides step-by-step instructions for setting up comprehensive performance monitoring for the rental ML system, including metrics collection, alerting, and visualization.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Prometheus Setup](#prometheus-setup)
4. [Grafana Dashboard Setup](#grafana-dashboard-setup)
5. [AlertManager Configuration](#alertmanager-configuration)
6. [Application Metrics Integration](#application-metrics-integration)
7. [Log Aggregation Setup](#log-aggregation-setup)
8. [Performance Testing Integration](#performance-testing-integration)
9. [CI/CD Performance Gates](#cicd-performance-gates)
10. [Troubleshooting](#troubleshooting)

## Overview

The monitoring stack consists of:

- **Prometheus**: Metrics collection and storage
- **Grafana**: Visualization and dashboards
- **AlertManager**: Alert routing and notification
- **ELK Stack**: Log aggregation and analysis
- **Application Metrics**: Custom metrics from the rental ML system
- **Performance Testing**: Automated performance regression detection

### Architecture Diagram

```mermaid
graph TB
    A[Rental ML API] --> B[Prometheus Metrics]
    A --> C[Application Logs]
    D[Database] --> B
    E[Redis] --> B
    F[Load Balancer] --> B
    
    B --> G[Prometheus Server]
    C --> H[Elasticsearch]
    
    G --> I[Grafana Dashboards]
    G --> J[AlertManager]
    H --> K[Kibana]
    
    J --> L[Slack/Email Notifications]
    I --> M[Performance Dashboard]
    K --> N[Log Analysis Dashboard]
```

## Prerequisites

### System Requirements

- Docker and Docker Compose
- Kubernetes cluster (for production)
- At least 4GB RAM for monitoring stack
- 50GB storage for metrics and logs

### Network Configuration

```yaml
# docker-compose.monitoring.yml
version: '3.8'

networks:
  monitoring:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16

volumes:
  prometheus_data:
  grafana_data:
  elasticsearch_data:
```

## Prometheus Setup

### 1. Prometheus Configuration

Create `prometheus/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'rental-ml-prod'
    replica: '1'

rule_files:
  - "alerts/*.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  # Prometheus self-monitoring
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  # Rental ML API instances
  - job_name: 'rental-ml-api'
    static_configs:
      - targets: ['api-1:9091', 'api-2:9091', 'api-3:9091']
    metrics_path: '/metrics'
    scrape_interval: 10s
    scrape_timeout: 10s

  # Database monitoring
  - job_name: 'postgresql'
    static_configs:
      - targets: ['postgres-exporter:9187']

  # Redis monitoring
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

  # Node exporter (system metrics)
  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']

  # ML model monitoring
  - job_name: 'ml-models'
    static_configs:
      - targets: ['ml-service:9092']
    metrics_path: '/ml-metrics'

  # Load balancer monitoring
  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx-exporter:9113']

  # Kubernetes metrics (if applicable)
  - job_name: 'kubernetes-apiservers'
    kubernetes_sd_configs:
      - role: endpoints
    scheme: https
    tls_config:
      ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
    bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
    relabel_configs:
      - source_labels: [__meta_kubernetes_namespace, __meta_kubernetes_service_name, __meta_kubernetes_endpoint_port_name]
        action: keep
        regex: default;kubernetes;https
```

### 2. Alert Rules Configuration

Create `prometheus/alerts/rental-ml-alerts.yml`:

```yaml
groups:
- name: rental_ml_performance_alerts
  rules:
  # API Performance Alerts
  - alert: HighAPIResponseTime
    expr: histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le, instance)) > 2.0
    for: 2m
    labels:
      severity: warning
      component: api
    annotations:
      summary: "High API response time detected"
      description: "95th percentile response time is {{ $value }}s on instance {{ $labels.instance }}"
      runbook_url: "https://docs.rental-ml.com/runbooks/high-response-time"

  - alert: HighAPIErrorRate
    expr: sum(rate(http_requests_total{status=~"5.."}[5m])) by (instance) / sum(rate(http_requests_total[5m])) by (instance) > 0.05
    for: 1m
    labels:
      severity: critical
      component: api
    annotations:
      summary: "High API error rate detected"
      description: "Error rate is {{ $value | humanizePercentage }} on instance {{ $labels.instance }}"
      runbook_url: "https://docs.rental-ml.com/runbooks/high-error-rate"

  - alert: LowAPIThroughput
    expr: sum(rate(http_requests_total[5m])) by (instance) < 10
    for: 5m
    labels:
      severity: warning
      component: api
    annotations:
      summary: "Low API throughput detected"
      description: "Request rate is {{ $value }} RPS on instance {{ $labels.instance }}"

  # Database Performance Alerts
  - alert: SlowDatabaseQueries
    expr: pg_stat_database_tup_fetched / pg_stat_database_tup_returned > 100
    for: 5m
    labels:
      severity: warning
      component: database
    annotations:
      summary: "Inefficient database queries detected"
      description: "Database is examining {{ $value }} rows per returned row"

  - alert: HighDatabaseConnections
    expr: pg_stat_activity_count > 80
    for: 2m
    labels:
      severity: warning
      component: database
    annotations:
      summary: "High database connection count"
      description: "{{ $value }} active database connections"

  - alert: DatabaseConnectionPoolExhaustion
    expr: database_connection_pool_active >= database_connection_pool_size * 0.9
    for: 1m
    labels:
      severity: critical
      component: database
    annotations:
      summary: "Database connection pool near exhaustion"
      description: "{{ $value }} active connections out of {{ database_connection_pool_size }} total"

  # Cache Performance Alerts
  - alert: LowCacheHitRate
    expr: redis_keyspace_hits_total / (redis_keyspace_hits_total + redis_keyspace_misses_total) < 0.7
    for: 10m
    labels:
      severity: warning
      component: cache
    annotations:
      summary: "Low cache hit rate"
      description: "Cache hit rate is {{ $value | humanizePercentage }}"

  - alert: HighRedisMemoryUsage
    expr: redis_memory_used_bytes / redis_memory_max_bytes > 0.85
    for: 5m
    labels:
      severity: warning
      component: cache
    annotations:
      summary: "High Redis memory usage"
      description: "Redis memory usage is {{ $value | humanizePercentage }}"

  # ML Model Performance Alerts
  - alert: SlowMLInference
    expr: histogram_quantile(0.95, sum(rate(ml_inference_duration_seconds_bucket[5m])) by (le, model_name)) > 1.0
    for: 3m
    labels:
      severity: warning
      component: ml
    annotations:
      summary: "Slow ML inference detected"
      description: "95th percentile inference time for {{ $labels.model_name }} is {{ $value }}s"

  - alert: HighMLErrorRate
    expr: sum(rate(ml_inference_errors_total[5m])) by (model_name) / sum(rate(ml_inference_total[5m])) by (model_name) > 0.02
    for: 2m
    labels:
      severity: critical
      component: ml
    annotations:
      summary: "High ML inference error rate"
      description: "ML error rate for {{ $labels.model_name }} is {{ $value | humanizePercentage }}"

  # System Resource Alerts
  - alert: HighCPUUsage
    expr: 100 - (avg by(instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 85
    for: 5m
    labels:
      severity: warning
      component: system
    annotations:
      summary: "High CPU usage detected"
      description: "CPU usage is {{ $value }}% on instance {{ $labels.instance }}"

  - alert: HighMemoryUsage
    expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes > 0.9
    for: 3m
    labels:
      severity: critical
      component: system
    annotations:
      summary: "High memory usage detected"
      description: "Memory usage is {{ $value | humanizePercentage }} on instance {{ $labels.instance }}"

  - alert: HighDiskUsage
    expr: (node_filesystem_size_bytes - node_filesystem_avail_bytes) / node_filesystem_size_bytes > 0.85
    for: 5m
    labels:
      severity: warning
      component: system
    annotations:
      summary: "High disk usage detected"
      description: "Disk usage is {{ $value | humanizePercentage }} on instance {{ $labels.instance }}"

  # Application-Specific Alerts
  - alert: HighRecommendationLatency
    expr: histogram_quantile(0.95, sum(rate(recommendation_generation_duration_seconds_bucket[5m])) by (le)) > 5.0
    for: 3m
    labels:
      severity: warning
      component: ml
    annotations:
      summary: "High recommendation generation latency"
      description: "95th percentile recommendation latency is {{ $value }}s"

  - alert: SearchIndexingLag
    expr: search_index_lag_seconds > 300
    for: 5m
    labels:
      severity: warning
      component: search
    annotations:
      summary: "Search index lagging behind"
      description: "Search index is {{ $value }}s behind real-time data"
```

### 3. Docker Compose for Monitoring Stack

Create `docker-compose.monitoring.yml`:

```yaml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:v2.45.0
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus:/etc/prometheus
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=90d'
      - '--web.enable-lifecycle'
      - '--web.enable-admin-api'
    networks:
      - monitoring

  grafana:
    image: grafana/grafana:10.0.0
    container_name: grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
      - GF_USERS_ALLOW_SIGN_UP=false
      - GF_INSTALL_PLUGINS=redis-datasource,postgres-datasource
    networks:
      - monitoring

  alertmanager:
    image: prom/alertmanager:v0.25.0
    container_name: alertmanager
    ports:
      - "9093:9093"
    volumes:
      - ./alertmanager:/etc/alertmanager
    command:
      - '--config.file=/etc/alertmanager/alertmanager.yml'
      - '--storage.path=/alertmanager'
    networks:
      - monitoring

  node-exporter:
    image: prom/node-exporter:v1.6.0
    container_name: node-exporter
    ports:
      - "9100:9100"
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.ignored-mount-points=^/(sys|proc|dev|host|etc)($$|/)'
    networks:
      - monitoring

  postgres-exporter:
    image: prometheuscommunity/postgres-exporter:v0.13.0
    container_name: postgres-exporter
    ports:
      - "9187:9187"
    environment:
      - DATA_SOURCE_NAME=postgresql://username:password@postgres:5432/rental_ml?sslmode=disable
    networks:
      - monitoring

  redis-exporter:
    image: oliver006/redis_exporter:v1.52.0
    container_name: redis-exporter
    ports:
      - "9121:9121"
    environment:
      - REDIS_ADDR=redis://redis:6379
    networks:
      - monitoring

networks:
  monitoring:
    driver: bridge

volumes:
  prometheus_data:
  grafana_data:
```

## Grafana Dashboard Setup

### 1. Datasource Configuration

Create `grafana/datasources/prometheus.yml`:

```yaml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
```

### 2. Dashboard Configuration

Create `grafana/dashboards/dashboard.yml`:

```yaml
apiVersion: 1

providers:
  - name: 'rental-ml-dashboards'
    orgId: 1
    folder: 'Rental ML System'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /etc/grafana/provisioning/dashboards
```

### 3. Main Performance Dashboard

Create `grafana/dashboards/rental-ml-performance.json`:

```json
{
  "dashboard": {
    "id": null,
    "title": "Rental ML System - Performance Overview",
    "tags": ["rental-ml", "performance"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "API Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))",
            "legendFormat": "P50"
          },
          {
            "expr": "histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))",
            "legendFormat": "P95"
          },
          {
            "expr": "histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))",
            "legendFormat": "P99"
          }
        ],
        "yAxes": [
          {
            "unit": "s",
            "min": 0
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 0,
          "y": 0
        }
      },
      {
        "id": 2,
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total[5m]))",
            "legendFormat": "Total RPS"
          },
          {
            "expr": "sum(rate(http_requests_total{status=~\"2..\"}[5m]))",
            "legendFormat": "Success RPS"
          },
          {
            "expr": "sum(rate(http_requests_total{status=~\"5..\"}[5m]))",
            "legendFormat": "Error RPS"
          }
        ],
        "yAxes": [
          {
            "unit": "reqps",
            "min": 0
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 12,
          "y": 0
        }
      },
      {
        "id": 3,
        "title": "System Resources",
        "type": "graph",
        "targets": [
          {
            "expr": "100 - (avg(irate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)",
            "legendFormat": "CPU Usage %"
          },
          {
            "expr": "(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes * 100",
            "legendFormat": "Memory Usage %"
          }
        ],
        "yAxes": [
          {
            "unit": "percent",
            "min": 0,
            "max": 100
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 0,
          "y": 8
        }
      },
      {
        "id": 4,
        "title": "Database Performance",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(pg_stat_database_tup_fetched[5m])",
            "legendFormat": "Rows Fetched/sec"
          },
          {
            "expr": "pg_stat_activity_count",
            "legendFormat": "Active Connections"
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 12,
          "y": 8
        }
      },
      {
        "id": 5,
        "title": "Cache Hit Rate",
        "type": "singlestat",
        "targets": [
          {
            "expr": "redis_keyspace_hits_total / (redis_keyspace_hits_total + redis_keyspace_misses_total) * 100"
          }
        ],
        "format": "percent",
        "gridPos": {
          "h": 4,
          "w": 6,
          "x": 0,
          "y": 16
        }
      },
      {
        "id": 6,
        "title": "ML Inference Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, sum(rate(ml_inference_duration_seconds_bucket[5m])) by (le, model_name))",
            "legendFormat": "{{model_name}} P95"
          }
        ],
        "yAxes": [
          {
            "unit": "s",
            "min": 0
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 6,
          "y": 16
        }
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "30s"
  }
}
```

## AlertManager Configuration

Create `alertmanager/alertmanager.yml`:

```yaml
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@rental-ml.com'
  smtp_auth_username: 'alerts@rental-ml.com'
  smtp_auth_password: 'password'

route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'default-receiver'
  routes:
  - match:
      severity: critical
    receiver: 'critical-alerts'
    group_wait: 10s
    repeat_interval: 5m
  - match:
      component: api
    receiver: 'api-alerts'
  - match:
      component: database
    receiver: 'database-alerts'
  - match:
      component: ml
    receiver: 'ml-alerts'

receivers:
- name: 'default-receiver'
  email_configs:
  - to: 'team@rental-ml.com'
    subject: '[{{ .Status | toUpper }}] {{ .GroupLabels.alertname }}'
    body: |
      {{ range .Alerts }}
      Alert: {{ .Annotations.summary }}
      Description: {{ .Annotations.description }}
      Instance: {{ .Labels.instance }}
      Severity: {{ .Labels.severity }}
      {{ end }}

- name: 'critical-alerts'
  slack_configs:
  - api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
    channel: '#critical-alerts'
    title: 'CRITICAL: {{ .GroupLabels.alertname }}'
    text: |
      {{ range .Alerts }}
      *Alert:* {{ .Annotations.summary }}
      *Description:* {{ .Annotations.description }}
      *Instance:* {{ .Labels.instance }}
      *Runbook:* {{ .Annotations.runbook_url }}
      {{ end }}
  email_configs:
  - to: 'oncall@rental-ml.com'
    subject: 'CRITICAL: {{ .GroupLabels.alertname }}'

- name: 'api-alerts'
  slack_configs:
  - api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
    channel: '#api-alerts'
    title: 'API Alert: {{ .GroupLabels.alertname }}'

- name: 'database-alerts'
  slack_configs:
  - api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
    channel: '#database-alerts'
    title: 'Database Alert: {{ .GroupLabels.alertname }}'

- name: 'ml-alerts'
  slack_configs:
  - api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
    channel: '#ml-alerts'
    title: 'ML Alert: {{ .GroupLabels.alertname }}'

inhibit_rules:
- source_match:
    severity: 'critical'
  target_match:
    severity: 'warning'
  equal: ['alertname', 'cluster', 'service']
```

## Application Metrics Integration

### 1. FastAPI Metrics Middleware

Create `metrics_middleware.py`:

```python
import time
import asyncio
from typing import Callable
from fastapi import Request, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import psutil

# Prometheus metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total number of HTTP requests',
    ['method', 'endpoint', 'status_code']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint']
)

ACTIVE_REQUESTS = Gauge(
    'http_requests_active',
    'Number of active HTTP requests'
)

DATABASE_CONNECTIONS = Gauge(
    'database_connection_pool_active',
    'Number of active database connections'
)

CACHE_OPERATIONS = Counter(
    'cache_operations_total',
    'Total cache operations',
    ['operation', 'result']
)

ML_INFERENCE_DURATION = Histogram(
    'ml_inference_duration_seconds',
    'ML inference duration in seconds',
    ['model_name']
)

ML_INFERENCE_TOTAL = Counter(
    'ml_inference_total',
    'Total ML inferences',
    ['model_name']
)

ML_INFERENCE_ERRORS = Counter(
    'ml_inference_errors_total',
    'Total ML inference errors',
    ['model_name', 'error_type']
)

SYSTEM_MEMORY_USAGE = Gauge(
    'system_memory_usage_bytes',
    'System memory usage in bytes'
)

SYSTEM_CPU_USAGE = Gauge(
    'system_cpu_usage_percent',
    'System CPU usage percentage'
)

class PrometheusMiddleware:
    def __init__(self):
        # Start background task for system metrics
        asyncio.create_task(self._update_system_metrics())
    
    async def __call__(self, request: Request, call_next: Callable) -> Response:
        # Increment active requests
        ACTIVE_REQUESTS.inc()
        
        # Extract request info
        method = request.method
        endpoint = request.url.path
        
        # Start timing
        start_time = time.time()
        
        try:
            # Process request
            response = await call_next(request)
            status_code = response.status_code
            
            # Record metrics
            REQUEST_COUNT.labels(
                method=method,
                endpoint=endpoint,
                status_code=status_code
            ).inc()
            
            REQUEST_DURATION.labels(
                method=method,
                endpoint=endpoint
            ).observe(time.time() - start_time)
            
            return response
            
        except Exception as e:
            # Record error
            REQUEST_COUNT.labels(
                method=method,
                endpoint=endpoint,
                status_code=500
            ).inc()
            
            REQUEST_DURATION.labels(
                method=method,
                endpoint=endpoint
            ).observe(time.time() - start_time)
            
            raise
        finally:
            # Decrement active requests
            ACTIVE_REQUESTS.dec()
    
    async def _update_system_metrics(self):
        """Update system metrics periodically"""
        while True:
            try:
                # Memory usage
                memory = psutil.virtual_memory()
                SYSTEM_MEMORY_USAGE.set(memory.used)
                
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                SYSTEM_CPU_USAGE.set(cpu_percent)
                
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                print(f"Error updating system metrics: {e}")
                await asyncio.sleep(10)

# FastAPI integration
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

app = FastAPI()

# Add metrics middleware
app.add_middleware(PrometheusMiddleware)

@app.get("/metrics")
async def get_metrics():
    """Endpoint for Prometheus to scrape metrics"""
    return PlainTextResponse(
        generate_latest(),
        media_type="text/plain"
    )

# Database metrics helper
async def record_database_metrics(db_pool):
    """Record database connection pool metrics"""
    DATABASE_CONNECTIONS.set(db_pool.get_size())

# Cache metrics helper
def record_cache_operation(operation: str, result: str):
    """Record cache operation metrics"""
    CACHE_OPERATIONS.labels(operation=operation, result=result).inc()

# ML metrics helper
async def record_ml_inference(model_name: str, inference_func, *args, **kwargs):
    """Record ML inference metrics"""
    start_time = time.time()
    
    try:
        result = await inference_func(*args, **kwargs)
        
        # Record success
        ML_INFERENCE_TOTAL.labels(model_name=model_name).inc()
        ML_INFERENCE_DURATION.labels(model_name=model_name).observe(
            time.time() - start_time
        )
        
        return result
        
    except Exception as e:
        # Record error
        ML_INFERENCE_ERRORS.labels(
            model_name=model_name,
            error_type=type(e).__name__
        ).inc()
        
        raise

# Usage examples
@app.get("/api/v1/recommendations/{user_id}")
async def get_recommendations(user_id: str):
    # Record ML inference with metrics
    recommendations = await record_ml_inference(
        "hybrid_recommender",
        recommendation_service.get_recommendations,
        user_id
    )
    
    return recommendations

@app.get("/api/v1/search")
async def search_properties(query: str):
    # Record cache operation
    cache_key = f"search:{query}"
    cached_result = await cache.get(cache_key)
    
    if cached_result:
        record_cache_operation("get", "hit")
        return cached_result
    else:
        record_cache_operation("get", "miss")
        
        # Perform search
        results = await search_service.search(query)
        
        # Cache results
        await cache.set(cache_key, results)
        record_cache_operation("set", "success")
        
        return results
```

### 2. Custom Metrics for Business Logic

```python
# business_metrics.py
from prometheus_client import Counter, Histogram, Gauge

# Business metrics
USER_REGISTRATIONS = Counter(
    'user_registrations_total',
    'Total user registrations'
)

PROPERTY_VIEWS = Counter(
    'property_views_total',
    'Total property views',
    ['property_type', 'location']
)

SEARCH_QUERIES = Counter(
    'search_queries_total',
    'Total search queries',
    ['query_type']
)

RECOMMENDATION_CLICKS = Counter(
    'recommendation_clicks_total',
    'Total recommendation clicks',
    ['recommendation_type']
)

RECOMMENDATION_GENERATION_TIME = Histogram(
    'recommendation_generation_duration_seconds',
    'Time taken to generate recommendations',
    ['user_type']
)

ACTIVE_USERS = Gauge(
    'active_users_current',
    'Current number of active users'
)

SEARCH_INDEX_LAG = Gauge(
    'search_index_lag_seconds',
    'Search index lag in seconds'
)

# Usage in business logic
class RecommendationService:
    async def get_recommendations(self, user_id: str):
        start_time = time.time()
        
        try:
            # Determine user type
            user_type = await self.get_user_type(user_id)
            
            # Generate recommendations
            recommendations = await self.generate_recommendations(user_id)
            
            # Record metrics
            RECOMMENDATION_GENERATION_TIME.labels(
                user_type=user_type
            ).observe(time.time() - start_time)
            
            return recommendations
            
        except Exception as e:
            # Handle errors
            raise

    async def record_recommendation_click(self, user_id: str, property_id: str, rec_type: str):
        """Record when user clicks on a recommendation"""
        RECOMMENDATION_CLICKS.labels(recommendation_type=rec_type).inc()

class SearchService:
    async def search(self, query: str, query_type: str = "text"):
        """Search properties and record metrics"""
        SEARCH_QUERIES.labels(query_type=query_type).inc()
        
        # Perform search
        results = await self.perform_search(query)
        
        return results

class PropertyService:
    async def record_property_view(self, property_id: str):
        """Record property view"""
        property_info = await self.get_property(property_id)
        
        PROPERTY_VIEWS.labels(
            property_type=property_info.get('type', 'unknown'),
            location=property_info.get('location', 'unknown')
        ).inc()
```

## Log Aggregation Setup

### 1. ELK Stack Configuration

Create `docker-compose.elk.yml`:

```yaml
version: '3.8'

services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.8.0
    container_name: elasticsearch
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    ports:
      - "9200:9200"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data
    networks:
      - monitoring

  logstash:
    image: docker.elastic.co/logstash/logstash:8.8.0
    container_name: logstash
    ports:
      - "5044:5044"
      - "5000:5000/tcp"
      - "5000:5000/udp"
      - "9600:9600"
    volumes:
      - ./logstash/pipeline:/usr/share/logstash/pipeline
      - ./logstash/config:/usr/share/logstash/config
    depends_on:
      - elasticsearch
    networks:
      - monitoring

  kibana:
    image: docker.elastic.co/kibana/kibana:8.8.0
    container_name: kibana
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    depends_on:
      - elasticsearch
    networks:
      - monitoring

volumes:
  elasticsearch_data:

networks:
  monitoring:
    external: true
```

### 2. Logstash Configuration

Create `logstash/config/logstash.yml`:

```yaml
http.host: "0.0.0.0"
xpack.monitoring.elasticsearch.hosts: [ "http://elasticsearch:9200" ]
```

Create `logstash/pipeline/logstash.conf`:

```ruby
input {
  beats {
    port => 5044
  }
  
  tcp {
    port => 5000
    codec => json_lines
  }
}

filter {
  if [fields][service] == "rental-ml-api" {
    mutate {
      add_tag => [ "api" ]
    }
    
    # Parse JSON log message
    json {
      source => "message"
    }
    
    # Extract response time from log
    if [response_time] {
      mutate {
        convert => { "response_time" => "float" }
      }
    }
    
    # Categorize log levels
    if [level] == "ERROR" {
      mutate {
        add_tag => [ "error" ]
      }
    }
    
    # Extract slow query information
    if [message] =~ /slow query/ {
      mutate {
        add_tag => [ "slow_query" ]
      }
      
      grok {
        match => { "message" => "slow query detected: %{NUMBER:query_time:float}s - %{GREEDYDATA:query_text}" }
      }
    }
  }
  
  # Add timestamp
  date {
    match => [ "timestamp", "ISO8601" ]
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "rental-ml-logs-%{+YYYY.MM.dd}"
  }
  
  # Output errors to separate index
  if "error" in [tags] {
    elasticsearch {
      hosts => ["elasticsearch:9200"]
      index => "rental-ml-errors-%{+YYYY.MM.dd}"
    }
  }
  
  # Output slow queries to separate index
  if "slow_query" in [tags] {
    elasticsearch {
      hosts => ["elasticsearch:9200"]
      index => "rental-ml-slow-queries-%{+YYYY.MM.dd}"
    }
  }
}
```

### 3. Application Logging Configuration

```python
# logging_config.py
import logging
import json
import time
from datetime import datetime
from typing import Dict, Any

class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        
        if hasattr(record, 'response_time'):
            log_entry['response_time'] = record.response_time
        
        if hasattr(record, 'status_code'):
            log_entry['status_code'] = record.status_code
        
        # Add exception info
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)

# Configure logging
def setup_logging():
    """Setup application logging"""
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove default handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create JSON formatter
    formatter = JSONFormatter()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler('/var/log/rental-ml/api.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

# Performance logging middleware
class PerformanceLoggingMiddleware:
    def __init__(self):
        self.logger = logging.getLogger('performance')
    
    async def __call__(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        # Add request ID to context
        request.state.request_id = request_id
        
        try:
            response = await call_next(request)
            
            # Log successful request
            response_time = time.time() - start_time
            
            self.logger.info(
                f"Request processed",
                extra={
                    'request_id': request_id,
                    'method': request.method,
                    'url': str(request.url),
                    'status_code': response.status_code,
                    'response_time': response_time,
                    'user_agent': request.headers.get('user-agent', ''),
                    'remote_addr': request.client.host
                }
            )
            
            # Log slow requests
            if response_time > 1.0:  # Slow request threshold
                self.logger.warning(
                    f"Slow request detected: {response_time:.3f}s",
                    extra={
                        'request_id': request_id,
                        'response_time': response_time,
                        'url': str(request.url)
                    }
                )
            
            return response
            
        except Exception as e:
            # Log error
            response_time = time.time() - start_time
            
            self.logger.error(
                f"Request failed: {str(e)}",
                extra={
                    'request_id': request_id,
                    'method': request.method,
                    'url': str(request.url),
                    'response_time': response_time,
                    'error_type': type(e).__name__
                },
                exc_info=True
            )
            
            raise

# Database query logging
class DatabaseQueryLogger:
    def __init__(self):
        self.logger = logging.getLogger('database')
    
    async def log_query(self, query: str, params: list, duration: float):
        """Log database query with performance info"""
        
        log_extra = {
            'query': query[:500],  # Truncate long queries
            'params_count': len(params) if params else 0,
            'duration': duration
        }
        
        if duration > 0.1:  # Slow query threshold
            self.logger.warning(
                f"Slow query detected: {duration:.3f}s",
                extra=log_extra
            )
        else:
            self.logger.debug(
                f"Query executed in {duration:.3f}s",
                extra=log_extra
            )
```

## Performance Testing Integration

### 1. Automated Performance Tests

Create `scripts/performance_test.py`:

```python
#!/usr/bin/env python3
"""
Automated performance testing script for CI/CD integration
"""

import asyncio
import json
import sys
import time
from typing import Dict, List
import aiohttp
import argparse
from datetime import datetime

class PerformanceTest:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.results = []
    
    async def run_load_test(self, endpoint: str, 
                           concurrent_users: int = 10,
                           duration_seconds: int = 60,
                           expected_avg_response_time: float = 1.0,
                           expected_p95_response_time: float = 2.0,
                           expected_error_rate: float = 0.01) -> Dict:
        """Run load test against endpoint"""
        
        print(f"Running load test: {endpoint}")
        print(f"Users: {concurrent_users}, Duration: {duration_seconds}s")
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        response_times = []
        error_count = 0
        success_count = 0
        
        async def user_session(session: aiohttp.ClientSession):
            while time.time() < end_time:
                request_start = time.time()
                
                try:
                    async with session.get(f"{self.base_url}{endpoint}") as response:
                        await response.text()
                        request_time = time.time() - request_start
                        response_times.append(request_time)
                        
                        if 200 <= response.status < 400:
                            success_count += 1
                        else:
                            error_count += 1
                            
                except Exception:
                    error_count += 1
                    response_times.append(time.time() - request_start)
                
                await asyncio.sleep(0.1)  # Small delay between requests
        
        # Run concurrent user sessions
        connector = aiohttp.TCPConnector(limit=concurrent_users * 2)
        timeout = aiohttp.ClientTimeout(total=30)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = [user_session(session) for _ in range(concurrent_users)]
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Calculate results
        total_requests = len(response_times)
        avg_response_time = sum(response_times) / max(total_requests, 1)
        p95_response_time = sorted(response_times)[int(0.95 * total_requests)] if response_times else 0
        error_rate = error_count / max(total_requests, 1)
        throughput = total_requests / duration_seconds
        
        result = {
            'endpoint': endpoint,
            'total_requests': total_requests,
            'avg_response_time': avg_response_time,
            'p95_response_time': p95_response_time,
            'error_rate': error_rate,
            'throughput_rps': throughput,
            'test_passed': True
        }
        
        # Check performance criteria
        failures = []
        
        if avg_response_time > expected_avg_response_time:
            failures.append(f"Average response time {avg_response_time:.3f}s > {expected_avg_response_time:.3f}s")
        
        if p95_response_time > expected_p95_response_time:
            failures.append(f"P95 response time {p95_response_time:.3f}s > {expected_p95_response_time:.3f}s")
        
        if error_rate > expected_error_rate:
            failures.append(f"Error rate {error_rate:.3f} > {expected_error_rate:.3f}")
        
        if failures:
            result['test_passed'] = False
            result['failures'] = failures
            print(f"❌ Performance test FAILED for {endpoint}:")
            for failure in failures:
                print(f"   - {failure}")
        else:
            print(f"✅ Performance test PASSED for {endpoint}")
        
        print(f"   Avg: {avg_response_time:.3f}s, P95: {p95_response_time:.3f}s, "
              f"Error Rate: {error_rate:.1%}, Throughput: {throughput:.1f} RPS")
        
        self.results.append(result)
        return result

async def main():
    parser = argparse.ArgumentParser(description='Run performance tests')
    parser.add_argument('--base-url', default='http://localhost:8000', 
                       help='Base URL for testing')
    parser.add_argument('--duration', type=int, default=60,
                       help='Test duration in seconds')
    parser.add_argument('--users', type=int, default=10,
                       help='Number of concurrent users')
    parser.add_argument('--output', default='performance_results.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # Test configuration
    test_config = [
        {
            'endpoint': '/health',
            'expected_avg_response_time': 0.1,
            'expected_p95_response_time': 0.2,
            'expected_error_rate': 0.001
        },
        {
            'endpoint': '/api/v1/search?query=apartment',
            'expected_avg_response_time': 0.5,
            'expected_p95_response_time': 1.0,
            'expected_error_rate': 0.01
        },
        {
            'endpoint': '/api/v1/recommendations/user/1',
            'expected_avg_response_time': 1.0,
            'expected_p95_response_time': 2.0,
            'expected_error_rate': 0.02
        }
    ]
    
    # Run tests
    tester = PerformanceTest(args.base_url)
    
    print(f"Starting performance tests against {args.base_url}")
    print(f"Test duration: {args.duration}s, Concurrent users: {args.users}")
    print("=" * 60)
    
    all_passed = True
    
    for test in test_config:
        result = await tester.run_load_test(
            endpoint=test['endpoint'],
            concurrent_users=args.users,
            duration_seconds=args.duration,
            expected_avg_response_time=test['expected_avg_response_time'],
            expected_p95_response_time=test['expected_p95_response_time'],
            expected_error_rate=test['expected_error_rate']
        )
        
        if not result['test_passed']:
            all_passed = False
        
        print()  # Empty line between tests
    
    # Save results
    summary = {
        'timestamp': datetime.now().isoformat(),
        'test_config': {
            'base_url': args.base_url,
            'duration': args.duration,
            'concurrent_users': args.users
        },
        'results': tester.results,
        'overall_passed': all_passed
    }
    
    with open(args.output, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("=" * 60)
    print(f"Performance test summary:")
    print(f"Total tests: {len(tester.results)}")
    print(f"Passed: {sum(1 for r in tester.results if r['test_passed'])}")
    print(f"Failed: {sum(1 for r in tester.results if not r['test_passed'])}")
    print(f"Results saved to: {args.output}")
    
    if not all_passed:
        print("\n❌ Some performance tests failed!")
        sys.exit(1)
    else:
        print("\n✅ All performance tests passed!")
        sys.exit(0)

if __name__ == '__main__':
    asyncio.run(main())
```

### 2. CI/CD Integration

Create `.github/workflows/performance-tests.yml`:

```yaml
name: Performance Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  performance-tests:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:14
        env:
          POSTGRES_PASSWORD: password
          POSTGRES_DB: rental_ml_test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements/base.txt
        pip install -r requirements/dev.txt
    
    - name: Start application
      run: |
        # Start the application in background
        python -m uvicorn src.application.api.main:app --host 0.0.0.0 --port 8000 &
        
        # Wait for application to start
        sleep 10
        
        # Verify application is running
        curl -f http://localhost:8000/health || exit 1
      env:
        DATABASE_URL: postgresql://postgres:password@localhost:5432/rental_ml_test
        REDIS_URL: redis://localhost:6379
    
    - name: Run performance tests
      run: |
        python scripts/performance_test.py \
          --base-url http://localhost:8000 \
          --duration 60 \
          --users 5 \
          --output performance_results.json
    
    - name: Upload performance results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: performance-results
        path: performance_results.json
    
    - name: Comment performance results
      if: github.event_name == 'pull_request'
      uses: actions/github-script@v6
      with:
        script: |
          const fs = require('fs');
          const results = JSON.parse(fs.readFileSync('performance_results.json', 'utf8'));
          
          let comment = '## Performance Test Results\n\n';
          comment += `**Overall Status:** ${results.overall_passed ? '✅ PASSED' : '❌ FAILED'}\n\n`;
          comment += '| Endpoint | Avg Response Time | P95 Response Time | Error Rate | Throughput |\n';
          comment += '|----------|-------------------|-------------------|------------|------------|\n';
          
          for (const result of results.results) {
            const status = result.test_passed ? '✅' : '❌';
            comment += `| ${status} ${result.endpoint} | ${result.avg_response_time.toFixed(3)}s | ${result.p95_response_time.toFixed(3)}s | ${(result.error_rate * 100).toFixed(2)}% | ${result.throughput_rps.toFixed(1)} RPS |\n`;
          }
          
          if (!results.overall_passed) {
            comment += '\n### Failures:\n';
            for (const result of results.results) {
              if (!result.test_passed && result.failures) {
                comment += `\n**${result.endpoint}:**\n`;
                for (const failure of result.failures) {
                  comment += `- ${failure}\n`;
                }
              }
            }
          }
          
          github.rest.issues.createComment({
            issue_number: context.issue.number,
            owner: context.repo.owner,
            repo: context.repo.repo,
            body: comment
          });
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Prometheus Not Scraping Metrics

**Symptoms:**
- No data in Grafana dashboards
- Prometheus targets showing as "down"

**Solutions:**
```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Check if metrics endpoint is accessible
curl http://localhost:8000/metrics

# Verify network connectivity
docker exec prometheus ping api-container

# Check Prometheus logs
docker logs prometheus
```

#### 2. High Memory Usage in Prometheus

**Symptoms:**
- Prometheus container using excessive memory
- Out of memory errors

**Solutions:**
```yaml
# Add memory limits to prometheus config
command:
  - '--storage.tsdb.retention.time=90d'
  - '--storage.tsdb.retention.size=10GB'
  - '--query.max-samples=50000000'
  - '--storage.tsdb.max-block-duration=2h'
```

#### 3. Grafana Dashboard Not Loading

**Symptoms:**
- Dashboard shows "No data"
- Panel errors

**Solutions:**
```bash
# Check Grafana logs
docker logs grafana

# Verify datasource connection
curl http://admin:admin123@localhost:3000/api/datasources

# Test Prometheus queries manually
curl 'http://localhost:9090/api/v1/query?query=up'
```

#### 4. AlertManager Not Sending Notifications

**Symptoms:**
- Alerts firing but no notifications received
- AlertManager web UI shows alerts but no notifications

**Solutions:**
```bash
# Check AlertManager configuration
curl http://localhost:9093/api/v1/status

# Test SMTP configuration
docker exec alertmanager amtool config check

# Check Slack webhook
curl -X POST -H 'Content-type: application/json' \
  --data '{"text":"Test message"}' \
  YOUR_SLACK_WEBHOOK_URL
```

### Performance Monitoring Best Practices

1. **Set Appropriate Retention Periods**
   - Prometheus: 90 days for detailed metrics
   - Logs: 30 days for application logs, 7 days for debug logs

2. **Use Appropriate Scrape Intervals**
   - System metrics: 15-30 seconds
   - Application metrics: 10-15 seconds
   - Business metrics: 30-60 seconds

3. **Implement Proper Alerting**
   - Set up escalation policies
   - Use severity levels appropriately
   - Include runbook links in alerts

4. **Regular Maintenance**
   - Clean up old metrics and logs
   - Update dashboard queries for performance
   - Review and update alert thresholds

5. **Security Considerations**
   - Use authentication for Grafana
   - Secure Prometheus endpoints
   - Encrypt sensitive data in alerts

This monitoring setup provides comprehensive visibility into the rental ML system's performance and enables proactive identification and resolution of performance issues.