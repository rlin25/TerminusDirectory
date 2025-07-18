#!/usr/bin/env python3
"""
Comprehensive Health Check Script for Rental ML System

This script performs extensive health checks across all system components:
- Application health endpoints
- Database connectivity and performance
- Redis cache functionality
- Service mesh health
- External dependencies
- Performance metrics validation
"""

import argparse
import json
import logging
import time
import sys
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import ssl
import socket

import requests
import psycopg2
import redis
from kubernetes import client, config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class HealthCheckResult:
    """Result of a health check"""
    name: str
    status: str  # 'healthy', 'warning', 'critical'
    message: str
    response_time: float
    details: Dict = None
    
    def is_healthy(self) -> bool:
        return self.status == 'healthy'

@dataclass
class HealthCheckConfig:
    """Configuration for health checks"""
    environment: str
    namespace: str
    timeout: int = 300
    parallel_checks: bool = True
    
    # URLs and endpoints
    api_url: str = None
    prometheus_url: str = None
    grafana_url: str = None
    jaeger_url: str = None
    
    # Database connection
    database_url: str = None
    
    # Redis connection
    redis_url: str = None
    
    # Expected response times (in seconds)
    api_response_time_threshold: float = 2.0
    database_response_time_threshold: float = 1.0
    redis_response_time_threshold: float = 0.5

class HealthChecker:
    """Comprehensive health checker for the rental ML system"""
    
    def __init__(self, config: HealthCheckConfig):
        self.config = config
        self.results: List[HealthCheckResult] = []
        
        # Initialize Kubernetes client
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()
        
        self.k8s_core_v1 = client.CoreV1Api()
        self.k8s_apps_v1 = client.AppsV1Api()
    
    def run_all_checks(self) -> bool:
        """Run all health checks and return overall health status"""
        logger.info(f"Starting comprehensive health checks for {self.config.environment} environment")
        
        # Define all health checks
        health_checks = [
            ("API Health", self._check_api_health),
            ("API Endpoints", self._check_api_endpoints),
            ("Database Health", self._check_database_health),
            ("Redis Health", self._check_redis_health),
            ("Kubernetes Resources", self._check_kubernetes_resources),
            ("Service Mesh", self._check_service_mesh),
            ("Monitoring Stack", self._check_monitoring_stack),
            ("External Dependencies", self._check_external_dependencies),
            ("Performance Metrics", self._check_performance_metrics),
            ("Security Checks", self._check_security),
            ("Backup Systems", self._check_backup_systems)
        ]
        
        if self.config.parallel_checks:
            self._run_checks_parallel(health_checks)
        else:
            self._run_checks_sequential(health_checks)
        
        # Generate report
        self._generate_health_report()
        
        # Return overall health status
        return self._get_overall_health_status()
    
    def _run_checks_parallel(self, health_checks: List[Tuple[str, callable]]):
        """Run health checks in parallel"""
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_check = {
                executor.submit(check_func): check_name 
                for check_name, check_func in health_checks
            }
            
            for future in as_completed(future_to_check, timeout=self.config.timeout):
                check_name = future_to_check[future]
                try:
                    result = future.result()
                    if isinstance(result, list):
                        self.results.extend(result)
                    else:
                        self.results.append(result)
                except Exception as e:
                    logger.error(f"Health check '{check_name}' failed with exception: {e}")
                    self.results.append(HealthCheckResult(
                        name=check_name,
                        status='critical',
                        message=f"Check failed with exception: {str(e)}",
                        response_time=0.0
                    ))
    
    def _run_checks_sequential(self, health_checks: List[Tuple[str, callable]]):
        """Run health checks sequentially"""
        for check_name, check_func in health_checks:
            try:
                result = check_func()
                if isinstance(result, list):
                    self.results.extend(result)
                else:
                    self.results.append(result)
            except Exception as e:
                logger.error(f"Health check '{check_name}' failed with exception: {e}")
                self.results.append(HealthCheckResult(
                    name=check_name,
                    status='critical',
                    message=f"Check failed with exception: {str(e)}",
                    response_time=0.0
                ))
    
    def _check_api_health(self) -> HealthCheckResult:
        """Check API health endpoint"""
        start_time = time.time()
        
        try:
            api_url = self.config.api_url or f"https://api.{self.config.environment}.rental-ml.com"
            response = requests.get(
                f"{api_url}/health",
                timeout=10,
                headers={'User-Agent': 'HealthChecker/1.0'}
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                health_data = response.json()
                
                if health_data.get('status') == 'healthy':
                    status = 'healthy' if response_time <= self.config.api_response_time_threshold else 'warning'
                    message = f"API healthy, response time: {response_time:.2f}s"
                else:
                    status = 'warning'
                    message = f"API reports unhealthy status: {health_data.get('message', 'Unknown')}"
                
                return HealthCheckResult(
                    name="API Health",
                    status=status,
                    message=message,
                    response_time=response_time,
                    details=health_data
                )
            else:
                return HealthCheckResult(
                    name="API Health",
                    status='critical',
                    message=f"API health check failed with status {response.status_code}",
                    response_time=response_time
                )
                
        except requests.RequestException as e:
            return HealthCheckResult(
                name="API Health",
                status='critical',
                message=f"API health check failed: {str(e)}",
                response_time=time.time() - start_time
            )
    
    def _check_api_endpoints(self) -> List[HealthCheckResult]:
        """Check critical API endpoints"""
        api_url = self.config.api_url or f"https://api.{self.config.environment}.rental-ml.com"
        
        endpoints = [
            ("/api/v1/properties", "GET", "Properties endpoint"),
            ("/api/v1/recommendations", "GET", "Recommendations endpoint"),
            ("/api/v1/search", "POST", "Search endpoint"),
            ("/metrics", "GET", "Metrics endpoint"),
            ("/docs", "GET", "API documentation")
        ]
        
        results = []
        
        for endpoint, method, description in endpoints:
            start_time = time.time()
            
            try:
                if method == "GET":
                    response = requests.get(
                        f"{api_url}{endpoint}",
                        timeout=10,
                        headers={'User-Agent': 'HealthChecker/1.0'}
                    )
                elif method == "POST":
                    response = requests.post(
                        f"{api_url}{endpoint}",
                        json={"query": "test"},
                        timeout=10,
                        headers={'User-Agent': 'HealthChecker/1.0'}
                    )
                
                response_time = time.time() - start_time
                
                if response.status_code in [200, 201, 422]:  # 422 for validation errors is OK
                    status = 'healthy' if response_time <= self.config.api_response_time_threshold else 'warning'
                    message = f"{description} accessible, response time: {response_time:.2f}s"
                else:
                    status = 'warning'
                    message = f"{description} returned status {response.status_code}"
                
                results.append(HealthCheckResult(
                    name=f"API Endpoint: {endpoint}",
                    status=status,
                    message=message,
                    response_time=response_time
                ))
                
            except Exception as e:
                results.append(HealthCheckResult(
                    name=f"API Endpoint: {endpoint}",
                    status='critical',
                    message=f"{description} failed: {str(e)}",
                    response_time=time.time() - start_time
                ))
        
        return results
    
    def _check_database_health(self) -> HealthCheckResult:
        """Check database connectivity and performance"""
        start_time = time.time()
        
        try:
            # Connect to database
            if self.config.database_url:
                conn = psycopg2.connect(self.config.database_url)
            else:
                # Try to get connection info from Kubernetes secret
                conn = self._get_database_connection()
            
            cursor = conn.cursor()
            
            # Test basic connectivity
            cursor.execute("SELECT 1")
            cursor.fetchone()
            
            # Test database performance
            cursor.execute("SELECT COUNT(*) FROM information_schema.tables")
            table_count = cursor.fetchone()[0]
            
            # Test a typical query
            cursor.execute("""
                SELECT 
                    schemaname,
                    tablename,
                    attname,
                    n_distinct,
                    correlation
                FROM pg_stats 
                LIMIT 1
            """)
            
            response_time = time.time() - start_time
            
            cursor.close()
            conn.close()
            
            status = 'healthy' if response_time <= self.config.database_response_time_threshold else 'warning'
            message = f"Database healthy, {table_count} tables, response time: {response_time:.2f}s"
            
            return HealthCheckResult(
                name="Database Health",
                status=status,
                message=message,
                response_time=response_time,
                details={"table_count": table_count}
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="Database Health",
                status='critical',
                message=f"Database check failed: {str(e)}",
                response_time=time.time() - start_time
            )
    
    def _check_redis_health(self) -> HealthCheckResult:
        """Check Redis connectivity and performance"""
        start_time = time.time()
        
        try:
            # Connect to Redis
            if self.config.redis_url:
                r = redis.from_url(self.config.redis_url)
            else:
                # Try to get connection info from Kubernetes
                r = self._get_redis_connection()
            
            # Test connectivity
            pong = r.ping()
            
            # Test basic operations
            test_key = "health_check_test"
            r.set(test_key, "test_value", ex=60)
            value = r.get(test_key)
            r.delete(test_key)
            
            # Get Redis info
            info = r.info()
            
            response_time = time.time() - start_time
            
            status = 'healthy' if response_time <= self.config.redis_response_time_threshold else 'warning'
            message = f"Redis healthy, memory: {info.get('used_memory_human', 'unknown')}, response time: {response_time:.2f}s"
            
            return HealthCheckResult(
                name="Redis Health",
                status=status,
                message=message,
                response_time=response_time,
                details={
                    "used_memory": info.get('used_memory_human'),
                    "connected_clients": info.get('connected_clients'),
                    "role": info.get('role')
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="Redis Health",
                status='critical',
                message=f"Redis check failed: {str(e)}",
                response_time=time.time() - start_time
            )
    
    def _check_kubernetes_resources(self) -> List[HealthCheckResult]:
        """Check Kubernetes resources health"""
        results = []
        
        # Check deployments
        try:
            deployments = self.k8s_apps_v1.list_namespaced_deployment(namespace=self.config.namespace)
            
            for deployment in deployments.items:
                name = deployment.metadata.name
                ready_replicas = deployment.status.ready_replicas or 0
                desired_replicas = deployment.spec.replicas or 0
                
                if ready_replicas == desired_replicas and desired_replicas > 0:
                    status = 'healthy'
                    message = f"Deployment healthy: {ready_replicas}/{desired_replicas} replicas ready"
                elif ready_replicas > 0:
                    status = 'warning'
                    message = f"Deployment partially ready: {ready_replicas}/{desired_replicas} replicas ready"
                else:
                    status = 'critical'
                    message = f"Deployment unhealthy: {ready_replicas}/{desired_replicas} replicas ready"
                
                results.append(HealthCheckResult(
                    name=f"K8s Deployment: {name}",
                    status=status,
                    message=message,
                    response_time=0.0,
                    details={
                        "ready_replicas": ready_replicas,
                        "desired_replicas": desired_replicas
                    }
                ))
        
        except Exception as e:
            results.append(HealthCheckResult(
                name="K8s Deployments",
                status='critical',
                message=f"Failed to check deployments: {str(e)}",
                response_time=0.0
            ))
        
        # Check services
        try:
            services = self.k8s_core_v1.list_namespaced_service(namespace=self.config.namespace)
            
            for service in services.items:
                name = service.metadata.name
                
                # Check if service has endpoints
                try:
                    endpoints = self.k8s_core_v1.read_namespaced_endpoints(
                        name=name,
                        namespace=self.config.namespace
                    )
                    
                    has_endpoints = bool(endpoints.subsets)
                    
                    status = 'healthy' if has_endpoints else 'warning'
                    message = f"Service {'has' if has_endpoints else 'has no'} endpoints"
                    
                    results.append(HealthCheckResult(
                        name=f"K8s Service: {name}",
                        status=status,
                        message=message,
                        response_time=0.0
                    ))
                    
                except client.ApiException:
                    # Service might not have endpoints (e.g., ExternalName service)
                    results.append(HealthCheckResult(
                        name=f"K8s Service: {name}",
                        status='healthy',
                        message="Service exists (no endpoints check)",
                        response_time=0.0
                    ))
        
        except Exception as e:
            results.append(HealthCheckResult(
                name="K8s Services",
                status='critical',
                message=f"Failed to check services: {str(e)}",
                response_time=0.0
            ))
        
        return results
    
    def _check_service_mesh(self) -> HealthCheckResult:
        """Check Istio service mesh health"""
        start_time = time.time()
        
        try:
            # Check if Istio is installed
            istio_pods = self.k8s_core_v1.list_namespaced_pod(
                namespace="istio-system",
                label_selector="app=istiod"
            )
            
            if not istio_pods.items:
                return HealthCheckResult(
                    name="Service Mesh",
                    status='warning',
                    message="Istio not detected",
                    response_time=time.time() - start_time
                )
            
            # Check Istio control plane health
            healthy_pods = sum(1 for pod in istio_pods.items 
                             if pod.status.phase == "Running")
            total_pods = len(istio_pods.items)
            
            # Check ingress gateway
            gateway_pods = self.k8s_core_v1.list_namespaced_pod(
                namespace="istio-system",
                label_selector="app=istio-ingressgateway"
            )
            
            healthy_gateways = sum(1 for pod in gateway_pods.items 
                                 if pod.status.phase == "Running")
            total_gateways = len(gateway_pods.items)
            
            if healthy_pods == total_pods and healthy_gateways == total_gateways:
                status = 'healthy'
                message = f"Service mesh healthy: {healthy_pods}/{total_pods} control plane, {healthy_gateways}/{total_gateways} gateways"
            else:
                status = 'warning'
                message = f"Service mesh issues: {healthy_pods}/{total_pods} control plane, {healthy_gateways}/{total_gateways} gateways"
            
            return HealthCheckResult(
                name="Service Mesh",
                status=status,
                message=message,
                response_time=time.time() - start_time,
                details={
                    "control_plane_pods": f"{healthy_pods}/{total_pods}",
                    "gateway_pods": f"{healthy_gateways}/{total_gateways}"
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="Service Mesh",
                status='critical',
                message=f"Service mesh check failed: {str(e)}",
                response_time=time.time() - start_time
            )
    
    def _check_monitoring_stack(self) -> List[HealthCheckResult]:
        """Check monitoring stack components"""
        results = []
        
        monitoring_components = [
            ("Prometheus", "prometheus-server", "monitoring"),
            ("Grafana", "grafana", "monitoring"),
            ("Jaeger", "jaeger-query", "monitoring"),
            ("Alertmanager", "alertmanager", "monitoring")
        ]
        
        for component_name, service_name, namespace in monitoring_components:
            start_time = time.time()
            
            try:
                # Check if pods are running
                pods = self.k8s_core_v1.list_namespaced_pod(
                    namespace=namespace,
                    label_selector=f"app.kubernetes.io/name={service_name}"
                )
                
                if not pods.items:
                    # Try alternative label selector
                    pods = self.k8s_core_v1.list_namespaced_pod(
                        namespace=namespace,
                        label_selector=f"app={service_name}"
                    )
                
                healthy_pods = sum(1 for pod in pods.items 
                                 if pod.status.phase == "Running")
                total_pods = len(pods.items)
                
                if total_pods == 0:
                    status = 'warning'
                    message = f"{component_name} not deployed"
                elif healthy_pods == total_pods:
                    status = 'healthy'
                    message = f"{component_name} healthy: {healthy_pods}/{total_pods} pods running"
                else:
                    status = 'warning'
                    message = f"{component_name} issues: {healthy_pods}/{total_pods} pods running"
                
                results.append(HealthCheckResult(
                    name=f"Monitoring: {component_name}",
                    status=status,
                    message=message,
                    response_time=time.time() - start_time
                ))
                
            except Exception as e:
                results.append(HealthCheckResult(
                    name=f"Monitoring: {component_name}",
                    status='critical',
                    message=f"Check failed: {str(e)}",
                    response_time=time.time() - start_time
                ))
        
        return results
    
    def _check_external_dependencies(self) -> List[HealthCheckResult]:
        """Check external dependencies"""
        results = []
        
        # DNS resolution check
        start_time = time.time()
        try:
            socket.gethostbyname('google.com')
            results.append(HealthCheckResult(
                name="External DNS",
                status='healthy',
                message="DNS resolution working",
                response_time=time.time() - start_time
            ))
        except Exception as e:
            results.append(HealthCheckResult(
                name="External DNS",
                status='critical',
                message=f"DNS resolution failed: {str(e)}",
                response_time=time.time() - start_time
            ))
        
        # Internet connectivity check
        start_time = time.time()
        try:
            response = requests.get('https://httpbin.org/status/200', timeout=10)
            status = 'healthy' if response.status_code == 200 else 'warning'
            message = f"Internet connectivity {'working' if response.status_code == 200 else 'issues'}"
            
            results.append(HealthCheckResult(
                name="Internet Connectivity",
                status=status,
                message=message,
                response_time=time.time() - start_time
            ))
        except Exception as e:
            results.append(HealthCheckResult(
                name="Internet Connectivity",
                status='warning',
                message=f"Connectivity check failed: {str(e)}",
                response_time=time.time() - start_time
            ))
        
        return results
    
    def _check_performance_metrics(self) -> HealthCheckResult:
        """Check if performance metrics are within acceptable ranges"""
        start_time = time.time()
        
        try:
            if not self.config.prometheus_url:
                return HealthCheckResult(
                    name="Performance Metrics",
                    status='warning',
                    message="Prometheus URL not configured",
                    response_time=time.time() - start_time
                )
            
            # Query key performance metrics
            queries = {
                'cpu_usage': 'avg(rate(container_cpu_usage_seconds_total[5m])) * 100',
                'memory_usage': 'avg(container_memory_working_set_bytes / container_spec_memory_limit_bytes) * 100',
                'error_rate': 'rate(http_requests_total{status=~"[45].."}[5m]) / rate(http_requests_total[5m]) * 100',
                'response_time': 'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) * 1000'
            }
            
            metrics = {}
            for metric_name, query in queries.items():
                try:
                    response = requests.get(
                        f"{self.config.prometheus_url}/api/v1/query",
                        params={'query': query},
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data['data']['result']:
                            value = float(data['data']['result'][0]['value'][1])
                            metrics[metric_name] = value
                        else:
                            metrics[metric_name] = 0.0
                except:
                    metrics[metric_name] = None
            
            # Evaluate metrics
            issues = []
            if metrics.get('cpu_usage', 0) > 80:
                issues.append(f"High CPU usage: {metrics['cpu_usage']:.1f}%")
            
            if metrics.get('memory_usage', 0) > 80:
                issues.append(f"High memory usage: {metrics['memory_usage']:.1f}%")
            
            if metrics.get('error_rate', 0) > 5:
                issues.append(f"High error rate: {metrics['error_rate']:.2f}%")
            
            if metrics.get('response_time', 0) > 2000:
                issues.append(f"High response time: {metrics['response_time']:.0f}ms")
            
            if issues:
                status = 'warning'
                message = f"Performance issues detected: {'; '.join(issues)}"
            else:
                status = 'healthy'
                message = "Performance metrics within acceptable ranges"
            
            return HealthCheckResult(
                name="Performance Metrics",
                status=status,
                message=message,
                response_time=time.time() - start_time,
                details=metrics
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="Performance Metrics",
                status='warning',
                message=f"Performance check failed: {str(e)}",
                response_time=time.time() - start_time
            )
    
    def _check_security(self) -> List[HealthCheckResult]:
        """Check security-related aspects"""
        results = []
        
        # SSL/TLS certificate check
        if self.config.api_url and self.config.api_url.startswith('https://'):
            start_time = time.time()
            
            try:
                import ssl
                import socket
                from urllib.parse import urlparse
                
                parsed_url = urlparse(self.config.api_url)
                hostname = parsed_url.hostname
                port = parsed_url.port or 443
                
                context = ssl.create_default_context()
                with socket.create_connection((hostname, port), timeout=10) as sock:
                    with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                        cert = ssock.getpeercert()
                        
                        # Check certificate expiry
                        import datetime
                        expiry_date = datetime.datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
                        days_until_expiry = (expiry_date - datetime.datetime.now()).days
                        
                        if days_until_expiry > 30:
                            status = 'healthy'
                            message = f"SSL certificate valid, expires in {days_until_expiry} days"
                        elif days_until_expiry > 7:
                            status = 'warning'
                            message = f"SSL certificate expires soon: {days_until_expiry} days"
                        else:
                            status = 'critical'
                            message = f"SSL certificate expires very soon: {days_until_expiry} days"
                
                results.append(HealthCheckResult(
                    name="SSL Certificate",
                    status=status,
                    message=message,
                    response_time=time.time() - start_time
                ))
                
            except Exception as e:
                results.append(HealthCheckResult(
                    name="SSL Certificate",
                    status='warning',
                    message=f"SSL check failed: {str(e)}",
                    response_time=time.time() - start_time
                ))
        
        return results
    
    def _check_backup_systems(self) -> HealthCheckResult:
        """Check backup systems health"""
        start_time = time.time()
        
        try:
            # Check if backup CronJobs exist and are scheduled
            batch_v1 = client.BatchV1Api()
            cronjobs = batch_v1.list_namespaced_cron_job(namespace=self.config.namespace)
            
            backup_jobs = [job for job in cronjobs.items 
                          if 'backup' in job.metadata.name.lower()]
            
            if not backup_jobs:
                return HealthCheckResult(
                    name="Backup Systems",
                    status='warning',
                    message="No backup jobs found",
                    response_time=time.time() - start_time
                )
            
            active_jobs = sum(1 for job in backup_jobs if job.spec.suspend != True)
            total_jobs = len(backup_jobs)
            
            if active_jobs == total_jobs:
                status = 'healthy'
                message = f"Backup systems healthy: {active_jobs}/{total_jobs} jobs active"
            else:
                status = 'warning'
                message = f"Some backup jobs suspended: {active_jobs}/{total_jobs} jobs active"
            
            return HealthCheckResult(
                name="Backup Systems",
                status=status,
                message=message,
                response_time=time.time() - start_time,
                details={"active_jobs": active_jobs, "total_jobs": total_jobs}
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="Backup Systems",
                status='warning',
                message=f"Backup check failed: {str(e)}",
                response_time=time.time() - start_time
            )
    
    def _generate_health_report(self):
        """Generate comprehensive health report"""
        healthy_count = sum(1 for result in self.results if result.is_healthy())
        warning_count = sum(1 for result in self.results if result.status == 'warning')
        critical_count = sum(1 for result in self.results if result.status == 'critical')
        total_count = len(self.results)
        
        logger.info("=" * 80)
        logger.info("HEALTH CHECK REPORT")
        logger.info("=" * 80)
        logger.info(f"Environment: {self.config.environment}")
        logger.info(f"Namespace: {self.config.namespace}")
        logger.info(f"Total Checks: {total_count}")
        logger.info(f"Healthy: {healthy_count}")
        logger.info(f"Warning: {warning_count}")
        logger.info(f"Critical: {critical_count}")
        logger.info("=" * 80)
        
        # Group results by status
        for status in ['critical', 'warning', 'healthy']:
            status_results = [r for r in self.results if r.status == status]
            if status_results:
                logger.info(f"\n{status.upper()} CHECKS:")
                for result in status_results:
                    logger.info(f"  {result.name}: {result.message}")
                    if result.details:
                        logger.info(f"    Details: {result.details}")
        
        logger.info("=" * 80)
    
    def _get_overall_health_status(self) -> bool:
        """Determine overall health status"""
        critical_count = sum(1 for result in self.results if result.status == 'critical')
        return critical_count == 0
    
    def _get_database_connection(self):
        """Get database connection from Kubernetes secrets"""
        # This would be implemented to read from K8s secrets
        # For now, return a placeholder
        raise NotImplementedError("Database connection from K8s secrets not implemented")
    
    def _get_redis_connection(self):
        """Get Redis connection from Kubernetes secrets"""
        # This would be implemented to read from K8s secrets
        # For now, return a placeholder
        raise NotImplementedError("Redis connection from K8s secrets not implemented")

def main():
    parser = argparse.ArgumentParser(description='Comprehensive Health Check Script')
    parser.add_argument('--environment', required=True, help='Environment (dev/staging/production)')
    parser.add_argument('--namespace', help='Kubernetes namespace')
    parser.add_argument('--timeout', type=int, default=300, help='Total timeout in seconds')
    parser.add_argument('--api-url', help='API base URL')
    parser.add_argument('--prometheus-url', help='Prometheus URL')
    parser.add_argument('--database-url', help='Database connection URL')
    parser.add_argument('--redis-url', help='Redis connection URL')
    parser.add_argument('--sequential', action='store_true', help='Run checks sequentially instead of parallel')
    
    args = parser.parse_args()
    
    config = HealthCheckConfig(
        environment=args.environment,
        namespace=args.namespace or f"rental-ml-{args.environment}",
        timeout=args.timeout,
        parallel_checks=not args.sequential,
        api_url=args.api_url,
        prometheus_url=args.prometheus_url,
        database_url=args.database_url,
        redis_url=args.redis_url
    )
    
    health_checker = HealthChecker(config)
    overall_health = health_checker.run_all_checks()
    
    if overall_health:
        logger.info("✅ Overall system health: HEALTHY")
        sys.exit(0)
    else:
        logger.error("❌ Overall system health: UNHEALTHY")
        sys.exit(1)

if __name__ == "__main__":
    main()