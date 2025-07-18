#!/usr/bin/env python3
"""
Canary Deployment Script for Rental ML System

This script implements a canary deployment strategy where:
1. A new version is deployed alongside the current version
2. A small percentage of traffic is routed to the canary
3. Metrics are monitored to ensure canary health
4. Traffic is gradually increased if canary is healthy
5. Full rollout or rollback based on success criteria
"""

import argparse
import json
import logging
import subprocess
import time
import sys
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

import requests
import yaml
from kubernetes import client, config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class CanaryConfig:
    """Configuration for canary deployment"""
    namespace: str
    app_name: str
    image_tag: str
    helm_chart_path: str
    values_file: str
    health_check_url: str
    prometheus_url: str
    initial_percentage: int = 10
    target_percentage: int = 100
    increment_percentage: int = 10
    analysis_interval: int = 300  # 5 minutes
    success_threshold: float = 0.99  # 99% success rate
    max_error_rate: float = 0.05  # 5% error rate
    timeout: int = 1800  # 30 minutes total
    rollback_on_failure: bool = True

@dataclass
class CanaryMetrics:
    """Metrics for canary analysis"""
    success_rate: float
    error_rate: float
    avg_response_time: float
    request_count: int
    cpu_usage: float
    memory_usage: float

class CanaryDeployer:
    """Canary deployment orchestrator"""
    
    def __init__(self, config: CanaryConfig):
        self.config = config
        self.k8s_apps_v1 = client.AppsV1Api()
        self.k8s_core_v1 = client.CoreV1Api()
        self.k8s_networking_v1 = client.NetworkingV1Api()
        self.start_time = time.time()
        
    def deploy(self) -> bool:
        """Execute canary deployment"""
        try:
            logger.info("Starting canary deployment")
            
            # Step 1: Deploy canary version
            if not self._deploy_canary():
                logger.error("Failed to deploy canary version")
                return False
            
            # Step 2: Initial traffic routing
            current_percentage = self.config.initial_percentage
            if not self._set_traffic_split(current_percentage):
                logger.error(f"Failed to set initial traffic split to {current_percentage}%")
                return False
            
            # Step 3: Progressive traffic increase with monitoring
            while current_percentage < self.config.target_percentage:
                # Wait for analysis interval
                logger.info(f"Waiting {self.config.analysis_interval}s for metrics collection...")
                time.sleep(self.config.analysis_interval)
                
                # Analyze metrics
                if not self._analyze_canary_metrics():
                    logger.error("Canary metrics analysis failed")
                    if self.config.rollback_on_failure:
                        self._rollback_canary()
                    return False
                
                # Increase traffic
                current_percentage = min(
                    current_percentage + self.config.increment_percentage,
                    self.config.target_percentage
                )
                
                if not self._set_traffic_split(current_percentage):
                    logger.error(f"Failed to increase traffic to {current_percentage}%")
                    return False
                
                # Check timeout
                if time.time() - self.start_time > self.config.timeout:
                    logger.error("Canary deployment timed out")
                    if self.config.rollback_on_failure:
                        self._rollback_canary()
                    return False
            
            # Step 4: Final analysis and promotion
            time.sleep(self.config.analysis_interval)
            if not self._analyze_canary_metrics():
                logger.error("Final canary analysis failed")
                if self.config.rollback_on_failure:
                    self._rollback_canary()
                return False
            
            # Step 5: Promote canary to production
            if not self._promote_canary():
                logger.error("Failed to promote canary")
                return False
            
            logger.info("Canary deployment completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Canary deployment failed with exception: {e}")
            if self.config.rollback_on_failure:
                self._rollback_canary()
            return False
    
    def _deploy_canary(self) -> bool:
        """Deploy canary version of the application"""
        try:
            canary_name = f"{self.config.app_name}-canary"
            
            # Prepare Helm command for canary deployment
            helm_cmd = [
                "helm", "upgrade", "--install", canary_name,
                self.config.helm_chart_path,
                "--namespace", self.config.namespace,
                "--values", self.config.values_file,
                "--set", f"image.tag={self.config.image_tag}",
                "--set", f"nameOverride={canary_name}",
                "--set", f"fullnameOverride={canary_name}",
                "--set", "canary.enabled=true",
                "--set", "service.type=ClusterIP",
                "--set", "ingress.enabled=false",  # Canary uses separate service
                "--wait", "--timeout", "300s"
            ]
            
            logger.info(f"Deploying canary with command: {' '.join(helm_cmd)}")
            
            result = subprocess.run(
                helm_cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                logger.error(f"Canary deployment failed: {result.stderr}")
                return False
            
            # Wait for canary to be ready
            if not self._wait_for_deployment_ready(canary_name):
                logger.error("Canary deployment not ready")
                return False
            
            logger.info("Canary deployment successful")
            return True
            
        except Exception as e:
            logger.error(f"Canary deployment failed: {e}")
            return False
    
    def _wait_for_deployment_ready(self, deployment_name: str) -> bool:
        """Wait for deployment to be ready"""
        logger.info(f"Waiting for deployment {deployment_name} to be ready")
        
        for attempt in range(20):  # 10 minutes max wait
            try:
                deployment = self.k8s_apps_v1.read_namespaced_deployment(
                    name=deployment_name,
                    namespace=self.config.namespace
                )
                
                if (deployment.status.ready_replicas and 
                    deployment.status.ready_replicas == deployment.spec.replicas):
                    logger.info(f"Deployment {deployment_name} is ready")
                    return True
                
                logger.info(f"Deployment not ready yet. Ready: {deployment.status.ready_replicas}/{deployment.spec.replicas}")
                time.sleep(30)
                
            except client.ApiException as e:
                logger.error(f"Error checking deployment status: {e}")
                time.sleep(30)
        
        return False
    
    def _set_traffic_split(self, canary_percentage: int) -> bool:
        """Set traffic split between stable and canary versions"""
        try:
            # Update Istio VirtualService for traffic splitting
            virtual_service_config = {
                "apiVersion": "networking.istio.io/v1beta1",
                "kind": "VirtualService",
                "metadata": {
                    "name": f"{self.config.app_name}-canary",
                    "namespace": self.config.namespace
                },
                "spec": {
                    "hosts": [f"api.{self.config.namespace}.rental-ml.com"],
                    "gateways": ["istio-system/rental-ml-gateway"],
                    "http": [{
                        "match": [{"uri": {"prefix": "/api/"}}],
                        "route": [
                            {
                                "destination": {
                                    "host": f"{self.config.app_name}-canary",
                                    "port": {"number": 8000}
                                },
                                "weight": canary_percentage
                            },
                            {
                                "destination": {
                                    "host": self.config.app_name,
                                    "port": {"number": 8000}
                                },
                                "weight": 100 - canary_percentage
                            }
                        ],
                        "fault": {
                            "delay": {
                                "percentage": {"value": 0.1},
                                "fixedDelay": "5s"
                            }
                        },
                        "retries": {
                            "attempts": 3,
                            "perTryTimeout": "30s"
                        }
                    }]
                }
            }
            
            # Apply the VirtualService
            with open(f"/tmp/canary-virtualservice-{self.config.app_name}.yaml", "w") as f:
                yaml.dump(virtual_service_config, f)
            
            apply_cmd = [
                "kubectl", "apply", "-f", f"/tmp/canary-virtualservice-{self.config.app_name}.yaml"
            ]
            
            result = subprocess.run(apply_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Failed to apply traffic split: {result.stderr}")
                return False
            
            logger.info(f"Traffic split set: {canary_percentage}% canary, {100-canary_percentage}% stable")
            
            # Wait for traffic split to propagate
            time.sleep(30)
            return True
            
        except Exception as e:
            logger.error(f"Failed to set traffic split: {e}")
            return False
    
    def _analyze_canary_metrics(self) -> bool:
        """Analyze canary metrics to determine if deployment should continue"""
        try:
            logger.info("Analyzing canary metrics...")
            
            # Get metrics for both canary and stable versions
            canary_metrics = self._get_metrics(f"{self.config.app_name}-canary")
            stable_metrics = self._get_metrics(self.config.app_name)
            
            if not canary_metrics or not stable_metrics:
                logger.error("Failed to retrieve metrics")
                return False
            
            # Log metrics
            logger.info(f"Canary metrics: Success rate: {canary_metrics.success_rate:.2%}, "
                       f"Error rate: {canary_metrics.error_rate:.2%}, "
                       f"Avg response time: {canary_metrics.avg_response_time:.2f}ms")
            
            logger.info(f"Stable metrics: Success rate: {stable_metrics.success_rate:.2%}, "
                       f"Error rate: {stable_metrics.error_rate:.2%}, "
                       f"Avg response time: {stable_metrics.avg_response_time:.2f}ms")
            
            # Success criteria checks
            success_checks = [
                (canary_metrics.success_rate >= self.config.success_threshold, 
                 f"Success rate {canary_metrics.success_rate:.2%} >= {self.config.success_threshold:.2%}"),
                
                (canary_metrics.error_rate <= self.config.max_error_rate,
                 f"Error rate {canary_metrics.error_rate:.2%} <= {self.config.max_error_rate:.2%}"),
                
                (canary_metrics.avg_response_time <= stable_metrics.avg_response_time * 1.5,
                 f"Response time {canary_metrics.avg_response_time:.2f}ms <= {stable_metrics.avg_response_time * 1.5:.2f}ms"),
                
                (canary_metrics.cpu_usage <= 90.0,
                 f"CPU usage {canary_metrics.cpu_usage:.1f}% <= 90%"),
                
                (canary_metrics.memory_usage <= 90.0,
                 f"Memory usage {canary_metrics.memory_usage:.1f}% <= 90%")
            ]
            
            failed_checks = []
            for check_passed, description in success_checks:
                if check_passed:
                    logger.info(f"✓ {description}")
                else:
                    logger.error(f"✗ {description}")
                    failed_checks.append(description)
            
            if failed_checks:
                logger.error(f"Canary analysis failed. Failed checks: {failed_checks}")
                return False
            
            logger.info("Canary analysis passed - proceeding with deployment")
            return True
            
        except Exception as e:
            logger.error(f"Metrics analysis failed: {e}")
            return False
    
    def _get_metrics(self, service_name: str) -> Optional[CanaryMetrics]:
        """Retrieve metrics for a service from Prometheus"""
        try:
            # Prometheus queries for metrics
            queries = {
                'success_rate': f'rate(http_requests_total{{service="{service_name}",status=~"2.."}}[5m]) / rate(http_requests_total{{service="{service_name}"}}[5m])',
                'error_rate': f'rate(http_requests_total{{service="{service_name}",status=~"[45].."}}[5m]) / rate(http_requests_total{{service="{service_name}"}}[5m])',
                'response_time': f'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{{service="{service_name}"}}[5m])) * 1000',
                'request_count': f'sum(rate(http_requests_total{{service="{service_name}"}}[5m]))',
                'cpu_usage': f'avg(rate(container_cpu_usage_seconds_total{{pod=~"{service_name}-.*"}}[5m])) * 100',
                'memory_usage': f'avg(container_memory_working_set_bytes{{pod=~"{service_name}-.*"}}) / avg(container_spec_memory_limit_bytes{{pod=~"{service_name}-.*"}}) * 100'
            }
            
            metrics = {}
            for metric_name, query in queries.items():
                response = requests.get(
                    f"{self.config.prometheus_url}/api/v1/query",
                    params={'query': query},
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data['data']['result']:
                        value = float(data['data']['result'][0]['value'][1])
                        metrics[metric_name] = value
                    else:
                        logger.warning(f"No data for metric {metric_name}")
                        metrics[metric_name] = 0.0
                else:
                    logger.error(f"Failed to query metric {metric_name}: {response.status_code}")
                    return None
            
            return CanaryMetrics(
                success_rate=metrics.get('success_rate', 0.0),
                error_rate=metrics.get('error_rate', 0.0),
                avg_response_time=metrics.get('response_time', 0.0),
                request_count=int(metrics.get('request_count', 0)),
                cpu_usage=metrics.get('cpu_usage', 0.0),
                memory_usage=metrics.get('memory_usage', 0.0)
            )
            
        except Exception as e:
            logger.error(f"Failed to get metrics for {service_name}: {e}")
            return None
    
    def _promote_canary(self) -> bool:
        """Promote canary to production by replacing stable version"""
        try:
            logger.info("Promoting canary to production")
            
            # Update the main deployment with canary image
            deployment = self.k8s_apps_v1.read_namespaced_deployment(
                name=self.config.app_name,
                namespace=self.config.namespace
            )
            
            # Update image tag
            for container in deployment.spec.template.spec.containers:
                if container.name == self.config.app_name:
                    current_image = container.image
                    new_image = f"{current_image.split(':')[0]}:{self.config.image_tag}"
                    container.image = new_image
                    break
            
            # Apply the update
            self.k8s_apps_v1.patch_namespaced_deployment(
                name=self.config.app_name,
                namespace=self.config.namespace,
                body=deployment
            )
            
            # Wait for rollout to complete
            time.sleep(60)
            
            # Remove traffic split (100% to main service)
            self._remove_traffic_split()
            
            # Clean up canary deployment
            self._cleanup_canary()
            
            logger.info("Canary promotion completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to promote canary: {e}")
            return False
    
    def _rollback_canary(self) -> bool:
        """Rollback canary deployment"""
        try:
            logger.info("Rolling back canary deployment")
            
            # Remove traffic split
            self._remove_traffic_split()
            
            # Clean up canary deployment
            self._cleanup_canary()
            
            logger.info("Canary rollback completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rollback canary: {e}")
            return False
    
    def _remove_traffic_split(self):
        """Remove traffic split configuration"""
        try:
            delete_cmd = [
                "kubectl", "delete", "virtualservice",
                f"{self.config.app_name}-canary",
                "-n", self.config.namespace,
                "--ignore-not-found=true"
            ]
            
            subprocess.run(delete_cmd, capture_output=True, text=True)
            logger.info("Traffic split configuration removed")
            
        except Exception as e:
            logger.warning(f"Failed to remove traffic split: {e}")
    
    def _cleanup_canary(self):
        """Clean up canary deployment resources"""
        try:
            canary_name = f"{self.config.app_name}-canary"
            
            # Delete canary Helm release
            delete_cmd = [
                "helm", "uninstall", canary_name,
                "-n", self.config.namespace
            ]
            
            subprocess.run(delete_cmd, capture_output=True, text=True)
            logger.info("Canary deployment cleaned up")
            
        except Exception as e:
            logger.warning(f"Failed to cleanup canary: {e}")

def main():
    parser = argparse.ArgumentParser(description='Canary Deployment Script')
    parser.add_argument('--namespace', required=True, help='Kubernetes namespace')
    parser.add_argument('--app-name', required=True, help='Application name')
    parser.add_argument('--image-tag', required=True, help='Docker image tag')
    parser.add_argument('--canary-percentage', type=int, default=10, help='Initial canary percentage')
    parser.add_argument('--prometheus-url', default='http://prometheus-server.monitoring.svc.cluster.local:80', help='Prometheus URL')
    parser.add_argument('--timeout', type=int, default=1800, help='Total deployment timeout')
    parser.add_argument('--config-file', help='Deployment configuration file')
    
    args = parser.parse_args()
    
    # Load Kubernetes config
    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()
    
    # Build configuration
    if args.config_file:
        with open(args.config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        canary_config = CanaryConfig(**config_data)
    else:
        canary_config = CanaryConfig(
            namespace=args.namespace,
            app_name=args.app_name,
            image_tag=args.image_tag,
            helm_chart_path="k8s/helm/rental-ml/",
            values_file=f"k8s/helm/rental-ml/values-{args.namespace}.yaml",
            health_check_url=f"https://api.{args.namespace}.rental-ml.com",
            prometheus_url=args.prometheus_url,
            initial_percentage=args.canary_percentage,
            timeout=args.timeout
        )
    
    # Execute deployment
    deployer = CanaryDeployer(canary_config)
    success = deployer.deploy()
    
    if success:
        logger.info("Canary deployment completed successfully")
        sys.exit(0)
    else:
        logger.error("Canary deployment failed")
        sys.exit(1)

if __name__ == "__main__":
    main()