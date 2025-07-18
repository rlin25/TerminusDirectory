#!/usr/bin/env python3
"""
Blue-Green Deployment Script for Rental ML System

This script implements a blue-green deployment strategy where:
1. A new version (green) is deployed alongside the current version (blue)
2. Health checks are performed on the green environment
3. Traffic is gradually switched from blue to green
4. Blue environment is kept for quick rollback if needed
"""

import argparse
import json
import logging
import subprocess
import time
import sys
from typing import Dict, List, Optional
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
class DeploymentConfig:
    """Configuration for blue-green deployment"""
    namespace: str
    app_name: str
    image_tag: str
    helm_chart_path: str
    values_file: str
    health_check_url: str
    timeout: int
    traffic_switch_delay: int = 60
    health_check_retries: int = 10
    health_check_interval: int = 30

class BlueGreenDeployer:
    """Blue-Green deployment orchestrator"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.k8s_apps_v1 = client.AppsV1Api()
        self.k8s_core_v1 = client.CoreV1Api()
        self.k8s_networking_v1 = client.NetworkingV1Api()
        
    def deploy(self) -> bool:
        """Execute blue-green deployment"""
        try:
            logger.info("Starting blue-green deployment")
            
            # Step 1: Determine current environment (blue/green)
            current_env = self._get_current_environment()
            new_env = "green" if current_env == "blue" else "blue"
            
            logger.info(f"Current environment: {current_env}, New environment: {new_env}")
            
            # Step 2: Deploy new version to inactive environment
            if not self._deploy_to_environment(new_env):
                logger.error(f"Failed to deploy to {new_env} environment")
                return False
            
            # Step 3: Wait for deployment to be ready
            if not self._wait_for_deployment_ready(new_env):
                logger.error(f"Deployment to {new_env} environment not ready")
                return False
            
            # Step 4: Perform health checks
            if not self._perform_health_checks(new_env):
                logger.error(f"Health checks failed for {new_env} environment")
                return False
            
            # Step 5: Perform smoke tests
            if not self._run_smoke_tests(new_env):
                logger.error(f"Smoke tests failed for {new_env} environment")
                return False
            
            # Step 6: Switch traffic to new environment
            if not self._switch_traffic(new_env):
                logger.error(f"Failed to switch traffic to {new_env}")
                return False
            
            # Step 7: Verify traffic switch
            if not self._verify_traffic_switch(new_env):
                logger.error("Traffic switch verification failed")
                if not self._rollback_traffic(current_env):
                    logger.critical("Failed to rollback traffic!")
                return False
            
            # Step 8: Update deployment labels
            self._update_deployment_labels(new_env)
            
            logger.info(f"Blue-green deployment completed successfully. Active environment: {new_env}")
            return True
            
        except Exception as e:
            logger.error(f"Deployment failed with exception: {e}")
            return False
    
    def _get_current_environment(self) -> str:
        """Determine the current active environment"""
        try:
            # Check the current service selector
            service = self.k8s_core_v1.read_namespaced_service(
                name=self.config.app_name,
                namespace=self.config.namespace
            )
            
            if 'environment' in service.spec.selector:
                return service.spec.selector['environment']
            else:
                # Default to blue if no environment label
                return "blue"
                
        except client.ApiException as e:
            if e.status == 404:
                logger.info("Service not found, starting with blue environment")
                return "blue"
            else:
                raise
    
    def _deploy_to_environment(self, environment: str) -> bool:
        """Deploy application to specified environment using Helm"""
        try:
            release_name = f"{self.config.app_name}-{environment}"
            
            # Prepare Helm command
            helm_cmd = [
                "helm", "upgrade", "--install", release_name,
                self.config.helm_chart_path,
                "--namespace", self.config.namespace,
                "--values", self.config.values_file,
                "--set", f"image.tag={self.config.image_tag}",
                "--set", f"environment={environment}",
                "--set", f"nameOverride={self.config.app_name}-{environment}",
                "--set", f"fullnameOverride={self.config.app_name}-{environment}",
                "--wait", "--timeout", f"{self.config.timeout}s"
            ]
            
            logger.info(f"Deploying to {environment} environment with command: {' '.join(helm_cmd)}")
            
            result = subprocess.run(
                helm_cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout
            )
            
            if result.returncode != 0:
                logger.error(f"Helm deployment failed: {result.stderr}")
                return False
            
            logger.info(f"Successfully deployed to {environment} environment")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error(f"Helm deployment timed out after {self.config.timeout} seconds")
            return False
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return False
    
    def _wait_for_deployment_ready(self, environment: str) -> bool:
        """Wait for deployment to be ready"""
        deployment_name = f"{self.config.app_name}-{environment}"
        
        logger.info(f"Waiting for deployment {deployment_name} to be ready")
        
        for attempt in range(self.config.health_check_retries):
            try:
                deployment = self.k8s_apps_v1.read_namespaced_deployment(
                    name=deployment_name,
                    namespace=self.config.namespace
                )
                
                # Check if deployment is ready
                if (deployment.status.ready_replicas and 
                    deployment.status.ready_replicas == deployment.spec.replicas):
                    logger.info(f"Deployment {deployment_name} is ready")
                    return True
                
                logger.info(f"Deployment not ready yet. Ready: {deployment.status.ready_replicas}/{deployment.spec.replicas}")
                time.sleep(self.config.health_check_interval)
                
            except client.ApiException as e:
                logger.error(f"Error checking deployment status: {e}")
                time.sleep(self.config.health_check_interval)
        
        logger.error(f"Deployment {deployment_name} not ready after {self.config.health_check_retries} attempts")
        return False
    
    def _perform_health_checks(self, environment: str) -> bool:
        """Perform health checks on the new environment"""
        service_name = f"{self.config.app_name}-{environment}"
        
        # Get service port
        try:
            service = self.k8s_core_v1.read_namespaced_service(
                name=service_name,
                namespace=self.config.namespace
            )
            port = service.spec.ports[0].port
        except client.ApiException:
            logger.error(f"Failed to get service {service_name}")
            return False
        
        # Perform health checks using port-forward
        health_url = f"http://localhost:8080/health"
        
        logger.info(f"Starting health checks for {environment} environment")
        
        # Start port-forward in background
        port_forward_cmd = [
            "kubectl", "port-forward",
            f"service/{service_name}",
            f"8080:{port}",
            "-n", self.config.namespace
        ]
        
        port_forward_process = subprocess.Popen(
            port_forward_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        try:
            # Wait for port-forward to be ready
            time.sleep(5)
            
            for attempt in range(self.config.health_check_retries):
                try:
                    response = requests.get(
                        health_url,
                        timeout=10,
                        headers={'User-Agent': 'BlueGreenDeployer/1.0'}
                    )
                    
                    if response.status_code == 200:
                        health_data = response.json()
                        if health_data.get('status') == 'healthy':
                            logger.info(f"Health check passed for {environment}")
                            return True
                    
                    logger.warning(f"Health check failed: {response.status_code}")
                    
                except requests.RequestException as e:
                    logger.warning(f"Health check request failed: {e}")
                
                if attempt < self.config.health_check_retries - 1:
                    logger.info(f"Retrying health check in {self.config.health_check_interval} seconds")
                    time.sleep(self.config.health_check_interval)
            
            logger.error(f"All health checks failed for {environment}")
            return False
            
        finally:
            # Clean up port-forward process
            port_forward_process.terminate()
            port_forward_process.wait()
    
    def _run_smoke_tests(self, environment: str) -> bool:
        """Run smoke tests against the new environment"""
        logger.info(f"Running smoke tests for {environment} environment")
        
        try:
            # Run smoke tests using kubectl exec
            pod_selector = f"app={self.config.app_name}-{environment}"
            
            # Get first pod
            pods = self.k8s_core_v1.list_namespaced_pod(
                namespace=self.config.namespace,
                label_selector=pod_selector
            )
            
            if not pods.items:
                logger.error(f"No pods found for selector {pod_selector}")
                return False
            
            pod_name = pods.items[0].metadata.name
            
            # Run smoke test command in pod
            smoke_test_cmd = [
                "kubectl", "exec", pod_name,
                "-n", self.config.namespace,
                "--", "python", "-m", "pytest",
                "tests/smoke/", "-v", "--tb=short"
            ]
            
            result = subprocess.run(
                smoke_test_cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout for smoke tests
            )
            
            if result.returncode == 0:
                logger.info("Smoke tests passed")
                return True
            else:
                logger.error(f"Smoke tests failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to run smoke tests: {e}")
            return False
    
    def _switch_traffic(self, new_environment: str) -> bool:
        """Switch traffic to new environment"""
        try:
            # Update main service selector to point to new environment
            service = self.k8s_core_v1.read_namespaced_service(
                name=self.config.app_name,
                namespace=self.config.namespace
            )
            
            # Update selector
            service.spec.selector['environment'] = new_environment
            
            # Patch the service
            self.k8s_core_v1.patch_namespaced_service(
                name=self.config.app_name,
                namespace=self.config.namespace,
                body=service
            )
            
            logger.info(f"Traffic switched to {new_environment} environment")
            
            # Wait for traffic switch to propagate
            time.sleep(self.config.traffic_switch_delay)
            
            return True
            
        except client.ApiException as e:
            logger.error(f"Failed to switch traffic: {e}")
            return False
    
    def _verify_traffic_switch(self, environment: str) -> bool:
        """Verify that traffic is correctly routed to new environment"""
        logger.info(f"Verifying traffic switch to {environment}")
        
        # Make several requests to verify traffic routing
        for attempt in range(5):
            try:
                # Use the health endpoint to verify which environment is serving
                response = requests.get(
                    f"{self.config.health_check_url}/health",
                    timeout=10,
                    headers={'User-Agent': 'BlueGreenDeployer/1.0'}
                )
                
                if response.status_code == 200:
                    health_data = response.json()
                    serving_env = health_data.get('environment', 'unknown')
                    
                    if serving_env == environment:
                        logger.info(f"Traffic verification successful - serving from {environment}")
                        return True
                    else:
                        logger.warning(f"Expected {environment}, but serving from {serving_env}")
                
                time.sleep(5)
                
            except requests.RequestException as e:
                logger.warning(f"Traffic verification request failed: {e}")
                time.sleep(5)
        
        logger.error("Traffic verification failed")
        return False
    
    def _rollback_traffic(self, environment: str) -> bool:
        """Rollback traffic to previous environment"""
        logger.info(f"Rolling back traffic to {environment}")
        return self._switch_traffic(environment)
    
    def _update_deployment_labels(self, environment: str):
        """Update deployment labels to track active environment"""
        try:
            # Update configmap with current environment info
            config_map_data = {
                'active_environment': environment,
                'deployment_time': time.strftime('%Y-%m-%d %H:%M:%S UTC'),
                'image_tag': self.config.image_tag
            }
            
            config_map = client.V1ConfigMap(
                metadata=client.V1ObjectMeta(
                    name=f"{self.config.app_name}-deployment-info",
                    namespace=self.config.namespace
                ),
                data=config_map_data
            )
            
            try:
                self.k8s_core_v1.create_namespaced_config_map(
                    namespace=self.config.namespace,
                    body=config_map
                )
            except client.ApiException as e:
                if e.status == 409:  # Already exists
                    self.k8s_core_v1.patch_namespaced_config_map(
                        name=f"{self.config.app_name}-deployment-info",
                        namespace=self.config.namespace,
                        body=config_map
                    )
                else:
                    raise
            
            logger.info("Updated deployment labels")
            
        except Exception as e:
            logger.warning(f"Failed to update deployment labels: {e}")

def main():
    parser = argparse.ArgumentParser(description='Blue-Green Deployment Script')
    parser.add_argument('--namespace', required=True, help='Kubernetes namespace')
    parser.add_argument('--app-name', required=True, help='Application name')
    parser.add_argument('--image-tag', required=True, help='Docker image tag')
    parser.add_argument('--helm-chart-path', default='k8s/helm/rental-ml/', help='Helm chart path')
    parser.add_argument('--values-file', help='Helm values file')
    parser.add_argument('--health-check-url', help='Health check URL')
    parser.add_argument('--timeout', type=int, default=600, help='Deployment timeout in seconds')
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
        deploy_config = DeploymentConfig(**config_data)
    else:
        deploy_config = DeploymentConfig(
            namespace=args.namespace,
            app_name=args.app_name,
            image_tag=args.image_tag,
            helm_chart_path=args.helm_chart_path,
            values_file=args.values_file or f"k8s/helm/rental-ml/values-{args.namespace}.yaml",
            health_check_url=args.health_check_url or f"https://api.{args.namespace}.rental-ml.com",
            timeout=args.timeout
        )
    
    # Execute deployment
    deployer = BlueGreenDeployer(deploy_config)
    success = deployer.deploy()
    
    if success:
        logger.info("Blue-green deployment completed successfully")
        sys.exit(0)
    else:
        logger.error("Blue-green deployment failed")
        sys.exit(1)

if __name__ == "__main__":
    main()