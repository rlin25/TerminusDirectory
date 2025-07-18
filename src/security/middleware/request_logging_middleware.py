"""
Request Logging Middleware

Comprehensive request logging middleware that captures detailed information
about incoming requests and responses for security monitoring and analytics.
"""

import json
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from urllib.parse import urlparse, parse_qs

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Request Logging Middleware that captures:
    - Request details (method, path, headers, query params)
    - Response details (status code, headers, size)
    - Timing information
    - User context (if available)
    - IP address and geolocation
    - User agent analysis
    - Security events
    - Performance metrics
    """
    
    def __init__(
        self,
        app,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(app)
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Logging configuration
        self.log_level = self.config.get("log_level", "INFO")
        self.log_requests = self.config.get("log_requests", True)
        self.log_responses = self.config.get("log_responses", True)
        self.log_request_body = self.config.get("log_request_body", False)
        self.log_response_body = self.config.get("log_response_body", False)
        self.log_headers = self.config.get("log_headers", True)
        self.log_query_params = self.config.get("log_query_params", True)
        
        # Security logging
        self.log_security_events = self.config.get("log_security_events", True)
        self.log_failed_requests = self.config.get("log_failed_requests", True)
        self.log_slow_requests = self.config.get("log_slow_requests", True)
        self.slow_request_threshold_ms = self.config.get("slow_request_threshold_ms", 1000)
        
        # Privacy configuration
        self.sensitive_headers = set(self.config.get("sensitive_headers", [
            "authorization", "x-api-key", "cookie", "x-auth-token",
            "x-csrf-token", "x-access-token", "x-refresh-token"
        ]))
        self.sensitive_query_params = set(self.config.get("sensitive_query_params", [
            "token", "api_key", "password", "secret", "key", "auth"
        ]))
        self.sensitive_form_fields = set(self.config.get("sensitive_form_fields", [
            "password", "confirm_password", "current_password", "new_password",
            "credit_card", "ssn", "social_security_number"
        ]))
        
        # Path filtering
        self.exclude_paths = set(self.config.get("exclude_paths", [
            "/health", "/ping", "/metrics", "/favicon.ico"
        ]))
        self.include_paths = set(self.config.get("include_paths", []))
        
        # Sampling configuration
        self.sampling_rate = self.config.get("sampling_rate", 1.0)  # 1.0 = 100%
        self.sample_counter = 0
        
        # Output configuration
        self.output_format = self.config.get("output_format", "json")  # json, text, structured
        self.output_destination = self.config.get("output_destination", "logger")  # logger, file, database, elasticsearch
        
        # Structured logging
        self.structured_logger = self._setup_structured_logger()
        
        # Statistics
        self._stats = {
            "total_requests": 0,
            "logged_requests": 0,
            "skipped_requests": 0,
            "slow_requests": 0,
            "error_responses": 0,
            "security_events": 0,
            "sampling_hits": 0,
            "sampling_misses": 0
        }
    
    def _setup_structured_logger(self):
        """Setup structured logger for request logging"""
        structured_logger = logging.getLogger(f"{__name__}.structured")
        
        # Configure handler based on output destination
        if self.output_destination == "file":
            handler = logging.FileHandler(
                self.config.get("log_file", "requests.log")
            )
        else:
            handler = logging.StreamHandler()
        
        # Set formatter
        if self.output_format == "json":
            formatter = logging.Formatter('%(message)s')
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        handler.setFormatter(formatter)
        structured_logger.addHandler(handler)
        structured_logger.setLevel(getattr(logging, self.log_level))
        
        return structured_logger
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Main middleware dispatch method"""
        start_time = time.time()
        
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Check if request should be logged
        if not self._should_log_request(request):
            self._stats["skipped_requests"] += 1
            return await call_next(request)
        
        # Apply sampling
        if not self._should_sample():
            self._stats["sampling_misses"] += 1
            return await call_next(request)
        
        self._stats["total_requests"] += 1
        self._stats["sampling_hits"] += 1
        
        # Capture request information
        request_info = await self._capture_request_info(request, start_time)
        
        # Process request
        response = await call_next(request)
        
        # Capture response information
        end_time = time.time()
        response_info = self._capture_response_info(response, end_time)
        
        # Calculate metrics
        duration_ms = (end_time - start_time) * 1000
        
        # Create log entry
        log_entry = self._create_log_entry(
            request_info, response_info, duration_ms, request_id
        )
        
        # Log the entry
        await self._log_entry(log_entry)
        
        # Update statistics
        self._update_statistics(response, duration_ms)
        
        return response
    
    def _should_log_request(self, request: Request) -> bool:
        """Check if request should be logged"""
        path = request.url.path
        
        # Check exclude paths
        if any(path.startswith(exclude_path) for exclude_path in self.exclude_paths):
            return False
        
        # Check include paths (if configured)
        if self.include_paths:
            return any(path.startswith(include_path) for include_path in self.include_paths)
        
        return True
    
    def _should_sample(self) -> bool:
        """Check if request should be sampled"""
        if self.sampling_rate >= 1.0:
            return True
        
        self.sample_counter += 1
        return (self.sample_counter % int(1 / self.sampling_rate)) == 0
    
    async def _capture_request_info(self, request: Request, start_time: float) -> Dict[str, Any]:
        """Capture request information"""
        request_info = {
            "timestamp": datetime.utcnow().isoformat(),
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "query_string": str(request.url.query) if request.url.query else None,
            "scheme": request.url.scheme,
            "host": request.url.hostname,
            "port": request.url.port,
            "client_ip": self._get_client_ip(request),
            "user_agent": request.headers.get("user-agent"),
            "referer": request.headers.get("referer"),
            "content_type": request.headers.get("content-type"),
            "content_length": request.headers.get("content-length"),
            "accept": request.headers.get("accept"),
            "accept_language": request.headers.get("accept-language"),
            "accept_encoding": request.headers.get("accept-encoding")
        }
        
        # Add headers if enabled
        if self.log_headers:
            request_info["headers"] = self._sanitize_headers(dict(request.headers))
        
        # Add query parameters if enabled
        if self.log_query_params and request.url.query:
            query_params = dict(request.query_params)
            request_info["query_params"] = self._sanitize_query_params(query_params)
        
        # Add user context if available
        security_context = getattr(request.state, 'security_context', None)
        if security_context:
            request_info["user"] = {
                "user_id": str(security_context.user_id),
                "username": security_context.username,
                "roles": [role.value for role in security_context.roles],
                "authenticated": True
            }
        else:
            request_info["user"] = {"authenticated": False}
        
        # Add request body if enabled and appropriate
        if self.log_request_body and request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if body:
                    content_type = request.headers.get("content-type", "")
                    if "application/json" in content_type:
                        try:
                            body_data = json.loads(body.decode('utf-8'))
                            request_info["body"] = self._sanitize_request_body(body_data)
                        except json.JSONDecodeError:
                            request_info["body"] = {"error": "Invalid JSON"}
                    else:
                        request_info["body"] = {"size": len(body), "type": content_type}
            except Exception as e:
                request_info["body"] = {"error": str(e)}
        
        return request_info
    
    def _capture_response_info(self, response: Response, end_time: float) -> Dict[str, Any]:
        """Capture response information"""
        response_info = {
            "status_code": response.status_code,
            "content_type": response.headers.get("content-type"),
            "content_length": response.headers.get("content-length"),
            "cache_control": response.headers.get("cache-control"),
            "location": response.headers.get("location"),
            "server": response.headers.get("server")
        }
        
        # Add response headers if enabled
        if self.log_headers:
            response_info["headers"] = self._sanitize_headers(dict(response.headers))
        
        # Add response body if enabled (be careful with large responses)
        if self.log_response_body and response.status_code >= 400:
            # Only log response body for error responses
            try:
                # This is a simplified approach - in production, you'd need
                # to handle streaming responses properly
                pass
            except Exception:
                pass
        
        return response_info
    
    def _create_log_entry(
        self,
        request_info: Dict[str, Any],
        response_info: Dict[str, Any],
        duration_ms: float,
        request_id: str
    ) -> Dict[str, Any]:
        """Create structured log entry"""
        log_entry = {
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "duration_ms": round(duration_ms, 2),
            "request": request_info,
            "response": response_info,
            "performance": {
                "duration_ms": round(duration_ms, 2),
                "is_slow": duration_ms > self.slow_request_threshold_ms
            }
        }
        
        # Add classification
        log_entry["classification"] = self._classify_request(
            request_info, response_info, duration_ms
        )
        
        return log_entry
    
    def _classify_request(
        self,
        request_info: Dict[str, Any],
        response_info: Dict[str, Any],
        duration_ms: float
    ) -> Dict[str, Any]:
        """Classify request for analysis"""
        classification = {
            "request_type": "normal",
            "security_level": "low",
            "performance_level": "normal",
            "tags": []
        }
        
        # Security classification
        if response_info["status_code"] in [401, 403]:
            classification["security_level"] = "medium"
            classification["tags"].append("auth_failure")
        elif response_info["status_code"] == 429:
            classification["security_level"] = "high"
            classification["tags"].append("rate_limited")
        elif response_info["status_code"] >= 500:
            classification["security_level"] = "medium"
            classification["tags"].append("server_error")
        
        # Performance classification
        if duration_ms > self.slow_request_threshold_ms:
            classification["performance_level"] = "slow"
            classification["tags"].append("slow_request")
        
        # Request type classification
        if request_info["method"] in ["POST", "PUT", "PATCH", "DELETE"]:
            classification["request_type"] = "mutating"
        
        path = request_info["path"]
        if "/auth/" in path:
            classification["tags"].append("authentication")
        elif "/admin/" in path:
            classification["tags"].append("admin")
        elif "/api/" in path:
            classification["tags"].append("api")
        
        return classification
    
    async def _log_entry(self, log_entry: Dict[str, Any]):
        """Log the entry based on configuration"""
        self._stats["logged_requests"] += 1
        
        if self.output_format == "json":
            message = json.dumps(log_entry)
        else:
            # Simple text format
            req = log_entry["request"]
            resp = log_entry["response"]
            message = (
                f'{req["client_ip"]} - {req["method"]} {req["path"]} '
                f'HTTP/{resp["status_code"]} {log_entry["duration_ms"]}ms'
            )
        
        # Log based on response status
        if log_entry["response"]["status_code"] >= 500:
            self.structured_logger.error(message)
        elif log_entry["response"]["status_code"] >= 400:
            self.structured_logger.warning(message)
        elif log_entry["performance"]["is_slow"]:
            self.structured_logger.warning(message)
        else:
            self.structured_logger.info(message)
        
        # Send to additional destinations if configured
        if self.output_destination == "database":
            await self._log_to_database(log_entry)
        elif self.output_destination == "elasticsearch":
            await self._log_to_elasticsearch(log_entry)
    
    async def _log_to_database(self, log_entry: Dict[str, Any]):
        """Log to database (implementation depends on your database)"""
        # This would be implemented based on your database choice
        pass
    
    async def _log_to_elasticsearch(self, log_entry: Dict[str, Any]):
        """Log to Elasticsearch"""
        # This would be implemented if you're using Elasticsearch
        pass
    
    def _sanitize_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Sanitize headers to remove sensitive information"""
        sanitized = {}
        for key, value in headers.items():
            if key.lower() in self.sensitive_headers:
                sanitized[key] = "[REDACTED]"
            else:
                sanitized[key] = value
        return sanitized
    
    def _sanitize_query_params(self, query_params: Dict[str, str]) -> Dict[str, str]:
        """Sanitize query parameters to remove sensitive information"""
        sanitized = {}
        for key, value in query_params.items():
            if key.lower() in self.sensitive_query_params:
                sanitized[key] = "[REDACTED]"
            else:
                sanitized[key] = value
        return sanitized
    
    def _sanitize_request_body(self, body_data: Any) -> Any:
        """Sanitize request body to remove sensitive information"""
        if isinstance(body_data, dict):
            sanitized = {}
            for key, value in body_data.items():
                if key.lower() in self.sensitive_form_fields:
                    sanitized[key] = "[REDACTED]"
                elif isinstance(value, dict):
                    sanitized[key] = self._sanitize_request_body(value)
                else:
                    sanitized[key] = value
            return sanitized
        elif isinstance(body_data, list):
            return [self._sanitize_request_body(item) for item in body_data]
        else:
            return body_data
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address"""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        if request.client:
            return request.client.host
        
        return "unknown"
    
    def _update_statistics(self, response: Response, duration_ms: float):
        """Update internal statistics"""
        if response.status_code >= 400:
            self._stats["error_responses"] += 1
        
        if duration_ms > self.slow_request_threshold_ms:
            self._stats["slow_requests"] += 1
    
    def add_exclude_path(self, path: str):
        """Add path to exclude from logging"""
        self.exclude_paths.add(path)
        self.logger.info(f"Added exclude path: {path}")
    
    def remove_exclude_path(self, path: str):
        """Remove path from exclude list"""
        self.exclude_paths.discard(path)
        self.logger.info(f"Removed exclude path: {path}")
    
    def add_sensitive_header(self, header: str):
        """Add header to sensitive list"""
        self.sensitive_headers.add(header.lower())
        self.logger.info(f"Added sensitive header: {header}")
    
    def remove_sensitive_header(self, header: str):
        """Remove header from sensitive list"""
        self.sensitive_headers.discard(header.lower())
        self.logger.info(f"Removed sensitive header: {header}")
    
    def get_logging_statistics(self) -> Dict[str, Any]:
        """Get request logging statistics"""
        return {
            "statistics": dict(self._stats),
            "configuration": {
                "log_level": self.log_level,
                "log_requests": self.log_requests,
                "log_responses": self.log_responses,
                "log_request_body": self.log_request_body,
                "log_response_body": self.log_response_body,
                "log_headers": self.log_headers,
                "log_query_params": self.log_query_params,
                "sampling_rate": self.sampling_rate,
                "slow_request_threshold_ms": self.slow_request_threshold_ms,
                "output_format": self.output_format,
                "output_destination": self.output_destination
            },
            "exclude_paths": len(self.exclude_paths),
            "sensitive_headers": len(self.sensitive_headers),
            "sensitive_query_params": len(self.sensitive_query_params),
            "sensitive_form_fields": len(self.sensitive_form_fields)
        }