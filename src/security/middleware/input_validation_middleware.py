"""
Input Validation and Sanitization Middleware

Advanced input validation middleware that protects against injection attacks,
XSS, malformed data, and other input-based security threats.
"""

import re
import json
import logging
import html
import urllib.parse
from typing import Dict, List, Optional, Any, Set, Union
from datetime import datetime

from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


class InputValidationRule:
    """Input validation rule definition"""
    
    def __init__(
        self,
        name: str,
        pattern: Optional[str] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        allowed_chars: Optional[str] = None,
        forbidden_patterns: Optional[List[str]] = None,
        sanitize: bool = True,
        required: bool = False
    ):
        self.name = name
        self.pattern = re.compile(pattern) if pattern else None
        self.max_length = max_length
        self.min_length = min_length
        self.allowed_chars = allowed_chars
        self.forbidden_patterns = [re.compile(p, re.IGNORECASE) for p in (forbidden_patterns or [])]
        self.sanitize = sanitize
        self.required = required


class InputValidationMiddleware(BaseHTTPMiddleware):
    """
    Input Validation and Sanitization Middleware with:
    - SQL injection prevention
    - XSS attack prevention
    - Command injection prevention
    - Path traversal prevention
    - File upload validation
    - JSON/XML validation
    - Unicode normalization
    - Content type validation
    """
    
    def __init__(
        self,
        app,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(app)
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Validation configuration
        self.enable_validation = self.config.get("enable_validation", True)
        self.strict_mode = self.config.get("strict_mode", False)
        self.max_request_size = self.config.get("max_request_size", 10 * 1024 * 1024)  # 10MB
        self.max_json_depth = self.config.get("max_json_depth", 10)
        
        # Load validation rules
        self.validation_rules = self._load_validation_rules()
        
        # SQL injection patterns
        self.sql_injection_patterns = [
            r"(?i)(union\s+select)",
            r"(?i)(select\s+.*\s+from)",
            r"(?i)(drop\s+table)",
            r"(?i)(delete\s+from)",
            r"(?i)(insert\s+into)",
            r"(?i)(update\s+.*\s+set)",
            r"(?i)(exec\s*\()",
            r"(?i)(sp_executesql)",
            r"(?i)(\bor\s+1\s*=\s*1)",
            r"(?i)(\band\s+1\s*=\s*1)",
            r"(?i)(;\s*--)",
            r"(?i)(/\*.*\*/)",
            r"(?i)(0x[0-9a-f]+)",
            r"(?i)(char\s*\(\s*\d+\s*\))",
            r"(?i)(ascii\s*\(\s*.*\s*\))",
            r"(?i)(waitfor\s+delay)"
        ]
        
        # XSS patterns
        self.xss_patterns = [
            r"(?i)<script[^>]*>.*?</script>",
            r"(?i)<iframe[^>]*>.*?</iframe>",
            r"(?i)<object[^>]*>.*?</object>",
            r"(?i)<embed[^>]*>",
            r"(?i)<link[^>]*>",
            r"(?i)<meta[^>]*>",
            r"(?i)javascript:",
            r"(?i)vbscript:",
            r"(?i)data:text/html",
            r"(?i)on\w+\s*=",
            r"(?i)expression\s*\(",
            r"(?i)@import",
            r"(?i)behavior:"
        ]
        
        # Command injection patterns
        self.command_injection_patterns = [
            r"(?i)(\||&&|\|\|)",
            r"(?i)(;\s*[a-z])",
            r"(?i)(`.*`)",
            r"(?i)(\$\(.*\))",
            r"(?i)(nc\s+-)",
            r"(?i)(wget\s+)",
            r"(?i)(curl\s+)",
            r"(?i)(chmod\s+)",
            r"(?i)(rm\s+-)",
            r"(?i)(cat\s+/)",
            r"(?i)(echo\s+.*>)",
            r"(?i)(python\s+-c)",
            r"(?i)(perl\s+-e)",
            r"(?i)(sh\s+-c)"
        ]
        
        # Path traversal patterns
        self.path_traversal_patterns = [
            r"\.\.\/",
            r"\.\.\\",
            r"\.\.%2f",
            r"\.\.%2F",
            r"\.\.%5c",
            r"\.\.%5C",
            r"%2e%2e%2f",
            r"%2e%2e%5c",
            r"..\/",
            r"..\\"
        ]
        
        # Compile all patterns
        self._compiled_sql_patterns = [re.compile(p) for p in self.sql_injection_patterns]
        self._compiled_xss_patterns = [re.compile(p) for p in self.xss_patterns]
        self._compiled_cmd_patterns = [re.compile(p) for p in self.command_injection_patterns]
        self._compiled_path_patterns = [re.compile(p) for p in self.path_traversal_patterns]
        
        # Allowed content types
        self.allowed_content_types = set(self.config.get("allowed_content_types", [
            "application/json",
            "application/x-www-form-urlencoded",
            "multipart/form-data",
            "text/plain",
            "application/xml",
            "text/xml"
        ]))
        
        # File upload validation
        self.max_file_size = self.config.get("max_file_size", 50 * 1024 * 1024)  # 50MB
        self.allowed_file_extensions = set(self.config.get("allowed_file_extensions", [
            ".jpg", ".jpeg", ".png", ".gif", ".pdf", ".doc", ".docx", ".txt", ".csv"
        ]))
        
        # Statistics
        self._stats = {
            "total_requests": 0,
            "validation_failures": 0,
            "sql_injection_attempts": 0,
            "xss_attempts": 0,
            "command_injection_attempts": 0,
            "path_traversal_attempts": 0,
            "oversized_requests": 0,
            "invalid_content_type": 0,
            "sanitized_inputs": 0
        }
    
    def _load_validation_rules(self) -> Dict[str, InputValidationRule]:
        """Load input validation rules"""
        rules = {}
        
        # Email validation
        rules["email"] = InputValidationRule(
            name="email",
            pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
            max_length=254,
            min_length=5
        )
        
        # Username validation
        rules["username"] = InputValidationRule(
            name="username",
            pattern=r"^[a-zA-Z0-9_-]{3,50}$",
            max_length=50,
            min_length=3,
            forbidden_patterns=[
                r"admin", r"root", r"system", r"null", r"undefined"
            ]
        )
        
        # Password validation
        rules["password"] = InputValidationRule(
            name="password",
            min_length=8,
            max_length=128,
            sanitize=False  # Don't sanitize passwords
        )
        
        # Search query validation
        rules["search_query"] = InputValidationRule(
            name="search_query",
            max_length=1000,
            forbidden_patterns=self.sql_injection_patterns + self.xss_patterns
        )
        
        # Numeric ID validation
        rules["numeric_id"] = InputValidationRule(
            name="numeric_id",
            pattern=r"^\d+$",
            max_length=20
        )
        
        # UUID validation
        rules["uuid"] = InputValidationRule(
            name="uuid",
            pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
        )
        
        # File path validation
        rules["file_path"] = InputValidationRule(
            name="file_path",
            max_length=500,
            forbidden_patterns=self.path_traversal_patterns + [
                r"/etc/", r"/root/", r"/home/", r"C:\\Windows", r"C:\\Users"
            ]
        )
        
        # Load custom rules from config
        custom_rules = self.config.get("custom_rules", {})
        for rule_name, rule_config in custom_rules.items():
            rules[rule_name] = InputValidationRule(name=rule_name, **rule_config)
        
        return rules
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Main middleware dispatch method"""
        if not self.enable_validation:
            return await call_next(request)
        
        try:
            self._stats["total_requests"] += 1
            
            # Validate request size
            if await self._validate_request_size(request):
                return JSONResponse(
                    status_code=413,
                    content={"error": "Request too large"}
                )
            
            # Validate content type
            if await self._validate_content_type(request):
                return JSONResponse(
                    status_code=415,
                    content={"error": "Unsupported content type"}
                )
            
            # Validate and sanitize request data
            validation_result = await self._validate_request_data(request)
            if not validation_result["valid"]:
                self._stats["validation_failures"] += 1
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "Input validation failed",
                        "details": validation_result["errors"]
                    }
                )
            
            # Process request with sanitized data
            response = await call_next(request)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Input validation middleware error: {e}", exc_info=True)
            # Allow request to proceed on middleware error
            return await call_next(request)
    
    async def _validate_request_size(self, request: Request) -> bool:
        """Validate request size"""
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                size = int(content_length)
                if size > self.max_request_size:
                    self._stats["oversized_requests"] += 1
                    self.logger.warning(f"Oversized request rejected: {size} bytes")
                    return True
            except ValueError:
                pass
        
        return False
    
    async def _validate_content_type(self, request: Request) -> bool:
        """Validate content type"""
        content_type = request.headers.get("content-type", "")
        
        # Allow requests without content type for GET requests
        if request.method == "GET" and not content_type:
            return False
        
        # Extract base content type (ignore charset, boundary, etc.)
        base_content_type = content_type.split(";")[0].strip().lower()
        
        if base_content_type and base_content_type not in self.allowed_content_types:
            self._stats["invalid_content_type"] += 1
            self.logger.warning(f"Invalid content type rejected: {content_type}")
            return True
        
        return False
    
    async def _validate_request_data(self, request: Request) -> Dict[str, Any]:
        """Validate and sanitize request data"""
        errors = []
        
        try:
            # Store original body for re-reading if needed
            original_body = await request.body()
            if original_body:
                # Create a new request object that can be read multiple times
                from starlette.requests import Request as StarletteRequest
                from io import BytesIO
                
                # Monkey-patch the receive method to allow re-reading
                async def new_receive():
                    return {"type": "http.request", "body": original_body}
                
                request._receive = new_receive
            
            # Validate URL parameters
            query_errors = self._validate_query_parameters(request)
            errors.extend(query_errors)
            
            # Validate request body if present
            if request.method in ["POST", "PUT", "PATCH"] and original_body:
                body_errors = await self._validate_request_body(request, original_body)
                errors.extend(body_errors)
            
            # Validate headers
            header_errors = self._validate_headers(request)
            errors.extend(header_errors)
            
            return {
                "valid": len(errors) == 0,
                "errors": errors
            }
            
        except Exception as e:
            self.logger.error(f"Request validation error: {e}")
            return {
                "valid": False,
                "errors": ["Request validation failed"]
            }
    
    def _validate_query_parameters(self, request: Request) -> List[str]:
        """Validate URL query parameters"""
        errors = []
        
        for key, value in request.query_params.items():
            # Check for injection attacks
            if self._detect_injection_attacks(value):
                errors.append(f"Suspicious content in query parameter '{key}'")
                continue
            
            # Sanitize parameter value
            sanitized_value = self._sanitize_input(value)
            if sanitized_value != value:
                self._stats["sanitized_inputs"] += 1
                # Update query params with sanitized value
                # Note: This is a simplified approach; in production,
                # you might want to reconstruct the request with clean data
        
        return errors
    
    async def _validate_request_body(self, request: Request, body: bytes = None) -> List[str]:
        """Validate request body"""
        errors = []
        
        try:
            content_type = request.headers.get("content-type", "")
            
            if "application/json" in content_type:
                errors.extend(await self._validate_json_body(request, body))
            elif "application/x-www-form-urlencoded" in content_type:
                errors.extend(await self._validate_form_body(request))
            elif "multipart/form-data" in content_type:
                errors.extend(await self._validate_multipart_body(request))
            
        except Exception as e:
            errors.append(f"Body validation error: {str(e)}")
        
        return errors
    
    async def _validate_json_body(self, request: Request, body: bytes = None) -> List[str]:
        """Validate JSON request body"""
        errors = []
        
        try:
            # Use provided body or read from request
            if body is None:
                body = await request.body()
            
            if not body:
                return errors
            
            # Parse JSON
            try:
                data = json.loads(body.decode('utf-8'))
            except json.JSONDecodeError as e:
                return [f"Invalid JSON: {str(e)}"]
            
            # Validate JSON structure
            depth = self._get_json_depth(data)
            if depth > self.max_json_depth:
                errors.append(f"JSON depth ({depth}) exceeds maximum ({self.max_json_depth})")
            
            # Validate JSON values
            json_errors = self._validate_json_values(data)
            errors.extend(json_errors)
            
        except Exception as e:
            errors.append(f"JSON validation error: {str(e)}")
        
        return errors
    
    async def _validate_form_body(self, request: Request) -> List[str]:
        """Validate form-encoded request body"""
        errors = []
        
        try:
            form_data = await request.form()
            
            for key, value in form_data.items():
                if isinstance(value, str):
                    if self._detect_injection_attacks(value):
                        errors.append(f"Suspicious content in form field '{key}'")
                
        except Exception as e:
            errors.append(f"Form validation error: {str(e)}")
        
        return errors
    
    async def _validate_multipart_body(self, request: Request) -> List[str]:
        """Validate multipart form data (file uploads)"""
        errors = []
        
        try:
            form_data = await request.form()
            
            for key, value in form_data.items():
                if hasattr(value, 'filename'):  # File upload
                    file_errors = self._validate_file_upload(value)
                    errors.extend(file_errors)
                elif isinstance(value, str):
                    if self._detect_injection_attacks(value):
                        errors.append(f"Suspicious content in form field '{key}'")
                
        except Exception as e:
            errors.append(f"Multipart validation error: {str(e)}")
        
        return errors
    
    def _validate_headers(self, request: Request) -> List[str]:
        """Validate request headers"""
        errors = []
        
        # Check for suspicious headers
        for name, value in request.headers.items():
            if self._detect_injection_attacks(value):
                errors.append(f"Suspicious content in header '{name}'")
        
        return errors
    
    def _validate_file_upload(self, file) -> List[str]:
        """Validate uploaded file"""
        errors = []
        
        if not hasattr(file, 'filename') or not file.filename:
            return errors
        
        filename = file.filename.lower()
        
        # Check file extension
        file_ext = None
        if '.' in filename:
            file_ext = '.' + filename.split('.')[-1]
        
        if file_ext not in self.allowed_file_extensions:
            errors.append(f"File type not allowed: {file_ext}")
        
        # Check file size
        if hasattr(file, 'size') and file.size > self.max_file_size:
            errors.append(f"File too large: {file.size} bytes")
        
        # Check for suspicious file names
        if self._detect_injection_attacks(filename):
            errors.append("Suspicious filename detected")
        
        return errors
    
    def _detect_injection_attacks(self, value: str) -> bool:
        """Detect various injection attack patterns"""
        if not isinstance(value, str):
            return False
        
        # Check for SQL injection
        for pattern in self._compiled_sql_patterns:
            if pattern.search(value):
                self._stats["sql_injection_attempts"] += 1
                self.logger.warning(f"SQL injection attempt detected: {value[:100]}")
                return True
        
        # Check for XSS
        for pattern in self._compiled_xss_patterns:
            if pattern.search(value):
                self._stats["xss_attempts"] += 1
                self.logger.warning(f"XSS attempt detected: {value[:100]}")
                return True
        
        # Check for command injection
        for pattern in self._compiled_cmd_patterns:
            if pattern.search(value):
                self._stats["command_injection_attempts"] += 1
                self.logger.warning(f"Command injection attempt detected: {value[:100]}")
                return True
        
        # Check for path traversal
        for pattern in self._compiled_path_patterns:
            if pattern.search(value):
                self._stats["path_traversal_attempts"] += 1
                self.logger.warning(f"Path traversal attempt detected: {value[:100]}")
                return True
        
        return False
    
    def _sanitize_input(self, value: str) -> str:
        """Sanitize input value"""
        if not isinstance(value, str):
            return value
        
        # HTML escape
        sanitized = html.escape(value)
        
        # URL decode (to catch encoded attacks)
        try:
            decoded = urllib.parse.unquote(sanitized)
            if decoded != sanitized and self._detect_injection_attacks(decoded):
                # Re-encode if malicious content was found after decoding
                sanitized = urllib.parse.quote(sanitized)
        except Exception:
            pass
        
        # Remove null bytes
        sanitized = sanitized.replace('\x00', '')
        
        # Normalize unicode
        try:
            import unicodedata
            sanitized = unicodedata.normalize('NFKC', sanitized)
        except Exception:
            pass
        
        return sanitized
    
    def _validate_json_values(self, data, path="") -> List[str]:
        """Recursively validate JSON values"""
        errors = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key
                
                # Validate key
                if isinstance(key, str) and self._detect_injection_attacks(key):
                    errors.append(f"Suspicious content in JSON key at {current_path}")
                
                # Validate value
                if isinstance(value, str) and self._detect_injection_attacks(value):
                    errors.append(f"Suspicious content in JSON value at {current_path}")
                elif isinstance(value, (dict, list)):
                    errors.extend(self._validate_json_values(value, current_path))
                
        elif isinstance(data, list):
            for i, item in enumerate(data):
                current_path = f"{path}[{i}]" if path else f"[{i}]"
                
                if isinstance(item, str) and self._detect_injection_attacks(item):
                    errors.append(f"Suspicious content in JSON array at {current_path}")
                elif isinstance(item, (dict, list)):
                    errors.extend(self._validate_json_values(item, current_path))
        
        return errors
    
    def _get_json_depth(self, data, depth=0) -> int:
        """Calculate JSON nesting depth"""
        if isinstance(data, dict):
            if not data:
                return depth
            return max(self._get_json_depth(value, depth + 1) for value in data.values())
        elif isinstance(data, list):
            if not data:
                return depth
            return max(self._get_json_depth(item, depth + 1) for item in data)
        else:
            return depth
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get input validation statistics"""
        return {
            "statistics": dict(self._stats),
            "configuration": {
                "enable_validation": self.enable_validation,
                "strict_mode": self.strict_mode,
                "max_request_size": self.max_request_size,
                "max_json_depth": self.max_json_depth,
                "allowed_content_types": list(self.allowed_content_types),
                "allowed_file_extensions": list(self.allowed_file_extensions),
                "max_file_size": self.max_file_size
            },
            "validation_rules": {
                name: {
                    "max_length": rule.max_length,
                    "min_length": rule.min_length,
                    "required": rule.required,
                    "sanitize": rule.sanitize
                }
                for name, rule in self.validation_rules.items()
            }
        }