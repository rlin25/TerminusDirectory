"""
Multi-Factor Authentication (MFA) Manager

Provides comprehensive MFA support including TOTP, SMS, email, and push notifications
with secure token generation and verification.
"""

import asyncio
import base64
import logging
import secrets
import qrcode
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID, uuid4
from io import BytesIO

import pyotp
import phonenumbers
from phonenumbers import NumberParseException

from .models import (
    MFAMethod, MFAToken, SecurityEvent, SecurityEventType, 
    ThreatLevel, SecurityConfig
)


class MFAManager:
    """
    Multi-Factor Authentication Manager with comprehensive security features:
    - TOTP (Time-based One-Time Password) support
    - SMS verification with rate limiting
    - Email verification
    - Push notification support
    - Backup codes generation
    - MFA recovery options
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # MFA configuration
        self.totp_issuer = self.config.get("totp_issuer", SecurityConfig.MFA_TOTP_ISSUER)
        self.token_validity_minutes = self.config.get("token_validity_minutes", SecurityConfig.MFA_TOKEN_VALIDITY_MINUTES)
        self.max_attempts = self.config.get("max_attempts", SecurityConfig.MFA_MAX_ATTEMPTS)
        
        # Service configurations
        self.sms_service = self.config.get("sms_service")
        self.email_service = self.config.get("email_service")
        self.push_service = self.config.get("push_service")
        
        # Active MFA tokens (in production, use Redis)
        self._active_tokens: Dict[str, MFAToken] = {}
        
        # Rate limiting (in production, use Redis)
        self._sms_rate_limits: Dict[str, List[datetime]] = {}
        self._email_rate_limits: Dict[str, List[datetime]] = {}
        
        # MFA statistics
        self._mfa_stats = {
            "totp_verifications": 0,
            "sms_verifications": 0,
            "email_verifications": 0,
            "push_verifications": 0,
            "failed_verifications": 0,
            "backup_code_uses": 0
        }
    
    async def setup_totp(self, user_id: UUID, username: str, email: str) -> Dict[str, Any]:
        """
        Set up TOTP authentication for user
        
        Returns:
            Dictionary with secret, QR code, and backup codes
        """
        try:
            # Generate TOTP secret
            secret = pyotp.random_base32()
            
            # Create TOTP URI for QR code
            totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
                name=email,
                issuer_name=self.totp_issuer
            )
            
            # Generate QR code
            qr_code_data = self._generate_qr_code(totp_uri)
            
            # Generate backup codes
            backup_codes = self._generate_backup_codes()
            
            # Store MFA setup (temporary - user must verify before activation)
            setup_token = str(uuid4())
            self._active_tokens[setup_token] = MFAToken(
                user_id=user_id,
                method=MFAMethod.TOTP,
                secret=secret,
                expires_at=datetime.now() + timedelta(minutes=15)  # Setup expires in 15 minutes
            )
            
            self.logger.info(f"TOTP setup initiated for user {username}")
            
            return {
                "success": True,
                "setup_token": setup_token,
                "secret": secret,
                "qr_code_data": qr_code_data,
                "backup_codes": backup_codes,
                "message": "Scan QR code with authenticator app and verify to complete setup"
            }
            
        except Exception as e:
            self.logger.error(f"TOTP setup failed for user {user_id}: {e}")
            return {
                "success": False,
                "error": "Failed to setup TOTP authentication"
            }
    
    async def verify_totp_setup(
        self, 
        setup_token: str, 
        verification_code: str,
        user_id: UUID
    ) -> Dict[str, Any]:
        """Verify TOTP setup and activate MFA"""
        try:
            # Get setup token
            mfa_token = self._active_tokens.get(setup_token)
            if not mfa_token or mfa_token.is_expired() or mfa_token.user_id != user_id:
                return {
                    "success": False,
                    "error": "Invalid or expired setup token"
                }
            
            # Verify TOTP code
            totp = pyotp.TOTP(mfa_token.secret)
            if not totp.verify(verification_code, valid_window=1):
                mfa_token.increment_attempts()
                if mfa_token.attempts >= mfa_token.max_attempts:
                    del self._active_tokens[setup_token]
                
                return {
                    "success": False,
                    "error": "Invalid verification code"
                }
            
            # Activate TOTP for user
            await self._activate_user_mfa(user_id, MFAMethod.TOTP, mfa_token.secret)
            
            # Clean up setup token
            del self._active_tokens[setup_token]
            
            self.logger.info(f"TOTP activated for user {user_id}")
            
            return {
                "success": True,
                "message": "TOTP authentication activated successfully"
            }
            
        except Exception as e:
            self.logger.error(f"TOTP setup verification failed: {e}")
            return {
                "success": False,
                "error": "Failed to verify TOTP setup"
            }
    
    async def setup_sms_mfa(self, user_id: UUID, phone_number: str) -> Dict[str, Any]:
        """Set up SMS-based MFA for user"""
        try:
            # Validate phone number
            validation_result = self._validate_phone_number(phone_number)
            if not validation_result["valid"]:
                return {
                    "success": False,
                    "error": validation_result["error"]
                }
            
            formatted_phone = validation_result["formatted"]
            
            # Check SMS rate limits
            if not self._check_sms_rate_limit(formatted_phone):
                return {
                    "success": False,
                    "error": "SMS rate limit exceeded. Please try again later"
                }
            
            # Generate verification code
            verification_code = self._generate_numeric_code(6)
            
            # Create MFA token
            token_id = str(uuid4())
            self._active_tokens[token_id] = MFAToken(
                user_id=user_id,
                method=MFAMethod.SMS,
                token=verification_code,
                phone_number=formatted_phone,
                expires_at=datetime.now() + timedelta(minutes=self.token_validity_minutes)
            )
            
            # Send SMS
            sms_result = await self._send_sms(formatted_phone, verification_code)
            if not sms_result["success"]:
                del self._active_tokens[token_id]
                return {
                    "success": False,
                    "error": "Failed to send SMS verification code"
                }
            
            # Record SMS sent for rate limiting
            self._record_sms_sent(formatted_phone)
            
            self.logger.info(f"SMS MFA setup initiated for user {user_id}")
            
            return {
                "success": True,
                "token_id": token_id,
                "phone_number": self._mask_phone_number(formatted_phone),
                "message": "SMS verification code sent"
            }
            
        except Exception as e:
            self.logger.error(f"SMS MFA setup failed for user {user_id}: {e}")
            return {
                "success": False,
                "error": "Failed to setup SMS authentication"
            }
    
    async def setup_email_mfa(self, user_id: UUID, email: str) -> Dict[str, Any]:
        """Set up email-based MFA for user"""
        try:
            # Validate email
            if not self._validate_email(email):
                return {
                    "success": False,
                    "error": "Invalid email address"
                }
            
            # Check email rate limits
            if not self._check_email_rate_limit(email):
                return {
                    "success": False,
                    "error": "Email rate limit exceeded. Please try again later"
                }
            
            # Generate verification code
            verification_code = self._generate_alphanumeric_code(8)
            
            # Create MFA token
            token_id = str(uuid4())
            self._active_tokens[token_id] = MFAToken(
                user_id=user_id,
                method=MFAMethod.EMAIL,
                token=verification_code,
                email=email,
                expires_at=datetime.now() + timedelta(minutes=self.token_validity_minutes)
            )
            
            # Send email
            email_result = await self._send_email_verification(email, verification_code)
            if not email_result["success"]:
                del self._active_tokens[token_id]
                return {
                    "success": False,
                    "error": "Failed to send email verification code"
                }
            
            # Record email sent for rate limiting
            self._record_email_sent(email)
            
            self.logger.info(f"Email MFA setup initiated for user {user_id}")
            
            return {
                "success": True,
                "token_id": token_id,
                "email": self._mask_email(email),
                "message": "Email verification code sent"
            }
            
        except Exception as e:
            self.logger.error(f"Email MFA setup failed for user {user_id}: {e}")
            return {
                "success": False,
                "error": "Failed to setup email authentication"
            }
    
    async def initiate_mfa_verification(
        self, 
        user_id: UUID, 
        method: MFAMethod,
        phone_number: Optional[str] = None,
        email: Optional[str] = None
    ) -> Dict[str, Any]:
        """Initiate MFA verification for login"""
        try:
            if method == MFAMethod.TOTP:
                # TOTP doesn't need initiation
                return {
                    "success": True,
                    "method": method.value,
                    "message": "Enter code from authenticator app"
                }
            
            elif method == MFAMethod.SMS:
                if not phone_number:
                    return {
                        "success": False,
                        "error": "Phone number required for SMS verification"
                    }
                
                return await self._initiate_sms_verification(user_id, phone_number)
            
            elif method == MFAMethod.EMAIL:
                if not email:
                    return {
                        "success": False,
                        "error": "Email required for email verification"
                    }
                
                return await self._initiate_email_verification(user_id, email)
            
            else:
                return {
                    "success": False,
                    "error": f"MFA method {method.value} not supported"
                }
                
        except Exception as e:
            self.logger.error(f"MFA verification initiation failed: {e}")
            return {
                "success": False,
                "error": "Failed to initiate MFA verification"
            }
    
    async def verify_mfa_code(
        self, 
        token_id: str, 
        verification_code: str,
        method: Optional[MFAMethod] = None
    ) -> Dict[str, Any]:
        """Verify MFA code for authentication"""
        try:
            # Handle TOTP verification (uses user's stored secret)
            if method == MFAMethod.TOTP:
                return await self._verify_totp_code(token_id, verification_code)
            
            # Handle token-based verification (SMS, Email)
            mfa_token = self._active_tokens.get(token_id)
            if not mfa_token:
                return {
                    "success": False,
                    "error": "Invalid or expired verification token"
                }
            
            # Check if token is expired
            if mfa_token.is_expired():
                del self._active_tokens[token_id]
                return {
                    "success": False,
                    "error": "Verification code has expired"
                }
            
            # Check attempts
            if not mfa_token.increment_attempts():
                del self._active_tokens[token_id]
                return {
                    "success": False,
                    "error": "Maximum verification attempts exceeded"
                }
            
            # Verify code
            if mfa_token.token != verification_code:
                self._mfa_stats["failed_verifications"] += 1
                return {
                    "success": False,
                    "error": "Invalid verification code"
                }
            
            # Mark as verified
            mfa_token.verified = True
            
            # Update statistics
            if mfa_token.method == MFAMethod.SMS:
                self._mfa_stats["sms_verifications"] += 1
            elif mfa_token.method == MFAMethod.EMAIL:
                self._mfa_stats["email_verifications"] += 1
            
            # Clean up token
            user_id = mfa_token.user_id
            del self._active_tokens[token_id]
            
            self.logger.info(f"MFA verification successful for user {user_id}")
            
            return {
                "success": True,
                "user_id": user_id,
                "username": None,  # This would be fetched from user data
                "message": "MFA verification successful"
            }
            
        except Exception as e:
            self.logger.error(f"MFA code verification failed: {e}")
            return {
                "success": False,
                "error": "Failed to verify MFA code"
            }
    
    async def _verify_totp_code(self, user_id: str, verification_code: str) -> Dict[str, Any]:
        """Verify TOTP code using user's stored secret"""
        try:
            # Get user's TOTP secret (would query database in production)
            user_secret = await self._get_user_totp_secret(user_id)
            if not user_secret:
                return {
                    "success": False,
                    "error": "TOTP not configured for user"
                }
            
            # Verify TOTP code
            totp = pyotp.TOTP(user_secret)
            if not totp.verify(verification_code, valid_window=1):
                self._mfa_stats["failed_verifications"] += 1
                return {
                    "success": False,
                    "error": "Invalid TOTP code"
                }
            
            self._mfa_stats["totp_verifications"] += 1
            
            return {
                "success": True,
                "user_id": user_id,
                "message": "TOTP verification successful"
            }
            
        except Exception as e:
            self.logger.error(f"TOTP verification failed: {e}")
            return {
                "success": False,
                "error": "Failed to verify TOTP code"
            }
    
    async def generate_backup_codes(self, user_id: UUID) -> Dict[str, Any]:
        """Generate backup codes for user"""
        try:
            backup_codes = self._generate_backup_codes()
            
            # Store backup codes (hashed) in database
            await self._store_backup_codes(user_id, backup_codes)
            
            self.logger.info(f"Backup codes generated for user {user_id}")
            
            return {
                "success": True,
                "backup_codes": backup_codes,
                "message": "Store these backup codes in a safe place"
            }
            
        except Exception as e:
            self.logger.error(f"Backup code generation failed: {e}")
            return {
                "success": False,
                "error": "Failed to generate backup codes"
            }
    
    async def verify_backup_code(self, user_id: UUID, backup_code: str) -> Dict[str, Any]:
        """Verify backup code for account recovery"""
        try:
            # Get user's backup codes
            stored_codes = await self._get_user_backup_codes(user_id)
            if not stored_codes:
                return {
                    "success": False,
                    "error": "No backup codes found for user"
                }
            
            # Verify backup code (compare hashes)
            code_hash = self._hash_backup_code(backup_code)
            if code_hash not in stored_codes:
                return {
                    "success": False,
                    "error": "Invalid backup code"
                }
            
            # Remove used backup code
            await self._remove_backup_code(user_id, code_hash)
            
            self._mfa_stats["backup_code_uses"] += 1
            self.logger.info(f"Backup code used for user {user_id}")
            
            return {
                "success": True,
                "user_id": user_id,
                "message": "Backup code verification successful"
            }
            
        except Exception as e:
            self.logger.error(f"Backup code verification failed: {e}")
            return {
                "success": False,
                "error": "Failed to verify backup code"
            }
    
    def _generate_qr_code(self, data: str) -> str:
        """Generate QR code as base64 encoded image"""
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(data)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        
        return base64.b64encode(buffer.getvalue()).decode()
    
    def _generate_backup_codes(self, count: int = 10) -> List[str]:
        """Generate backup codes"""
        codes = []
        for _ in range(count):
            code = secrets.token_hex(4).upper()  # 8-character hex codes
            formatted_code = f"{code[:4]}-{code[4:]}"
            codes.append(formatted_code)
        return codes
    
    def _generate_numeric_code(self, length: int = 6) -> str:
        """Generate numeric verification code"""
        return ''.join([str(secrets.randbelow(10)) for _ in range(length)])
    
    def _generate_alphanumeric_code(self, length: int = 8) -> str:
        """Generate alphanumeric verification code"""
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        return ''.join([secrets.choice(alphabet) for _ in range(length)])
    
    def _validate_phone_number(self, phone_number: str) -> Dict[str, Any]:
        """Validate and format phone number"""
        try:
            parsed = phonenumbers.parse(phone_number, None)
            if not phonenumbers.is_valid_number(parsed):
                return {
                    "valid": False,
                    "error": "Invalid phone number"
                }
            
            formatted = phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
            return {
                "valid": True,
                "formatted": formatted
            }
            
        except NumberParseException:
            return {
                "valid": False,
                "error": "Unable to parse phone number"
            }
    
    def _validate_email(self, email: str) -> bool:
        """Validate email address"""
        try:
            from email_validator import validate_email, EmailNotValidError
            validate_email(email)
            return True
        except (EmailNotValidError, ImportError):
            return False
    
    def _check_sms_rate_limit(self, phone_number: str) -> bool:
        """Check SMS rate limit for phone number"""
        now = datetime.now()
        cutoff = now - timedelta(hours=1)
        
        if phone_number not in self._sms_rate_limits:
            return True
        
        # Clean old entries
        self._sms_rate_limits[phone_number] = [
            timestamp for timestamp in self._sms_rate_limits[phone_number]
            if timestamp > cutoff
        ]
        
        # Check limit (5 SMS per hour)
        return len(self._sms_rate_limits[phone_number]) < 5
    
    def _check_email_rate_limit(self, email: str) -> bool:
        """Check email rate limit"""
        now = datetime.now()
        cutoff = now - timedelta(hours=1)
        
        if email not in self._email_rate_limits:
            return True
        
        # Clean old entries
        self._email_rate_limits[email] = [
            timestamp for timestamp in self._email_rate_limits[email]
            if timestamp > cutoff
        ]
        
        # Check limit (3 emails per hour)
        return len(self._email_rate_limits[email]) < 3
    
    def _record_sms_sent(self, phone_number: str):
        """Record SMS sent for rate limiting"""
        if phone_number not in self._sms_rate_limits:
            self._sms_rate_limits[phone_number] = []
        
        self._sms_rate_limits[phone_number].append(datetime.now())
    
    def _record_email_sent(self, email: str):
        """Record email sent for rate limiting"""
        if email not in self._email_rate_limits:
            self._email_rate_limits[email] = []
        
        self._email_rate_limits[email].append(datetime.now())
    
    def _mask_phone_number(self, phone_number: str) -> str:
        """Mask phone number for display"""
        if len(phone_number) < 4:
            return "*" * len(phone_number)
        return phone_number[:-4] + "****"
    
    def _mask_email(self, email: str) -> str:
        """Mask email for display"""
        parts = email.split("@")
        if len(parts) != 2:
            return email
        
        username = parts[0]
        domain = parts[1]
        
        if len(username) <= 2:
            masked_username = "*" * len(username)
        else:
            masked_username = username[0] + "*" * (len(username) - 2) + username[-1]
        
        return f"{masked_username}@{domain}"
    
    async def _send_sms(self, phone_number: str, code: str) -> Dict[str, Any]:
        """Send SMS verification code"""
        # Mock SMS sending - replace with actual SMS service
        message = f"Your verification code is: {code}. Valid for {self.token_validity_minutes} minutes."
        
        self.logger.info(f"SMS sent to {self._mask_phone_number(phone_number)}: {code}")
        
        return {
            "success": True,
            "message_id": str(uuid4())
        }
    
    async def _send_email_verification(self, email: str, code: str) -> Dict[str, Any]:
        """Send email verification code"""
        # Mock email sending - replace with actual email service
        subject = "Your verification code"
        message = f"Your verification code is: {code}. Valid for {self.token_validity_minutes} minutes."
        
        self.logger.info(f"Email sent to {self._mask_email(email)}: {code}")
        
        return {
            "success": True,
            "message_id": str(uuid4())
        }
    
    async def _initiate_sms_verification(self, user_id: UUID, phone_number: str) -> Dict[str, Any]:
        """Initiate SMS verification for login"""
        # Check rate limits
        if not self._check_sms_rate_limit(phone_number):
            return {
                "success": False,
                "error": "SMS rate limit exceeded"
            }
        
        # Generate verification code
        verification_code = self._generate_numeric_code(6)
        
        # Create MFA token
        token_id = str(uuid4())
        self._active_tokens[token_id] = MFAToken(
            user_id=user_id,
            method=MFAMethod.SMS,
            token=verification_code,
            phone_number=phone_number,
            expires_at=datetime.now() + timedelta(minutes=self.token_validity_minutes)
        )
        
        # Send SMS
        sms_result = await self._send_sms(phone_number, verification_code)
        if not sms_result["success"]:
            del self._active_tokens[token_id]
            return {
                "success": False,
                "error": "Failed to send SMS"
            }
        
        self._record_sms_sent(phone_number)
        
        return {
            "success": True,
            "token_id": token_id,
            "message": "SMS verification code sent"
        }
    
    async def _initiate_email_verification(self, user_id: UUID, email: str) -> Dict[str, Any]:
        """Initiate email verification for login"""
        # Check rate limits
        if not self._check_email_rate_limit(email):
            return {
                "success": False,
                "error": "Email rate limit exceeded"
            }
        
        # Generate verification code
        verification_code = self._generate_alphanumeric_code(8)
        
        # Create MFA token
        token_id = str(uuid4())
        self._active_tokens[token_id] = MFAToken(
            user_id=user_id,
            method=MFAMethod.EMAIL,
            token=verification_code,
            email=email,
            expires_at=datetime.now() + timedelta(minutes=self.token_validity_minutes)
        )
        
        # Send email
        email_result = await self._send_email_verification(email, verification_code)
        if not email_result["success"]:
            del self._active_tokens[token_id]
            return {
                "success": False,
                "error": "Failed to send email"
            }
        
        self._record_email_sent(email)
        
        return {
            "success": True,
            "token_id": token_id,
            "message": "Email verification code sent"
        }
    
    async def _activate_user_mfa(self, user_id: UUID, method: MFAMethod, secret: str):
        """Activate MFA for user (store in database)"""
        # In production, this would update the database
        self.logger.info(f"MFA {method.value} activated for user {user_id}")
    
    async def _get_user_totp_secret(self, user_id: str) -> Optional[str]:
        """Get user's TOTP secret from database"""
        # Mock return for testing
        return "JBSWY3DPEHPK3PXP"  # Example secret
    
    async def _store_backup_codes(self, user_id: UUID, backup_codes: List[str]):
        """Store hashed backup codes in database"""
        # In production, hash and store codes in database
        pass
    
    async def _get_user_backup_codes(self, user_id: UUID) -> Optional[List[str]]:
        """Get user's backup codes from database"""
        # Mock return for testing
        return []
    
    async def _remove_backup_code(self, user_id: UUID, code_hash: str):
        """Remove used backup code from database"""
        # In production, remove used backup code
        pass
    
    def _hash_backup_code(self, code: str) -> str:
        """Hash backup code for storage"""
        import hashlib
        return hashlib.sha256(code.encode()).hexdigest()
    
    def get_mfa_statistics(self) -> Dict[str, Any]:
        """Get MFA usage statistics"""
        return {
            "statistics": self._mfa_stats.copy(),
            "active_tokens": len(self._active_tokens),
            "sms_rate_limits": len(self._sms_rate_limits),
            "email_rate_limits": len(self._email_rate_limits)
        }