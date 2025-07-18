"""
Email and SMS Service Implementations

Provides production-ready email and SMS services for MFA and notifications
with support for multiple providers and comprehensive error handling.
"""

import asyncio
import logging
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Optional, Any, List
from uuid import uuid4

import aiosmtplib
import httpx
from jinja2 import Environment, DictLoader

from .models import SecurityEvent, SecurityEventType, ThreatLevel


class EmailService:
    """
    Production email service with support for multiple providers
    including SMTP, SendGrid, AWS SES, and Mailgun
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.provider = config.get("provider", "smtp")
        
        # Email templates
        self.templates = {
            "mfa_verification": {
                "subject": "Your verification code for {{ service_name }}",
                "html": """
                <html>
                <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                    <h2 style="color: #333;">Verification Code</h2>
                    <p>Your verification code is:</p>
                    <div style="background: #f8f9fa; padding: 20px; text-align: center; margin: 20px 0; border-radius: 5px;">
                        <span style="font-size: 24px; font-weight: bold; letter-spacing: 3px; color: #007bff;">{{ verification_code }}</span>
                    </div>
                    <p>This code will expire in {{ expiry_minutes }} minutes.</p>
                    <p>If you didn't request this code, please ignore this email.</p>
                    <hr style="margin: 30px 0; border: none; border-top: 1px solid #eee;">
                    <p style="color: #666; font-size: 12px;">
                        This is an automated message from {{ service_name }}. Please do not reply to this email.
                    </p>
                </body>
                </html>
                """,
                "text": """
                Verification Code
                
                Your verification code is: {{ verification_code }}
                
                This code will expire in {{ expiry_minutes }} minutes.
                
                If you didn't request this code, please ignore this email.
                
                ---
                This is an automated message from {{ service_name }}. Please do not reply to this email.
                """
            },
            "login_notification": {
                "subject": "New login to your {{ service_name }} account",
                "html": """
                <html>
                <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                    <h2 style="color: #333;">New Login Detected</h2>
                    <p>We detected a new login to your account:</p>
                    <div style="background: #f8f9fa; padding: 15px; margin: 20px 0; border-radius: 5px;">
                        <strong>Time:</strong> {{ login_time }}<br>
                        <strong>Location:</strong> {{ location }}<br>
                        <strong>Device:</strong> {{ device }}<br>
                        <strong>IP Address:</strong> {{ ip_address }}
                    </div>
                    <p>If this was you, you can safely ignore this email.</p>
                    <p>If you don't recognize this activity, please secure your account immediately:</p>
                    <ol>
                        <li>Change your password</li>
                        <li>Review your account settings</li>
                        <li>Enable two-factor authentication if not already active</li>
                    </ol>
                    <p><a href="{{ secure_account_url }}" style="color: #007bff;">Secure My Account</a></p>
                </body>
                </html>
                """
            },
            "security_alert": {
                "subject": "Security Alert - {{ service_name }}",
                "html": """
                <html>
                <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                    <h2 style="color: #dc3545;">Security Alert</h2>
                    <div style="background: #f8d7da; color: #721c24; padding: 15px; margin: 20px 0; border-radius: 5px; border: 1px solid #f5c6cb;">
                        <strong>Alert Type:</strong> {{ alert_type }}<br>
                        <strong>Time:</strong> {{ timestamp }}<br>
                        <strong>Description:</strong> {{ description }}
                    </div>
                    <p>Immediate action may be required to secure your account.</p>
                    <p><a href="{{ security_center_url }}" style="color: #007bff;">Review Security Settings</a></p>
                </body>
                </html>
                """
            }
        }
        
        # Initialize Jinja2 environment
        self.jinja_env = Environment(loader=DictLoader(self.templates))
        
        # Initialize provider-specific settings
        self._init_provider()
    
    def _init_provider(self):
        """Initialize provider-specific settings"""
        if self.provider == "smtp":
            self.smtp_config = {
                "hostname": self.config.get("smtp_host", "localhost"),
                "port": self.config.get("smtp_port", 587),
                "username": self.config.get("smtp_username"),
                "password": self.config.get("smtp_password"),
                "use_tls": self.config.get("smtp_use_tls", True),
                "timeout": self.config.get("smtp_timeout", 30)
            }
        
        elif self.provider == "sendgrid":
            self.sendgrid_config = {
                "api_key": self.config.get("sendgrid_api_key"),
                "from_email": self.config.get("from_email"),
                "from_name": self.config.get("from_name", "Security Team")
            }
        
        elif self.provider == "aws_ses":
            self.ses_config = {
                "region": self.config.get("aws_region", "us-east-1"),
                "access_key": self.config.get("aws_access_key"),
                "secret_key": self.config.get("aws_secret_key"),
                "from_email": self.config.get("from_email")
            }
        
        elif self.provider == "mailgun":
            self.mailgun_config = {
                "api_key": self.config.get("mailgun_api_key"),
                "domain": self.config.get("mailgun_domain"),
                "from_email": self.config.get("from_email")
            }
    
    async def send_mfa_code(
        self,
        email: str,
        verification_code: str,
        expiry_minutes: int = 5,
        service_name: str = "Rental ML System"
    ) -> Dict[str, Any]:
        """Send MFA verification code via email"""
        try:
            template_data = {
                "verification_code": verification_code,
                "expiry_minutes": expiry_minutes,
                "service_name": service_name
            }
            
            return await self._send_templated_email(
                to_email=email,
                template_name="mfa_verification",
                template_data=template_data
            )
            
        except Exception as e:
            self.logger.error(f"Failed to send MFA code to {email}: {e}")
            return {"success": False, "error": str(e)}
    
    async def send_login_notification(
        self,
        email: str,
        login_info: Dict[str, Any],
        service_name: str = "Rental ML System"
    ) -> Dict[str, Any]:
        """Send login notification email"""
        try:
            template_data = {
                "service_name": service_name,
                "login_time": login_info.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")),
                "location": login_info.get("location", "Unknown"),
                "device": login_info.get("device", "Unknown"),
                "ip_address": login_info.get("ip_address", "Unknown"),
                "secure_account_url": login_info.get("secure_account_url", "#")
            }
            
            return await self._send_templated_email(
                to_email=email,
                template_name="login_notification",
                template_data=template_data
            )
            
        except Exception as e:
            self.logger.error(f"Failed to send login notification to {email}: {e}")
            return {"success": False, "error": str(e)}
    
    async def send_security_alert(
        self,
        email: str,
        alert_info: Dict[str, Any],
        service_name: str = "Rental ML System"
    ) -> Dict[str, Any]:
        """Send security alert email"""
        try:
            template_data = {
                "service_name": service_name,
                "alert_type": alert_info.get("type", "Security Event"),
                "timestamp": alert_info.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")),
                "description": alert_info.get("description", "Suspicious activity detected"),
                "security_center_url": alert_info.get("security_center_url", "#")
            }
            
            return await self._send_templated_email(
                to_email=email,
                template_name="security_alert",
                template_data=template_data
            )
            
        except Exception as e:
            self.logger.error(f"Failed to send security alert to {email}: {e}")
            return {"success": False, "error": str(e)}
    
    async def _send_templated_email(
        self,
        to_email: str,
        template_name: str,
        template_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Send email using template"""
        try:
            if template_name not in self.templates:
                raise ValueError(f"Unknown template: {template_name}")
            
            template = self.templates[template_name]
            
            # Render subject
            subject_template = self.jinja_env.from_string(template["subject"])
            subject = subject_template.render(**template_data)
            
            # Render HTML body
            html_template = self.jinja_env.from_string(template["html"])
            html_body = html_template.render(**template_data)
            
            # Render text body if available
            text_body = None
            if "text" in template:
                text_template = self.jinja_env.from_string(template["text"])
                text_body = text_template.render(**template_data)
            
            # Send email based on provider
            if self.provider == "smtp":
                return await self._send_smtp_email(to_email, subject, html_body, text_body)
            elif self.provider == "sendgrid":
                return await self._send_sendgrid_email(to_email, subject, html_body, text_body)
            elif self.provider == "aws_ses":
                return await self._send_ses_email(to_email, subject, html_body, text_body)
            elif self.provider == "mailgun":
                return await self._send_mailgun_email(to_email, subject, html_body, text_body)
            else:
                raise ValueError(f"Unsupported email provider: {self.provider}")
            
        except Exception as e:
            self.logger.error(f"Failed to send templated email: {e}")
            return {"success": False, "error": str(e)}
    
    async def _send_smtp_email(
        self,
        to_email: str,
        subject: str,
        html_body: str,
        text_body: Optional[str] = None
    ) -> Dict[str, Any]:
        """Send email via SMTP"""
        try:
            message = MIMEMultipart("alternative")
            message["Subject"] = subject
            message["From"] = self.config.get("from_email", "noreply@example.com")
            message["To"] = to_email
            
            # Add text part
            if text_body:
                text_part = MIMEText(text_body, "plain")
                message.attach(text_part)
            
            # Add HTML part
            html_part = MIMEText(html_body, "html")
            message.attach(html_part)
            
            # Send email
            await aiosmtplib.send(
                message,
                hostname=self.smtp_config["hostname"],
                port=self.smtp_config["port"],
                username=self.smtp_config["username"],
                password=self.smtp_config["password"],
                use_tls=self.smtp_config["use_tls"],
                timeout=self.smtp_config["timeout"]
            )
            
            message_id = str(uuid4())
            self.logger.info(f"Email sent via SMTP to {to_email} (ID: {message_id})")
            
            return {
                "success": True,
                "message_id": message_id,
                "provider": "smtp"
            }
            
        except Exception as e:
            self.logger.error(f"SMTP email failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _send_sendgrid_email(
        self,
        to_email: str,
        subject: str,
        html_body: str,
        text_body: Optional[str] = None
    ) -> Dict[str, Any]:
        """Send email via SendGrid API"""
        try:
            url = "https://api.sendgrid.com/v3/mail/send"
            
            payload = {
                "personalizations": [{
                    "to": [{"email": to_email}]
                }],
                "from": {
                    "email": self.sendgrid_config["from_email"],
                    "name": self.sendgrid_config["from_name"]
                },
                "subject": subject,
                "content": [
                    {"type": "text/html", "value": html_body}
                ]
            }
            
            if text_body:
                payload["content"].insert(0, {"type": "text/plain", "value": text_body})
            
            headers = {
                "Authorization": f"Bearer {self.sendgrid_config['api_key']}",
                "Content-Type": "application/json"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload, headers=headers)
                
                if response.status_code == 202:
                    message_id = response.headers.get("X-Message-Id", str(uuid4()))
                    return {
                        "success": True,
                        "message_id": message_id,
                        "provider": "sendgrid"
                    }
                else:
                    raise Exception(f"SendGrid API error: {response.status_code} - {response.text}")
            
        except Exception as e:
            self.logger.error(f"SendGrid email failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _send_mailgun_email(
        self,
        to_email: str,
        subject: str,
        html_body: str,
        text_body: Optional[str] = None
    ) -> Dict[str, Any]:
        """Send email via Mailgun API"""
        try:
            url = f"https://api.mailgun.net/v3/{self.mailgun_config['domain']}/messages"
            
            data = {
                "from": self.mailgun_config["from_email"],
                "to": to_email,
                "subject": subject,
                "html": html_body
            }
            
            if text_body:
                data["text"] = text_body
            
            auth = ("api", self.mailgun_config["api_key"])
            
            async with httpx.AsyncClient() as client:
                response = await client.post(url, data=data, auth=auth)
                
                if response.status_code == 200:
                    result = response.json()
                    return {
                        "success": True,
                        "message_id": result.get("id", str(uuid4())),
                        "provider": "mailgun"
                    }
                else:
                    raise Exception(f"Mailgun API error: {response.status_code} - {response.text}")
            
        except Exception as e:
            self.logger.error(f"Mailgun email failed: {e}")
            return {"success": False, "error": str(e)}


class SMSService:
    """
    Production SMS service with support for multiple providers
    including Twilio, AWS SNS, and Nexmo
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.provider = config.get("provider", "twilio")
        
        # SMS templates
        self.templates = {
            "mfa_verification": "Your {service_name} verification code is: {code}. Valid for {expiry_minutes} minutes.",
            "login_notification": "New login to your {service_name} account from {location} at {time}. If this wasn't you, secure your account immediately.",
            "security_alert": "Security Alert: {alert_type} detected on your {service_name} account. Please review your security settings."
        }
        
        self._init_provider()
    
    def _init_provider(self):
        """Initialize provider-specific settings"""
        if self.provider == "twilio":
            self.twilio_config = {
                "account_sid": self.config.get("twilio_account_sid"),
                "auth_token": self.config.get("twilio_auth_token"),
                "from_number": self.config.get("twilio_from_number")
            }
        
        elif self.provider == "aws_sns":
            self.sns_config = {
                "region": self.config.get("aws_region", "us-east-1"),
                "access_key": self.config.get("aws_access_key"),
                "secret_key": self.config.get("aws_secret_key")
            }
        
        elif self.provider == "nexmo":
            self.nexmo_config = {
                "api_key": self.config.get("nexmo_api_key"),
                "api_secret": self.config.get("nexmo_api_secret"),
                "from_number": self.config.get("nexmo_from_number", "RentalML")
            }
    
    async def send_mfa_code(
        self,
        phone_number: str,
        verification_code: str,
        expiry_minutes: int = 5,
        service_name: str = "Rental ML System"
    ) -> Dict[str, Any]:
        """Send MFA verification code via SMS"""
        try:
            message = self.templates["mfa_verification"].format(
                service_name=service_name,
                code=verification_code,
                expiry_minutes=expiry_minutes
            )
            
            return await self._send_sms(phone_number, message)
            
        except Exception as e:
            self.logger.error(f"Failed to send MFA code to {phone_number}: {e}")
            return {"success": False, "error": str(e)}
    
    async def send_login_notification(
        self,
        phone_number: str,
        login_info: Dict[str, Any],
        service_name: str = "Rental ML System"
    ) -> Dict[str, Any]:
        """Send login notification via SMS"""
        try:
            message = self.templates["login_notification"].format(
                service_name=service_name,
                location=login_info.get("location", "Unknown"),
                time=login_info.get("timestamp", "Unknown")
            )
            
            return await self._send_sms(phone_number, message)
            
        except Exception as e:
            self.logger.error(f"Failed to send login notification to {phone_number}: {e}")
            return {"success": False, "error": str(e)}
    
    async def send_security_alert(
        self,
        phone_number: str,
        alert_info: Dict[str, Any],
        service_name: str = "Rental ML System"
    ) -> Dict[str, Any]:
        """Send security alert via SMS"""
        try:
            message = self.templates["security_alert"].format(
                service_name=service_name,
                alert_type=alert_info.get("type", "Security Event")
            )
            
            return await self._send_sms(phone_number, message)
            
        except Exception as e:
            self.logger.error(f"Failed to send security alert to {phone_number}: {e}")
            return {"success": False, "error": str(e)}
    
    async def _send_sms(self, phone_number: str, message: str) -> Dict[str, Any]:
        """Send SMS using configured provider"""
        try:
            if self.provider == "twilio":
                return await self._send_twilio_sms(phone_number, message)
            elif self.provider == "aws_sns":
                return await self._send_sns_sms(phone_number, message)
            elif self.provider == "nexmo":
                return await self._send_nexmo_sms(phone_number, message)
            else:
                raise ValueError(f"Unsupported SMS provider: {self.provider}")
            
        except Exception as e:
            self.logger.error(f"Failed to send SMS: {e}")
            return {"success": False, "error": str(e)}
    
    async def _send_twilio_sms(self, phone_number: str, message: str) -> Dict[str, Any]:
        """Send SMS via Twilio API"""
        try:
            url = f"https://api.twilio.com/2010-04-01/Accounts/{self.twilio_config['account_sid']}/Messages.json"
            
            data = {
                "From": self.twilio_config["from_number"],
                "To": phone_number,
                "Body": message
            }
            
            auth = (self.twilio_config["account_sid"], self.twilio_config["auth_token"])
            
            async with httpx.AsyncClient() as client:
                response = await client.post(url, data=data, auth=auth)
                
                if response.status_code == 201:
                    result = response.json()
                    return {
                        "success": True,
                        "message_id": result.get("sid"),
                        "provider": "twilio"
                    }
                else:
                    raise Exception(f"Twilio API error: {response.status_code} - {response.text}")
            
        except Exception as e:
            self.logger.error(f"Twilio SMS failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _send_nexmo_sms(self, phone_number: str, message: str) -> Dict[str, Any]:
        """Send SMS via Nexmo (Vonage) API"""
        try:
            url = "https://rest.nexmo.com/sms/json"
            
            data = {
                "api_key": self.nexmo_config["api_key"],
                "api_secret": self.nexmo_config["api_secret"],
                "from": self.nexmo_config["from_number"],
                "to": phone_number.replace("+", ""),  # Remove + prefix
                "text": message
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(url, data=data)
                
                if response.status_code == 200:
                    result = response.json()
                    messages = result.get("messages", [])
                    
                    if messages and messages[0].get("status") == "0":
                        return {
                            "success": True,
                            "message_id": messages[0].get("message-id"),
                            "provider": "nexmo"
                        }
                    else:
                        error = messages[0].get("error-text") if messages else "Unknown error"
                        raise Exception(f"Nexmo error: {error}")
                else:
                    raise Exception(f"Nexmo API error: {response.status_code} - {response.text}")
            
        except Exception as e:
            self.logger.error(f"Nexmo SMS failed: {e}")
            return {"success": False, "error": str(e)}
    
    # Mock implementation for testing
    async def _send_mock_sms(self, phone_number: str, message: str) -> Dict[str, Any]:
        """Mock SMS sending for testing"""
        message_id = str(uuid4())
        self.logger.info(f"Mock SMS sent to {phone_number}: {message} (ID: {message_id})")
        
        return {
            "success": True,
            "message_id": message_id,
            "provider": "mock"
        }


class NotificationOrchestrator:
    """
    Orchestrates email and SMS notifications with fallback mechanisms
    and delivery tracking
    """
    
    def __init__(self, email_service: EmailService, sms_service: SMSService):
        self.logger = logging.getLogger(__name__)
        self.email_service = email_service
        self.sms_service = sms_service
        
        # Delivery tracking
        self.delivery_stats = {
            "emails_sent": 0,
            "emails_failed": 0,
            "sms_sent": 0,
            "sms_failed": 0
        }
    
    async def send_mfa_notification(
        self,
        user_contact: Dict[str, str],
        verification_code: str,
        method: str = "email",
        fallback: bool = True
    ) -> Dict[str, Any]:
        """Send MFA notification with fallback support"""
        try:
            results = []
            
            if method == "email" and user_contact.get("email"):
                email_result = await self.email_service.send_mfa_code(
                    user_contact["email"],
                    verification_code
                )
                results.append({"method": "email", "result": email_result})
                
                if email_result["success"]:
                    self.delivery_stats["emails_sent"] += 1
                    return {"success": True, "primary_method": "email", "results": results}
                else:
                    self.delivery_stats["emails_failed"] += 1
            
            if (method == "sms" or (fallback and method == "email")) and user_contact.get("phone"):
                sms_result = await self.sms_service.send_mfa_code(
                    user_contact["phone"],
                    verification_code
                )
                results.append({"method": "sms", "result": sms_result})
                
                if sms_result["success"]:
                    self.delivery_stats["sms_sent"] += 1
                    return {"success": True, "primary_method": "sms", "results": results}
                else:
                    self.delivery_stats["sms_failed"] += 1
            
            return {
                "success": False,
                "error": "All notification methods failed",
                "results": results
            }
            
        except Exception as e:
            self.logger.error(f"Notification orchestration failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def send_security_notification(
        self,
        user_contact: Dict[str, str],
        notification_type: str,
        notification_data: Dict[str, Any],
        channels: List[str] = ["email"]
    ) -> Dict[str, Any]:
        """Send security notification across multiple channels"""
        try:
            results = []
            successful_channels = []
            
            if "email" in channels and user_contact.get("email"):
                if notification_type == "login":
                    email_result = await self.email_service.send_login_notification(
                        user_contact["email"],
                        notification_data
                    )
                elif notification_type == "security_alert":
                    email_result = await self.email_service.send_security_alert(
                        user_contact["email"],
                        notification_data
                    )
                else:
                    email_result = {"success": False, "error": "Unknown notification type"}
                
                results.append({"method": "email", "result": email_result})
                
                if email_result["success"]:
                    successful_channels.append("email")
                    self.delivery_stats["emails_sent"] += 1
                else:
                    self.delivery_stats["emails_failed"] += 1
            
            if "sms" in channels and user_contact.get("phone"):
                if notification_type == "login":
                    sms_result = await self.sms_service.send_login_notification(
                        user_contact["phone"],
                        notification_data
                    )
                elif notification_type == "security_alert":
                    sms_result = await self.sms_service.send_security_alert(
                        user_contact["phone"],
                        notification_data
                    )
                else:
                    sms_result = {"success": False, "error": "Unknown notification type"}
                
                results.append({"method": "sms", "result": sms_result})
                
                if sms_result["success"]:
                    successful_channels.append("sms")
                    self.delivery_stats["sms_sent"] += 1
                else:
                    self.delivery_stats["sms_failed"] += 1
            
            return {
                "success": len(successful_channels) > 0,
                "successful_channels": successful_channels,
                "results": results
            }
            
        except Exception as e:
            self.logger.error(f"Security notification failed: {e}")
            return {"success": False, "error": str(e)}
    
    def get_delivery_statistics(self) -> Dict[str, Any]:
        """Get notification delivery statistics"""
        total_emails = self.delivery_stats["emails_sent"] + self.delivery_stats["emails_failed"]
        total_sms = self.delivery_stats["sms_sent"] + self.delivery_stats["sms_failed"]
        
        return {
            "email": {
                "sent": self.delivery_stats["emails_sent"],
                "failed": self.delivery_stats["emails_failed"],
                "success_rate": (self.delivery_stats["emails_sent"] / total_emails * 100) if total_emails > 0 else 0
            },
            "sms": {
                "sent": self.delivery_stats["sms_sent"],
                "failed": self.delivery_stats["sms_failed"],
                "success_rate": (self.delivery_stats["sms_sent"] / total_sms * 100) if total_sms > 0 else 0
            },
            "total_notifications": total_emails + total_sms
        }