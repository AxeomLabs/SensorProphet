"""
Email Notification Module for Predictive Maintenance System
Sends anomaly alerts via email with attached data.
"""
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
from typing import Dict, Optional, List
import json
import os

from src.utils import setup_logging

logger = setup_logging("email_notifier")


class EmailNotifier:
    """Handles email notifications for anomaly alerts."""
    
    def __init__(
        self,
        smtp_server: str = "smtp.gmail.com",
        smtp_port: int = 587,
        sender_email: str = "",
        sender_password: str = "",
        recipient_email: str = "snox.kali@usa.com",
    ):
        """
        Initialize email notifier.
        
        Args:
            smtp_server: SMTP server address
            smtp_port: SMTP port (587 for TLS, 465 for SSL)
            sender_email: Sender's email address
            sender_password: Sender's email password or app password
            recipient_email: Recipient's email address
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.recipient_email = recipient_email
        
        # Rate limiting - avoid spam
        self._last_sent = None
        self._min_interval = 60  # Minimum seconds between emails
        
    def configure(self, sender_email: str, sender_password: str, 
                  recipient_email: str = None, smtp_server: str = None):
        """Update email configuration."""
        self.sender_email = sender_email
        self.sender_password = sender_password
        if recipient_email:
            self.recipient_email = recipient_email
        if smtp_server:
            self.smtp_server = smtp_server
            
    def is_configured(self) -> bool:
        """Check if email is properly configured."""
        return bool(self.sender_email and self.sender_password and self.recipient_email)
    
    def send_anomaly_alert(
        self,
        sample_number: int,
        rms: float,
        kurtosis: float,
        mean: float,
        z_score: float,
        health_score: float,
        additional_data: Optional[Dict] = None,
    ) -> bool:
        """
        Send anomaly alert email.
        
        Args:
            sample_number: Sample index where anomaly was detected
            rms: RMS value
            kurtosis: Kurtosis value
            mean: Mean value
            z_score: Z-score value
            health_score: Current health score
            additional_data: Any additional data to include
            
        Returns:
            True if email was sent successfully
        """
        if not self.is_configured():
            logger.warning("Email not configured. Cannot send alert.")
            return False
        
        # Rate limiting
        now = datetime.now()
        if self._last_sent and (now - self._last_sent).total_seconds() < self._min_interval:
            logger.info("Rate limited - skipping email")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"🚨 ANOMALY DETECTED - Sample #{sample_number} - Health: {health_score:.0f}%"
            msg["From"] = self.sender_email
            msg["To"] = self.recipient_email
            
            # Create email content
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
            
            # Plain text version
            text = f"""
PREDICTIVE MAINTENANCE SYSTEM - ANOMALY ALERT
=============================================

Timestamp: {timestamp}
Sample Number: {sample_number}

SENSOR READINGS:
- RMS: {rms:.6f}
- Kurtosis: {kurtosis:.4f}
- Mean: {mean:.6f}

ANALYSIS:
- Z-Score: {z_score:.2f}
- Health Score: {health_score:.1f}%
- Status: {'CRITICAL' if health_score < 30 else 'WARNING'}

RECOMMENDATION:
{'Immediate maintenance required!' if health_score < 30 else 'Schedule maintenance soon.'}

This is an automated alert from the Predictive Maintenance System.
            """
            
            # HTML version
            status_color = "#f85149" if health_score < 30 else "#d29922"
            status_text = "CRITICAL" if health_score < 30 else "WARNING"
            
            html = f"""
            <html>
            <body style="font-family: Arial, sans-serif; background: #0d1117; color: #c9d1d9; padding: 20px;">
                <div style="max-width: 600px; margin: 0 auto; background: #161b22; border-radius: 8px; overflow: hidden;">
                    <div style="background: {status_color}; padding: 15px; text-align: center;">
                        <h1 style="margin: 0; color: white; font-size: 20px;">⚠️ ANOMALY DETECTED</h1>
                    </div>
                    
                    <div style="padding: 20px;">
                        <div style="background: #21262d; border-radius: 6px; padding: 15px; margin-bottom: 15px;">
                            <p style="margin: 0; color: #8b949e; font-size: 12px;">TIMESTAMP</p>
                            <p style="margin: 5px 0 0; font-size: 16px;">{timestamp}</p>
                        </div>
                        
                        <div style="background: #21262d; border-radius: 6px; padding: 15px; margin-bottom: 15px;">
                            <p style="margin: 0; color: #8b949e; font-size: 12px;">SAMPLE NUMBER</p>
                            <p style="margin: 5px 0 0; font-size: 24px; font-weight: bold; color: #f78166;">#{sample_number}</p>
                        </div>
                        
                        <h3 style="color: #8b949e; font-size: 12px; margin: 20px 0 10px;">SENSOR READINGS</h3>
                        <table style="width: 100%; border-collapse: collapse;">
                            <tr>
                                <td style="padding: 10px; background: #21262d; border-radius: 4px; text-align: center; width: 33%;">
                                    <div style="color: #8b949e; font-size: 10px;">RMS</div>
                                    <div style="font-size: 16px; font-family: monospace; color: #f78166;">{rms:.6f}</div>
                                </td>
                                <td style="padding: 10px; background: #21262d; border-radius: 4px; text-align: center; width: 33%;">
                                    <div style="color: #8b949e; font-size: 10px;">KURTOSIS</div>
                                    <div style="font-size: 16px; font-family: monospace; color: #f78166;">{kurtosis:.4f}</div>
                                </td>
                                <td style="padding: 10px; background: #21262d; border-radius: 4px; text-align: center; width: 33%;">
                                    <div style="color: #8b949e; font-size: 10px;">MEAN</div>
                                    <div style="font-size: 16px; font-family: monospace; color: #f78166;">{mean:.6f}</div>
                                </td>
                            </tr>
                        </table>
                        
                        <h3 style="color: #8b949e; font-size: 12px; margin: 20px 0 10px;">ANALYSIS</h3>
                        <div style="display: flex; gap: 10px;">
                            <div style="flex: 1; background: #21262d; border-radius: 4px; padding: 15px; text-align: center;">
                                <div style="color: #8b949e; font-size: 10px;">Z-SCORE</div>
                                <div style="font-size: 24px; font-weight: bold; color: #d29922;">{z_score:.2f}</div>
                            </div>
                            <div style="flex: 1; background: #21262d; border-radius: 4px; padding: 15px; text-align: center;">
                                <div style="color: #8b949e; font-size: 10px;">HEALTH</div>
                                <div style="font-size: 24px; font-weight: bold; color: {status_color};">{health_score:.0f}%</div>
                            </div>
                            <div style="flex: 1; background: {status_color}; border-radius: 4px; padding: 15px; text-align: center;">
                                <div style="color: white; font-size: 10px;">STATUS</div>
                                <div style="font-size: 18px; font-weight: bold; color: white;">{status_text}</div>
                            </div>
                        </div>
                        
                        <div style="margin-top: 20px; padding: 15px; background: #21262d; border-left: 4px solid {status_color}; border-radius: 4px;">
                            <strong>Recommendation:</strong> 
                            {'⚠️ Immediate maintenance required!' if health_score < 30 else '⏰ Schedule maintenance soon.'}
                        </div>
                    </div>
                    
                    <div style="padding: 15px; text-align: center; color: #6e7681; font-size: 11px; border-top: 1px solid #30363d;">
                        Predictive Maintenance System - Automated Alert
                    </div>
                </div>
            </body>
            </html>
            """
            
            # Attach parts
            msg.attach(MIMEText(text, "plain"))
            msg.attach(MIMEText(html, "html"))
            
            # Attach JSON data file
            anomaly_data = {
                "timestamp": timestamp,
                "sample_number": sample_number,
                "sensor_readings": {
                    "rms": rms,
                    "kurtosis": kurtosis,
                    "mean": mean,
                },
                "analysis": {
                    "z_score": z_score,
                    "health_score": health_score,
                    "status": status_text,
                },
                "additional": additional_data or {},
            }
            
            json_data = json.dumps(anomaly_data, indent=2)
            attachment = MIMEBase("application", "json")
            attachment.set_payload(json_data.encode())
            encoders.encode_base64(attachment)
            attachment.add_header("Content-Disposition", f"attachment; filename=anomaly_{sample_number}.json")
            msg.attach(attachment)
            
            # Send email
            context = ssl.create_default_context()
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.sender_email, self.sender_password)
                server.sendmail(self.sender_email, self.recipient_email, msg.as_string())
            
            self._last_sent = now
            logger.info(f"Anomaly alert email sent to {self.recipient_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False


# Global instance
email_notifier = EmailNotifier()


def configure_email(sender_email: str, sender_password: str, 
                    recipient_email: str = "snox.kali@usa.com"):
    """Configure email settings."""
    email_notifier.configure(sender_email, sender_password, recipient_email)


def send_anomaly_email(sample: int, rms: float, kurtosis: float, 
                       mean: float, z_score: float, health: float) -> bool:
    """Send anomaly alert email."""
    return email_notifier.send_anomaly_alert(
        sample_number=sample,
        rms=rms,
        kurtosis=kurtosis,
        mean=mean,
        z_score=z_score,
        health_score=health,
    )
