"""
Alert Engine Module

Manages alert generation, prioritization, and tracking for predictive maintenance.
"""
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict

from src.config import ALERT_LEVELS
from src.utils import setup_logging

logger = setup_logging("alert_engine")


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of alerts."""
    ANOMALY = "anomaly"
    HEALTH_SCORE = "health_score"
    THRESHOLD = "threshold"
    PREDICTION = "prediction"
    MAINTENANCE = "maintenance"
    EARLY_WARNING = "early_warning"


@dataclass
class Alert:
    """Alert data structure."""
    id: str
    equipment_id: str
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    timestamp: str
    source_value: Optional[float] = None
    threshold: Optional[float] = None
    acknowledged: bool = False
    resolved: bool = False
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = asdict(self)
        result["alert_type"] = self.alert_type.value
        result["severity"] = self.severity.value
        return result


class AlertEngine:
    """Engine for generating and managing alerts."""
    
    def __init__(
        self,
        thresholds: Optional[Dict[str, Dict]] = None,
        cooldown_minutes: int = 15,
    ):
        """
        Initialize alert engine.
        
        Args:
            thresholds: Custom thresholds for alerts.
            cooldown_minutes: Minimum time between duplicate alerts.
        """
        self.thresholds = thresholds or self._default_thresholds()
        self.cooldown_minutes = cooldown_minutes
        
        # Alert tracking
        self.alerts: List[Alert] = []
        self.alert_counts: Dict[str, int] = defaultdict(int)
        self._last_alert_time: Dict[str, datetime] = {}
        
        # Callbacks for alert notifications
        self._callbacks: List[Callable[[Alert], None]] = []
        
        # Alert counter for IDs
        self._alert_counter = 0
    
    def _default_thresholds(self) -> Dict[str, Dict]:
        """Get default alert thresholds."""
        return {
            "health_score": {
                "warning": 50,
                "critical": 30,
            },
            "anomaly_rate": {
                "warning": 0.3,
                "critical": 0.5,
            },
            "rms": {
                "warning": 0.15,
                "critical": 0.25,
            },
            "kurtosis": {
                "warning": 2.0,
                "critical": 4.0,
            },
            "temperature": {
                "warning": 70,
                "critical": 85,
            },
        }
    
    def register_callback(self, callback: Callable[[Alert], None]) -> None:
        """Register a callback for alert notifications."""
        self._callbacks.append(callback)
    
    def check_thresholds(
        self,
        equipment_id: str,
        metrics: Dict[str, float],
    ) -> List[Alert]:
        """
        Check metrics against thresholds and generate alerts.
        
        Args:
            equipment_id: Equipment identifier.
            metrics: Dictionary of metric values.
            
        Returns:
            List of generated alerts.
        """
        alerts = []
        
        for metric_name, value in metrics.items():
            if metric_name not in self.thresholds:
                continue
            
            threshold_config = self.thresholds[metric_name]
            
            # Determine if metric goes up (bad) or down (bad)
            is_lower_bad = metric_name in ["health_score"]
            
            # Check critical threshold
            if "critical" in threshold_config:
                critical_threshold = threshold_config["critical"]
                is_critical = value < critical_threshold if is_lower_bad else value > critical_threshold
                
                if is_critical:
                    alert = self._create_alert(
                        equipment_id=equipment_id,
                        alert_type=AlertType.THRESHOLD,
                        severity=AlertSeverity.CRITICAL,
                        message=f"{metric_name} is at critical level: {value:.2f}",
                        source_value=value,
                        threshold=critical_threshold,
                        metadata={"metric": metric_name},
                    )
                    if alert:
                        alerts.append(alert)
                    continue
            
            # Check warning threshold
            if "warning" in threshold_config:
                warning_threshold = threshold_config["warning"]
                is_warning = value < warning_threshold if is_lower_bad else value > warning_threshold
                
                if is_warning:
                    alert = self._create_alert(
                        equipment_id=equipment_id,
                        alert_type=AlertType.THRESHOLD,
                        severity=AlertSeverity.WARNING,
                        message=f"{metric_name} is elevated: {value:.2f}",
                        source_value=value,
                        threshold=warning_threshold,
                        metadata={"metric": metric_name},
                    )
                    if alert:
                        alerts.append(alert)
        
        return alerts
    
    def check_health_score(
        self,
        equipment_id: str,
        health_score: float,
    ) -> Optional[Alert]:
        """
        Check health score and generate alert if needed.
        
        Args:
            equipment_id: Equipment identifier.
            health_score: Current health score (0-100).
            
        Returns:
            Alert if threshold exceeded, None otherwise.
        """
        thresholds = self.thresholds.get("health_score", {})
        
        if health_score < thresholds.get("critical", 30):
            return self._create_alert(
                equipment_id=equipment_id,
                alert_type=AlertType.HEALTH_SCORE,
                severity=AlertSeverity.CRITICAL,
                message=f"Equipment health critically low: {health_score:.1f}%",
                source_value=health_score,
                threshold=thresholds.get("critical"),
            )
        elif health_score < thresholds.get("warning", 50):
            return self._create_alert(
                equipment_id=equipment_id,
                alert_type=AlertType.HEALTH_SCORE,
                severity=AlertSeverity.WARNING,
                message=f"Equipment health warning: {health_score:.1f}%",
                source_value=health_score,
                threshold=thresholds.get("warning"),
            )
        
        return None
    
    def check_anomaly(
        self,
        equipment_id: str,
        is_anomaly: bool,
        anomaly_score: Optional[float] = None,
    ) -> Optional[Alert]:
        """
        Generate alert for detected anomaly.
        
        Args:
            equipment_id: Equipment identifier.
            is_anomaly: Whether anomaly was detected.
            anomaly_score: Optional anomaly score.
            
        Returns:
            Alert if anomaly detected, None otherwise.
        """
        if not is_anomaly:
            return None
        
        severity = AlertSeverity.CRITICAL if (anomaly_score and anomaly_score < -1) else AlertSeverity.WARNING
        
        return self._create_alert(
            equipment_id=equipment_id,
            alert_type=AlertType.ANOMALY,
            severity=severity,
            message=f"Anomaly detected in sensor readings",
            source_value=anomaly_score,
            metadata={"anomaly_score": anomaly_score},
        )
    
    def check_prediction(
        self,
        equipment_id: str,
        predicted_failure_hours: float,
    ) -> Optional[Alert]:
        """
        Generate alert for predicted failure.
        
        Args:
            equipment_id: Equipment identifier.
            predicted_failure_hours: Hours until predicted failure.
            
        Returns:
            Alert if failure predicted soon, None otherwise.
        """
        if predicted_failure_hours <= 6:
            return self._create_alert(
                equipment_id=equipment_id,
                alert_type=AlertType.PREDICTION,
                severity=AlertSeverity.CRITICAL,
                message=f"Failure predicted within {predicted_failure_hours:.1f} hours",
                source_value=predicted_failure_hours,
            )
        elif predicted_failure_hours <= 24:
            return self._create_alert(
                equipment_id=equipment_id,
                alert_type=AlertType.PREDICTION,
                severity=AlertSeverity.WARNING,
                message=f"Failure predicted within {predicted_failure_hours:.1f} hours",
                source_value=predicted_failure_hours,
            )
        
        return None
    
    def check_early_warning(
        self,
        equipment_id: str,
        predicted_hours: float,
        confidence: float,
        affected_metrics: List[str],
        risk_score: float,
    ) -> Optional[Alert]:
        """
        Generate alert for early warning prediction.
        
        Args:
            equipment_id: Equipment identifier.
            predicted_hours: Hours until predicted anomaly.
            confidence: Prediction confidence (0-1).
            affected_metrics: List of affected metric names.
            risk_score: Overall risk score (0-100).
            
        Returns:
            Alert if warning should be issued, None otherwise.
        """
        if predicted_hours is None or confidence < 0.5:
            return None
        
        # Determine severity based on time horizon
        if predicted_hours <= 6:
            severity = AlertSeverity.CRITICAL
            message = f"EARLY WARNING: Anomaly predicted in {predicted_hours:.1f} hours"
        elif predicted_hours <= 24:
            severity = AlertSeverity.WARNING
            message = f"Early Warning: Anomaly predicted in {predicted_hours:.1f} hours"
        elif predicted_hours <= 48:
            severity = AlertSeverity.INFO
            message = f"Advisory: Potential anomaly in {predicted_hours:.1f} hours"
        else:
            return None  # Too far ahead to alert
        
        metrics_str = ", ".join(affected_metrics[:3])
        message += f" (Affected: {metrics_str}, Confidence: {confidence:.0%})"
        
        return self._create_alert(
            equipment_id=equipment_id,
            alert_type=AlertType.EARLY_WARNING,
            severity=severity,
            message=message,
            source_value=predicted_hours,
            threshold=confidence,
            metadata={
                "predicted_hours": predicted_hours,
                "confidence": confidence,
                "affected_metrics": affected_metrics,
                "risk_score": risk_score,
            },
        )
    
    def _create_alert(
        self,
        equipment_id: str,
        alert_type: AlertType,
        severity: AlertSeverity,
        message: str,
        source_value: Optional[float] = None,
        threshold: Optional[float] = None,
        metadata: Optional[Dict] = None,
    ) -> Optional[Alert]:
        """Create an alert if not in cooldown."""
        # Check cooldown
        alert_key = f"{equipment_id}:{alert_type.value}:{severity.value}"
        last_time = self._last_alert_time.get(alert_key)
        
        if last_time:
            elapsed = datetime.now() - last_time
            if elapsed < timedelta(minutes=self.cooldown_minutes):
                return None  # Still in cooldown
        
        # Create alert
        self._alert_counter += 1
        alert = Alert(
            id=f"alert_{self._alert_counter:06d}",
            equipment_id=equipment_id,
            alert_type=alert_type,
            severity=severity,
            message=message,
            timestamp=datetime.now().isoformat(),
            source_value=source_value,
            threshold=threshold,
            metadata=metadata,
        )
        
        # Track alert
        self.alerts.append(alert)
        self.alert_counts[severity.value] += 1
        self._last_alert_time[alert_key] = datetime.now()
        
        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
        
        logger.info(f"Alert generated: [{severity.value}] {equipment_id} - {message}")
        return alert
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Mark an alert as acknowledged."""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                logger.info(f"Alert acknowledged: {alert_id}")
                return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Mark an alert as resolved."""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.resolved = True
                logger.info(f"Alert resolved: {alert_id}")
                return True
        return False
    
    def get_active_alerts(
        self,
        equipment_id: Optional[str] = None,
        severity: Optional[AlertSeverity] = None,
    ) -> List[Alert]:
        """Get all active (unresolved) alerts."""
        alerts = [a for a in self.alerts if not a.resolved]
        
        if equipment_id:
            alerts = [a for a in alerts if a.equipment_id == equipment_id]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def get_alert_summary(self) -> Dict:
        """Get summary of alert statistics."""
        active = [a for a in self.alerts if not a.resolved]
        
        return {
            "total_alerts": len(self.alerts),
            "active_alerts": len(active),
            "critical_active": len([a for a in active if a.severity == AlertSeverity.CRITICAL]),
            "warning_active": len([a for a in active if a.severity == AlertSeverity.WARNING]),
            "acknowledged": len([a for a in active if a.acknowledged]),
        }
    
    def update_thresholds(self, metric: str, thresholds: Dict[str, float]) -> None:
        """Update thresholds for a metric."""
        self.thresholds[metric] = thresholds
        logger.info(f"Updated thresholds for {metric}: {thresholds}")
    
    def clear_old_alerts(self, days: int = 7) -> int:
        """Clear resolved alerts older than specified days."""
        cutoff = datetime.now() - timedelta(days=days)
        original_count = len(self.alerts)
        
        self.alerts = [
            a for a in self.alerts
            if not a.resolved or datetime.fromisoformat(a.timestamp) > cutoff
        ]
        
        removed = original_count - len(self.alerts)
        logger.info(f"Cleared {removed} old alerts")
        return removed


if __name__ == "__main__":
    # Test alert engine
    engine = AlertEngine()
    
    # Register callback
    def on_alert(alert: Alert):
        print(f"  Callback received: {alert.severity.value} - {alert.message}")
    
    engine.register_callback(on_alert)
    
    # Test threshold alerts
    print("Testing threshold alerts:")
    metrics = {
        "health_score": 45,
        "rms": 0.18,
        "temperature": 72,
    }
    alerts = engine.check_thresholds("pump_001", metrics)
    print(f"Generated {len(alerts)} alerts")
    
    # Test health score alert
    print("\nTesting health score alert:")
    alert = engine.check_health_score("pump_001", 25)
    if alert:
        print(f"  Alert: {alert.message}")
    
    # Test anomaly alert
    print("\nTesting anomaly alert:")
    alert = engine.check_anomaly("pump_001", True, anomaly_score=-2.5)
    if alert:
        print(f"  Alert: {alert.message}")
    
    # Get summary
    print("\nAlert Summary:")
    print(engine.get_alert_summary())
