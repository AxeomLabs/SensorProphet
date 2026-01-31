"""
Health Score Calculator

Computes equipment health scores from sensor data and features.
Provides aggregated health metrics for predictive maintenance.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from src.config import HEALTH_SCORE_WEIGHTS, ALERT_LEVELS
from src.utils import setup_logging, calculate_rms, calculate_kurtosis

logger = setup_logging("health_scorer")


class HealthStatus(Enum):
    """Equipment health status levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class HealthReport:
    """Structured health report for equipment."""
    equipment_id: str
    timestamp: str
    overall_score: float
    status: HealthStatus
    component_scores: Dict[str, float]
    trend: str
    recommendations: List[str]
    alerts: List[Dict]


class HealthScorer:
    """Calculate and track equipment health scores."""
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        baseline_window: int = 100,
    ):
        """
        Initialize health scorer.
        
        Args:
            weights: Feature weights for score calculation.
            baseline_window: Window size for baseline statistics.
        """
        self.weights = weights or HEALTH_SCORE_WEIGHTS.copy()
        self.baseline_window = baseline_window
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.items()}
        
        # Baseline statistics
        self._baseline: Dict[str, Dict] = {}
        self._history: List[Dict] = []
        self.is_calibrated = False
    
    def calibrate(
        self,
        baseline_data: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
    ) -> "HealthScorer":
        """
        Calibrate the scorer with baseline (healthy) data.
        
        Args:
            baseline_data: DataFrame with healthy equipment data.
            feature_columns: Columns to use for scoring.
            
        Returns:
            Self for chaining.
        """
        if feature_columns is None:
            feature_columns = baseline_data.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in feature_columns:
            values = baseline_data[col].dropna().values
            self._baseline[col] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "p5": np.percentile(values, 5),
                "p95": np.percentile(values, 95),
            }
        
        self.is_calibrated = True
        self.feature_columns = feature_columns
        logger.info(f"Calibrated health scorer with {len(feature_columns)} features")
        return self
    
    def calculate_score(
        self,
        data: Union[Dict[str, float], pd.Series, pd.DataFrame],
        equipment_id: str = "equipment_1",
    ) -> float:
        """
        Calculate overall health score (0-100).
        
        Args:
            data: Current sensor readings or features.
            equipment_id: Equipment identifier.
            
        Returns:
            Health score from 0 (critical) to 100 (excellent).
        """
        if isinstance(data, pd.DataFrame):
            data = data.iloc[-1].to_dict()
        elif isinstance(data, pd.Series):
            data = data.to_dict()
        
        component_scores = self._calculate_component_scores(data)
        
        # Weighted average
        overall_score = 0.0
        total_weight = 0.0
        
        for feature, score in component_scores.items():
            weight = self.weights.get(feature, 0.1)
            overall_score += score * weight
            total_weight += weight
        
        if total_weight > 0:
            overall_score = overall_score / total_weight
        
        # Ensure score is in valid range
        overall_score = max(0, min(100, overall_score))
        
        # Track history
        self._history.append({
            "equipment_id": equipment_id,
            "score": overall_score,
            "components": component_scores,
        })
        
        return overall_score
    
    def _calculate_component_scores(self, data: Dict[str, float]) -> Dict[str, float]:
        """Calculate individual component scores."""
        scores = {}
        
        for feature, value in data.items():
            if feature not in self._baseline and not self.is_calibrated:
                # Use simple heuristic if not calibrated
                scores[feature] = self._simple_score(feature, value)
            elif feature in self._baseline:
                baseline = self._baseline[feature]
                scores[feature] = self._statistical_score(value, baseline)
            else:
                # Feature not in baseline, use default
                scores[feature] = 75.0
        
        return scores
    
    def _simple_score(self, feature: str, value: float) -> float:
        """Calculate score using simple heuristics when not calibrated."""
        # Map feature types to expected ranges
        if "rms" in feature.lower():
            # Lower RMS is better
            if value < 0.05:
                return 100
            elif value < 0.1:
                return 80
            elif value < 0.2:
                return 50
            else:
                return max(0, 100 - value * 100)
        
        elif "kurtosis" in feature.lower():
            # Kurtosis close to 0 (normal distribution) is better
            deviation = abs(value)
            return max(0, 100 - deviation * 20)
        
        elif "temperature" in feature.lower():
            # Normal operating temperature around 40-60
            if 40 <= value <= 60:
                return 100
            elif 30 <= value <= 70:
                return 80
            elif 20 <= value <= 80:
                return 60
            else:
                return max(0, 100 - abs(value - 50))
        
        else:
            # Default: assume current value should be stable
            return 75.0
    
    def _statistical_score(self, value: float, baseline: Dict) -> float:
        """Calculate score based on statistical deviation from baseline."""
        mean = baseline["mean"]
        std = baseline["std"]
        
        if std == 0:
            return 100 if value == mean else 50
        
        # Calculate z-score
        z = abs(value - mean) / std
        
        # Convert z-score to health score
        # z=0 -> 100, z=1 -> 85, z=2 -> 50, z=3 -> 15, z>=4 -> 0
        score = 100 - (z ** 2) * 12.5
        return max(0, min(100, score))
    
    def get_status(self, score: float) -> HealthStatus:
        """Convert numeric score to health status."""
        if score >= 90:
            return HealthStatus.EXCELLENT
        elif score >= 70:
            return HealthStatus.GOOD
        elif score >= 50:
            return HealthStatus.FAIR
        elif score >= 30:
            return HealthStatus.WARNING
        else:
            return HealthStatus.CRITICAL
    
    def get_trend(self, window: int = 10) -> str:
        """Analyze recent trend in health scores."""
        if len(self._history) < 2:
            return "stable"
        
        recent = self._history[-window:]
        scores = [h["score"] for h in recent]
        
        if len(scores) < 2:
            return "stable"
        
        # Calculate trend
        slope = (scores[-1] - scores[0]) / len(scores)
        
        if slope > 1:
            return "improving"
        elif slope < -1:
            return "degrading"
        else:
            return "stable"
    
    def generate_report(
        self,
        data: Union[Dict, pd.Series, pd.DataFrame],
        equipment_id: str = "equipment_1",
        timestamp: Optional[str] = None,
    ) -> HealthReport:
        """
        Generate comprehensive health report.
        
        Args:
            data: Current sensor data.
            equipment_id: Equipment identifier.
            timestamp: Report timestamp.
            
        Returns:
            HealthReport object.
        """
        from datetime import datetime
        
        if isinstance(data, pd.DataFrame):
            data = data.iloc[-1].to_dict()
        elif isinstance(data, pd.Series):
            data = data.to_dict()
        
        # Calculate scores
        component_scores = self._calculate_component_scores(data)
        overall_score = self.calculate_score(data, equipment_id)
        status = self.get_status(overall_score)
        trend = self.get_trend()
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            overall_score, component_scores, status
        )
        
        # Generate alerts
        alerts = self._generate_alerts(overall_score, component_scores, status)
        
        return HealthReport(
            equipment_id=equipment_id,
            timestamp=timestamp or datetime.now().isoformat(),
            overall_score=overall_score,
            status=status,
            component_scores=component_scores,
            trend=trend,
            recommendations=recommendations,
            alerts=alerts,
        )
    
    def _generate_recommendations(
        self,
        score: float,
        components: Dict[str, float],
        status: HealthStatus,
    ) -> List[str]:
        """Generate maintenance recommendations."""
        recommendations = []
        
        if status == HealthStatus.CRITICAL:
            recommendations.append("Immediate maintenance inspection required")
            recommendations.append("Consider shutting down equipment to prevent damage")
        elif status == HealthStatus.WARNING:
            recommendations.append("Schedule maintenance within 24-48 hours")
            recommendations.append("Monitor equipment closely")
        elif status == HealthStatus.FAIR:
            recommendations.append("Plan maintenance within the next week")
        
        # Component-specific recommendations
        for comp, comp_score in components.items():
            if comp_score < 50:
                if "vibration" in comp.lower() or "rms" in comp.lower():
                    recommendations.append(f"Check {comp}: possible bearing or alignment issue")
                elif "temperature" in comp.lower():
                    recommendations.append(f"Check {comp}: possible cooling or lubrication issue")
                elif "kurtosis" in comp.lower():
                    recommendations.append(f"Check {comp}: signal indicates potential impact events")
        
        return recommendations
    
    def _generate_alerts(
        self,
        score: float,
        components: Dict[str, float],
        status: HealthStatus,
    ) -> List[Dict]:
        """Generate alert notifications."""
        alerts = []
        
        if status in [HealthStatus.CRITICAL, HealthStatus.WARNING]:
            alerts.append({
                "type": "health_score",
                "severity": status.value,
                "message": f"Health score is {score:.1f}%",
            })
        
        for comp, comp_score in components.items():
            if comp_score < 30:
                alerts.append({
                    "type": "component",
                    "severity": "critical",
                    "message": f"{comp} score is critically low: {comp_score:.1f}%",
                })
            elif comp_score < 50:
                alerts.append({
                    "type": "component",
                    "severity": "warning",
                    "message": f"{comp} score is concerning: {comp_score:.1f}%",
                })
        
        return alerts
    
    def get_history_dataframe(self) -> pd.DataFrame:
        """Get health score history as DataFrame."""
        if not self._history:
            return pd.DataFrame()
        
        records = []
        for h in self._history:
            record = {
                "equipment_id": h["equipment_id"],
                "score": h["score"],
            }
            record.update(h["components"])
            records.append(record)
        
        return pd.DataFrame(records)


def calculate_rul(
    health_scores: List[float],
    failure_threshold: float = 30.0,
    min_samples: int = 10,
) -> Optional[float]:
    """
    Estimate Remaining Useful Life (RUL) from health score trends.
    
    Args:
        health_scores: List of recent health scores.
        failure_threshold: Health score at which failure occurs.
        min_samples: Minimum samples for estimation.
        
    Returns:
        Estimated steps until failure, or None if cannot estimate.
    """
    if len(health_scores) < min_samples:
        return None
    
    scores = np.array(health_scores)
    
    # Check if degrading
    if scores[-1] >= scores[0]:
        return None  # Not degrading
    
    # Linear regression for trend
    x = np.arange(len(scores))
    slope, intercept = np.polyfit(x, scores, 1)
    
    if slope >= 0:
        return None  # Not degrading
    
    # Calculate time to threshold
    current_step = len(scores) - 1
    current_score = intercept + slope * current_step
    
    # Steps until failure_threshold
    if slope != 0:
        failure_step = (failure_threshold - intercept) / slope
        rul = failure_step - current_step
        return max(0, rul)
    
    return None


if __name__ == "__main__":
    # Test health scorer
    np.random.seed(42)
    
    # Generate baseline data (healthy operation)
    n_baseline = 100
    baseline_data = pd.DataFrame({
        "rms": np.random.normal(0.05, 0.01, n_baseline),
        "kurtosis": np.random.normal(0.5, 0.1, n_baseline),
        "temperature": np.random.normal(45, 2, n_baseline),
        "peak_to_peak": np.random.normal(0.1, 0.02, n_baseline),
    })
    
    # Initialize and calibrate
    scorer = HealthScorer()
    scorer.calibrate(baseline_data)
    
    # Test with healthy data
    healthy_reading = {"rms": 0.06, "kurtosis": 0.5, "temperature": 46, "peak_to_peak": 0.11}
    score = scorer.calculate_score(healthy_reading, "pump_001")
    print(f"Healthy reading score: {score:.1f}%")
    
    # Test with degraded data
    degraded_reading = {"rms": 0.15, "kurtosis": 2.0, "temperature": 65, "peak_to_peak": 0.3}
    score = scorer.calculate_score(degraded_reading, "pump_001")
    print(f"Degraded reading score: {score:.1f}%")
    
    # Generate report
    report = scorer.generate_report(degraded_reading, "pump_001")
    print(f"\nHealth Report:")
    print(f"  Status: {report.status.value}")
    print(f"  Trend: {report.trend}")
    print(f"  Recommendations: {report.recommendations}")
